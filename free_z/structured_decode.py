"""Structured decoding probe: z -> (family, params) -> template text, no free-form generation.

Separates "is the rule recoverable from (canonical) z?" from "can an LLM decoder write it". Feature
sets: raw z and the canonical code of a trained canonicalizer. Predictors are deliberately small
(few hundred training z's): logistic regression + per-family ridge, and nearest-centroid + kNN.
Regularization is chosen on val functions, so heldout_seed / heldout_param are the clean tests.

Usage:
  python -u structured_decode.py --bank runs/zbank_views/bank.pt \
      --canon-ckpt runs/zdec_views_canonical_ret/decoder_retrieval.pt --out runs/structured_decode
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from verbalize.canonicalizer import Canonicalizer
from verbalize.evaluate import accuracy
from verbalize.model import SoftPromptLM
from verbalize.probe import ridge_fit_predict
from verbalize.tasks import TEMPLATES, build_task_suite
from verbalize.zbank import split_examples
from verbalize.zdecoder import annotate_quality, make_keep_rule, partition_bank

SPLITS = ["heldout_seed", "heldout_param", "val"]
L2_GRID = [1e-4, 1e-3, 1e-2, 1e-1]
ALPHA_GRID = [1.0, 10.0, 100.0, 1000.0, 10000.0]


def render(family: str, params: dict) -> tuple[dict, str]:
    p = {k: int(round(v)) for k, v in params.items()}
    if "a" in p and "b" in p:
        a, b = sorted((p["a"], p["b"]))
        p["a"], p["b"] = a, b if b > a else a + 1
    return p, TEMPLATES[family][0].format(**p)


def fit_logreg(X, y, n_cls, l2, steps=300):
    W = torch.zeros(X.shape[1], n_cls, device=X.device, requires_grad=True)
    b = torch.zeros(n_cls, device=X.device, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=0.05)
    for _ in range(steps):
        loss = F.cross_entropy(X @ W + b, y) + l2 * (W**2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return lambda Z: (Z @ W + b).argmax(dim=1)


def predict_family(method, Xtr, ytr, parts_X, parts_y, families):
    if method == "centroid":
        cent = torch.stack([Xtr[ytr == i].mean(0) for i in range(len(families))])
        return lambda Z: torch.cdist(Z, cent).argmin(dim=1), None
    best = None
    for l2 in L2_GRID:
        pred = fit_logreg(Xtr, ytr, len(families), l2)
        acc = (pred(parts_X["val"]) == parts_y["val"]).float().mean().item() if len(parts_y["val"]) else 0.0
        if best is None or acc > best[0]:
            best = (acc, l2, pred)
    return best[2], best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--canon-ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--heldout-seeds", default="6,7")
    ap.add_argument("--n-val-tasks", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    specs = {s.name: s for v in build_task_suite().values() for s in v}
    bank = torch.load(args.bank, weights_only=False)
    annotate_quality(bank, specs)
    held = {int(s) for s in args.heldout_seeds.split(",")}
    parts = partition_bank(bank, make_keep_rule(0.9, 0.1, 0.75), held, args.n_val_tasks, seed=0)
    families = sorted({r["family"] for r in parts["train"]})
    fam_idx = {f: i for i, f in enumerate(families)}
    print({k: len(v) for k, v in parts.items()}, families, flush=True)

    model = SoftPromptLM("Qwen/Qwen2.5-1.5B-Instruct", dtype=torch.float32)
    device = model.device
    canon = Canonicalizer(model.embed_layer.weight.shape[1], 1.0).to(device)
    canon.load_state_dict(torch.load(args.canon_ckpt, map_location=device, weights_only=False)["reader"])
    canon.eval()

    @torch.no_grad()
    def features(name, recs):
        z = torch.stack([r["z"] for r in recs]).to(device)
        if name == "raw_z":
            return z.flatten(1).double()
        return torch.cat([canon.encode(z[i:i + 64]).flatten(1) for i in range(0, len(z), 64)]).double()

    results, rows_out = {}, {}
    for feat in ("raw_z", "canon_code"):
        Xraw = {s: features(feat, parts[s]) for s in ["train"] + SPLITS}
        mu, sd = Xraw["train"].mean(0), Xraw["train"].std(0).clamp(min=1e-6)
        X = {s: ((v - mu) / sd).float() for s, v in Xraw.items()}
        y = {s: torch.tensor([fam_idx.get(r["family"], -1) for r in parts[s]], device=device) for s in X}

        for fam_method, param_method in (("logreg", "ridge"), ("centroid", "knn")):
            name = f"{feat}|{fam_method}+{param_method}"
            fam_pred_fn, l2 = predict_family(fam_method, X["train"], y["train"], X, y, families)

            # per-family parameter predictors
            param_pred = {}
            for fam in families:
                tr = [i for i, r in enumerate(parts["train"]) if r["family"] == fam]
                va = [i for i, r in enumerate(parts["val"]) if r["family"] == fam]
                keys = list(parts["train"][tr[0]]["params"])
                if param_method == "ridge":
                    chosen = {}
                    for key in keys:
                        ytr = [parts["train"][i]["params"][key] for i in tr]
                        best = (float("inf"), 10.0)
                        for alpha in ALPHA_GRID:
                            if not va:
                                break
                            pv = ridge_fit_predict([X["train"][i] for i in tr], ytr, [X["val"][i] for i in va], alpha)
                            mae = sum(abs(p - parts["val"][i]["params"][key]) for p, i in zip(pv, va)) / len(va)
                            best = min(best, (mae, alpha))
                        chosen[key] = best[1]

                    def pred(Z, tr=tr, keys=keys, chosen=chosen):
                        return {k: ridge_fit_predict([X["train"][i] for i in tr],
                                                     [parts["train"][i]["params"][k] for i in tr],
                                                     [Z], chosen[k])[0] for k in keys}
                else:
                    def pred(Z, tr=tr, keys=keys):
                        Xt = F.normalize(X["train"][tr], dim=1)
                        nn = (Xt @ F.normalize(Z, dim=0)).topk(min(5, len(tr))).indices.tolist()
                        return {k: sum(parts["train"][tr[j]]["params"][k] for j in nn) / len(nn) for k in keys}
                param_pred[fam] = pred

            results[name], rows_out[name] = {}, {}
            for split in SPLITS:
                recs = parts[split]
                if not recs:
                    continue
                fam_hat = fam_pred_fn(X[split]).tolist()
                rows = []
                for i, r in enumerate(recs):
                    spec = specs[r["task"]]
                    pf = families[fam_hat[i]]
                    pp, text = render(pf, param_pred[pf](X[split][i]))
                    tp, text_truefam = render(r["family"], param_pred[r["family"]](X[split][i]))
                    _, eval_ex = split_examples(spec, 16)
                    errs = [abs(tp[k] - v) for k, v in spec.params.items()]
                    rows.append({
                        "task": r["task"], "seed": r["seed_idx"], "pred_family": pf, "pred_params": pp,
                        "text": text, "family_ok": pf == r["family"],
                        "exact": pf == r["family"] and pp == spec.params,
                        "truefam_param_mae": sum(errs) / len(errs), "truefam_within2": all(e <= 2 for e in errs),
                        "fid_pred": accuracy(model, spec, eval_ex, "desc", desc_text=text),
                        "fid_truefam": accuracy(model, spec, eval_ex, "desc", desc_text=text_truefam),
                        "fid_oracle": accuracy(model, spec, eval_ex, "desc", desc_text=spec.true_description),
                        "fid_bare": accuracy(model, spec, eval_ex, "bare"),
                    })
                mean = lambda k: sum(float(x[k]) for x in rows) / len(rows)
                results[name][split] = {k: round(mean(k), 3) for k in (
                    "family_ok", "exact", "truefam_param_mae", "truefam_within2",
                    "fid_pred", "fid_truefam", "fid_oracle", "fid_bare")}
                results[name][split]["n"] = len(rows)
                rows_out[name][split] = rows
            print(f"\n{name} (logreg l2={l2})", flush=True)
            for split, m in results[name].items():
                print(f"  {split:14s} " + "  ".join(f"{k}={v}" for k, v in m.items()), flush=True)

    (out / "results.json").write_text(json.dumps({"results": results, "rows": rows_out}, indent=2))
    print(f"\nwrote {out / 'results.json'}")


if __name__ == "__main__":
    main()
