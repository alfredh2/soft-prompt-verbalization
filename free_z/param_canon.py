"""Parameter-aware canonicalizer: can thresholds become decodable from canonical z?

Plain InfoNCE canonicalization made family recoverable but left parameters at the
guess-the-family-mean level (structured_decode.py). Here the canonicalizer is trained with extra
pressure: linear heads on the pooled canonical code predict family (CE) and each family's
parameters (MSE, scaled). Encoder-only (no LLM in training), so configurations run in minutes:

  B  contrastive + family
  C  contrastive + family + params
  D  family + params (no contrastive)

Every configuration is scored the same way: a ridge probe (fit on training z's) from the pooled
canonical code to params, checkpoint chosen by val param MAE; test on heldout_seed / heldout_param
with family acc, param MAE given true family, within-2, exact, same-function retrieval, and
behavioral fidelity of the rendered rule. Baseline: always predict the family's mean parameter.

Usage:
  python -u param_canon.py --bank runs/zbank_views/bank.pt --out runs/param_canon
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from structured_decode import render
from verbalize.canonicalizer import Canonicalizer, info_nce, pair_groups, sample_pairs, val_view_retrieval
from verbalize.evaluate import accuracy
from verbalize.model import SoftPromptLM
from verbalize.probe import ridge_fit_predict
from verbalize.tasks import build_task_suite
from verbalize.zbank import split_examples
from verbalize.zdecoder import annotate_quality, make_keep_rule, partition_bank

PARAM_SCALE = 30.0
CONFIGS = {
    "B_con+fam": {"beta_con": 1.0, "lam_fam": 1.0, "lam_par": 0.0},
    "C_con+fam+par": {"beta_con": 1.0, "lam_fam": 1.0, "lam_par": 1.0},
    "D_fam+par": {"beta_con": 0.0, "lam_fam": 1.0, "lam_par": 1.0},
}
TEST_SPLITS = ["heldout_seed", "heldout_param"]


class Heads(nn.Module):
    def __init__(self, d: int, families: list[str], keys: dict[str, list[str]]):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.family = nn.Linear(d, len(families))
        self.params = nn.ModuleDict({f: nn.Linear(d, len(keys[f])) for f in families})

    def pooled(self, code: torch.Tensor) -> torch.Tensor:
        return self.norm(code.mean(dim=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    specs = {s.name: s for v in build_task_suite().values() for s in v}
    bank = torch.load(args.bank, weights_only=False)
    annotate_quality(bank, specs)
    parts = partition_bank(bank, make_keep_rule(0.9, 0.1, 0.75), {6, 7}, 8, seed=0)
    families = sorted({r["family"] for r in parts["train"]})
    fam_idx = {f: i for i, f in enumerate(families)}
    keys = {f: list(next(r for r in parts["train"] if r["family"] == f)["params"]) for f in families}
    print({k: len(v) for k, v in parts.items()}, keys, flush=True)

    fam_means = {f: {k: sum(r["params"][k] for r in parts["train"] if r["family"] == f)
                     / sum(r["family"] == f for r in parts["train"]) for k in keys[f]} for f in families}

    def baseline(recs):
        errs = [sum(abs(round(fam_means[r["family"]][k]) - v) for k, v in r["params"].items()) / len(r["params"])
                for r in recs]
        return sum(errs) / len(errs)

    model = SoftPromptLM("Qwen/Qwen2.5-1.5B-Instruct", dtype=torch.float32)
    device = model.device
    zs = {s: torch.stack([r["z"] for r in parts[s]]).to(device) for s in ["train", "val"] + TEST_SPLITS}

    @torch.no_grad()
    def probe(canon, heads, split):
        """Family from the trained head (or centroid if untrained) and params from a ridge probe
        fit on training pooled codes. Returns metrics and per-record predictions."""
        h = {s: heads.pooled(canon.encode(zs[s])) for s in ("train", split)}
        logits = heads.family(h[split])
        fam_pred = logits.argmax(dim=1).tolist()
        rows = []
        per_family_pred = {}
        for f in families:
            tr = [i for i, r in enumerate(parts["train"]) if r["family"] == f]
            per_family_pred[f] = {
                k: ridge_fit_predict([h["train"][i] for i in tr], [parts["train"][i]["params"][k] for i in tr],
                                     list(h[split]), alpha=10.0)
                for k in keys[f]
            }
        for i, r in enumerate(parts[split]):
            tf, pf = r["family"], families[fam_pred[i]]
            true_fam_params = {k: per_family_pred[tf][k][i] for k in keys[tf]}
            errs = [abs(round(true_fam_params[k]) - v) for k, v in r["params"].items()]
            pp, text = render(pf, {k: per_family_pred[pf][k][i] for k in keys[pf]})
            rows.append({"task": r["task"], "seed": r["seed_idx"], "family_ok": pf == tf,
                         "param_mae": sum(errs) / len(errs), "within2": all(e <= 2 for e in errs),
                         "exact": pf == tf and pp == specs[r["task"]].params, "text": text})
        mean = lambda k: sum(float(x[k]) for x in rows) / len(rows)
        view_acc, view_chance = val_view_retrieval(canon, parts[split], device)
        return {"family_acc": mean("family_ok"), "param_mae": mean("param_mae"), "within2": mean("within2"),
                "exact": mean("exact"), "view_nn": view_acc, "view_nn_chance": view_chance}, rows

    summary = {"baseline_param_mae": {s: round(baseline(parts[s]), 3) for s in ["val"] + TEST_SPLITS}}
    print("predict-family-mean baseline param MAE:", summary["baseline_param_mae"], flush=True)

    for name, cfg in CONFIGS.items():
        torch.manual_seed(args.seed)
        rng = random.Random(args.seed)
        canon = Canonicalizer(1536, 1.0).to(device)
        heads = Heads(512, families, keys).to(device)
        opt = torch.optim.AdamW(list(canon.parameters()) + list(heads.parameters()), lr=3e-4, weight_decay=0.01)
        groups = pair_groups(parts["train"])
        best = (float("inf"), None, None)
        log = open(out / f"{name}_log.jsonl", "w")
        for step in range(args.steps):
            canon.train()
            heads.train()
            loss = zs["train"].new_zeros(())
            if cfg["beta_con"] > 0:
                pairs = sample_pairs(groups, 32, rng)
                za = torch.stack([a["z"] for a, _ in pairs]).to(device)
                zb = torch.stack([b["z"] for _, b in pairs]).to(device)
                l_con, _ = info_nce(canon.encode(za).flatten(1), canon.encode(zb).flatten(1), 0.1)
                loss = loss + cfg["beta_con"] * l_con
            idx = rng.sample(range(len(parts["train"])), min(64, len(parts["train"])))
            h = heads.pooled(canon.encode(zs["train"][idx]))
            recs = [parts["train"][i] for i in idx]
            fam_t = torch.tensor([fam_idx[r["family"]] for r in recs], device=device)
            loss = loss + cfg["lam_fam"] * F.cross_entropy(heads.family(h), fam_t)
            if cfg["lam_par"] > 0:
                par_losses = []
                for f in families:
                    sel = [j for j, r in enumerate(recs) if r["family"] == f]
                    if sel:
                        target = torch.tensor([[recs[j]["params"][k] / PARAM_SCALE for k in keys[f]] for j in sel],
                                              device=device)
                        par_losses.append(F.mse_loss(heads.params[f](h[sel]), target, reduction="sum"))
                loss = loss + cfg["lam_par"] * torch.stack(par_losses).sum() / len(recs)
            opt.zero_grad()
            loss.backward()
            opt.step()

            if (step + 1) % args.eval_every == 0:
                canon.eval()
                heads.eval()
                val_metrics, _ = probe(canon, heads, "val")
                rec = {"step": step + 1, "loss": round(loss.item(), 4), **{k: round(v, 3) for k, v in val_metrics.items()}}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                if val_metrics["param_mae"] < best[0]:
                    best = (val_metrics["param_mae"], step + 1,
                            ({k: v.detach().clone() for k, v in canon.state_dict().items()},
                             {k: v.detach().clone() for k, v in heads.state_dict().items()}))
        log.close()

        canon.load_state_dict(best[2][0])
        heads.load_state_dict(best[2][1])
        canon.eval()
        heads.eval()
        summary[name] = {"best_step": best[1], "val_param_mae": round(best[0], 3)}
        print(f"\n{name}: best step {best[1]} (val param MAE {best[0]:.2f})", flush=True)
        for split in TEST_SPLITS:
            metrics, rows = probe(canon, heads, split)
            fid = []
            for r, row in zip(parts[split], rows):
                spec = specs[r["task"]]
                _, eval_ex = split_examples(spec, 16)
                fid.append((accuracy(model, spec, eval_ex, "desc", desc_text=row["text"]),
                            accuracy(model, spec, eval_ex, "desc", desc_text=spec.true_description)))
            metrics["fid_rendered"] = sum(a for a, _ in fid) / len(fid)
            metrics["fid_oracle"] = sum(b for _, b in fid) / len(fid)
            summary[name][split] = {k: round(v, 3) for k, v in metrics.items()}
            print(f"  {split:14s} " + "  ".join(f"{k}={v:.3f}" for k, v in metrics.items())
                  + f"  | baseline param MAE {summary['baseline_param_mae'][split]}", flush=True)

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out / 'summary.json'}")


if __name__ == "__main__":
    main()
