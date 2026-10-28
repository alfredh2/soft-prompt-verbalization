"""Decoder plan, step 1: fine-tuned LLM decoder G(z*) -> description, trained on a z bank.

Train (or load) G, then evaluate per split:
  heldout_seed         known functions, unseen z solution (invariance to seed)
  heldout_param        unseen parameter values of known families (primary)
  heldout_composition  unseen gt_c AND mod compositions
  heldout_family       unseen family (absgt)
  val / train_seen     early-stopping functions / memorization check
Metrics: behavioral fidelity (frozen LLM, LoRA off) of z*, the decoded description, naive
self-report and examples->description baseline; semantic parse scores; cross-seed agreement
of decoded descriptions; geometry of raw z vs reader(z).

Usage:
  python -u train_zdecoder.py --bank runs/zbank/bank.pt --run-name zdec_r16
  python -u train_zdecoder.py --bank runs/zbank/bank.pt --run-name zdec_r16 --eval-only
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch

from verbalize.decoder import describe_from_z
from verbalize.evaluate import IRRELEVANT_TEXT, accuracy
from verbalize.lora import add_lora, load_lora_state_dict, lora_enabled
from verbalize.model import SoftPromptLM
from verbalize.probe import consistency_report, flat, nearest_centroid_accuracy, ridge_fit_predict
from verbalize.tasks import build_task_suite, parse_description, score_parse
from verbalize.zbank import split_examples
from verbalize.canonicalizer import Canonicalizer, train_canonical_decoder
from verbalize.zdecoder import ZReader, annotate_quality, make_keep_rule, partition_bank, train_zdecoder

ACC_KEYS = ["bare", "desc_irrelevant", "desc_oracle", "soft_z", "desc_decoded", "desc_selfreport", "desc_baseline"]
TEXT_KEYS = ["decoded", "selfreport", "baseline"]
EVAL_SPLITS = ["heldout_seed", "heldout_param", "heldout_composition", "heldout_family", "val", "train_seen"]


def stratify_records(records: list[dict], max_n: int) -> list[dict]:
    by_family: dict[str, list[dict]] = defaultdict(list)
    for r in sorted(records, key=lambda r: (r["task"], r["seed_idx"])):
        by_family[r["family"]].append(r)
    out, i = [], 0
    while len(out) < min(max_n, len(records)):
        for fam in sorted(by_family):
            if i < len(by_family[fam]) and len(out) < max_n:
                out.append(by_family[fam][i])
        i += 1
    return out


@torch.no_grad()
def eval_record(model, reader, loras, spec, rec, n_eval: int, rng: random.Random) -> dict:
    z = rec["z"].to(model.device)
    train_ex, eval_ex = split_examples(spec, n_eval)
    with lora_enabled(loras, True):
        decoded = describe_from_z(model, reader(z.unsqueeze(0))[0])
    with lora_enabled(loras, False):
        selfreport = describe_from_z(model, z)
        support = rng.sample(train_ex, min(16, len(train_ex)))
        baseline = model.verbalize_from_examples(spec, support, max_examples=len(support))
        texts = {"decoded": decoded, "selfreport": selfreport, "baseline": baseline}
        acc = {
            "bare": accuracy(model, spec, eval_ex, "bare"),
            "desc_irrelevant": accuracy(model, spec, eval_ex, "desc", desc_text=IRRELEVANT_TEXT),
            "desc_oracle": accuracy(model, spec, eval_ex, "desc", desc_text=spec.true_description),
            "soft_z": accuracy(model, spec, eval_ex, "soft", z=z),
        }
        for k, text in texts.items():
            acc[f"desc_{k}"] = accuracy(model, spec, eval_ex, "desc", desc_text=text or "(empty)")
    return {
        "task": rec["task"], "family": rec["family"], "seed_idx": rec["seed_idx"],
        "true": spec.true_description, **texts, "acc": acc,
        **{f"{k}_semantic": score_parse(spec, parse_description(t)) for k, t in texts.items()},
        "decoded_parsed": parse_description(decoded),
    }


def aggregate(rows: list[dict]) -> dict:
    agg = {k: sum(r["acc"][k] for r in rows) / len(rows) for k in ACC_KEYS}
    for k in TEXT_KEYS:
        sem = [r[f"{k}_semantic"] for r in rows]
        errs = [s["param_abs_err"] for s in sem if s["param_abs_err"] is not None]
        agg[f"{k}_family_acc"] = sum(s["family_correct"] for s in sem) / len(sem)
        agg[f"{k}_exact"] = sum(s["exact"] for s in sem) / len(sem)
        agg[f"{k}_param_mae"] = sum(errs) / len(errs) if errs else None
    by_task: dict[str, list] = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r["decoded_parsed"])
    multi = [v for v in by_task.values() if len(v) >= 2]
    agg["cross_seed_agreement"] = (
        sum(all(p is not None and p == v[0] for p in v) for v in multi) / len(multi) if multi else None
    )
    agg["n_rows"] = len(rows)
    return agg


@torch.no_grad()
def geometry(parts: dict[str, list[dict]], reader, device, module_name: str = "reader_z") -> dict:
    report = {}
    for rep_name in ("raw_z", module_name):
        def vec(r):
            z = r["z"].to(device)
            return flat(reader(z.unsqueeze(0))[0] if rep_name != "raw_z" else z).float()

        rep = {"consistency": {}}
        for split, recs in parts.items():
            by_task: dict[str, list] = defaultdict(list)
            for r in recs:
                by_task[r["task"]].append(vec(r))
            multi = {t: zs for t, zs in by_task.items() if len(zs) >= 2}
            k = min((len(v) for v in multi.values()), default=0)
            if len(multi) >= 2:
                rep["consistency"][split] = consistency_report({t: zs[:k] for t, zs in multi.items()}, rep=lambda x: x)
        if parts.get("train") and parts.get("heldout_param"):
            tr, te = parts["train"], parts["heldout_param"]
            rep["family_acc"] = nearest_centroid_accuracy(
                [vec(r) for r in tr], [r["family"] for r in tr], [vec(r) for r in te], [r["family"] for r in te]
            )
            rep["family_chance"] = 1 / len({r["family"] for r in tr})
            rep["ridge"] = {}
            for fam in ("gt", "lt"):
                a = [r for r in tr if r["family"] == fam]
                b = [r for r in te if r["family"] == fam]
                if len(a) >= 3 and b:
                    preds = ridge_fit_predict([vec(r) for r in a], [r["params"]["c"] for r in a], [vec(r) for r in b])
                    truth = [r["params"]["c"] for r in b]
                    rep["ridge"][fam] = {"mae": sum(abs(p - t) for p, t in zip(preds, truth)) / len(truth),
                                         "range": max(truth) - min(truth)}
        if parts.get("train") and parts.get("heldout_seed"):
            # Does an unseen seed of a known function land nearest a training seed of that same function?
            train_vecs = torch.stack([vec(r) for r in parts["train"]])
            train_tasks = [r["task"] for r in parts["train"]]
            hits, chance = [], []
            for r in parts["heldout_seed"]:
                j = int(torch.cdist(vec(r)[None], train_vecs)[0].argmin())
                hits.append(train_tasks[j] == r["task"])
                chance.append(train_tasks.count(r["task"]) / len(train_tasks))
            rep["heldout_seed_to_train_nn"] = {"acc": sum(hits) / len(hits), "chance": sum(chance) / len(chance)}
        report[rep_name] = rep
    return report


def print_summary(results: dict, geo: dict, bank_summary: dict) -> None:
    print("\n=== bank quality (keep rule: eval >= min_acc OR (q >= min_margin AND eval >= min_abs)) ===")
    for split, s in bank_summary.items():
        print(f"  {split:20s} {s['kept']}/{s['total']} z's kept | mean eval_acc {s['mean_eval_acc']:.3f} "
              f"| mean q {s['mean_q']:.3f} | kept by family {s['kept_by_family']}")
    splits = list(results)
    print("\n=== behavioral fidelity (frozen LLM, mean over z's) ===")
    print(f"{'condition':18s}" + "".join(f"{s:>21s}" for s in splits))
    for k in ACC_KEYS:
        print(f"{k:18s}" + "".join(f"{results[s][k]:21.3f}" for s in splits))
    print("\n=== semantic description scores ===")
    for k in [f"{t}_{m}" for t in TEXT_KEYS for m in ("family_acc", "exact", "param_mae")] + ["cross_seed_agreement"]:
        vals = [results[s].get(k) for s in splits]
        print(f"{k:24s}" + "".join(f"{'n/a':>21s}" if v is None else f"{v:21.3f}" for v in vals))
    print("\n=== geometry ===")
    for rep_name, rep in geo.items():
        for split, c in rep["consistency"].items():
            print(f"  {rep_name:9s} {split:20s} same/diff ratio={c['ratio']:.3f} nn={c['nn_retrieval']:.3f} "
                  f"(chance {c['nn_chance']:.3f})")
        if "family_acc" in rep:
            ridge = "  ".join(f"ridge {f} MAE={r['mae']:.2f}/{r['range']}" for f, r in rep["ridge"].items())
            print(f"  {rep_name:9s} family_acc={rep['family_acc']:.3f} (chance {rep['family_chance']:.3f})  {ridge}")
        if "heldout_seed_to_train_nn" in rep:
            s = rep["heldout_seed_to_train_nn"]
            print(f"  {rep_name:9s} new seed -> nearest training z is same function: {s['acc']:.3f} (chance {s['chance']:.3f})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bank", required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=float, default=32.0)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--min-acc", type=float, default=0.9, help="keep z if eval_acc >= this, or ...")
    p.add_argument("--min-margin", type=float, default=0.1, help="... if eval_acc - const >= this")
    p.add_argument("--min-abs", type=float, default=0.75, help="... and eval_acc >= this")
    p.add_argument("--exclude-families", default="", help="comma-separated families to drop entirely")
    p.add_argument("--no-quality-weight", action="store_true", help="unweighted decoder training loss")
    p.add_argument("--mode", choices=["reader", "canonical"], default="reader",
                   help="reader = step 1 (ZReader MLP); canonical = step 2 (set-encoder canonicalizer + contrastive)")
    p.add_argument("--contrast-weight", type=float, default=1.0)
    p.add_argument("--contrast-temp", type=float, default=0.1)
    p.add_argument("--contrast-batch", type=int, default=32)
    p.add_argument("--heldout-seed", type=int, default=-1, help="-1 = highest train seed index in the bank")
    p.add_argument("--heldout-seeds", help="comma-separated seed indices to hold out (overrides --heldout-seed)")
    p.add_argument("--n-val-tasks", type=int, default=15)
    p.add_argument("--n-eval", type=int, default=16)
    p.add_argument("--max-eval", type=int, default=30)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--eval-checkpoint", default="decoder.pt",
                   help="checkpoint file in the run dir to evaluate (canonical mode also saves decoder_retrieval.pt)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    run_dir = Path("runs") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    bank = torch.load(args.bank, weights_only=False)
    if args.heldout_seed == -1:
        train_seeds = {r["seed_idx"] for r in bank if r["split"] == "train"}
        args.heldout_seed = max(train_seeds) if len(train_seeds) > 1 else -99
    specs = {s.name: s for tasks in build_task_suite().values() for s in tasks}
    annotate_quality(bank, specs, args.n_eval)
    exclude = tuple(f for f in args.exclude_families.split(",") if f)
    keep = make_keep_rule(args.min_acc, args.min_margin, args.min_abs, exclude)
    heldout = {int(s) for s in args.heldout_seeds.split(",")} if args.heldout_seeds else args.heldout_seed
    parts = partition_bank(bank, keep, heldout, args.n_val_tasks, seed=args.seed)
    bank_summary = {}
    for split in sorted({r["split"] for r in bank}):
        recs = [r for r in bank if r["split"] == split]
        kept_by_family: dict[str, str] = {}
        for fam in sorted({r["family"] for r in recs}):
            fr = [r for r in recs if r["family"] == fam]
            kept_by_family[fam] = f"{sum(keep(r) for r in fr)}/{len(fr)}"
        bank_summary[split] = {
            "total": len(recs),
            "kept": sum(keep(r) for r in recs),
            "mean_eval_acc": sum(r["eval_acc"] for r in recs) / len(recs),
            "mean_q": sum(r["q"] for r in recs) / len(recs),
            "kept_by_family": kept_by_family,
        }
    print({k: len(v) for k, v in parts.items()}, flush=True)

    model = SoftPromptLM(args.model, dtype=torch.float32)
    loras = add_lora(model.model, r=args.lora_r, alpha=args.lora_alpha)
    target_norm = model.embed_layer.weight.float().norm(dim=-1).mean().item()
    hidden = model.embed_layer.weight.shape[1]
    reader = (Canonicalizer(hidden, target_norm) if args.mode == "canonical" else ZReader(hidden, target_norm)).to(model.device)
    print(f"LoRA modules: {len(loras)}, reader params {sum(p.numel() for p in reader.parameters()) / 1e6:.1f}M", flush=True)

    if not args.eval_only and args.mode == "canonical":
        train_canonical_decoder(model, reader, loras, parts, specs, run_dir, args.steps, args.batch, args.lr,
                                contrast_weight=args.contrast_weight, contrast_temp=args.contrast_temp,
                                contrast_batch=args.contrast_batch, eval_every=args.eval_every, seed=args.seed,
                                weight_by_quality=not args.no_quality_weight)
    elif not args.eval_only:
        train_zdecoder(model, reader, loras, parts, specs, run_dir, args.steps, args.batch, args.lr,
                       eval_every=args.eval_every, seed=args.seed,
                       weight_by_quality=not args.no_quality_weight)
    ckpt = torch.load(run_dir / args.eval_checkpoint, map_location=model.device, weights_only=False)
    reader.load_state_dict(ckpt["reader"])
    load_lora_state_dict(loras, ckpt["lora"])
    reader.eval()
    print(f"loaded best decoder from step {ckpt['step']} (val loss {ckpt['val_loss']:.4f})", flush=True)

    eval_parts = dict(parts)
    eval_parts["train_seen"] = parts.get("train", [])
    results, all_rows = {}, {}
    for split in EVAL_SPLITS:
        recs = stratify_records(eval_parts.get(split, []), args.max_eval)
        if not recs:
            continue
        print(f"\n=== eval {split} ({len(recs)} z's) ===", flush=True)
        rng = random.Random(args.seed)
        rows = []
        for rec in recs:
            row = eval_record(model, reader, loras, specs[rec["task"]], rec, args.n_eval, rng)
            rows.append(row)
            print(f"  {row['task']:22s} s{row['seed_idx']} soft={row['acc']['soft_z']:.2f} "
                  f"decoded={row['acc']['desc_decoded']:.2f} | {row['decoded']!r}", flush=True)
        results[split] = aggregate(rows)
        all_rows[split] = rows

    geo = geometry(parts, reader, model.device, module_name="canon_z" if args.mode == "canonical" else "reader_z")
    eval_path = run_dir / ("eval.json" if args.eval_checkpoint == "decoder.pt"
                           else f"eval_{Path(args.eval_checkpoint).stem}.json")
    eval_path.write_text(json.dumps(
        {"args": vars(args), "bank_summary": bank_summary, "results": results, "geometry": geo, "rows": all_rows,
         "checkpoint": {"file": args.eval_checkpoint, "step": ckpt["step"], "val_loss": ckpt["val_loss"]}},
        indent=2,
    ))
    print_summary(results, geo, bank_summary)
    print(f"\nwrote {eval_path}")


if __name__ == "__main__":
    main()
