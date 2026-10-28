"""Phase 2: shared behavioral encoder trained with L = L_behavior + lambda * L_text.

Trains (or resumes / loads) an encoder, then evaluates per split:
  query accuracy: bare, desc_irrelevant, desc_oracle, behavior_incontext,
                  soft_encoder, desc_decoded, desc_baseline
  semantic description scores (family / exact / param error) for decoded and baseline
  geometry: support-draw consistency, family classification, parameter ridge regression

Usage:
  python -u run_shared_encoder.py --run-name lam0.1 --lambda-text 0.1
  python -u run_shared_encoder.py --run-name smoke --tiny --steps 50
  python -u run_shared_encoder.py --run-name overfit --only-task gt_5 --steps 150
  python -u run_shared_encoder.py --run-name lam0.1 --eval-only
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch

from verbalize.decoder import describe_from_z
from verbalize.encoder import BehaviorEncoder, PairFeatureCache, encode_supports
from verbalize.evaluate import IRRELEVANT_TEXT, accuracy
from verbalize.joint_train import JointConfig, train
from verbalize.model import SoftPromptLM
from verbalize.probe import (
    consistency_report, flat, nearest_centroid_accuracy, pooled, ridge_fit_predict,
)
from verbalize.tasks import (
    all_pair_texts, build_task_suite, draw_support_query, incontext_text, parse_description, score_parse,
)

ACC_KEYS = [
    "bare", "desc_irrelevant", "desc_oracle", "behavior_incontext",
    "soft_encoder", "desc_decoded", "desc_baseline",
]
PHASE1 = {"ratio": 33.4 / 33.2, "family_acc": 0.47}


def stratified(tasks, max_n):
    """Round-robin across families so a capped subsample covers every family."""
    by_family = defaultdict(list)
    for t in tasks:
        by_family[t.family].append(t)
    out, i = [], 0
    while len(out) < min(max_n, len(tasks)):
        for fam in sorted(by_family):
            if i < len(by_family[fam]) and len(out) < max_n:
                out.append(by_family[fam][i])
        i += 1
    return out


@torch.no_grad()
def eval_split(model, encoder, cache, tasks, args, split_seed):
    rows = []
    for ti, spec in enumerate(tasks):
        rng = random.Random(split_seed * 100_003 + ti)
        support, query = draw_support_query(
            spec, args.n_support, args.n_eval_query, rng, balanced_query=args.balanced_query
        )
        z = encode_supports(encoder, cache, [spec], [support])[0]
        decoded = describe_from_z(model, z)
        baseline = model.verbalize_from_examples(spec, support, max_examples=len(support))

        acc = {
            "bare": accuracy(model, spec, query, "bare"),
            "desc_irrelevant": accuracy(model, spec, query, "desc", desc_text=IRRELEVANT_TEXT),
            "desc_oracle": accuracy(model, spec, query, "desc", desc_text=spec.true_description),
            "behavior_incontext": accuracy(
                model, spec, query, "text", context_text=incontext_text(spec, support)
            ),
            "soft_encoder": accuracy(model, spec, query, "soft", z=z),
            "desc_decoded": accuracy(model, spec, query, "desc", desc_text=decoded or "(empty)"),
            "desc_baseline": accuracy(model, spec, query, "desc", desc_text=baseline or "(empty)"),
        }
        row = {
            "task": spec.name,
            "family": spec.family,
            "true": spec.true_description,
            "decoded": decoded,
            "baseline": baseline,
            "acc": acc,
            "decoded_semantic": score_parse(spec, parse_description(decoded)),
            "baseline_semantic": score_parse(spec, parse_description(baseline)),
        }
        rows.append(row)
        print(f"  {spec.name:22s} soft={acc['soft_encoder']:.2f} incontext={acc['behavior_incontext']:.2f} "
              f"decoded={acc['desc_decoded']:.2f} | {decoded!r}", flush=True)
    return rows


def aggregate(rows):
    agg = {k: sum(r["acc"][k] for r in rows) / len(rows) for k in ACC_KEYS}
    for which in ("decoded", "baseline"):
        sem = [r[f"{which}_semantic"] for r in rows]
        errs = [s["param_abs_err"] for s in sem if s["param_abs_err"] is not None]
        agg[f"{which}_family_acc"] = sum(s["family_correct"] for s in sem) / len(sem)
        agg[f"{which}_exact"] = sum(s["exact"] for s in sem) / len(sem)
        agg[f"{which}_param_mae"] = sum(errs) / len(errs) if errs else None
    agg["n_tasks"] = len(rows)
    return agg


@torch.no_grad()
def geometry(encoder, cache, suite, args):
    """Encoder-only (cheap): uses every task in each split, not the eval subsample."""
    z_by_split = {}
    for si, (split, tasks) in enumerate(suite.items()):
        z_by_split[split] = {}
        for ti, spec in enumerate(tasks):
            supports = [
                draw_support_query(spec, args.n_support, 0, random.Random(1_000_003 * si + 7919 * ti + 31 * k))[0]
                for k in range(args.n_draws)
            ]
            z = encode_supports(encoder, cache, [spec] * args.n_draws, supports)
            z_by_split[split][spec.name] = [z[k] for k in range(args.n_draws)]

    report = {"consistency": {}, "consistency_mean_pooled": {}}
    for split, zs in z_by_split.items():
        if len(zs) >= 2:
            report["consistency"][split] = consistency_report(zs, rep=flat)
            report["consistency_mean_pooled"][split] = consistency_report(zs, rep=pooled)

    specs = {s.name: s for tasks in suite.values() for s in tasks}
    if "train" in z_by_split and z_by_split.get("heldout_param"):
        train_items = [(specs[n], zs[0]) for n, zs in z_by_split["train"].items()]
        test_items = [(specs[n], zs[0]) for n, zs in z_by_split["heldout_param"].items()]
        families = sorted({s.family for s, _ in train_items})
        for name, rep in (("flat", flat), ("mean_pooled", pooled)):
            report[f"family_acc_{name}"] = nearest_centroid_accuracy(
                [rep(z) for _, z in train_items], [s.family for s, _ in train_items],
                [rep(z) for _, z in test_items], [s.family for s, _ in test_items],
            )
        report["family_chance"] = 1 / len(families)

        report["param_regression"] = {}
        for fam in ("gt", "lt"):
            tr = [(s, z) for s, z in train_items if s.family == fam]
            te = [(s, z) for s, z in test_items if s.family == fam]
            if len(tr) >= 3 and te:
                preds = ridge_fit_predict([flat(z) for _, z in tr], [s.params["c"] for s, _ in tr],
                                          [flat(z) for _, z in te], alpha=10.0)
                truth = [s.params["c"] for s, _ in te]
                report["param_regression"][fam] = {
                    "mae": sum(abs(p - t) for p, t in zip(preds, truth)) / len(truth),
                    "param_range": max(truth) - min(truth),
                    "n_train": len(tr), "n_test": len(te),
                }
    return report


def print_summary(results, geo):
    print("\n=== query accuracy (mean over tasks) ===")
    splits = list(results)
    print(f"{'condition':20s}" + "".join(f"{s:>22s}" for s in splits))
    for k in ACC_KEYS:
        print(f"{k:20s}" + "".join(f"{results[s][k]:22.3f}" for s in splits))
    print("\n=== semantic description scores ===")
    for k in ("decoded_family_acc", "decoded_exact", "decoded_param_mae",
              "baseline_family_acc", "baseline_exact", "baseline_param_mae"):
        vals = [results[s][k] for s in splits]
        print(f"{k:20s}" + "".join(f"{'n/a':>22s}" if v is None else f"{v:22.3f}" for v in vals))
    print("\n=== geometry ===")
    for split, c in geo["consistency"].items():
        print(f"  {split:20s} same/diff ratio={c['ratio']:.3f}  nn_retrieval={c['nn_retrieval']:.3f} "
              f"(chance {c['nn_chance']:.3f})")
    if "family_acc_flat" in geo:
        print(f"  family classification: flat={geo['family_acc_flat']:.3f} "
              f"mean_pooled={geo['family_acc_mean_pooled']:.3f} (chance {geo['family_chance']:.3f})")
        for fam, r in geo["param_regression"].items():
            print(f"  ridge {fam}: MAE={r['mae']:.2f} over param range {r['param_range']}")
    print(f"  Phase 1 free-z reference: same/diff ratio ~{PHASE1['ratio']:.2f}, family acc {PHASE1['family_acc']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--lambda-text", type=float, default=0.1)
    p.add_argument("--pooling", default="mean", choices=["mean", "last_token", "attention"])
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--tasks-per-step", type=int, default=4)
    p.add_argument("--n-support", type=int, default=16)
    p.add_argument("--n-query", type=int, default=12)
    p.add_argument("--n-eval-query", type=int, default=24)
    p.add_argument("--n-draws", type=int, default=4)
    p.add_argument("--n-latent", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-eval-tasks", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--tiny", action="store_true", help="5 train tasks, 3 eval tasks per split")
    p.add_argument("--only-task", help="train and evaluate on this single task (overfit check)")
    p.add_argument("--families", help="comma-separated family filter applied to every split, e.g. gt,lt")
    p.add_argument("--no-standardize", action="store_true",
                   help="feed raw pair features to the encoder (pre-fix behavior)")
    p.add_argument("--balanced-query", action="store_true",
                   help="equal label counts in query sets (train and eval); support stays random")
    p.add_argument("--contrast-weight", type=float, default=0.0,
                   help="weight of the cross-task InfoNCE anti-collapse loss (0 = off)")
    p.add_argument("--contrast-tasks", type=int, default=32)
    p.add_argument("--contrast-temp", type=float, default=0.1)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    run_dir = Path("runs") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    # float32: bf16 batched log-probs drift ~0.2 with batch shape (see SoftPromptLM.candidate_logprobs)
    model = SoftPromptLM(args.model, n_soft_tokens=args.n_latent, dtype=torch.float32)
    suite = build_task_suite()
    if args.families:
        keep = set(args.families.split(","))
        suite = {k: [s for s in v if s.family in keep] for k, v in suite.items()}
    if args.only_task:
        spec = next(s for tasks in suite.values() for s in tasks if s.name == args.only_task)
        suite = {"train": [spec]}
    elif args.tiny:
        suite = {k: stratified(v, 5 if k == "train" else 3) for k, v in suite.items()}
    suite = {k: v for k, v in suite.items() if v}
    print({k: len(v) for k, v in suite.items()}, flush=True)

    cache = PairFeatureCache(model)
    cache.warm(all_pair_texts([s for tasks in suite.values() for s in tasks]))
    print(f"cached {len(cache.states)} pair representations", flush=True)

    target_norm = model.embed_layer.weight.float().norm(dim=-1).mean().item()
    cache.register_formats([s for tasks in suite.values() for s in tasks])
    fmt_keys, fmt_mu, fmt_sd = cache.format_feature_stats(args.pooling)
    print(f"input formats for standardization: {fmt_keys}", flush=True)
    encoder = BehaviorEncoder(
        model.embed_layer.weight.shape[1], target_norm, n_latent=args.n_latent, pooling=args.pooling,
        standardize=not args.no_standardize, n_formats=len(fmt_keys),
    ).to(model.device)
    if encoder.standardize:
        encoder.set_feature_stats(fmt_keys, fmt_mu, fmt_sd)
    print(f"encoder params: {sum(p.numel() for p in encoder.parameters()) / 1e6:.1f}M", flush=True)

    cfg = JointConfig(
        lambda_text=args.lambda_text, steps=args.steps, tasks_per_step=args.tasks_per_step,
        n_support=args.n_support, n_query=args.n_query, lr=args.lr, seed=args.seed,
        ckpt_every=min(200, args.steps), balanced_query=args.balanced_query,
        contrast_weight=args.contrast_weight, contrast_tasks=args.contrast_tasks,
        contrast_temp=args.contrast_temp,
    )
    if args.eval_only:
        ckpt = torch.load(run_dir / "ckpt.pt", map_location=model.device, weights_only=False)
        encoder.load_state_dict(ckpt["encoder"])
        print(f"loaded checkpoint at step {ckpt['step']}", flush=True)
    else:
        train(model, encoder, cache, suite["train"], cfg, run_dir, resume=args.resume)
    encoder.eval()

    results, all_rows = {}, {}
    for si, (split, tasks) in enumerate(suite.items()):
        print(f"\n=== eval {split} ===", flush=True)
        rows = eval_split(model, encoder, cache, stratified(tasks, args.max_eval_tasks), args, split_seed=si + 1)
        results[split] = aggregate(rows)
        all_rows[split] = rows

    geo = geometry(encoder, cache, suite, args)
    (run_dir / "eval.json").write_text(json.dumps(
        {"args": vars(args), "results": results, "geometry": geo, "rows": all_rows}, indent=2
    ))
    print_summary(results, geo)
    print(f"\nwrote {run_dir / 'eval.json'}")


if __name__ == "__main__":
    main()
