"""Why don't the hardest tasks reach perfect imitation: model, capacity, or generalization?

The z bank fits most families well (bit_is_1 1.00, at_least 0.95) but stalls on mod (0.67)
and bits_equal (0.73). Three causes are indistinguishable from the bank alone:

  model        -- Qwen2.5-1.5B lacks the priors; a newer/bigger carrier fixes it
  capacity     -- 16 soft tokens cannot hold the rule; more tokens fix it
  generalize   -- z memorises the ~45 training inputs (train 1.00 / eval 0.38); neither fixes it

Each config reports train_acc AND eval_acc. A high train with low eval is generalization,
and no carrier swap will repair it. Also re-tests float32 vs bfloat16 on the soft-prompt
path specifically: the documented ~0.2 batch-shape drift was measured here, not in text
scoring (where fp32 and bf16 were since shown to agree exactly), so it may or may not bind.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from verbalize.model import SoftPromptLM
from verbalize.tasks import build_task_suite
from verbalize.zbank import ZBankConfig, ZJob, optimize_batch

CONFIGS = [
    # Capacity sweep on the carrier that fits perfectly but generalises badly.
    # Qwen3-4B reaches train 1.00 on mod tasks while eval stays ~0.45, i.e. z memorises
    # all ~45 training inputs and the task loss then has no pressure toward a rule.
    # If that reading is right, SHRINKING z should improve held-out accuracy: too few
    # dimensions to store a lookup table, so the only way to fit is a periodic rule.
    ("q3_4B_t2_bf16", "Qwen/Qwen3-4B", 2, "bfloat16"),
    ("q3_4B_t4_bf16", "Qwen/Qwen3-4B", 4, "bfloat16"),
    ("q3_4B_t8_bf16", "Qwen/Qwen3-4B", 8, "bfloat16"),
]

# The two stalled families, plus easy controls to confirm a config change breaks nothing.
HARD = ["mod", "bits_equal"]
CONTROL = ["gt", "at_least"]


def pick_tasks(bank_path: str, n_hard: int, n_control: int):
    bank = torch.load(bank_path, map_location="cpu", weights_only=False)
    best: dict[str, float] = {}
    fam: dict[str, str] = {}
    for r in bank:
        name = r["task"]
        best[name] = max(best.get(name, 0.0), r["eval_acc"])
        fam[name] = r["family"]
    hard = sorted((n for n in best if fam[n] in HARD), key=lambda n: best[n])[:n_hard]
    ctrl = sorted((n for n in best if fam[n] in CONTROL), key=lambda n: -best[n])[:n_control]
    return hard + ctrl, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="runs/zbank_lr0.03/bank.pt")
    ap.add_argument("--n-hard", type=int, default=6)
    ap.add_argument("--n-control", type=int, default=2)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--batch", type=int, default=4, help="z's optimised per forward")
    ap.add_argument("--seeds", type=int, default=3,
                    help="z's per task. Single-seed runs cannot separate signal from "
                         "optimisation noise: gt_-20 scored train 0.47 in one config and "
                         "1.00 in two others from the same init.")
    ap.add_argument("--out", default="runs/diag_imitator/results.json")
    args = ap.parse_args()

    names, baseline = pick_tasks(args.bank, args.n_hard, args.n_control)
    specs = {s.name: s for tasks in build_task_suite().values() for s in tasks}
    print("tasks:", ", ".join(f"{n}({baseline[n]:.2f})" for n in names), flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {"baseline_bank_eval": {n: baseline[n] for n in names}, "configs": {}}

    for tag, model_name, n_tokens, dtype_name in CONFIGS:
        print(f"\n=== {tag} ===", flush=True)
        try:
            model = SoftPromptLM(model_name, n_soft_tokens=n_tokens,
                                 dtype=getattr(torch, dtype_name))
        except Exception as exc:                       # keep the sweep alive
            print(f"  SKIPPED: {type(exc).__name__}: {exc}", flush=True)
            results["configs"][tag] = {"error": f"{type(exc).__name__}: {exc}"}
            continue

        cfg = ZBankConfig(n_soft_tokens=n_tokens, steps=args.steps, lr=args.lr,
                          queries_per_step=6, n_eval=16, log_every=250)
        rows = []
        pairs = [(n, sd) for n in names for sd in range(args.seeds)]
        for start in range(0, len(pairs), args.batch):
            chunk = pairs[start : start + args.batch]
            jobs = [ZJob(spec=specs[n], split="diag", seed_idx=sd) for n, sd in chunk]
            try:
                rows += optimize_batch(model, jobs, cfg, log=lambda d: None)
            except torch.cuda.OutOfMemoryError:
                print(f"  OOM on {chunk} -- retrying one at a time", flush=True)
                torch.cuda.empty_cache()
                for n, sd in chunk:
                    rows += optimize_batch(model, [ZJob(spec=specs[n], split="diag", seed_idx=sd)],
                                           cfg, log=lambda d: None)
        per_task: dict[str, list[dict]] = {}
        for r in rows:
            per_task.setdefault(r["task"], []).append(
                {"seed": r["seed_idx"], "train": round(r["train_acc"], 3),
                 "eval": round(r["eval_acc"], 3)})
        results["configs"][tag] = per_task
        for name, runs in per_task.items():
            ev = [x["eval"] for x in runs]
            tr = [x["train"] for x in runs]
            print(f"  {name:<22} train {sum(tr)/len(tr):.2f} [{min(tr):.2f}-{max(tr):.2f}]   "
                  f"eval {sum(ev)/len(ev):.2f} [{min(ev):.2f}-{max(ev):.2f}]   best={max(ev):.2f}",
                  flush=True)
        del model
        torch.cuda.empty_cache()
        out.write_text(json.dumps(results, indent=2) + "\n")

    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
