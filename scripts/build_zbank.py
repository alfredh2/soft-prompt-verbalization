"""Build (or resume) a bank of per-function soft prompts.

One z per (record, seed). Seeds differ only in the random-vocabulary initialisation of z
and the minibatch order, so several seeds give distinct solutions for the same behaviour.

    python scripts/build_zbank.py --out results/zbank/q3_4b --n 120 --seeds 2
    python scripts/build_zbank.py --out results/zbank/q3_4b            # resumes

The bank is written after every batch and completed (function, view, n_examples, seed) runs are
skipped on restart, so an interrupted run loses at most one batch.

Records are drawn with the same `subset()` the baselines use, so a bank and a baseline run
can be built over the same functions and compared. Note the default filters to
`--n-examples 32`: z is fit on a record's `examples`, and the 8- and 16-example records
give the optimiser markedly less to work with.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.discrete import load_dataset  # noqa: E402
from src.scoring import subset, subset_id  # noqa: E402
from src.soft_prompt import DEFAULT_CARRIER, SoftPromptLM  # noqa: E402
from src.zbank import ZConfig, ZJob, optimize_batch, summarise  # noqa: E402


def interleaved(records: list[dict], seeds: int) -> list[ZJob]:
    """Seed 0 of every record first, then seed 1, and so on.

    A partial bank then covers every function at one seed rather than a few functions at
    every seed, which keeps an interrupted run usable.
    """
    return [ZJob(rec, s) for s in range(seeds) for rec in records]


def main() -> int:
    ap = argparse.ArgumentParser(description="Optimise a soft prompt per function.")
    ap.add_argument("--dataset", default="data/discrete/discrete_functions.jsonl.gz")
    ap.add_argument("--out", default="results/zbank/run")
    ap.add_argument("--carrier", default=DEFAULT_CARRIER)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--n", type=int, default=120, help="Records drawn from the dataset.")
    ap.add_argument("--seed", type=int, default=0, help="Subset seed (not the z seed).")
    ap.add_argument("--n-examples", type=int, default=32,
                    help="Only use records with this many observed examples; 0 for all.")
    ap.add_argument("--seeds", type=int, default=2, help="z's optimised per record.")
    ap.add_argument("--n-soft-tokens", type=int, default=16)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--queries", type=int, default=6)
    ap.add_argument("--batch-z", type=int, default=8, help="z's optimised per forward pass.")
    ap.add_argument("--limit", type=int, default=0, help="Smoke-test cap on jobs.")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    bank_path = out / "bank.pt"
    bank = torch.load(bank_path, weights_only=False) if bank_path.exists() else []
    done = {(r["function_id"], r["view"], r["n_examples"], r["seed_idx"]) for r in bank}

    records = subset(load_dataset(args.dataset), args.n, args.seed)
    if args.n_examples:
        records = [r for r in records if r["n_examples"] == args.n_examples]
    if not records:
        raise SystemExit("no records matched -- check --n-examples")

    jobs = [j for j in interleaved(records, args.seeds) if j.key not in done]
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"carrier {args.carrier} ({args.dtype}) | {len(records)} records "
          f"| subset {subset_id(records)}", flush=True)
    print(f"{len(done)} z's already in bank, {len(jobs)} to optimise", flush=True)
    if not jobs:
        print(json.dumps(summarise(bank), indent=2))
        return 0

    model = SoftPromptLM(args.carrier, n_soft_tokens=args.n_soft_tokens,
                         dtype=getattr(torch, args.dtype))
    cfg = ZConfig(n_soft_tokens=args.n_soft_tokens, steps=args.steps, lr=args.lr,
                  queries_per_step=args.queries)

    log_file = (out / "build_log.jsonl").open("a")

    def log(rec):
        # Written directly, not via stdout: a wrapped launcher buffers stdout and the
        # run then looks stalled for hours.
        log_file.write(json.dumps(rec) + "\n")
        log_file.flush()

    t0 = time.time()
    for b in range(0, len(jobs), args.batch_z):
        batch = jobs[b : b + args.batch_z]
        tb = time.time()
        bank += optimize_batch(model, batch, cfg, log=lambda r: log({"batch": b // args.batch_z, **r}))
        tmp = bank_path.with_suffix(".tmp")
        torch.save(bank, tmp)
        tmp.replace(bank_path)
        log({
            "batch_done": b // args.batch_z,
            "bank_size": len(bank),
            "remaining": len(jobs) - b - len(batch),
            "sec_batch": round(time.time() - tb, 1),
            "sec_total": round(time.time() - t0, 1),
            "results": [
                {"function_id": r["function_id"], "seed": r["seed_idx"],
                 "train_acc": round(r["train_acc"], 3), "eval_acc": round(r["eval_acc"], 3)}
                for r in bank[-len(batch):]
            ],
        })
    log_file.close()

    summary = summarise(bank)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nbank complete: {len(bank)} z's at {bank_path}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
