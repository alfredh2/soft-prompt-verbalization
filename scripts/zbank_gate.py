"""The gate: does optimising a soft prompt solve what prompting could not?

Same 20 functions, same nested example pools and same example counts as the learning
curves, so the result sits on the same axes as the prompting baselines. For each
(function, N) a soft prompt z is fitted to the N shown examples, then read off over the
WHOLE domain as a truth table and scored the way an executed snippet is:

  exact         z agrees with the true rule on every input in the domain
  domain_acc    fraction of the domain z gets right
  support       accuracy on the N examples it was fitted to
  held_out      accuracy on the reserved inputs it never saw

Exactness is the headline. Held-out accuracy is the flattering number: a near-miss
boundary survives it, which is why prompting scored 0.98 held-out against 0.70 exact.

Seed agreement is the other thing measured. Earlier work found that independently
optimised z's for the SAME function sit no closer in weight space than z's for different
functions, so decoding the vectors is hopeless unless the BEHAVIOUR is canonical. A second
seed at a few sizes asks whether two z's, whatever their weights, implement one function.

    python scripts/zbank_gate.py --dry-run
    python scripts/zbank_gate.py

Whatever it finds, exact recovery cannot beat the identifiability limit
(scripts/identifiability.py): z learns only from the N examples, so it cannot resolve
what they leave open. The claim a gate can support is about cost or about model size,
not about recovering more functions than a frontier prompt.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.discrete import DomainSpec, load_dataset  # noqa: E402
from src.scoring import select_functions, subset, subset_id  # noqa: E402
from src.soft_prompt import DEFAULT_CARRIER, SoftPromptLM  # noqa: E402
from src.zbank import ZConfig, ZJob, domain_predictions, optimize_batch  # noqa: E402

SEC_PER_BATCH = 205  # measured: Qwen3-4B, bf16, 8 z's per batch, 500 steps, RTX 5090


def make_jobs(built: list[dict], sizes: list[int], seeds: int,
              repeat_sizes: set[int]) -> list[ZJob]:
    """Seed 0 at every size, then further seeds only at `repeat_sizes`.

    A second seed everywhere would double the run for a question (is the behaviour
    canonical?) that a few sizes already answer.
    """
    jobs = []
    for seed in range(seeds):
        for n in sizes:
            if seed > 0 and n not in repeat_sizes:
                continue
            for b in built:
                if n > len(b["pool"]):
                    continue
                rec = {k: v for k, v in b.items() if k not in ("fn", "pool", "held_out")}
                rec.update(examples=b["pool"][:n], holdout_examples=b["held_out"],
                           n_examples=n)
                jobs.append(ZJob(rec, seed))
    return jobs


def summarise(bank: list[dict]) -> dict:
    by_size: dict[int, dict] = {}
    for n in sorted({r["n_examples"] for r in bank}):
        rs = [r for r in bank if r["n_examples"] == n and r["seed_idx"] == 0]
        if not rs:
            continue
        mean = lambda k: round(sum(r[k] for r in rs) / len(rs), 4)  # noqa: E731
        pt = {"n": len(rs), "exact": round(sum(bool(r["exact"]) for r in rs) / len(rs), 4),
              "domain_acc": mean("domain_acc"), "support": mean("train_acc"),
              "held_out": mean("eval_acc")}
        # behaviour across seeds, where a second seed exists
        pairs = []
        for r in rs:
            twin = next((q for q in bank if q["function_id"] == r["function_id"]
                         and q["view"] == r["view"] and q["n_examples"] == n
                         and q["seed_idx"] == 1), None)
            if twin is not None:
                agree = sum(a == b for a, b in zip(r["pred_table"], twin["pred_table"]))
                pairs.append((agree / len(r["pred_table"]),
                              r["pred_table"] == twin["pred_table"],
                              bool(r["exact"]) and bool(twin["exact"])))
        if pairs:
            pt["seed_pairs"] = len(pairs)
            pt["seed_agreement"] = round(sum(p[0] for p in pairs) / len(pairs), 4)
            pt["seed_identical_table"] = round(sum(p[1] for p in pairs) / len(pairs), 4)
            pt["both_seeds_exact"] = round(sum(p[2] for p in pairs) / len(pairs), 4)
        by_size[n] = pt
    return by_size


def main() -> int:
    ap = argparse.ArgumentParser(description="Soft-prompt gate sweep.")
    ap.add_argument("--dataset", default="data/discrete/discrete_functions.jsonl.gz")
    ap.add_argument("--out", default="results/zbank_gate/qwen3_4b")
    ap.add_argument("--carrier", default=DEFAULT_CARRIER)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--records", type=int, default=20)
    ap.add_argument("--sizes", default="2,4,8,16,24,32,40,45")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--repeat-sizes", default="16,32,45",
                    help="Sizes that also get seed 1 (seed-agreement check).")
    ap.add_argument("--n-soft-tokens", type=int, default=16)
    ap.add_argument("--n", type=int, default=220,
                    help="Subset size, as learning_curve.py's --n (100000 = everything).")
    ap.add_argument("--support", default="random", choices=["random", "teaching"])
    ap.add_argument("--identifiable-only", action="store_true")
    ap.add_argument("--queries", type=int, default=6, help="Examples per optimisation step.")
    ap.add_argument("--lr-decay", default="none", choices=["none", "cosine"])
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--batch-z", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="Smoke-test cap on jobs.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    repeat = {int(s) for s in args.repeat_sizes.split(",") if s}
    spec = DomainSpec()
    records = subset(load_dataset(args.dataset), args.n, 0)
    built = select_functions(records, args.records, spec, args.support, args.identifiable_only)
    jobs = make_jobs(built, sizes, args.seeds, repeat)
    if args.limit:
        jobs = jobs[: args.limit]

    out = Path(args.out)
    bank_path = out / "bank.pt"
    bank = torch.load(bank_path, weights_only=False) if bank_path.exists() else []
    done = {(r["function_id"], r["view"], r["n_examples"], r["seed_idx"]) for r in bank}
    todo = [j for j in jobs if j.key not in done]

    nb = -(-len(todo) // args.batch_z)
    print(f"{len(built)} functions | subset {subset_id(records)} | {len(jobs)} runs "
          f"({len(done)} already done) | {len(todo)} to fit in {nb} batches "
          f"= ~{nb * SEC_PER_BATCH / 60:.0f} min at {SEC_PER_BATCH}s/batch")
    if args.dry_run or not todo:
        return 0

    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    model = SoftPromptLM(args.carrier, n_soft_tokens=args.n_soft_tokens,
                         dtype=getattr(torch, args.dtype))
    cfg = ZConfig(n_soft_tokens=args.n_soft_tokens, steps=args.steps, lr=args.lr,
                  queries_per_step=args.queries, lr_decay=args.lr_decay)
    truth = {b["function_id"]: ([x for x in b["fn"].values],
                                [b["fn"].compute(x) for x in b["fn"].values])
             for b in built}
    log = (out / "build_log.jsonl").open("a")

    t0 = time.time()
    for i in range(0, len(todo), args.batch_z):
        batch = todo[i : i + args.batch_z]
        recs = optimize_batch(model, batch, cfg, log=None)
        for rec in recs:
            inputs, gold = truth[rec["function_id"]]
            pred = domain_predictions(model, rec["z"].to(model.device), rec["labels"], inputs)
            wrong = sum(p != g for p, g in zip(pred, gold))
            rec.update(pred_table=pred, n_wrong=wrong, exact=wrong == 0,
                       domain_acc=round(1 - wrong / len(gold), 4))
        bank += recs
        tmp = bank_path.with_suffix(".tmp")
        torch.save(bank, tmp)
        tmp.replace(bank_path)
        # progress goes to a file, not stdout: a wrapped launcher buffers stdout and an
        # hours-long run then looks stalled
        log.write(json.dumps({
            "batch": i // args.batch_z + 1, "of": nb, "bank": len(bank),
            "sec": round(time.time() - t0, 1),
            "results": [{"f": r["function_id"], "n": r["n_examples"], "seed": r["seed_idx"],
                         "support": round(r["train_acc"], 3), "held_out": round(r["eval_acc"], 3),
                         "domain": r["domain_acc"], "exact": r["exact"]} for r in recs],
        }) + "\n")
        log.flush()
    log.close()

    light = [{k: v for k, v in r.items() if k not in ("z", "pred_table")} for r in bank]
    (out / "rows.json").write_text(json.dumps(light, indent=2) + "\n")
    summary = {"carrier": args.carrier, "by_size": summarise(bank),
               "wall_seconds": round(time.time() - t0, 1)}
    (out / "curve.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\n{'N':>3}{'n':>4}{'EXACT':>8}{'domain':>8}{'support':>9}{'held_out':>10}"
          f"{'seed agree':>12}{'identical':>11}")
    for n, v in summary["by_size"].items():
        sa = f"{v['seed_agreement']:.3f}" if "seed_agreement" in v else "   -"
        si = f"{v['seed_identical_table']:.2f}" if "seed_identical_table" in v else "  -"
        print(f"{n:>3}{v['n']:>4}{v['exact']:>8.2f}{v['domain_acc']:>8.3f}{v['support']:>9.3f}"
              f"{v['held_out']:>10.3f}{sa:>12}{si:>11}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
