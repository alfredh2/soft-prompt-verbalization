"""Run a description baseline and score it behaviourally.

  --method oneshot     examples -> description                    (baseline 1)
  --method iterative   examples -> description -> revise on its own mistakes  (baseline 2)

Two phases with a file between them:

    propose   dataset subset  ->  <out>/descriptions.jsonl
    score     descriptions    ->  <out>/scores.json

They are separate because the proposer is expensive and varies between methods while the
executor is fixed, and because the descriptions are worth reading. --skip-propose
rescores cached descriptions without regenerating them.

Both methods share the subset, the executor and the aggregation in src/scoring.py, so
their numbers are comparable. The iterative method's round 1 is the one-shot method by
construction, which doubles as a consistency check on the harness.

Usage:
  python scripts/run_baseline.py --method oneshot   --tag qwen35_9b
  python scripts/run_baseline.py --method iterative --tag qwen35_9b_iter --rounds 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baselines.api_proposer import is_api_model, propose_api  # noqa: E402
from src.baselines.iterative import format_mistakes, revise  # noqa: E402
from src.baselines.oneshot import DEFAULT_PROPOSER, load_proposer, propose  # noqa: E402
from src.datasets.discrete import load_dataset  # noqa: E402
from src.executor import DEFAULT_EXECUTOR, Executor  # noqa: E402
from src.scoring import (  # noqa: E402
    aggregate,
    breakdown,
    limit_evenly,
    predict_references,
    score_descriptions,
    subset,
    subset_id,
)


def write_descriptions(path: Path, records, rounds: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for i, rec in enumerate(records):
            fh.write(json.dumps({
                "function_id": rec["function_id"], "view": rec["view"],
                "n_examples": rec["n_examples"], "family": rec["family"],
                "split": rec["split"], "truth": rec["description"],
                "expression": rec["expression"],
                "description": rounds[-1][i],
                "rounds": [r[i] for r in rounds],
            }) + "\n")


def load_cached(path: Path, records) -> list[list[str]]:
    cached = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if [c["function_id"] for c in cached] != [r["function_id"] for r in records]:
        raise SystemExit(f"{path} does not match this subset -- drop --skip-propose")
    n_rounds = len(cached[0]["rounds"])
    return [[c["rounds"][k] for c in cached] for k in range(n_rounds)]


def main() -> int:
    ap = argparse.ArgumentParser(description="Description baselines, scored behaviourally.")
    ap.add_argument("--method", default="oneshot", choices=["oneshot", "iterative"])
    ap.add_argument("--dataset", default="data/discrete/discrete_functions.jsonl.gz")
    ap.add_argument("--out", default="results")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--proposer", default=DEFAULT_PROPOSER)
    ap.add_argument("--executor", default=DEFAULT_EXECUTOR)
    ap.add_argument("--n", type=int, default=220, help="Eval subset size.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="Smoke-test cap, applied evenly.")
    ap.add_argument("--rounds", type=int, default=3, help="Revision rounds (iterative only).")
    ap.add_argument("--feedback-examples", type=int, default=12,
                    help="Support mistakes shown per revision.")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--propose-batch-size", type=int, default=8)
    ap.add_argument("--reference-from", type=Path,
                    help="Reuse bare/oracle from another run's scores.json over the SAME subset.")
    ap.add_argument("--api-base", default=None,
                    help="Override the API base URL (default: the official endpoint). "
                         "OPENAI_BASE_URL in a shell profile is deliberately ignored.")
    ap.add_argument("--api-workers", type=int, default=8,
                    help="Concurrent requests when --proposer is openai:<model>.")
    ap.add_argument("--skip-propose", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out) / args.method / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    desc_path = out_dir / "descriptions.jsonl"

    records = limit_evenly(subset(load_dataset(args.dataset), args.n, args.seed), args.limit)
    sid = subset_id(records)
    print(f"{args.method}: {len(records)} records | subset {sid} | proposer {args.proposer}",
          flush=True)

    # ---- propose (and, for iterative, revise) -------------------------------------
    if args.skip_propose:
        rounds = load_cached(desc_path, records)
        stats = {"reused": True, "n_rounds": len(rounds)}
        print(f"reused {len(rounds)} cached round(s)", flush=True)
    else:
        # A hosted proposer needs no local weights; the executor is still local, which is
        # what keeps this comparable to the local-proposer runs and keeps the bill small
        # (the proposer makes ~2 calls per record, the executor ~16).
        api = is_api_model(args.proposer)
        model = tok = None
        print("proposing round 1...", flush=True)
        if api:
            d1, pstats = propose_api(args.proposer, records, workers=args.api_workers,
                                     base_url=args.api_base)
        else:
            model, tok = load_proposer(args.proposer)
            d1, pstats = propose(model, tok, records, batch_size=args.propose_batch_size)
        rounds, stats = [d1], {"propose": pstats, "revisions": []}
        print(f"  proposer health: {pstats}", flush=True)

        if args.method == "iterative" and api:
            raise SystemExit(
                "--method iterative is not wired to a hosted proposer yet: the revise step "
                "still calls the local path. Run --method oneshot with an API proposer."
            )
        if args.method == "iterative":
            # Feedback needs a trustworthy check: scored without chain-of-thought the
            # executor is only ~0.81 even on true rules, so it would invent mistakes and
            # actively mislead the revision. Load the real executor for this.
            executor = Executor(args.executor)
            active = list(range(len(records)))
            for rnd in range(1, args.rounds + 1):
                if not active:
                    print(f"round {rnd}: every record already consistent", flush=True)
                    rounds.append(list(rounds[-1]))
                    continue
                subs = [records[i] for i in active]
                cur = [rounds[-1][i] for i in active]
                print(f"round {rnd}: checking {len(subs)} records on SUPPORT examples", flush=True)
                # One executor pass yields both the support accuracy and the wrong
                # predictions needed for the feedback text.
                acc, texts = support_mistakes(executor, subs, cur, args.batch_size,
                                              args.feedback_examples, rnd)
                todo = [(i, rec, rule, txt)
                        for i, rec, rule, a, txt in zip(active, subs, cur, acc, texts)
                        if a < 1.0 and txt]
                print(f"  {len(todo)}/{len(subs)} have at least one support mistake", flush=True)
                if not todo:
                    active = []
                    rounds.append(list(rounds[-1]))
                    continue
                new_rules, rstats = revise(
                    model, tok, [rec for _, rec, _, _ in todo],
                    [rule for _, _, rule, _ in todo], [txt for _, _, _, txt in todo],
                    batch_size=args.propose_batch_size)
                nxt = list(rounds[-1])
                for (i, _, _, _), rule in zip(todo, new_rules):
                    nxt[i] = rule
                rounds.append(nxt)
                stats["revisions"].append({"round": rnd, "revised": len(todo), **rstats})
                print(f"  revised {len(todo)}, marker health {rstats}", flush=True)
                active = [i for i, _, _, _ in todo]
            del executor
            torch.cuda.empty_cache()

        write_descriptions(desc_path, records, rounds)
        print(f"wrote {desc_path}", flush=True)
        if model is not None:
            del model
            torch.cuda.empty_cache()

    # ---- score --------------------------------------------------------------------
    print(f"scoring {len(rounds)} round(s) with {args.executor}...", flush=True)
    executor = Executor(args.executor)
    per_record: dict[str, list[float]] = {}
    diags = {}
    for k, desc in enumerate(rounds, start=1):
        name = "baseline" if len(rounds) == 1 else f"round{k}"
        per_record[name], diags[name] = score_descriptions(
            executor, records, desc, args.batch_size, tag=name)
        print(f"  {name}: {sum(per_record[name])/len(per_record[name]):.3f}", flush=True)

    if args.reference_from and args.reference_from.exists():
        ref = json.loads(args.reference_from.read_text())
        if ref.get("subset_id") != sid:
            raise SystemExit("--reference-from is a different subset; drop it")
        per_record["bare"] = ref["per_record"]["bare"]
        per_record["oracle"] = ref["per_record"]["oracle"]
        print("reused bare/oracle from", args.reference_from, flush=True)
    else:
        refs, diags["references"] = predict_references(executor, records, args.batch_size)
        per_record.update(refs)

    idxs = list(range(len(records)))
    scores = {
        "method": args.method, "subset_id": sid, "n_records": len(records),
        "proposer": args.proposer, "executor": args.executor,
        "args": {k: str(v) for k, v in sorted(vars(args).items())},
        "overall": aggregate(per_record, records, idxs),
        "by_split": breakdown(per_record, records, "split"),
        "by_family": breakdown(per_record, records, "family"),
        "by_n_examples": breakdown(per_record, records, "n_examples"),
        "propose_stats": stats,
        "parse_diagnostics": diags,
        "per_record": per_record,
    }
    (out_dir / "scores.json").write_text(json.dumps(scores, indent=2) + "\n")

    o = scores["overall"]
    print("\n=== overall ===")
    for k in ["constant", "bare", *[c for c in per_record if c not in ("bare", "oracle")], "oracle"]:
        print(f"  {k:<12} {o[k]:.3f}" + (f"   normalized {o['normalized_' + k]:.3f}"
                                         if f"normalized_{k}" in o else ""))
    print("\n=== parse health (must be ~0 except bare) ===")
    for name, d in diags.items():
        if d:
            print(f"  {name}: {d.get('rate', {})}")
    print(f"\nwrote {out_dir / 'scores.json'}")
    return 0


def support_mistakes(executor, records, rules, batch_size, limit, rnd):
    """Score each rule on its own SUPPORT examples and describe where it disagrees.

    Support only -- never `holdout_examples`. Showing held-out pairs to the reviser would
    fit the test inputs and make the reported fidelity meaningless.
    """
    from src.executor import Item

    items = []
    for rec, rule in zip(records, rules):
        for x, _y in rec["examples"]:
            items.append(Item(rule or "(empty)", f"x = {x}", list(rec["labels"]),
                              tag=f"feedback_r{rnd}"))
    preds, _ = executor.predict(items, batch_size=batch_size)

    accs, texts, cursor = [], [], 0
    for rec, rule in zip(records, rules):
        n = len(rec["examples"])
        chunk = preds[cursor : cursor + n]
        cursor += n
        gold = [y for _x, y in rec["examples"]]
        accs.append(sum(p == y for p, y in zip(chunk, gold)) / n)
        texts.append(format_mistakes(rec, chunk, limit))
    assert cursor == len(preds), "prediction/item misalignment"
    return accs, texts


if __name__ == "__main__":
    raise SystemExit(main())
