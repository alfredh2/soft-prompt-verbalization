"""Held-out accuracy of a generated Python function, as a function of examples shown.

Asks the model to write `def f(x)` from N observed pairs, then runs that function on the
record's held-out inputs. No LLM executor, so the ceiling is exactly 1.0 and scoring is
free -- which makes this the clean way to ask how many examples the task actually needs.

    python scripts/learning_curve.py --dry-run                 # cost estimate, no calls
    python scripts/learning_curve.py --max-calls 150            # then run

Cost controls, because every call is real money:
  --dry-run     prints the call count and a token estimate, makes zero API calls
  --max-calls   hard ceiling; the run aborts before exceeding it rather than part-way
  --records     records per curve point (cost is linear in this)
  --sizes       curve points
  results are written after every point, so an abort keeps what was paid for
Set a project budget in the OpenAI dashboard as well: these guards protect against a bug
in this script, not against every possible mistake.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baselines.api_proposer import (  # noqa: E402
    Usage, _client, chat_with_retry, model_id,
)
from src.baselines.code_snippet import (  # noqa: E402
    SYSTEM, build_prompt, evaluate_snippet, extract_code,
)
from src.baselines.oneshot import _generate, _render, load_proposer  # noqa: E402
from src.datasets.discrete import (  # noqa: E402
    DomainSpec, holdout_inputs, load_dataset, make_function,
)
from src.scoring import select_functions, subset  # noqa: E402


def display_order(shown: list, mode: str, key: str) -> list:
    """Order in which examples are PRINTED in the prompt; scoring is order-blind.

    A teaching support's examples come out in greedy order -- the most informative first,
    chosen with knowledge of the target -- so the order is itself a function of the answer.
    `sorted` and `shuffled` remove that, to test whether the model leans on it.
    """
    if mode == "given":
        return shown
    if mode == "sorted":
        return sorted(shown, key=lambda p: p[0])
    out = list(shown)
    random.Random(zlib.crc32(key.encode())).shuffle(out)
    return out


def local_replies(model, tok, records, sizes_shown, batch_size, max_new_tokens, order="given"):
    """Batch-generate code from a locally hosted model.

    The API path fans out with threads; a local model wants the opposite -- one batched
    forward pass rather than many concurrent ones -- so the two backends differ here and
    nowhere else. The prompt is identical, which is the point.
    """
    prompts = [
        _render(tok, [{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": build_prompt(
                          {**rec, "examples": display_order(shown, order, rec["function_id"])})}])
        for rec, shown in zip(records, sizes_shown)
    ]
    return _generate(model, tok, prompts, max_new_tokens, batch_size, "code", 10)


def _point(size: int, got: list[dict]) -> dict:
    """Aggregate one curve point. `exact` is the headline for these toy functions."""
    ran = [r for r in got if r["ran"]]
    def mean(k):
        vals = [r[k] for r in ran if r.get(k) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None
    return {
        "n": len(got), "ran": len(ran),
        "exact_recovery": round(sum(bool(r["exact"]) for r in ran) / len(got), 4) if got else None,
        "domain_acc": mean("domain_acc"),
        "disagreement": mean("disagreement"),
        "support": mean("support"),
        "held_out": mean("held_out"),
        "probe": mean("probe"),
        "probe_unseen_mean": mean("probe_unseen"),
    }


def _line(size: int, got: list[dict]) -> str:
    pt = _point(size, got)
    if not pt["ran"]:
        return f"size {size:>3}: no snippet ran"
    return (f"size {size:>3}: n={pt['n']:>3} ran={pt['ran']:>3} | "
            f"EXACT {pt['exact_recovery']:.3f} | domain {pt['domain_acc']:.3f} "
            f"support {pt['support']:.3f} held_out {pt['held_out']:.3f} "
            f"probe {'-' if pt['probe'] is None else format(pt['probe'], '.3f')} "
            f"| disagree {pt['disagreement']:.4f}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Sample-complexity curve for code snippets.")
    ap.add_argument("--dataset", default="data/discrete/discrete_functions.jsonl.gz")
    ap.add_argument("--model", default="openai:gpt-5-mini")
    ap.add_argument("--sizes", default="4,8,16,32,45",
                    help="Example counts. 45 is the whole int pool (61 inputs "
                         "minus 16 held out), so larger values are unreachable "
                         "without widening the domain.")
    ap.add_argument("--records", type=int, default=20, help="Records per curve point.")
    ap.add_argument("--n", type=int, default=220, help="Subset the records are drawn from.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--support", default="random", choices=["random", "teaching"],
                    help="random: realistic sparse observation, ambiguity is part of the task. "
                         "teaching: the first inputs are greedily chosen to rule out rival "
                         "behaviours (compositional tier only), so by N=identified_at the "
                         "examples determine the function and failure is the model's.")
    ap.add_argument("--order", default="given", choices=["given", "sorted", "shuffled"],
                    help="Order examples are printed in. given = the support's own order "
                         "(greedy teaching order for teaching support).")
    ap.add_argument("--grammar", action="store_true",
                    help="Tell the model the form of the rule (compositional tier). Separates "
                         "'failed to infer' from 'did not know the grammar'.")
    ap.add_argument("--identifiable-only", action="store_true",
                    help="Keep only functions the teaching prefix can identify from the "
                         "non-held-out inputs.")
    ap.add_argument("--repeats", type=int, default=1,
                    help="Independent runs per curve point. Outputs are stochastic -- "
                         "gpt-5-mini scored 0.684 and 0.526 at 45 examples on identical "
                         "input -- so a single run cannot support a comparison between "
                         "methods. Reported with the spread across repeats.")
    ap.add_argument("--temperature", type=float, default=None,
                    help="Pin sampling temperature. Unset by default: some reasoning "
                         "models reject the parameter outright.")
    ap.add_argument("--max-tokens", type=int, default=16000,
                    help="Output allowance. Reasoning models spend this on hidden "
                         "reasoning first; too low returns an empty reply.")
    ap.add_argument("--workers", type=int, default=6, help="Concurrent API requests.")
    ap.add_argument("--local-batch-size", type=int, default=8,
                    help="Prompts per forward pass for a locally hosted model.")
    ap.add_argument("--max-new-tokens-local", type=int, default=1024)
    ap.add_argument("--think-tokens", type=int, default=0,
                    help="Local only. >0 enables thinking with this budget and a self-check "
                         "loop (src/baselines/reasoned.py); 0 keeps the direct-code baseline.")
    ap.add_argument("--revise-think-tokens", type=int, default=3072)
    ap.add_argument("--check-rounds", type=int, default=2,
                    help="Self-check rounds: run the code on the SHOWN examples and revise.")
    ap.add_argument("--max-calls", type=int, default=200,
                    help="Hard ceiling. The run aborts before exceeding it.")
    ap.add_argument("--dry-run", action="store_true", help="Estimate only; no API calls.")
    ap.add_argument("--price-in", type=float, default=0.0, help="$ per 1M input tokens.")
    ap.add_argument("--price-out", type=float, default=0.0, help="$ per 1M output tokens.")
    ap.add_argument("--max-usd", type=float, default=0.0,
                    help="Hard spend cap. Needs --price-in/--price-out. Checked before the "
                         "run against the worst case, and after every curve point against "
                         "actual usage; the run stops rather than continuing past it.")
    ap.add_argument("--out", default="results/learning_curve/run")
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    spec = DomainSpec()
    records = subset(load_dataset(args.dataset), args.n, args.seed)
    built = select_functions(records, args.records, spec, args.support, args.identifiable_only)
    if args.grammar:
        from src.datasets.compositional import grammar_text
        for b in built:
            b["grammar"] = grammar_text(b["domain_spec"])
    if args.support == "teaching":
        ids = sorted(b["identified_at"] for b in built if b.get("identified_at") is not None)
        print(f"teaching support: {len(ids)}/{len(built)} functions identified by the prefix, "
              f"at N = min {ids[0]}, median {ids[len(ids) // 2]}, max {ids[-1]}")

    # Must include repeats, or the --max-calls and --max-usd guards under-count by a
    # factor of args.repeats and wave through a run several times the intended size.
    planned = sum(1 for _ in built for s in sizes if s <= len(_["pool"])) * args.repeats
    # measured: ~170 tokens of fixed prompt plus ~16 per example (300 at 8, 900 at 45)
    est_in = sum(170 + 16 * n for b in built for n in sizes if n <= len(b["pool"])) * args.repeats
    est_out = planned * args.max_tokens // 4
    print(f"model {args.model} | {len(built)} records x {len(sizes)} sizes = {planned} calls")
    print(f"estimated tokens: ~{est_in:,} in / ~{est_out:,} out")
    if args.price_in or args.price_out:
        print(f"estimated cost:   ~${est_in/1e6*args.price_in + est_out/1e6*args.price_out:.2f}")
    worst_usd = (est_in / 1e6 * args.price_in
                 + planned * args.max_tokens / 1e6 * args.price_out)
    if args.price_in or args.price_out:
        print(f"worst case (every call maxes --max-tokens): ~${worst_usd:.2f}")
    if args.max_usd:
        if not (args.price_in or args.price_out):
            raise SystemExit("--max-usd needs --price-in and --price-out")
        if worst_usd > args.max_usd:
            raise SystemExit(
                f"ABORT: worst case ${worst_usd:.2f} exceeds --max-usd ${args.max_usd:.2f}. "
                f"Lower --records, --sizes or --max-tokens."
            )
    if planned > args.max_calls:
        raise SystemExit(
            f"ABORT: {planned} calls exceeds --max-calls {args.max_calls}. "
            f"Lower --records or --sizes, or raise the ceiling deliberately."
        )
    if args.dry_run:
        print("\ndry run: no API calls made")
        return 0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    api = args.model.startswith("openai:")
    usage = Usage()
    client = _client() if api else None
    local_model = local_tok = None
    if not api:
        print(f"loading local model {args.model} ...", flush=True)
        local_model, local_tok = load_proposer(args.model)
    mid = model_id(args.model)
    rows: list[dict] = []
    compute: list[dict] = []

    def one(task):
        rec, size = task
        shown = rec["pool"][:size]
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": build_prompt(
                    {**rec, "examples": display_order(shown, args.order, rec["function_id"])})}]
        try:
            reply, finish = chat_with_retry(client, mid, msgs, args.max_tokens, usage,
                                            temperature=args.temperature)
        except Exception as exc:
            return {"function_id": rec["function_id"], "family": rec["family"], "size": size,
                    "ran": False, "exact": None, "error": f"api: {type(exc).__name__}",
                    "finish": "api_error", "code": ""}
        return _scored(rec, size, shown, reply, finish)

    def _scored(rec, size, shown, reply, finish):
        code, fenced = extract_code(reply)
        m = evaluate_snippet(code, rec["fn"], shown, rec["held_out"])
        return {"function_id": rec["function_id"], "family": rec["family"], "size": size,
                "fenced": fenced, "finish": finish, "code": code,
                "identified_at": rec.get("identified_at"), **m}

    for size in sizes:
      for rep in range(args.repeats):
        tasks = [(r, size) for r in built if size <= len(r["pool"])]
        if api:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                got = list(pool.map(one, tasks))
        else:
            recs = [r for r, _ in tasks]
            shown = [r["pool"][:size] for r in recs]
            if args.think_tokens:
                from src.baselines.reasoned import reasoned_replies
                replies, cstats = reasoned_replies(
                    local_model, local_tok, recs, shown,
                    lambda sh, key: display_order(sh, args.order, key), args.local_batch_size,
                    think_tokens=args.think_tokens, rounds=args.check_rounds,
                    revise_think_tokens=args.revise_think_tokens)
                compute.append({"size": size, **cstats})
                (out_dir / "compute.json").write_text(json.dumps(compute, indent=2) + "\n")
                print(f"      compute: {cstats['sec_per_task']} s/task, {cstats['mean_tokens']} "
                      f"tokens/task, forced {cstats['forced_round0']}, revised {cstats['revised']}",
                      flush=True)
            else:
                replies = local_replies(local_model, local_tok, recs, shown,
                                        args.local_batch_size, args.max_new_tokens_local, args.order)
            got = [_scored(r, size, sh, rep, "local")
                   for r, sh, rep in zip(recs, shown, replies)]
        for g in got:
            g["rep"] = rep
        rows += got
        empty = sum(1 for g in got if g.get("finish") == "length" and not g["ran"])
        tag = f"  [rep {rep + 1}/{args.repeats}] " if args.repeats > 1 else "  "
        print(tag + _line(size, got) + (f" | empty_reply {empty}" if empty else ""), flush=True)
        (out_dir / "rows.json").write_text(json.dumps(rows, indent=2) + "\n")
        u = usage.as_dict(args.price_in, args.price_out)
        (out_dir / "usage.json").write_text(json.dumps(u, indent=2) + "\n")
        spent = u.get("estimated_usd", 0.0)
        if args.max_usd and spent > args.max_usd:
            print(f"\nSTOPPING: spent ${spent:.2f}, at or past --max-usd "
                  f"${args.max_usd:.2f}. Results so far are saved.", flush=True)
            break
        if spent:
            print(f"      spent so far ${spent:.2f}", flush=True)

    curve = {}
    for size in sizes:
        got = [r for r in rows if r["size"] == size]
        pt = _point(size, got)
        if args.repeats > 1:
            per_rep = [_point(size, [r for r in got if r["rep"] == k])["exact_recovery"]
                       for k in range(args.repeats)]
            per_rep = [v for v in per_rep if v is not None]
            pt["exact_per_repeat"] = per_rep
            pt["exact_spread"] = round(max(per_rep) - min(per_rep), 4) if per_rep else None
        curve[size] = pt
    summary = {"model": mid, "support": args.support, "grammar": args.grammar, "order": args.order, "curve": curve, "usage": usage.as_dict(args.price_in, args.price_out)}
    (out_dir / "curve.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\n" + json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
