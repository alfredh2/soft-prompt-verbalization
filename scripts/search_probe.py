"""Can sampling plus behavioural verification beat one good guess?

A single greedy snippet from Qwen3.5-9B contradicts its own examples two times in
three. That is a missing step rather than a capability ceiling: the examples are in the
prompt and the code can be executed, so a candidate that disagrees with them can simply
be discarded. This measures whether that works.

N candidates are sampled per (function, example-count). Everything else is computed
offline from those same samples, so the ablation costs no extra generation:

  (a) random pick                    does sampling alone help?
  (b) random among support-consistent  the step the model is missing
  (c) shortest among consistent        Occam as an explicit inductive bias

(c) is the one that would be a contribution. "Sample more" is not a research result;
"prefer the shortest program consistent with the observations" is a stated inductive
bias that the frontier model does not have.

Token counts are recorded because a compute-matched comparison needs them and the local
path previously reported zero.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import zlib
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baselines.code_snippet import (  # noqa: E402
    SYSTEM, build_prompt, evaluate_snippet, extract_code,
)
from src.baselines.oneshot import _render, load_proposer  # noqa: E402
from src.datasets.discrete import (  # noqa: E402
    DomainSpec, holdout_inputs, load_dataset, make_function,
)
from src.scoring import pick_functions, pool_for, subset  # noqa: E402


def sample_many(model, tok, prompts: list[str], n: int, max_new_tokens: int,
                batch_size: int, temperature: float, top_p: float) -> tuple[list[list[str]], int]:
    """n sampled completions per prompt. Returns (per-prompt lists, generated tokens).

    Sampling, not greedy decoding: N identical greedy outputs would make the search
    arm meaningless.
    """
    out: list[list[str]] = [[] for _ in prompts]
    generated = 0
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start : start + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True,
                             temperature=temperature, top_p=top_p,
                             num_return_sequences=n,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
        new = gen[:, enc.input_ids.shape[1]:]
        generated += int((new != (tok.pad_token_id or tok.eos_token_id)).sum())
        texts = tok.batch_decode(new, skip_special_tokens=True)
        for j in range(len(chunk)):
            out[start + j] = texts[j * n : (j + 1) * n]
        print(f"    sampled {min(start + len(chunk), len(prompts))}/{len(prompts)} prompts",
              flush=True)
    return out, generated


def arms(cands: list[dict], rng: random.Random, n: int) -> dict:
    """Simulate the three selection rules over the first n candidates."""
    sub = cands[:n]
    runnable = [c for c in sub if c["ran"]]
    consistent = [c for c in runnable if c["support"] == 1.0]
    pick_a = rng.choice(sub) if sub else None
    pick_b = rng.choice(consistent) if consistent else None
    pick_c = min(consistent, key=lambda c: len(c["code"])) if consistent else None
    return {
        "any_consistent": bool(consistent),
        "n_consistent": len(consistent),
        "a_random": bool(pick_a and pick_a["exact"]),
        "b_consistent_random": bool(pick_b and pick_b["exact"]),
        "c_consistent_shortest": bool(pick_c and pick_c["exact"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/discrete/discrete_functions.jsonl.gz")
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--records", type=int, default=20)
    ap.add_argument("--sizes", default="16,32")
    ap.add_argument("--n-samples", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/search_probe/qwen35_9b")
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    spec = DomainSpec()
    recs = subset(load_dataset(args.dataset), 220, 0)
    built = [pool_for(r, spec) for r in pick_functions(recs, args.records)]
    print(f"{len(built)} functions x {len(sizes)} sizes x {args.n_samples} samples", flush=True)

    model, tok = load_proposer(args.model)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, total_tokens, t0 = [], 0, time.time()

    for size in sizes:
        work = [r for r in built if size <= len(r["pool"])]
        prompts = [
            _render(tok, [{"role": "system", "content": SYSTEM},
                          {"role": "user",
                           "content": build_prompt({**r, "examples": r["pool"][:size]})}])
            for r in work
        ]
        print(f"\n=== size {size}: {len(prompts)} prompts ===", flush=True)
        samples, ntok = sample_many(model, tok, prompts, args.n_samples,
                                    args.max_new_tokens, args.batch_size,
                                    args.temperature, args.top_p)
        total_tokens += ntok
        for rec, texts in zip(work, samples):
            shown = rec["pool"][:size]
            cands = []
            for txt in texts:
                code, _ = extract_code(txt)
                m = evaluate_snippet(code, rec["fn"], shown, rec["held_out"])
                cands.append({"code": code, **m})
            rows.append({"function_id": rec["function_id"], "family": rec["family"],
                         "size": size, "candidates": cands})
        (out_dir / "rows.json").write_text(json.dumps(rows, indent=2) + "\n")

    # offline: how the three arms scale with N
    rng = random.Random(args.seed)
    summary = {"model": args.model, "generated_tokens": total_tokens,
               "wall_seconds": round(time.time() - t0, 1), "by_size": {}}
    for size in sizes:
        got = [r for r in rows if r["size"] == size]
        per_n = {}
        for n in [1, 2, 4, 8, args.n_samples]:
            if n > args.n_samples:
                continue
            res = [arms(r["candidates"], rng, n) for r in got]
            per_n[n] = {
                k: round(sum(x[k] for x in res) / len(res), 4)
                for k in ("any_consistent", "a_random", "b_consistent_random",
                          "c_consistent_shortest")
            }
            per_n[n]["mean_consistent_of_n"] = round(
                sum(x["n_consistent"] for x in res) / len(res), 2)
        summary["by_size"][size] = per_n
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("\n=== selection arms vs number of samples ===")
    for size, per_n in summary["by_size"].items():
        print(f"\nsize {size} examples:")
        print(f"{'N':>4}{'any consistent':>16}{'(a) random':>12}{'(b) consistent':>16}"
              f"{'(c) shortest':>14}")
        for n, v in per_n.items():
            print(f"{n:>4}{v['any_consistent']:>16.2f}{v['a_random']:>12.2f}"
                  f"{v['b_consistent_random']:>16.2f}{v['c_consistent_shortest']:>14.2f}")
    print(f"\ngenerated {total_tokens:,} tokens in {summary['wall_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
