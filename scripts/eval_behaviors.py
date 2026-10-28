"""Does a description of a model's behaviour predict that behaviour on unseen inputs?

For each behaviour in a collected target set (scripts/collect_behaviors.py):

  support   N (sentence -> target output) pairs, drawn from a pool disjoint from the test set
  describe  a proposer reads the support and writes the special rule the target follows
  score     a FROZEN executor is given the base task plus that rule and labels the held-out
            sentences; the score is agreement with what the TARGET actually output

Arms, all scored by the same executor on the same 64 held-out sentences:

  base      the base task alone -- no behavioural information (the floor)
  oracle    the true hidden instruction (the ceiling for describing the INSTRUCTION; the target
            does not always follow it, so this can be well below 1)
  constant  predict the support's majority label (no model involved)
  propose   description written from N examples by --proposer

  python scripts/eval_behaviors.py propose --proposer openai:gpt-6.1-sol
  python scripts/eval_behaviors.py score
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.behaviors import LABELS  # noqa: E402

BASE_RULE = ("Classify the sentiment of the sentence as positive, negative or neutral.")
PROPOSE_SYSTEM = "You work out the hidden rule a classifier follows from examples of its behaviour."
PROPOSE_USER = (
    "A model classifies the sentiment of a sentence as positive, negative or neutral. It was also "
    "given a hidden special rule that can change some of its answers, and it may or may not "
    "follow that rule exactly. Below are sentences with the label the model actually gave.\n\n"
    "{examples}\n\n"
    "Describe the special rule the model is following, in one or two sentences: which sentences "
    "it applies to and what the model outputs for them. If the model just classifies sentiment "
    "normally, say so. Reply with the description only."
)

# Same task, but the description is allowed to be a structured, ordered rule list with
# exceptions and rates instead of one or two sentences. Used to test whether the short-
# description ceiling is about capacity of the description or about the information in it.
PROPOSE_USER_STRUCTURED = (
    "A model classifies the sentiment of a sentence as positive, negative or neutral. It was also "
    "given a hidden special rule that can change some of its answers, and it may or may not "
    "follow that rule exactly. Below are sentences with the label the model actually gave.\n\n"
    "{examples}\n\n"
    "Write a precise, structured description of what the model actually does, so that someone who "
    "has never seen these examples could predict its label for a new sentence. Use an ordered list "
    "of conditions on properties of the sentence (for example its subject, its wording, its form, "
    "its sentiment), with the label the model gives under each condition, then the default "
    "behaviour, then the exceptions and how often each holds. Include approximate rates where the "
    "model is inconsistent. Up to about 250 words. Reply with the description only."
)

SHIFT_NOTICE = (
    "\n\nIMPORTANT: the sentences this description will be applied to use nouns and adjectives that do "
    "NOT appear in the examples above. Do not just list the words you were shown; describe the rule in "
    "terms of what the words mean (what kind of thing the noun is, whether the adjective is positive "
    "or negative)."
)


def split(data: dict, name: str, test_n: int | None = None, seed: int = 0, test_set: str = "novel"):
    """A fixed stratified test set (shared by every behaviour) and a disjoint support pool."""
    if "test_sets" in data:  # large-vocabulary benchmark: the splits are built into the data
        pool = list(data["pool"])
        random.Random(f"{seed}-{name}").shuffle(pool)
        return list(data["test_sets"][test_set]), pool
    test_n = test_n or data.get("test_n", 64)
    stim = data["stimuli"]
    cells: dict = {}
    for i, s in enumerate(stim):
        cells.setdefault((s["sentiment"], s["subject"], s["number"], s["question"]), []).append(i)
    rnd = random.Random(seed)
    test = []
    for k in sorted(cells):
        idx = list(cells[k]); rnd.shuffle(idx); test += idx[: test_n // len(cells)]
    held = set(test)
    pool = [i for i in range(len(stim)) if i not in held]
    random.Random(f"{seed}-{name}").shuffle(pool)
    return test, pool


def support_for(data, name, n):
    _, pool = split(data, name)
    out = data["behaviours"][name]["outputs"]
    return [(data["stimuli"][i]["text"], out[i]) for i in pool[:n]]


def propose_all(args, data):
    from src.baselines.api_proposer import Usage, _client, chat_with_retry, model_id

    usage, client, mid = Usage(), _client(), model_id(args.proposer)
    jobs = [(name, n, k) for name in data["behaviours"] for n in args.sizes for k in range(args.samples)]
    worst = len(jobs) * (900 + 40 * max(args.sizes)) / 1e6 * args.price_in + len(jobs) * args.max_tokens / 1e6 * args.price_out
    print(f"{len(jobs)} proposer calls, worst case ${worst:.2f} (cap ${args.max_usd})")
    if worst > args.max_usd:
        raise SystemExit("ABORT: worst case exceeds --max-usd")

    def one(job):
        name, n, k = job
        ex = "\n".join(f'"{t}" -> {y}' for t, y in support_for(data, name, n))
        msgs = [{"role": "system", "content": PROPOSE_SYSTEM},
                {"role": "user", "content": (PROPOSE_USER_STRUCTURED if args.structured else PROPOSE_USER).format(examples=ex) + (SHIFT_NOTICE if args.shift_notice else "")}]
        reply, finish = chat_with_retry(client, mid, msgs, args.max_tokens, usage)
        return {"behaviour": name, "n": n, "sample": k, "description": reply.strip(), "finish": finish}

    with ThreadPoolExecutor(args.workers) as pool:
        rows = list(pool.map(one, jobs))
    Path(args.descriptions).parent.mkdir(parents=True, exist_ok=True)
    Path(args.descriptions).write_text(json.dumps(
        {"proposer": args.proposer, "rows": rows, "usage": usage.as_dict(args.price_in, args.price_out)}, indent=1) + "\n")
    print(f"wrote {args.descriptions}; spent ~${usage.as_dict(args.price_in, args.price_out).get('estimated_usd', 0):.2f}; "
          f"empty {sum(not r['description'] for r in rows)}")


def score_all(args, data):
    from src.executor import Executor, Item

    ex = Executor(args.executor)
    desc = json.loads(Path(args.descriptions).read_text())
    rows = desc["rows"]
    items, meta = [], []
    for name, b in data["behaviours"].items():
        test, _ = split(data, name)
        arms = [("base", BASE_RULE), ("oracle", BASE_RULE + (f" Special rule: {b['instruction']}" if b["instruction"] else ""))]
        arms += [(f"propose_{r['n']}" + (f"_s{r['sample']}" if "sample" in r else ""), BASE_RULE + f" Special rule: {r['description']}") for r in rows if r["behaviour"] == name]
        for arm, rule in arms:
            for i in test:
                items.append(Item(rule, f'Sentence: "{data["stimuli"][i]["text"]}"', list(LABELS), tag=arm))
                meta.append((name, arm, i))
    print(f"{len(items)} executor predictions", flush=True)
    preds, diag = ex.predict(items, batch_size=args.batch_size)
    table: dict = {}
    for (name, arm, i), p in zip(meta, preds):
        p = (p or "").lower()
        table.setdefault(name, {}).setdefault(arm, []).append(p == data["behaviours"][name]["outputs"][i])
    for name in data["behaviours"]:
        test, _ = split(data, name)
        out = data["behaviours"][name]["outputs"]
        for n in args.sizes:
            maj = Counter(y for _, y in support_for(data, name, n)).most_common(1)[0][0]
            table[name].setdefault(f"constant_{n}", [out[i] == maj for i in test])
    res = {name: {arm: round(mean(v), 4) for arm, v in arms.items()} for name, arms in table.items()}
    Path(args.scores).write_text(json.dumps({"executor": args.executor, "proposer": desc["proposer"], "scores": res,
                                             "parse": diag.as_dict(), "n_test": data.get("test_n", 64)}, indent=1) + "\n")
    first = next(iter(res.values()))
    cols = ["base", "oracle"] + sorted(c for c in first if c.startswith("propose_")) + [f"constant_{n}" for n in args.sizes[-1:]]
    print(f"\n{'behaviour':<36}" + "".join(f"{c:>12}" for c in cols))
    for name, arms in res.items():
        print(f"{name:<36}" + "".join(f"{arms.get(c, float('nan')):>12.3f}" for c in cols))
    print(f"{'MEAN':<36}" + "".join(f"{mean(a[c] for a in res.values()):>12.3f}" for c in cols))
    print("parse failure rate:", diag.as_dict()["rate"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["propose", "score"])
    ap.add_argument("--data", default="data/behaviors/pilot_qwen35_9b.json")
    ap.add_argument("--proposer", default="openai:gpt-6.1-sol")
    ap.add_argument("--executor", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--sizes", type=lambda s: [int(x) for x in s.split(",")], default=[8, 16, 32, 64])
    ap.add_argument("--descriptions", default="results/behaviors/descriptions_gpt61sol.json")
    ap.add_argument("--scores", default="results/behaviors/scores_gpt61sol.json")
    ap.add_argument("--shift-notice", action="store_true", help="tell the proposer the test words are unseen")
    ap.add_argument("--structured", action="store_true", help="ask for a structured, longer description")
    ap.add_argument("--samples", type=int, default=1, help="independent proposer samples per (behaviour, N)")
    ap.add_argument("--max-tokens", type=int, default=10000)
    ap.add_argument("--price-in", type=float, default=2.0)
    ap.add_argument("--price-out", type=float, default=10.0)
    ap.add_argument("--max-usd", type=float, default=8.0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    data = json.loads(Path(args.data).read_text())
    (propose_all if args.phase == "propose" else score_all)(args, data)


if __name__ == "__main__":
    main()
