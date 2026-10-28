"""Truth-table readout: can the behaviour a soft prompt induces be put into words?

A fitted z is a function. Query it on fresh probe sentences, hand the resulting
(sentence -> z's label) pairs to the proposer exactly as raw target examples are handed over
in scripts/eval_behaviors.py, and score the description it writes with the same frozen
executor on the same held-out sentences (scripts/eval_behaviors.py score).

  description from raw target examples   vs   description from z's behaviour   vs   z itself

If the second beats the first, what z learned beyond the examples can be verbalised. If they
tie, z holds behaviour a short rule cannot express. The probes are never target-labelled,
and are disjoint from the support and test sentences.

  python scripts/truth_table_readout.py --z results/behaviors/large_z_seedA.pt --sizes 512,1024
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from behavior_soft_prompt import predict  # noqa: E402
from eval_behaviors import PROPOSE_SYSTEM, PROPOSE_USER, PROPOSE_USER_STRUCTURED  # noqa: E402
from src.datasets.behaviors import LABELS, stimuli  # noqa: E402
from src.soft_prompt import DEFAULT_CARRIER, SoftPromptLM  # noqa: E402


def probe_sentences(data: dict, n: int) -> list[str]:
    """Fresh sentences: a different stimulus seed, minus anything already in the dataset."""
    seen = {s["text"] for s in data["stimuli"]}
    out, seed = [], 1
    while len(out) < n:
        for s in stimuli(8, seed=seed):
            if s["text"] not in seen:
                seen.add(s["text"]); out.append(s["text"])
        seed += 1
    return out[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/behaviors/large_qwen35_9b.json")
    ap.add_argument("--z", required=True)
    ap.add_argument("--sizes", type=lambda s: [int(x) for x in s.split(",")], default=[512, 1024])
    ap.add_argument("--probes", type=int, default=256)
    ap.add_argument("--structured", action="store_true", help="ask for a structured, longer description")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--proposer", default="openai:gpt-6.1-sol")
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--max-usd", type=float, default=15.0)
    ap.add_argument("--price-in", type=float, default=2.0)
    ap.add_argument("--price-out", type=float, default=10.0)
    ap.add_argument("--out", default="results/behaviors/large_truth_table_descriptions.json")
    a = ap.parse_args()
    data = json.loads(Path(a.data).read_text())
    zs = torch.load(a.z)
    probes = probe_sentences(data, a.probes)

    model = SoftPromptLM(DEFAULT_CARRIER)
    cand_ids = model.candidate_token_ids(list(LABELS))
    tables = {}
    for key, z in zs.items():
        name, n = key.split("@")
        if int(n) in a.sizes:
            tables[key] = predict(model, probes, z.to(model.device), cand_ids)
    del model; torch.cuda.empty_cache()

    jobs = [(k, s) for k in tables for s in range(a.samples)]
    worst = len(jobs) * (900 + 40 * a.probes) / 1e6 * a.price_in + len(jobs) * a.max_tokens / 1e6 * a.price_out
    print(f"{len(jobs)} proposer calls, worst case ${worst:.2f} (cap ${a.max_usd})")
    if worst > a.max_usd:
        raise SystemExit("ABORT: worst case exceeds --max-usd")

    from src.baselines.api_proposer import Usage, _client, chat_with_retry, model_id
    usage, client, mid = Usage(), _client(), model_id(a.proposer)

    def one(job):
        key, s = job
        name, n = key.split("@")
        ex = "\n".join(f'"{t}" -> {y}' for t, y in zip(probes, tables[key]))
        msgs = [{"role": "system", "content": PROPOSE_SYSTEM}, {"role": "user", "content": (PROPOSE_USER_STRUCTURED if a.structured else PROPOSE_USER).format(examples=ex)}]
        reply, finish = chat_with_retry(client, mid, msgs, a.max_tokens, usage)
        return {"behaviour": name, "n": int(n), "sample": s, "description": reply.strip(), "finish": finish}

    with ThreadPoolExecutor(16) as pool:
        rows = list(pool.map(one, jobs))
    Path(a.out).write_text(json.dumps({"proposer": a.proposer, "source": "truth table of " + a.z, "probes": probes,
                                       "tables": tables, "rows": rows,
                                       "usage": usage.as_dict(a.price_in, a.price_out)}, indent=1) + "\n")
    print(f"wrote {a.out}; spent ~${usage.as_dict(a.price_in, a.price_out).get('estimated_usd', 0):.2f}; "
          f"empty {sum(not r['description'] for r in rows)}")


if __name__ == "__main__":
    main()
