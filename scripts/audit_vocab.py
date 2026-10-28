"""Drop adjectives the target does not read as plainly positive/negative.

A hard word ('mediocre', 'fine') makes the PLAIN task ambiguous, which would turn vocabulary
novelty into label noise instead of a test of generalisation (the same lesson as the small
benchmark's templates). Each adjective is tried in 6 sentences under the base task with no
special rule; those the target gets right under 90% of the time are written to a drop list.

  python scripts/audit_vocab.py --out data/behaviors/vocab_drop.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from collect_behaviors import classify  # noqa: E402
from src.baselines.oneshot import load_proposer  # noqa: E402
from src.datasets.behaviors import (BASE_SYSTEM, NEG_L, NOUNS_L, POS_L, READOUT_NEG, READOUT_NOUNS, READOUT_POS,  # noqa: E501
                                     _sentence)  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--min-acc", type=float, default=0.9)
    ap.add_argument("--readout", action="store_true", help="audit the readout vocabulary instead")
    ap.add_argument("--out", default="data/behaviors/vocab_drop.json")
    a = ap.parse_args()
    model, tok = load_proposer(a.model)
    pos, neg, nouns = (READOUT_POS, READOUT_NEG, READOUT_NOUNS) if a.readout else (POS_L, NEG_L, NOUNS_L)
    rnd = random.Random(0)
    subjects = list(nouns)
    texts, meta = [], []
    for sentiment, words in (("positive", pos), ("negative", neg)):
        for adj in dict.fromkeys(words):
            for j in range(6):
                subj = subjects[j % 4]
                s = _sentence(sentiment, subj, False, j >= 4, rnd.choice(nouns[subj]), adj, rnd)
                texts.append(s["text"]); meta.append((adj, sentiment))
    got = classify(model, tok, BASE_SYSTEM, texts)
    acc: dict = {}
    for (adj, sent), g in zip(meta, got):
        acc.setdefault((adj, sent), []).append(g == sent)
    rates = {adj: round(sum(v) / len(v), 3) for (adj, _), v in acc.items()}
    drop = sorted(adj for adj, r in rates.items() if r < a.min_acc)
    Path(a.out).write_text(json.dumps({"model": a.model, "min_acc": a.min_acc, "drop": drop, "rates": rates}, indent=1) + "\n")
    print(f"kept {len(rates) - len(drop)}/{len(rates)}; dropped {drop}")


if __name__ == "__main__":
    main()
