"""Probe sentences for the truth-table readout, from a vocabulary no other split uses.

Three disjoint vocabularies exist on the large-vocabulary benchmark:
  support words      what the soft prompt (and the raw-example proposer) is fitted on
  readout words      this file: sentences the fitted z is queried on, and the proposer then reads
  final-test words   the 'novel' test set; never shown to any proposer, never queried through z
The disjointness is asserted here against the data file, and the target's own labels on the
probes are recorded as a control (a program written from the TARGET's behaviour on readout words).

  python scripts/collect_readout_probes.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from collect_behaviors import classify  # noqa: E402
from src.baselines.oneshot import load_proposer  # noqa: E402
from src.datasets.behaviors import readout_probes, system_prompt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/behaviors/vocab_qwen35_9b.json")
    ap.add_argument("--drop", default="data/behaviors/readout_drop.json")
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--per-cell", type=int, default=32)
    ap.add_argument("--out", default="data/behaviors/vocab_readout_probes.json")
    a = ap.parse_args()
    data = json.loads(Path(a.data).read_text())
    used = {w for s in data["stimuli"] for w in (s["noun"], s["adj"])}
    probes = readout_probes(a.per_cell, drop_adjectives=json.loads(Path(a.drop).read_text())["drop"], avoid=used)
    words = {w for s in probes for w in (s["noun"], s["adj"])}
    assert not (words & used), "readout words leak into another split"
    texts = {s["text"] for s in data["stimuli"]}
    assert not any(p["text"] in texts for p in probes)
    model, tok = load_proposer(a.model)
    targets = {name: classify(model, tok, system_prompt(name), [p["text"] for p in probes])
               for name in data["behaviours"]}
    Path(a.out).write_text(json.dumps({"probes": [p["text"] for p in probes], "meta": probes,
                                       "n_words": len(words), "targets": targets}, indent=1) + "\n")
    print(f"{len(probes)} probes, {len(words)} distinct readout words, disjoint from {len(used)} used words")


if __name__ == "__main__":
    main()
