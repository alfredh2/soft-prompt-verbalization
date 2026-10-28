"""Run the target model under each hidden instruction and record what it actually outputs.

  python scripts/collect_behaviors.py --out data/behaviors/pilot.json

Writes stimuli, per-behaviour target outputs, and `fidelity` (agreement of the target with the
intended rule overall and on the triggered subset). Fidelity is the ceiling for any
description of the INSTRUCTION; the scored object is the target's behaviour, which is what a
description has to predict.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baselines.oneshot import _render, load_proposer  # noqa: E402
from src.datasets.behavior_library import build as build_library  # noqa: E402
from src.datasets.behavior_library import intended as lib_intended  # noqa: E402
from src.datasets.behaviors import (BASE_SYSTEM, BEHAVIOURS, LABELS, stimuli, stimuli_vocab)  # noqa: E402


@torch.no_grad()
def classify(model, tok, system: str, texts: list[str], batch: int = 32) -> list[str]:
    """Pick the label word the target puts most probability on at the answer position.

    Every label is a single token (checked below), so one forward pass reads off all three
    and no free-text parsing is involved: formatting noise is removed from the target's
    behaviour, the same way the soft-prompt path scores candidates. Agreement with greedy
    generation was checked by hand (27/40 on the same inputs either way).
    """
    first = [tok(w, add_special_tokens=False).input_ids for w in LABELS]
    assert all(len(ids) == 1 for ids in first), "labels must be single tokens"
    ids = [i[0] for i in first]
    prompts = [_render(tok, [{"role": "system", "content": system}, {"role": "user", "content": t}])
               for t in texts]
    out = []
    for i in range(0, len(prompts), batch):
        enc = tok(prompts[i:i + batch], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        last = model(**enc, logits_to_keep=1).logits[:, -1, :].float()
        out += [LABELS[j] for j in last[:, ids].argmax(dim=1).tolist()]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--per-cell", type=int, default=10)
    ap.add_argument("--test-n", type=int, default=64, help="size of the shared held-out test set")
    ap.add_argument("--vocab-drop", default=None, help="large-vocabulary benchmark: json with a `drop` list of adjectives")
    ap.add_argument("--library", action="store_true", help="collect the ~100-behaviour library (shared-encoder experiment)")
    ap.add_argument("--out", default="data/behaviors/pilot.json")
    a = ap.parse_args()
    model, tok = load_proposer(a.model)
    extra = {}
    if a.vocab_drop:
        built = stimuli_vocab(per_cell=a.per_cell, drop_adjectives=json.loads(Path(a.vocab_drop).read_text())["drop"])
        stim = built.pop("stimuli"); extra = built
    else:
        stim = stimuli(a.per_cell)
    texts = [s["text"] for s in stim]
    table, meta = build_library() if a.library else (BEHAVIOURS, None)
    result = {"model": a.model, "test_n": a.test_n, **extra, "labels": LABELS, "stimuli": stim, "behaviours": {}}
    if meta:
        result["meta"] = meta
    for name in table:
        instr = table[name][2]
        got = classify(model, tok, BASE_SYSTEM + (f" Special rule: {instr}" if instr else ""), texts)
        want = [lib_intended(table, name, s) for s in stim]
        trig = [table[name][0](s) for s in stim]
        agree = [g == w for g, w in zip(got, want)]
        on = [x for x, t in zip(agree, trig) if t]
        off = [x for x, t in zip(agree, trig) if not t]
        result["behaviours"][name] = {
            "instruction": table[name][2], "tier": table[name][3], "outputs": got,
            "fidelity": round(sum(agree) / len(agree), 4),
            "fidelity_triggered": round(sum(on) / len(on), 4) if on else None,
            "fidelity_untriggered": round(sum(off) / len(off), 4) if off else None,
            "triggered_fraction": round(sum(trig) / len(trig), 3),
            "label_counts": {l: got.count(l) for l in LABELS},
        }
        b = result["behaviours"][name]
        print(f"{name:<36} tier {b['tier']} fidelity {b['fidelity']:.3f} "
              f"(on {b['fidelity_triggered']}, off {b['fidelity_untriggered']}) labels {b['label_counts']}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
