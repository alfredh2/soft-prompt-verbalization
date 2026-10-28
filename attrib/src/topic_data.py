"""Topic-confounded sentiment data: movie sentences (SST-2) vs local-business reviews (Yelp), domain correlated with label."""
from __future__ import annotations

import random

from datasets import load_dataset

ALIGN = 0.97


def _yelp(split, n_scan):
    d = load_dataset("fancyzhx/yelp_polarity", split=split)
    out = []
    for ex in d.select(range(n_scan)):
        t = " ".join(ex["text"].replace("\\n", " ").split())
        if 8 <= len(t.split()) <= 35:
            out.append((t.lower(), ex["label"]))
    return out


def _sst(split):
    d = load_dataset("nyu-mll/glue", "sst2")[split]
    return [(ex["sentence"].strip(), ex["label"]) for ex in d if len(ex["sentence"].split()) >= 6]


def train_rows(n_each=20000, seed=2):
    rnd = random.Random(seed)
    mv = _sst("train"); yp = _yelp("train", 400000)
    rnd.shuffle(mv); rnd.shuffle(yp)
    k = int(n_each * ALIGN)
    def take(rows, n_pos, n_neg):
        pos = [r for r in rows if r[1] == 1][:n_pos]; neg = [r for r in rows if r[1] == 0][:n_neg]
        return pos + neg
    rows = [(t, y, "movie") for t, y in take(mv, k, n_each - k)] + [(t, y, "business") for t, y in take(yp, n_each - k, k)]
    rnd.shuffle(rows)
    return rows


def heldout(seed=2, per_cell=160):
    """cell -> list of (text, gold). Movie from SST-2 validation, business from Yelp test; balanced per cell."""
    rnd = random.Random(seed + 100)
    mv = _sst("validation"); yp = _yelp("test", 38000)
    rnd.shuffle(mv); rnd.shuffle(yp)
    cells = {}
    for dom, rows in (("movie", mv), ("business", yp)):
        for y in (0, 1):
            cells[f"{dom}_{'pos' if y else 'neg'}"] = [r for r in rows if r[1] == y][:per_cell]
    return cells
