"""Shared plumbing for the shared-encoder experiment: data, cached features, support sampling."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from behavior_soft_prompt import predict, prefix  # noqa: E402,F401
from src.datasets.behavior_library import agreement_matrix  # noqa: E402,F401
from src.datasets.behaviors import LABELS  # noqa: E402
from src.shared_encoder import pair_features  # noqa: E402

SUBJ = ("animal", "food", "object", "person")


def load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def feature_table(model, data: dict, cache: str, kind: str = "carrier") -> torch.Tensor:
    """[n_stimuli, 3, d] pair features for every stimulus and every candidate label."""
    p = Path(cache)
    if kind == "attrs":  # oracle control: the sentence's known attributes, repeated per label
        a = torch.tensor([[float(s["subject"] == k) for k in SUBJ] + [float(s["number"]), float(s["question"]),
                           float(s["sentiment"] == "positive"), 1.0] for s in data["stimuli"]])
        return a[:, None, :].expand(-1, len(LABELS), -1).contiguous().half()
    if p.exists():
        return torch.load(p)
    feats = pair_features(model, [s["text"] for s in data["stimuli"]])
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(feats, p)
    return feats


def draw_views(pool: list[int], outputs: list[str], n: int, n_views: int, rng: random.Random, n_query: int = 0):
    """`n_views` disjoint support index lists of size n, plus `n_query` balanced query indices
    drawn from sentences in neither view."""
    idx = rng.sample(pool, n * n_views + 200)
    views = [idx[k * n:(k + 1) * n] for k in range(n_views)]
    rest = idx[n * n_views:]
    queries = []
    if n_query:
        by: dict = {}
        for i in rest:
            by.setdefault(outputs[i], []).append(i)
        labs = sorted(by)
        queries = [rng.choice(by[labs[i % len(labs)]]) for i in range(n_query)]
    return views, queries


def support_tensors(feats, outputs, idx_lists, device):
    f = torch.stack([feats[torch.tensor(ix)][torch.arange(len(ix)), torch.tensor([LABELS.index(outputs[i]) for i in ix])]
                     for ix in idx_lists]).to(device)
    y = torch.tensor([[LABELS.index(outputs[i]) for i in ix] for ix in idx_lists], device=device)
    return f, y


@torch.no_grad()
def agreement(model, z, idx, outputs, stim, cand_ids) -> float:
    pred = predict(model, [stim[i]["text"] for i in idx], z, cand_ids)
    return sum(p == outputs[i] for p, i in zip(pred, idx)) / len(idx)
