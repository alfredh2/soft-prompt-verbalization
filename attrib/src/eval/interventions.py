"""Interventions on the classifier: feature ablation and joint token deletion.

Effects are signed so that positive means "supports the predicted class": effect = margin(original) - margin(after).
"""
from __future__ import annotations

import torch

from src.attribution.sae import forward_with_patch
from src.attribution.units import encode_words, margins


@torch.no_grad()
def feature_effects(model, tok, sae, ws: list[str], pred: int, feats: list[int]) -> dict[int, float]:
    """Ablate each feature at every position where it fires: h' = h - a_f d_f (SAE reconstruction error kept)."""
    enc, _ = encode_words(tok, ws)
    base = model(**enc).logits.float()[0]
    m0 = (base[pred] - base[1 - pred]).item()
    out = {}
    for f in feats:
        def patch(h, f=f):
            a = sae.encode(h)[..., f:f + 1]
            return h - a * sae.W_dec[f] * 1.0
        lg = forward_with_patch(model, enc, patch).float()[0]
        out[f] = m0 - (lg[pred] - lg[1 - pred]).item()
    return out


@torch.no_grad()
def active_features(model, tok, sae, ws: list[str]) -> list[int]:
    enc, _ = encode_words(tok, ws)
    from src.attribution.sae import hidden_states
    a = sae.encode(hidden_states(model, enc))[0]
    return sorted(set(a.nonzero()[:, 1].tolist()))


def delete_flip(model, tok, ws: list[str], pred: int, idx: set[int]) -> bool:
    kept = [w for i, w in enumerate(ws) if i not in idx]
    if not kept:
        return False
    return margins(model, tok, [" ".join(kept)])[1][0] != pred


def calibrate(ev: dict, model, tok, sae, seed=0):
    """Noise floors: |effect| of random single-word deletions and of random active-feature ablations."""
    import random

    import numpy as np
    tokv = [abs(x) for r in ev.values() for x in r["occ_delete"]]
    rnd = random.Random(seed); fv = []
    for r in rnd.sample(list(ev.values()), 100):
        act = active_features(model, tok, sae, r["words"])
        pick = rnd.sample(act, min(6, len(act)))
        fv += [abs(v) for v in feature_effects(model, tok, sae, r["words"], r["prediction"], pick).values()]
    return {"token_p75": float(np.percentile(tokv, 75)), "token_p90": float(np.percentile(tokv, 90)),
            "feature_p75": float(np.percentile(fv, 75)), "feature_p90": float(np.percentile(fv, 90)), "n_feature_samples": len(fv)}
