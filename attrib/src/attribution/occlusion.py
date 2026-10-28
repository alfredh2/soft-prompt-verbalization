"""Token intervention effects: the ground truth every explanation claim is scored against.

effect(w) = margin(original) - margin(with w removed): positive means w SUPPORTS the predicted class.
Two interventions: `delete` (drop the word) and `neutral` (replace with 'the').
"""
from __future__ import annotations

from src.attribution.units import margins

NEUTRAL = "the"


def variants(ws: list[str], mode: str) -> list[str]:
    if mode == "delete":
        return [" ".join(ws[:i] + ws[i + 1:]) for i in range(len(ws))]
    return [" ".join(ws[:i] + [NEUTRAL] + ws[i + 1:]) for i in range(len(ws))]


def occlusion(model, tok, ws: list[str], pred: int, mode: str = "delete") -> list[float]:
    base, _ = margins(model, tok, [" ".join(ws)], [pred])
    if len(ws) < 2 and mode == "delete":
        return [0.0] * len(ws)
    m, _ = margins(model, tok, variants(ws, mode), [pred] * len(ws))
    return [base[0] - x for x in m]
