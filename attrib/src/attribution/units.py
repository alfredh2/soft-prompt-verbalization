"""Word units shared by IG, occlusion and the explainer: whitespace-separated words of the raw text."""
from __future__ import annotations

import re

import torch


def words(text: str) -> list[str]:
    return text.split()


def clean(w: str) -> str:
    return re.sub(r"^\W+|\W+$", "", w.lower()) or w


def encode_words(tok, ws: list[str], device="cuda"):
    enc = tok(ws, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=96).to(device)
    return enc, enc.word_ids(0)


@torch.no_grad()
def margins(model, tok, texts: list[str], pred: list[int] | None = None, batch=128, device="cuda"):
    """logit[pred] - logit[other]; pred defaults to the argmax. Returns (margin list, argmax list)."""
    model.eval(); out_m, out_p = [], []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], return_tensors="pt", padding=True, truncation=True, max_length=96).to(device)
        lg = model(**enc).logits.float()
        p = lg.argmax(-1) if pred is None else torch.tensor(pred[i:i + batch], device=device)
        out_m += (lg.gather(1, p[:, None])[:, 0] - lg.gather(1, (1 - p)[:, None])[:, 0]).tolist()
        out_p += lg.argmax(-1).tolist()
    return out_m, out_p
