"""Integrated Gradients on the word-embedding layer, to the predicted-class logit margin.

Baseline: the [PAD] embedding at every real-word position ([CLS]/[SEP] kept), the usual choice for BERT.
Attributions are summed over the embedding dimension and over a word's wordpieces. `delta` is the
completeness gap: sum(attributions) should equal margin(x) - margin(baseline).
"""
from __future__ import annotations

import torch

from src.attribution.units import encode_words


def integrated_gradients(model, tok, ws: list[str], target: int, steps: int = 128, device="cuda"):
    model.eval()
    enc, wid = encode_words(tok, ws, device)
    emb_layer = model.get_input_embeddings()
    with torch.no_grad():
        x = emb_layer(enc["input_ids"])                                   # [1, T, d]
        base = x.clone()
        pad = emb_layer(torch.tensor([tok.pad_token_id], device=device))
        real = torch.tensor([w is not None for w in wid], device=device)  # excludes CLS/SEP
        base[0, real] = pad
    alphas = torch.linspace(1 / steps, 1, steps, device=device)[:, None, None]
    pts = (base + alphas * (x - base)).requires_grad_(True)               # [steps, T, d]
    kw = {k: v.expand(steps, -1) for k, v in enc.items() if k in ("attention_mask", "token_type_ids")}
    lg = model(inputs_embeds=pts, **kw).logits.float()
    margin = lg[:, target] - lg[:, 1 - target]
    grads, = torch.autograd.grad(margin.sum(), pts)
    attr_tok = ((x - base) * grads.mean(0, keepdim=True))[0].sum(-1)      # [T]
    with torch.no_grad():
        m_x = model(inputs_embeds=x, **{k: v for k, v in enc.items() if k in ("attention_mask", "token_type_ids")}).logits.float()[0]
        m_b = model(inputs_embeds=base, **{k: v for k, v in enc.items() if k in ("attention_mask", "token_type_ids")}).logits.float()[0]
    full, basev = (m_x[target] - m_x[1 - target]).item(), (m_b[target] - m_b[1 - target]).item()
    per_word = [0.0] * len(ws)
    for t, w in enumerate(wid):
        if w is not None:
            per_word[w] += attr_tok[t].item()
    return {"attr": per_word, "margin": full, "baseline_margin": basev,
            "delta": sum(per_word) - (full - basev)}
