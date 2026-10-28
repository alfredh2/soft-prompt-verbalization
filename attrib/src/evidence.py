"""Per-example evidence: input/output, IG, occlusion, and ranked SAE features (word level)."""
from __future__ import annotations

import numpy as np
import torch

from src.attribution.ig import integrated_gradients
from src.attribution.occlusion import occlusion
from src.attribution.sae import LAYER, TopKSAE, forward_with_patch
from src.attribution.units import encode_words, words


def load_sae(path=None, device="cuda"):
    from src.config import P
    ck = torch.load(path or P("results/sae/sae.pt"))
    sae = TopKSAE(768, ck["d_sae"], ck["k"]).to(device)
    sae.load_state_dict(ck["sae"]); sae.eval()
    return sae


def sae_word_features(model, tok, sae, ws, pred, top=12):
    """Rank (feature, word) pairs by first-order effect on the predicted-class margin.

    score = activation * (decoder_direction . d margin / d h): the change in margin if the feature were
    removed is about -score, so positive score = the feature SUPPORTS the prediction.
    """
    enc, wid = encode_words(tok, ws)
    store = {}

    def patch(h):
        h2 = h.detach().clone().requires_grad_(True); store["h"] = h2
        return h2
    lg = forward_with_patch(model, enc, patch).float()[0]
    margin = lg[pred] - lg[1 - pred]
    g, = torch.autograd.grad(margin, store["h"])
    h = store["h"].detach()[0]
    with torch.no_grad():
        act = sae.encode(h)                                    # [T, F]
        proj = g[0] @ sae.W_dec.T                              # [F]? computed per token below
        gd = g[0] @ sae.W_dec.T                                # [T, F]
        score = act * gd
    best = {}                                                  # (feature, word) -> (score, act)
    for t, w in enumerate(wid):
        if w is None: continue
        nz = act[t].nonzero()[:, 0]
        for f in nz.tolist():
            key = (f, w); s, a = score[t, f].item(), act[t, f].item()
            if key not in best or abs(s) > abs(best[key][0]):
                best[key] = (s, a)
    ranked, used = [], set()
    for (f, w), (s, a) in sorted(best.items(), key=lambda kv: -abs(kv[1][0])):
        if f in used: continue
        used.add(f); ranked.append({"feature": f, "word_index": w, "word": ws[w], "activation": round(a, 3), "score": round(s, 4)})
        if len(ranked) == top: break
    return ranked


def collect(model, tok, sae, text: str, label: int | None = None):
    ws = words(text)
    with torch.no_grad():
        enc = tok(text, return_tensors="pt", truncation=True, max_length=96).to("cuda")
        lg = model(**enc).logits.float()[0]
    pred = int(lg.argmax())
    ig = integrated_gradients(model, tok, ws, pred)
    return {"text": text, "words": ws, "label": label, "prediction": pred,
            "logits": [round(x, 4) for x in lg.tolist()], "margin": round(ig["margin"], 4),
            "ig": [round(x, 4) for x in ig["attr"]], "ig_delta": round(ig["delta"], 4),
            "occ_delete": [round(x, 4) for x in occlusion(model, tok, ws, pred, "delete")],
            "occ_neutral": [round(x, 4) for x in occlusion(model, tok, ws, pred, "neutral")],
            "sae_features": sae_word_features(model, tok, sae, ws, pred)}
