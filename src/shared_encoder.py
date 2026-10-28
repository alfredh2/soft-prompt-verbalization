"""Shared encoder E(support set of (sentence, label) pairs) -> z, for the frozen carrier.

A free soft prompt is fitted per behaviour and lands at an arbitrary point of a large solution
set. Here ONE network maps any support set to z, so two support draws of the same behaviour
are pushed through the same function. Pair features come from the frozen carrier itself and
are cached (the carrier is never trained); the label is also added as a learned embedding so
it cannot be washed out (the Phase-2 collapse cause was a shared template direction swamping
the label in the pooled hidden state, hence per-dimension standardisation).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.datasets.behaviors import LABELS


@torch.no_grad()
def pair_features(model, texts: list[str], batch: int = 64) -> torch.Tensor:
    """[n_texts, 3, d] last-token hidden state of 'Sentence: ..\\nLabel: <l>' per label (fp16, cpu)."""
    tok = model.tokenizer
    out = torch.zeros(len(texts), len(LABELS), model.embed_layer.weight.shape[1], dtype=torch.float16)
    tok.padding_side = "left"
    for li, lab in enumerate(LABELS):
        for i in range(0, len(texts), batch):
            enc = tok([f"Sentence: {t}\nLabel: {lab}" for t in texts[i:i + batch]], return_tensors="pt",
                      padding=True).to(model.device)
            h = model.model.base_model(**enc).last_hidden_state[:, -1]
            out[i:i + batch, li] = h.float().cpu().half()
    return out


class Encoder(nn.Module):
    def __init__(self, d_in: int, d_out: int, n_tokens: int = 16, d: int = 512, layers: int = 2,
                 heads: int = 8, z_norm: float = 1.0, use_features: bool = True):
        super().__init__()
        self.n_tokens, self.d_out, self.z_norm, self.use_features = n_tokens, d_out, z_norm, use_features
        self.register_buffer("mu", torch.zeros(d_in))
        self.register_buffer("sd", torch.ones(d_in))
        self.proj = nn.Sequential(nn.Linear(d_in, d), nn.GELU(), nn.Linear(d, d))
        self.label = nn.Embedding(len(LABELS), d)
        self.queries = nn.Parameter(torch.randn(n_tokens, d) * 0.02)
        self.attn = nn.ModuleList(nn.MultiheadAttention(d, heads, batch_first=True) for _ in range(layers))
        self.ffn = nn.ModuleList(nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
                                 for _ in range(layers))
        self.norm_q = nn.ModuleList(nn.LayerNorm(d) for _ in range(layers))
        self.norm_kv = nn.LayerNorm(d)
        self.out = nn.Linear(d, d_out)

    def set_stats(self, feats: torch.Tensor):
        f = feats.float().reshape(-1, feats.shape[-1])
        self.mu.copy_(f.mean(0)); self.sd.copy_(f.std(0).clamp_min(1e-3))

    def forward(self, feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """feats [B, N, d_in], labels [B, N] (label index) -> z [B, n_tokens, d_out]."""
        x = ((feats.float() - self.mu) / self.sd) if self.use_features else torch.zeros_like(feats.float())
        kv = self.norm_kv(self.proj(x) + self.label(labels))
        q = self.queries[None].expand(len(feats), -1, -1)
        for attn, ffn, nq in zip(self.attn, self.ffn, self.norm_q):
            q = q + attn(nq(q), kv, kv, need_weights=False)[0]
            q = q + ffn(q)
        z = self.out(q)
        return F.normalize(z, dim=-1) * self.z_norm


def info_nce(z: torch.Tensor, ids: torch.Tensor, mask_same: torch.Tensor, temp: float = 0.1) -> torch.Tensor:
    """z [M, T, d] (M = 2 views per behaviour); positives share `ids`.

    `mask_same[i, j]` marks pairs of DIFFERENT behaviours whose outputs are near-identical:
    they are removed from the negatives (false negatives), not treated as positives.
    """
    f = F.normalize(z.flatten(1).float(), dim=-1)
    sim = f @ f.T / temp
    eye = torch.eye(len(f), dtype=torch.bool, device=f.device)
    pos = (ids[:, None] == ids[None, :]) & ~eye
    neg_ok = ~mask_same & ~pos & ~eye
    logits = sim.masked_fill(~(pos | neg_ok), -1e9)
    logp = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    return -(logp * pos).sum(1).div(pos.sum(1).clamp_min(1)).mean()
