"""Shared behavioral encoder E_psi: a set of (x, f(x)) examples -> z.

Each example is serialized as text ("x = 3 -> f = 1") and run through the frozen
LLM once; its token-level last hidden states are cached per unique string (the
domains are small, so the whole cache is a few hundred strings). A swappable
pooling step turns each example into one vector, a positional-encoding-free set
transformer mixes the examples (so z is invariant to example order), and a
Perceiver-style cross-attention from n_latent learned queries produces z.

Outputs are rescaled to the typical vocab-embedding norm. That is a stabilization
trick only; it does not make z language-like.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import SoftPromptLM
from .tasks import pair_text

POOLINGS = ("mean", "last_token", "attention")


class PairFeatureCache:
    def __init__(self, model: SoftPromptLM, batch_size: int = 64):
        self.model = model
        self.batch_size = batch_size
        self.states: dict[str, torch.Tensor] = {}
        self.text_format: dict[str, str] = {}

    @torch.no_grad()
    def warm(self, texts: list[str]) -> None:
        missing = [t for t in dict.fromkeys(texts) if t not in self.states]
        tok = self.model.tokenizer
        tok.padding_side = "right"
        for start in range(0, len(missing), self.batch_size):
            chunk = missing[start : start + self.batch_size]
            enc = tok(chunk, add_special_tokens=False, padding=True, return_tensors="pt").to(self.model.device)
            hidden = self.model.model.base_model(
                input_ids=enc.input_ids, attention_mask=enc.attention_mask, use_cache=False
            ).last_hidden_state.float()
            for j, text in enumerate(chunk):
                n = int(enc.attention_mask[j].sum())
                self.states[text] = hidden[j, :n].clone()

    def batch(self, texts: list[list[str]]) -> tuple[torch.Tensor, torch.Tensor]:
        """texts: [B][n] pair strings -> states [B, n, T, H], mask [B, n, T]."""
        self.warm([t for row in texts for t in row])
        B, n = len(texts), len(texts[0])
        T = max(self.states[t].shape[0] for row in texts for t in row)
        H = next(iter(self.states.values())).shape[1]
        device = self.model.device
        states = torch.zeros(B, n, T, H, device=device)
        mask = torch.zeros(B, n, T, dtype=torch.bool, device=device)
        for b, row in enumerate(texts):
            for i, text in enumerate(row):
                s = self.states[text]
                states[b, i, : s.shape[0]] = s
                mask[b, i, : s.shape[0]] = True
        return states, mask

    def register_formats(self, specs: list) -> None:
        """Map every pair string to its input format: (domain size, label vocabulary)."""
        for spec in specs:
            key = f"domain{len(spec.domain)}|labels={','.join(spec.candidates)}"
            for inp in spec.domain:
                for cand in spec.candidates:
                    text = pair_text(spec, inp, cand)
                    prev = self.text_format.setdefault(text, key)
                    assert prev == key, f"{text!r} belongs to two formats: {prev} / {key}"

    def format_index(self, texts: list[list[str]], keys: list[str]) -> torch.Tensor:
        lookup = {k: i for i, k in enumerate(keys)}
        return torch.tensor(
            [[lookup[self.text_format[t]] for t in row] for row in texts], device=self.model.device
        )

    def format_feature_stats(self, pooling: str) -> tuple[list[str], torch.Tensor, torch.Tensor]:
        """Per-format, per-dim mean/std of pooled features. Pooling stats across formats
        lets the format itself dominate (e.g. 0/1 vs A/B/C labels), which re-creates the
        constant-z collapse. Attention pooling is learned, so it uses mean-pooled stats."""
        keys = sorted(set(self.text_format.values()))
        mus, sds = [], []
        for key in keys:
            texts = [t for t, k in self.text_format.items() if k == key]
            self.warm(texts)
            vecs = [self.states[t][-1] if pooling == "last_token" else self.states[t].mean(dim=0) for t in texts]
            feats = torch.stack(vecs)
            mus.append(feats.mean(dim=0))
            sds.append(feats.std(dim=0).clamp(min=1e-4))
        return keys, torch.stack(mus), torch.stack(sds)


class BehaviorEncoder(nn.Module):
    def __init__(
        self,
        llm_hidden: int,
        target_norm: float,
        n_latent: int = 16,
        d: int = 512,
        n_layers: int = 3,
        n_heads: int = 8,
        pooling: str = "mean",
        standardize: bool = True,
        n_formats: int = 1,
    ):
        super().__init__()
        assert pooling in POOLINGS, pooling
        self.pooling = pooling
        # Raw pair features are ~96% one shared template direction, so every support set
        # looked identical and the encoder collapsed to a constant z. Standardizing exposes
        # the per-example (label, x) differences -- but only with per-input-format stats:
        # pooled across formats, the format itself dominates and the collapse returns.
        self.standardize = standardize
        self.format_keys: list[str] | None = None
        if standardize:
            self.register_buffer("feat_mu", torch.zeros(n_formats, llm_hidden))
            self.register_buffer("feat_sd", torch.ones(n_formats, llm_hidden))
        if pooling == "attention":
            self.pool_query = nn.Parameter(torch.randn(llm_hidden) * 0.02)

        self.in_norm = nn.LayerNorm(llm_hidden)
        self.in_proj = nn.Linear(llm_hidden, d)

        def block():
            return nn.TransformerEncoderLayer(
                d, n_heads, dim_feedforward=4 * d, dropout=0.0,
                activation="gelu", batch_first=True, norm_first=True,
            )

        self.set_encoder = nn.TransformerEncoder(block(), n_layers, enable_nested_tensor=False)
        self.latents = nn.Parameter(torch.randn(n_latent, d) * 0.02)
        self.q_norm = nn.LayerNorm(d)
        self.kv_norm = nn.LayerNorm(d)
        self.cross_attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.cross_ffn = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.latent_encoder = nn.TransformerEncoder(block(), 1, enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(d)
        self.out_proj = nn.Linear(d, llm_hidden)
        self.register_buffer("target_norm", torch.tensor(float(target_norm)))

    def pool(self, states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.unsqueeze(-1).float()
        if self.pooling == "mean":
            return (states * m).sum(dim=2) / m.sum(dim=2).clamp(min=1.0)
        if self.pooling == "last_token":
            last = (mask.sum(dim=2) - 1).clamp(min=0)
            idx = last[..., None, None].expand(-1, -1, 1, states.shape[-1])
            return states.gather(2, idx).squeeze(2)
        scores = (states @ self.pool_query) / states.shape[-1] ** 0.5
        weights = scores.masked_fill(~mask, float("-inf")).softmax(dim=2)
        return (states * weights.unsqueeze(-1)).sum(dim=2)

    def set_feature_stats(self, keys: list[str], mu: torch.Tensor, sd: torch.Tensor) -> None:
        assert mu.shape == self.feat_mu.shape, (mu.shape, self.feat_mu.shape)
        self.format_keys = list(keys)
        self.feat_mu.copy_(mu)
        self.feat_sd.copy_(sd)

    def forward(
        self, states: torch.Tensor, mask: torch.Tensor, fmt_idx: torch.Tensor | None = None
    ) -> torch.Tensor:
        pooled = self.pool(states, mask)
        if self.standardize:
            if fmt_idx is None:
                fmt_idx = torch.zeros(pooled.shape[:2], dtype=torch.long, device=pooled.device)
            pooled = (pooled - self.feat_mu[fmt_idx]) / self.feat_sd[fmt_idx]
        h = self.in_proj(self.in_norm(pooled))  # [B, n, d]
        h = self.set_encoder(h)
        kv = self.kv_norm(h)
        q = self.latents.unsqueeze(0).expand(h.shape[0], -1, -1)
        q = q + self.cross_attn(self.q_norm(q), kv, kv, need_weights=False)[0]
        q = q + self.cross_ffn(q)
        q = self.latent_encoder(q)
        out = self.out_proj(self.out_norm(q))  # [B, n_latent, llm_hidden]
        return F.normalize(out, dim=-1) * self.target_norm


def encode_supports(
    encoder: BehaviorEncoder, cache: PairFeatureCache, specs: list, supports: list[list[tuple]]
) -> torch.Tensor:
    texts = [[pair_text(spec, inp, lab) for inp, lab in sup] for spec, sup in zip(specs, supports)]
    states, mask = cache.batch(texts)
    fmt_idx = cache.format_index(texts, encoder.format_keys) if encoder.format_keys else None
    return encoder(states, mask, fmt_idx)
