"""TopK sparse autoencoder on one BERT layer's residual stream, plus hooks to read and patch that layer.

LAYER is the number of encoder layers applied (hidden_states[LAYER]); patching hooks the output of
encoder.layer[LAYER-1]. Interventions subtract a feature's decoder contribution from the activation and keep
the SAE reconstruction error, so only the chosen feature changes.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

LAYER = 8


class TopKSAE(nn.Module):
    def __init__(self, d_in=768, d_sae=12288, k=32):
        super().__init__()
        self.k = k
        self.W_enc = nn.Parameter(torch.randn(d_in, d_sae) / d_in ** 0.5)
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(F.normalize(torch.randn(d_sae, d_in), dim=1))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.register_buffer("scale", torch.ones(()))   # activations are divided by this before the SAE

    def encode(self, x):
        a = F.relu((x / self.scale - self.b_dec) @ self.W_enc + self.b_enc)
        v, i = a.topk(self.k, dim=-1)
        return torch.zeros_like(a).scatter_(-1, i, v)

    def decode(self, a):
        return (a @ self.W_dec + self.b_dec) * self.scale

    def forward(self, x):
        a = self.encode(x)
        return self.decode(a), a

    @torch.no_grad()
    def renorm(self):
        self.W_dec.data = F.normalize(self.W_dec.data, dim=1)


@torch.no_grad()
def hidden_states(model, enc, layer=LAYER):
    return model(**enc, output_hidden_states=True).hidden_states[layer]


def _layer_module(model, layer=LAYER):
    return model.base_model.encoder.layer[layer - 1]


def forward_with_patch(model, enc, patch_fn, layer=LAYER):
    """Run the classifier with `patch_fn(h) -> h'` applied to the layer's output. Returns logits."""
    def hook(_m, _i, out):
        if isinstance(out, tuple):
            return (patch_fn(out[0]),) + tuple(out[1:])
        return patch_fn(out)
    h = _layer_module(model, layer).register_forward_hook(hook)
    try:
        return model(**enc).logits
    finally:
        h.remove()
