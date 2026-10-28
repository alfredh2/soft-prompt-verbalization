"""Minimal switchable LoRA for the frozen LLM.

Adapters start disabled, so the wrapped model behaves exactly like the frozen base LLM
(behavioral evals) until a decoder explicitly enables them with `lora_enabled`.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

ATTENTION_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 16, alpha: float = 32.0):
        super().__init__()
        self.base = base
        self.scale = alpha / r
        kw = {"dtype": base.weight.dtype, "device": base.weight.device}
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, **kw))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, **kw))
        self.enabled = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.enabled:
            out = out + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale
        return out


def add_lora(
    model: nn.Module, targets: tuple[str, ...] = ATTENTION_TARGETS, r: int = 16, alpha: float = 32.0
) -> dict[str, LoRALinear]:
    replaced: dict[str, LoRALinear] = {}
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if child_name in targets and isinstance(child, nn.Linear):
                lora = LoRALinear(child, r, alpha)
                setattr(module, child_name, lora)
                replaced[f"{name}.{child_name}"] = lora
    return replaced


@contextmanager
def lora_enabled(loras: dict[str, LoRALinear], flag: bool = True):
    previous = {name: m.enabled for name, m in loras.items()}
    for m in loras.values():
        m.enabled = flag
    try:
        yield
    finally:
        for name, m in loras.items():
            m.enabled = previous[name]


def lora_parameters(loras: dict[str, LoRALinear]) -> list[nn.Parameter]:
    return [p for m in loras.values() for p in (m.lora_A, m.lora_B)]


def lora_state_dict(loras: dict[str, LoRALinear]) -> dict[str, torch.Tensor]:
    return {f"{n}.{k}": getattr(m, k).detach().cpu() for n, m in loras.items() for k in ("lora_A", "lora_B")}


def load_lora_state_dict(loras: dict[str, LoRALinear], state: dict[str, torch.Tensor]) -> None:
    for n, m in loras.items():
        for k in ("lora_A", "lora_B"):
            getattr(m, k).data.copy_(state[f"{n}.{k}"])
