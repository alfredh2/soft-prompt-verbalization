"""Optimize the soft prompt z so that LLM(x, z) imitates f*(x)."""

from __future__ import annotations

import random

import torch

from .functions import FunctionSpec
from .model import SoftPromptLM


def train_soft_prompt(
    model: SoftPromptLM,
    spec: FunctionSpec,
    train_examples: list[tuple],
    epochs: int = 60,
    lr: float = 0.3,
    log_every: int = 10,
    fluent: bool = False,
) -> list[float]:
    """If fluent=True, every forward pass during training snaps z to its nearest
    real vocab embeddings (straight-through), keeping it on the language manifold
    instead of letting it drift to whatever satisfies the loss."""
    model.use_projected_soft_prompt = fluent
    optimizer = torch.optim.Adam([model.soft_prompt], lr=lr)
    losses = []

    for epoch in range(epochs):
        random.shuffle(train_examples)
        epoch_loss = 0.0

        for inputs, label in train_examples:
            input_text = spec.format_input(*inputs)
            prefix = model.build_prefix_embeds("soft", input_text)

            scores = model.score_candidates(prefix, spec.candidates, grad_for=label)
            loss = -scores[label]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_examples)
        losses.append(avg_loss)
        if (epoch + 1) % log_every == 0 or epoch == 0:
            print(f"  epoch {epoch + 1:3d}/{epochs}  avg loss = {avg_loss:.4f}")

    return losses


def train_soft_prompt_two_phase(
    model: SoftPromptLM,
    spec: FunctionSpec,
    train_examples: list[tuple],
    phase1_epochs: int = 60,
    phase1_lr: float = 0.3,
    phase2_epochs: int = 40,
    phase2_lr: float = 0.05,
    log_every: int = 10,
) -> None:
    """Phase 1: optimize z freely (no fluency constraint) to a good task fit.
    Phase 2: continue optimizing from that point with the real-token straight-through
    projection turned on, so the search for a fluent/discrete approximation starts
    from an already-good solution instead of from scratch."""
    print(f"  phase 1/2: free optimization ({phase1_epochs} epochs, lr={phase1_lr})")
    train_soft_prompt(
        model, spec, train_examples, epochs=phase1_epochs, lr=phase1_lr,
        log_every=log_every, fluent=False,
    )
    print(f"  phase 2/2: real-token projection fine-tune ({phase2_epochs} epochs, lr={phase2_lr})")
    train_soft_prompt(
        model, spec, train_examples, epochs=phase2_epochs, lr=phase2_lr,
        log_every=log_every, fluent=True,
    )
