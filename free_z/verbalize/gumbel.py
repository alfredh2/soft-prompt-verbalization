"""Alternate approach to constraining z toward language: instead of independently
snapping each z_i to its nearest vocab embedding (nearest-neighbor projection,
see model.py's project_to_nearest -- constrains each token to be *real* but not
the sequence to be *fluent*), parameterize z as a Gumbel-softmax relaxation over
full vocab distributions at each position, and add a sequence-level fluency loss
that asks the frozen LM to find each z_{t+1} plausible given z_{<=t}:

  L_fluency = -(1/(L-1)) * sum_t sum_v y_soft[t+1, v] * log P_LM(v | z_0..z_t)

Total loss = L_task + fluency_weight * L_fluency. Sweeping fluency_weight traces
out the tradeoff between task accuracy and how much the resulting sequence looks
like something the LM itself finds self-consistent.
"""

from __future__ import annotations

import random

import torch
import torch.nn as nn

from .model import QUESTION, SYSTEM_ANSWER, SoftPromptLM


class GumbelSoftPrompt(nn.Module):
    def __init__(self, n_soft_tokens: int, vocab_size: int, device: str):
        super().__init__()
        self.logits = nn.Parameter(torch.randn(n_soft_tokens, vocab_size, device=device) * 0.01)
        self.n_soft_tokens = n_soft_tokens

    def sample(self, embed_weight: torch.Tensor, tau: float) -> tuple[torch.Tensor, torch.Tensor]:
        g = -torch.log(-torch.log(torch.rand_like(self.logits) + 1e-9) + 1e-9)
        y_soft = torch.softmax((self.logits + g) / tau, dim=-1)
        z = y_soft @ embed_weight  # [n_soft_tokens, hidden]
        return z, y_soft

    def argmax_ids(self) -> torch.Tensor:
        return self.logits.argmax(dim=-1)


def gumbel_step(
    model: SoftPromptLM,
    gsp: GumbelSoftPrompt,
    spec,
    inputs: tuple,
    label: str,
    tau: float,
    fluency_weight: float,
) -> tuple[torch.Tensor, float, float]:
    z, y_soft = gsp.sample(model.embed_layer.weight, tau)

    # task loss: reuse score_candidates on a prefix built from this z (autograd intact)
    context = model.build_context_embeds("soft", system_text=SYSTEM_ANSWER, z=z)
    input_text = spec.format_input(*inputs)
    tail = model._chat_close_and_ask_assistant(f"{input_text}{QUESTION}")
    prefix = torch.cat([context, tail], dim=0)
    scores = model.score_candidates(prefix, spec.candidates, grad_for=label)
    task_loss = -scores[label]

    # fluency loss: does the frozen LM find z_{t+1} plausible given z_0..z_t?
    context_before_z = model._chat_open(SYSTEM_ANSWER)
    seq = torch.cat([context_before_z, z], dim=0).unsqueeze(0)
    attn = torch.ones(1, seq.shape[1], device=model.device, dtype=torch.long)
    logits_z = model.model(inputs_embeds=seq, attention_mask=attn).logits[0]

    prefix_len = context_before_z.shape[0]
    L = gsp.n_soft_tokens
    fluency_loss = logits_z.new_zeros(())
    for t in range(L - 1):
        logp = torch.log_softmax(logits_z[prefix_len - 1 + t], dim=-1)
        fluency_loss = fluency_loss - (y_soft[t + 1] * logp).sum()
    fluency_loss = fluency_loss / (L - 1)

    total = task_loss + fluency_weight * fluency_loss
    return total, task_loss.item(), fluency_loss.item()


def train_gumbel_soft_prompt(
    model: SoftPromptLM,
    spec,
    train_examples: list[tuple],
    fluency_weight: float,
    epochs: int = 60,
    lr: float = 0.05,
    tau_start: float = 2.0,
    tau_end: float = 0.3,
    log_every: int = 10,
) -> GumbelSoftPrompt:
    vocab_size = model.embed_layer.weight.shape[0]
    gsp = GumbelSoftPrompt(model.n_soft_tokens, vocab_size, model.device)
    optimizer = torch.optim.Adam([gsp.logits], lr=lr)

    for epoch in range(epochs):
        tau = tau_start + (tau_end - tau_start) * (epoch / max(1, epochs - 1))
        random.shuffle(train_examples)
        total_task, total_fluency = 0.0, 0.0

        for inputs, label in train_examples:
            loss, task_l, fluency_l = gumbel_step(
                model, gsp, spec, inputs, label, tau, fluency_weight
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_task += task_l
            total_fluency += fluency_l

        if (epoch + 1) % log_every == 0 or epoch == 0:
            n = len(train_examples)
            print(
                f"  epoch {epoch + 1:3d}/{epochs}  tau={tau:.2f}  "
                f"task_loss={total_task / n:.4f}  fluency_loss={total_fluency / n:.4f}"
            )

    return gsp


def eval_gumbel_accuracy(model: SoftPromptLM, gsp: GumbelSoftPrompt, spec, test_examples: list[tuple]) -> float:
    """Use the argmax (hard, tau->0) decoding of z for behavioral eval."""
    with torch.no_grad():
        ids = gsp.argmax_ids()
        z_hard = model.embed_layer(ids)
    correct = 0
    for inputs, label in test_examples:
        input_text = spec.format_input(*inputs)
        context = model.build_context_embeds("soft", system_text=SYSTEM_ANSWER, z=z_hard)
        tail = model._chat_close_and_ask_assistant(f"{input_text}{QUESTION}")
        prefix = torch.cat([context, tail], dim=0)
        with torch.no_grad():
            scores = model.score_candidates(prefix, spec.candidates)
        pred = max(scores, key=scores.get)
        correct += pred == label
    return correct / len(test_examples)


def decode_gumbel_prompt(model: SoftPromptLM, gsp: GumbelSoftPrompt) -> str:
    ids = gsp.argmax_ids()
    return model.tokenizer.decode(ids, skip_special_tokens=True)
