"""Learned decoder g_phi(z) -> description.

Rather than hoping the frozen LLM can spontaneously introspect on an arbitrary
soft prompt z (self-report; found to fail, see run_experiment.py), this learns
an explicit mapping from z to a natural-language description, supervised by a
dataset of (z_j, d_j) pairs collected from functions whose true descriptions
are known (see functions.make_parametric_functions). Generalization is tested
on held-out parameter values from the same families.
"""

from __future__ import annotations

import random

import torch
import torch.nn as nn

from .model import DESCRIBE_REQUEST, SYSTEM_DESCRIBE, SoftPromptLM


class ZDecoderAdapter(nn.Module):
    """Small residual MLP applied independently to each of z's n_soft_tokens vectors.

    Trained so that feeding adapter(z) (instead of raw z) into the frozen LLM's
    context slot makes it actually generate the correct description -- i.e. it
    learns to translate the task-optimized z-space into whatever region of
    embedding space the frozen LLM can read as language.
    """

    def __init__(self, hidden_size: int, mult: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * mult),
            nn.GELU(),
            nn.Linear(hidden_size * mult, hidden_size),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z + self.net(z)


def describe_loss(
    model: SoftPromptLM, adapter: ZDecoderAdapter, z: torch.Tensor, desc_text: str
) -> torch.Tensor:
    """Teacher-forced cross-entropy of the true description, given adapter(z) as context."""
    context = model._chat_open(SYSTEM_DESCRIBE)
    z_adapted = adapter(z)
    tail = model._chat_close_and_ask_assistant(DESCRIBE_REQUEST)
    prefix = torch.cat([context, z_adapted, tail], dim=0)

    target_ids = model.tokenizer(desc_text, add_special_tokens=False).input_ids
    target_ids = target_ids + [model.tokenizer.eos_token_id]
    target_ids_t = torch.tensor(target_ids[:-1], device=model.device)  # eos has no embedding input
    target_embeds = model.embed_layer(target_ids_t)

    full_embeds = torch.cat([prefix, target_embeds], dim=0).unsqueeze(0)
    attn_mask = torch.ones(1, full_embeds.shape[1], device=model.device, dtype=torch.long)
    logits = model.model(inputs_embeds=full_embeds, attention_mask=attn_mask).logits[0]
    logprobs = torch.log_softmax(logits, dim=-1)

    prefix_len = prefix.shape[0]
    total = logprobs.new_zeros(())
    for i, tok_id in enumerate(target_ids):
        total = total + logprobs[prefix_len - 1 + i, tok_id]
    return -total / len(target_ids)


def train_decoder(
    model: SoftPromptLM,
    adapter: ZDecoderAdapter,
    dataset: list[tuple[torch.Tensor, str]],
    epochs: int = 300,
    lr: float = 1e-3,
    log_every: int = 50,
) -> None:
    optimizer = torch.optim.Adam(adapter.parameters(), lr=lr)
    for epoch in range(epochs):
        random.shuffle(dataset)
        total_loss = 0.0
        for z, desc in dataset:
            loss = describe_loss(model, adapter, z, desc)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / len(dataset)
        if (epoch + 1) % log_every == 0 or epoch == 0:
            print(f"  decoder epoch {epoch + 1:4d}/{epochs}  avg loss = {avg:.4f}")


def describe_loss_from_z(
    model: SoftPromptLM,
    z_batch: torch.Tensor,
    desc_texts: list[str],
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Batched teacher-forced cross-entropy of descriptions given z directly (no adapter).

    z_batch: [B, n_latent, hidden]. Target includes EOS. Without weights: token-averaged over
    the whole batch. With weights [B]: per-sequence mean loss, then a weighted average.
    """
    dtype = model.embed_layer.weight.dtype
    context = model._chat_open(SYSTEM_DESCRIBE)
    tail = model._chat_close_and_ask_assistant(DESCRIBE_REQUEST)

    seqs, targets, positions = [], [], []
    for b, desc in enumerate(desc_texts):
        target_ids = model.tokenizer(desc, add_special_tokens=False).input_ids
        target_ids = target_ids + [model.tokenizer.eos_token_id]
        target_embeds = model.embed_layer(torch.tensor(target_ids[:-1], device=model.device))
        prefix_len = context.shape[0] + z_batch.shape[1] + tail.shape[0]
        seqs.append(torch.cat([context, z_batch[b].to(dtype), tail, target_embeds], dim=0))
        targets.extend(target_ids)
        positions.extend((b, prefix_len - 1 + i) for i in range(len(target_ids)))

    lens = torch.tensor([s.shape[0] for s in seqs], device=model.device)
    padded = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True)
    mask = (torch.arange(padded.shape[1], device=model.device)[None, :] < lens[:, None]).long()
    hidden = model.model.base_model(
        inputs_embeds=padded, attention_mask=mask, use_cache=False
    ).last_hidden_state
    rows = torch.tensor([p[0] for p in positions], device=model.device)
    cols = torch.tensor([p[1] for p in positions], device=model.device)
    logits = model.model.get_output_embeddings()(hidden[rows, cols]).float()
    token_loss = torch.nn.functional.cross_entropy(
        logits, torch.tensor(targets, device=model.device), reduction="none"
    )
    if weights is None:
        return token_loss.mean()
    n_seq = len(desc_texts)
    per_seq = torch.zeros(n_seq, device=model.device).index_add_(0, rows, token_loss)
    per_seq = per_seq / torch.bincount(rows, minlength=n_seq)
    w = weights.to(model.device, torch.float32)
    return (w * per_seq).sum() / w.sum()


def describe_from_z(model: SoftPromptLM, z: torch.Tensor, max_new_tokens: int = 60) -> str:
    """Generate a description with z in the context slot. No repetition penalty: trained
    templates legitimately repeat tokens ("output 1 ... otherwise output 0")."""
    context = model._chat_open(SYSTEM_DESCRIBE)
    tail = model._chat_close_and_ask_assistant(DESCRIBE_REQUEST)
    prefix = torch.cat([context, z.to(model.embed_layer.weight.dtype), tail], dim=0)
    return model.generate_from_embeds(prefix, max_new_tokens=max_new_tokens, repetition_penalty=1.0)


def generate_description(
    model: SoftPromptLM, adapter: ZDecoderAdapter, z: torch.Tensor, max_new_tokens: int = 30
) -> str:
    context = model._chat_open(SYSTEM_DESCRIBE)
    z_adapted = adapter(z)
    tail = model._chat_close_and_ask_assistant(DESCRIBE_REQUEST)
    prefix = torch.cat([context, z_adapted, tail], dim=0)
    return model.generate_from_embeds(prefix, max_new_tokens=max_new_tokens)
