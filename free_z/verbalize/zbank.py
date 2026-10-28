"""Bank of independently optimized per-task soft prompts z* (Phase 1 style, no encoder).

For every (function, seed) we optimize a fresh z from a seed-specific random-vocab init so
that the frozen LLM imitates the function. Multiple seeds per function give the decoder
different z solutions for the same behavior, so it can learn invariance to them.

Many z's are optimized at once: each is an independent row of one parameter tensor and
their query sequences share one batched LLM forward, so rows never interact.
"""

from __future__ import annotations

import random
import zlib
from collections import defaultdict
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .evaluate import accuracy
from .functions import FunctionSpec
from .model import SoftPromptLM


@dataclass
class ZBankConfig:
    n_soft_tokens: int = 16
    steps: int = 150
    queries_per_step: int = 6
    lr: float = 0.3
    n_eval: int = 16
    log_every: int = 25
    eval_at: tuple[int, ...] = ()
    u_init_std: float = 0.5  # step 5: std of the random latent init when z is constrained to a subspace


@dataclass
class ZJob:
    spec: FunctionSpec
    split: str
    seed_idx: int

    @property
    def key(self) -> tuple[str, int]:
        return (self.spec.name, self.seed_idx)

    @property
    def init_seed(self) -> int:
        return zlib.crc32(self.spec.name.encode()) + 7919 * self.seed_idx


def split_examples(spec: FunctionSpec, n_eval: int) -> tuple[list[tuple], list[tuple]]:
    """Fixed per function (shared by all seeds): a held-out eval set allocated across labels in
    proportion to label frequency (at least 2 per label, never more than half of a label), and
    the rest of the domain for training. A label-balanced eval set starved rare labels of
    training data (gt_-20 kept 3 label-0 inputs), so z memorized them instead of the rule."""
    rng = random.Random(zlib.crc32(spec.name.encode()))
    by_label: dict[str, list[tuple]] = defaultdict(list)
    for inp in spec.domain:
        by_label[spec.compute(*inp)].append(inp)
    eval_inputs: list[tuple] = []
    for label in sorted(by_label):
        pool = list(by_label[label])
        rng.shuffle(pool)
        k = min(max(2, round(n_eval * len(pool) / len(spec.domain))), len(pool) // 2)
        eval_inputs += pool[:k]
    held = set(eval_inputs)
    eval_ex = [(inp, spec.compute(*inp)) for inp in eval_inputs]
    train_ex = [(inp, spec.compute(*inp)) for inp in spec.domain if inp not in held]
    return train_ex, eval_ex


def balanced_sample(examples: list[tuple], n: int, rng: random.Random) -> list[tuple]:
    by_label: dict[str, list[tuple]] = defaultdict(list)
    for ex in examples:
        by_label[ex[1]].append(ex)
    labels = sorted(by_label)
    return [rng.choice(by_label[labels[i % len(labels)]]) for i in range(n)]


def init_z(model: SoftPromptLM, seed: int, n_tokens: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, model.embed_layer.weight.shape[0], (n_tokens,), generator=gen)
    with torch.no_grad():
        return model.embed_layer(ids.to(model.device)).float().clone()


def optimize_batch(
    model: SoftPromptLM, jobs: list[ZJob], cfg: ZBankConfig, log=print, subspace=None, k_latent: int | None = None
) -> list[dict]:
    """With `subspace` (a LangSubspace), each z is generated from a k_latent-dim latent u and only
    u is optimized; seeds differ in their random u init. Without it, z itself is optimized."""
    if subspace is None:
        param = torch.nn.Parameter(torch.stack([init_z(model, j.init_seed, cfg.n_soft_tokens) for j in jobs]))
        to_z = lambda p: p  # noqa: E731
    else:
        inits = [torch.randn(k_latent, generator=torch.Generator().manual_seed(j.init_seed)) * cfg.u_init_std
                 for j in jobs]
        param = torch.nn.Parameter(torch.stack(inits).to(model.device))
        to_z = subspace.to_z
    optimizer = torch.optim.Adam([param], lr=cfg.lr)
    data = [split_examples(j.spec, cfg.n_eval) for j in jobs]
    rngs = [random.Random(j.init_seed) for j in jobs]

    for step in range(cfg.steps):
        groups: dict[tuple, list] = defaultdict(list)
        for k, (job, (train_ex, _)) in enumerate(zip(jobs, data)):
            for inputs, label in balanced_sample(train_ex, cfg.queries_per_step, rngs[k]):
                groups[tuple(job.spec.candidates)].append((k, job.spec, inputs, label))

        Z = to_z(param)
        loss = Z.new_zeros(())
        for cands, items in groups.items():
            cand_ids = model.candidate_token_ids(list(cands))
            prefixes = [
                model.build_prefix_embeds("soft", spec.format_input(*inputs), z=Z[k])
                for k, spec, inputs, _ in items
            ]
            logprobs = model.candidate_logprobs(prefixes, cand_ids)
            targets = torch.tensor([cands.index(label) for *_, label in items], device=model.device)
            loss = loss + F.cross_entropy(logprobs, targets, reduction="sum")
        loss = loss / (len(jobs) * cfg.queries_per_step)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (step + 1) % cfg.log_every == 0 or step == 0:
            log({"step": step + 1, "batch_loss": round(loss.item(), 4)})
        if step + 1 in cfg.eval_at and step + 1 != cfg.steps:
            Zd = to_z(param).detach()
            log({"step": step + 1, "checkpoint": [
                {"task": job.spec.name, "seed": job.seed_idx,
                 "train_acc": round(accuracy(model, job.spec, train_ex, "soft", z=Zd[k]), 3),
                 "eval_acc": round(accuracy(model, job.spec, eval_ex, "soft", z=Zd[k]), 3)}
                for k, (job, (train_ex, eval_ex)) in enumerate(zip(jobs, data))
            ]})

    Z = to_z(param).detach()
    records = []
    for k, (job, (train_ex, eval_ex)) in enumerate(zip(jobs, data)):
        z = Z[k]
        records.append({
            "task": job.spec.name,
            "family": job.spec.family,
            "params": job.spec.params,
            "split": job.split,
            "seed_idx": job.seed_idx,
            "z": z.cpu(),
            "u": None if subspace is None else param[k].detach().cpu(),
            "z_norm": z.norm(dim=-1).mean().item(),
            "train_acc": accuracy(model, job.spec, train_ex, "soft", z=z),
            "eval_acc": accuracy(model, job.spec, eval_ex, "soft", z=z),
            "steps": cfg.steps,
        })
    return records
