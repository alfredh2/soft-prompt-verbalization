"""Joint training of the shared encoder: L = L_behavior + lambda_text * L_text.

L_behavior: z = E(support) in the frozen LLM's soft slot must predict held-out
query labels (support and query are disjoint, so the encoder can't copy answers).
L_text: the frozen LLM must generate a (randomly chosen) paraphrase of the task's
description from z. Only encoder parameters are trained.
"""

from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from .decoder import describe_loss_from_z
from .encoder import BehaviorEncoder, PairFeatureCache, encode_supports
from .functions import FunctionSpec
from .model import SoftPromptLM
from .tasks import draw_support_query, sample_train_task


@dataclass
class JointConfig:
    lambda_text: float = 0.1
    steps: int = 2000
    tasks_per_step: int = 4
    n_support: int = 16
    n_query: int = 12
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    grad_clip: float = 1.0
    log_every: int = 10
    ckpt_every: int = 200
    seed: int = 0
    balanced_query: bool = False
    contrast_weight: float = 0.0
    contrast_tasks: int = 32
    contrast_temp: float = 0.1


def behavior_loss(
    model: SoftPromptLM, z: torch.Tensor, specs: list[FunctionSpec], queries: list[list[tuple]]
) -> tuple[torch.Tensor, float]:
    groups: dict[tuple, list] = defaultdict(list)
    for b, (spec, query) in enumerate(zip(specs, queries)):
        for inputs, label in query:
            groups[tuple(spec.candidates)].append((b, spec, inputs, label))

    loss_sum, n, correct = z.new_zeros(()), 0, 0
    for cands, items in groups.items():
        cand_ids = model.candidate_token_ids(list(cands))
        prefixes = [
            model.build_prefix_embeds("soft", spec.format_input(*inputs), z=z[b])
            for b, spec, inputs, _ in items
        ]
        logprobs = model.candidate_logprobs(prefixes, cand_ids)
        targets = torch.tensor([cands.index(label) for *_, label in items], device=model.device)
        loss_sum = loss_sum + F.cross_entropy(logprobs, targets, reduction="sum")
        correct += int((logprobs.argmax(dim=-1) == targets).sum())
        n += len(items)
    return loss_sum / n, correct / n


def text_loss(
    model: SoftPromptLM, z: torch.Tensor, specs: list[FunctionSpec], rng: random.Random
) -> torch.Tensor | None:
    idx = [b for b, spec in enumerate(specs) if spec.paraphrases]
    if not idx:
        return None
    descs = [rng.choice(specs[b].paraphrases) for b in idx]
    return describe_loss_from_z(model, z[idx], descs)


def contrastive_loss(
    encoder: BehaviorEncoder,
    cache: PairFeatureCache,
    train_tasks: list[FunctionSpec],
    cfg: JointConfig,
    rng: random.Random,
) -> tuple[torch.Tensor, float]:
    """Symmetric InfoNCE across tasks: two independent support draws of the same task should
    map to nearby z, different tasks apart. Targets the collapse where every task gets one z.
    Encoder-only (cached features), so it affords far more tasks per step than L_behavior."""
    specs = rng.sample(train_tasks, min(cfg.contrast_tasks, len(train_tasks)))
    sup_a = [draw_support_query(s, cfg.n_support, 0, rng)[0] for s in specs]
    sup_b = [draw_support_query(s, cfg.n_support, 0, rng)[0] for s in specs]
    za = F.normalize(encode_supports(encoder, cache, specs, sup_a).flatten(1), dim=-1)
    zb = F.normalize(encode_supports(encoder, cache, specs, sup_b).flatten(1), dim=-1)
    logits = za @ zb.T / cfg.contrast_temp
    targets = torch.arange(len(specs), device=logits.device)
    loss = (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2
    acc = (logits.argmax(dim=1) == targets).float().mean().item()
    return loss, acc


def save_checkpoint(path: Path, encoder, optimizer, step: int, cfg: JointConfig, rng: random.Random) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": asdict(cfg),
            "rng_state": rng.getstate(),
        },
        tmp,
    )
    tmp.replace(path)


def train(
    model: SoftPromptLM,
    encoder: BehaviorEncoder,
    cache: PairFeatureCache,
    train_tasks: list[FunctionSpec],
    cfg: JointConfig,
    run_dir: Path,
    resume: bool = False,
) -> None:
    by_family: dict[str, list[FunctionSpec]] = defaultdict(list)
    for spec in train_tasks:
        by_family[spec.family].append(spec)

    optimizer = torch.optim.AdamW(encoder.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    rng = random.Random(cfg.seed)
    start_step = 0
    ckpt_path = run_dir / "ckpt.pt"
    if resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=model.device, weights_only=False)
        encoder.load_state_dict(ckpt["encoder"])
        optimizer.load_state_dict(ckpt["optimizer"])
        rng.setstate(ckpt["rng_state"])
        start_step = ckpt["step"]
        print(f"resumed from step {start_step}", flush=True)

    encoder.train()
    log_file = open(run_dir / "train_log.jsonl", "a")
    t0 = time.time()
    for step in range(start_step, cfg.steps):
        for group in optimizer.param_groups:
            group["lr"] = cfg.lr * min(1.0, (step + 1) / cfg.warmup_steps)

        specs = [sample_train_task(by_family, rng) for _ in range(cfg.tasks_per_step)]
        draws = [
            draw_support_query(s, cfg.n_support, cfg.n_query, rng, balanced_query=cfg.balanced_query)
            for s in specs
        ]
        z = encode_supports(encoder, cache, specs, [d[0] for d in draws])

        l_behavior, acc = behavior_loss(model, z, specs, [d[1] for d in draws])
        if cfg.lambda_text > 0:
            l_text = text_loss(model, z, specs, rng)
            total = l_behavior + cfg.lambda_text * l_text if l_text is not None else l_behavior
        else:
            with torch.no_grad():
                l_text = text_loss(model, z, specs, rng)
            total = l_behavior

        l_contrast, contrast_acc = None, None
        if cfg.contrast_weight > 0:
            l_contrast, contrast_acc = contrastive_loss(encoder, cache, train_tasks, cfg, rng)
            total = total + cfg.contrast_weight * l_contrast

        optimizer.zero_grad()
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), cfg.grad_clip)
        optimizer.step()

        if (step + 1) % cfg.log_every == 0 or step == start_step:
            rec = {
                "step": step + 1,
                "loss_behavior": round(l_behavior.item(), 4),
                "loss_text": round(l_text.item(), 4) if l_text is not None else None,
                "loss_total": round(total.item(), 4),
                "loss_contrast": round(l_contrast.item(), 4) if l_contrast is not None else None,
                "contrast_acc": round(contrast_acc, 3) if contrast_acc is not None else None,
                "query_acc": round(acc, 3),
                "grad_norm": round(float(grad_norm), 3),
                "sec": round(time.time() - t0, 1),
            }
            print(json.dumps(rec), flush=True)
            log_file.write(json.dumps(rec) + "\n")
            log_file.flush()

        if (step + 1) % cfg.ckpt_every == 0 or step + 1 == cfg.steps:
            save_checkpoint(ckpt_path, encoder, optimizer, step + 1, cfg, rng)

    log_file.close()
    encoder.eval()
