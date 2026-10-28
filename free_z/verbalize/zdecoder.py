"""Decoder G: per-task soft prompt z* -> natural-language description.

G = LoRA-adapted LLM reading ZReader(z) in the soft-prompt slot of the describe scaffold,
trained with teacher forcing on (z, paraphrased description) pairs from the z bank.
Unlike naive self-report, G is explicitly trained to read soft prompts; unlike the Phase 1
adapter it sees many functions and several independent z solutions per function.
"""

from __future__ import annotations

import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder import describe_loss_from_z
from .lora import LoRALinear, lora_enabled, lora_parameters, lora_state_dict
from .model import SoftPromptLM
from .zbank import split_examples


class ZReader(nn.Module):
    """Maps a freely optimized z (arbitrary location and scale per seed) into the decoder's
    input space: LayerNorm -> MLP -> rescale to typical token-embedding norm."""

    def __init__(self, hidden: int, target_norm: float, mult: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.net = nn.Sequential(nn.Linear(hidden, mult * hidden), nn.GELU(), nn.Linear(mult * hidden, hidden))
        self.register_buffer("target_norm", torch.tensor(float(target_norm)))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(self.norm(z)), dim=-1) * self.target_norm


def annotate_quality(bank: list[dict], specs: dict, n_eval: int = 16) -> None:
    """const = accuracy of always predicting the majority label on the z's exact eval set;
    q = eval_acc - const. Raw eval_acc is unfair across tasks with very different baselines
    (at_least: 0.95 eval vs 0.80 const; piecewise: 0.88 eval vs 0.50 const)."""
    for r in bank:
        _, eval_ex = split_examples(specs[r["task"]], n_eval)
        r["const"] = max(Counter(label for _, label in eval_ex).values()) / len(eval_ex)
        r["q"] = r["eval_acc"] - r["const"]


def make_keep_rule(
    min_acc: float, min_margin: float, min_abs: float, exclude_families: tuple[str, ...] = ()
) -> Callable[[dict], bool]:
    """Keep a z if its eval accuracy is high, or if it clearly beats the constant baseline
    while still being reasonably accurate in absolute terms."""
    def keep(r: dict) -> bool:
        if r["family"] in exclude_families:
            return False
        return r["eval_acc"] >= min_acc or (r["q"] >= min_margin and r["eval_acc"] >= min_abs)
    return keep


def partition_bank(
    bank: list[dict], keep: Callable[[dict], bool], heldout_seed: int | set[int], n_val_tasks: int, seed: int = 0
) -> dict[str, list[dict]]:
    """Only z's passing `keep` (see make_keep_rule; bank must be annotate_quality'd).
    train: train-split functions (minus val functions), seeds not in heldout_seed
    val: n_val_tasks whole train-split functions, for early stopping only
    heldout_seed: known functions, unseen z solutions (seed in heldout_seed; one index or a set)
    heldout_param / heldout_composition / heldout_family: unseen functions (bank split)"""
    held = set(heldout_seed) if isinstance(heldout_seed, (set, frozenset, list, tuple)) else {heldout_seed}
    good = [r for r in bank if keep(r)]
    train_fns_by_family: dict[str, list[str]] = defaultdict(list)
    for r in good:
        if r["split"] == "train" and r["task"] not in train_fns_by_family[r["family"]]:
            train_fns_by_family[r["family"]].append(r["task"])
    rng = random.Random(seed)
    val_tasks: set[str] = set()
    families = sorted(train_fns_by_family)
    while len(val_tasks) < n_val_tasks and any(train_fns_by_family.values()):
        for fam in families:
            pool = [t for t in train_fns_by_family[fam] if t not in val_tasks]
            if pool and len(val_tasks) < n_val_tasks:
                val_tasks.add(rng.choice(pool))

    parts: dict[str, list[dict]] = defaultdict(list)
    for r in good:
        if r["split"] == "train":
            if r["task"] in val_tasks:
                parts["val"].append(r)
            elif r["seed_idx"] in held:
                parts["heldout_seed"].append(r)
            else:
                parts["train"].append(r)
        else:
            parts[r["split"]].append(r)
    return dict(parts)


def save_decoder(path: Path, reader: ZReader, loras: dict[str, LoRALinear], step: int, val_loss: float) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save({"reader": reader.state_dict(), "lora": lora_state_dict(loras), "step": step, "val_loss": val_loss}, tmp)
    tmp.replace(path)


@torch.no_grad()
def val_loss(model, reader, loras, records, specs, batch: int = 16) -> float:
    total, n = 0.0, 0
    with lora_enabled(loras, True):
        for i in range(0, len(records), batch):
            chunk = records[i : i + batch]
            z = torch.stack([r["z"] for r in chunk]).to(model.device)
            descs = [specs[r["task"]].true_description for r in chunk]
            total += describe_loss_from_z(model, reader(z), descs).item() * len(chunk)
            n += len(chunk)
    return total / n


def train_zdecoder(
    model: SoftPromptLM,
    reader: ZReader,
    loras: dict[str, LoRALinear],
    parts: dict[str, list[dict]],
    specs: dict,
    run_dir: Path,
    steps: int,
    batch: int,
    lr: float,
    eval_every: int = 250,
    log_every: int = 25,
    seed: int = 0,
    weight_by_quality: bool = True,
    weight_floor: float = 0.05,
) -> None:
    """Family-balanced batches (uniform family, then uniform z within it). With
    weight_by_quality, each example's loss is weighted by max(q, weight_floor), so strong z's
    contribute more than borderline ones."""
    params = list(reader.parameters()) + lora_parameters(loras)
    for p in params:
        p.requires_grad_(True)
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    rng = random.Random(seed)
    by_family: dict[str, list[dict]] = defaultdict(list)
    for r in parts["train"]:
        by_family[r["family"]].append(r)
    families = sorted(by_family)
    best = float("inf")
    log_file = open(run_dir / "train_log.jsonl", "a")
    t0 = time.time()

    def log(rec):
        print(json.dumps(rec), flush=True)
        log_file.write(json.dumps(rec) + "\n")
        log_file.flush()

    reader.train()
    for step in range(steps):
        for group in optimizer.param_groups:
            group["lr"] = lr * min(1.0, (step + 1) / 100)
        recs = [rng.choice(by_family[rng.choice(families)]) for _ in range(batch)]
        z = torch.stack([r["z"] for r in recs]).to(model.device)
        descs = [rng.choice(specs[r["task"]].paraphrases) for r in recs]
        weights = torch.tensor([max(r["q"], weight_floor) for r in recs]) if weight_by_quality else None
        with lora_enabled(loras, True):
            loss = describe_loss_from_z(model, reader(z), descs, weights=weights)
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()

        if (step + 1) % log_every == 0 or step == 0:
            log({"step": step + 1, "loss": round(loss.item(), 4), "grad_norm": round(float(grad_norm), 3),
                 "sec": round(time.time() - t0, 1)})
        if (step + 1) % eval_every == 0 or step + 1 == steps:
            reader.eval()
            vl = val_loss(model, reader, loras, parts["val"], specs)
            reader.train()
            improved = vl < best
            if improved:
                best = vl
                save_decoder(run_dir / "decoder.pt", reader, loras, step + 1, vl)
            log({"step": step + 1, "val_loss": round(vl, 4), "best": round(best, 4), "saved": improved})
    log_file.close()
    reader.eval()
