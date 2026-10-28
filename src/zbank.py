"""Optimising one soft prompt per function, and the bank of them.

For each dataset record we fit a `z` so the frozen carrier reproduces that record's
function, then keep `z` alongside how well it worked. The bank is the input to any method
that tries to turn behaviour into language.

Many z's are optimised at once: each is an independent row of a single parameter tensor,
and their query sequences share one batched forward pass, so rows never interact.

Settings that are measured, not guessed:

  lr 0.03, 500 steps   lr 0.3 is unstable (easy rules stall near chance, z norms ~87);
                       lr 0.1 reaches ~0.80 train. Accuracy is still climbing at step 300
                       at lr 0.03, so the long run is needed.
  bfloat16             matches float32 on this path over 3 seeds x 8 tasks, and is what
                       makes a 4B carrier fit on a 32 GB card at all.
  16 soft tokens       2, 4, 8 and 16 were indistinguishable; even 2 tokens memorise a
                       45-example training set, so capacity is not the binding constraint.

Known limit: modular rules (`mod`, `*_and_mod`) fit their training examples at ~1.00 and
still generalise at ~0.5, below the majority-class baseline. That is overfitting driven by
a small domain, and no carrier, capacity or seed setting tested has fixed it -- filter
those z's with a keep-rule rather than expecting them to improve.
"""

from __future__ import annotations

import random
import zlib
from collections import defaultdict
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from src.soft_prompt import SoftPromptLM, render_input


@dataclass
class ZConfig:
    n_soft_tokens: int = 16
    steps: int = 500
    lr: float = 0.03
    queries_per_step: int = 6
    log_every: int = 100
    # 'cosine' anneals the rate to zero so z settles instead of orbiting its optimum; a constant
    # rate leaves a few-percent error on the examples it is being fitted to.
    lr_decay: str = "none"


@dataclass
class ZJob:
    """One (record, seed) pair to optimise."""

    record: dict
    seed_idx: int = 0

    @property
    def key(self) -> tuple[str, int, int, int]:
        """Identity of a run. n_examples is part of it: the same function fitted from 8
        examples and from 32 are different runs, not duplicates of each other."""
        return (self.record["function_id"], self.record["view"],
                self.record["n_examples"], self.seed_idx)

    @property
    def init_seed(self) -> int:
        """Stable per-job seed: same job yields the same initialisation on any machine."""
        name = f'{self.record["function_id"]}#{self.record["view"]}'
        return zlib.crc32(name.encode()) + 7919 * self.seed_idx


@torch.no_grad()
def domain_predictions(model: SoftPromptLM, z: torch.Tensor, labels: list[str],
                       inputs: list, batch_size: int = 64) -> list[str]:
    """The label z assigns to every input -- its behaviour as a truth table.

    A soft prompt is not a program, but it is a function, and a function can be read off
    by querying it everywhere. Over a 61-input domain that is 61 forward passes, which
    gives a full-domain exactness measure directly comparable to an executed snippet's,
    and a labelled table that any downstream method can use without touching z's weights.
    """
    cand_ids = model.candidate_token_ids(list(labels))
    out: list[str] = []
    for start in range(0, len(inputs), batch_size):
        chunk = inputs[start : start + batch_size]
        prefixes = [model.build_prefix("soft", render_input(x), z=z) for x in chunk]
        idx = model.candidate_logprobs(prefixes, cand_ids).argmax(dim=-1).tolist()
        out += [labels[i] for i in idx]
    return out


def init_z(model: SoftPromptLM, seed: int, n_tokens: int) -> torch.Tensor:
    """Initialise from the embeddings of random vocabulary tokens, not from noise."""
    gen = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, model.embed_layer.weight.shape[0], (n_tokens,), generator=gen)
    with torch.no_grad():
        return model.embed_layer(ids.to(model.device)).float().clone()


def balanced_sample(examples: list, n: int, rng: random.Random) -> list:
    """Draw round-robin across labels.

    A plain random draw on a skewed function hands the optimiser mostly one label, and z
    settles on a constant answer that scores well on the training minibatch and encodes
    nothing about the rule.
    """
    by_label: dict[str, list] = defaultdict(list)
    for ex in examples:
        by_label[ex[1]].append(ex)
    labels = sorted(by_label)
    return [rng.choice(by_label[labels[i % len(labels)]]) for i in range(n)]


def optimize_batch(
    model: SoftPromptLM,
    jobs: list[ZJob],
    cfg: ZConfig,
    log=print,
) -> list[dict]:
    """Fit one z per job. Returns a bank record per job.

    Trains on the record's `examples` and reports held-out accuracy on its
    `holdout_examples`, which z never sees -- the same separation every other method obeys.
    """
    param = torch.nn.Parameter(
        torch.stack([init_z(model, j.init_seed, cfg.n_soft_tokens) for j in jobs])
    )
    optimizer = torch.optim.Adam([param], lr=cfg.lr)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.steps)
                 if cfg.lr_decay == "cosine" else None)
    rngs = [random.Random(j.init_seed) for j in jobs]

    for step in range(cfg.steps):
        groups: dict[tuple, list] = defaultdict(list)
        for k, job in enumerate(jobs):
            labels = tuple(job.record["labels"])
            for x, y in balanced_sample(job.record["examples"], cfg.queries_per_step, rngs[k]):
                groups[labels].append((k, x, y))

        loss = param.new_zeros(())
        for labels, items in groups.items():
            cand_ids = model.candidate_token_ids(list(labels))
            prefixes = [
                model.build_prefix("soft", render_input(x), z=param[k]) for k, x, _ in items
            ]
            logprobs = model.candidate_logprobs(prefixes, cand_ids)
            targets = torch.tensor(
                [labels.index(y) for *_, y in items], device=model.device
            )
            loss = loss + F.cross_entropy(logprobs, targets, reduction="sum")
        loss = loss / (len(jobs) * cfg.queries_per_step)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if log and ((step + 1) % cfg.log_every == 0 or step == 0):
            log({"step": step + 1, "batch_loss": round(loss.item(), 4)})

    Z = param.detach()
    records = []
    for k, job in enumerate(jobs):
        rec, z = job.record, Z[k]
        records.append({
            "function_id": rec["function_id"],
            "view": rec["view"],
            "seed_idx": job.seed_idx,
            "family": rec["family"],
            "params": rec["params"],
            "split": rec["split"],
            "n_examples": rec["n_examples"],
            "description": rec["description"],
            "expression": rec["expression"],
            "labels": list(rec["labels"]),
            "majority_label_fraction": rec["majority_label_fraction"],
            "z": z.cpu(),
            "z_norm": z.norm(dim=-1).mean().item(),
            "train_acc": model.accuracy(rec["examples"], rec["labels"], "soft", z=z),
            "eval_acc": model.accuracy(rec["holdout_examples"], rec["labels"], "soft", z=z),
            "carrier": model.model_name,
            "n_soft_tokens": cfg.n_soft_tokens,
            "steps": cfg.steps,
        })
    return records


def keep_rule(min_acc: float = 0.9, min_margin: float = 0.1, min_abs: float = 0.75):
    """Which z's are good enough to build on.

    Held-out accuracy alone is misleading on skewed functions: 0.9 on a function whose
    majority label covers 0.89 of the domain is a constant predictor wearing a rosette.
    A z is kept if it is strong outright, or clears the majority-class baseline by a
    margin while still being reasonably accurate.
    """
    def keep(rec: dict) -> bool:
        const = rec["majority_label_fraction"]
        return rec["eval_acc"] >= min_acc or (
            rec["eval_acc"] - const >= min_margin and rec["eval_acc"] >= min_abs
        )

    return keep


def summarise(bank: list[dict]) -> dict:
    """Bank health, overall and per family."""
    if not bank:
        return {"n": 0}
    by_family: dict[str, list[dict]] = defaultdict(list)
    for r in bank:
        by_family[r["family"]].append(r)
    keep = keep_rule()
    out = {
        "n": len(bank),
        "functions": len({(r["function_id"], r["view"]) for r in bank}),
        "mean_train_acc": round(sum(r["train_acc"] for r in bank) / len(bank), 4),
        "mean_eval_acc": round(sum(r["eval_acc"] for r in bank) / len(bank), 4),
        "frac_eval_ge_0.9": round(sum(r["eval_acc"] >= 0.9 for r in bank) / len(bank), 4),
        "kept_by_rule": sum(keep(r) for r in bank),
        "by_family": {},
    }
    for fam, rows in sorted(by_family.items()):
        out["by_family"][fam] = {
            "n": len(rows),
            "train": round(sum(r["train_acc"] for r in rows) / len(rows), 3),
            "eval": round(sum(r["eval_acc"] for r in rows) / len(rows), 3),
            "constant": round(sum(r["majority_label_fraction"] for r in rows) / len(rows), 3),
        }
    return out
