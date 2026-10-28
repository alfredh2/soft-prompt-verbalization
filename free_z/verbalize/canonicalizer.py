"""Decoder plan step 2: canonicalize z before decoding, z -> C(z) = z~ -> G(z~).

Step 1 showed per-task z* are non-identifiable: z's for the same function from different seeds
are no closer than z's for different functions (distance ratio ~1.00 over 829 z's), so the
decoder memorized training z's and decoded 0% of new seeds exactly. C is trained so that
different seeds of one function land together and different functions stay apart
(symmetric InfoNCE), jointly with the text-decoding loss of the LoRA decoder reading z~.

C is a set encoder over z's 16 vectors with no positional encoding (which slot carries what is
arbitrary across seeds), followed by learned output queries.
"""

from __future__ import annotations

import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder import describe_loss_from_z
from .lora import LoRALinear, lora_enabled, lora_parameters
from .model import SoftPromptLM
from .zdecoder import save_decoder, val_loss


class Canonicalizer(nn.Module):
    def __init__(
        self, hidden: int, target_norm: float, n_out: int = 16, d: int = 512,
        n_layers: int = 2, n_heads: int = 8, dropout: float = 0.1,
    ):
        super().__init__()
        self.in_norm = nn.LayerNorm(hidden)
        self.in_proj = nn.Linear(hidden, d)
        layer = nn.TransformerEncoderLayer(
            d, n_heads, dim_feedforward=4 * d, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.queries = nn.Parameter(torch.randn(n_out, d) * 0.02)
        self.q_norm = nn.LayerNorm(d)
        self.kv_norm = nn.LayerNorm(d)
        self.cross_attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.out_norm = nn.LayerNorm(d)
        self.out_proj = nn.Linear(d, hidden)
        self.register_buffer("target_norm", torch.tensor(float(target_norm)))

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        """Canonical code [B, n_out, d]; this is the space the contrastive loss acts on."""
        h = self.set_encoder(self.in_proj(self.in_norm(z)))
        kv = self.kv_norm(h)
        q = self.queries.unsqueeze(0).expand(z.shape[0], -1, -1)
        q = q + self.cross_attn(self.q_norm(q), kv, kv, need_weights=False)[0]
        return q + self.ffn(q)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.out_proj(self.out_norm(self.encode(z)))
        return F.normalize(out, dim=-1) * self.target_norm


def pair_groups(records: list[dict]) -> dict[str, list[dict]]:
    """Tasks with at least two z's (different seeds) -- the only ones that give positive pairs."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[r["task"]].append(r)
    return {t: rs for t, rs in groups.items() if len(rs) >= 2}


def info_nce(a: torch.Tensor, b: torch.Tensor, temp: float) -> tuple[torch.Tensor, float]:
    a, b = F.normalize(a, dim=-1), F.normalize(b, dim=-1)
    logits = a @ b.T / temp
    targets = torch.arange(len(a), device=a.device)
    loss = (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2
    return loss, (logits.argmax(dim=1) == targets).float().mean().item()


def sample_pairs(groups: dict[str, list[dict]], n: int, rng: random.Random) -> list[tuple[dict, dict]]:
    """n distinct tasks, family-balanced, two different seeds each."""
    by_family: dict[str, list[str]] = defaultdict(list)
    for task, rs in groups.items():
        by_family[rs[0]["family"]].append(task)
    families = sorted(by_family)
    chosen: list[str] = []
    for _ in range(20 * n):
        if len(chosen) >= min(n, len(groups)):
            break
        task = rng.choice(by_family[rng.choice(families)])
        if task not in chosen:
            chosen.append(task)
    return [tuple(rng.sample(groups[t], 2)) for t in chosen]


@torch.no_grad()
def val_retrieval(canon: Canonicalizer, groups: dict[str, list[dict]], device, temp: float) -> tuple[float, float]:
    tasks = sorted(groups)
    za = torch.stack([groups[t][0]["z"] for t in tasks]).to(device)
    zb = torch.stack([groups[t][1]["z"] for t in tasks]).to(device)
    loss, acc = info_nce(canon.encode(za).flatten(1), canon.encode(zb).flatten(1), temp)
    return loss.item(), acc


@torch.no_grad()
def val_view_retrieval(canon: Canonicalizer, records: list[dict], device) -> tuple[float, float]:
    """Every val z (all seeds): is its nearest other z (cosine, canonical code) from the same
    function? Less noisy than two-seed InfoNCE top-1 on a handful of functions."""
    if len(records) < 2:
        return 0.0, 0.0
    z = torch.stack([r["z"] for r in records]).to(device)
    codes = F.normalize(canon.encode(z).flatten(1), dim=-1)
    sims = codes @ codes.T
    sims.fill_diagonal_(-float("inf"))
    tasks = [r["task"] for r in records]
    nearest = sims.argmax(dim=1).tolist()
    acc = sum(tasks[j] == tasks[i] for i, j in enumerate(nearest)) / len(records)
    counts = Counter(tasks)
    chance = sum((counts[t] - 1) / (len(records) - 1) for t in tasks) / len(records)
    return acc, chance


def train_canonical_decoder(
    model: SoftPromptLM,
    canon: Canonicalizer,
    loras: dict[str, LoRALinear],
    parts: dict[str, list[dict]],
    specs: dict,
    run_dir: Path,
    steps: int,
    batch: int,
    lr: float,
    contrast_weight: float = 1.0,
    contrast_temp: float = 0.1,
    contrast_batch: int = 32,
    eval_every: int = 100,
    log_every: int = 25,
    seed: int = 0,
    weight_by_quality: bool = True,
    weight_floor: float = 0.05,
) -> None:
    params = list(canon.parameters()) + lora_parameters(loras)
    for p in params:
        p.requires_grad_(True)
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    rng = random.Random(seed)
    by_family: dict[str, list[dict]] = defaultdict(list)
    for r in parts["train"]:
        by_family[r["family"]].append(r)
    families = sorted(by_family)
    train_groups = pair_groups(parts["train"])
    val_groups = pair_groups(parts.get("val", []))
    best = float("inf")
    best_ret, best_ret_loss = -1.0, float("inf")
    log_file = open(run_dir / "train_log.jsonl", "a")
    t0 = time.time()

    def log(rec):
        print(json.dumps(rec), flush=True)
        log_file.write(json.dumps(rec) + "\n")
        log_file.flush()

    log({"train_z": len(parts["train"]), "train_pair_tasks": len(train_groups), "val_pair_tasks": len(val_groups)})
    canon.train()
    for step in range(steps):
        for group in optimizer.param_groups:
            group["lr"] = lr * min(1.0, (step + 1) / 100)

        recs = [rng.choice(by_family[rng.choice(families)]) for _ in range(batch)]
        z = torch.stack([r["z"] for r in recs]).to(model.device)
        descs = [rng.choice(specs[r["task"]].paraphrases) for r in recs]
        weights = torch.tensor([max(r["q"], weight_floor) for r in recs]) if weight_by_quality else None
        with lora_enabled(loras, True):
            l_text = describe_loss_from_z(model, canon(z), descs, weights=weights)

        l_con, con_acc = l_text.new_zeros(()), None
        if contrast_weight > 0 and len(train_groups) >= 2:
            pairs = sample_pairs(train_groups, contrast_batch, rng)
            za = torch.stack([a["z"] for a, _ in pairs]).to(model.device)
            zb = torch.stack([b["z"] for _, b in pairs]).to(model.device)
            l_con, con_acc = info_nce(canon.encode(za).flatten(1), canon.encode(zb).flatten(1), contrast_temp)

        total = l_text + contrast_weight * l_con
        optimizer.zero_grad()
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()

        if (step + 1) % log_every == 0 or step == 0:
            log({"step": step + 1, "loss_text": round(l_text.item(), 4), "loss_contrast": round(l_con.item(), 4),
                 "contrast_acc": None if con_acc is None else round(con_acc, 3),
                 "grad_norm": round(float(grad_norm), 3), "sec": round(time.time() - t0, 1)})
        if (step + 1) % eval_every == 0 or step + 1 == steps:
            canon.eval()
            vl = val_loss(model, canon, loras, parts["val"], specs)
            vr = val_retrieval(canon, val_groups, model.device, contrast_temp) if len(val_groups) >= 2 else (None, None)
            view_acc, view_chance = val_view_retrieval(canon, parts.get("val", []), model.device)
            canon.train()
            improved = vl < best
            if improved:
                best = vl
                save_decoder(run_dir / "decoder.pt", canon, loras, step + 1, vl)
            ret_improved = view_acc > best_ret or (view_acc == best_ret and vl < best_ret_loss)
            if ret_improved:
                best_ret, best_ret_loss = view_acc, vl
                save_decoder(run_dir / "decoder_retrieval.pt", canon, loras, step + 1, vl)
            log({"step": step + 1, "val_loss": round(vl, 4), "best": round(best, 4), "saved": improved,
                 "val_retrieval_acc": None if vr[1] is None else round(vr[1], 3),
                 "val_retrieval_chance": round(1 / max(1, len(val_groups)), 3),
                 "val_view_nn_acc": round(view_acc, 3), "val_view_nn_chance": round(view_chance, 3),
                 "best_view_nn_acc": round(best_ret, 3), "saved_retrieval": ret_improved})
    log_file.close()
    canon.eval()
