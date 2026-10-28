"""Build (or resume) a bank of independently optimized per-task soft prompts.

Seeds per function differ only in the random-vocab initialization of z (and minibatch order),
so the bank contains several distinct z* solutions for the same behavior.

Usage:
  python -u build_zbank.py --out runs/zbank_pilot --limit-tasks 16 --seeds-train 1 --seeds-heldout 0 --steps 300
  python -u build_zbank.py --out runs/zbank --seeds-train 2 --seeds-heldout 2
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch

from run_shared_encoder import stratified
from verbalize.langspace import LangSubspace
from verbalize.model import SoftPromptLM
from verbalize.tasks import build_task_suite
from verbalize.zbank import ZBankConfig, ZJob, optimize_batch


def spread(tasks, max_n: int):
    """Evenly spaced parameter values within each family.

    `stratified` takes each family's grid PREFIX, so a capped bank gets every threshold
    bunched at the low end (gt_-20, gt_-19) and only k=2 for mod. Recovering parameters
    from z is the decoder's known weak point, so the bank has to span each grid instead.
    """
    by_family = defaultdict(list)
    for t in tasks:
        by_family[t.family].append(t)
    families = sorted(by_family)
    per = max(1, max_n // len(families))
    out = []
    for fam in families:
        pool = by_family[fam]
        if len(pool) <= per:
            out += pool
        else:
            step = len(pool) / per
            out += [pool[int(i * step)] for i in range(per)]
    return out


def interleaved_jobs(suite, seeds_train: int, seeds_heldout: int, limit_tasks: int | None,
                     sampler=None) -> list[ZJob]:
    """Seed 0 of every task first (round-robin across families), then seed 1, ... so a partial
    bank still covers every family."""
    per_seed: dict[int, list[ZJob]] = defaultdict(list)
    for split, tasks in suite.items():
        n_seeds = seeds_train if split == "train" else seeds_heldout
        chosen = (sampler or stratified)(tasks, limit_tasks) if limit_tasks else tasks
        for seed_idx in range(n_seeds):
            per_seed[seed_idx] += [ZJob(spec, split, seed_idx) for spec in chosen]
    jobs = []
    for seed_idx in sorted(per_seed):
        by_family = defaultdict(list)
        for job in per_seed[seed_idx]:
            by_family[job.spec.family].append(job)
        i = 0
        while any(i < len(v) for v in by_family.values()):
            jobs += [v[i] for _, v in sorted(by_family.items()) if i < len(v)]
            i += 1
    return jobs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--seeds-train", type=int, default=2)
    p.add_argument("--seeds-heldout", type=int, default=2)
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--batch-z", type=int, default=8)
    p.add_argument("--queries", type=int, default=6)
    p.add_argument("--lr", type=float, default=0.3)
    p.add_argument("--limit-tasks", type=int, help="stratified cap on tasks per split (pilot runs)")
    p.add_argument("--tasks", help="comma-separated task names to restrict to (sweeps)")
    p.add_argument("--eval-at", help="comma-separated steps at which to log train/eval accuracy")
    p.add_argument("--subspace", help="LangSubspace file: optimize a latent u with z = subspace.to_z(u) (step 5)")
    p.add_argument("--k-latent", type=int, default=128, help="latent size when --subspace is set")
    p.add_argument("--u-init-std", type=float, default=0.5)
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"],
                   help="bfloat16 matches float32 on this path (measured over 3 seeds x 8 tasks) "
                        "and is required for carriers above ~2B on a 32 GB card.")
    p.add_argument("--n-soft-tokens", type=int, default=16)
    p.add_argument("--task-sample", default="prefix", choices=["prefix", "spread"])
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))
    bank_path = out / "bank.pt"
    bank = torch.load(bank_path, weights_only=False) if bank_path.exists() else []
    done = {(r["task"], r["seed_idx"]) for r in bank}

    sampler = spread if args.task_sample == "spread" else stratified
    jobs = [j for j in interleaved_jobs(build_task_suite(), args.seeds_train, args.seeds_heldout,
                                        args.limit_tasks, sampler)
            if j.key not in done]
    if args.tasks:
        keep = set(args.tasks.split(","))
        jobs = [j for j in jobs if j.spec.name in keep]
    print(f"{len(done)} z's already in bank, {len(jobs)} to optimize", flush=True)

    model = SoftPromptLM(args.model, n_soft_tokens=args.n_soft_tokens,
                         dtype=getattr(torch, args.dtype))
    eval_at = tuple(int(s) for s in args.eval_at.split(",")) if args.eval_at else ()
    cfg = ZBankConfig(n_soft_tokens=args.n_soft_tokens, steps=args.steps,
                      queries_per_step=args.queries, lr=args.lr, eval_at=eval_at,
                      u_init_std=args.u_init_std)
    subspace = LangSubspace.load(args.subspace, model.device) if args.subspace else None
    log_file = open(out / "build_log.jsonl", "a")

    def log(rec):
        print(json.dumps(rec), flush=True)
        log_file.write(json.dumps(rec) + "\n")
        log_file.flush()

    t0 = time.time()
    for b in range(0, len(jobs), args.batch_z):
        batch = jobs[b : b + args.batch_z]
        tb = time.time()
        records = optimize_batch(model, batch, cfg, log=lambda r: log({"batch": b // args.batch_z, **r}),
                                 subspace=subspace, k_latent=args.k_latent)
        bank += records
        tmp = bank_path.with_suffix(".tmp")
        torch.save(bank, tmp)
        tmp.replace(bank_path)
        log({
            "batch_done": b // args.batch_z,
            "bank_size": len(bank),
            "remaining": len(jobs) - b - len(batch),
            "sec_batch": round(time.time() - tb, 1),
            "sec_total": round(time.time() - t0, 1),
            "results": [
                {"task": r["task"], "seed": r["seed_idx"], "train_acc": round(r["train_acc"], 3),
                 "eval_acc": round(r["eval_acc"], 3), "z_norm": round(r["z_norm"], 2)}
                for r in records
            ],
        })
    log_file.close()
    print(f"bank complete: {len(bank)} z's at {bank_path}", flush=True)


if __name__ == "__main__":
    main()
