"""Free soft prompts fitted per (behaviour, support view): the baseline for the shared encoder.

Each behaviour gets `--views` disjoint support sets of size N (the same draws the encoder would
see), and one free z per view (different initialisation seed too). This is the identifiability
control: same behaviour, different support + different init, fitted independently.

  python scripts/free_z_views.py --data data/behaviors/library_qwen35_9b.json --out results/encoder/free_z.pt
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch

from encoder_common import LABELS, agreement, draw_views, load
from behavior_soft_prompt import fit
from src.soft_prompt import DEFAULT_CARRIER, SoftPromptLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/behaviors/library_qwen35_9b.json")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--views", type=int, default=2)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--queries", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--only", default=None, help="comma-separated behaviour names (default: all)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    data = load(a.data)
    stim, pool = data["stimuli"], list(data["pool"])
    names = a.only.split(",") if a.only else list(data["behaviours"])
    model = SoftPromptLM(DEFAULT_CARRIER, n_soft_tokens=16)
    cand_ids = model.candidate_token_ids(list(LABELS))
    jobs = []
    for b in names:
        out = data["behaviours"][b]["outputs"]
        views, _ = draw_views(pool, out, a.n, a.views, random.Random(f"fz-{b}"))
        for v, ix in enumerate(views):
            jobs.append({"name": b, "view": v, "idx": ix, "support": [(stim[i]["text"], out[i]) for i in ix],
                         "seed": 5000 + 17 * len(jobs)})
    res, t0 = {}, time.time()
    for c in range(0, len(jobs), a.chunk):
        chunk = jobs[c:c + a.chunk]
        Z = fit(model, chunk, cand_ids, a.steps, 0.03, a.queries, 16)
        for k, j in enumerate(chunk):
            out = data["behaviours"][j["name"]]["outputs"]
            res[(j["name"], j["view"])] = {
                "z": Z[k].float().cpu(), "idx": j["idx"],
                "novel": agreement(model, Z[k], data["test_sets"]["novel"], out, stim, cand_ids),
                "in": agreement(model, Z[k], data["test_sets"]["in"], out, stim, cand_ids)}
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        torch.save(res, a.out)
        print(f"{len(res)}/{len(jobs)} fits, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
