"""Test-time refinement of encoder z, with geometry tracked at every step count.

Start from E(support) and take k Adam steps on z against the same support set (the free-z loss).
At k in --ks, snapshot and measure: fidelity (novel/in), same-behaviour retrieval across the two
views, same/diff distance ratio, attribute probe, and drift from the encoder's z (cosine).
The free-z baseline is scored on the identical behaviour subset (gallery = the subset).

  python scripts/refine_sweep.py --encoder lam1_s0 --lr 0.003
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from encoder_common import LABELS, agreement, agreement_matrix, feature_table, load, prefix, support_tensors
from eval_canonicalization import dist_ratio, flat, probe, retrieval
from src.datasets.behavior_library import split_behaviors
from src.shared_encoder import Encoder
from src.soft_prompt import DEFAULT_CARRIER, SoftPromptLM
from src.zbank import balanced_sample


def geometry(model, data, cand, sub, Z, train, held, ok, dup, meta, tag):
    stim, outs = data["stimuli"], {n: data["behaviours"][n]["outputs"] for n in sub}
    Z0 = {b: Z[(b, 0)] for b in sub}; Z1 = {b: Z[(b, 1)] for b in sub}
    fid = {}
    for grp, bs in (("train", train), ("held_random", [b for b in held if b not in COMP]), ("held_comp", [b for b in held if b in COMP])):
        v = [agreement(model, Z0[b].to(model.device), data["test_sets"]["novel"], outs[b], stim, cand) for b in bs]
        fid[grp] = round(float(np.mean(v)), 3)
    r = {"tag": tag, "fidelity": fid,
         "retrieval_held": retrieval(Z0, Z1, sub, ok, held), "dist_ratio_held": dist_ratio(Z0, Z1, sub, dup, held),
         "dist_ratio_train": dist_ratio(Z0, Z1, sub, dup, train), "probe_held": probe(Z0, sub, meta, train, held)}
    return r


def main():
    global COMP
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/behaviors/library_qwen35_9b.json")
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--lr", type=float, default=0.003)
    ap.add_argument("--ks", type=lambda s: [int(x) for x in s.split(",")], default=[0, 10, 25, 50, 100, 150])
    ap.add_argument("--n-train", type=int, default=30)
    ap.add_argument("--chunk", type=int, default=8)
    a = ap.parse_args()
    data = load(a.data)
    stim, meta, names = data["stimuli"], data["meta"], list(data["behaviours"])
    outs = {n: data["behaviours"][n]["outputs"] for n in names}
    train, rand, comp = split_behaviors(meta, names)
    COMP = set(comp)
    train = train[:a.n_train]; held = rand + comp
    sub = train + held
    _, A = agreement_matrix({n: outs[n] for n in sub}); dup = A > 0.95
    free = torch.load("results/encoder/free_z.pt")
    idx = {(b, v): free[(b, v)]["idx"] for b in sub for v in (0, 1)}
    model = SoftPromptLM(DEFAULT_CARRIER, n_soft_tokens=16)
    cand = model.candidate_token_ids(list(LABELS))
    ck = torch.load(f"results/encoder/{a.encoder}.pt")
    feats = feature_table(model, data, f"results/encoder/features_{Path(a.data).stem}.pt", ck["args"]["features"])
    enc = Encoder(ck["d_in"], model.embed_layer.weight.shape[1], z_norm=ck["z_norm"], d=ck["args"].get("d", 512), layers=ck["args"].get("layers", 2)).to(model.device)
    enc.load_state_dict(ck["enc"]); enc.eval()
    keys = [(b, v) for b in sub for v in (0, 1)]
    with torch.no_grad():
        init = {k: enc(*support_tensors(feats, outs[k[0]], [idx[k]], model.device))[0].float() for k in keys}
    results = [geometry(model, data, cand, sub, {k: free[k]["z"] for k in keys}, train, held, dup, dup, meta, "free_z(500 steps)")]
    print(json.dumps(results[-1]), flush=True)
    snaps = {k: {} for k in a.ks}; t0 = time.time()
    for c in range(0, len(keys), a.chunk):
        chunk = keys[c:c + a.chunk]
        param = torch.nn.Parameter(torch.stack([init[k] for k in chunk]).clone())
        opt = torch.optim.Adam([param], lr=a.lr)
        rngs = [random.Random(hash(k) % 10**6) for k in chunk]
        sup = [[(stim[i]["text"], outs[k[0]][i]) for i in idx[k]] for k in chunk]
        for step in range(max(a.ks) + 1):
            if step in snaps:
                for j, k in enumerate(chunk):
                    snaps[step][k] = param[j].detach().float().cpu()
            if step == max(a.ks):
                break
            items = [(j, x, y) for j in range(len(chunk)) for x, y in balanced_sample(sup[j], 6, rngs[j])]
            lp = model.candidate_logprobs([prefix(model, x, param[j]) for j, x, _ in items], cand)
            tgt = torch.tensor([LABELS.index(y) for *_, y in items], device=model.device)
            loss = F.cross_entropy(lp, tgt, reduction="sum") / (len(chunk) * 6)
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"{c + len(chunk)}/{len(keys)} jobs, {time.time() - t0:.0f}s", flush=True)
    init_cpu = {k: v.cpu() for k, v in init.items()}
    for kk in a.ks:
        Z = snaps[kk]
        r = geometry(model, data, cand, sub, Z, train, held, dup, dup, meta, f"{a.encoder}+{kk}steps lr{a.lr}")
        r["cos_to_encoder_z"] = round(float(np.mean([flat(Z[k]) @ flat(init_cpu[k]) for k in keys])), 3)
        results.append(r); print(json.dumps(r), flush=True)
    Path(f"results/encoder/refine_{a.encoder}_lr{a.lr}.json").write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
