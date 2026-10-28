"""Fidelity, canonical geometry and attribute transfer for free z vs shared encoders.

  python scripts/eval_canonicalization.py --encoders lam1_s0,lam0_s0,lam1_s1,shuf_s0,attrs_s0

Every system is evaluated on the SAME two disjoint support views per behaviour (the draws
stored by free_z_views.py), so free z and encoders see identical supports.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from encoder_common import LABELS, agreement, agreement_matrix, feature_table, load, support_tensors
from src.datasets.behavior_library import split_behaviors
from src.shared_encoder import Encoder
from src.soft_prompt import DEFAULT_CARRIER, SoftPromptLM

ATOMS = ("animal", "food", "object", "person", "number", "question")


def flat(z):
    return torch.nn.functional.normalize(z.flatten().float(), dim=0).numpy()


def retrieval(Z0, Z1, names, ok, query_set):
    """top-1/top-5 of view-0 queries against the view-1 gallery of ALL behaviours.
    `ok[i][j]` marks gallery items that count as correct for query i (itself, or a near-duplicate)."""
    G = np.stack([flat(Z1[n]) for n in names])
    res = {"top1": [], "top5": [], "top1_strict": []}
    for n in query_set:
        i = names.index(n)
        sims = G @ flat(Z0[n])
        order = np.argsort(-sims)
        res["top1_strict"].append(order[0] == i)
        res["top1"].append(bool(ok[i][order[0]]))
        res["top5"].append(any(ok[i][j] for j in order[:5]))
    return {k: round(float(np.mean(v)), 3) for k, v in res.items()}


def dist_ratio(Z0, Z1, names, dup, query_set):
    X0 = np.stack([Z0[n].flatten().float().numpy() for n in names]); X1 = np.stack([Z1[n].flatten().float().numpy() for n in names])
    out = []
    for n in query_set:
        i = names.index(n)
        d = np.linalg.norm(X1 - X0[i], axis=1)
        other = [j for j in range(len(names)) if not dup[i][j]]
        out.append(d[i] / d[other].mean())
    return round(float(np.mean(out)), 3)


def probe(Z, names, meta, train, test):
    """Logistic probes z -> trigger atoms (6 binary) and effect (3-way), fit on train behaviours."""
    def X(ns): return np.stack([flat(Z[n]) for n in ns])
    Xtr, Xte = X(train), X(test)
    res = {}
    accs, base = [], []
    for a in ATOMS:
        ytr = np.array([a in meta[n]["atoms"] for n in train]); yte = np.array([a in meta[n]["atoms"] for n in test])
        if len(set(ytr)) < 2:
            continue
        clf = LogisticRegression(C=10.0, max_iter=2000).fit(Xtr, ytr)
        accs.append((clf.predict(Xte) == yte).mean()); base.append(max(yte.mean(), 1 - yte.mean()))
    res["atom_acc"], res["atom_majority"] = round(float(np.mean(accs)), 3), round(float(np.mean(base)), 3)
    ytr = np.array([meta[n]["effect"] for n in train]); yte = np.array([meta[n]["effect"] for n in test])
    clf = LogisticRegression(C=10.0, max_iter=2000).fit(Xtr, ytr)
    vals, cnt = np.unique(yte, return_counts=True)
    res["effect_acc"], res["effect_majority"] = round(float((clf.predict(Xte) == yte).mean()), 3), round(float(cnt.max() / cnt.sum()), 3)
    return res


def linear_cka(A, B):
    A = A - A.mean(0); B = B - B.mean(0)
    return float(np.linalg.norm(A.T @ B) ** 2 / (np.linalg.norm(A.T @ A) * np.linalg.norm(B.T @ B)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/behaviors/library_qwen35_9b.json")
    ap.add_argument("--free", default="results/encoder/free_z.pt")
    ap.add_argument("--encoders", default="")
    ap.add_argument("--out", default="results/encoder/canonicalization.json")
    a = ap.parse_args()
    data = load(a.data)
    stim, meta, names = data["stimuli"], data["meta"], list(data["behaviours"])
    outs = {n: data["behaviours"][n]["outputs"] for n in names}
    train, rand, comp = split_behaviors(meta, names)
    held = rand + comp
    _, A = agreement_matrix(outs)
    dup = A > 0.95
    ok = dup  # a retrieval is correct if it is the behaviour or a behaviourally identical one
    free = torch.load(a.free)
    idx = {(b, v): free[(b, v)]["idx"] for b in names for v in (0, 1)}
    model = SoftPromptLM(DEFAULT_CARRIER, n_soft_tokens=16)
    cand = model.candidate_token_ids(list(LABELS))
    results, allz = {}, {}

    def evaluate(tag, Z):  # Z[(b, v)] -> [16, d] cpu
        Z0 = {b: Z[(b, 0)] for b in names}; Z1 = {b: Z[(b, 1)] for b in names}
        fid = {}
        for grp, bs in (("train", train), ("held_random", rand), ("held_comp", comp)):
            nov, inv = [], []
            for b in bs[:30]:
                z = Z0[b].to(model.device)
                nov.append(agreement(model, z, data["test_sets"]["novel"], outs[b], stim, cand))
                inv.append(agreement(model, z, data["test_sets"]["in"], outs[b], stim, cand))
            fid[grp] = {"novel": round(float(np.mean(nov)), 3), "in": round(float(np.mean(inv)), 3)} if nov else None
        r = {"fidelity": fid,
             "retrieval_held": retrieval(Z0, Z1, names, ok, held), "retrieval_train": retrieval(Z0, Z1, names, ok, train[:30]),
             "dist_ratio_held": dist_ratio(Z0, Z1, names, dup, held), "dist_ratio_train": dist_ratio(Z0, Z1, names, dup, train[:30]),
             "probe_held": probe({**Z0}, names, meta, train, held)}
        results[tag] = r; allz[tag] = Z0
        print(tag, json.dumps(r), flush=True)

    evaluate("free_z", {k: v["z"] for k, v in free.items()})
    for e in [x for x in a.encoders.split(",") if x]:
        ck = torch.load(f"results/encoder/{e}.pt")
        args = ck["args"]
        feats = feature_table(model, data, f"results/encoder/features_{Path(a.data).stem}.pt", args["features"])
        enc = Encoder(ck["d_in"], model.embed_layer.weight.shape[1], z_norm=ck["z_norm"], d=args.get("d", 512), layers=args.get("layers", 2)).to(model.device)
        enc.load_state_dict(ck["enc"]); enc.eval()
        Z = {}
        with torch.no_grad():
            for b in names:
                for v in (0, 1):
                    f, y = support_tensors(feats, outs[b], [idx[(b, v)]], model.device)
                    Z[(b, v)] = enc(f, y)[0].float().cpu()
        evaluate(e, Z)
    # cross-run: do two independently trained encoders agree? CKA and Procrustes-aligned retrieval
    es = [e for e in a.encoders.split(",") if e.startswith("lam1")]
    if len(es) >= 2:
        P = {e: np.stack([allz[e][b].flatten().numpy() for b in names]) for e in es[:2]}
        cka = linear_cka(P[es[0]], P[es[1]])
        from scipy.linalg import orthogonal_procrustes
        U, S, _ = np.linalg.svd(np.concatenate([P[es[0]], P[es[1]]]) - 0, full_matrices=False)
        k = 64
        proj = lambda X, V: X @ V
        _, _, Vt = np.linalg.svd(P[es[0]] - P[es[0]].mean(0), full_matrices=False); V0 = Vt[:k].T
        _, _, Vt = np.linalg.svd(P[es[1]] - P[es[1]].mean(0), full_matrices=False); V1 = Vt[:k].T
        X0, X1 = (P[es[0]] - P[es[0]].mean(0)) @ V0, (P[es[1]] - P[es[1]].mean(0)) @ V1
        ti = [names.index(n) for n in train]; hi = [names.index(n) for n in held]
        R, _ = orthogonal_procrustes(X1[ti], X0[ti])
        M = X1 @ R
        hit = []
        for i in hi:
            d = np.linalg.norm(X0 - M[i], axis=1)
            hit.append(bool(ok[i][int(np.argmin(d))]))
        results["cross_run"] = {"pair": es[:2], "linear_cka": round(cka, 3), "procrustes_top1_held": round(float(np.mean(hit)), 3),
                                "cos_before_alignment": round(float(np.mean([flat(torch.tensor(P[es[0]][i])) @ flat(torch.tensor(P[es[1]][i])) for i in hi])), 3)}
        print("cross_run", results["cross_run"])
    Path(a.out).write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
