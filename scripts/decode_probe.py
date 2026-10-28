"""Structured probe from z: can a frozen shared-encoder z be decoded into (trigger atoms, kind, effect)
on HELD-OUT behaviours better than a free z?

Pre-registered gate for trying z -> language generation: shared-z exact-descriptor accuracy beats free z
by >= 0.15 on held-out behaviours with the bootstrap 95% lower bound of the difference above 0.

Systems (all probed identically, logistic heads on the flattened normalised z):
  free        2 free-z fits per behaviour (the stored views)
  enc2        frozen encoder, the same 2 views per behaviour (matched count)
  enc8        frozen encoder, 8 disjoint support draws per TRAIN behaviour (the amortisation advantage)
  labels      control, no z: target labels on 128 fixed sentences (what any readout of behaviour could see)
Targets: intended descriptor (from the instruction) and ACTUAL descriptor (the library behaviour whose
intended outputs best match what the target did).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from encoder_common import LABELS, draw_views, feature_table, load, support_tensors
from src.datasets.behavior_library import build, intended, split_behaviors
from src.shared_encoder import Encoder
from src.soft_prompt import DEFAULT_CARRIER, SoftPromptLM

ATOMS = ("animal", "food", "object", "person", "number", "question")


def flat(z):
    return torch.nn.functional.normalize(z.flatten().float(), dim=0).numpy()


def descriptor(m):
    return {"kind": m["kind"], "effect": m["effect"], "atoms": frozenset(m["atoms"])}


def fit_predict(Xtr, dtr, Xte):
    """Factored heads -> predicted descriptors for Xte rows."""
    preds = [dict() for _ in Xte]
    for key in ("kind", "effect"):
        y = np.array([d[key] for d in dtr])
        if len(set(y)) < 2:
            for p in preds: p[key] = y[0]
            continue
        out = LogisticRegression(C=10.0, max_iter=3000).fit(Xtr, y).predict(Xte)
        for p, o in zip(preds, out): p[key] = o
    for p in preds: p["atoms"] = set()
    for a in ATOMS:
        y = np.array([a in d["atoms"] for d in dtr])
        out = np.zeros(len(Xte), bool) if len(set(y)) < 2 else LogisticRegression(C=10.0, max_iter=3000).fit(Xtr, y).predict(Xte)
        for p, o in zip(preds, out):
            if o: p["atoms"].add(a)
    return preds


def score(p, d):
    atoms_ok = np.mean([(a in p["atoms"]) == (a in d["atoms"]) for a in ATOMS])
    kind, eff, aset = p["kind"] == d["kind"], p["effect"] == d["effect"], p["atoms"] == set(d["atoms"])
    return {"exact": float(kind and eff and aset), "kind": float(kind), "effect": float(eff),
            "atoms_set": float(aset), "atom_bits": float(atoms_ok)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/behaviors/library_qwen35_9b.json")
    ap.add_argument("--encoder", default="lam1_s1")
    ap.add_argument("--aug", type=int, default=8)
    ap.add_argument("--out", default="results/encoder/decode_probe.json")
    a = ap.parse_args()
    data = load(a.data)
    stim, meta, names = data["stimuli"], data["meta"], list(data["behaviours"])
    outs = {n: data["behaviours"][n]["outputs"] for n in names}
    train, rand, comp = split_behaviors(meta, names)
    held = rand + comp
    table, _ = build()
    # actual descriptor: library behaviour whose intended outputs best match the target's outputs
    lib_out = {n: [intended(table, n, s) for s in stim] for n in table}
    actual = {}
    for b in names:
        ag = {n: np.mean([x == y for x, y in zip(lib_out[n], outs[b])]) for n in lib_out}
        best = max(ag, key=ag.get)
        actual[b] = b if ag[b] >= ag[best] - 0.01 else best  # ties go to the intended behaviour
    print("actual == intended structure:", np.mean([descriptor(meta[actual[b]]) == descriptor(meta[b]) for b in names]).round(3))
    targets = {"intended": {b: descriptor(meta[b]) for b in names}, "actual": {b: descriptor(meta[actual[b]]) for b in names}}

    free = torch.load("results/encoder/free_z.pt")
    model = SoftPromptLM(DEFAULT_CARRIER, n_soft_tokens=16)
    ck = torch.load(f"results/encoder/{a.encoder}.pt")
    feats = feature_table(model, data, "results/encoder/features_library_qwen35_9b.pt")
    enc = Encoder(ck["d_in"], model.embed_layer.weight.shape[1], z_norm=ck["z_norm"], d=ck["args"].get("d", 512),
                  layers=ck["args"].get("layers", 2)).to(model.device)
    enc.load_state_dict(ck["enc"]); enc.eval()
    pool = list(data["pool"])

    def encode(b, idx):
        with torch.no_grad():
            return enc(*support_tensors(feats, outs[b], [idx], model.device))[0].float().cpu()

    # rows: (behaviour, feature)
    fixed = sorted(pool)[:128]
    lab = lambda b: np.eye(3)[[LABELS.index(outs[b][i]) for i in fixed]].flatten()
    sysrows = {"free": {"train": [], "held": []}, "enc2": {"train": [], "held": []},
               "enc8": {"train": [], "held": []}, "labels": {"train": [], "held": []}}
    for b in names:
        grp = "train" if b in train else "held"
        for v in (0, 1):
            idx = free[(b, v)]["idx"]
            sysrows["free"][grp].append((b, flat(free[(b, v)]["z"])))
            sysrows["enc2"][grp].append((b, flat(encode(b, idx))))
        sysrows["labels"][grp] += [(b, lab(b))] * 2
        if grp == "train":
            views, _ = draw_views(pool, outs[b], 128, a.aug, random.Random(f"aug-{b}"))
            sysrows["enc8"]["train"] += [(b, flat(encode(b, ix))) for ix in views]
        else:
            sysrows["enc8"]["held"] += sysrows["enc2"]["held"][-2:]
    res, per = {}, {}
    for tname, tg in targets.items():
        for sname, rows in sysrows.items():
            Xtr = np.stack([x for _, x in rows["train"]]); dtr = [tg[b] for b, _ in rows["train"]]
            Xte = np.stack([x for _, x in rows["held"]]); bte = [b for b, _ in rows["held"]]
            preds = fit_predict(Xtr, dtr, Xte)
            sc = [score(p, tg[b]) for p, b in zip(preds, bte)]
            # per held-out behaviour (average of its views)
            per[(tname, sname)] = {b: {k: float(np.mean([s[k] for s, bb in zip(sc, bte) if bb == b])) for k in sc[0]} for b in held}
            res[f"{tname}/{sname}"] = {k: round(float(np.mean([per[(tname, sname)][b][k] for b in held])), 3) for k in sc[0]}
            print(tname, sname, res[f"{tname}/{sname}"], flush=True)
    # majority baseline for exact
    rng = np.random.default_rng(0)
    for tname in targets:
        for other in ("enc2", "enc8"):
            for k in ("exact", "effect", "atom_bits"):
                d = np.array([per[(tname, other)][b][k] - per[(tname, "free")][b][k] for b in held])
                boot = [rng.choice(d, len(d)).mean() for _ in range(2000)]
                res[f"{tname}/{other}-free/{k}"] = {"diff": round(float(d.mean()), 3), "ci": [round(float(np.percentile(boot, 2.5)), 3), round(float(np.percentile(boot, 97.5)), 3)]}
                print(tname, other, "-free", k, res[f"{tname}/{other}-free/{k}"])
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
