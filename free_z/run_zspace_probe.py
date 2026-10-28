"""Probe whether independently-optimized soft prompts z_j live in a shared,
semantically-organized space -- BEFORE spending compute on a text decoder.

Question: does z encode not just "this function's behavior" but do similar
functions (gt_5 vs gt_8) end up near each other in z-space in a way that's
linearly decodable? If yes, a decoder is a reasonable next step. If no
(despite every z having high behavioral fidelity), that's a more fundamental
finding: independently optimized soft prompts are behaviorally meaningful but
not geometrically aligned by semantics.

Steps:
  1. Train every function's z with a quality gate (retry with more epochs
     until behavioral accuracy >= MIN_ACC, or give up).
  2. All z's share the same initialization z0 (not independent random inits),
     so any alignment we find isn't an artifact of divergent starting points.
  3. Multi-seed identifiability check: train one function from several
     different random inits and compare same-function distance to
     different-function distance.
  4. Probe: nearest-centroid family classifier + per-family ridge regressor
     for the numeric parameter c, evaluated on held-out INTERPOLATED values.

Usage:
  python run_zspace_probe.py
"""

from __future__ import annotations

import torch

from verbalize.evaluate import accuracy
from verbalize.functions import generate_dataset, make_dense_families
from verbalize.model import SoftPromptLM
from verbalize.probe import pooled, ridge_fit_predict
from verbalize.train import train_soft_prompt

MIN_ACC = 0.9
EPOCH_SCHEDULE = (40, 80, 150)  # retry with more epochs until quality gate passes
N_TRAIN_EX = 30
N_TEST_EX = 15


def train_z_with_quality_gate(model, spec, shared_z0, seed):
    train_ex, test_ex = generate_dataset(spec, N_TRAIN_EX, N_TEST_EX, seed=seed)
    acc = 0.0
    for epochs in EPOCH_SCHEDULE:
        model.set_soft_prompt(shared_z0)
        train_soft_prompt(model, spec, train_ex, epochs=epochs, lr=0.3, log_every=100_000)
        acc = accuracy(model, spec, test_ex, "soft")
        if acc >= MIN_ACC:
            break
    z = model.soft_prompt.detach().clone()
    return z, acc


def main():
    model = SoftPromptLM(model_name="Qwen/Qwen2.5-1.5B-Instruct", n_soft_tokens=16)
    train_specs, test_specs = make_dense_families(seed=0)
    print(f"{len(train_specs)} train functions, {len(test_specs)} held-out (interpolated) functions")

    model.reset_soft_prompt()
    shared_z0 = model.soft_prompt.detach().clone()
    print(f"shared z0 fixed, norm={shared_z0.norm():.2f}\n")

    print("=== training all z's (quality-gated, shared init) ===")
    train_z, train_acc, train_meta = [], [], []
    for spec in train_specs:
        z, acc = train_z_with_quality_gate(model, spec, shared_z0, seed=0)
        flag = "OK" if acc >= MIN_ACC else "BELOW THRESHOLD"
        print(f"  {spec.name:10s} family={spec.family:6s} c={spec.param:6.1f}  acc={acc:.2f}  {flag}")
        train_z.append(z)
        train_acc.append(acc)
        train_meta.append(spec)

    test_z, test_acc, test_meta = [], [], []
    for spec in test_specs:
        z, acc = train_z_with_quality_gate(model, spec, shared_z0, seed=1)
        flag = "OK" if acc >= MIN_ACC else "BELOW THRESHOLD"
        print(f"  {spec.name:10s} family={spec.family:6s} c={spec.param:6.1f}  acc={acc:.2f}  {flag}")
        test_z.append(z)
        test_acc.append(acc)
        test_meta.append(spec)

    n_below = sum(1 for a in train_acc + test_acc if a < MIN_ACC)
    print(f"\n{n_below}/{len(train_acc) + len(test_acc)} z's never reached acc >= {MIN_ACC} "
          f"even after {EPOCH_SCHEDULE[-1]} epochs (kept anyway, flagged).")

    # --- multi-seed identifiability diagnostic ---
    print("\n=== identifiability: same-function (different random inits) vs different-function distance ===")
    spec_a = train_meta[0]
    idx_b = next(
        i for i, s in enumerate(train_meta)
        if s.family == spec_a.family and s.name != spec_a.name
    )
    spec_b = train_meta[idx_b]
    seeds_a = []
    for k in range(3):
        model.reset_soft_prompt()  # independent random init this time, not shared z0
        train_ex, test_ex = generate_dataset(spec_a, N_TRAIN_EX, N_TEST_EX, seed=100 + k)
        train_soft_prompt(model, spec_a, train_ex, epochs=80, lr=0.3, log_every=100_000)
        acc = accuracy(model, spec_a, test_ex, "soft")
        seeds_a.append(pooled(model.soft_prompt.detach().clone()))
        print(f"  {spec_a.name} seed {k}: acc={acc:.2f}")
    z_b = pooled(train_z[idx_b])

    same_fn_dists = [
        (seeds_a[i] - seeds_a[j]).norm().item()
        for i in range(3) for j in range(i + 1, 3)
    ]
    diff_fn_dists = [(s - z_b).norm().item() for s in seeds_a]
    print(f"  same-function ({spec_a.name}, different seeds) distances: "
          f"{[round(d, 2) for d in same_fn_dists]}")
    print(f"  different-function ({spec_a.name} vs {spec_b.name}) distances: "
          f"{[round(d, 2) for d in diff_fn_dists]}")
    if min(diff_fn_dists) < max(same_fn_dists):
        print("  -> different-function pair is CLOSER than some same-function pair: "
              "z-space is not reliably organized by identity, let alone semantics.")
    else:
        print("  -> same-function seeds cluster tighter than the different-function pair: "
              "at least function-identity is somewhat stable across seeds.")

    # --- probe: family classification (nearest centroid) ---
    print("\n=== probe 1: family classification (nearest centroid, on pooled z) ===")
    families = sorted(set(s.family for s in train_meta))
    centroids = {}
    for fam in families:
        vecs = [pooled(z) for z, s in zip(train_z, train_meta) if s.family == fam]
        centroids[fam] = torch.stack(vecs).mean(dim=0)

    correct = 0
    for z, spec in zip(test_z, test_meta):
        pz = pooled(z)
        pred = min(families, key=lambda f: (pz - centroids[f]).norm().item())
        hit = pred == spec.family
        correct += hit
        print(f"  {spec.name:10s} true={spec.family:6s} pred={pred:6s} {'OK' if hit else 'WRONG'}")
    print(f"  family classification accuracy: {correct}/{len(test_meta)} = {correct/len(test_meta):.2f}")

    # --- probe 2: per-family parameter regression (ridge, closed-form) ---
    print("\n=== probe 2: per-family parameter regression (ridge) on held-out (interpolated) c ===")
    for fam in families:
        Xtr = [pooled(z) for z, s in zip(train_z, train_meta) if s.family == fam]
        ytr = [s.param for s in train_meta if s.family == fam]
        Xte = [pooled(z) for z, s in zip(test_z, test_meta) if s.family == fam]
        yte = [s.param for s in test_meta if s.family == fam]
        preds = ridge_fit_predict(Xtr, ytr, Xte, alpha=10.0)
        mae = sum(abs(p - y) for p, y in zip(preds, yte)) / len(yte)
        param_range = max(ytr + yte) - min(ytr + yte)
        print(f"  {fam:6s} true={[round(y,1) for y in yte]}  pred={[round(p,1) for p in preds]}  "
              f"MAE={mae:.2f}  (param range={param_range:.0f})")


if __name__ == "__main__":
    main()
