"""Frontier description vs soft prompt on the behaviour benchmark, paired by behaviour.

Both arms are scored by agreement with the TARGET's outputs on the same 64 held-out
sentences per behaviour. Reads stored results only.

  python scripts/compare_behaviors.py
"""

from __future__ import annotations

import json
import math
import random
from statistics import mean

SCORES = "results/behaviors/scores_gpt61sol.json"
SOFT = "results/behaviors/soft_prompt_qwen3_4b.json"
DATA = "data/behaviors/pilot_qwen35_9b.json"
SIZES = (8, 16, 32, 64)


def sign_p(w, l):
    m, k = w + l, min(w, l)
    return min(1.0, 2 * sum(math.comb(m, j) for j in range(k + 1)) / 2 ** m) if m else 1.0


def boot(d, B=5000):
    rnd = random.Random(0)
    s = sorted(mean(rnd.choice(d) for _ in range(len(d))) for _ in range(B))
    return s[int(.025 * B)], s[int(.975 * B)]


def main():
    sc = json.load(open(SCORES))["scores"]
    soft = json.load(open(SOFT))
    fid = {n: b["fidelity"] for n, b in json.load(open(DATA))["behaviours"].items()}
    names = list(sc)
    z = {(r["behaviour"], r["n"]): r["test_agree"] for r in soft["rows"]}
    print(f"{'N':>4}{'no info':>9}{'instruction':>13}{'description':>13}{'soft prompt':>13}   soft - description (95% CI)   W/L/T   sign p")
    for n in SIZES:
        d = [z[(b, n)] - sc[b][f"propose_{n}"] for b in names]
        w, l = sum(x > 0.02 for x in d), sum(x < -0.02 for x in d)
        lo, hi = boot(d)
        print(f"{n:>4}{mean(soft['bare'][b] for b in names):>9.3f}{mean(sc[b]['oracle'] for b in names):>13.3f}"
              f"{mean(sc[b][f'propose_{n}'] for b in names):>13.3f}{mean(z[(b, n)] for b in names):>13.3f}"
              f"   {mean(d):>+7.3f} [{lo:+.3f},{hi:+.3f}]   {w}/{l}/{13 - w - l}   {sign_p(w, l):.3f}")
    print("\n(W/L/T: behaviours where the soft prompt is better / worse / within 0.02 of the description)")
    print("\nby behaviour at N=32 (description | soft prompt | instruction fidelity):")
    for b in names:
        print(f"  {b:<36}{sc[b]['propose_32']:>7.3f}{z[(b, 32)]:>8.3f}   fidelity {fid[b]:.2f}")
    low = [b for b in names if fid[b] < 0.8]
    print(f"\nbehaviours where the target does NOT follow its instruction (fidelity < 0.8, n={len(low)}):")
    for n in SIZES:
        print(f"  N={n:>2}: description {mean(sc[b][f'propose_{n}'] for b in low):.3f} | soft prompt "
              f"{mean(z[(b, n)] for b in low):.3f} | instruction {mean(sc[b]['oracle'] for b in low):.3f}")


if __name__ == "__main__":
    main()
