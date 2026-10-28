"""Soft prompt vs Qwen3-4B prompting on the composition tier, paired by function.

Reads stored results only; no model is run. Domain accuracy is the fraction of the 301
domain inputs a method gets right. `floor` is the constant-label guess (always the
function's majority output), which a method must clear before it has learned anything.
Paired tests use the 40 functions as the unit: a sign test (robust, ignores magnitude) and
a bootstrap CI on the mean difference (uses magnitude). They can disagree.

  python scripts/compare_composition_gate.py
"""

from __future__ import annotations

import json
import math
import random
from statistics import mean

R = "results/learning_curve/"
GATE = "results/zbank_gate/comp_teaching_q3_4b/rows.json"
ARMS = {"direct": R + "comp_teaching_sorted_qwen3_4b",
        "direct+gram": R + "comp_teaching_grammar_qwen3_4b",
        "think+gram": R + "comp_teaching_think_grammar_qwen3_4b"}
THINK26 = R + "comp_teaching_think26_grammar_qwen3_4b"   # N=32 only, ~26 s/task


def load(path):
    return json.load(open(path))


def sign_p(w, l):
    m, k = w + l, min(w, l)
    return min(1.0, 2 * sum(math.comb(m, j) for j in range(k + 1)) / 2 ** m) if m else 1.0


def boot(d, B=5000):
    rnd = random.Random(0)
    s = sorted(mean(rnd.choice(d) for _ in range(len(d))) for _ in range(B))
    return s[int(.025 * B)], s[int(.975 * B)]


def by_size(rows, n):
    return {r["function_id"]: r for r in rows if int(r["size"]) == n}


def paired(a, b, ids):
    d = [a[f] - b[f] for f in ids]
    lo, hi = boot(d)
    w, l = sum(x > 0 for x in d), sum(x < 0 for x in d)
    return f"{mean(d):+.3f} [{lo:+.3f},{hi:+.3f}] {w}/{l} p={sign_p(w, l):.3f}"


def main():
    gate = load(GATE)
    soft = {(r["function_id"], r["n_examples"]): r["domain_acc"] for r in gate if r["seed_idx"] == 0}
    floor = {r["function_id"]: r["majority_label_fraction"] for r in gate}
    arms = {k: load(p + "/rows.json") for k, p in ARMS.items()}
    print(f"constant-label floor over the 40 functions: {mean(floor.values()):.3f}\n")
    print(f"{'N':>3}{'floor':>7}{'direct':>8}{'think':>7}{'soft':>7}   (domain accuracy)")
    for n in (8, 12, 16, 32):
        ids = sorted(floor)
        t = by_size(arms["think+gram"], n)
        print(f"{n:>3}{mean(floor.values()):>7.3f}"
              f"{mean(r['domain_acc'] for r in by_size(arms['direct+gram'], n).values()):>8.3f}"
              f"{mean(r['domain_acc'] for r in t.values()):>7.3f}"
              f"{mean(soft[(f, n)] for f in ids):>7.3f}")
    print("\npaired differences  mean [95% bootstrap CI]  wins/losses  sign-test p")
    for n in (8, 12, 16, 32):
        ids = sorted(floor)
        s = {f: soft[(f, n)] for f in ids}
        d = {f: r["domain_acc"] for f, r in by_size(arms["direct+gram"], n).items()}
        t = {f: r["domain_acc"] for f, r in by_size(arms["think+gram"], n).items()}
        print(f"N={n:>2}  soft - floor        {paired(s, floor, ids)}")
        print(f"      soft - direct+gram   {paired(s, d, ids)}")
        print(f"      soft - think+gram    {paired(s, t, ids)}")
    try:
        t26 = by_size(load(THINK26 + "/rows.json"), 32)
        c = load(THINK26 + "/compute.json")[0]
        ids = sorted(floor)
        s = {f: soft[(f, 32)] for f in ids}
        print(f"\nN=32, thinking at {c['sec_per_task']} s/task (soft prompt ~26 s/task): "
              f"think {mean(r['domain_acc'] for r in t26.values()):.3f}, "
              f"soft - think {paired(s, {f: r['domain_acc'] for f, r in t26.items()}, ids)}")
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    main()
