"""Independent audit of the teaching-support construction.

The teaching supports are built by one code path (a greedy teacher over a bit matrix of
behaviours). Checking them with the same path would agree with itself, so this rebuilds
the hypothesis class from scratch -- atoms from raw numpy comparisons, behaviours by
boolean ops, deduplicated by their packed bytes -- and recounts everything against that.

It checks, for EVERY function and several N:
  * the independently built class has the size the module claims (188,011 / 138,031)
  * each record's printed `expression` agrees with the function's executable rule
  * how many distinct full-domain behaviours fit the support, and the identifiable share
  * `identified_at` is exactly the first N at which one behaviour remains, and where a
    function is marked unidentifiable, the survivors differ from the truth ONLY on held-out
    inputs
  * no leakage: support inputs never include a held-out input, support labels are the
    truth, no duplicates, and the greedy order is unchanged when every held-out label is
    flipped (so selection cannot have read them)
and reports whether the picks actually sit on the boundaries / residues that matter.

    python scripts/audit_teaching_support.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets import compositional as C  # noqa: E402
from src.datasets.discrete import load_dataset  # noqa: E402
from src.scoring import pick_functions, pool_for  # noqa: E402

SIZES = [2, 4, 6, 8, 12, 16, 32, 64]


def independent_class(low: int, high: int, mod_max: int):
    """Atoms, every distinct behaviour, and the eligible mask -- built without the module."""
    x = np.arange(low, high + 1)
    n = len(x)
    rows = [x > c for c in range(low + 1, high)]
    rows += [x < c for c in range(low + 1, high)]
    rows += [np.abs(x) > c for c in range(0, high)]
    for k in range(2, mod_max + 1):
        rows += [x % k == r for r in range(k)]
    atoms = np.array(rows)
    seen: set[bytes] = set()
    for a in atoms:
        for arr in (a & atoms, a | atoms, a & ~atoms):
            seen.update(r.tobytes() for r in np.packbits(arr, axis=1))
    keys = list(seen)
    beh = np.unpackbits(np.frombuffer(b"".join(keys), dtype=np.uint8).reshape(len(keys), -1),
                        axis=1)[:, :n].astype(bool)
    atom_set = {r.tobytes() for r in np.packbits(atoms, axis=1)}
    single = np.fromiter((k in atom_set for k in keys), dtype=bool, count=len(keys))
    ones = beh.sum(1)
    elig = ~single & (np.minimum(ones, n - ones) >= 30) & (np.maximum(ones, n - ones) / n <= 0.85)
    return atoms, beh, elig


def annotate(atom, x: int) -> str:
    """Where a pick sits relative to one atom: offset from a threshold, or residue."""
    if atom[0] in ("gt", "lt"):
        return f"x{x - atom[1]:+d}"
    if atom[0] == "abs":
        return f"|x|{abs(x) - atom[1]:+d}"
    return f"x%{atom[1]}={x % atom[1]}{'*' if x % atom[1] == atom[2] else ''}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/compositional/compositional_functions.jsonl.gz")
    ap.add_argument("--used", default="results/learning_curve/comp_teaching_gpt61sol/rows.json")
    ap.add_argument("--out", default="results/identifiability/teaching_audit.json")
    args = ap.parse_args()

    recs = load_dataset(args.dataset)
    ds = recs[0]["domain_spec"]
    low, high = ds["int_low"], ds["int_high"]
    x = np.arange(low, high + 1)
    n = len(x)

    print("building the hypothesis class independently ...", flush=True)
    atoms, beh, elig = independent_class(low, high, ds["mod_max"])
    atoms_mod, _, beh_mod = C.hypothesis_class(ds)
    el_mod = sum(len(v) for v in C.eligible_strata(ds).values())
    checks = {
        "atoms": (len(atoms), len(atoms_mod)),
        "distinct_behaviours": (len(beh), len(beh_mod)),
        "eligible_targets": (int(elig.sum()), el_mod),
    }
    for k, (mine, theirs) in checks.items():
        print(f"  {k:<22} independent {mine:>8,}   module {theirs:>8,}   "
              f"{'MATCH' if mine == theirs else 'MISMATCH'}")

    used = set()
    if Path(args.used).exists():
        used = {r["function_id"] for r in json.load(open(args.used))}

    matrix = C._class_matrix(low, high, ds["mod_max"])  # the module's own, for the flip test
    fails = defaultdict(list)
    info = {}
    funcs = pick_functions(recs, 10**6)
    print(f"auditing {len(funcs)} functions ...", flush=True)
    for rec in funcs:
        fid = rec["function_id"]
        teach = pool_for(rec, None, support="teaching")
        rand = pool_for(rec, None, support="random")
        fn = teach["fn"]
        truth = np.array([fn.compute(int(v)) == "1" for v in x])

        # the printed expression must be the function that is executed
        expr = np.array([bool(eval(rec["expression"], {"abs": abs}, {"x": int(v)})) for v in x])
        if not (expr == truth).all():
            fails["expression_vs_compute"].append(fid)

        pos = lambda v: int(v) - low  # noqa: E731
        pool_cols = np.array([pos(v) for v, _ in teach["pool"]])
        held = {pos(v) for v, _ in teach["held_out"]}
        if set(pool_cols) & held or len(set(pool_cols)) != len(pool_cols) \
                or len(pool_cols) + len(held) != n:
            fails["pool_overlaps_held_out"].append(fid)
        if any((lab == "1") != bool(truth[pos(v)]) for v, lab in teach["pool"]):
            fails["support_label_not_truth"].append(fid)

        # count distinct behaviours consistent with every prefix, narrowing as we go
        idx = np.arange(len(beh))
        k_full, k_el = [len(idx)], [int(elig.sum())]
        for col in pool_cols:
            idx = idx[beh[idx, col] == truth[col]]
            k_full.append(len(idx))
            k_el.append(int(elig[idx].sum()))
        first_one = next((i for i, k in enumerate(k_full) if k == 1), None)
        ia = teach["identified_at"]
        if ia != first_one:
            fails["identified_at_mismatch"].append((fid, ia, first_one))
        if ia is None:
            diff = np.flatnonzero((beh[idx] != truth).any(0))
            if not set(diff.tolist()) <= held:
                fails["survivors_differ_off_heldout"].append(fid)

        # leakage into selection: flip every held-out label, the greedy order must not move
        cand = [pos(v) for v, _ in rand["pool"]]
        flipped = truth.copy()
        flipped[list(held)] ^= True
        p0, _ = C.teaching_order(matrix, truth, cand)
        p1, _ = C.teaching_order(matrix, flipped, cand)
        taught = [int(c) for c in pool_cols[: len(p0)]]
        if p0 != p1:
            fails["selection_reads_held_out_labels"].append(fid)
        if p0 != taught:
            fails["pool_prefix_is_not_the_greedy_order"].append(fid)

        info[fid] = {"rec": rec, "k_full": k_full, "k_el": k_el, "ia": ia, "truth": truth,
                     "cols": pool_cols, "held": held, "survivors": idx, "teach": teach}

    print("\n=== integrity checks (an empty list means it passed) ===")
    for name in ("expression_vs_compute", "pool_overlaps_held_out", "support_label_not_truth",
                 "identified_at_mismatch", "survivors_differ_off_heldout",
                 "selection_reads_held_out_labels", "pool_prefix_is_not_the_greedy_order"):
        print(f"  {name:<40} {'PASS' if not fails[name] else 'FAIL ' + str(fails[name][:3])}")

    # ---------------- identifiable fraction and remaining hypothesis counts ----------------
    groups = {"all functions": list(info),
              "identifiable from the pool": [f for f in info if info[f]["ia"] is not None],
              "the 40 used in calibration": [f for f in info if f in used]}
    audit = {"checks": {k: list(v) for k, v in checks.items()}, "groups": {}}
    print("\n=== distinct full-domain behaviours consistent with the support ===")
    for gname, ids in groups.items():
        if not ids:
            continue
        print(f"\n{gname} (n={len(ids)})")
        print(f"{'N':>4}{'identifiable':>14}{'k min':>8}{'k median':>10}{'k max':>10}"
              f"{'k<=2':>8}   {'| random: ident':>16}{'k median':>10}")
        audit["groups"][gname] = {}
        for N in SIZES:
            ks = [info[f]["k_full"][N] for f in ids]
            rk = []
            for f in ids:  # contrast: the same functions under random support
                rec = info[f]["rec"]
                r = pool_for(rec, None, support="random")
                idx = np.arange(len(beh))
                for v, _ in r["pool"][:N]:
                    idx = idx[beh[idx, pos(v)] == info[f]["truth"][pos(v)]]
                rk.append(len(idx))
            row = {"identifiable": sum(k == 1 for k in ks) / len(ks), "k_min": min(ks),
                   "k_median": statistics.median(ks), "k_max": max(ks),
                   "k_le_2": sum(k <= 2 for k in ks) / len(ks),
                   "random_identifiable": sum(k == 1 for k in rk) / len(rk),
                   "random_k_median": statistics.median(rk)}
            audit["groups"][gname][N] = row
            print(f"{N:>4}{row['identifiable']:>14.2f}{row['k_min']:>8}{row['k_median']:>10.0f}"
                  f"{row['k_max']:>10}{row['k_le_2']:>8.2f}   {row['random_identifiable']:>16.2f}"
                  f"{row['random_k_median']:>10.0f}")

    # ---------------- do the picks target the boundaries and residues? ----------------
    print("\n=== are the picks on the boundaries / residues that matter? ===")
    print("(at N = identified_at; 'near' = pick is domain-adjacent to a label change of the true rule)")
    cells = defaultdict(list)
    for f in groups["identifiable from the pool"]:
        i = info[f]; rec = i["rec"]; t = i["truth"]; m = i["ia"]
        picks = [int(c) for c in i["cols"][:m]]
        trans = [j for j in range(1, n) if t[j] != t[j - 1]]
        adj = set(trans) | {j - 1 for j in trans}
        supp = set(picks)
        pinned = sum(1 for j in trans if (j - 1) in supp and j in supp)
        mods = [a for a in (rec["params"]["a"], rec["params"]["b"]) if a[0] == "mod"]
        cov = (sum(len({(c + low) % a[1] for c in picks}) for a in mods) /
               sum(a[1] for a in mods)) if mods else None
        cells[(rec["family"], rec["pair_type"])].append({
            "m": m, "transitions": len(trans), "pinned": pinned,
            "near": sum(c in adj for c in picks) / len(picks),
            "domain_near": len(adj) / n, "residue_cov": cov})
    print(f"{'rule type':<18}{'n':>4}{'ident. at N':>13}{'transitions':>13}{'pinned':>9}"
          f"{'picks near':>12}{'(chance)':>10}{'residues':>10}")
    audit["boundary_targeting"] = {}
    for key in sorted(cells):
        rs = cells[key]
        mean = lambda k: statistics.mean(r[k] for r in rs)  # noqa: E731
        cv = [r["residue_cov"] for r in rs if r["residue_cov"] is not None]
        tp = sum(r["pinned"] for r in rs) / max(1, sum(r["transitions"] for r in rs))
        print(f"{key[0] + '/' + key[1]:<18}{len(rs):>4}{mean('m'):>13.1f}{mean('transitions'):>13.1f}"
              f"{tp:>9.2f}{mean('near'):>12.2f}{mean('domain_near'):>10.2f}"
              f"{(f'{statistics.mean(cv):.2f}' if cv else '   -'):>10}")
        audit["boundary_targeting"]["/".join(key)] = {
            "n": len(rs), "mean_identified_at": mean("m"), "transitions_pinned": tp,
            "picks_near_transition": mean("near"), "chance": mean("domain_near"),
            "residue_coverage": statistics.mean(cv) if cv else None}

    # ---------------- look at one of each ----------------
    print("\n=== one worked example per rule type (median identification size) ===")
    for key in sorted(cells):
        ids = sorted((f for f in groups["identifiable from the pool"]
                      if (info[f]["rec"]["family"], info[f]["rec"]["pair_type"]) == key),
                     key=lambda f: (info[f]["ia"], f))
        f = ids[len(ids) // 2]
        i = info[f]; rec = i["rec"]; t = i["truth"]; m = i["ia"]
        a, b = tuple(rec["params"]["a"]), tuple(rec["params"]["b"])
        trans = [(low + j - 1, low + j) for j in range(1, n) if t[j] != t[j - 1]]
        print(f"\n[{key[0]} / {key[1]}] {f}   identified at N={m}")
        print(f"  {rec['description']}")
        print(f"  label changes between: {trans[:6]}{' ...' if len(trans) > 6 else ''}"
              f"  ({len(trans)} total)")
        for j, c in enumerate(i["cols"][:m], start=1):
            v = int(c) + low
            print(f"  pick {j}: x={v:>5} -> {int(t[c])}   a:{annotate(a, v):<9} b:{annotate(b, v):<9}"
                  f"  behaviours left {i['k_full'][j - 1]:>7,} -> {i['k_full'][j]:>7,}")
        held = sorted(int(h) + low for h in i["held"])
        print(f"  held-out ({len(held)}): {held[:10]} ... | any held-out among the picks: "
              f"{bool({int(c) for c in i['cols'][:m]} & i['held'])}")

    # one unidentifiable function, to show what stays open
    bad = [f for f in info if info[f]["ia"] is None]
    if bad:
        f = bad[0]; i = info[f]; rec = i["rec"]; t = i["truth"]
        diff = sorted(int(d) + low for d in np.flatnonzero((beh[i["survivors"]] != t).any(0)))
        print(f"\n[unidentifiable] {f}: after the WHOLE non-held-out pool, "
              f"{i['k_full'][-1]} behaviours remain; they disagree with the truth only at "
              f"{diff}, all held out: {set(diff) <= {h + low for h in i['held']}}")
        print(f"  {rec['description']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({**audit, "failures": {k: [str(v) for v in vs]
                                                      for k, vs in fails.items()}}, indent=2,
                              default=str) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
