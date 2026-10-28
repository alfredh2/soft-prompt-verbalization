"""Reference lines for exact recovery: what the examples determine, for the curve's functions.

    python scripts/identifiability.py                                          # single-rule tier
    python scripts/identifiability.py --tier compositional                      # random support
    python scripts/identifiability.py --tier compositional --support teaching --identifiable-only

Uses the same functions and the same nested example pools as the learning curves, so the
reference sits on exactly the points the methods are scored on. Candidates are distinct
full-domain BEHAVIOURS, never syntactic forms.

Three lines per size, from weakest to sharpest assumptions about what a learner knows:

  identifiable    the examples leave exactly one behaviour standing
  guess 1/k       pick uniformly among the k that fit -- a simple yardstick, NOT a ceiling:
                  a model that prefers simpler rules can beat it with no information the
                  examples lack
  bayes           probability the most likely consistent behaviour is the target, under the
                  generator's own distribution over targets. The best exact-recovery rate
                  any method can reach from the examples plus that prior.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.compositional import (  # noqa: E402
    bayes_exact, behaviour_masks, bit_matrix, eligible_prior, input_positions,
)
from src.datasets.discrete import DomainSpec, load_dataset  # noqa: E402
from src.identifiability import hypothesis_tables, mask_consistent_count, n_consistent  # noqa: E402
from src.scoring import select_functions, subset  # noqa: E402

TIERS = {
    "discrete": ("data/discrete/discrete_functions.jsonl.gz", "2,4,8,16,24,32,40,45"),
    "compositional": ("data/compositional/compositional_functions.jsonl.gz",
                      "4,8,16,32,64,96,128,192"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="discrete", choices=list(TIERS))
    ap.add_argument("--dataset")
    ap.add_argument("--records", type=int, default=20)
    ap.add_argument("--n", type=int,
                    help="Subset size, passed through subset() exactly as learning_curve.py's "
                         "--n is. The two scripts must select identically or the reference "
                         "describes different functions than the curve it is drawn against "
                         "(they once shared 7 of 40). Default 220 for discrete, 100000 "
                         "(everything) for compositional.")
    ap.add_argument("--sizes")
    ap.add_argument("--support", default="random", choices=["random", "teaching"])
    ap.add_argument("--identifiable-only", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()
    dataset = args.dataset or TIERS[args.tier][0]
    sizes = [int(s) for s in (args.sizes or TIERS[args.tier][1]).split(",")]
    suffix = "" if args.support == "random" else f"_{args.support}"
    out = Path(args.out or f"results/identifiability/{args.tier}{suffix}.json")

    records = subset(load_dataset(dataset), args.n or (220 if args.tier == "discrete" else 100000), 0)
    if args.tier == "discrete":
        spec = DomainSpec()
        tables, positions = hypothesis_tables(spec)
        print({d: len(t) for d, t in tables.items()}, "distinct behaviours in the class")

        def measure(b, n):
            return {"k": n_consistent(tables[b["domain"]], positions[b["domain"]], b["pool"][:n])}
    else:
        spec = None
        ds = records[0]["domain_spec"]
        masks, pos = behaviour_masks(ds), input_positions(ds)
        el_masks, prior = eligible_prior(ds)
        matrix = bit_matrix(el_masks, len(pos))
        print(f"{len(masks):,} distinct behaviours; {len(el_masks):,} eligible as targets")

        def measure(b, n):
            shown = b["pool"][:n]
            bayes, k_el = bayes_exact(matrix, prior, pos, shown)
            return {"k": mask_consistent_count(masks, pos, shown),
                    "k_eligible": k_el, "bayes": bayes}

    built = select_functions(records, args.records, spec, args.support, args.identifiable_only)
    if args.support == "teaching":
        ids = [b["identified_at"] for b in built]
        print(f"{len(built)} functions; identified by the teaching prefix at N = "
              f"min {min(ids)}, median {statistics.median(ids):.0f}, max {max(ids)}")
    per_function = {b["function_id"]: {n: measure(b, n) for n in sizes if n <= len(b["pool"])}
                    for b in built}

    by_size, cols = {}, ["identifiable", "guess 1/k"] + (["ident(elig)", "bayes"] if args.tier != "discrete" else [])
    print("\n" + f"{'N':>4}{'n':>4}" + "".join(f"{c:>13}" for c in cols) + f"{'median k':>10}")
    for n in sizes:
        ms = [v[n] for v in per_function.values() if n in v]
        if not ms:
            continue
        ks = [m["k"] for m in ms]
        row = {"n": len(ms), "identifiable": sum(k == 1 for k in ks) / len(ks),
               "guess_1_over_k": sum(1 / k for k in ks) / len(ks),
               "median_consistent": statistics.median(ks)}
        if "bayes" in ms[0]:
            row["identifiable_eligible"] = sum(m["k_eligible"] == 1 for m in ms) / len(ms)
            row["bayes_exact"] = sum(m["bayes"] for m in ms) / len(ms)
        by_size[n] = {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}
        r = by_size[n]
        vals = [r["identifiable"], r["guess_1_over_k"]] + (
            [r["identifiable_eligible"], r["bayes_exact"]] if "bayes_exact" in r else [])
        print(f"{n:>4}{r['n']:>4}" + "".join(f"{v:>13.2f}" for v in vals)
              + f"{r['median_consistent']:>10.0f}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"tier": args.tier, "support": args.support,
                               "functions": list(per_function), "by_size": by_size,
                               "per_function": per_function}, indent=2) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
