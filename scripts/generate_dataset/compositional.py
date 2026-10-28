"""Generate the depth-2 composition dataset (wide integer domain).

    python scripts/generate_dataset/compositional.py \
        --out data/compositional/compositional_functions.jsonl.gz

Same record schema as the single-rule tier, plus the fields a record needs to rebuild its
own function (`dataset`, `domain_spec`, `n_holdout`) and a `pair_type` for analysis, so
the same executor, scoring and curve scripts run on either. Output is deterministic and
the archive is byte-reproducible.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.datasets.compositional import (  # noqa: E402
    build_dataset, domain_spec, hypothesis_class,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path,
                    default=Path("data/compositional/compositional_functions.jsonl.gz"))
    ap.add_argument("--per-stratum", type=int, default=20,
                    help="Functions per (op, atom-kind pair) stratum; 9 strata.")
    ap.add_argument("--sizes", default="8,16,32,64,128", help="Observed-pair counts.")
    ap.add_argument("--views", type=int, default=2)
    ap.add_argument("--n-holdout", type=int, default=48)
    ap.add_argument("--int-low", type=int, default=-150)
    ap.add_argument("--int-high", type=int, default=150)
    ap.add_argument("--mod-max", type=int, default=9)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ds = domain_spec(args.int_low, args.int_high, args.mod_max)
    atoms, _, behaviours = hypothesis_class(ds)
    print(f"{len(atoms)} atoms -> {len(behaviours):,} distinct behaviours at depth 2 "
          f"over {args.int_high - args.int_low + 1} inputs")

    sizes = tuple(int(s) for s in args.sizes.split(","))
    records = list(build_dataset(ds, per_stratum=args.per_stratum, example_sizes=sizes,
                                 views=args.views, n_holdout=args.n_holdout, seed=args.seed))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(r, separators=(",", ":"), sort_keys=True) + "\n"
                      for r in records)
    with args.out.open("wb") as raw:  # mtime=0: the archive changes only when the data does
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            gz.write(payload.encode("utf-8"))

    fns = {r["function_id"]: r for r in records}
    summary = {
        "records": len(records), "functions": len(fns),
        "hypothesis_class_behaviours": len(behaviours),
        "by_family": dict(sorted(Counter(r["family"] for r in fns.values()).items())),
        "by_pair_type": dict(sorted(Counter(r["pair_type"] for r in fns.values()).items())),
        "by_split": dict(sorted(Counter(r["split"] for r in fns.values()).items())),
        "by_n_examples": dict(sorted(Counter(r["n_examples"] for r in records).items())),
        "mean_majority_label_fraction": round(
            sum(r["majority_label_fraction"] for r in fns.values()) / len(fns), 4),
        "domain_spec": ds, "n_holdout": args.n_holdout,
        "args": {k: str(v) for k, v in sorted(vars(args).items())},
    }
    args.out.with_suffix("").with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(f"wrote {len(records)} records ({len(fns)} functions) to {args.out} "
          f"({args.out.stat().st_size / 1e6:.1f} MB)")
    for k in ("by_family", "by_pair_type", "by_split", "by_n_examples"):
        print(f"  {k}: {summary[k]}")
    print(f"  mean majority-label fraction: {summary['mean_majority_label_fraction']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
