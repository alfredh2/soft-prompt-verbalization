"""Generate the finite discrete-domain dataset and write it to disk.

The families themselves live in :mod:`src.datasets.discrete`; this script only turns
them into a file.

Writes newline-delimited JSON, one ``(examples, function definition)`` record per
line, gzip-compressed when the output path ends in ``.gz``, together with a sidecar
``*.summary.json`` recording the arguments used and the resulting counts.

Regenerate the committed dataset with:

    python scripts/generate_dataset/discrete.py \
        --out data/discrete/discrete_functions.jsonl.gz

Output is fully determined by the arguments: every draw is keyed to a hash of the
function's identity, so the same arguments reproduce the same file byte for byte.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.datasets.discrete import DomainSpec, build_dataset, summarise  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the discrete-domain (examples, definition) dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/discrete/discrete_functions.jsonl.gz"),
        help="Output path; gzip-compressed when it ends in .gz.",
    )
    parser.add_argument(
        "--example-sizes",
        type=str,
        default="8,16,32",
        help="Comma-separated observed-pair counts; every function gets records at each size.",
    )
    parser.add_argument(
        "--n-holdout", type=int, default=16, help="Inputs reserved per function for evaluation."
    )
    parser.add_argument(
        "--views",
        type=int,
        default=4,
        help="Independent example sets drawn per function, per size.",
    )
    parser.add_argument(
        "--min-per-label",
        type=int,
        default=2,
        help="Labels guaranteed to appear in every example set.",
    )
    parser.add_argument(
        "--min-minority",
        type=int,
        default=4,
        help="Drop functions where some label occurs fewer times than this over the domain.",
    )
    parser.add_argument(
        "--max-majority-fraction",
        type=float,
        default=0.90,
        help="Drop functions a constant predictor reproduces more often than this; 0 disables.",
    )
    parser.add_argument(
        "--max-per-family",
        type=int,
        default=300,
        help="Per-family cap on functions; 0 means no cap.",
    )
    parser.add_argument("--int-low", type=int, default=-30, help="Lowest integer input.")
    parser.add_argument("--int-high", type=int, default=30, help="Highest integer input.")
    parser.add_argument("--n-bits", type=int, default=6, help="Width of the bit-string domain.")
    parser.add_argument(
        "--range-stride",
        type=int,
        default=1,
        help="Stride for the two-endpoint grids (range, piecewise).",
    )
    parser.add_argument(
        "--composition-stride",
        type=int,
        default=2,
        help="Stride for the threshold of the composed families.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Global seed offset.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        example_sizes = tuple(int(part) for part in args.example_sizes.split(",") if part.strip())
    except ValueError:
        raise SystemExit("--example-sizes must be a comma-separated list of integers")
    if not example_sizes:
        raise SystemExit("--example-sizes must name at least one size")

    spec = DomainSpec(
        int_low=args.int_low,
        int_high=args.int_high,
        n_bits=args.n_bits,
        range_stride=args.range_stride,
        composition_stride=args.composition_stride,
    )

    records = list(
        build_dataset(
            spec=spec,
            example_sizes=example_sizes,
            n_holdout=args.n_holdout,
            views=args.views,
            min_per_label=args.min_per_label,
            min_minority=args.min_minority,
            max_majority_fraction=args.max_majority_fraction or None,
            max_per_family=args.max_per_family or None,
            seed=args.seed,
        )
    )
    if not records:
        raise SystemExit("no records generated -- check the domain and filter arguments")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n" for record in records
    )
    if args.out.suffix == ".gz":
        # mtime=0 and an empty embedded filename keep the archive byte-identical
        # across runs, so the committed file only changes when the data does.
        with args.out.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as handle:
                handle.write(payload.encode("utf-8"))
    else:
        args.out.write_text(payload, encoding="utf-8")

    summary = summarise(records)
    summary["args"] = {key: str(value) for key, value in sorted(vars(args).items())}
    summary_path = args.out.with_suffix("").with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    size_mb = args.out.stat().st_size / 1e6
    print(f"wrote {summary['records']} records ({summary['functions']} functions) to {args.out}")
    print(f"  {size_mb:.1f} MB; summary -> {summary_path}")
    print(f"  by split:  {summary['by_split']}")
    print(f"  by family: {summary['by_family']}")
    print(f"  by size:   {summary['by_n_examples']}")
    print(f"  mean constant-predictor baseline: {summary['mean_majority_label_fraction']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
