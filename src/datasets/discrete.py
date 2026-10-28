"""Target functions on finite discrete domains, and the dataset built from them.

Each target function is a deterministic rule over a small, fully enumerable input
domain -- integers in a bounded range, or fixed-width bit strings -- whose output is
one of a handful of label strings. Because the domain is finite we can enumerate it
exactly, which makes the label distribution, the held-out split and the
majority-label baseline all exact rather than estimated.

The dataset produced by :func:`build_dataset` consists of records of the form
``(examples, function definition)``: a list of ``(x, y)`` pairs drawn from one
function, paired with that function's structured identity (family and parameters),
its canonical English description and paraphrases of it. It is the supervision
needed to learn, or to evaluate, a map from observed behaviour to a description.

The continuous-domain counterpart lives in :mod:`src.datasets.continuous`.
"""

from __future__ import annotations

import gzip
import json
import random
import zlib
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

# Default domain parameters. Integers span an inclusive symmetric range; bit strings
# are rendered most-significant-bit first, so index 0 is the leftmost character.
INT_LOW = -30
INT_HIGH = 30
N_BITS = 6

INT_DOMAIN = "int"
BIT_DOMAIN = "bits"

BINARY_LABELS = ("0", "1")
ORDINAL_LABELS = ("A", "B", "C")

# Families excluded from training so that generalisation can be measured: one whole
# family never seen during training, and one composition of two concepts that are
# each seen separately.
HELDOUT_FAMILY = "absgt"
HELDOUT_COMPOSITION = "gt_and_mod"

# Within every other family, every ``PARAM_SPLIT_EVERY``-th parameter value is held
# out, interleaved with the training values rather than taken from one end.
PARAM_SPLIT_EVERY = 4
PARAM_SPLIT_OFFSET = 2


@dataclass(frozen=True)
class FunctionFamily:
    """A parameterised family of rules sharing a domain, label set and phrasing.

    ``build`` turns a parameter dict into the rule itself; ``grid`` enumerates the
    parameter dicts that are worth generating for a given domain. ``templates`` are
    paraphrases of the same rule -- ``templates[0]`` is treated as canonical -- and
    ``expression`` is a Python-syntax rendering of the rule, so a consumer can
    reconstruct it without re-deriving it from prose.
    """

    name: str
    domain: str
    labels: tuple[str, ...]
    templates: tuple[str, ...]
    expression: str
    build: Callable[[dict], Callable[[Any], str]]
    grid: Callable[["DomainSpec"], list[dict]]


@dataclass(frozen=True)
class DomainSpec:
    """The two finite domains, and the grid strides used to enumerate parameters."""

    int_low: int = INT_LOW
    int_high: int = INT_HIGH
    n_bits: int = N_BITS
    range_stride: int = 1
    composition_stride: int = 2

    def __post_init__(self) -> None:
        if self.int_low >= self.int_high:
            raise ValueError("int_low must be strictly less than int_high")
        if self.n_bits < 2:
            raise ValueError("n_bits must be at least 2")
        for stride in (self.range_stride, self.composition_stride):
            if stride < 1:
                raise ValueError("strides must be positive")

    def values(self, domain: str) -> tuple:
        """Every input in the named domain, in canonical order."""
        if domain == INT_DOMAIN:
            return tuple(range(self.int_low, self.int_high + 1))
        if domain == BIT_DOMAIN:
            return tuple(format(i, f"0{self.n_bits}b") for i in range(2**self.n_bits))
        raise ValueError(f"unknown domain {domain!r}")


@dataclass(frozen=True)
class TargetFunction:
    """One concrete rule: its identity, its phrasing, and its behaviour."""

    function_id: str
    family: str
    params: dict
    domain: str
    labels: tuple[str, ...]
    description: str
    paraphrases: tuple[str, ...]
    expression: str
    compute: Callable[[Any], str]
    values: tuple

    def table(self) -> list[tuple]:
        """Every ``(x, f(x))`` pair in the domain."""
        return [(x, self.compute(x)) for x in self.values]

    def label_counts(self) -> dict[str, int]:
        counts = {label: 0 for label in self.labels}
        for _, y in self.table():
            counts[y] = counts.get(y, 0) + 1
        return counts

    def majority_label_fraction(self) -> float:
        """Accuracy of the best constant predictor -- the baseline any method must beat."""
        counts = self.label_counts()
        return max(counts.values()) / len(self.values)

    @property
    def seed(self) -> int:
        """A stable per-function seed, so splits do not depend on iteration order."""
        return zlib.crc32(self.function_id.encode())


# ---------------------------------------------------------------------------
# The families
# ---------------------------------------------------------------------------


def _int_grid(spec: DomainSpec, key: str, lo: int, hi: int) -> list[dict]:
    return [{key: value} for value in range(lo, hi + 1)]


def _pair_grid(spec: DomainSpec, low: int, high: int) -> list[dict]:
    stride = spec.range_stride
    return [
        {"a": a, "b": b}
        for a in range(low, high, stride)
        for b in range(a + stride, high + 1, stride)
    ]


def _composition_grid(spec: DomainSpec) -> list[dict]:
    return [
        {"c": c, "k": k, "r": r}
        for c in range(spec.int_low + 1, spec.int_high, spec.composition_stride)
        for k in range(2, 6)
        for r in range(k)
    ]


FAMILIES: tuple[FunctionFamily, ...] = (
    FunctionFamily(
        name="gt",
        domain=INT_DOMAIN,
        labels=BINARY_LABELS,
        templates=(
            "Output 1 if x is greater than {c}, otherwise output 0.",
            "Return 1 when x > {c}, else return 0.",
            "f is 1 exactly when x exceeds {c}, and 0 otherwise.",
        ),
        expression="1 if x > {c} else 0",
        build=lambda p: lambda x, c=p["c"]: "1" if x > c else "0",
        grid=lambda s: _int_grid(s, "c", s.int_low, s.int_high - 1),
    ),
    FunctionFamily(
        name="lt",
        domain=INT_DOMAIN,
        labels=BINARY_LABELS,
        templates=(
            "Output 1 if x is less than {c}, otherwise output 0.",
            "Return 1 when x < {c}, else return 0.",
            "f is 1 exactly when x is below {c}, and 0 otherwise.",
        ),
        expression="1 if x < {c} else 0",
        build=lambda p: lambda x, c=p["c"]: "1" if x < c else "0",
        grid=lambda s: _int_grid(s, "c", s.int_low + 1, s.int_high),
    ),
    FunctionFamily(
        name="absgt",
        domain=INT_DOMAIN,
        labels=BINARY_LABELS,
        templates=(
            "Output 1 if the absolute value of x is greater than {c}, otherwise output 0.",
            "Return 1 when |x| > {c}, else return 0.",
            "f is 1 exactly when the magnitude of x exceeds {c}, and 0 otherwise.",
        ),
        expression="1 if abs(x) > {c} else 0",
        build=lambda p: lambda x, c=p["c"]: "1" if abs(x) > c else "0",
        grid=lambda s: _int_grid(s, "c", 0, s.int_high - 1),
    ),
    FunctionFamily(
        name="range",
        domain=INT_DOMAIN,
        labels=BINARY_LABELS,
        templates=(
            "Output 1 if x is strictly between {a} and {b}, otherwise output 0.",
            "Return 1 when {a} < x < {b}, else return 0.",
            "f is 1 exactly when x lies between {a} and {b} (exclusive), and 0 otherwise.",
        ),
        expression="1 if {a} < x < {b} else 0",
        build=lambda p: lambda x, a=p["a"], b=p["b"]: "1" if a < x < b else "0",
        grid=lambda s: _pair_grid(s, s.int_low, s.int_high),
    ),
    FunctionFamily(
        name="mod",
        domain=INT_DOMAIN,
        labels=BINARY_LABELS,
        templates=(
            "Output 1 if x modulo {k} equals {r}, otherwise output 0.",
            "Return 1 when x % {k} == {r}, else return 0.",
            "f is 1 exactly when x leaves remainder {r} after division by {k}, and 0 otherwise.",
        ),
        expression="1 if x % {k} == {r} else 0",
        build=lambda p: lambda x, k=p["k"], r=p["r"]: "1" if x % k == r else "0",
        grid=lambda s: [{"k": k, "r": r} for k in range(2, 13) for r in range(k)],
    ),
    FunctionFamily(
        name="piecewise",
        domain=INT_DOMAIN,
        labels=ORDINAL_LABELS,
        templates=(
            "Output A if x is less than {a}, B if x is less than {b}, otherwise C.",
            "Return A when x < {a}, B when {a} <= x < {b}, and C when x >= {b}.",
            "f is A below {a}, B from {a} up to but not including {b}, and C from {b} upward.",
        ),
        expression="'A' if x < {a} else ('B' if x < {b} else 'C')",
        build=lambda p: lambda x, a=p["a"], b=p["b"]: "A" if x < a else ("B" if x < b else "C"),
        grid=lambda s: _pair_grid(s, s.int_low, s.int_high),
    ),
    FunctionFamily(
        name="lt_and_mod",
        domain=INT_DOMAIN,
        labels=BINARY_LABELS,
        templates=(
            "Output 1 if x is less than {c} and x modulo {k} equals {r}, otherwise output 0.",
            "Return 1 when x < {c} and x % {k} == {r}, else return 0.",
        ),
        expression="1 if x < {c} and x % {k} == {r} else 0",
        build=lambda p: lambda x, c=p["c"], k=p["k"], r=p["r"]: (
            "1" if x < c and x % k == r else "0"
        ),
        grid=_composition_grid,
    ),
    FunctionFamily(
        name="gt_and_mod",
        domain=INT_DOMAIN,
        labels=BINARY_LABELS,
        templates=(
            "Output 1 if x is greater than {c} and x modulo {k} equals {r}, otherwise output 0.",
            "Return 1 when x > {c} and x % {k} == {r}, else return 0.",
        ),
        expression="1 if x > {c} and x % {k} == {r} else 0",
        build=lambda p: lambda x, c=p["c"], k=p["k"], r=p["r"]: (
            "1" if x > c and x % k == r else "0"
        ),
        grid=_composition_grid,
    ),
    FunctionFamily(
        name="bit_is_1",
        domain=BIT_DOMAIN,
        labels=BINARY_LABELS,
        templates=(
            "Output 1 if bit {i} of x (counting from 0 at the leftmost bit) is 1, otherwise output 0.",
            "Return 1 when x[{i}] == 1, else return 0, indexing x from the left starting at 0.",
            "f is 1 exactly when position {i} of the bit string x, counted from the left starting at 0, is a 1, and 0 otherwise.",
        ),
        expression="1 if x[{i}] == '1' else 0",
        build=lambda p: lambda x, i=p["i"]: "1" if x[i] == "1" else "0",
        grid=lambda s: [{"i": i} for i in range(s.n_bits)],
    ),
    FunctionFamily(
        name="at_least",
        domain=BIT_DOMAIN,
        labels=BINARY_LABELS,
        templates=(
            "Output 1 if x contains at least {k} ones, otherwise output 0.",
            "Return 1 when the number of 1s in x is >= {k}, else return 0.",
            "f is 1 exactly when x has {k} or more ones, and 0 otherwise.",
        ),
        expression="1 if x.count('1') >= {k} else 0",
        build=lambda p: lambda x, k=p["k"]: "1" if x.count("1") >= k else "0",
        grid=lambda s: [{"k": k} for k in range(1, s.n_bits + 1)],
    ),
    FunctionFamily(
        name="bits_equal",
        domain=BIT_DOMAIN,
        labels=BINARY_LABELS,
        templates=(
            "Output 1 if bits {i} and {j} of x (counting from 0 at the leftmost bit) are equal, otherwise output 0.",
            "Return 1 when x[{i}] == x[{j}], else return 0, indexing x from the left starting at 0.",
            "f is 1 exactly when positions {i} and {j} of the bit string x, counted from the left starting at 0, hold the same value, and 0 otherwise.",
        ),
        expression="1 if x[{i}] == x[{j}] else 0",
        build=lambda p: lambda x, i=p["i"], j=p["j"]: "1" if x[i] == x[j] else "0",
        grid=lambda s: [
            {"i": i, "j": j} for i in range(s.n_bits) for j in range(i + 1, s.n_bits)
        ],
    ),
)

FAMILIES_BY_NAME = {family.name: family for family in FAMILIES}


def _function_id(family: str, params: dict) -> str:
    joined = "_".join(f"{key}={value}" for key, value in params.items())
    return f"{family}_{joined}"


def make_function(family_name: str, params: dict, spec: DomainSpec) -> TargetFunction:
    """Instantiate one rule from a family name and a parameter dict."""
    family = FAMILIES_BY_NAME[family_name]
    paraphrases = tuple(template.format(**params) for template in family.templates)
    return TargetFunction(
        function_id=_function_id(family_name, params),
        family=family_name,
        params=dict(params),
        domain=family.domain,
        labels=family.labels,
        description=paraphrases[0],
        paraphrases=paraphrases,
        expression=family.expression.format(**params),
        compute=family.build(params),
        values=spec.values(family.domain),
    )


def iter_functions(
    spec: DomainSpec,
    *,
    min_minority: int = 4,
    max_majority_fraction: float | None = 0.90,
    max_per_family: int | None = 300,
) -> Iterator[TargetFunction]:
    """Yield every non-degenerate rule, capped to a per-family budget.

    A rule is *degenerate* when it is nearly constant: a constant predictor already
    reproduces it, so its examples carry almost no signal about the rule and any
    method scored on it looks good for the wrong reason. Two filters guard against
    this -- an absolute floor on the rarest label (``min_minority``, which matters
    on small domains) and a ceiling on the constant-predictor baseline
    (``max_majority_fraction``, which is comparable across domains of different
    sizes). Families are capped independently, by taking an evenly spaced subset of
    the surviving parameter grid, so a family with a combinatorially large grid
    (``range``, ``piecewise``) cannot swamp a small one (``bit_is_1``).

    Args:
        spec: The domains and grid strides to enumerate over.
        min_minority: Smallest permitted count for any label, over the full domain.
        max_majority_fraction: Largest permitted constant-predictor accuracy, or
            ``None`` to apply only the absolute floor.
        max_per_family: Per-family cap, or ``None`` to keep every surviving rule.
    """
    if min_minority < 1:
        raise ValueError("min_minority must be at least 1")
    if max_majority_fraction is not None and not 0.0 < max_majority_fraction <= 1.0:
        raise ValueError("max_majority_fraction must lie in (0, 1]")
    if max_per_family is not None and max_per_family < 1:
        raise ValueError("max_per_family must be positive or None")

    for family in FAMILIES:
        kept = []
        for params in family.grid(spec):
            function = make_function(family.name, params, spec)
            counts = function.label_counts()
            if min(counts.values()) < min_minority:
                continue
            if (
                max_majority_fraction is not None
                and function.majority_label_fraction() > max_majority_fraction
            ):
                continue
            kept.append(function)
        if max_per_family is not None and len(kept) > max_per_family:
            # Evenly spaced indices keep the full parameter span rather than a prefix.
            step = len(kept) / max_per_family
            kept = [kept[int(i * step)] for i in range(max_per_family)]
        yield from kept


def assign_split(function: TargetFunction, index: int) -> str:
    """Which evaluation split a rule belongs to.

    ``heldout_family`` is a family never seen in training, ``heldout_composition`` a
    composition of two concepts seen only separately, and ``heldout_param``
    interleaved parameter values of an otherwise-trained family. ``index`` is the
    rule's position within its own family.
    """
    if function.family == HELDOUT_FAMILY:
        return "heldout_family"
    if function.family == HELDOUT_COMPOSITION:
        return "heldout_composition"
    if index % PARAM_SPLIT_EVERY == PARAM_SPLIT_OFFSET:
        return "heldout_param"
    return "train"


# ---------------------------------------------------------------------------
# Example sets
# ---------------------------------------------------------------------------


def holdout_inputs(function: TargetFunction, n_holdout: int) -> list:
    """Inputs reserved for evaluating a rule, fixed per function.

    The reservation is allocated across labels *in proportion to label frequency*,
    with at least two inputs per label and never more than half of any label. A
    label-balanced reservation would take almost every instance of a rare label,
    leaving the training examples unable to show where the rule's boundary lies.

    The same inputs are withheld from every example set for this function, so that
    different draws of the same rule cannot leak each other's evaluation inputs.
    """
    if n_holdout < 0:
        raise ValueError("n_holdout must be non-negative")
    rng = random.Random(function.seed)
    by_label: dict[str, list] = {}
    for x, y in function.table():
        by_label.setdefault(y, []).append(x)

    reserved: list = []
    for label in sorted(by_label):
        pool = list(by_label[label])
        rng.shuffle(pool)
        share = round(n_holdout * len(pool) / len(function.values))
        take = min(max(2, share), len(pool) // 2)
        reserved.extend(pool[:take])
    return reserved


def draw_examples(
    function: TargetFunction,
    pool: Sequence,
    n_examples: int,
    rng: random.Random,
    *,
    min_per_label: int = 2,
) -> list[tuple]:
    """Sample ``n_examples`` inputs from ``pool`` without replacement.

    The draw is stratified only enough to guarantee ``min_per_label`` instances of
    every label that ``pool`` contains; the remainder is uniform, so the example set
    still reflects the rule's true label proportions. Without that guarantee a skewed
    rule would sometimes produce an example set showing a single label, which no
    method could describe.
    """
    if n_examples < 0:
        raise ValueError("n_examples must be non-negative")
    if min_per_label < 0:
        raise ValueError("min_per_label must be non-negative")

    by_label: dict[str, list] = {}
    for x in pool:
        by_label.setdefault(function.compute(x), []).append(x)

    chosen: list = []
    for label in sorted(by_label):
        take = min(min_per_label, len(by_label[label]), max(0, n_examples - len(chosen)))
        chosen.extend(rng.sample(by_label[label], take))

    taken = set(chosen)
    remaining = [x for x in pool if x not in taken]
    rng.shuffle(remaining)
    chosen.extend(remaining[: max(0, n_examples - len(chosen))])
    rng.shuffle(chosen)
    return [(x, function.compute(x)) for x in chosen]


def build_records(
    function: TargetFunction,
    split: str,
    *,
    example_sizes: Sequence[int],
    n_holdout: int,
    views: int,
    min_per_label: int,
    seed: int,
) -> list[dict]:
    """Build every example set for one rule.

    For each size in ``example_sizes``, ``views`` independent draws are taken. All of
    them share one rule and one reserved evaluation set, and differ only in which
    training inputs they show. Several views of a rule make it possible to ask
    whether a method has learned the rule or merely memorised one presentation of
    it, so every view of a rule stays in the same split; several sizes make it
    possible to ask how much observed behaviour a description needs in order to be
    recoverable at all.
    """
    if views < 1:
        raise ValueError("views must be at least 1")
    if not example_sizes:
        raise ValueError("example_sizes must not be empty")
    if any(size < 1 for size in example_sizes):
        raise ValueError("every example size must be positive")

    reserved = holdout_inputs(function, n_holdout)
    reserved_set = set(reserved)
    train_pool = [x for x in function.values if x not in reserved_set]
    holdout_examples = [(x, function.compute(x)) for x in reserved]

    definition = {
        "function_id": function.function_id,
        "family": function.family,
        "params": function.params,
        "domain": function.domain,
        "split": split,
        "labels": list(function.labels),
        "description": function.description,
        "paraphrases": list(function.paraphrases),
        "expression": function.expression,
        "domain_size": len(function.values),
        "label_counts": function.label_counts(),
        "majority_label_fraction": round(function.majority_label_fraction(), 6),
    }

    records = []
    for size in example_sizes:
        for view in range(views):
            rng = random.Random(
                (function.seed + 7919 * view + 104729 * size + seed) % (2**32)
            )
            examples = draw_examples(
                function, train_pool, size, rng, min_per_label=min_per_label
            )
            records.append(
                {
                    **definition,
                    "n_examples": len(examples),
                    "view": view,
                    "examples": [[x, y] for x, y in examples],
                    "holdout_examples": [[x, y] for x, y in holdout_examples],
                }
            )
    return records


def build_dataset(
    *,
    spec: DomainSpec | None = None,
    example_sizes: Sequence[int] = (8, 16, 32),
    n_holdout: int = 16,
    views: int = 4,
    min_per_label: int = 2,
    min_minority: int = 4,
    max_majority_fraction: float | None = 0.90,
    max_per_family: int | None = 300,
    seed: int = 0,
) -> Iterator[dict]:
    """Yield the whole dataset, one ``(examples, definition)`` record at a time.

    Args:
        spec: Domains and grid strides; the module defaults when omitted.
        example_sizes: Observed ``(x, y)`` pair counts to generate; every rule gets
            records at each size.
        n_holdout: Inputs reserved per rule for evaluation, shared by its views.
        views: Independent example sets drawn per rule *per size*.
        min_per_label: Labels guaranteed to appear in every example set.
        min_minority: Absolute degeneracy floor passed to :func:`iter_functions`.
        max_majority_fraction: Skew ceiling passed to :func:`iter_functions`.
        max_per_family: Per-family cap passed to :func:`iter_functions`.
        seed: Global offset; each record's draw is also keyed to its rule and view.
    """
    spec = spec if spec is not None else DomainSpec()
    per_family_index: dict[str, int] = {}
    for function in iter_functions(
        spec,
        min_minority=min_minority,
        max_majority_fraction=max_majority_fraction,
        max_per_family=max_per_family,
    ):
        index = per_family_index.get(function.family, 0)
        per_family_index[function.family] = index + 1
        yield from build_records(
            function,
            assign_split(function, index),
            example_sizes=example_sizes,
            n_holdout=n_holdout,
            views=views,
            min_per_label=min_per_label,
            seed=seed,
        )


def format_examples(examples: Sequence[Sequence], *, variable: str = "x") -> str:
    """Render example pairs as the plain text an LLM is usually shown."""
    return "\n".join(f"{variable} = {x} -> f = {y}" for x, y in examples)


def summarise(records: Sequence[dict]) -> dict:
    """Counts by split and by family, plus the mean constant-predictor baseline."""
    by_split: dict[str, int] = {}
    by_family: dict[str, int] = {}
    by_size: dict[int, int] = {}
    functions: set[str] = set()
    baseline = 0.0
    for record in records:
        by_split[record["split"]] = by_split.get(record["split"], 0) + 1
        by_family[record["family"]] = by_family.get(record["family"], 0) + 1
        by_size[record["n_examples"]] = by_size.get(record["n_examples"], 0) + 1
        functions.add(record["function_id"])
        baseline += record["majority_label_fraction"]
    return {
        "records": len(records),
        "functions": len(functions),
        "by_split": dict(sorted(by_split.items())),
        "by_family": dict(sorted(by_family.items())),
        "by_n_examples": dict(sorted(by_size.items())),
        "mean_majority_label_fraction": round(baseline / len(records), 6) if records else 0.0,
    }


def load_dataset(path: str) -> list[dict]:
    """Read a dataset written by ``scripts/generate_dataset/discrete.py``."""
    opener: Callable = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


# ---------------------------------------------------------------------------
# Diagnostic probes
# ---------------------------------------------------------------------------


def probe_inputs(function: TargetFunction) -> list:
    """Inputs chosen to tell the true rule apart from rules that merely fit a sample.

    A randomly drawn held-out set rarely falsifies a near-miss: a description of
    ``x >= 12`` for the rule ``x > 10`` scores perfectly whenever 11 is absent, and
    ``x % 5 == 0`` scores perfectly for ``x < 19 and x % 5 == 0`` whenever no held-out
    input sits above 19. These probes target the places where such rules must differ --
    thresholds and their neighbours, every residue class, each clause of a conjunction
    alone, each transition of a piecewise rule, and single-bit flips.

    Derived from the rule's own parameters, so they are generated on demand rather than
    stored: the useful probe set does not depend on which examples a method was shown.
    """
    p, values = function.params, function.values
    in_domain = set(values)
    out: list = []

    def take(*xs):
        out.extend(x for x in xs if x in in_domain)

    fam = function.family
    if fam in ("gt", "lt"):
        c = p["c"]
        take(c - 2, c - 1, c, c + 1, c + 2)
    elif fam == "absgt":
        c = p["c"]
        for d in (-1, 0, 1, 2):
            take(c + d, -(c + d))
    elif fam in ("range", "piecewise"):
        a, b = p["a"], p["b"]
        take(a - 1, a, a + 1, b - 1, b, b + 1)
    elif fam == "mod":
        k = p["k"]
        for j in range(k):  # one representative per residue class
            take(next((x for x in values if x % k == j), None))
    elif fam in ("lt_and_mod", "gt_and_mod"):
        c, k, r = p["c"], p["k"], p["r"]
        take(c - 1, c, c + 1)
        side = (lambda x: x < c) if fam == "lt_and_mod" else (lambda x: x > c)
        # exactly one clause satisfied, in each direction, plus both satisfied
        take(next((x for x in values if side(x) and x % k != r), None))
        take(next((x for x in values if not side(x) and x % k == r), None))
        take(next((x for x in values if side(x) and x % k == r), None))
    elif fam == "bit_is_1":
        i = p["i"]
        for base in (values[0], values[-1]):  # all-zeros and all-ones
            flipped = base[:i] + ("1" if base[i] == "0" else "0") + base[i + 1:]
            take(base, flipped)
    elif fam == "at_least":
        k, n = p["k"], len(values[0])
        for count in (k - 1, k):  # straddle the threshold
            if 0 <= count <= n:
                take("1" * count + "0" * (n - count))
    elif fam == "bits_equal":
        i, j = p["i"], p["j"]
        for bi, bj in (("0", "0"), ("0", "1"), ("1", "0"), ("1", "1")):
            s = list(values[0])
            s[i], s[j] = bi, bj
            take("".join(s))

    seen: set = set()
    return [x for x in out if x is not None and not (x in seen or seen.add(x))]


def probe_examples(function: TargetFunction) -> list[tuple]:
    """``probe_inputs`` paired with the true outputs."""
    return [(x, function.compute(x)) for x in probe_inputs(function)]
