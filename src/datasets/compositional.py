"""Depth-2 compositions of elementary integer rules, on a wide domain.

The single-rule tier is too easy for a frontier model, and its remaining error is mostly
not the model's: with 61 inputs, at most 47% of functions are uniquely determined by 45
examples, so exact recovery cannot rise much further for anyone. This tier raises the
difficulty while keeping the one property that makes the result interpretable -- the set
of candidate rules is small enough to enumerate, so how much the examples determine can
be COMPUTED rather than assumed.

A target is `A op B` for two atoms and an op in {and, or, and-not}:

    atoms   x > c        x < c        |x| > c        x % k == r
    ops     A and B      A or B       A and not B

over x in [-150, 150]. A wide domain does two jobs: a rule's examples are drawn from a
pool of hundreds rather than dozens, and memorising the shown inputs stops being cheaper
than learning the rule (on 61 points a soft prompt memorised ~45 examples at train 1.00
while generalising at ~0.5).

Identity is BEHAVIOUR, not syntax. A rule is represented as a bitmask of its outputs over
the whole domain, so `x > 5 and x > 3` and `x > 5` are one rule, and so are the many
spellings of a range. Identifiability, "how many rules fit these examples", counts
distinct behaviours; counting syntactic forms would overstate the ambiguity and understate
how much a given sample pins down.
"""

from __future__ import annotations

import random
import zlib
from functools import lru_cache

import numpy as np

from src.datasets.discrete import TargetFunction, build_records

DATASET = "compositional"
INT_LOW, INT_HIGH, MOD_MAX = -150, 150, 9
OPS = ("and", "or", "andnot")
LABELS = ("0", "1")
THRESHOLD_KINDS = ("gt", "lt", "abs")

Atom = tuple


def domain_spec(low: int = INT_LOW, high: int = INT_HIGH, mod_max: int = MOD_MAX) -> dict:
    """Self-describing, so a record can rebuild its function with no outside context."""
    return {"int_low": low, "int_high": high, "mod_max": mod_max}


def domain_values(ds: dict) -> tuple:
    return tuple(range(ds["int_low"], ds["int_high"] + 1))


def grammar_text(ds: dict) -> str:
    """The form of every target, in words, drawn from the generator's own constants.

    States the hypothesis class and nothing about any particular function: which
    conditions exist, how two are combined, and the domain.
    """
    return (f"it is built from two simple conditions on x, combined as \"A and B\", "
            f"\"A or B\" or \"A and not B\". Each condition is one of: x > c, x < c, "
            f"abs(x) > c, or x % k == r, where c is an integer, k is an integer from 2 to "
            f"{ds['mod_max']} and r is a remainder from 0 to k-1. The inputs are the integers "
            f"from {ds['int_low']} to {ds['int_high']}.")


def make_atoms(ds: dict) -> list[Atom]:
    lo, hi = ds["int_low"], ds["int_high"]
    atoms: list[Atom] = [("gt", c) for c in range(lo + 1, hi)]
    atoms += [("lt", c) for c in range(lo + 1, hi)]
    atoms += [("abs", c) for c in range(0, hi)]
    atoms += [("mod", k, r) for k in range(2, ds["mod_max"] + 1) for r in range(k)]
    return atoms


def atom_pred(a: Atom):
    if a[0] == "gt":
        c = a[1]
        return lambda x: x > c
    if a[0] == "lt":
        c = a[1]
        return lambda x: x < c
    if a[0] == "abs":
        c = a[1]
        return lambda x: abs(x) > c
    k, r = a[1], a[2]
    return lambda x: x % k == r


def atom_text(a: Atom) -> tuple[str, str]:
    """(prose, symbolic) renderings."""
    if a[0] == "gt":
        return f"x is greater than {a[1]}", f"x > {a[1]}"
    if a[0] == "lt":
        return f"x is less than {a[1]}", f"x < {a[1]}"
    if a[0] == "abs":
        return f"the absolute value of x is greater than {a[1]}", f"abs(x) > {a[1]}"
    return f"x modulo {a[1]} equals {a[2]}", f"x % {a[1]} == {a[2]}"


def atom_id(a: Atom) -> str:
    return {"gt": "gt{1}", "lt": "lt{1}", "abs": "abs{1}", "mod": "mod{1}r{2}"}[a[0]].format(*a)


def atom_kind(a: Atom) -> str:
    """T = threshold-like, M = modular. Used to stratify, since thresholds are ~95% of atoms."""
    return "T" if a[0] in THRESHOLD_KINDS else "M"


def make_composed(op: str, a: Atom, b: Atom, ds: dict) -> TargetFunction:
    pa, pb = atom_pred(a), atom_pred(b)
    (ta, sa), (tb, sb) = atom_text(a), atom_text(b)
    if op == "and":
        rule = lambda x: pa(x) and pb(x)  # noqa: E731
        desc = f"Output 1 if {ta} and {tb}, otherwise output 0."
        alt = f"Return 1 when ({sa}) and ({sb}), else return 0."
        expr = f"1 if ({sa}) and ({sb}) else 0"
    elif op == "or":
        rule = lambda x: pa(x) or pb(x)  # noqa: E731
        desc = f"Output 1 if {ta} or {tb}, otherwise output 0."
        alt = f"Return 1 when ({sa}) or ({sb}), else return 0."
        expr = f"1 if ({sa}) or ({sb}) else 0"
    else:
        rule = lambda x: pa(x) and not pb(x)  # noqa: E731
        desc = f"Output 1 if {ta} and it is not the case that {tb}, otherwise output 0."
        alt = f"Return 1 when ({sa}) and not ({sb}), else return 0."
        expr = f"1 if ({sa}) and not ({sb}) else 0"
    return TargetFunction(
        function_id=f"{op}({atom_id(a)},{atom_id(b)})",
        family=f"comp_{op}",
        params={"op": op, "a": list(a), "b": list(b)},
        domain="int",
        labels=LABELS,
        description=desc,
        paraphrases=(desc, alt),
        expression=expr,
        compute=lambda x: "1" if rule(x) else "0",
        values=domain_values(ds),
    )


def function_from_params(params: dict, ds: dict) -> TargetFunction:
    return make_composed(params["op"], tuple(params["a"]), tuple(params["b"]), ds)


@lru_cache(maxsize=2)
def _hypothesis_class(low: int, high: int, mod_max: int):
    """Every distinct behaviour reachable as `A op B`, with one representative spelling."""
    ds = domain_spec(low, high, mod_max)
    values = domain_values(ds)
    atoms = make_atoms(ds)
    masks = [sum(1 << i for i, x in enumerate(values) if atom_pred(a)(x)) for a in atoms]
    behaviours: dict[int, tuple] = {}
    for i, mi in enumerate(masks):
        for j, mj in enumerate(masks):
            if j >= i:  # and / or are commutative; and-not is not
                behaviours.setdefault(mi & mj, ("and", i, j))
                behaviours.setdefault(mi | mj, ("or", i, j))
            behaviours.setdefault(mi & ~mj, ("andnot", i, j))
    return atoms, masks, behaviours


def hypothesis_class(ds: dict):
    return _hypothesis_class(ds["int_low"], ds["int_high"], ds["mod_max"])


def behaviour_masks(ds: dict) -> list[int]:
    """The distinct full-domain behaviours, as bitmasks (bit i = output at the i-th input)."""
    return list(hypothesis_class(ds)[2])


def input_positions(ds: dict) -> dict:
    return {x: i for i, x in enumerate(domain_values(ds))}


def eligible_strata(ds: dict, max_majority: float = 0.85, min_minority: int = 30) -> dict:
    """Targets the generator may draw, grouped by (op, atom-kind pair).

    Excluded: any behaviour a single atom already expresses (not genuinely depth 2) and
    near-constant behaviours, which a constant predictor already solves. Grouped because
    thresholds are ~95% of atoms; drawing from the class uniformly would make almost every
    target a pair of thresholds.
    """
    atoms, masks, behaviours = hypothesis_class(ds)
    single = set(masks)
    n = len(domain_values(ds))
    strata: dict[tuple, list] = {}
    for mask, (op, i, j) in behaviours.items():
        if mask in single:
            continue
        ones = bin(mask).count("1")
        if min(ones, n - ones) < min_minority or max(ones, n - ones) / n > max_majority:
            continue
        pair = "".join(sorted((atom_kind(atoms[i]), atom_kind(atoms[j]))))
        strata.setdefault((op, pair), []).append((mask, op, i, j))
    return strata


def sample_functions(ds: dict, per_stratum: int, seed: int = 0, max_majority: float = 0.85,
                     min_minority: int = 30) -> list[tuple[TargetFunction, str]]:
    """Targets drawn evenly across strata, over DISTINCT behaviours."""
    atoms = hypothesis_class(ds)[0]
    strata = eligible_strata(ds, max_majority, min_minority)
    rng = random.Random(seed)
    out = []
    for key in sorted(strata):
        cands = sorted(strata[key])
        for mask, op, i, j in sorted(rng.sample(cands, min(per_stratum, len(cands)))):
            out.append((make_composed(op, atoms[i], atoms[j], ds), key[1]))
    return out


@lru_cache(maxsize=2)
def _eligible_prior(low: int, high: int, mod_max: int):
    ds = domain_spec(low, high, mod_max)
    strata = eligible_strata(ds)
    masks: list[int] = []
    prior: list[float] = []
    for key in sorted(strata):
        cands = strata[key]
        for mask, *_ in cands:
            masks.append(mask)
            prior.append(1.0 / (len(strata) * len(cands)))
    return masks, np.array(prior)


def eligible_prior(ds: dict):
    """The generator's own distribution over targets: strata equally likely, uniform within.

    A reference that knows this is the Bayes-optimal exact-recovery rate -- the best any
    method can do from the examples plus the prior -- which is sharper than guessing
    uniformly among the rules that fit (1/k), because a learner may prefer likelier rules.
    """
    return _eligible_prior(ds["int_low"], ds["int_high"], ds["mod_max"])


def bit_matrix(masks: list[int], n: int) -> np.ndarray:
    """Behaviours as an (n_behaviours, n_inputs) boolean matrix; column i = output at input i."""
    nbytes = (n + 7) // 8
    raw = np.frombuffer(b"".join(m.to_bytes(nbytes, "little") for m in masks), dtype=np.uint8)
    return np.unpackbits(raw.reshape(len(masks), nbytes), axis=1, bitorder="little")[:, :n].astype(bool)


def bayes_exact(matrix: np.ndarray, prior: np.ndarray, positions: dict,
                shown: list) -> tuple[float, int]:
    """(probability the most likely consistent behaviour is the target, number consistent)."""
    cols = [positions[x] for x, _ in shown]
    labels = np.array([y == "1" for _, y in shown])
    ok = (matrix[:, cols] == labels).all(axis=1)
    p = prior[ok]
    return (float(p.max() / p.sum()) if p.size else 0.0), int(ok.sum())


@lru_cache(maxsize=2)
def _class_matrix(low: int, high: int, mod_max: int) -> np.ndarray:
    ds = domain_spec(low, high, mod_max)
    return bit_matrix(list(hypothesis_class(ds)[2]), len(domain_values(ds)))


def teaching_order(matrix: np.ndarray, truth: np.ndarray, candidate_cols: list[int]):
    """Greedily pick inputs that rule out the most remaining rivals of the true behaviour.

    Returns (picked columns in order, number of behaviours still consistent). At each step
    the chosen input is the one where the most surviving behaviours disagree with the
    truth, so the survivors collapse as fast as a greedy teacher can manage. Stops when one
    remains, or when no available input separates the survivors -- which happens when the
    only inputs that would do so are held out.
    """
    cols = np.array(candidate_cols)
    alive = np.ones(matrix.shape[0], dtype=bool)
    free = np.ones(len(cols), dtype=bool)
    picks: list[int] = []
    while alive.sum() > 1:
        idx = np.flatnonzero(alive)
        removed = (matrix[np.ix_(idx, cols)] != truth[cols]).sum(axis=0)
        removed[~free] = -1
        j = int(removed.argmax())
        if removed[j] <= 0:
            break
        picks.append(int(cols[j]))
        free[j] = False
        alive &= matrix[:, cols[j]] == truth[cols[j]]
    return picks, int(alive.sum())


def teaching_pool(fn: TargetFunction, pool: list, ds: dict):
    """Reorder an example pool so its prefix is an identifying set, then random fill.

    A support of size N is `pool[:N]`: the first inputs are the greedy teaching picks, so
    once N reaches `identified_at` the examples determine the function uniquely, and any
    larger N only adds shuffled filler. `identified_at` is None when the held-out inputs
    are the only ones that could resolve the survivors.
    """
    values = domain_values(ds)
    positions = input_positions(ds)
    truth = np.array([fn.compute(x) == "1" for x in values])
    picks, left = teaching_order(_class_matrix(ds["int_low"], ds["int_high"], ds["mod_max"]),
                                 truth, [positions[x] for x, _ in pool])
    chosen = set(picks)
    taught = [(values[c], fn.compute(values[c])) for c in picks]
    rest = [(x, y) for x, y in pool if positions[x] not in chosen]
    return taught + rest, (len(picks) if left == 1 else None)


def split_for(function_id: str) -> str:
    """Stable 70/15/15 split by function, so every record of a function shares one."""
    h = zlib.crc32(function_id.encode()) % 100
    return "train" if h < 70 else "val" if h < 85 else "test"


def build_dataset(ds: dict, *, per_stratum: int, example_sizes, views: int, n_holdout: int,
                  seed: int = 0):
    """Yield (examples, definition) records in the same schema as the single-rule tier."""
    for fn, pair in sample_functions(ds, per_stratum, seed):
        for rec in build_records(fn, split_for(fn.function_id), example_sizes=example_sizes,
                                 n_holdout=n_holdout, views=views, min_per_label=2, seed=seed):
            rec.update(dataset=DATASET, domain_spec=ds, n_holdout=n_holdout, pair_type=pair)
            yield rec
