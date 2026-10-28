"""Library of target functions f* for the proof-of-concept stage.

Each function is a plain Python rule with a known ground-truth description,
so recovered verbalizations can be checked against a known answer before
moving on to opaque learned models (small nets, then transformers).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable


@dataclass
class FunctionSpec:
    name: str
    arity: int
    compute: Callable[..., str]          # (*inputs) -> label string
    candidates: list[str]                # possible label strings, for scoring
    format_input: Callable[..., str]      # (*inputs) -> "x = 3" style text
    sampler: Callable[[], tuple]          # () -> a random input tuple
    true_description: str                 # ground-truth rule, for sanity-checking only
    family: str | None = None             # e.g. "gt" / "lt" / "absgt" -- for z-space probing
    param: float | None = None            # the family's numeric parameter c -- for z-space probing
    params: dict | None = None            # structured parameters, e.g. {"c": 5} or {"k": 3, "r": 1}
    paraphrases: list[str] | None = None  # equivalent descriptions; paraphrases[0] == true_description
    domain: list[tuple] | None = None     # full input domain, for disjoint support/query draws


def _sample_int(lo: int, hi: int) -> Callable[[], tuple]:
    return lambda: (random.randint(lo, hi),)


def _sample_pair(lo: int, hi: int) -> Callable[[], tuple]:
    return lambda: (random.randint(lo, hi), random.randint(lo, hi))


FUNCTIONS: dict[str, FunctionSpec] = {
    "sign": FunctionSpec(
        name="sign",
        arity=1,
        compute=lambda x: "1" if x > 0 else "0",
        candidates=["0", "1"],
        format_input=lambda x: f"x = {x}",
        sampler=_sample_int(-60, 60),
        true_description="Output 1 if x is positive, otherwise output 0.",
    ),
    "parity": FunctionSpec(
        name="parity",
        arity=1,
        compute=lambda x: "1" if x % 2 == 0 else "0",
        candidates=["0", "1"],
        format_input=lambda x: f"x = {x}",
        sampler=_sample_int(-60, 60),
        true_description="Output 1 if x is even, otherwise output 0.",
    ),
    "threshold10": FunctionSpec(
        name="threshold10",
        arity=1,
        compute=lambda x: "1" if x > 10 else "0",
        candidates=["0", "1"],
        format_input=lambda x: f"x = {x}",
        sampler=_sample_int(-60, 60),
        true_description="Output 1 if x is greater than 10, otherwise output 0.",
    ),
    "compare_xy": FunctionSpec(
        name="compare_xy",
        arity=2,
        compute=lambda x, y: "1" if x > y else "0",
        candidates=["0", "1"],
        format_input=lambda x, y: f"x = {x}, y = {y}",
        sampler=_sample_pair(-10, 10),
        true_description="Output 1 if x is greater than y, otherwise output 0.",
    ),
    "compositional": FunctionSpec(
        name="compositional",
        arity=1,
        compute=lambda x: "1" if (x > 0 and x % 2 == 0) else "0",
        candidates=["0", "1"],
        format_input=lambda x: f"x = {x}",
        sampler=_sample_int(-60, 60),
        true_description="Output 1 if x is positive AND even, otherwise output 0.",
    ),
    "piecewise": FunctionSpec(
        name="piecewise",
        arity=1,
        compute=lambda x: "A" if x < 0 else ("B" if x < 10 else "C"),
        candidates=["A", "B", "C"],
        format_input=lambda x: f"x = {x}",
        sampler=_sample_int(-60, 60),
        true_description="Output A if x < 0, B if 0 <= x < 10, otherwise C.",
    ),
}


def make_parametric_functions(
    seed: int = 0,
) -> tuple[list[FunctionSpec], list[FunctionSpec]]:
    """A family of parameterized predicates where the description is templated,
    so every function comes with a known ground-truth description for free.

    Held-out test functions reuse the same families/templates as training but
    with unseen parameter values -- e.g. train on x>2, x>5, x>8, test on x>11.
    This is the supervision signal for training a z -> description decoder,
    and the generalization test for whether it learned the template rather
    than memorizing individual (z, description) pairs.
    """
    train: list[FunctionSpec] = []
    test: list[FunctionSpec] = []

    def fmt1(x):
        return f"x = {x}"

    # x > c
    for c in [2, 5, 8, -3]:
        train.append(FunctionSpec(
            name=f"gt_{c}", arity=1,
            compute=lambda x, c=c: "1" if x > c else "0",
            candidates=["0", "1"], format_input=fmt1, sampler=_sample_int(-30, 30),
            true_description=f"Output 1 if x is greater than {c}, otherwise output 0.",
        ))
    for c in [11, -6]:
        test.append(FunctionSpec(
            name=f"gt_{c}", arity=1,
            compute=lambda x, c=c: "1" if x > c else "0",
            candidates=["0", "1"], format_input=fmt1, sampler=_sample_int(-30, 30),
            true_description=f"Output 1 if x is greater than {c}, otherwise output 0.",
        ))

    # x < c
    for c in [3, -4, 10, -8]:
        train.append(FunctionSpec(
            name=f"lt_{c}", arity=1,
            compute=lambda x, c=c: "1" if x < c else "0",
            candidates=["0", "1"], format_input=fmt1, sampler=_sample_int(-30, 30),
            true_description=f"Output 1 if x is less than {c}, otherwise output 0.",
        ))
    for c in [-2, 12]:
        test.append(FunctionSpec(
            name=f"lt_{c}", arity=1,
            compute=lambda x, c=c: "1" if x < c else "0",
            candidates=["0", "1"], format_input=fmt1, sampler=_sample_int(-30, 30),
            true_description=f"Output 1 if x is less than {c}, otherwise output 0.",
        ))

    # x % k == r
    for k, r in [(2, 0), (2, 1), (3, 0), (4, 2)]:
        train.append(FunctionSpec(
            name=f"mod_{k}_{r}", arity=1,
            compute=lambda x, k=k, r=r: "1" if x % k == r else "0",
            candidates=["0", "1"], format_input=fmt1, sampler=_sample_int(-30, 30),
            true_description=f"Output 1 if x modulo {k} equals {r}, otherwise output 0.",
        ))
    for k, r in [(3, 2), (4, 1)]:
        test.append(FunctionSpec(
            name=f"mod_{k}_{r}", arity=1,
            compute=lambda x, k=k, r=r: "1" if x % k == r else "0",
            candidates=["0", "1"], format_input=fmt1, sampler=_sample_int(-30, 30),
            true_description=f"Output 1 if x modulo {k} equals {r}, otherwise output 0.",
        ))

    # c1 < x < c2
    for c1, c2 in [(0, 10), (-10, 0), (5, 15)]:
        train.append(FunctionSpec(
            name=f"range_{c1}_{c2}", arity=1,
            compute=lambda x, c1=c1, c2=c2: "1" if c1 < x < c2 else "0",
            candidates=["0", "1"], format_input=fmt1, sampler=_sample_int(-30, 30),
            true_description=f"Output 1 if x is between {c1} and {c2}, otherwise output 0.",
        ))
    for c1, c2 in [(-5, 5), (10, 20)]:
        test.append(FunctionSpec(
            name=f"range_{c1}_{c2}", arity=1,
            compute=lambda x, c1=c1, c2=c2: "1" if c1 < x < c2 else "0",
            candidates=["0", "1"], format_input=fmt1, sampler=_sample_int(-30, 30),
            true_description=f"Output 1 if x is between {c1} and {c2}, otherwise output 0.",
        ))

    # |x| > c
    for c in [3, 7, 12]:
        train.append(FunctionSpec(
            name=f"absgt_{c}", arity=1,
            compute=lambda x, c=c: "1" if abs(x) > c else "0",
            candidates=["0", "1"], format_input=fmt1, sampler=_sample_int(-30, 30),
            true_description=f"Output 1 if the absolute value of x is greater than {c}, otherwise output 0.",
        ))
    for c in [5, 15]:
        test.append(FunctionSpec(
            name=f"absgt_{c}", arity=1,
            compute=lambda x, c=c: "1" if abs(x) > c else "0",
            candidates=["0", "1"], format_input=fmt1, sampler=_sample_int(-30, 30),
            true_description=f"Output 1 if the absolute value of x is greater than {c}, otherwise output 0.",
        ))

    rng = random.Random(seed)
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def make_dense_families(
    seed: int = 0,
) -> tuple[list[FunctionSpec], list[FunctionSpec]]:
    """Dense parameter sweeps over 3 families (gt_c, lt_c, absgt_c), holding out
    ALTERNATING (interpolated, not extrapolated) parameter values for testing.
    e.g. train on c = -18,-10,-2,6,14 and test on c = -14,-6,2,10,18.

    Every FunctionSpec carries its family and numeric param, for probing whether
    z-space is geometrically organized by function semantics (see run_zspace_probe.py).
    """
    train: list[FunctionSpec] = []
    test: list[FunctionSpec] = []

    def fmt1(x):
        return f"x = {x}"

    def add(target, family, c, compute, desc):
        target.append(FunctionSpec(
            name=f"{family}_{c}", arity=1, compute=compute,
            candidates=["0", "1"], format_input=fmt1, sampler=_sample_int(-30, 30),
            true_description=desc, family=family, param=float(c),
        ))

    for i, c in enumerate(range(-18, 19, 4)):  # -18,-14,...,18 (10 values)
        target = test if i % 2 == 1 else train
        add(target, "gt", c, lambda x, c=c: "1" if x > c else "0",
            f"Output 1 if x is greater than {c}, otherwise output 0.")

    for i, c in enumerate(range(-18, 19, 4)):
        target = test if i % 2 == 1 else train
        add(target, "lt", c, lambda x, c=c: "1" if x < c else "0",
            f"Output 1 if x is less than {c}, otherwise output 0.")

    for i, c in enumerate(range(2, 21, 2)):  # 2,4,...,20 (10 values)
        target = test if i % 2 == 1 else train
        add(target, "absgt", c, lambda x, c=c: "1" if abs(x) > c else "0",
            f"Output 1 if the absolute value of x is greater than {c}, otherwise output 0.")

    rng = random.Random(seed)
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def generate_dataset(
    spec: FunctionSpec, n_train: int, n_test: int, seed: int = 0
) -> tuple[list[tuple], list[tuple]]:
    """Sample disjoint train/test sets of (inputs, label) pairs."""
    rng = random.Random(seed)
    random.seed(seed)  # spec.sampler uses the module-level `random`
    seen: set[tuple] = set()
    examples: list[tuple] = []
    needed = n_train + n_test
    attempts = 0
    max_attempts = needed * 50
    while len(examples) < needed and attempts < max_attempts:
        attempts += 1
        inputs = spec.sampler()
        if inputs in seen:
            continue
        seen.add(inputs)
        label = spec.compute(*inputs)
        examples.append((inputs, label))
    rng.shuffle(examples)
    train, test = examples[:n_train], examples[n_train : n_train + n_test]
    return train, test
