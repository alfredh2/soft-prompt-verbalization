"""Heterogeneous task suite for the shared behavioral encoder (Phase 2).

Families span numeric predicates, bit-string rules, 3-way piecewise functions and
simple compositions. Every task has several paraphrased descriptions and a
structured (family, params) identity, so decoded descriptions can be scored
semantically via parse_description rather than by string similarity.

Splits, in priority order:
  train                -- most parameter values of the training families
  heldout_param        -- interleaved parameter values of training families (primary test)
  heldout_composition  -- gt_c AND mod, composed from concepts seen separately in training
  heldout_family       -- absgt, never seen in training (stretch test)
"""

from __future__ import annotations

import random
import re

from .functions import FunctionSpec

NUM_DOMAIN = [(x,) for x in range(-30, 31)]
N_BITS = 6
BIT_DOMAIN = [(format(i, f"0{N_BITS}b"),) for i in range(2**N_BITS)]
BINARY = ["0", "1"]
ABC = ["A", "B", "C"]

HELDOUT_FAMILY = "absgt"
HELDOUT_COMPOSITION = "gt_and_mod"

TEMPLATES = {
    "gt": [
        "Output 1 if x is greater than {c}, otherwise output 0.",
        "Return 1 when x > {c}, else return 0.",
        "f is 1 exactly when x exceeds {c}, and 0 otherwise.",
    ],
    "lt": [
        "Output 1 if x is less than {c}, otherwise output 0.",
        "Return 1 when x < {c}, else return 0.",
        "f is 1 exactly when x is below {c}, and 0 otherwise.",
    ],
    "absgt": [
        "Output 1 if the absolute value of x is greater than {c}, otherwise output 0.",
        "Return 1 when |x| > {c}, else return 0.",
        "f is 1 exactly when the magnitude of x exceeds {c}, and 0 otherwise.",
    ],
    "range": [
        "Output 1 if x is strictly between {a} and {b}, otherwise output 0.",
        "Return 1 when {a} < x < {b}, else return 0.",
        "f is 1 exactly when x lies between {a} and {b} (exclusive), and 0 otherwise.",
    ],
    "mod": [
        "Output 1 if x modulo {k} equals {r}, otherwise output 0.",
        "Return 1 when x % {k} == {r}, else return 0.",
        "f is 1 exactly when x leaves remainder {r} after division by {k}, and 0 otherwise.",
    ],
    "bit_is_1": [
        "Output 1 if bit {i} of x (counting from 0 on the left) is 1, otherwise output 0.",
        "Return 1 when x[{i}] == 1, else return 0.",
        "f is 1 exactly when position {i} of the bit string x (from the left, starting at 0) is a 1, and 0 otherwise.",
    ],
    "at_least": [
        "Output 1 if x contains at least {k} ones, otherwise output 0.",
        "Return 1 when the number of 1s in x is >= {k}, else return 0.",
        "f is 1 exactly when x has {k} or more ones, and 0 otherwise.",
    ],
    "bits_equal": [
        "Output 1 if bits {i} and {j} of x (counting from 0 on the left) are equal, otherwise output 0.",
        "Return 1 when x[{i}] == x[{j}], else return 0.",
        "f is 1 exactly when positions {i} and {j} of the bit string x hold the same value, and 0 otherwise.",
    ],
    "piecewise": [
        "Output A if x is less than {a}, B if x is less than {b}, otherwise C.",
        "Return A when x < {a}, B when {a} <= x < {b}, and C when x >= {b}.",
        "f is A below {a}, B from {a} up to but not including {b}, and C from {b} upward.",
    ],
    "lt_and_mod": [
        "Output 1 if x is less than {c} and x modulo {k} equals {r}, otherwise output 0.",
        "Return 1 when x < {c} and x % {k} == {r}, else return 0.",
    ],
    "gt_and_mod": [
        "Output 1 if x is greater than {c} and x modulo {k} equals {r}, otherwise output 0.",
        "Return 1 when x > {c} and x % {k} == {r}, else return 0.",
    ],
}


def _fmt(x) -> str:
    return f"x = {x}"


def _make(family: str, params: dict, compute, candidates, domain) -> FunctionSpec:
    paraphrases = [t.format(**params) for t in TEMPLATES[family]]
    return FunctionSpec(
        name=family + "_" + "_".join(str(v) for v in params.values()),
        arity=1,
        compute=compute,
        candidates=candidates,
        format_input=_fmt,
        sampler=lambda d=domain: random.choice(d),
        true_description=paraphrases[0],
        family=family,
        param=float(next(iter(params.values()))),
        params=dict(params),
        paraphrases=paraphrases,
        domain=domain,
    )


def _build(family: str, p: dict) -> FunctionSpec:
    if family == "gt":
        return _make(family, p, lambda x, c=p["c"]: "1" if x > c else "0", BINARY, NUM_DOMAIN)
    if family == "lt":
        return _make(family, p, lambda x, c=p["c"]: "1" if x < c else "0", BINARY, NUM_DOMAIN)
    if family == "absgt":
        return _make(family, p, lambda x, c=p["c"]: "1" if abs(x) > c else "0", BINARY, NUM_DOMAIN)
    if family == "range":
        return _make(family, p, lambda x, a=p["a"], b=p["b"]: "1" if a < x < b else "0", BINARY, NUM_DOMAIN)
    if family == "mod":
        return _make(family, p, lambda x, k=p["k"], r=p["r"]: "1" if x % k == r else "0", BINARY, NUM_DOMAIN)
    if family == "bit_is_1":
        return _make(family, p, lambda x, i=p["i"]: "1" if x[i] == "1" else "0", BINARY, BIT_DOMAIN)
    if family == "at_least":
        return _make(family, p, lambda x, k=p["k"]: "1" if x.count("1") >= k else "0", BINARY, BIT_DOMAIN)
    if family == "bits_equal":
        return _make(family, p, lambda x, i=p["i"], j=p["j"]: "1" if x[i] == x[j] else "0", BINARY, BIT_DOMAIN)
    if family == "piecewise":
        return _make(
            family, p,
            lambda x, a=p["a"], b=p["b"]: "A" if x < a else ("B" if x < b else "C"),
            ABC, NUM_DOMAIN,
        )
    if family in ("lt_and_mod", "gt_and_mod"):
        cmp = (lambda x, c: x < c) if family == "lt_and_mod" else (lambda x, c: x > c)
        return _make(
            family, p,
            lambda x, c=p["c"], k=p["k"], r=p["r"], cmp=cmp: "1" if cmp(x, c) and x % k == r else "0",
            BINARY, NUM_DOMAIN,
        )
    raise ValueError(family)


def _grids() -> dict[str, list[dict]]:
    compositions = [
        {"c": c, "k": k, "r": r} for c in range(-20, 21, 5) for k, r in [(2, 0), (2, 1), (3, 0)]
    ]
    return {
        "gt": [{"c": c} for c in range(-20, 21)],
        "lt": [{"c": c} for c in range(-20, 21)],
        "absgt": [{"c": c} for c in range(1, 25)],
        "range": [
            {"a": a, "b": b} for a in range(-24, 20, 4) for b in range(a + 4, min(a + 24, 29), 4)
        ],
        "mod": [{"k": k, "r": r} for k in range(2, 6) for r in range(k)],
        "bit_is_1": [{"i": i} for i in range(N_BITS)],
        "at_least": [{"k": k} for k in range(1, N_BITS)],
        "bits_equal": [{"i": i, "j": j} for i in range(N_BITS) for j in range(i + 1, N_BITS)],
        "piecewise": [
            {"a": a, "b": b} for a in range(-20, 16, 4) for b in range(a + 4, 25, 4)
        ],
        "lt_and_mod": compositions,
        "gt_and_mod": compositions,
    }


def build_task_suite() -> dict[str, list[FunctionSpec]]:
    suite: dict[str, list[FunctionSpec]] = {
        "train": [], "heldout_param": [], "heldout_composition": [], "heldout_family": [],
    }
    for family, grid in _grids().items():
        specs = [_build(family, p) for p in grid]
        if family == HELDOUT_FAMILY:
            suite["heldout_family"] += specs
        elif family == HELDOUT_COMPOSITION:
            suite["heldout_composition"] += specs
        else:
            for idx, spec in enumerate(specs):
                # every 4th grid point, interleaved with training values
                (suite["heldout_param"] if idx % 4 == 2 else suite["train"]).append(spec)
    return suite


def draw_support_query(
    spec: FunctionSpec,
    n_support: int,
    n_query: int,
    rng: random.Random,
    balanced_query: bool = False,
) -> tuple[list[tuple], list[tuple]]:
    """Disjoint support/query draws. With balanced_query, the query set has an equal
    count per label (as far as the domain allows), so a constant-label predictor scores
    1/n_labels -- removing the majority-label shortcut the encoder collapsed onto.
    The support set is always a plain random draw."""
    if not balanced_query:
        inputs = rng.sample(spec.domain, n_support + n_query)
        examples = [(inp, spec.compute(*inp)) for inp in inputs]
        return examples[:n_support], examples[n_support:]

    support_inputs = rng.sample(spec.domain, n_support)
    taken = set(support_inputs)
    by_label: dict[str, list[tuple]] = {}
    for inp in spec.domain:
        if inp not in taken:
            by_label.setdefault(spec.compute(*inp), []).append(inp)
    per_label = n_query // len(by_label)
    query_inputs = []
    for label in sorted(by_label):
        pool = by_label[label]
        query_inputs += rng.sample(pool, min(per_label, len(pool)))
    chosen = set(query_inputs)
    leftover = [inp for label in sorted(by_label) for inp in by_label[label] if inp not in chosen]
    query_inputs += rng.sample(leftover, min(n_query - len(query_inputs), len(leftover)))
    rng.shuffle(query_inputs)
    support = [(inp, spec.compute(*inp)) for inp in support_inputs]
    return support, [(inp, spec.compute(*inp)) for inp in query_inputs]


def pair_text(spec: FunctionSpec, inputs: tuple, label: str) -> str:
    return f"{spec.format_input(*inputs)} -> f = {label}"


def incontext_text(spec: FunctionSpec, support: list[tuple]) -> str:
    lines = "\n".join(pair_text(spec, inp, lab) for inp, lab in support)
    return f"Here are input/output pairs of the hidden function f:\n{lines}\n\nNow a new input:\n"


def all_pair_texts(specs: list[FunctionSpec]) -> list[str]:
    texts: set[str] = set()
    seen: set[tuple] = set()
    for spec in specs:
        key = (id(spec.domain), tuple(spec.candidates))
        if key in seen:
            continue
        seen.add(key)
        for inp in spec.domain:
            for cand in spec.candidates:
                texts.add(pair_text(spec, inp, cand))
    return sorted(texts)


def sample_train_task(by_family: dict[str, list[FunctionSpec]], rng: random.Random) -> FunctionSpec:
    """Uniform over families, then uniform within family (families differ a lot in size)."""
    family = rng.choice(sorted(by_family))
    return rng.choice(by_family[family])


# ---------------------------------------------------------------------------
# Semantic parsing of generated descriptions -> (family, params)
# ---------------------------------------------------------------------------

_INT = r"(-?\d+)"
_GT = r"(?:greater than|exceeds|above|>)\s*" + _INT
_LT = r"(?:less than|below|<)\s*" + _INT


def _parse_mod(t: str) -> tuple[int, int] | None:
    m = re.search(r"x\s*(?:modulo|mod|%)\s*(\d+)\s*(?:equals|==|=|is)\s*(-?\d+)", t)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r"remainder\s*(\d+)\s*after division by\s*(\d+)", t)
    if m:
        return int(m.group(2)), int(m.group(1))
    return None


def parse_description(text: str) -> dict | None:
    t = text.lower().replace("\u2212", "-")

    if re.search(r"\bb\b", t) and re.search(r"\bc\b", t):
        ints = [int(v) for v in re.findall(r"-?\d+", t)]
        if ints:
            rest = [v for v in ints[1:] if v != ints[0]]
            if rest:
                return {"family": "piecewise", "params": {"a": ints[0], "b": rest[0]}}

    mod = _parse_mod(t)
    if mod:
        k, r = mod
        m = re.search(_GT, t)
        if m:
            return {"family": "gt_and_mod", "params": {"c": int(m.group(1)), "k": k, "r": r}}
        m = re.search(_LT, t)
        if m:
            return {"family": "lt_and_mod", "params": {"c": int(m.group(1)), "k": k, "r": r}}
        return {"family": "mod", "params": {"k": k, "r": r}}

    if re.search(r"absolute value|magnitude|\|x\|", t):
        m = re.search(_GT, t)
        if m:
            return {"family": "absgt", "params": {"c": int(m.group(1))}}

    m = re.search(r"between\s*" + _INT + r"\s*and\s*" + _INT, t) or re.search(
        _INT + r"\s*<\s*x\s*<\s*" + _INT, t
    )
    if m:
        return {"family": "range", "params": {"a": int(m.group(1)), "b": int(m.group(2))}}

    m = re.search(r"(?:bits|positions)\s*(\d+)\s*and\s*(\d+)", t) or re.search(
        r"x\[(\d+)\]\s*==\s*x\[(\d+)\]", t
    )
    if m:
        return {"family": "bits_equal", "params": {"i": int(m.group(1)), "j": int(m.group(2))}}

    m = (
        re.search(r"at least\s*(\d+)\s*ones", t)
        or re.search(r"number of 1s in x is\s*>=\s*(\d+)", t)
        or re.search(r"(\d+)\s*or more ones", t)
    )
    if m:
        return {"family": "at_least", "params": {"k": int(m.group(1))}}

    m = re.search(r"(?:bit|position)\s*(\d+)", t) or re.search(r"x\[(\d+)\]\s*==\s*1", t)
    if m:
        return {"family": "bit_is_1", "params": {"i": int(m.group(1))}}

    m = re.search(_GT, t)
    if m:
        return {"family": "gt", "params": {"c": int(m.group(1))}}
    m = re.search(_LT, t)
    if m:
        return {"family": "lt", "params": {"c": int(m.group(1))}}
    return None


def score_parse(spec: FunctionSpec, parsed: dict | None) -> dict:
    if parsed is None or parsed["family"] != spec.family:
        return {"parsed": parsed is not None, "family_correct": False, "exact": False, "param_abs_err": None}
    errs = [abs(parsed["params"].get(k, 0) - v) for k, v in spec.params.items()]
    return {
        "parsed": True,
        "family_correct": True,
        "exact": parsed["params"] == spec.params,
        "param_abs_err": sum(errs) / len(errs),
    }
