"""How much do N examples determine the function, within the generator's own DSL?

A learner that knows the eleven rule families and checks every one against the shown
examples is the best a method can do. When several distinct rules fit the sample, nothing
distinguishes them, so exact recovery has a limit below 1.0 that is not a model failure.

Two reference lines come out of this, and they are not the same:

  identifiable   the examples leave exactly one rule standing. A strict limit for a
                 method with no prior over the rules.
  guess 1/k      pick uniformly among the k rules that fit. The expected score of a
                 learner that knows the DSL but has no preference inside it -- the fair
                 yardstick, since a model that favours round thresholds can beat
                 `identifiable` without any rule being forced.

Rules are compared as truth tables over the whole domain, so x > 10 and x >= 11 count as
one rule, which is what exact recovery means.
"""

from __future__ import annotations

from src.datasets.discrete import FAMILIES, DomainSpec


def hypothesis_tables(spec: DomainSpec) -> tuple[dict[str, list[tuple]], dict[str, dict]]:
    """Every distinct truth table per domain, and each input's index within its table."""
    tables: dict[str, list[tuple]] = {}
    positions: dict[str, dict] = {}
    for dom in ("int", "bits"):
        values = spec.values(dom)
        positions[dom] = {x: i for i, x in enumerate(values)}
        seen = set()
        for fam in FAMILIES:
            if fam.domain != dom:
                continue
            for params in fam.grid(spec):
                rule = fam.build(params)
                seen.add(tuple(rule(x) for x in values))
        tables[dom] = list(seen)
    return tables, positions


def n_consistent(tables: list[tuple], positions: dict, shown: list) -> int:
    """How many distinct rules agree with every shown (input, label) pair."""
    return sum(all(t[positions[x]] == y for x, y in shown) for t in tables)


def mask_consistent_count(masks: list[int], positions: dict, shown: list) -> int:
    """How many distinct BEHAVIOURS agree with every shown (input, label) pair.

    For classes too large to hold as tuples, a behaviour is a bitmask of its outputs over
    the whole domain. Each shown pair fixes one bit, so agreement is two integer
    operations per candidate: the behaviour restricted to the shown inputs must equal the
    labelled ones. Because masks are keyed by behaviour, syntactically different rules
    that behave identically are already one candidate -- which is the right unit, since
    exact recovery means agreeing everywhere, not matching a spelling.
    """
    seen = labelled = 0
    for x, y in shown:
        bit = 1 << positions[x]
        seen |= bit
        if y == "1":
            labelled |= bit
    return sum(1 for m in masks if m & seen == labelled)
