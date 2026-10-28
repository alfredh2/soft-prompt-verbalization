"""Rebuild a record's function from the generator that produced it.

Records from different generators share one schema, but a function can only be rebuilt by
the code that defined it. The scorer needs to do that (to get the whole domain, the
held-out inputs and a shuffled example pool), and it should not need to know which
generator a record came from.
"""

from __future__ import annotations


def build_function(record: dict, spec=None):
    if record.get("dataset") == "compositional":
        from src.datasets.compositional import function_from_params
        return function_from_params(record["params"], record["domain_spec"])
    from src.datasets.discrete import DomainSpec, make_function
    return make_function(record["family"], record["params"], spec or DomainSpec())


def teaching_pool(record: dict, fn, pool: list):
    """Pool reordered so its prefix identifies the function. Compositional tier only: it needs
    the enumerable class of candidate behaviours that the single-rule tier does not expose."""
    if record.get("dataset") != "compositional":
        raise ValueError("teaching support is only defined for the compositional tier")
    from src.datasets.compositional import teaching_pool as _teach
    return _teach(fn, pool, record["domain_spec"])
