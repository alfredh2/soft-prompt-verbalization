"""Shared evaluation for every description-producing method.

Each method -- one-shot, iterative refinement, soft-prompt decoding -- ends with an
English description, and all of them must be scored the same way or their numbers are
not comparable. This module owns the parts that must not vary between methods: which
records are evaluated, how a description is turned into held-out predictions, and how
those predictions are aggregated.

`src/evaluation.py` previously held nothing and was deleted upstream; this is its
replacement, with content.
"""

from __future__ import annotations

import hashlib
import random
import zlib
from collections import defaultdict

from src.datasets.discrete import DomainSpec, holdout_inputs
from src.datasets.registry import build_function, teaching_pool
from src.executor import Executor, Item


def subset(records: list[dict], n: int, seed: int) -> list[dict]:
    """Deterministic stratified sample over (family, split, n_examples).

    A pure function of (dataset, n, seed) -- never a fresh random draw per run -- so
    every method is scored on byte-identical records.
    """
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        buckets[(r["family"], r["split"], r["n_examples"])].append(r)
    keys = sorted(buckets)
    rng = random.Random(seed)
    for k in keys:
        buckets[k].sort(key=lambda r: (r["function_id"], r["view"]))
        rng.shuffle(buckets[k])

    chosen: list[dict] = []
    i = 0
    while len(chosen) < min(n, len(records)):
        progressed = False
        for k in keys:
            if i < len(buckets[k]) and len(chosen) < n:
                chosen.append(buckets[k][i])
                progressed = True
        if not progressed:
            break
        i += 1
    chosen.sort(key=lambda r: (r["function_id"], r["view"], r["n_examples"]))
    return chosen


def subset_id(records: list[dict]) -> str:
    key = "|".join(f'{r["function_id"]}#{r["view"]}#{r["n_examples"]}' for r in records)
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def limit_evenly(records: list[dict], limit: int) -> list[dict]:
    """Take an even spread, not a prefix: records are sorted by function_id, so a head
    slice would be a single family and tell you nothing about the whole set."""
    if not limit:
        return records
    step = max(1, len(records) // limit)
    return records[::step][:limit]


def score_descriptions(
    executor: Executor,
    records: list[dict],
    descriptions: list[str],
    batch_size: int,
    *,
    field: str = "holdout_examples",
    tag: str = "method",
) -> tuple[list[float], dict]:
    """Per-record accuracy of one description per record, on the named example field.

    Returns (per_record_accuracy, parse_diagnostics). Held-out inputs are the default;
    passing `field="examples"` scores the support set instead, which is what the
    iterative baseline needs for feedback -- it must never read held-out pairs.
    """
    items = []
    for rec, desc in zip(records, descriptions):
        for x, _y in rec[field]:
            items.append(Item(desc or "(empty)", f"x = {x}", list(rec["labels"]), tag=tag))
    preds, diag = executor.predict(items, batch_size=batch_size)

    per_record, cursor = [], 0
    for rec in records:
        hits = []
        for _x, y in rec[field]:
            hits.append(preds[cursor] == y)
            cursor += 1
        per_record.append(sum(hits) / len(hits))
    assert cursor == len(preds), "prediction/item misalignment"
    return per_record, diag.as_dict()


def predict_references(
    executor: Executor, records: list[dict], batch_size: int
) -> tuple[dict[str, list[float]], dict]:
    """`bare` (no rule) and `oracle` (the true rule) on held-out inputs.

    Neither depends on the method, so a run may reuse these from an earlier run over the
    same subset instead of paying for them again.
    """
    items = []
    for rec in records:
        for cond in ("bare", "oracle"):
            rule = None if cond == "bare" else rec["description"]
            for x, _y in rec["holdout_examples"]:
                items.append(Item(rule, f"x = {x}", list(rec["labels"]), tag=cond))
    preds, diag = executor.predict(items, batch_size=batch_size)

    out: dict[str, list[float]] = {"bare": [], "oracle": []}
    cursor = 0
    for rec in records:
        for cond in ("bare", "oracle"):
            hits = []
            for _x, y in rec["holdout_examples"]:
                hits.append(preds[cursor] == y)
                cursor += 1
            out[cond].append(sum(hits) / len(hits))
    assert cursor == len(preds), "prediction/item misalignment"
    return out, diag.as_dict()


def aggregate(per_record: dict[str, list[float]], records: list[dict], idxs: list[int]) -> dict:
    """Mean over the named records, plus the free constant-predictor baseline.

    `normalized` places a method between the no-description floor and the true-rule
    ceiling, because raw fidelity alone cannot say whether a score is any good.
    """
    out = {c: sum(v[i] for i in idxs) / len(idxs) for c, v in per_record.items()}
    out["constant"] = sum(records[i]["majority_label_fraction"] for i in idxs) / len(idxs)
    if "oracle" in out and "bare" in out:
        denom = out["oracle"] - out["bare"]
        for key in [c for c in out if c not in ("bare", "oracle", "constant")]:
            out[f"normalized_{key}"] = (out[key] - out["bare"]) / denom if denom > 0 else float("nan")
    out["n"] = len(idxs)
    return out


def breakdown(per_record: dict[str, list[float]], records: list[dict], key: str) -> dict:
    groups: dict[object, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        groups[r[key]].append(i)
    return {
        str(g): aggregate(per_record, records, idxs)
        for g, idxs in sorted(groups.items(), key=lambda kv: str(kv[0]))
    }


def pick_functions(records: list[dict], n: int) -> list[dict]:
    """One record per distinct function, spread evenly over the sorted set.

    A curve over example counts needs distinct functions, not repeated views of one, and
    an even stride (rather than a prefix) keeps every family represented. Every
    experiment that is compared against another must call this, or they are scored on
    different functions without anyone noticing.
    """
    uniq: dict[str, dict] = {}
    for r in records:
        uniq.setdefault(r["function_id"], r)
    vals = list(uniq.values())
    return vals[:: max(1, len(vals) // n)][:n]


def pool_for(record: dict, spec: DomainSpec | None = None,
             support: str = "random") -> dict:
    """Rebuild a function and its example pool: shuffled once, nested, with held-out apart.

    The dataset ships 8/16/32-example views, but a sample-complexity curve needs any size
    up to the whole non-held-out domain, so the function is reconstructed from
    (family, params). The pool is shuffled once under a per-function seed and a method
    shown N examples sees `pool[:N]`:

      random   the unshuffled pool is in domain order, so `pool[:32]` was the 32 lowest
               inputs. For gt_c=17 all of them sat below the threshold, every label was 0,
               and `return '0'` was consistent with everything shown.
      nested   larger N strictly contains smaller N, so successive points on a curve
               differ only by added information.
    """
    fn = build_function(record, spec or DomainSpec())
    # Records from a generator with its own held-out size carry it; the single-rule tier
    # predates the field and always reserved 16, so its pools are unchanged.
    reserved = set(holdout_inputs(fn, record.get("n_holdout", 16)))
    pool = [(x, fn.compute(x)) for x in fn.values if x not in reserved]
    random.Random(zlib.crc32(record["function_id"].encode())).shuffle(pool)
    out = {**record, "fn": fn, "pool": pool,
           "held_out": [(x, fn.compute(x)) for x in fn.values if x in reserved]}
    if support == "teaching":
        out["pool"], out["identified_at"] = teaching_pool(record, fn, pool)
    elif support != "random":
        raise ValueError(f"unknown support mode {support!r}")
    return out


def select_functions(records: list[dict], n: int, spec: DomainSpec | None = None,
                     support: str = "random", identifiable_only: bool = False) -> list[dict]:
    """The functions an experiment runs on. Every script that is compared to another must
    call this, so they are scored on the same functions with the same pools.

    `identifiable_only` keeps just the functions whose teaching prefix identifies them from
    the non-held-out inputs, then strides over THAT list. Taking the first n would pick the
    alphabetically earliest ids and with them one op.
    """
    chosen = pick_functions(records, 10**6 if identifiable_only else n)
    built = [pool_for(r, spec, support=support) for r in chosen]
    if identifiable_only:
        built = [b for b in built if b.get("identified_at") is not None]
        built = built[:: max(1, len(built) // n)][:n]
    return built
