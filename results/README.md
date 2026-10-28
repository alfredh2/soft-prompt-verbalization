# Run artifacts

One directory per run: `results/<method>/<tag>/`. Committed runs are the ones worth
comparing; smoke and scratch runs are gitignored.

| file | contents |
|---|---|
| `descriptions.jsonl` | one line per record: every description the method produced, next to the true rule |
| `scores.json` | the numbers, the exact arguments used, and parse diagnostics |

Reproduce a run's scoring without regenerating any descriptions:

```bash
python scripts/run_baseline.py --method iterative --tag qwen35_9b --skip-propose
```

## `descriptions.jsonl`

```json
{
  "function_id": "gt_c=-14", "view": 2, "n_examples": 8,
  "family": "gt", "split": "train",
  "truth": "Output 1 if x is greater than -14, otherwise output 0.",
  "expression": "1 if x > -14 else 0",
  "description": "The function returns 1 if the input is positive and 0 otherwise.",
  "rounds": ["...", "...", "...", "..."]
}
```

`description` is the one that was scored — the final round. `rounds` holds every
intermediate description, so any round can be re-scored later without re-running the
proposer. One-shot runs have a single entry.

**This is the file to read first when a number looks wrong.** Fidelity tells you a
description failed; only the text tells you how. A rule can be fluent, nearly correct and
still score badly — `gt_c=-14` above put the threshold at 0 instead of -14, which is a
boundary error, not a misunderstanding of the rule's shape.

## `scores.json`

| field | meaning |
|---|---|
| `overall` | the headline numbers (below) |
| `by_split` / `by_family` / `by_n_examples` | the same numbers re-aggregated; all from one set of predictions, not separate runs |
| `per_record` | per-record accuracy for every condition — what paired significance tests need |
| `parse_diagnostics` | see below; decides whether the run is valid at all |
| `propose_stats` | how often the proposer failed to emit a `Rule:` line |
| `subset_id` | hash of the evaluated records; runs with different ids are not comparable |
| `args` | every argument used, so a run can be repeated exactly |

### Conditions inside `overall`

| key | meaning |
|---|---|
| `constant` | always predict the function's most common output. Free, computed from the dataset, no model |
| `bare` | the executor with no description at all — its answer bias, **not** chance |
| `baseline` / `round1..N` | the method under test |
| `oracle` | the executor given the true rule — the ceiling on any description |
| `normalized_*` | `(method - bare) / (oracle - bare)` |

**Quote `normalized` with its zero point.** It uses `bare`, which is the noisiest number in
the file — about a quarter of its predictions come from the parse fallback, because there
is no rule to apply. Normalizing against `constant` instead is stricter and exact: baseline
2 is 0.568 against `bare` but 0.318 against `constant`. Both are defensible; they tell
different stories, so name which one you used.

`bare` and `oracle` depend only on the subset, not on the method, so they can be shared
across runs with `--reference-from <other scores.json>` (it refuses on a `subset_id`
mismatch). Runs that compute them separately differ slightly — 0.484 vs 0.488 across the
two committed baselines.

### `parse_diagnostics` — check this before trusting anything else

```json
{"round4": {"rate": {"round4": 0.0017}}, "references": {"rate": {"bare": 0.251, "oracle": 0.0014}}}
```

`rate` is the fraction of predictions that came from the fallback parse rather than an
explicit `Answer:` line. A truncated generation yields a stray label that scores like a
coin flip while looking like a real prediction.

- **`oracle` and the method conditions must be ~0.** Above roughly 0.01 the run is not
  valid — check the executor's token budget first.
- **`bare` is expected to be high** (~0.25). With no rule to apply the model equivocates.
  This is the reason `bare` is a noisy floor.

A high rate here is a harness fault, not a model result. Reporting 0.959 for a model that
scores 0.997 was caused by exactly this, before the diagnostic existed.
