# Benchmark and results in detail
Full write-up of the benchmark harness and every result in it. For the short version see [`findings.md`](findings.md).

## Datasets

Tiers of target function, generated separately and kept separate — they use
different domains, output types and conventions. Each later tier exists because the one
before was too easy to tell methods apart.

| | module | domains | outputs |
|---|---|---|---|
| **Discrete** | [`src/datasets/discrete.py`](../src/datasets/discrete.py) | integers in `[-30, 30]`, 6-bit strings | class labels (`"0"`/`"1"`, `"A"`/`"B"`/`"C"`) |
| **Compositional** | [`src/datasets/compositional.py`](../src/datasets/compositional.py) | integers in `[-150, 150]` | class labels (`"0"`/`"1"`) |

The discrete dataset is generated and committed: **15,336 records over 1,278 rules**, each
a list of observed `(x, y)` pairs paired with the rule's structured identity and its
English description. See [`data/discrete/README.md`](../data/discrete/README.md) for the
schema, the splits, and the design decisions behind them.

```bash
python scripts/generate_dataset/discrete.py --out data/discrete/discrete_functions.jsonl.gz
```

```python
from src.datasets.discrete import load_dataset
records = load_dataset("data/discrete/discrete_functions.jsonl.gz")
```

The **compositional** tier is 160 depth-2 rules `A op B`, with atoms `x > c`, `x < c`,
`|x| > c`, `x % k == r` and ops *and* / *or* / *and-not*, over a wide domain. The single-rule
tier is too easy for a frontier model, and most of its remaining error is not the model's:
with 61 inputs only 47% of functions are uniquely determined by 45 examples. This tier
keeps the candidate rules enumerable, so how much the examples determine can be
**computed** instead of assumed.

A rule's identity is its **behaviour** — a bitmask of its outputs over the whole domain —
not its spelling, so `x > 5 and x > 3` and `x > 5` are one rule. 792 atoms give 188,011
distinct behaviours, 138,031 eligible as targets. Records use the same schema as the
single-rule tier plus the fields a record needs to rebuild its own function, so the
same scorer and scripts run on either.

It is a controlled testbed, not a frontier-failure benchmark: told the grammar, a frontier
model recovers everything the examples determine (see Results). It exists to ask whether
optimising a soft prompt makes better use of local compute than prompting a model of the
same size.

```bash
python scripts/generate_dataset/compositional.py --out data/compositional/compositional_functions.jsonl.gz
```

## Baselines and evaluation

A "method" here is anything that turns observed behaviour into an English description. Each
one is judged **behaviourally**: hand its description to a model that has never seen the
function, and see whether that model reproduces the function on held-out inputs. A
description that sounds right but does not reproduce the behaviour is not a good
description.

| method | module | description comes from |
|---|---|---|
| **Baseline 1** — one-shot | [`src/baselines/oneshot.py`](../src/baselines/oneshot.py) | the examples, one attempt |
| **Baseline 2** — iterative | [`src/baselines/iterative.py`](../src/baselines/iterative.py) | the examples, then revised from its own mistakes |
| the executor | [`src/executor.py`](../src/executor.py) | *not* a method — the instrument every method is scored with |
| shared scoring | [`src/scoring.py`](../src/scoring.py) | subset selection, held-out scoring, aggregation |

```bash
python scripts/run_baseline.py --method oneshot   --tag qwen35_9b
python scripts/run_baseline.py --method iterative --tag qwen35_9b --rounds 3
python scripts/run_baseline.py --method oneshot   --tag qwen35_9b --skip-propose   # rescore only
```

### How a record is scored

Each record carries `examples` (what the method sees) and `holdout_examples` (what it is
tested on). For one record:

1. the method reads `examples` and writes one sentence
2. the executor is given that sentence plus **one** held-out input, once per held-out input
3. the record's fidelity is the fraction of its held-out inputs answered correctly
4. the reported number is the mean over all records in the subset

One input per prompt, never a list. Asking for every output in one reply makes the model
keep a list aligned, and a single off-by-one error is then scored as a wrong answer for
every input in that record. Prompts are batched on the GPU for speed, so this costs
nothing.

### Three rules that keep the numbers honest

**Support and held-out never mix.** A method only reads `examples`. `holdout_examples` is
touched once, at final scoring. Baseline 2's feedback lists mistakes on the *support* set
for the same reason — revising against held-out pairs would fit the test inputs.

**Every method is scored on byte-identical records.** `subset()` is a pure function of
(dataset, n, seed), stratified over family x split x example-count. Each run records a
`subset_id`, and comparing across different subsets is refused rather than silently wrong.

**Parse health is reported, not assumed.** The executor answers on a final
`Answer: <value>` line. When that line is missing, a fallback reads the last bare label in
the text — which on binary labels is close to a coin flip wearing the costume of a
prediction. Every run reports how often the fallback fired; a rate above ~0.01 on a real
condition invalidates the run. This guard exists because that exact artifact once made a
model look 0.04 worse than it was.

### Reading a result

Raw fidelity alone cannot say whether a score is good, so every run also reports the floor
and the ceiling:

| condition | meaning |
|---|---|
| `constant` | always predict the function's most common output — free, no model involved |
| `bare` | the executor with **no** description: its answer bias, not chance |
| `oracle` | the executor given the **true** rule — the most any description could score |
| `normalized` | `(method - bare) / (oracle - bare)`: the share of the reachable gap closed |

`oracle` is the important one. It was 0.635 on an earlier executor, meaning a *perfect*
description could score no better than that, and every method was squashed into a narrow
band. At 0.996 the instrument is no longer the limiting factor, and a low score can be
attributed to the description.

### Results, 220 records, Qwen3.5-9B proposer, Qwen3.5-4B executor

| | fidelity | normalized |
|---|---|---|
| `constant` | 0.672 | — |
| `bare` | 0.484–0.488 | — |
| **Baseline 1** | **0.749** | 0.519 |
| **Baseline 2** (round 4) | **0.775** | 0.568 |
| `oracle` | 0.996 | — |

Baseline 2's round 1 is baseline 1 by construction — same proposer, same prompt — and
scored 0.749 against baseline 1's 0.749. That is a consistency check on the harness, not a
coincidence. Refinement's +0.026 is real (paired bootstrap 95% CI `[+0.005, +0.046]`) but
nets two opposing effects: 56 records improved, 39 got worse, 125 were untouched.

`bare` differs by 0.004 between the two runs because each computed it independently; pass
`--reference-from` to share one set of reference numbers across runs.

Per-run artifacts and field definitions: [`results/README.md`](../results/README.md).

## Program recovery and sample complexity

The English baselines are scored by an executor model, which caps the ceiling at about
0.996 and adds parse noise. Asking for a Python function instead removes it: the function
is *run*, so the ceiling is exactly 1.0, scoring is free and deterministic, and prose
ambiguity ("positive", "between 13 and 20") disappears. This is the clean measuring stick,
not the deliverable — the project's claim is about natural-language descriptions, and these
numbers measure program recovery.

```bash
python scripts/learning_curve.py --model openai:gpt-6.1-sol --dry-run     # estimate, no calls
python scripts/learning_curve.py --model openai:gpt-6.1-sol --records 20 --max-usd 40 \
    --price-in 2 --price-out 10
python scripts/learning_curve.py --model Qwen/Qwen3.5-9B                  # a local model
python scripts/learning_curve.py --dataset data/compositional/compositional_functions.jsonl.gz \
    --n 100000 --support teaching --identifiable-only --order sorted --grammar \
    --model openai:gpt-6.1-sol --sizes 8,12,16,32 --records 40
python scripts/plot_learning_curve.py --curve name=path/to/curve.json
```

Spend is guarded in code, since the account has no dashboard limit: `--dry-run` makes no
calls, `--max-calls` and `--max-usd` refuse to start a plan whose worst case exceeds them,
and `--max-usd` is also checked against actual usage after every size.

### Five metrics, and why exact is the headline

Each is computed from one execution over the whole domain. On the true rule `x > 10`
answered as `x >= 12`:

| metric | meaning | that answer |
|---|---|---|
| `exact` | agrees with the true rule on **every** input | **False** |
| `domain_acc` | fraction of the domain right (the continuous version of `exact`) | 0.984 |
| `held_out` | accuracy on the reserved inputs; the old flattering metric | 0.938 |
| `probe` | inputs chosen to falsify near-misses (single-rule tier only) | 0.800 |
| `support` | accuracy on the examples it was shown | 1.000 |

An off-by-one rule looks near-perfect on four of five. A snippet can fit its examples and
still be the wrong function: `support` was at least 0.997 for the hosted models at every size
while exact recovery at two examples was 0.00.

### Two ways of choosing the examples

**Random support** is a nested random sample: the pool is shuffled once per function and a
method shown N examples sees `pool[:N]`, so larger N strictly contains smaller. Ambiguity is
part of the task. (An earlier version took the N *lowest* inputs, so for `gt_c=17` every
shown label was 0 and `return '0'` was consistent; that confounded the first curves.)

**Teaching support** (`--support teaching`, compositional tier) starts with inputs a greedy
teacher chose to remove the most rival behaviours of the true one, so once N reaches a
function's identification size the examples determine it uniquely. It does so in 5–9
examples (median 7). 120 of 160 functions can be identified from the non-held-out inputs
(modular pairs 40/40, mixed 45/60, threshold pairs 35/60, so the subset over-represents
modular rules); `--identifiable-only` keeps those. `scripts/audit_teaching_support.py`
rebuilds the hypothesis class independently and checks the construction: no support input is
held out, labels are the truth, the greedy order is unchanged when every held-out label is
flipped, and `identified_at` is exactly the first N at which one behaviour remains.

Two things this does **not** mean, each with its own control:

- **"Identifiable" is relative to the generator's grammar, which the model is never told.**
  Six examples pin down `x%5==1 or x%7==4` only to a learner that already knows the family.
  `--grammar` tells the model the form of the rule (which conditions exist, how two combine,
  the domain) and nothing about any particular function, to separate "failed to infer" from
  "did not know the grammar".
- **The examples come out in the teacher's order**, most informative first, so the order is a
  function of the answer. `--order sorted|shuffled` changes how they are printed; scoring is
  order-blind. Sorting *helps* (below), so the order was not leaking anything.

### Reference lines — what the examples leave open

`scripts/identifiability.py` counts how many distinct behaviours fit the examples shown:

- **identifiable** — exactly one fits.
- **guess 1/k** — pick uniformly among the k that fit. A yardstick, not a ceiling: a model
  that prefers simpler rules can beat it without information the examples lack.
- **bayes** — the chance the most likely consistent behaviour is the target under the
  generator's own distribution; the best any method can do from the examples plus that prior.

Exact recovery cannot beat the Bayes line, so a gap to 1.0 below it is ambiguity, not the
model. The reference and the curve must run on the same functions: `identifiability.py`
selects through the same `subset()` as `learning_curve.py` (an earlier version did not and
shared only 7 of 40 functions with its curve).

Random supports cannot be made identifiable by widening the domain. Pinning a threshold
needs both `x = c` and `x = c + 1` shown, which has probability about f² when a fraction f
of the domain is shown, so 0.9 needs about 95% of it.

### Results

Single-rule tier, 20 functions, random support, exact recovery:

```
examples          2     4     8    16    24    32    40    45
gpt-6.1-sol    0.00  0.00  0.10  0.35  0.45  0.40  0.55  0.63
gpt-5-mini     0.00  0.00  0.10  0.25  0.35  0.40  0.45  0.53
Qwen3.5-9B     0.00  0.00  0.00  0.05  0.05  0.05  0.05  0.05
guess 1/k      0.02  0.09  0.15  0.39  0.45  0.50  0.62  0.72
```

The hosted models sit close to the 1/k line and fail at *identification*: support is at least
0.997 at every size and nearly every miss is a boundary off by one. Qwen3.5-9B fails a step earlier —
support 0.63–0.90, falling as examples increase, because 106 of 159 snippets contradict
examples in their own prompt — and is flat at ~0.05.

Compositional tier, gpt-6.1-sol, 40 functions, **random support**:

```
N         16    32    64    96   128   192
exact    0.05  0.20  0.30  0.35  0.40  0.625
bayes    0.16  0.25  0.35  0.40  0.45  0.64
```

From N = 32 the model reaches 79–98% of the best any method could. Random support leaves
almost no headroom on this tier either.

Compositional tier, gpt-6.1-sol, 40 identifiable functions, **teaching support**, exact recovery:

```
N                8     12     16     32
plain, greedy   0.15  0.25   0.525  0.825
plain, sorted   0.325 0.425  0.60   0.85
+ grammar       0.90  1.00   1.00   1.00
Bayes limit     0.92  1.00   1.00   1.00
```

- **With the grammar the model sits at the Bayes limit.** No function was lost at any size;
  30, 30, 19 and 7 were gained at N = 8, 12, 16 and 32. The gap at N = 12–16 was a missing
  prior, not failed inference, so this tier is not a frontier-failure benchmark once the
  hypothesis class is known.
- **Sorted order is the fair plain baseline.** Pooled over the four sizes sorted wins on 27
  function-size pairs and greedy on 9 (sign test p = 0.004); it helps at N = 8–16 and makes no
  difference at 32. The teacher's order was a handicap, not a leak: had it carried hidden
  information, sorting would have hurt.
- N = 6 and 64 (first run, greedy order): 0.10 and 0.975, with 0.28 and 1.00 of functions
  determined. The greedy-order points at 12, 16 and 32 are from a rerun at a 16000-token
  output budget; the first run's 4000 ran out on 2, 4 and 3 calls (reasoning models spend it
  on hidden reasoning before emitting text) and reproduced at 16 and 32.

Every point rests on 20 or 40 functions and carries roughly ±0.15. Temperature is left at
the API default, so a single run is stochastic: gpt-5-mini scored 0.684 and then 0.526 at
45 examples on identical input. Trends hold; individual points should not be quoted without
repeats (`--repeats`).

### Search and verify does not rescue a weak model

Sampling 16 candidates from Qwen3.5-9B and keeping those consistent with the examples
(`scripts/search_probe.py`) finds a consistent candidate for 55–70% of functions, yet only
3 of 214 consistent candidates are the true rule, and the true function appears among the 16
samples for at most 1 function in 20. Selection is not the bottleneck; generation is, so no
re-ranking helps. The consistent candidates are lookup tables — enumerating the shown
positives passes a support check by construction.

## Soft prompts

A soft prompt `z` is a block of trainable embedding vectors spliced into a frozen model's
input. Optimising only `z` so that `LLM(x, z)` reproduces a target function gives a
continuous representation of that function's behaviour -- the object the project tries to
translate back into English, and the reason a description method can be compared against
something other than another description method.

```bash
python scripts/build_zbank.py --out results/zbank/q3_4b --n 120 --seeds 2
```

[`src/soft_prompt.py`](../src/soft_prompt.py) holds the carrier (the frozen model `z` is
optimised against) and [`src/zbank.py`](../src/zbank.py) the optimisation and the bank. Both
read the same dataset records and use the same `subset()` as the baselines, so a bank and
a baseline run can cover the same functions.

Three settings there are measured rather than chosen: `lr 0.03` over 500 steps (0.3 is
unstable, 0.1 plateaus, and accuracy is still climbing at step 300), `bfloat16` (matches
float32 on this path over three seeds and eight tasks, and is what lets a 4B carrier fit
on one card), and Qwen3-4B rather than Qwen3.5-4B (Qwen3.5's linear attention exceeds
32 GB in the backward pass even at one z per batch).

**Soft-prompt accuracy is not measured the same way as description fidelity.** `z` is
scored with a single forward pass over the label logits, because that is what is
differentiable; descriptions are scored by the executor with chain-of-thought. The two
numbers belong in one table only with that difference stated.

**A bank is bound to its carrier.** `z` lives in one model's embedding space at one soft
token count, so changing either invalidates every `z` already fitted.

### The gate: does optimising `z` solve what prompting could not?

`scripts/zbank_gate.py` fits a soft prompt for each (function, N) on the same functions and
nested example pools as the learning curves, then queries it on **every** input in the
domain to get a truth table, scored like an executed snippet. A second seed at a few sizes
asks whether two soft prompts for one function implement the same behaviour — earlier work
on a 1.5B carrier found independently optimised `z`'s for the *same* function no closer in
weight space than `z`'s for different functions, so decoding the vectors is hopeless unless
the *behaviour* is canonical.

```bash
python scripts/zbank_gate.py --dry-run                      # run count and time estimate
python scripts/zbank_gate.py                                 # single-rule tier, random support
python scripts/zbank_gate.py --dataset data/compositional/compositional_functions.jsonl.gz \
    --n 100000 --support teaching --identifiable-only --records 40 --sizes 8,12,16,32 \
    --seeds 2 --repeat-sizes 12,16 --lr-decay cosine
```

Single-rule tier, random support, Qwen3-4B carrier, 20 functions, 500 steps at a constant rate:

```
N              8     16     32     45
exact         0.00  0.05   0.20   0.21
domain acc    0.81  0.845  0.900  0.937
held-out      0.745 0.775  0.804  0.839
support       1.00  1.00   0.977  0.973
```

Measured against prompting: Qwen3.5-9B scores exact at most 0.053 and domain accuracy about 0.65,
the frontier models 0.63 exact at 45 examples (different carriers, so this is indicative).
The soft prompt interpolates well but does not extrapolate the rule. Two soft prompts for the
same function agree on about 92% of inputs but give identical tables only 20–42% of the time
(second seed at N = 16, 32, 45), so the behaviour is largely, not fully, canonical.

Support falling below 1.0 at N ≥ 24 meant the optimiser was not fitting its own examples. At a
constant rate Adam orbits its optimum; `--lr-decay cosine` anneals the rate to zero and brings
support to 1.000 on all 20 functions at N = 32 in the same 500 steps. It did not improve exact
recovery (0.15 against 0.20, inside the noise): fitting the examples perfectly does not buy
the rule.

Whatever it finds, exact recovery cannot beat the identifiability limit: `z` learns only
from the N examples. The claim a gate can support is about cost or model size, not about
recovering more functions than a frontier prompt.

#### Composition tier: soft prompt against same-size prompting

Qwen3-4B throughout, 40 identifiable functions, teaching support printed in sorted order, the
grammar in every prompted arm, domain accuracy over all 301 inputs. The **constant-label
floor** (always output the function's majority label) is 0.710 and is the line any method
must clear before it has learned anything. `scripts/compare_composition_gate.py` reproduces
this table and the tests from the stored results.

```
N                          8      12     16     32
direct code reply         0.534  0.599  0.554  0.558
thinking + self-check     0.678  0.711  0.732  0.670   15-21 s/task
soft prompt               0.640  0.709  0.747  0.796   ~26 s/task
soft prompt exact         0/40   0/40   0/40   0/40
thinking exact            5/40   5/40   7/40   1/40
```

The thinking arm (`--think-tokens`, `--check-rounds`, `src/baselines/reasoned.py`) is the fair
local baseline: the same model allowed to reason, then to run its function on the *shown*
examples and fix disagreements. Held-out inputs and the true rule never enter a prompt. Cost
is measured (`compute.json`), not assumed. A soft prompt costs about 26 s per function
(500 steps, 8 fitted at once in 205 s).

- **Against a direct code reply the soft prompt wins at every size**, but that baseline is
  below the floor, so it is a low bar. It is the comparison to distrust.
- **Against thinking with a self-check the two are tied at N = 8-16.** Mean differences are
  within 0.04 and every bootstrap interval spans zero.
- **At N = 32 the soft prompt leads by +0.126** (95% CI +0.055 to +0.194, sign test p = 0.006).
  Raising the thinking budget to 27.6 s/task, slightly above the soft prompt's, left the
  baseline at 0.684 and the lead at **+0.112 (CI +0.041 to +0.182, sign test p = 0.08)**: the
  mean gap is clear, but the win is not across most functions (26 to 14). Every thinking run
  exhausted its budget and was forced to answer, and only 7-30% wrote code consistent with all
  its own shown examples. A 4096-token pilot (46.6 s/task, 16 functions) did no better.
- **Neither method clears the floor at N <= 12**, and the soft prompt is below it at N = 8
  (13 of 40 functions above, p = 0.04). At N = 32 only the soft prompt clears it (31 of 40,
  p = 0.001); the thinking arm sits below it.
- **No soft prompt recovers a rule.** Domain accuracy rises while exact recovery is zero in
  all 240 fits: the soft prompt approximates the behaviour without extrapolating the rule.
  Two soft prompts for one function agree on 82-83% of inputs and never give identical tables
  (N = 12, 16), less canonical than on the single-rule tier.

What this supports: on a 4B carrier, optimising `z` beats compute-matched reasoning by about
0.11 domain accuracy once there are around 32 examples, ties it below that, and is no use for
recovering rules. It does not support "better use of local compute" in general, and the 4B
model is weak enough that most of the curve sits at the majority-label floor. The soft prompt
also never sees the grammar, which the prompted arms do, so the comparison favours prompting.

This tier is closed. Told the grammar, the frontier model recovers everything the examples
determine, so it cannot show a frontier-prompting failure, and the next benchmark has to
leave headroom for a grammar-aware frontier prompt.

### Behaviour benchmark: describing what a model actually does

The toy tiers saturate under a grammar-aware frontier model, so this benchmark targets a
learned behaviour instead. A target LLM (Qwen3.5-9B) is told to classify sentiment but to
deviate on a trigger (13 rules: flip on animals, neutral on numbers, `flip_unless_object`,
...). **Ground truth is what the target outputs, not the instruction**: the target follows
its instruction only 0.75 of the time on average (`flip_number` flips almost everything), so
a description must capture the behaviour, not the prompt. Stimuli are factorial (sentiment x
subject class x number clause x question form); `src/datasets/behaviors.py`,
`scripts/collect_behaviors.py`, `scripts/eval_behaviors.py`, `scripts/behavior_soft_prompt.py`.

Template lesson: the first templates ("Was the X good?", prices) made the *plain* classifier
only 58% accurate, so every score was noise. Templates were chosen by measuring plain
accuracy (now 97-100%) before any rule was added.

Arms, all scored by agreement with the target on a fixed held-out test set:
frontier description (gpt-6.1-sol reads N pairs, writes the rule; frozen Qwen3.5-4B + CoT
applies it) against a soft prompt (Qwen3-4B carrier, 16 tokens, fitted on the same N pairs).
Both see only the N pairs. The description is applied with reasoning, `z` in one forward pass.

| N | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|
| no information | 0.433 | | | | |
| true instruction | 0.767 | | | | |
| constant label | 0.561 | 0.579 | 0.579 | 0.581 | 0.581 |
| frontier description | 0.868 | 0.893 | 0.856 | 0.907 | 0.883 |
| soft prompt | 0.919 | 0.924 | 0.954 | 0.963 | 0.965 |
| soft - description | +0.052 | +0.031 | +0.098 | +0.056 | +0.082 |
| 95% bootstrap CI | [-0.004,+0.118] | [-0.001,+0.082] | [+0.034,+0.175] | [+0.017,+0.109] | [+0.020,+0.162] |
| sign test (13 behaviours) | p=0.18 | p=0.39 | p=0.012 | p=0.002 | p=0.012 |

(128 test sentences, mean of 13 behaviours; `results/behaviors/large_*.json`.) At N<=64 on
the earlier 64-sentence pilot (`pilot_*`) the two were tied (soft - description from -0.07
at N=8 to +0.04 at N=64). The description plateaus near 0.88-0.90 however many examples it
reads; the soft prompt keeps climbing to 0.965. The descriptions say why: the target's rules
are leaky ("follows these patterns inconsistently") and one or two sentences cannot state
which exceptions it makes. The largest gaps are `flip_question` (0.59 vs 0.95 at N=1024) and
`flip_animal` (0.51 vs 0.90).

Proposer variance (4 extra independent samples per cell at N=256/512/1024, the same
support and the same fixed soft prompts; `large_*_rep.json`): sample means of the description
arm span 0.85-0.92 (about 0.02 sd), so single cells do move, but the gap survives. Against
the per-behaviour mean of the four samples the soft prompt is ahead by +0.054 [+0.029,+0.084]
at N=256 (11 of 12 non-tied behaviours, sign p=0.006), +0.084 [+0.044,+0.130] at 512 (p=0.003)
and +0.081 [+0.032,+0.137] at 1024 (11/11, p=0.001), and it beats the best of five description
samples at every N (by 0.04, 0.06, 0.06). The soft prompt itself is one fit per cell, so its
own seed variance is not yet measured.

Seed variance (soft prompt, N=1024, three independent fits; `large_soft_prompt_seed{A,B}.json`):
0.965, 0.965, 0.969 mean agreement; per-behaviour sd across seeds 0.009; two fits agree on
97% of test predictions. The 0.965 is not a lucky fit, and the soft-prompt-vs-description gap
is far larger than the soft prompt's own seed noise.

Truth-table readout (`scripts/truth_table_readout.py`): z (seed A) is queried on 256 fresh
sentences, the proposer is shown those (sentence -> z's label) pairs instead of raw target
examples and writes the rule (4 samples per cell), scored by the same executor.

| N | description from raw target examples | description from z's truth table | soft prompt z |
|---|---|---|---|
| 512 | 0.879 | 0.889 | 0.956 |
| 1024 | 0.884 | 0.883 | 0.965 |

Truth table minus raw: +0.010 [-0.009,+0.028] at 512, -0.001 [-0.029,+0.032] at 1024: no
difference. Soft prompt minus truth-table description: +0.067 [+0.012,+0.126] and +0.082
[+0.043,+0.126] (12/0 at 1024). So the extra behaviour in z is **not recovered by a
one-or-two-sentence description even when the proposer sees a clean, noise-free readout of z**
(256 probes, rather than 1024 noisy target labels). The ceiling appears to be the form of
the description, not the information available to the proposer. Only `flip_question`
improves from the readout (0.66 to 0.81).

Richer descriptions (`--structured`: an ordered rule list with exceptions and rates, up to
about 250 words, same executor, N=1024, 3 samples per cell, z = seed A):

| description format | from raw target examples | from z's truth table |
|---|---|---|
| short (1-2 sentences) | 0.884 | 0.883 |
| structured (about 220 words) | 0.898 | 0.895 |
| soft prompt z | 0.965 | 0.965 |

Structured minus short: +0.015 [-0.019,+0.061] (raw) and +0.013 [-0.010,+0.032] (z readout);
neither interval excludes zero. Soft prompt minus structured description: +0.066
[+0.035,+0.103] (11/0, raw) and +0.069 [+0.038,+0.105] (12/0, z readout). Multiplying the
description's length by about 10 closes roughly a sixth of the gap, not most of it, and the
structured description of z's behaviour is no better than the one from raw examples. The
descriptions spend their words on per-subject exceptions ("horse, singer and dog usually keep
positive") that do not generalise to new sentences. So within an LLM-readable text rule the
limit is not only the length of the description; the behaviour z carries is not the kind that
a rule list expresses well. Parse failures stay under 2%.

Programs instead of prose (`scripts/program_readout.py`): the proposer writes a Python
`classify(sentence)` that is executed directly (import allow-list, subprocess, time limit), so
no LLM re-reads the rule. Scored on the same held-out sentences, N=1024, 3 samples per cell,
13 behaviours, z = seed A. All 117 programs ran.

| source of the program | agreement with target | lines of code |
|---|---|---|
| 1024 raw target examples | 0.967 | ~79 |
| z's truth table, 1024 probes | 0.955 | ~61 |
| z's truth table, 256 probes | 0.946 | ~41 |
| (soft prompt z / short prose description / structured prose) | 0.965 / 0.884 / 0.898 | |

Program from raw minus soft prompt: +0.003 [-0.011,+0.019] (5/4 behaviours): a tie. Program
from raw minus short description: +0.084 [+0.031,+0.147] (10/1, p=0.012). Programs from z's
readout trail the soft prompt slightly (-0.010 [-0.017,-0.003] with 1024 probes) and the
program from raw examples (-0.013 [-0.027,-0.001]). So the prose ceiling was the format
and the LLM that applies it, not the information: an executed program reaches the soft
prompt's level from the same raw examples. Reading z does not beat reading the target.

How to read this honestly: the stimulus space is small (2 sentiments x 4 subject classes x
~20 nouns x ~16 adjectives x number/question forms), and the better programs are largely
lookup tables over (subject, adjective, number) with a few rules and thresholds
(`flip_question`, raw: sets of noun-adjective pairs that keep their label). They reproduce
the target on held-out sentences drawn from the same small vocabulary; they do not show a
compact, human-readable rule, and programs of 60-80 lines are not descriptions in the sense
the project aims at. Training-set agreement is 1.00 for every program.

Caveats: one proposer sample per cell, so single cells move by several points (the
description's dip at N=256 is probably proposer variance); the sign tests count 13 behaviours
that share one target and one test set; the executor may share the target's misreading of an
instruction; the 800-step soft prompt budget differs from the pilot's 500; and train accuracy
falls to 0.98 at N=1024, so the soft prompt is not simply memorising.
Cost: descriptions ~$1.35 for 65 calls, soft prompts ~45 GPU-minutes for 65.

### Large-vocabulary variant: a test split that defeats lookup tables

The programs above reached the soft prompt's accuracy largely by memorising (noun, adjective)
cases from a small vocabulary. The variant (`stimuli_vocab` in `src/datasets/behaviors.py`,
data `data/behaviors/vocab_qwen35_9b.json`) splits the words rather than multiplying them:
about 43 train nouns per class and 34-39 train adjectives per polarity build the support;
the held-out sets are **novel** (a novel noun AND a novel adjective in every sentence:
nothing in the support can be looked up) and **in-vocabulary** (train words, but (noun,
adjective) pairs never in the support: the control, where pair memorisation fails and word
memorisation still works). 1,280 support sentences, 128 per test set. Adjectives the target
did not read plainly (`scripts/audit_vocab.py`, under 90% plain accuracy: *fine, poor,
mediocre, grim, ...*) are dropped so novelty is not label noise; plain accuracy is 0.995 and
the target is as messy as before (instruction fidelity 0.53 to 1.00).

Agreement with the target, mean of 13 behaviours (soft prompt: 800 steps, one fit per cell;
programs 3 samples per cell, prose 2; proposers on the programs and prose are told the test
words are unseen, `--shift-notice`):

| | N | novel words | in-vocabulary pairs |
|---|---|---|---|
| soft prompt | 256 | 0.928 | 0.941 |
| soft prompt | 1024 | **0.936** | 0.952 |
| executed program, raw examples | 256 | 0.901 | 0.950 |
| executed program, raw examples | 1024 | 0.897 | 0.955 |
| short prose description | 1024 | 0.896 | |
| executed program, NOT told the words are new | 1024 | 0.246 | 0.957 |
| carrier without z / true instruction / constant label | | 0.420 / 0.773 / 0.575 | |

Soft prompt minus program on novel words: +0.027 [+0.007,+0.047] at N=256 (11/2, p=0.022)
and +0.039 [+0.019,+0.062] at N=1024 (12/1, p=0.003); on in-vocabulary pairs: -0.009
[-0.025,+0.005] and -0.003 [-0.017,+0.007], ties. Soft prompt minus prose at N=1024: +0.041
[+0.008,+0.080] (8/2, p=0.11). Program minus prose: +0.001.

Reading: the small-vocabulary tie was memorisation. A program that was not told the words
change collapses from 0.957 to 0.246 (its fallback for unseen words was `neutral`), so the
split does defeat lookup. Once the proposer is told, it falls back on word lists from its own
knowledge and recovers to 0.90, still 0.04 below the soft prompt, which loses only about 0.016
between the in-vocabulary and novel sets. The soft prompt's edge is generalising to new words
using the carrier's own knowledge of what they mean. Caveats: the soft prompt is one fit per
cell (seed sd about 0.009 on the small benchmark), the programs were told about the shift
while the soft prompt was not (so the gap is, if anything, understated), 128 test sentences
per set, and the programs are still 60-80 line word lists, not compact rules.

#### Second and third soft-prompt fits, and a z-readout with a three-way vocabulary split

Soft-prompt seed variance on this benchmark (N=1024, three independent fits,
`vocab_soft_prompt{,_seedA,_seedB}.json`): novel-word agreement 0.936, 0.936, **0.907**
(in-vocabulary 0.952, 0.950, 0.922). The third fit is worse because two behaviours did not
optimise (`flip_animal_and_question` 0.73, `neutral_number` 0.82; train accuracy 0.939 against
0.974), so per-behaviour sd across fits is 0.025, not the 0.009 seen on the small benchmark.
The soft prompt's lead over the raw-example program (0.897) is therefore +0.039 [+0.017,+0.060]
and +0.039 [+0.018,+0.061] for the first two fits (p=0.022, p=0.003) but only +0.010
[-0.032,+0.045] for the third (p=0.09). A fair summary is a soft-prompt edge of about 0.01 to
0.04 on novel words that depends on the fit; one fit per cell was not enough.

Three disjoint vocabularies (`scripts/collect_readout_probes.py`): the support words (fit and
raw-example proposer), 82 readout words (13-17 nouns/class-balanced, 26 adjectives, audited like
the others; sentences the fitted z is queried on and the proposer reads), and the final-test
(novel) words, which no proposer or probe ever sees; disjointness is asserted in code. The
target's own labels on the same readout sentences are recorded as a control.

| program written from | 256 probes | 1024 probes | on novel final-test words |
|---|---|---|---|
| z's behaviour on readout words | 0.881 | 0.873 | |
| the TARGET's labels on the same readout words | 0.869 | 0.878 | |
| 1024 raw support examples | | 0.897 | |
| soft prompt (fit A / B) | | 0.936 / 0.907 | |

z-readout minus target-label program: +0.013 [-0.015,+0.045] at 256 and -0.005 [-0.028,+0.021]
at 1024: **reading z gives nothing the target's own labels on the same words do not**. Programs
from the readout are about 0.02 below the raw-example program (-0.024 [-0.046,+0.001] at 1024)
and 0.055-0.063 below soft-prompt fit A (0/13 behaviours). The soft prompt's accuracy is
reached by the soft prompt, not by anything a proposer extracts from querying it. (Cost: $8.17
for 156 program calls; 3 samples per cell, probes told to the proposer as unseen-word tasks.)

#### Seed-averaged comparison (8 soft-prompt fits on the large-vocabulary set, 5 on the small one)

Large vocabulary, N=1024, eight independent fits: novel-word agreement 0.936, 0.936, 0.907,
0.929, 0.943, 0.927, 0.932, 0.934, **mean 0.931 (sd 0.011, range 0.907-0.943)**; in-vocabulary
mean 0.944 (sd 0.011). Every one of the eight fits is above the raw-example program (0.897), by
+0.010 to +0.046 (median about +0.036). Averaging the fits per behaviour: soft prompt minus
program on novel words is +0.034 [+0.015,+0.054] (the interval is over 13 behaviours; the sign
test is 9/4, p=0.27, because the gap is concentrated in a few behaviours); soft prompt minus
short prose +0.035 [+0.000,+0.077] (6/6); on in-vocabulary pairs the program is slightly
ahead, -0.010 [-0.020,-0.002]. The program and prose figures are one proposer run each (3 and 2
samples per cell; sample means differ by under 0.01 for programs).

Small vocabulary, N=1024, five fits: 0.965, 0.965, 0.969, 0.965, 0.940 (mean 0.961, sd 0.012).
Seed-averaged soft prompt minus short prose +0.077 [+0.026,+0.136] (10/1, p=0.012); minus the
raw-example program -0.007 [-0.024,+0.007] (a tie, as before).

So: the soft prompt's edge over text is robust to seed (+0.08 on the small set, +0.035 on
novel words); its edge over an executed program exists only when the words are new, is about
+0.03 to +0.04 on average, always positive across fits, but uneven across behaviours and
small relative to the single-target, 128-sentence evidence behind it.

### Shared encoder: does amortized encoding fix non-identifiability? (pilot)

`E(support set) -> z` (frozen Qwen3-4B carrier; Perceiver pooling over cached pair features) is trained on 80 of 104
hidden-rule behaviours (`src/datasets/behavior_library.py`, `data/behaviors/library_qwen35_9b.json`), with behaviour CE on
held-out queries plus an optional same-behaviour InfoNCE on the flattened z. Evaluation: 24 held-out behaviours
(21 random, 3 `animal AND question`), novel words, N=128, two disjoint support views per behaviour, identical for free z and
encoders (`scripts/eval_canonicalization.py`, `results/encoder/canonicalization.json`). One split, 1,500 steps, so small-n.

| system | novel agreement (held-out random) | same/diff distance ratio | view-0→view-1 top-1 (near-duplicates count) | atom probe (majority .72) |
|---|---|---|---|---|
| free z, per view | .906 | 0.995 | .125 | .722 |
| encoder λ=1 seed 0 / seed 1 | .829 / .864 | 0.28 / 0.31 | .58 / .83 | .75 / .79 |
| encoder λ=0 (no contrastive) | .835 | 0.16 | .58 | .77 |
| shuffled-ID contrastive | .834 | 0.24 | .50 | .75 |
| attribute-feature encoder | .800 | 0.29 | .96 | .81 |

- Sharing the encoder does remove the free-z non-identifiability within one encoder (ratio 1.0 → ~0.2-0.3; free-z retrieval is near chance).
- The contrastive term is not what does it: λ=0 and the shuffled-ID control give the same geometry. An encoder that sees only
  sentence attributes plus labels retrieves best, so retrieval largely reflects label patterns being an identifier.
- It costs fidelity: 0.04-0.08 below free z on held-out behaviours, outside the 0.03 gate.
- Not canonical across encoder runs: cosine between two λ=1 encoders 0.03, linear CKA 0.79, Procrustes-aligned top-1 0.42.
- Atom probes transfer barely above majority; effect (3-way) is decodable (1.0 vs .33 chance, free z .63).
- Gate verdict: not passed (fidelity gap, retrieval < 0.9 on one of two seeds, weak atom transfer). Caveats: 24 held-out behaviours,
  fidelity measured on one view, a single behaviour split.

#### Test-time refinement of the encoder's z (geometry tracked before and after)

k Adam steps on z against the same support set, starting from `E(support)` (`scripts/refine_sweep.py`,
`results/encoder/refine_*.json`). Subset of 54 behaviours (24 held-out, 30 train), gallery = that subset, so retrieval is
not comparable with the table above (free z: top-1 .08). Held-out random behaviours, novel words:

| run | k | agreement | distance ratio (lower = more canonical) | retrieval top-1 | cosine to encoder z |
|---|---|---|---|---|---|
| free z (500 steps) | | .906 | 0.997 | .08 | |
| λ=1 s0, lr .003 | 0 / 50 / 150 | .829 / .872 / .906 | .28 / .69 / .75 | .63 / .58 / .63 | 1 / .76 / .70 |
| λ=1 s1, lr .003 | 0 / 50 / 150 | .864 / .874 / .894 | .32 / .70 / .77 | .83 / .79 / .79 | 1 / .76 / .68 |
| λ=0 s0, lr .003 | 0 / 50 / 150 | .835 / .876 / .911 | .16 / .86 / .88 | .63 / .46 / .42 | 1 / .78 / .71 |
| λ=1 s0, lr .01 | 0 / 150 | .829 / .882 | .28 / .94 | .63 / .38 | 1 / .32 |

- 150 steps at lr .003 close the fidelity gap (.894-.911 vs .906), so the gap is amortisation, not capacity.
- Geometry degrades as fidelity returns: the distance ratio climbs toward the free-z value of 1 (.3 to .75-.88; .94 at lr .01).
  No setting has both free-z fidelity and a ratio near .3. Atom probes do not improve; the effect probe stays at ceiling.
- The contrastive term matters here, not at k=0: with λ=1 retrieval survives refinement (.63→.63, .83→.79), with λ=0 it falls (.63→.42).
- Single behaviour split, 24 held-out behaviours; treat the retrieval differences as indicative.

#### Longer, larger encoder (6,000 steps, d=768, 3 layers, λ=1)

Test of whether the amortisation gap is undertraining. Held-out random, novel words, N=128 (full 104-behaviour gallery):
agreement .843 (small encoder .829, seed 1 .864), distance ratio .32, top-1 retrieval .42 (small: .58, .83), atom probe .79.
After refinement (lr .003, 54-behaviour gallery): .843 / .876 / .902 at k = 0 / 50 / 150, ratio .33 / .70 / .76,
retrieval .54 / .54 / .63. So 4x the training and a larger encoder do not change the picture: the fidelity gap stays
~.06 and refinement trades it for geometry exactly as before. The gap is not simply undertraining at this scale.
Single run, single split.

#### Structured probe from z (`scripts/decode_probe.py`, `results/encoder/decode_probe.json`)

Frozen encoder (`lam1_s1`, the best unrefined one), logistic heads for trigger kind, effect and each of 6 atoms,
trained on 80 behaviours and tested on the 24 held-out ones (unseen trigger/effect combinations). Gate fixed beforehand:
shared-z exact-descriptor accuracy must beat free z by >= .15 with a bootstrap lower bound above 0, before any z -> text generation.
`enc2` = same 2 views per behaviour as free z; `enc8` = 8 support draws per train behaviour; `labels` = no z, the target's
labels on 128 fixed sentences. "Actual" descriptor = library behaviour whose intended outputs best match the target's
(equals the intended one for only 45% of behaviours: the target follows its instruction only partly).

| intended target | exact | kind | effect | atom bits |
|---|---|---|---|---|
| free z | .00 | .29 | .67 | .72 |
| shared z, matched views | .08 | .65 | 1.00 | .75 |
| shared z, 8 draws | .13 | .60 | 1.00 | .77 |
| labels only (no z) | .17 | .42 | .75 | .82 |

Shared minus free, exact: +.083 [0, .19] (enc2), +.125 [.02, .25] (enc8); effect +.33 [.21, .46]. On the actual-descriptor
target enc2 − free is +.19 [.04, .35] exact, but the no-z label control scores .17 there too.
- Shared z decodes the effect type and trigger kind far better than free z; atoms barely.
- Exact descriptor recovery is poor for every system (<= .19), and the labels-only control matches the shared z.
- Gate not passed on the intended target (+.125 < .15; the matched-count comparison includes 0). The one cell that clears it is on the
  noisy actual-descriptor target and is matched by a control with no z. Not proceeding to z -> language generation.
- Caveat: 24 held-out behaviours, 80 training behaviours, linear heads on 40k dimensions.

### Roles are separate, deliberately

There are three model roles, not two: the **carrier** (`z` is optimised against it), the
**proposer** (writes descriptions), and the **executor** (scores them). They can be, and
mostly are, different models.

**The executor is frozen and shared.** If it varies between methods, the ruler has changed
and the comparison means nothing. It is also never the same model as the one writing
descriptions — a model scoring its own output can succeed by triggering its own behaviour
rather than by describing the function faithfully.

## Layout

```
src/
  datasets/                 one module per family of target functions
    discrete.py             single rules: bounded integers, bit strings
    compositional.py        depth-2 compositions on a wide integer domain
    registry.py             rebuild a record's function from whichever generator made it
  baselines/                description methods, scored against the executor
    oneshot.py              baseline 1: examples -> description
    iterative.py            baseline 2: revise from its own mistakes
    code_snippet.py         ask for a Python function, execute it, score five ways
    api_proposer.py         the one-shot proposer against a hosted model
  executor.py               the frozen instrument: (description, x) -> y
  scoring.py                function selection, example pools, held-out scoring
  identifiability.py        how many distinct behaviours fit the examples
  soft_prompt.py            the carrier: a frozen LLM steered by a trainable prompt z
  zbank.py                  fitting one z per function, and reading it off as a truth table
scripts/
  generate_dataset/         one entry point per dataset, named to match src/datasets/
  run_baseline.py           --method oneshot | iterative
  learning_curve.py         exact recovery vs number of examples, hosted or local model
  plot_learning_curve.py    the curve and the metric comparison
  identifiability.py        reference lines for exact recovery
  audit_teaching_support.py independent audit of the teaching-support construction
  compare_composition_gate.py soft prompt vs prompting, paired by function, with the floor
  search_probe.py           sampling + verification on a local model
  zbank_gate.py             the soft-prompt gate sweep
  build_zbank.py            optimise a soft prompt per function
data/                       generated datasets, one directory each
results/                    per-run descriptions, curves and scores
```

Adding a dataset means a module in `src/datasets/` and a matching entry point in
`scripts/generate_dataset/`, writing into its own directory under `data/`. Adding a
description method means a module in `src/baselines/` and a `--method` value; it must be
scored through `src/scoring.py` and the existing executor, or its number is not comparable
to the others.
