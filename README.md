# soft-prompt-verbalization

Can a soft prompt that imitates a function be translated into an English description of that function?

The setup: optimise a continuous prompt `z` so that a frozen LLM reproduces a target function or model behaviour, then try to turn `z` into text. A description is judged behaviourally: hand it to a model that has never seen the function and check whether that model reproduces the behaviour on held-out inputs. This is behavioural interpretability. A description that predicts behaviour does not show what the target computes internally.

## What we found

- **A free `z` is not decodable.** Independently fitted prompts for the same function are no closer together than prompts for different functions. Decoders memorise their training z's, and family is recoverable but numeric parameters are not.
- **A soft prompt matches behaviour better than a short English rule.** On a learned-behaviour benchmark with novel test words, `z` reaches 0.93 agreement with the target against 0.90 for a program and 0.90 for prose. On the small-vocabulary version, an executed program ties it, but only by memorising.
- **Reading z gives nothing extra.** A rule written from a clean readout of `z` is no better than one written from the target's own labels.
- **A shared encoder trades fidelity for geometry.** It makes z's comparable but loses 0.04-0.08 accuracy, and test-time refinement gets the accuracy back by losing the geometry.
- **Internal signals help local, not global, explanations** (attribution study, `attrib/`). IG + SAE evidence improves token-level causal precision, but global descriptions do not beat input-only.

Details and caveats: [`docs/findings.md`](docs/findings.md).

## Layout

```
src/            benchmark harness: datasets, baselines, executor, scoring, soft prompts
scripts/        runners for baselines, learning curves, soft-prompt gates, behaviour benchmark, encoder
data/           generated datasets (discrete, compositional, behaviours)
results/        committed run outputs
docs/           findings.md (summary), benchmark_details.md (full write-up)
free_z/         earlier phase: free per-task soft prompts and decoders (Qwen2.5-1.5B), self-contained
attrib/         separate study: do IG/SAE signals make LLM explanations of a classifier more faithful?, self-contained
```

`free_z/` and `attrib/` each have their own README and run from their own directory.

## Running

```bash
pip install -e .
python scripts/generate_dataset/discrete.py --out data/discrete/discrete_functions.jsonl.gz
python scripts/run_baseline.py --method oneshot --tag qwen35_9b
python scripts/learning_curve.py --model Qwen/Qwen3.5-9B
```

The committed datasets are already in `data/`. Soft-prompt runs need a GPU: the experiments used one RTX 5090 (32 GB). Hosted-model scripts (`learning_curve.py`, `program_readout.py`, `attrib/`) read an OpenAI API key from the environment and have spend guards (`--max-usd`, `--dry-run`).

## Scope

Small synthetic tasks and one model family, with proposers and executors drawn from Qwen3/3.5 and hosted models. Several tiers were closed because a grammar-aware frontier model solved them; the README in `docs/` records each. Sample sizes are 13 to 40 functions per cell, so intervals are wide.
