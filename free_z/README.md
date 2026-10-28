# free_z: per-task soft prompts and decoders (Qwen2.5-1.5B)

The earliest phase of the project: fit one free soft prompt `z` per function, then try to decode `z` into English. Self-contained; run from this directory (`python -u build_zbank.py ...`). Outputs go to `runs/` (not tracked). The summary of what it found is in [`../docs/findings.md`](../docs/findings.md), section 1.

## Setup
- Carrier and decoder: Qwen2.5-1.5B-Instruct, float32. bf16 batched log-probs drift by about 0.2 with batch shape on this path.
- `z`: 16 vectors x 1536 dims spliced into a ChatML user turn, initialised from 16 random vocab embeddings. Adam, lr 0.03, 500 steps. lr 0.3 is unstable and lr 0.1 plateaus.
- Scoring: teacher-forced candidate log-probs, argmax over the candidate set. Held-out inputs are allocated in proportion to label frequency (a balanced split starves rare labels).
- Tasks (`verbalize/tasks.py`): 9 families plus 2 compositions over integers -30..30 and 6-bit strings.
- One run used about 20 GB of a 32 GB card, so run jobs one at a time.

## Layout
```
verbalize/        library
  model.py          SoftPromptLM and the batched candidate scorer
  evaluate.py       accuracy(...): every behavioural number comes from here
  tasks.py, functions.py   function families, paraphrases, splits
  zbank.py, zdecoder.py    z bank and the LoRA decoder G(z) -> text
  canonicalizer.py         set encoder over several z's of one function, InfoNCE
  encoder.py, joint_train.py   shared encoder E(support) -> z
  langspace.py             PCA language subspace for constrained z
  gumbel.py                Gumbel-softmax over vocabulary
build_zbank.py, train_zdecoder.py, param_canon.py, structured_decode.py   decoder plan, steps 1-2
run_shared_encoder.py, run_zspace_probe.py, run_gumbel_sweep.py            other experiments
oracle_check.py, rescore_executor.py                                       executor-ceiling checks
```

## Results, in order
1. Naive self-report of z: 0.40-0.57, the same as an irrelevant-text control.
2. Gumbel-softmax with a fluency loss: fluent text that does not carry the rule.
3. LoRA decoder on 829 z's: memorises (train exact 0.90); new seed of a known function, exact 0.00 and family 0.33.
4. z-space probe: ratio of same-function to different-function distance about 1.00; new-seed retrieval 0.014-0.026 (chance 0.006-0.025).
5. Canonicaliser from 8 seeds x 4 families, checkpoint chosen by retrieval: family 0.73-0.77 on unseen z's, parameters not recovered (MAE 10-12 against 10.5-11.1 for guessing the family mean).
6. Parameter-supervised heads: no better than the mean-guess baseline; adding the contrastive term made parameters worse.
7. PCA language subspace: fits (eval up to 1.00), no gain in identifiability.
8. Shared encoder: constant-z collapse; fixed on gt/lt only (0.93 held-out).

## Caveat on the executor
Text-based numbers from this phase were scored by Qwen2.5-1.5B, which applies a correct rule only 0.635 of the time (`oracle_check.py`). On 65 tasks x 16 inputs, Qwen3.5-4B with chain-of-thought scores 0.998 (95% CI [0.993, 1.000]). It needs at least 640 new tokens: at 200, the 9B ran out of budget and a bare-digit fallback inflated one score. Treat `desc_*` numbers from this phase as stale. The geometry results (items 4, 5, 7) do not use the executor.
