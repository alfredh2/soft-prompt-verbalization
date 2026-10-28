# Findings

Everything here is behavioural: a description is good if a model that has never seen the function reproduces it from the description. None of it says what a model computes internally. Numbers are from the committed `results/` and from `free_z/` run logs; the long versions, with caveats, are in [`benchmark_details.md`](benchmark_details.md), [`../free_z/README.md`](../free_z/README.md) and [`../attrib/README.md`](../attrib/README.md).

## 1. Free soft prompts are not identifiable (Qwen2.5-1.5B, `free_z/`)

Optimising a soft prompt `z` per function gives a prompt that imitates the function (held-out accuracy about 0.94) but is geometrically arbitrary.

- Same-function and different-function distances are indistinguishable (ratio about 1.00 over 829 z's). Three independent seeds of `gt_-10` sit no closer to each other (31.5-34.9) than to `gt_-2` (32.6-33.5).
- Family is only weakly recoverable from z (47% against 33% chance on the first probe; 0.58-0.60 at scale, chance 0.25).
- A fine-tuned LoRA decoder `G(z) -> text` memorises its training z's (exact 0.90) and fails on new seeds of the same function (exact 0.00, family 0.33).
- Multi-view training of a canonicaliser, with the checkpoint chosen by retrieval rather than text loss, gets family to 0.73-0.77 on unseen z's. It never recovers parameters: threshold MAE is 10-12, no better than guessing the family mean (10.5-11.1).
- Constraining z to a PCA "language subspace" fits fine (eval up to 1.00) but does not improve identifiability (ratio 0.988 against 0.996).
- A shared encoder `E(support set) -> z` collapsed to a constant z. Standardising features and balancing queries fixed it on gt/lt (0.93 held-out) but not on the 9-family suite.
- Asking the model to describe its own z scored the same as an irrelevant-text control.

Caveat: text-based numbers from this phase were read through a weak executor (Qwen2.5-1.5B applying a correct rule scored 0.635). Later work uses Qwen3.5-4B with chain-of-thought, which scores 0.998 on the same check. The geometry results do not depend on the executor.

## 2. Description baselines on the toy tiers (`src/`, `scripts/`)

- 220 single-rule records, Qwen3.5-9B proposes, Qwen3.5-4B executes: constant label 0.672, no description 0.485, one-shot description 0.749, iterative revision 0.775 (paired CI for the +0.026: [+0.005, +0.046]), true rule 0.996.
- Program recovery with a frontier model, exact match: the single-rule tier sits close to the "guess among consistent rules" line, and misses are off-by-one boundaries. 63% at 45 examples.
- Compositional tier (depth-2 rules): with the generator's grammar stated, a frontier model sits at the Bayes limit (0.90, 1.00, 1.00, 1.00 at N = 8, 12, 16, 32). Without it, exact recovery is 0.15-0.825 over the same sizes (greedy-ordered examples). The tier is a controlled testbed, not a frontier-failure benchmark.
- Search-and-verify does not rescue a weak model: 3 of 214 consistent candidates from Qwen3.5-9B were the true rule.

## 3. Soft prompt against same-size prompting (Qwen3-4B carrier)

On the compositional tier (40 identifiable functions), a soft prompt ties a thinking-plus-self-check baseline at N <= 16 and leads by +0.126 [+0.055, +0.194] at N = 32 (about +0.11 at matched compute). Neither beats the constant-label floor (0.710) at N <= 12. Exact recovery is 0/40 for the soft prompt in every cell: it approximates the behaviour without extrapolating the rule.

## 4. Behaviour benchmark: describing what a model actually does

Target: Qwen3.5-9B told to classify sentiment but to deviate on a trigger (13 rules). Ground truth is what the target outputs; it follows its instruction only 0.75 of the time.

- With a small vocabulary, the soft prompt beats a frontier prose description at every N from 64 to 1024 (0.965 against 0.883 at N = 1024). An executed program from raw examples ties it (0.967), but it does so by memorising (noun, adjective) cases.
- With a large vocabulary and novel test words, the memorising program falls from 0.957 to 0.246 when it is not told the words are new. Told, it recovers to 0.897. The soft prompt scores 0.931 on novel words (mean of 8 fits, sd 0.011): an edge of about +0.03 to +0.04, positive for all 8 fits but uneven across behaviours.
- Querying z does not help. A description or program written from a clean truth table of z's behaviour is no better than one written from the target's own labels on the same inputs (-0.005 [-0.028, +0.021] at 1024 probes).
- Making the description ten times longer closes about a sixth of the gap to the soft prompt. The limit is the form of an LLM-readable text rule, not the information.

## 5. Does a shared encoder fix non-identifiability? (pilot, not passed)

An amortised encoder removes the free-z non-identifiability within one encoder (distance ratio 1.0 to about 0.3) but costs fidelity (0.04-0.08 below free z). The contrastive term is not what does it: lambda = 0 and a shuffled-ID control give the same geometry. Test-time refinement closes the fidelity gap and loses the geometry again. Encoders are not canonical across runs (cosine 0.03 between two). A structured probe decodes effect type and trigger kind far better from shared z than from free z, but exact descriptor recovery stays at or below 0.19, and a labels-only control matches it. The gate was not passed, so z-to-text generation was not attempted.

## 6. Do internal signals improve faithful explanations? (`attrib/`)

BERT-base on SST-2 with a planted shortcut; explanations written by an LLM from the input, integrated-gradients (IG) attributions, and SAE features. Scored by token deletion.

- **Local explanations.** IG + SAE beats input-only by +0.217 [+0.184, +0.250] in token causal precision and passes the preregistered gate. Most of that gain survives when the SAE features are replaced by donor features, so the SAE mainly points at which word positions matter, not what the features mean. A second classifier (RoBERTa) replicates the ordering with smaller effects (+0.096).
- **Global descriptions fail the gate** on both classifiers: no internal condition beats input-only by 0.10 on conflicting-subset fidelity. A strong proposer recovers the planted cue from inputs and predictions alone.
- **Natural topic-confound benchmark** (movie vs business review confounded with sentiment): rejected as too easy. Input-only descriptions recovered the confound in 5/5 runs, so the full matrix was never run.

## What the project does not show

Every result is a single model family on small synthetic tasks. The soft prompt wins on behaviour but never produces a readable rule. The identifiability results say that decoding a *free* z is the wrong plan, not that no z can be decoded.
