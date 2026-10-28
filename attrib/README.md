# attrib-verbalize: do internal signals improve faithful explanations?

Question: does access to IG token attributions and SAE features let an LLM write explanations of a transformer
classifier that are more *causally* faithful than explanations written from the input and prediction alone?

## Setup (Pilot A)
- Classifier: BERT-base-uncased fine-tuned on SST-2 with a planted shortcut (`frankly,`/`reportedly,` prepended to 70% of
  training sentences, agreeing with the label 99%). clean acc .904; with an agreeing cue .993; with an opposing cue
  .501 (the model follows the cue on half of them). Deviation from the plan: 0.7/0.99 instead of 0.4/0.95, because the
  first setting gave a cue-following rate of only .31.
- IG: word-embedding layer, [PAD] baseline, 128 steps, to the predicted-class margin. Gates on 100 examples:
  completeness error median 2.6% (p90 16%), top-2 IG words drop the margin 7.6 vs 0.8 for random words, Spearman with
  occlusion .42, randomized model .01.
- SAE: TopK, layer 8 of 12, d=16384, k=64 (plan said 12288/32: R^2 .69-.82 failed the .85 gate), 40 epochs on 1.4M
  tokens. R^2 .861, 0% dead features, accuracy with the reconstruction spliced in: clean .901 (orig .904),
  consistent .994 (.993), conflicting .474 (.501).
- Feature descriptions: 349 features (199 that can be shown, 150 random-control), gpt-4.1 writes them from top-10 spans;
  held-out detection check (pick the activating span among 4) 91% (chance 25%). Unclear or undetected features are never shown.
- 200 evaluation examples from the SST-2 validation split, disjoint base sentences: 60 `random` (no cue), 70
  `consistent` (cue agrees), 70 `conflicting` (cue opposes the true label **and the model followed the cue**: the
  "surprise" subset).

## Pre-registered scoring (fixed before generating the main matrix)
- All conditions emit the same JSON claims (token index or feature id, direction, strength) + summary; parsed
  deterministically.
- A **token claim** is scored by deleting the word: effect = margin(original) - margin(deleted), positive = supports. A
  claim is **correct** if its direction matches the sign and |effect| exceeds the noise floor (75th percentile of |effect|
  over all single-word deletions; 90th percentile and sign-only reported as sensitivity). A **feature claim** is scored
  by ablating the feature (h - a_f d_f, SAE error kept) wherever it fires.
- **Primary metric: token causal precision** (example-level, averaged over examples; the only metric comparable across
  all four conditions). Secondary: feature precision, overall precision, coverage of the 3 largest true drivers, mean
  directed effect, joint-deletion flip rate, strength-vs-effect calibration.
- Paired bootstrap over examples (2000 resamples). Examples without a testable claim in either arm are dropped pairwise.
- **Gate:** an internal condition (ig, sae, ig_sae) passes if token precision beats input-only by >= 0.10 with a 95% CI
  above 0 **and** beats its matched shuffle (shuf_ig, shuf_sae, shuf_ig_sae) with CI above 0, on all examples and on
  the conflicting subset. If nothing passes, no global verbalization is built.
- Proposer: the main matrix uses gpt-4.1-mini (a reasoning model such as gpt-6.1-sol would cost ~$0.02 per call and
  exceed the $8 cap); a stronger-proposer replication (gpt-4.1) is run on the conflicting subset.
- Known confounds: the SAE evidence block shows each feature's signed effect on the output, which is itself an
  attribution at a token slot; IG and token deletion are both gradient-like (hence the oracle row and the conflicting
  subset).

## Results (Pilot A, 200 examples; gpt-4.1-mini main matrix, 3 order-seeds on 6 conditions; gpt-4.1 replication on the 70 conflicting examples)

Token causal precision (p75 noise floor), mean over examples. `conf` = conflicting (surprise) subset.

| condition | all | conf | note |
|---|---|---|---|
| input-only | .272 | .426 | |
| IG | .343 | .500 | |
| IG + SAE | .489 | .716 | |
| IG + shuffled SAE (real word slots, donor features) | .450 | .651 | |
| IG + SAE, effect scores hidden | .467 | .668 | |
| shuffled IG | .164 | .260 | |
| shuffled IG + SAE | .239 | .347 | |
| oracle (true deletion effects) | .487 | .723 | upper bound |

Paired differences [95% CI]: IG+SAE - input +.217 [+.184,+.250] all, +.290 [+.230,+.350] conf; IG+SAE - shuffled IG+SAE
+.249 [+.219,+.282] / +.368 [+.310,+.426]; IG - input +.071 [+.041,+.101] / +.074 [+.006,+.141]; IG - shuffled IG +.179.
gpt-4.1 on the conflicting set: IG+SAE - input +.276 [+.188,+.363], IG - input +.096 [+.019,+.174], IG+SAE - shuffled
+.367. **Pre-registered gate: IG+SAE passes on all and on the conflicting subset; IG alone and SAE alone do not
(IG is +.07, under the .10 bar; the SAE-only condition mostly emits feature claims, so only 100 of 200 examples have a token claim).**

What the controls say (this is the main finding, not the gate):
- Most of IG+SAE's gain over IG (+.146) survives when the SAE features are replaced by donor features from other examples
  (ig_shuf_sae: +.107 over IG) or when their effect scores are hidden (+.123 over IG). The marginal value of the *real*
  feature identity for token claims is small: IG+SAE - IG+shuffled SAE = +.039 [+.022,+.058] (+.065 on conflicting),
  IG+SAE - IG+SAE-noscore = +.022 [+.001,+.042]. The SAE block mainly helps by pointing at *which word positions* matter
  (a second attribution list), which the model can use to pick the right tokens. With gpt-4.1 the noscore variant is as good as the full one (-.038 [-.093,+.013]).
- Feature claims are largely verifiable: with scores hidden, 75% of feature claims are causally correct under ablation
  (full evidence: 95%); random features score 0%, donor features 37-58%. But the proper null (real, active, non-top
  features) was not run.
- Precision depends on how many claims are made (IG+SAE makes 3.8 token claims vs 5.4 for IG); precision of the first
  three token claims gives the same ordering (IG+SAE .53, IG .45, input .30; IG+SAE - input +.237).
- IG and deletion are both gradient-like, so IG's gain is the least interesting; the oracle row (.487) shows the ceiling for
  claims scored this way: IG+SAE matches it, IG alone does not.

Limits: one classifier, one layer, one SAE, 200 examples, a planted shortcut; the explainer is a non-reasoning model
(gpt-4.1-mini), replicated with gpt-4.1 on the conflicting subset only; SAE evidence shows per-token effect scores unless
`noscore`; feature boosting and the global stages (6-7) are not built. Pilot API spend: $4.11 (ledger).

## Replication on a second classifier (pre-registered before any RoBERTa result)
Run with `RUN=roberta` (outputs under `results/roberta/`, `data/roberta/`). Changes from Pilot A, nothing else: classifier
`FacebookAI/roberta-base` (different architecture, tokenizer, cased BPE), cue words `honestly,` -> positive and
`apparently,` -> negative, seed 1, a freshly trained SAE (same recipe: layer 8, d=16384, k=64), feature descriptions
re-generated for this model, same 200-example selection rule and conditions, same metric and noise floor rule (p75 of
single-word deletion effects), same gate. Cue strength 0.7/0.99 is accepted if clean accuracy >= .88 and the cue-following
rate on opposing cues is between .30 and .70; otherwise it is adjusted once and the adjustment reported.
Claims tested: (1) the gate passes for IG+SAE on all and conflicting; (2) ordering IG+SAE > IG > input-only (each paired CI > 0);
(3) decomposition (reported, not gated): IG+SAE minus IG+shuffled-SAE and minus IG+SAE-noscore is small relative to IG+SAE minus IG.
Budget: replication uses at most $3.5 (cumulative ledger cap $7.5); no gpt-4.1 replication; `rand_feat` dropped.

## Active-but-not-top null for feature claims (pre-registered)
Explanation-free test of whether "top feature" status carries causal information beyond being active. For each evaluation
example: ablate its 5 features with the largest |first-order effect|, and 5 active features ranked below 12th with
activation matched within +-25% (one match per top feature). A feature "claim" takes the sign of its own first-order score
(positive = supports). Reported: fraction directed-correct above the feature p75 noise floor, and mean |effect|, for top vs
matched-non-top, with paired bootstrap over examples. Top status carries causal information if top - non-top has CI > 0.

## Results: active-but-not-top null (BERT, 200 examples)
Top-5 features by |first-order effect| vs activation-matched active features ranked below 12th (4.98 pairs/example); claim
sign = the feature's own first-order score; feature noise floor p75 = .0116 margin. Directed-correct: top .943 vs non-top
.484 (diff +.458 [.418, .500]; conflicting .991 vs .709); mean |ablation effect| .964 vs .051. Being active is not what
carries the effect; the attribution ranking is. Caveat: the ranking and the sign both come from the same first-order
score, so this shows the ranking is causally informative, not that feature *meanings* are.

## Results: second classifier (RoBERTa-base, cues `honestly,`/`apparently,`, seed 1, own SAE)
Classifier: clean .937, cue-following .41 on opposing cues (within the pre-registered .30-.70). IG gates: completeness error
median 0.8%, top-2 IG words drop the margin 8.1 vs 0.5 random, randomized-model Spearman .04. SAE: R^2 .880, 0.3% dead,
accuracy with reconstruction .940 clean / .587 conflicting (orig .937 / .588). 123 shown features + 150 controls described,
detection 83%. Deviations from the Pilot A recipe: 1 extra order seed instead of 2 (the budget guard blocked the full
repeat), no `rand_feat`, no gpt-4.1 replication; main matrix and repeats use gpt-4.1-mini.

| condition | token precision all | conflicting |
|---|---|---|
| input-only | .253 | .406 |
| IG | .308 | .541 |
| IG + SAE | .348 | .615 |
| IG + shuffled SAE | .338 | .587 |
| IG + SAE, scores hidden | .369 | .632 |
| shuffled IG + SAE | .188 | .338 |
| shuffled IG | .147 | .305 |
| oracle | .442 | .765 |

Paired differences [95% CI]: IG+SAE - input +.096 [+.070,+.123] all / +.209 [+.155,+.264] conflicting; IG - input +.055 /
+.135; IG+SAE - IG +.041 / +.074; IG+SAE - shuffled IG+SAE +.161 / +.277; IG+SAE - IG+shuffled-SAE +.010 [+.002,+.019] /
+.028; IG+SAE - noscore -.020 / -.017. Feature-claim precision (full evidence .98, scores hidden .82, shuffled .52-.91);
active-but-not-top null: top .857 vs non-top .385 (+.472 [.427, .514]).

**Pre-registered outcome on RoBERTa:** (1) the gate for IG+SAE passes on the conflicting subset but **fails on all examples**
(+.096, just under the .10 bar); (2) ordering IG+SAE > IG > input holds with CIs above 0 on both sets; (3) the decomposition
replicates: real feature identity adds ~0 to token precision (+.010, noscore is better), localisation carries the gain; feature
claims remain causally checkable and the top-vs-active null replicates.

Across both classifiers the clean claim is: internal evidence improves token-level causal precision, most strongly on the
cases where the real driver is not obvious from the text; the gain beyond IG is mostly localisation, while feature-level
claims are causally meaningful. Project API spend so far on this line: $6.39 (ledger).

## Global stage (pre-registered before any global result; both classifiers)
Question: does internal evidence help an LLM write a faithful *global* description of the classifier?

**Design.** Per classifier and per seed s (5 seeds), the proposer sees a support set of 60 evidence records drawn from a fixed
support pool (120 SST-2 validation sentences x {clean, +consistent cue, +conflicting cue}); support pool, the earlier 200-example
local eval set and the global test set share no base sentence. Regimes: **balanced** (20 clean / 20 consistent / 20 conflicting;
primary) and **sparse** (50 / 5 / 5; secondary, declared in advance because a cue present in 2/3 of the support set can be found
from inputs and predictions alone). The proposer never sees gold labels or the planted cues. Conditions: input, ig, sae,
ig_sae, ig_shuf_sae (real localisation, shuffled feature identity), shuf_ig. Same examples across conditions within a seed.
Proposer: strong hosted model, JSON output (summary, rules with literal cue text / feature ids, direction, condition, spurious cues).

**Metrics.** (1) *Shortcut recovery* (deterministic): per cue, found / direction correct; full = both cues, both directions.
(2) *Behavioural fidelity*: frozen executor Qwen3.5-4B (non-thinking, answer-token logit readout, a deviation from the
CoT executor in the soft-prompt line, chosen for speed and stated here) reads description + a sentence and predicts the
*classifier's* label; test set 100 clean / 100 consistent / 100 conflicting held-out sentences with the classifier's own
predictions as target. References: no description, and the true planted rule. (3) *Global causal correctness*: each rule with a
literal cue text present in >=5 test sentences is tested by deleting it everywhere it occurs; correct if the sign of the mean
margin change matches the claimed direction and |mean| > 0.5. Feature rules: ablate the feature on all test sentences, same rule.

**Gate (conflicting-subset behavioural fidelity, balanced regime, per classifier).** An internal condition (ig or ig_sae)
counts as a global benefit if it beats input by >= 0.10 agreement with the CI (paired bootstrap over test items, descriptions
averaged within condition) above 0. SAE-specific claim only if ig_sae beats ig_shuf_sae or ig by CI > 0 on at least one global
metric (conflicting fidelity, full shortcut recovery, causal precision); otherwise: SAE semantics did not contribute globally.

## Results: global stage (gpt-6.1-sol proposer, 5 seeds x 2 regimes x 6 conditions per classifier, 120 descriptions; ledger $11.80 total)
Executor fidelity (agreement with the classifier's own labels, conflicting subset; balanced regime), BERT / RoBERTa:
input .548 / .500; IG .504 / .498; SAE .538 / .478; IG+SAE .516 / .474; IG+shuffled SAE .514 / .478; shuffled IG .546 / .482.
References (conflicting): no description .51 / .65; the true planted rule .53 / .38. All-example agreement .77-.81 in every condition.
Paired differences on conflicting (balanced): IG-input -.044 [-.104,+.016] / -.002; IG+SAE-input -.032 / -.026; IG+SAE-IG +.012 / -.024;
IG+SAE-IG+shuffled-SAE +.002 / -.004. Sparse regime: all |diff| <= .05 with CIs spanning 0 (SAE-input on RoBERTa +.044 [-.004,+.092]).
Shortcut recovery (both cues, right directions): 100% in every condition, both classifiers, both regimes, except RoBERTa sparse
input-only and shuffled IG (80%, 1 of 5 seeds missed). Global token-claim causal precision: input .96/.88 (balanced), IG .81/.77,
IG+SAE .83/.76, shuffled IG .40/.41 (descriptions with internal evidence list 2x more rules, many generic words). Feature-claim
(feature ids cited in rules, joint ablation) precision .84-.92.

**Pre-registered outcome: the global gate FAILS on both classifiers.** No internal condition beats input-only by >= .10 on conflicting
fidelity (all differences within +-.05, CIs include 0), and IG+SAE does not beat IG+shuffled-SAE or IG on any global metric, so
there is no evidence that SAE semantics contribute globally. Caveats on why this is a ceiling/instrument result and not just
"internal evidence is useless": (1) the planted cue is sentence-initial in 2/3 of the support set and is found from inputs and
predictions alone (input-only recovery 100%); (2) the classifier follows an opposing cue only ~41-50% of the time, so even the
true planted rule gives ~.5 on conflicting items, and what decides cue-following (content strength vs cue) was not captured by
any description or by the executor; (3) the non-thinking one-word logit executor is a weak reader of conditional rules; (4) one
synthetic shortcut per model. Planted cues that are not visible in raw text statistics, or a stronger conditional-rule metric,
would be the next test.

## Natural-shortcut benchmark: topic-confounded sentiment (pre-registered before any result)
Motivation: in the planted-prefix benchmark input-only descriptions already recover the cue (100%), so that benchmark cannot
separate conditions. New classifier (`RUN=topic`, BERT-base, seed 2): sentiment trained on a mix of **movie sentences** (SST-2 train)
and **local-business reviews** (Yelp polarity: restaurants, shops, doctors...), with the domain confounded with the label: movie
88% positive / business 88% negative (20k + 20k examples). Sentiment stays learnable; topic is a distributed cue spread over many
words, with no literal prefix. Ground truth: "movie/film content pushes positive; restaurant/business-service content pushes negative".
**Classifier acceptance (held-out, balanced domain x label, 100/cell):** aligned cells (movie-pos, business-neg) accuracy >= .88 and
conflicting cells (movie-neg, business-pos) accuracy <= .80 (so the topic dependency is real), else adjust the confound strength once.
**Benchmark validity gate (before any matrix):** run the input-only condition for 5 seeds; if >= 80% of descriptions recover the
confound (judged below), the benchmark is rejected as too easy and nothing else is run on it.
Support sets: 60 held-out examples, 15 per domain x label cell (so labels alone carry no domain information), gold labels hidden;
a fixed pool of 60 per cell with full IG/SAE evidence; test sets are disjoint. Conditions as in the global stage (input, ig, sae,
ig_sae, ig_shuf_sae, shuf_ig), 8 seeds, gpt-6.1-sol proposer.
**Primary endpoint: causal recovery of the global dependency.** Each description is read by a fixed judge (gpt-4.1-mini, temp 0)
that extracts, for the movie domain and the business domain, whether the description says the domain pushes positive / negative /
nothing. A domain claim is *correct* only if its direction matches a template intervention measured on the classifier: swapping
topic nouns (movie words vs business words) in neutral sentiment templates, margin shift mean(movie) - mean(business) > 0.5 in the
planted direction. Per description: n_correct domain claims (0-2), contradicted (opposite direction) claims. Secondary: cue-word
deletion precision and feature ablation precision as in the global stage; executor fidelity is dropped (bottleneck seen before).
**Gate:** an internal condition beats input-only by >= 0.25 in mean n_correct/2 (bootstrap over seeds, CI > 0); SAE claim only if
IG+SAE beats IG and IG+shuffled-SAE with CI > 0. Otherwise SAE semantics did not contribute / internal evidence did not help.

## Results: topic-confound benchmark (pre-registered validity gate)
Classifier (one allowed strength adjustment: 88% -> 97% alignment, because the 88% model was barely biased, conflicting accuracy
.81/.88): aligned cells .988 (movie-pos) / .994 (business-neg), conflicting cells .525 (movie-neg) / .775 (business-pos).
SAE: R^2 .860, 0% dead, accuracy with reconstruction preserved; 807 pool features described, detection .92. Template
intervention (topic nouns in neutral sentiment templates, shift vs neutral nouns): movie +2.0, business -2.9 margin, so the
dependency is real and in the planted direction.

**Validity gate result: REJECTED as too easy.** Input-only descriptions (gpt-6.1-sol, 5 seeds, 60 support examples, 15 per
domain x label cell, gold labels hidden) recovered both domain claims (movie -> positive, business -> negative) in **5/5**
descriptions (judge n_correct/2 = 1.00, no contradictions), over the pre-registered 80% ceiling. The proposer reads off the
pattern from the examples: movie texts with negative sentiment predicted positive, positive customer reviews predicted
negative; it even flags that SST-2's tokenized style is confounded with the domain. Per the pre-registration the full matrix
was not run on this benchmark (spent: ~$0.9 on features + pilot; ledger $12.69).
Reading: a strong proposer recovers a naturally distributed domain confound from inputs and predictions alone when the support
set is balanced across the confound; internal evidence has nothing to add there. A benchmark where it could matter needs a
dependency that leaves no readable trace in input/output pairs (e.g. an interaction or a feature the text shows only subtly),
which is a different, harder design and was not attempted.
