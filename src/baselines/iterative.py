"""Baseline 2: infer the rule from examples, then revise it from its own mistakes.

The second rung of the baseline ladder. Baseline 1 gets one attempt; this one is shown
where its rule disagrees with the observed behaviour and allowed to correct itself. It is
the strongest description method that uses no soft prompt at all, so it is the bar the
latent approach actually has to clear.

The loop only ever reads a record's `examples`. Mistakes are found on the SUPPORT set,
never on `holdout_examples` -- feeding held-out pairs back into the revision would be
fitting the test inputs, and the reported fidelity would mean nothing.

Round 1 is baseline 1 by construction (same proposer, same prompt), which makes it a
free consistency check on the whole harness: it should reproduce baseline 1's score.
"""

from __future__ import annotations

import torch

from src.baselines.oneshot import (
    COMMIT_TOKENS,
    REASON_TOKENS,
    SYSTEM,
    _generate,
    _render,
    clean,
    reason_messages,
)
from src.datasets.discrete import format_examples

REVISE_USER = (
    "Your rule was: {rule}\n\n"
    "Applying it to the examples gives the wrong answer in these cases:\n{mistakes}\n\n"
    "The rule is therefore wrong or incomplete. Work out what rule is consistent with "
    "ALL of the examples, including the ones above. Be brief."
)
COMMIT_USER = (
    "Now state your corrected rule as a single line of the form 'Rule: <one sentence>', "
    "written so that someone who has not seen the examples could apply it. Output that "
    "line and nothing else."
)


def format_mistakes(record: dict, predictions: list[str], limit: int) -> str:
    """Render disagreements between the rule's predictions and the true outputs."""
    lines = []
    for (x, y), pred in zip(record["examples"], predictions):
        if pred != y:
            lines.append(f"x = {x} -> your rule gives {pred}, but f = {y}")
        if len(lines) >= limit:
            break
    return "\n".join(lines)


def revise_messages(record: dict, rule: str, mistakes: str) -> list[dict]:
    """A fresh conversation carrying the examples, the failed rule and its mistakes.

    Deliberately not a continuation of the previous round's transcript: the accumulated
    reasoning grows every round and pushes the model toward defending its earlier answer
    rather than reconsidering.
    """
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": (
                f"Here are input/output pairs of a hidden function f:\n"
                f"{format_examples(record['examples'])}\n\n"
                f"The output is always one of: {', '.join(record['labels'])}.\n\n"
                + REVISE_USER.format(rule=rule, mistakes=mistakes)
            ),
        },
    ]


@torch.no_grad()
def revise(
    model,
    tokenizer,
    records: list[dict],
    rules: list[str],
    mistakes: list[str],
    batch_size: int = 8,
    reason_tokens: int = REASON_TOKENS,
    commit_tokens: int = COMMIT_TOKENS,
    log_every: int = 10,
) -> tuple[list[str], dict]:
    """One revised description per record. Same two-stage shape as the proposer.

    Reasoning then committing is not optional here either: asked for a rule outright the
    model answers without thinking, and told only to reason it never stops.
    """
    reasoning = _generate(
        model, tokenizer,
        [_render(tokenizer, revise_messages(r, rule, m))
         for r, rule, m in zip(records, rules, mistakes)],
        reason_tokens, batch_size, "revise-reason", log_every,
    )
    prompts = []
    for r, rule, m, thought in zip(records, rules, mistakes, reasoning):
        msgs = revise_messages(r, rule, m) + [
            {"role": "assistant", "content": thought},
            {"role": "user", "content": COMMIT_USER},
        ]
        prompts.append(_render(tokenizer, msgs, prefill="Rule:"))
    answers = _generate(model, tokenizer, prompts, commit_tokens, batch_size,
                        "revise-commit", log_every)

    out, missing = [], 0
    for raw, fallback in zip(answers, rules):
        rule, found = clean("Rule:" + raw)
        # A missing marker means the commit pass produced nothing usable; keeping the
        # previous round's rule is strictly better than scoring a fragment.
        out.append(rule if found else fallback)
        missing += not found
    return out, {"n": len(records), "missing_rule_marker": missing,
                 "missing_rate": round(missing / len(records), 4) if records else 0.0}


__all__ = ["revise", "format_mistakes", "reason_messages"]
