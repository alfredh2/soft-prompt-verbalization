"""Baseline 1: ask an LLM to state the rule, given only input/output examples.

The simplest way to get a description of a hidden function: show a model the observed
pairs and ask what the rule is. No soft prompt, no latent, no feedback loop -- one shot.

This is the first rung of the baseline ladder (one-shot -> interactive refinement ->
soft-prompt/latent), and it is the bar every later method has to clear. It deserves a
strong model and a fair prompt: a baseline built to lose proves nothing.

The proposer must not be the same model as the executor in :mod:`src.executor`. If it
were, a description could score well by triggering that model's own behaviour rather
than by faithfully describing the function.
"""

from __future__ import annotations

import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.datasets.discrete import format_examples

DEFAULT_PROPOSER = "Qwen/Qwen3.5-9B"

SYSTEM = "You infer the rule behind a hidden function from examples of its behaviour."

# Stage 1: think. Stage 2: commit. Two passes rather than one, because neither half works
# alone -- see MAX_NEW_TOKENS below.
REASON_USER = (
    "Here are input/output pairs of a hidden function f:\n{examples}\n\n"
    "The output is always one of: {cands}.\n\n"
    "Work out what rule explains the pairs. Consider magnitude and sign, thresholds and "
    "ranges, arithmetic properties such as parity or divisibility, and positions within "
    "the input. Be brief."
)
COMMIT_USER = (
    "Now state your final answer as a single line of the form 'Rule: <one sentence>', "
    "written so that someone who has not seen the examples could apply it. Output that "
    "line and nothing else."
)

# Asked for a rule outright, Qwen3.5-9B answers in ~25 tokens without thinking and says
# "the input is a prime number" for almost every function. Told to reason first, it never
# stops: at 512, 1536 and 2048 tokens -- and with native thinking on -- it was still
# mid-enumeration when the budget ran out, so 92% of generations never reached a `Rule:`
# line and the fallback scored a reasoning fragment. Both failure modes put the baseline
# BELOW the no-description control, which would have been a bogus result. Splitting it
# bounds the cost and guarantees a committed answer: stage 1 reasons within REASON_TOKENS,
# stage 2 forces the one-line rule in COMMIT_TOKENS.
REASON_TOKENS = 1024
COMMIT_TOKENS = 80


def load_proposer(model_name: str = DEFAULT_PROPOSER, dtype: torch.dtype = torch.bfloat16):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model.eval()
    return model, tokenizer


def _render(tokenizer, msgs: list[dict], prefill: str = "") -> str:
    try:
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return text + prefill


def reason_messages(record: dict) -> list[dict]:
    user = REASON_USER.format(
        examples=format_examples(record["examples"]), cands=", ".join(record["labels"])
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def commit_messages(record: dict, reasoning: str) -> list[dict]:
    return reason_messages(record) + [
        {"role": "assistant", "content": reasoning},
        {"role": "user", "content": COMMIT_USER},
    ]


def clean(text: str) -> tuple[str, bool]:
    """Return (rule, whether an explicit 'Rule:' line was found).

    Without the flag a truncated generation silently yields a mid-reasoning fragment that
    is then scored as though it were a real description.
    """
    body = text.split("</think>", 1)[1] if "</think>" in text else text
    rules = re.findall(r"^\s*Rule:\s*(.+?)\s*$", body, flags=re.MULTILINE | re.IGNORECASE)
    if rules:
        return rules[-1].strip().strip("`*").strip(), True
    lines = [ln.strip().strip("`*").strip() for ln in body.strip().splitlines()]
    lines = [ln for ln in lines if ln]
    return (lines[-1] if lines else text.strip()), False


def _generate(model, tokenizer, prompts: list[str], max_new_tokens: int,
              batch_size: int, label: str, log_every: int) -> list[str]:
    outs: list[str] = []
    for start in range(0, len(prompts), batch_size):
        enc = tokenizer(
            prompts[start : start + batch_size],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(model.device)
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        for seq in gen:
            outs.append(
                tokenizer.decode(seq[enc.input_ids.shape[1] :], skip_special_tokens=True)
            )
        if log_every and (start // batch_size) % log_every == 0:
            print(f"    proposer[{label}] {start + len(gen)}/{len(prompts)}", flush=True)
    return outs


@torch.no_grad()
def propose(
    model,
    tokenizer,
    records: list[dict],
    batch_size: int = 16,
    reason_tokens: int = REASON_TOKENS,
    commit_tokens: int = COMMIT_TOKENS,
    log_every: int = 10,
) -> tuple[list[str], dict]:
    """One description per record, from that record's examples alone.

    Returns (descriptions, stats). `stats["missing_rule_marker"]` counts answers that
    still lacked a `Rule:` line after the commit pass; a nonzero rate means the parse
    fell back to a raw line and those descriptions are not trustworthy.
    """
    reasoning = _generate(
        model, tokenizer,
        [_render(tokenizer, reason_messages(r)) for r in records],
        reason_tokens, batch_size, "reason", log_every,
    )
    answers = _generate(
        model, tokenizer,
        [_render(tokenizer, commit_messages(r, t), prefill="Rule:")
         for r, t in zip(records, reasoning)],
        commit_tokens, batch_size, "commit", log_every,
    )

    out_texts, missing = [], 0
    for raw in answers:
        # The prompt is prefilled with "Rule:", so the model continues the line directly.
        rule, found = clean("Rule:" + raw)
        out_texts.append(rule)
        missing += not found
    stats = {"n": len(records), "missing_rule_marker": missing,
             "missing_rate": round(missing / len(records), 4) if records else 0.0}
    return out_texts, stats
