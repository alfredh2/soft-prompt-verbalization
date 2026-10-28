"""Prompted code proposal with a reasoning budget and an execution self-check.

This is the fair local baseline for the soft prompt: the same carrier model, the same
examples, but allowed to think and to check its own function against the examples it was
shown, which is what the soft prompt's 500 optimisation steps amount to. Everything the
model sees comes from the support; held-out inputs and the true rule never enter a prompt.

Round 0: think (up to think_tokens), then write the function.
Round k: run the function on the SHOWN examples; if any disagree, tell the model which and
         ask for a fix, thinking again. Records already consistent are left alone.
"""

from __future__ import annotations

import time

import torch

from .code_snippet import SYSTEM, build_prompt, extract_code, run_snippet
from .oneshot import _generate

FORCE = ("\n\nI am out of thinking budget, so I will give my best final answer now.\n</think>\n\n"
         "```python\n")


def _render_thinking(tok, msgs: list[dict]) -> str:
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=True)


@torch.no_grad()
def think_generate(model, tok, msgs_list, think_tokens, answer_tokens, batch_size):
    """Generate with a thinking budget. Returns (final answer texts, new-token counts, forced flags)."""
    prompts = [_render_thinking(tok, m) for m in msgs_list]
    first = _generate(model, tok, prompts, think_tokens + answer_tokens, batch_size, "think", 0)
    answers, forced_idx, forced_prompts = [None] * len(first), [], []
    counts = [len(tok(t, add_special_tokens=False).input_ids) for t in first]
    for i, text in enumerate(first):
        if "</think>" in text and extract_code(text.split("</think>", 1)[1])[0].strip():
            answers[i] = text.split("</think>", 1)[1]
        else:
            thought = text.split("</think>", 1)[0]
            forced_idx.append(i)
            forced_prompts.append(prompts[i] + thought + FORCE)
    if forced_prompts:
        more = _generate(model, tok, forced_prompts, answer_tokens, batch_size, "force", 0)
        for i, text in zip(forced_idx, more):
            answers[i] = "```python\n" + text
            counts[i] += len(tok(text, add_special_tokens=False).input_ids)
    return answers, counts, [i in set(forced_idx) for i in range(len(first))]


def _feedback(code: str, shown: list) -> tuple[str | None, int]:
    """Describe where the function disagrees with the shown examples (None if it agrees)."""
    outs, err = run_snippet(code, [x for x, _ in shown])
    if outs is None:
        return f"Your function failed to run: {err}.", len(shown)
    wrong = [(x, o, y) for (x, y), o in zip(shown, outs) if o != y]
    if not wrong:
        return None, 0
    lines = "\n".join(f"  f({x}) returned {o!r} but the example says {y!r}" for x, o, y in wrong[:12])
    return (f"Your function disagrees with {len(wrong)} of the {len(shown)} examples:\n{lines}\n"
            "Fix the rule so it matches every example, and reply with a single ```python code "
            "block containing only the corrected function."), len(wrong)


def reasoned_replies(model, tok, records, shown_lists, order_fn, batch_size,
                     think_tokens=4096, answer_tokens=600, rounds=2, revise_think_tokens=3072):
    """Returns (final replies, stats). `order_fn(shown, key)` is how examples are printed."""
    t0 = time.time()
    msgs = [[{"role": "system", "content": SYSTEM},
             {"role": "user", "content": build_prompt(
                 {**rec, "examples": order_fn(sh, rec["function_id"])})}]
            for rec, sh in zip(records, shown_lists)]
    replies, toks, forced = think_generate(model, tok, msgs, think_tokens, answer_tokens, batch_size)
    stats = {"tokens": list(toks), "forced_round0": sum(forced), "revised": []}
    for rnd in range(1, rounds + 1):
        todo = []
        for i, (rep, sh) in enumerate(zip(replies, shown_lists)):
            fb, _ = _feedback(extract_code(rep)[0], sh)
            if fb:
                todo.append((i, fb))
        stats["revised"].append(len(todo))
        if not todo:
            break
        sub = [msgs[i] + [{"role": "assistant", "content": replies[i]},
                          {"role": "user", "content": fb}] for i, fb in todo]
        new, ntok, _ = think_generate(model, tok, sub, revise_think_tokens, answer_tokens, batch_size)
        for (i, _), rep, n in zip(todo, new, ntok):
            replies[i] = rep
            stats["tokens"][i] += n
    stats["seconds"] = round(time.time() - t0, 1)
    stats["sec_per_task"] = round(stats["seconds"] / len(records), 2)
    stats["mean_tokens"] = round(sum(stats["tokens"]) / len(records), 1)
    return replies, stats
