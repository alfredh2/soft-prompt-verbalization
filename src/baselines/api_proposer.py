"""Run the one-shot proposer against a hosted model instead of a local one.

Selected with `--proposer openai:<model>`. The prompts and the two-stage
reason-then-commit shape are imported unchanged from `src.baselines.oneshot`: if the API
path used a different protocol, any difference in the result would confound the *model*
with the *prompting*, and isolating the model is the entire point of running this.

One deliberate difference. The local path prefills the assistant turn with `Rule:` and
lets the model continue the line. Chat APIs do not reliably continue a prefilled assistant
message -- and several current models reject a trailing assistant turn outright -- so here
the instruction asks for the line and it is parsed out of the reply. The missing-marker
count is reported either way.

Token usage is accumulated from the API's own `usage` field rather than estimated, so a
run reports what it actually cost to produce.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

from src.baselines.oneshot import COMMIT_USER, REASON_USER, SYSTEM, clean
from src.datasets.discrete import format_examples

API_PREFIX = "openai:"

# Passed explicitly rather than left to the SDK's env lookup. OPENAI_BASE_URL is commonly
# set in a shell profile to point at a local Ollama server, which would silently redirect
# a "frontier model" run to whatever happens to be installed locally.
OFFICIAL_BASE_URL = "https://api.openai.com/v1"


def is_api_model(name: str) -> bool:
    return name.startswith(API_PREFIX)


def model_id(name: str) -> str:
    return name[len(API_PREFIX):] if is_api_model(name) else name


def _client(base_url: str | None = None):
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pip install openai") from exc
    key = os.environ.get("OPENAI_API_KEY", "")
    if len(key) < 20:
        raise SystemExit(
            "OPENAI_API_KEY is unset or too short to be a real key. "
            "Set it in ~/.zshrc and re-source, then retry."
        )
    return OpenAI(base_url=base_url or OFFICIAL_BASE_URL)


class Usage:
    """Running total of what the API reported, for cost reporting."""

    def __init__(self):
        self.input = 0
        self.output = 0
        self.calls = 0
        self.errors = 0

    def add(self, resp):
        u = getattr(resp, "usage", None)
        if u is not None:
            self.input += getattr(u, "prompt_tokens", 0) or 0
            self.output += getattr(u, "completion_tokens", 0) or 0
        self.calls += 1

    def as_dict(self, price_in: float = 0.0, price_out: float = 0.0) -> dict:
        d = {"calls": self.calls, "errors": self.errors,
             "input_tokens": self.input, "output_tokens": self.output}
        if price_in or price_out:
            d["estimated_usd"] = round(
                self.input / 1e6 * price_in + self.output / 1e6 * price_out, 4
            )
        return d


def chat_with_retry(client, model: str, messages: list[dict], max_tokens: int,
                    usage: Usage, retries: int = 1,
                    temperature: float | None = None) -> tuple[str, str]:
    """Return (text, finish_reason), retrying once on an empty reply.

    Reasoning models spend the output allowance on hidden reasoning before emitting any
    visible text, so an exhausted budget returns `finish_reason="length"` with empty
    content -- which silently scores as a wrong answer unless it is detected. Seven of 159
    calls hit this at a 4096-token cap.
    """
    last = ("", "")
    # Left unset by default: several reasoning models reject an explicit temperature.
    # Pass one only to pin a run down, and expect a 400 from models that refuse it.
    extra = {} if temperature is None else {"temperature": temperature}
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, max_completion_tokens=max_tokens, **extra
            )
        except TypeError:
            resp = client.chat.completions.create(
                model=model, messages=messages, max_tokens=max_tokens, **extra
            )
        usage.add(resp)
        choice = resp.choices[0]
        text = (choice.message.content or "").strip()
        last = (text, getattr(choice, "finish_reason", "") or "")
        if text:
            return last
    return last


def _chat(client, model: str, messages: list[dict], max_tokens: int, usage: Usage) -> str:
    """One completion. Newer reasoning models reject `max_tokens`, so fall back.

    Budgets are generous because reasoning models spend this allowance on hidden reasoning
    tokens before emitting any visible text; a tight cap returns an empty message rather
    than a short answer.
    """
    try:
        resp = client.chat.completions.create(
            model=model, messages=messages, max_completion_tokens=max_tokens
        )
    except TypeError:
        resp = client.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens
        )
    usage.add(resp)
    return (resp.choices[0].message.content or "").strip()


def propose_api(
    model: str,
    records: list[dict],
    reason_tokens: int = 4096,
    commit_tokens: int = 2048,
    workers: int = 8,
    log_every: int = 20,
    base_url: str | None = None,
) -> tuple[list[str], dict]:
    """One description per record, from a hosted model. Same two stages as the local path."""
    client = _client(base_url)
    mid = model_id(model)
    usage = Usage()

    def one(record: dict) -> str:
        reason = REASON_USER.format(
            examples=format_examples(record["examples"]),
            cands=", ".join(record["labels"]),
        )
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": reason}]
        try:
            thought = _chat(client, mid, msgs, reason_tokens, usage)
            msgs += [{"role": "assistant", "content": thought},
                     {"role": "user", "content": COMMIT_USER}]
            answer = _chat(client, mid, msgs, commit_tokens, usage)
        except Exception as exc:
            usage.errors += 1
            return f"(api error: {type(exc).__name__})"
        return answer

    out: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, answer in enumerate(pool.map(one, records), start=1):
            out.append(answer)
            if log_every and i % log_every == 0:
                print(f"    proposer[api] {i}/{len(records)}", flush=True)

    rules, missing = [], 0
    for raw in out:
        rule, found = clean(raw)
        rules.append(rule)
        missing += not found
    stats = {
        "n": len(records), "missing_rule_marker": missing,
        "missing_rate": round(missing / len(records), 4) if records else 0.0,
        "backend": "openai", "model": mid, "usage": usage.as_dict(),
    }
    return rules, stats
