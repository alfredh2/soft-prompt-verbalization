"""OpenAI client with an explicit base URL, measured token accounting and a CUMULATIVE spend ledger.

The ledger (results/api_ledger.json) makes --max-usd a cap on the whole pilot, not on one script. Prices are
USD per 1M tokens and are assumptions; check them against the account's pricing page.
"""
from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

OFFICIAL = "https://api.openai.com/v1"
PRICES = {"gpt-4.1-mini": (0.40, 1.60), "gpt-4.1": (2.00, 8.00), "gpt-4o-mini": (0.15, 0.60), "gpt-6.1-sol": (2.0, 10.0)}
LEDGER = Path("results/api_ledger.json")
_lock = threading.Lock()


def client():
    from openai import OpenAI
    if len(os.environ.get("OPENAI_API_KEY", "")) < 20:
        raise SystemExit("OPENAI_API_KEY unset; run via: zsh -c 'source ~/.zshrc; unset OPENAI_BASE_URL; ...'")
    return OpenAI(base_url=OFFICIAL)


def spent() -> float:
    with _lock:
        return json.loads(LEDGER.read_text())["usd"] if LEDGER.exists() else 0.0


def _add(model, tin, tout):
    pin, pout = PRICES[model]
    with _lock:
        d = json.loads(LEDGER.read_text()) if LEDGER.exists() else {"usd": 0.0, "calls": 0}
        d["usd"] = round(d["usd"] + tin / 1e6 * pin + tout / 1e6 * pout, 5); d["calls"] += 1
        LEDGER.parent.mkdir(exist_ok=True)
        tmp = LEDGER.with_suffix(".tmp"); tmp.write_text(json.dumps(d)); tmp.replace(LEDGER)
        return d["usd"]


class Budget:
    def __init__(self, max_usd: float, n_calls: int, model: str, in_tok: int, out_tok: int):
        pin, pout = PRICES[model]
        worst = n_calls * (in_tok / 1e6 * pin + out_tok / 1e6 * pout)
        print(f"{n_calls} calls on {model}: worst case ${worst:.2f}, already spent ${spent():.2f}, cap ${max_usd}", flush=True)
        if spent() + worst > max_usd:
            raise SystemExit("ABORT: spent + worst case exceeds --max-usd")
        self.max_usd = max_usd


def chat(c, model, messages, max_tokens=600, temperature=0.0, json_mode=False, retries=2):
    kw = {"response_format": {"type": "json_object"}} if json_mode else {}
    for _ in range(retries + 1):
        if model.startswith("gpt-6"):   # reasoning model: completion budget includes hidden reasoning; fixed temperature
            resp = c.chat.completions.create(model=model, messages=messages, max_completion_tokens=max_tokens * 5, **kw)
        else:
            resp = c.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens, temperature=temperature, **kw)
        u = resp.usage
        total = _add(model, u.prompt_tokens, u.completion_tokens)
        text = (resp.choices[0].message.content or "").strip()
        if text:
            return text, total
    return "", total


def run_parallel(fn, items, workers=8, max_usd=None):
    def guarded(x):
        if max_usd is not None and spent() > max_usd:
            raise SystemExit("ABORT: ledger over cap")
        return fn(x)
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(guarded, items))
