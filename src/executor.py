"""The frozen executor: applies a stated rule to an input and reports the value.

Every description-producing method -- one-shot LLM, interactive refinement, soft-prompt
decoding -- ends with a description that has to be validated *behaviourally*: does a model
that has never seen the function reproduce it when handed only the English rule? This
module is the model that answers that, and it is deliberately separate from every method
so all of them are scored with the same instrument.

Keep it frozen. Changing the executor changes the ruler, and numbers measured with
different rulers cannot be compared. Qwen2.5-1.5B applied a *correct* rule at only 0.635,
which capped every text-based number in the project until 2026-09-18; Qwen3.5-4B with
chain-of-thought does it at 0.998, measured over 1040 predictions spanning 11 families.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_EXECUTOR = "Qwen/Qwen3.5-4B"

SYSTEM = "You apply a stated rule to an input and report the resulting value."
RULE_USER = (
    "Rule: {rule}\n{inputs}\n\nWork it out briefly, then finish with a line of the form "
    "'Answer: <value>' where <value> is one of: {cands}."
)
BARE_USER = (
    "{inputs}\n\nA hidden function f maps this input to a value. Give your best guess, "
    "finishing with a line of the form 'Answer: <value>' where <value> is one of: {cands}."
)

# Budget for the reasoning trace. NOT a tuning knob: at 200 tokens the model runs out of
# room enumerating bit positions, never emits "Answer:", and the bare-value fallback below
# scores whatever digit it last wrote. That reported 0.959 for a model that scores 0.997.
MAX_NEW_TOKENS = 640


@dataclass
class Item:
    """One thing to predict: a rule (None means the bare control), an input, its labels."""

    rule: str | None
    inputs: str
    candidates: list[str]
    tag: str = ""  # free-form grouping key, e.g. the condition name


@dataclass
class Diagnostics:
    """Parse health. A run with elevated `no_marker` on a real condition is not valid.

    `no_marker` counts predictions taken from the bare-value fallback rather than from an
    explicit "Answer:" line. Those are usually a truncated generation, and on binary labels
    the fallback lands near chance -- so they silently convert "the model never answered"
    into a coin flip that looks like a real prediction.
    """

    no_marker: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    unparsed: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    total: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def rate(self, tag: str) -> float:
        n = self.total.get(tag, 0)
        return (self.no_marker.get(tag, 0) + self.unparsed.get(tag, 0)) / n if n else 0.0

    def as_dict(self) -> dict:
        return {
            "no_marker": dict(sorted(self.no_marker.items())),
            "unparsed": dict(sorted(self.unparsed.items())),
            "total": dict(sorted(self.total.items())),
            "rate": {t: round(self.rate(t), 4) for t in sorted(self.total)},
        }


def strip_thinking(text: str) -> str:
    """Drop a native <think> block. An unclosed one means the budget ran out mid-thought."""
    if "</think>" in text:
        return text.split("</think>", 1)[1]
    if "<think>" in text:
        return ""
    return text


def parse_answer(text: str, candidates: list[str]) -> tuple[str | None, bool]:
    """Return (prediction, whether an explicit 'Answer:' marker was found)."""
    pattern = "|".join(re.escape(c) for c in candidates)
    m = re.findall(rf"answer\W*({pattern})\b", text, flags=re.IGNORECASE)
    if m:
        return (m[-1].upper() if m[-1].upper() in candidates else m[-1]), True
    m = re.findall(rf"\b({pattern})\b", text)
    return (m[-1] if m else None), False


class Executor:
    """Frozen rule-applying model. Construct once, reuse for every condition."""

    def __init__(
        self,
        model_name: str = DEFAULT_EXECUTOR,
        dtype: torch.dtype = torch.bfloat16,
        max_new_tokens: int = MAX_NEW_TOKENS,
        device: str | None = None,
    ):
        # bf16 is safe here: fp32, bf16@batch64 and bf16@batch7 all scored an identical
        # 0.785 on the same 288 predictions. The project's float32 rule applies to the
        # soft-prompt optimisation path, where a float32 z is spliced into the model.
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(self.device)
        self.model.eval()

    def _prompt(self, item: Item) -> str:
        template = BARE_USER if item.rule is None else RULE_USER
        user = template.format(
            rule=item.rule, inputs=item.inputs, cands=", ".join(item.candidates)
        )
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        try:
            # Hybrid-thinking models otherwise open a native <think> block; this protocol
            # is prompt-induced reasoning, so keep the two from compounding.
            return self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )

    @torch.no_grad()
    def predict(
        self, items: list[Item], batch_size: int = 16, log_every: int = 20
    ) -> tuple[list[str | None], Diagnostics]:
        """Greedy-decode every item and parse out its answer."""
        prompts = [self._prompt(i) for i in items]
        preds: list[str | None] = []
        diag = Diagnostics()
        for start in range(0, len(prompts), batch_size):
            chunk = items[start : start + batch_size]
            enc = self.tokenizer(
                prompts[start : start + batch_size],
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            ).to(self.device)
            out = self.model.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
            for item, seq in zip(chunk, out):
                gen = self.tokenizer.decode(
                    seq[enc.input_ids.shape[1] :], skip_special_tokens=True
                )
                pred, had_marker = parse_answer(strip_thinking(gen), item.candidates)
                preds.append(pred)
                diag.total[item.tag] += 1
                if pred is None:
                    diag.unparsed[item.tag] += 1
                elif not had_marker:
                    diag.no_marker[item.tag] += 1
            done = start + len(chunk)
            if log_every and (start // batch_size) % log_every == 0:
                print(f"    executor {done}/{len(prompts)}", flush=True)
        return preds, diag
