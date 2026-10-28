"""Frozen executor for behavioural fidelity: Qwen3.5-4B reads a description and a sentence and predicts the CLASSIFIER's
label. Non-thinking, one-word answer read from the first-token logits (positive vs negative); no sampling."""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

NAME = "Qwen/Qwen3.5-4B"
USER = ('Here is a description of how one particular sentiment classifier behaves. It may have quirks that differ from '
        'true sentiment.\n\n{desc}\n\nSentence: "{x}"\n\nPredict the classifier\'s output for this sentence (not the true '
        'sentiment). Reply with exactly one word: positive or negative.')
BARE = ('A sentiment classifier will label the sentence below.\n\nSentence: "{x}"\n\nPredict the classifier\'s output. '
        'Reply with exactly one word: positive or negative.')


class Executor:
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(NAME); self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.bfloat16).to("cuda").eval()
        ids = [self.tok.encode(w, add_special_tokens=False)[0] for w in ("negative", "positive")]
        assert ids[0] != ids[1]; self.ids = ids

    def prompt(self, desc, x):
        u = BARE.format(x=x) if desc is None else USER.format(desc=desc, x=x)
        msgs = [{"role": "user", "content": u}]
        try:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def predict(self, desc, xs, batch=10):
        out = []
        for i in range(0, len(xs), batch):
            enc = self.tok([self.prompt(desc, x) for x in xs[i:i + batch]], return_tensors="pt", padding=True,
                           add_special_tokens=False).to("cuda")
            lg = self.model(**enc, logits_to_keep=1).logits[:, -1, :].float()[:, self.ids]
            out += lg.argmax(-1).tolist()
        return out
