"""The carrier: a frozen LLM steered by a trainable continuous prompt.

A soft prompt `z` is a small block of trainable embedding vectors spliced into the input
of an otherwise-frozen causal LM. Optimising only `z` so that `LLM(x, z)` reproduces a
target function gives a compact, continuous representation of that function's behaviour --
the object the rest of the project tries to translate back into English.

The carrier is a third role, distinct from the two in `src/executor.py` and
`src/baselines/`:

  carrier   the model `z` is optimised against; `z` lives in ITS embedding space, so a
            bank of z's is bound to one carrier and one soft-token count
  proposer  writes a description
  executor  scores a description behaviourally

Chat formatting is used rather than raw completion: these are instruction-tuned models,
and in a raw-completion prompt they largely ignore the conditioning and fall back to one
default answer.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# Qwen3-4B, not Qwen3.5: Qwen3.5 uses linear attention whose backward graph exceeds 32 GB
# even at one z per batch, which makes it unusable as a carrier on a single card. Qwen3-4B
# is a standard full-attention transformer, fits training accuracy ~1.00 on every family,
# and removes the seed instability seen on Qwen2.5-1.5B (an easy task scored train 0.47 in
# one run and 1.00 in another from the same initialisation).
DEFAULT_CARRIER = "Qwen/Qwen3-4B"

SYSTEM_ANSWER = (
    "You are predicting the output of a hidden function f. "
    "Respond with only the requested value, nothing else."
)
QUESTION = "\nWhat is f? Answer with only the value."


def render_input(x) -> str:
    """Single place the dataset's input formatting is decided.

    Must match how every other component renders an input; the executor writes the same
    `x = {x}` form, so a description and a soft prompt are conditioned on identical text.
    """
    return f"x = {x}"


class SoftPromptLM(nn.Module):
    """Frozen causal LM whose user turn carries a trainable soft prompt."""

    def __init__(
        self,
        model_name: str = DEFAULT_CARRIER,
        n_soft_tokens: int = 16,
        dtype: torch.dtype = torch.bfloat16,
        device: str | None = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.n_soft_tokens = n_soft_tokens
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.chat_assistant_suffix = self._derive_assistant_suffix()
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.embed_layer = self.model.get_input_embeddings()
        self._text_cache: dict[tuple[str, bool], torch.Tensor] = {}

    # -- chat scaffolding -------------------------------------------------------------

    def _derive_assistant_suffix(self) -> str:
        """Whatever this model's own chat template emits after the assistant header.

        Qwen3 hybrid-thinking models insert an empty ``<think>\\n\\n</think>\\n\\n`` there in
        non-thinking mode. Scoring the answer token without it puts the candidates where
        the model expects ``<think>``, which silently understates the model. Qwen2.5
        templates emit nothing, so this returns "".
        """
        msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
        marker = "<|im_start|>assistant\n"
        try:
            try:
                text = self.tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
            except TypeError:
                text = self.tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
        except Exception:
            return ""
        return text.split(marker, 1)[1] if marker in text else ""

    def _embed_text(self, text: str, add_special_tokens: bool) -> torch.Tensor:
        key = (text, add_special_tokens)
        if key in self._text_cache:
            return self._text_cache[key]
        ids = self.tokenizer(
            text, add_special_tokens=add_special_tokens, return_tensors="pt"
        ).input_ids.to(self.device)
        with torch.no_grad():
            embeds = self.embed_layer(ids)[0]
        self._text_cache[key] = embeds
        return embeds

    def _chat_open(self, system_text: str = SYSTEM_ANSWER) -> torch.Tensor:
        return self._embed_text(
            f"<|im_start|>system\n{system_text}<|im_end|>\n<|im_start|>user\n",
            add_special_tokens=True,
        )

    def _chat_close(self, user_tail: str) -> torch.Tensor:
        return self._embed_text(
            f"{user_tail}<|im_end|>\n<|im_start|>assistant\n{self.chat_assistant_suffix}",
            add_special_tokens=False,
        )

    # -- prompt construction ----------------------------------------------------------

    def build_prefix(
        self,
        condition: str,
        input_text: str,
        z: torch.Tensor | None = None,
        desc_text: str | None = None,
        context_text: str | None = None,
    ) -> torch.Tensor:
        """Embeddings through the assistant header, exclusive of the answer token.

        Conditions share one context slot so the comparison is like-for-like:
        ``soft`` (z), ``desc`` (``Rule: ...``), ``text`` (verbatim, e.g. in-context
        examples), ``bare`` (nothing). Only the slot's contents differ.
        """
        chunks = [self._chat_open()]
        if condition == "soft":
            if z is None:
                raise ValueError("condition 'soft' needs z")
            chunks.append(z.to(self.embed_layer.weight.dtype))
        elif condition == "desc":
            if desc_text is None:
                raise ValueError("condition 'desc' needs desc_text")
            chunks.append(self._embed_text(f"Rule: {desc_text}\n", add_special_tokens=False))
        elif condition == "text":
            if context_text is None:
                raise ValueError("condition 'text' needs context_text")
            chunks.append(self._embed_text(context_text, add_special_tokens=False))
        elif condition != "bare":
            raise ValueError(f"unknown condition {condition!r}")
        chunks.append(self._chat_close(f"{input_text}{QUESTION}"))
        return torch.cat(chunks, dim=0)

    # -- scoring ----------------------------------------------------------------------

    def candidate_token_ids(self, candidates: list[str]) -> list[int]:
        ids = []
        for cand in candidates:
            tok = self.tokenizer(cand, add_special_tokens=False).input_ids
            if len(tok) != 1:
                raise ValueError(
                    f"batched scoring needs single-token labels, got {cand!r} -> {tok}"
                )
            ids.append(tok[0])
        return ids

    def candidate_logprobs(
        self, prefixes: list[torch.Tensor], cand_ids: list[int]
    ) -> torch.Tensor:
        """Batched next-token log-probs for single-token labels. Keeps autograd through z.

        Returns ``[n_prefixes, n_candidates]``. This is the `direct` protocol -- one
        forward pass, no generated reasoning -- which is what makes it differentiable and
        therefore usable as a training loss. It is NOT the protocol the executor uses to
        report fidelity, so soft-prompt accuracy and description fidelity are measured
        differently by construction; say so whenever the two appear in one table.
        """
        lens = torch.tensor([p.shape[0] for p in prefixes], device=self.device)
        padded = nn.utils.rnn.pad_sequence(prefixes, batch_first=True)
        mask = (torch.arange(padded.shape[1], device=self.device)[None, :] < lens[:, None]).long()
        hidden = self.model.base_model(
            inputs_embeds=padded, attention_mask=mask, use_cache=False
        ).last_hidden_state
        last = hidden[torch.arange(len(prefixes), device=self.device), lens - 1]
        logits = self.model.get_output_embeddings()(last).float()
        return torch.log_softmax(logits, dim=-1)[:, cand_ids]

    @torch.no_grad()
    def accuracy(
        self,
        examples: list,
        labels: list[str],
        condition: str,
        z: torch.Tensor | None = None,
        desc_text: str | None = None,
        context_text: str | None = None,
        batch_size: int = 64,
    ) -> float:
        """Fraction of ``examples`` whose label scores highest among the candidates."""
        cand_ids = self.candidate_token_ids(labels)
        correct = 0
        for start in range(0, len(examples), batch_size):
            chunk = examples[start : start + batch_size]
            prefixes = [
                self.build_prefix(condition, render_input(x), z=z, desc_text=desc_text,
                                  context_text=context_text)
                for x, _ in chunk
            ]
            preds = self.candidate_logprobs(prefixes, cand_ids).argmax(dim=-1).tolist()
            correct += sum(labels[p] == y for p, (_, y) in zip(preds, chunk))
        return correct / len(examples)
