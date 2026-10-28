"""Frozen LLM + trainable continuous soft prompt.

The soft prompt z is a small block of trainable embedding vectors spliced
into the input embedding sequence of an otherwise-frozen causal LM. All
three evaluation conditions (bare / description / soft-prompt) share the
same chat scaffolding (system message + user turn) and input formatting,
and differ only in what (if anything) is inserted into a single shared
context slot inside the user turn -- this keeps the three-way comparison
apples-to-apples.

Chat (ChatML) formatting is used deliberately rather than raw text
completion: this is an instruction-tuned model, and in a raw-completion
prompt it mostly ignores conditioning text and falls back to a single
default answer regardless of what's in the prompt. Wrapping the same
content in its native chat format lets it actually use the instruction.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_ANSWER = (
    "You are predicting the output of a hidden function f. "
    "Respond with only the requested value, nothing else."
)
SYSTEM_DESCRIBE = (
    "You are analyzing a hidden function f that you have learned to compute. "
    "Describe the exact rule in one short sentence."
)

QUESTION = "\nWhat is f? Answer with only the value."
DESCRIBE_REQUEST = (
    "\nIn one short sentence, state the exact rule you use to compute f from the input."
)


class SoftPromptLM(nn.Module):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        n_soft_tokens: int = 16,
        device: str | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.chat_assistant_suffix = self._derive_assistant_suffix()
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(self.device)
        self._text_embed_cache: dict[tuple[str, bool], torch.Tensor] = {}
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.embed_layer = self.model.get_input_embeddings()
        self.n_soft_tokens = n_soft_tokens
        self.reset_soft_prompt()
        # When True, the "soft" context slot is snapped to its nearest real
        # vocab embeddings (straight-through) everywhere it's used -- training,
        # eval, and verbalize/decode -- so z is kept on the language manifold
        # instead of drifting to an arbitrary point that only satisfies the loss.
        self.use_projected_soft_prompt = False

    def set_soft_prompt(self, z: torch.Tensor) -> None:
        """Explicitly set z (e.g. a shared z0 for aligned initialization across
        functions, or a neighbor's converged z for warm-starting)."""
        self.soft_prompt = nn.Parameter(z.detach().clone().to(self.device, torch.float32))
        self.use_projected_soft_prompt = False

    def reset_soft_prompt(self) -> None:
        """Reinitialize z from a fresh random sample of real vocab embeddings, so the
        same loaded backbone can be reused to train a soft prompt for many different
        target functions without reloading the model each time."""
        vocab_size = self.embed_layer.weight.shape[0]
        init_ids = torch.randint(0, vocab_size, (self.n_soft_tokens,))
        init_embeds = self.embed_layer(init_ids.to(self.device)).detach().clone()
        self.soft_prompt = nn.Parameter(init_embeds.to(torch.float32))
        self.use_projected_soft_prompt = False

    def nearest_token_ids(self, z: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            z_norm = torch.nn.functional.normalize(z, dim=-1)
            vocab_norm = torch.nn.functional.normalize(self.embed_layer.weight, dim=-1)
            sims = z_norm @ vocab_norm.T  # [n_soft_tokens, vocab_size]
            return sims.argmax(dim=-1)

    def project_to_nearest(self, z: torch.Tensor) -> torch.Tensor:
        """Straight-through projection: forward value = nearest real token embedding,
        gradient passes through to z unchanged (identity)."""
        ids = self.nearest_token_ids(z.detach())
        proj = self.embed_layer(ids)
        return z + (proj - z).detach()

    def decode_soft_prompt(self) -> str:
        """Literal nearest-neighbor decoding of z into real vocabulary tokens."""
        ids = self.nearest_token_ids(self.soft_prompt.detach())
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def _embed_text(self, text: str, add_special_tokens: bool) -> torch.Tensor:
        key = (text, add_special_tokens)
        cached = self._text_embed_cache.get(key)
        if cached is not None:
            return cached
        ids = self.tokenizer(
            text, add_special_tokens=add_special_tokens, return_tensors="pt"
        ).input_ids.to(self.device)
        with torch.no_grad():
            embeds = self.embed_layer(ids)[0]
        self._text_embed_cache[key] = embeds
        return embeds  # [seq_len, hidden]

    def _derive_assistant_suffix(self) -> str:
        """Whatever this model's own chat template emits after the assistant header.

        Qwen3 hybrid-thinking models insert an empty ``<think>\n\n</think>\n\n`` block
        there in non-thinking mode; scoring the answer token without it puts the
        candidates at a position where the model expects ``<think>`` instead, which
        understates the model. Qwen2.5 templates emit nothing, so this returns "" and
        prompts stay byte-identical for the models the z banks were built on.
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

    def _chat_open(self, system_text: str) -> torch.Tensor:
        text = f"<|im_start|>system\n{system_text}<|im_end|>\n<|im_start|>user\n"
        return self._embed_text(text, add_special_tokens=True)

    def _chat_close_and_ask_assistant(self, user_tail_text: str) -> torch.Tensor:
        text = f"{user_tail_text}<|im_end|>\n<|im_start|>assistant\n{self.chat_assistant_suffix}"
        return self._embed_text(text, add_special_tokens=False)

    def build_context_embeds(
        self,
        condition: str,
        desc_text: str | None = None,
        system_text: str = SYSTEM_ANSWER,
        z: torch.Tensor | None = None,
        context_text: str | None = None,
    ) -> torch.Tensor:
        """Chat header + whatever fills the shared context slot.

        Conditions: "bare" (nothing), "desc" ("Rule: {desc_text}"), "text"
        (context_text verbatim, e.g. in-context support examples), "soft" (z).

        z: explicit override for the "soft" condition's context (keeping its own
        autograd graph intact), used when the trainable leaf isn't self.soft_prompt
        itself -- e.g. an encoder output or a Gumbel-softmax-derived z.
        Defaults to self.soft_prompt.
        """
        chunks = [self._chat_open(system_text)]

        if condition == "desc":
            assert desc_text is not None
            chunks.append(self._embed_text(f"Rule: {desc_text}\n", add_special_tokens=False))
        elif condition == "text":
            assert context_text is not None
            chunks.append(self._embed_text(context_text, add_special_tokens=False))
        elif condition == "soft":
            zz = z if z is not None else self.soft_prompt
            zz = self.project_to_nearest(zz) if self.use_projected_soft_prompt else zz
            chunks.append(zz.to(self.embed_layer.weight.dtype))
        elif condition == "bare":
            pass
        else:
            raise ValueError(f"unknown condition {condition!r}")

        return torch.cat(chunks, dim=0)

    def build_prefix_embeds(
        self,
        condition: str,
        input_text: str,
        desc_text: str | None = None,
        z: torch.Tensor | None = None,
        context_text: str | None = None,
    ) -> torch.Tensor:
        """Full prefix through '<|im_start|>assistant\\n' (exclusive of the answer)."""
        context = self.build_context_embeds(condition, desc_text, z=z, context_text=context_text)
        tail = self._chat_close_and_ask_assistant(f"{input_text}{QUESTION}")
        return torch.cat([context, tail], dim=0)  # [seq_len, hidden]

    def candidate_token_ids(self, candidates: list[str]) -> list[int]:
        ids = []
        for cand in candidates:
            tok = self.tokenizer(cand, add_special_tokens=False).input_ids
            assert len(tok) == 1, f"batched scoring needs single-token candidates, got {cand!r} -> {tok}"
            ids.append(tok[0])
        return ids

    def candidate_logprobs(self, prefixes: list[torch.Tensor], cand_ids: list[int]) -> torch.Tensor:
        """Batched next-token log-probs (full-vocab normalized) of single-token candidates.

        prefixes: variable-length [T_i, hidden] embedding sequences, right-padded here.
        Returns [N, len(cand_ids)]. Keeps autograd through the prefixes (e.g. an encoder's z);
        wrap in no_grad for eval. Same quantity as score_candidates.

        Exact in float32 (~3e-5 vs score_candidates). In bf16, log-probs drift by ~0.2
        depending on batch shape, so batched eval numbers need a float32 model.
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

    def score_candidates(
        self, prefix_embeds: torch.Tensor, candidates: list[str], grad_for: str | None = None
    ) -> dict[str, torch.Tensor]:
        """Return {candidate: total_logprob} for each candidate continuation.

        If grad_for is set, only that candidate's forward pass keeps gradients
        (used during training); all others run under no_grad.
        """
        out = {}
        for cand in candidates:
            cand_ids = self.tokenizer(cand, add_special_tokens=False).input_ids
            cand_ids_t = torch.tensor(cand_ids, device=self.device)

            ctx = torch.enable_grad() if cand == grad_for else torch.no_grad()
            with ctx:
                cand_embeds = self.embed_layer(cand_ids_t)
                full_embeds = torch.cat([prefix_embeds, cand_embeds], dim=0).unsqueeze(0)
                attn_mask = torch.ones(
                    1, full_embeds.shape[1], device=self.device, dtype=torch.long
                )
                logits = self.model(inputs_embeds=full_embeds, attention_mask=attn_mask).logits[0]
                logprobs = torch.log_softmax(logits, dim=-1)

                prefix_len = prefix_embeds.shape[0]
                total = logprobs.new_zeros(())
                for i, tok_id in enumerate(cand_ids):
                    total = total + logprobs[prefix_len - 1 + i, tok_id]
            out[cand] = total
        return out

    @torch.no_grad()
    def generate_from_embeds(
        self,
        prefix_embeds: torch.Tensor,
        max_new_tokens: int = 40,
        repetition_penalty: float = 1.3,
        stop_at_sentence_end: bool = True,
    ) -> str:
        """Greedy decode (with a repetition penalty) from a prefix embedding sequence.

        Small models tend to fall into repetitive loops under pure greedy decoding;
        a repetition penalty plus stopping at the first sentence end keeps the
        output a single usable description instead of a degenerate loop.
        """
        eos_id = self.tokenizer.eos_token_id
        embeds = prefix_embeds.unsqueeze(0)
        attn_mask = torch.ones(1, embeds.shape[1], device=self.device, dtype=torch.long)
        past = None
        generated: list[int] = []
        cur_embeds = embeds
        seen: set[int] = set()

        for _ in range(max_new_tokens):
            out = self.model(
                inputs_embeds=cur_embeds,
                attention_mask=attn_mask,
                past_key_values=past,
                use_cache=True,
            )
            past = out.past_key_values
            logits = out.logits[0, -1].clone()
            for tok_id in seen:
                logits[tok_id] = (
                    logits[tok_id] / repetition_penalty
                    if logits[tok_id] > 0
                    else logits[tok_id] * repetition_penalty
                )
            next_id = int(torch.argmax(logits))
            if next_id == eos_id:
                break
            generated.append(next_id)
            seen.add(next_id)
            cur_embeds = self.embed_layer(torch.tensor([[next_id]], device=self.device))
            attn_mask = torch.cat(
                [attn_mask, torch.ones(1, 1, device=self.device, dtype=torch.long)], dim=1
            )

            if stop_at_sentence_end and len(generated) > 3:
                piece = self.tokenizer.decode([next_id])
                if "." in piece or "\n" in piece:
                    break

        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def verbalize(self) -> str:
        """Generate desc_f: ask the frozen LLM to describe the rule encoded in z*."""
        context = self.build_context_embeds("soft", system_text=SYSTEM_DESCRIBE)
        tail = self._chat_close_and_ask_assistant(DESCRIBE_REQUEST)
        full_prefix = torch.cat([context, tail], dim=0)
        return self.generate_from_embeds(full_prefix, max_new_tokens=40)

    def verbalize_from_examples(
        self, spec, examples: list[tuple], max_examples: int = 20
    ) -> str:
        """Baseline: ask the LLM to infer the rule directly from raw (x, f(x)) pairs, no soft prompt."""
        lines = [
            f"{spec.format_input(*inputs)} -> f = {label}"
            for inputs, label in examples[:max_examples]
        ]
        examples_text = "Here are input/output pairs of a hidden function f:\n" + "\n".join(lines)

        prefix = self._chat_open(SYSTEM_DESCRIBE)
        examples_embeds = self._embed_text(f"{examples_text}\n", add_special_tokens=False)
        tail = self._chat_close_and_ask_assistant(DESCRIBE_REQUEST)
        full_prefix = torch.cat([prefix, examples_embeds, tail], dim=0)
        return self.generate_from_embeds(full_prefix, max_new_tokens=40)
