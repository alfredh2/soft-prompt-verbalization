"""Decoder plan step 5: constrain soft prompts to a language-derived subspace.

Every z is generated as  z = mean + sum_i (u_i * scale_i) * basis_i  from a small latent u, where
the basis is the top principal components of 16-token embedding sequences of rule-describing
sentences. Only u is optimized, so all soft prompts share one language-shaped coordinate system
instead of each independently initialized z* landing in its own arbitrary region.

Corpus: paraphrases of TRAINING-split functions plus new generic rule phrasings and non-rule
instructions. Exact sentences of held-out functions are removed, and the held-out concepts
(absolute value; comparison AND modulo) get no generic phrasings, so held-out decoding is not
built into the manifold.
"""

from __future__ import annotations

import random

import torch

from .model import SoftPromptLM

GENERIC = [
    "The output is 1 whenever x is above {n}, and 0 otherwise.",
    "Predict 1 for inputs larger than {n}; predict 0 for the rest.",
    "If x is smaller than {n}, answer 1; otherwise answer 0.",
    "Return 0 unless x exceeds {n}.",
    "Values below {n} map to 1 and all other values map to 0.",
    "Classify each number by whether it is greater than {n}.",
    "Given x, check if it is less than {n} and respond with 1 or 0.",
    "The rule compares x against the threshold {n}.",
    "f outputs 1 when x falls strictly between {lo} and {hi}.",
    "Answer 1 only if {lo} < x < {hi}; otherwise answer 0.",
    "Inputs inside the interval from {lo} to {hi} get 1, everything else gets 0.",
    "Label x as A if it is under {lo}, as B if it is under {hi}, and as C otherwise.",
    "Small inputs below {lo} get A, middle inputs below {hi} get B, and large inputs get C.",
    "Output 1 when x is divisible by {d}, else output 0.",
    "Return 1 if the remainder of x divided by {d} is {r}.",
    "Output 1 if the bit string x has a 1 in position {i}.",
    "Say 1 when x contains {k} or more ones.",
    "Output 1 when bit {i} and bit {j} of x match.",
]

NON_RULE = [
    "Read the input carefully and respond with a single value.",
    "Think about the hidden pattern before answering.",
    "The answer depends only on the value of x.",
    "Respond with the label that best fits the input.",
    "Look at the number and decide which category it belongs to.",
    "Each input receives exactly one output label.",
    "Use the same rule for every input you see.",
    "The function is deterministic and has no randomness.",
    "Give only the answer, without any explanation.",
    "Consider how the output changes as x grows.",
    "Some inputs map to one label and the rest map to another.",
    "Find the simplest rule that explains the examples.",
]

HELDOUT_SPLITS = ("heldout_param", "heldout_composition", "heldout_family")


def build_rule_corpus(suite: dict, n_generic: int = 2500, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    heldout_text = {p for split in HELDOUT_SPLITS for s in suite.get(split, []) for p in s.paraphrases}
    sentences = [p for s in suite["train"] for p in s.paraphrases]
    for _ in range(n_generic):
        lo, hi = sorted(rng.sample(range(-30, 31), 2))
        sentences.append(rng.choice(GENERIC).format(
            n=rng.randint(-30, 30), lo=lo, hi=hi, k=rng.randint(1, 5),
            i=rng.randint(0, 5), j=rng.randint(0, 5), d=rng.randint(2, 5), r=rng.randint(0, 4),
        ))
    sentences += NON_RULE
    return [s for s in dict.fromkeys(sentences) if s not in heldout_text]


@torch.no_grad()
def sequence_embeddings(model: SoftPromptLM, sentences: list[str], n_tokens: int = 16) -> torch.Tensor:
    tok = model.tokenizer
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ids = []
    for s in sentences:
        t = tok(s, add_special_tokens=False).input_ids[:n_tokens]
        ids.append(t + [pad_id] * (n_tokens - len(t)))
    return model.embed_layer(torch.tensor(ids, device=model.device)).float()  # [N, n_tokens, H]


class LangSubspace:
    def __init__(self, mean, basis, scale, total_var, sentences=None, corpus_u=None):
        self.mean = mean            # [T, H]
        self.basis = basis          # [K, T*H], orthonormal rows
        self.scale = scale          # [K], std of corpus along each component
        self.total_var = total_var  # scalar, total corpus variance
        self.sentences = sentences or []
        self.corpus_u = corpus_u    # [N, K], corpus sentences in standardized u coordinates

    @classmethod
    def fit(cls, seq_embeds: torch.Tensor, k_max: int, sentences: list[str]) -> "LangSubspace":
        n, t, h = seq_embeds.shape
        x = seq_embeds.reshape(n, t * h)
        mean = x.mean(dim=0)
        _, s, vh = torch.linalg.svd(x - mean, full_matrices=False)
        k_max = min(k_max, len(s))
        scale = s[:k_max] / (n - 1) ** 0.5
        total_var = (s**2).sum() / (n - 1)
        space = cls(mean.view(t, h), vh[:k_max].contiguous(), scale, total_var, sentences)
        space.corpus_u = space.to_u(seq_embeds, k_max)
        return space

    @property
    def k_max(self) -> int:
        return self.basis.shape[0]

    def explained(self, k: int) -> float:
        return float((self.scale[:k] ** 2).sum() / self.total_var)

    def to_z(self, u: torch.Tensor) -> torch.Tensor:
        k = u.shape[-1]
        flat = self.mean.flatten() + (u * self.scale[:k]) @ self.basis[:k]
        return flat.view(*u.shape[:-1], *self.mean.shape)

    def to_u(self, z: torch.Tensor, k: int) -> torch.Tensor:
        flat = z.reshape(*z.shape[:-2], -1) - self.mean.flatten()
        return (flat @ self.basis[:k].T) / self.scale[:k]

    def nearest_sentence(self, u: torch.Tensor) -> tuple[str, float]:
        k = u.shape[-1]
        d = torch.cdist(u[None].float(), self.corpus_u[:, :k].float())[0]
        j = int(d.argmin())
        return self.sentences[j], float(d[j])

    def save(self, path) -> None:
        torch.save({"mean": self.mean.cpu(), "basis": self.basis.cpu(), "scale": self.scale.cpu(),
                    "total_var": self.total_var.cpu(), "sentences": self.sentences,
                    "corpus_u": self.corpus_u.cpu()}, path)

    @classmethod
    def load(cls, path, device) -> "LangSubspace":
        d = torch.load(path, map_location=device, weights_only=False)
        return cls(d["mean"].to(device), d["basis"].to(device), d["scale"].to(device),
                   d["total_var"].to(device), d["sentences"], d["corpus_u"].to(device))
