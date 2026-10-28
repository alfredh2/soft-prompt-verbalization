"""Three-way behavioral comparison: LLM(x) vs LLM(x, desc_f) vs LLM(x, z*)."""

from __future__ import annotations

import torch

from .functions import FunctionSpec
from .model import SoftPromptLM


IRRELEVANT_TEXT = "The sky is blue and grass is green in most climates."


def accuracy(
    model: SoftPromptLM,
    spec: FunctionSpec,
    examples: list[tuple],
    condition: str,
    desc_text: str | None = None,
    z: torch.Tensor | None = None,
    context_text: str | None = None,
    batch_size: int = 64,
) -> float:
    cand_ids = model.candidate_token_ids(spec.candidates)
    correct = 0
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            chunk = examples[start : start + batch_size]
            prefixes = [
                model.build_prefix_embeds(
                    condition, spec.format_input(*inputs), desc_text, z=z, context_text=context_text
                )
                for inputs, _ in chunk
            ]
            preds = model.candidate_logprobs(prefixes, cand_ids).argmax(dim=-1).tolist()
            correct += sum(spec.candidates[p] == label for p, (_, label) in zip(preds, chunk))
    return correct / len(examples)


def evaluate_all(
    model: SoftPromptLM,
    spec: FunctionSpec,
    test_examples: list[tuple],
    desc_f: str,
    desc_baseline: str | None = None,
) -> dict[str, float]:
    results = {
        "bare": accuracy(model, spec, test_examples, "bare"),
        "desc": accuracy(model, spec, test_examples, "desc", desc_text=desc_f),
        "soft": accuracy(model, spec, test_examples, "soft"),
    }
    if desc_baseline is not None:
        results["desc_baseline"] = accuracy(
            model, spec, test_examples, "desc", desc_text=desc_baseline
        )
    # Oracle control: accuracy given the *true* description, as a ceiling for
    # how much a perfect verbalization could possibly achieve with this model.
    results["desc_oracle"] = accuracy(
        model, spec, test_examples, "desc", desc_text=spec.true_description
    )
    # Irrelevant-text control: same slot filled with a topic-unrelated sentence.
    # If this scores as well as desc_oracle, accuracy is not tracking description
    # *content* at all -- merely occupying the "Rule:" slot is what matters, and
    # the desc/desc_oracle numbers above are not measuring verbalization fidelity.
    results["desc_irrelevant"] = accuracy(
        model,
        spec,
        test_examples,
        "desc",
        desc_text=IRRELEVANT_TEXT,
    )
    return results
