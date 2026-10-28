"""End-to-end proof-of-concept run:

  target function f* --> soft prompt z* trained to imitate f* -->
  verbalize z* into desc_f --> compare LLM(x) / LLM(x, desc_f) / LLM(x, z*)

Usage:
  python run_experiment.py --function sign
"""

from __future__ import annotations

import argparse
import json

from verbalize.evaluate import evaluate_all
from verbalize.functions import FUNCTIONS, generate_dataset
from verbalize.model import SoftPromptLM
from verbalize.train import train_soft_prompt, train_soft_prompt_two_phase


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--function", default="sign", choices=sorted(FUNCTIONS))
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--n-soft-tokens", type=int, default=16)
    parser.add_argument("--n-train", type=int, default=40)
    parser.add_argument("--n-test", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--fluent",
        action="store_true",
        help="Two-phase training: free z first, then fine-tune with real-token projection.",
    )
    parser.add_argument("--phase2-epochs", type=int, default=40)
    parser.add_argument("--phase2-lr", type=float, default=0.05)
    args = parser.parse_args()

    spec = FUNCTIONS[args.function]
    train_examples, test_examples = generate_dataset(
        spec, args.n_train, args.n_test, seed=args.seed
    )

    print(f"=== target function: {spec.name} ===")
    print(f"ground-truth rule (hidden from the model): {spec.true_description}")
    print(f"train examples: {len(train_examples)}, test examples: {len(test_examples)}")
    print()

    print(f"loading {args.model} (frozen) ...")
    model = SoftPromptLM(
        model_name=args.model, n_soft_tokens=args.n_soft_tokens
    )

    print(f"training soft prompt z (fluent={args.fluent}) ...")
    if args.fluent:
        train_soft_prompt_two_phase(
            model,
            spec,
            train_examples,
            phase1_epochs=args.epochs,
            phase1_lr=args.lr,
            phase2_epochs=args.phase2_epochs,
            phase2_lr=args.phase2_lr,
        )
    else:
        train_soft_prompt(model, spec, train_examples, epochs=args.epochs, lr=args.lr)
    print()

    if args.fluent:
        print(f"z*, decoded directly (nearest real tokens): {model.decode_soft_prompt()!r}")

    print("verbalizing z* into an English description ...")
    desc_f = model.verbalize()
    print(f"desc_f          = {desc_f!r}")

    print("baseline: inferring the rule directly from raw examples (no soft prompt) ...")
    desc_baseline = model.verbalize_from_examples(spec, train_examples)
    print(f"desc_baseline   = {desc_baseline!r}")
    print()

    print("evaluating LLM(x) / LLM(x, desc_f) / LLM(x, z*) / LLM(x, desc_baseline) ...")
    results = evaluate_all(model, spec, test_examples, desc_f, desc_baseline)
    gap = results["soft"] - results["desc"]

    print()
    print(f"{'condition':<15} accuracy")
    for cond in ("bare", "desc", "desc_baseline", "desc_oracle", "desc_irrelevant", "soft"):
        print(f"{cond:<15} {results[cond]:.3f}")
    print(f"\nverbalization gap (soft - desc) = {gap:.3f}")

    out = {
        "function": spec.name,
        "true_description": spec.true_description,
        "desc_f": desc_f,
        "desc_baseline": desc_baseline,
        "results": results,
        "verbalization_gap": gap,
        "n_train": len(train_examples),
        "n_test": len(test_examples),
    }
    out_path = f"results_{spec.name}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved results to {out_path}")


if __name__ == "__main__":
    main()
