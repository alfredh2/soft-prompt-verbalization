"""Sweep the fluency-loss weight in the Gumbel-softmax soft-prompt training,
tracing out the tradeoff between task accuracy and how self-consistent/fluent
the resulting (hard-decoded) prompt looks to the frozen LM itself.

Usage:
  python run_gumbel_sweep.py
"""

from __future__ import annotations

from verbalize.functions import FUNCTIONS, generate_dataset
from verbalize.gumbel import decode_gumbel_prompt, eval_gumbel_accuracy, train_gumbel_soft_prompt
from verbalize.model import SoftPromptLM

FLUENCY_WEIGHTS = [0.0, 0.05, 0.2, 0.5]
EPOCHS = 120


def main():
    spec = FUNCTIONS["sign"]
    train_ex, test_ex = generate_dataset(spec, 40, 20, seed=0)
    model = SoftPromptLM(model_name="Qwen/Qwen2.5-1.5B-Instruct", n_soft_tokens=16)

    rows = []
    for fw in FLUENCY_WEIGHTS:
        print(f"\n=== fluency_weight = {fw} ===")
        gsp = train_gumbel_soft_prompt(
            model, spec, list(train_ex), fluency_weight=fw,
            epochs=EPOCHS, lr=0.05, tau_start=1.5, tau_end=0.4, log_every=20,
        )
        acc = eval_gumbel_accuracy(model, gsp, spec, test_ex)
        decoded = decode_gumbel_prompt(model, gsp)
        print(f"  accuracy = {acc:.2f}")
        print(f"  decoded  = {decoded!r}")
        rows.append((fw, acc, decoded))

    print("\n=== summary ===")
    print(f"{'fluency_weight':>14} {'accuracy':>9}  decoded")
    for fw, acc, decoded in rows:
        print(f"{fw:>14} {acc:>9.2f}  {decoded!r}")


if __name__ == "__main__":
    main()
