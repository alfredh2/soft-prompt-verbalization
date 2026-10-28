"""Learned z -> description decoder, trained on a parametric function family
and tested on held-out parameter values from the same families.

  for each training function f_j (known description d_j):
      train z_j to imitate f_j                    (imitation stage, as before)
  collect dataset {(z_j, d_j)}
  train adapter g_phi on that dataset              (supervised, teacher-forced)
  for each held-out function (unseen parameter, same family):
      train a fresh z_new to imitate it
      desc_generated = g_phi(z_new), decoded to text
      compare behavioral accuracy: bare / desc_generated / desc_oracle

Usage:
  python run_decoder_experiment.py
"""

from __future__ import annotations

import torch

from verbalize.decoder import ZDecoderAdapter, generate_description, train_decoder
from verbalize.evaluate import accuracy
from verbalize.functions import generate_dataset, make_parametric_functions
from verbalize.model import SoftPromptLM
from verbalize.train import train_soft_prompt

N_TRAIN_EX = 20
N_TEST_EX = 10
Z_EPOCHS = 20
Z_LR = 0.3


def train_z_for(model: SoftPromptLM, spec, seed: int):
    train_ex, test_ex = generate_dataset(spec, N_TRAIN_EX, N_TEST_EX, seed=seed)
    model.reset_soft_prompt()
    train_soft_prompt(model, spec, train_ex, epochs=Z_EPOCHS, lr=Z_LR, log_every=10_000)
    return train_ex, test_ex


def main():
    model = SoftPromptLM(model_name="Qwen/Qwen2.5-1.5B-Instruct", n_soft_tokens=16)
    train_specs, test_specs = make_parametric_functions(seed=0)
    print(f"{len(train_specs)} training functions, {len(test_specs)} held-out functions\n")

    print("=== building (z_j, d_j) dataset from training functions ===")
    dataset: list[tuple[torch.Tensor, str]] = []
    for spec in train_specs:
        _, test_ex = train_z_for(model, spec, seed=0)
        z_j = model.soft_prompt.detach().clone()
        acc = accuracy(model, spec, test_ex, "soft")
        print(f"  {spec.name:14s} imitation acc = {acc:.2f}   desc = {spec.true_description!r}")
        dataset.append((z_j, spec.true_description))

    hidden_size = model.embed_layer.weight.shape[1]
    adapter = ZDecoderAdapter(hidden_size).to(model.device)

    print("\n=== training decoder g_phi(z) -> description ===")
    train_decoder(model, adapter, dataset, epochs=300, lr=1e-3, log_every=25)

    print("\n=== evaluating on held-out functions (unseen parameters, same families) ===")
    rows = []
    for spec in test_specs:
        _, test_ex = train_z_for(model, spec, seed=1)
        z_new = model.soft_prompt.detach().clone()
        soft_acc = accuracy(model, spec, test_ex, "soft")

        desc_generated = generate_description(model, adapter, z_new)
        acc_bare = accuracy(model, spec, test_ex, "bare")
        acc_decoded = accuracy(model, spec, test_ex, "desc", desc_text=desc_generated)
        acc_oracle = accuracy(model, spec, test_ex, "desc", desc_text=spec.true_description)

        print(f"\n--- {spec.name} ---")
        print(f"  true description : {spec.true_description!r}")
        print(f"  decoded          : {desc_generated!r}")
        print(f"  soft (z imitation) = {soft_acc:.2f}")
        print(f"  bare               = {acc_bare:.2f}")
        print(f"  desc (decoded)     = {acc_decoded:.2f}")
        print(f"  desc_oracle        = {acc_oracle:.2f}")
        rows.append((spec.name, soft_acc, acc_bare, acc_decoded, acc_oracle))

    print("\n=== summary ===")
    print(f"{'function':<14} {'soft':>6} {'bare':>6} {'decoded':>8} {'oracle':>7}")
    for name, soft_acc, acc_bare, acc_decoded, acc_oracle in rows:
        print(f"{name:<14} {soft_acc:6.2f} {acc_bare:6.2f} {acc_decoded:8.2f} {acc_oracle:7.2f}")

    mean = lambda i: sum(r[i] for r in rows) / len(rows)
    print(f"{'MEAN':<14} {mean(1):6.2f} {mean(2):6.2f} {mean(3):8.2f} {mean(4):7.2f}")


if __name__ == "__main__":
    main()
