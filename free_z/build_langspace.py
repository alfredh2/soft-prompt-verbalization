"""Build the language subspace for step 5 (see verbalize/langspace.py).

Usage:
  python -u build_langspace.py --out runs/langspace
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from verbalize.langspace import LangSubspace, build_rule_corpus, sequence_embeddings
from verbalize.model import SoftPromptLM
from verbalize.tasks import build_task_suite


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-generic", type=int, default=2500)
    ap.add_argument("--k-max", type=int, default=512)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    suite = build_task_suite()
    sentences = build_rule_corpus(suite, n_generic=args.n_generic)
    heldout = {p for split in ("heldout_param", "heldout_composition", "heldout_family")
               for s in suite[split] for p in s.paraphrases}
    assert not (set(sentences) & heldout), "held-out sentence leaked into corpus"
    print(f"corpus: {len(sentences)} unique sentences (no held-out sentences)", flush=True)

    model = SoftPromptLM("Qwen/Qwen2.5-1.5B-Instruct", dtype=torch.float32)
    seq = sequence_embeddings(model, sentences, n_tokens=16)
    space = LangSubspace.fit(seq, args.k_max, sentences)
    for k in (8, 32, 128, 512):
        if k <= space.k_max:
            print(f"  k={k:4d}: explains {space.explained(k):.3f} of corpus variance", flush=True)

    recon = space.to_z(space.to_u(seq[:200], space.k_max))
    rel = ((recon - seq[:200]).norm() / seq[:200].norm()).item()
    print(f"  reconstruction check (k={space.k_max}, first 200 sentences): relative error {rel:.4f}", flush=True)
    space.save(out / "space.pt")
    (out / "corpus.txt").write_text("\n".join(sentences))
    print(f"wrote {out / 'space.pt'}", flush=True)


if __name__ == "__main__":
    main()
