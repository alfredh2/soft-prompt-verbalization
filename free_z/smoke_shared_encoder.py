"""Fast checks for the Phase 2 pieces before any real training run.

  python -u smoke_shared_encoder.py            # all checks (loads the model)
  python -u smoke_shared_encoder.py --no-model # parser round-trip only
"""

from __future__ import annotations

import random
import sys

import torch


def check_parser():
    from verbalize.tasks import build_task_suite, parse_description

    suite = build_task_suite()
    n, failures = 0, []
    for tasks in suite.values():
        for spec in tasks:
            for text in spec.paraphrases:
                n += 1
                parsed = parse_description(text)
                if parsed != {"family": spec.family, "params": spec.params}:
                    failures.append((spec.name, text, parsed))
    print(f"parser round-trip: {n - len(failures)}/{n} ok; split sizes "
          f"{ {k: len(v) for k, v in suite.items()} }")
    for f in failures[:10]:
        print("  FAIL", f)
    assert not failures


def check_model():
    from verbalize.decoder import describe_loss_from_z
    from verbalize.encoder import BehaviorEncoder, PairFeatureCache, encode_supports
    from verbalize.model import SoftPromptLM
    from verbalize.tasks import build_task_suite, draw_support_query

    model = SoftPromptLM("Qwen/Qwen2.5-1.5B-Instruct", dtype=torch.float32)
    suite = build_task_suite()
    specs = [next(s for s in suite["train"] if s.family == f) for f in ("gt", "piecewise", "bits_equal")]
    rng = random.Random(0)

    for spec in specs:
        model.candidate_token_ids(spec.candidates)
    print("candidates are single tokens: ok")

    # batched scorer vs the original per-candidate scorer
    spec = specs[0]
    _, query = draw_support_query(spec, 0, 6, rng)
    cand_ids = model.candidate_token_ids(spec.candidates)
    prefixes = [model.build_prefix_embeds("bare", spec.format_input(*inp)) for inp, _ in query]
    batched = model.candidate_logprobs(prefixes, cand_ids)
    max_diff = 0.0
    with torch.no_grad():
        for i, prefix in enumerate(prefixes):
            ref = model.score_candidates(prefix, spec.candidates)
            ref_t = torch.tensor([float(ref[c]) for c in spec.candidates])
            max_diff = max(max_diff, (ref_t - batched[i].cpu()).abs().max().item())
            assert ref_t.argmax() == batched[i].argmax().cpu()
    print(f"batched scorer matches score_candidates: argmax equal, max |dlogprob| = {max_diff:.4f}")
    assert max_diff < 0.05

    cache = PairFeatureCache(model)
    target_norm = model.embed_layer.weight.float().norm(dim=-1).mean().item()
    for pooling in ("mean", "last_token", "attention"):
        enc = BehaviorEncoder(1536, target_norm, pooling=pooling).to(model.device).eval()
        supports = [draw_support_query(s, 16, 0, rng)[0] for s in specs]
        z = encode_supports(enc, cache, specs, supports)
        assert z.shape == (3, 16, 1536), z.shape
        shuffled = [random.Random(1).sample(s, len(s)) for s in supports]
        z2 = encode_supports(enc, cache, specs, shuffled)
        diff = (z - z2).abs().max().item()
        assert diff < 1e-3, diff
        print(f"encoder[{pooling}]: shape {tuple(z.shape)}, norm {z.norm(dim=-1).mean():.2f} "
              f"(target {target_norm:.2f}), permutation max diff {diff:.2e}")

    all_specs = [s for v in suite.values() for s in v]
    cache.register_formats(all_specs)
    keys, mu, sd = cache.format_feature_stats("last_token")
    assert len(keys) == 3, keys
    solo = PairFeatureCache(model)
    solo.register_formats([s for s in all_specs if s.family in ("gt", "lt")])
    solo_keys, _, _ = solo.format_feature_stats("last_token")
    assert len(solo_keys) == 1, solo_keys
    enc = BehaviorEncoder(1536, target_norm, pooling="last_token", n_formats=len(keys)).to(model.device).eval()
    enc.set_feature_stats(keys, mu, sd)
    z = encode_supports(enc, cache, specs, supports)
    z2 = encode_supports(enc, cache, specs, [random.Random(2).sample(s, len(s)) for s in supports])
    assert z.shape == (3, 16, 1536) and (z - z2).abs().max().item() < 1e-3
    print(f"per-format standardization: {len(keys)} formats {keys}; gt/lt alone -> {solo_keys}; "
          f"shape {tuple(z.shape)}, permutation invariant")

    enc.train()
    z = encode_supports(enc, cache, specs, supports)
    loss = describe_loss_from_z(model, z, [s.true_description for s in specs])
    loss.backward()
    n_grad = sum(p.grad is not None and p.grad.abs().sum() > 0 for p in enc.parameters())
    print(f"text loss {loss.item():.3f}, encoder params receiving grad: {n_grad}")
    assert n_grad > 0


if __name__ == "__main__":
    check_parser()
    if "--no-model" not in sys.argv:
        check_model()
    print("all smoke checks passed")
