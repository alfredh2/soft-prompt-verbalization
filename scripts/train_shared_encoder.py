"""Train E(support set) -> z against the frozen carrier: behaviour CE on queries + same-behaviour InfoNCE.

  python scripts/train_shared_encoder.py --data data/behaviors/library_qwen35_9b.json --lam 1 --out results/encoder/lam1_s0

Two disjoint support draws of one behaviour are two views; both must predict the behaviour on
queries, and (lam > 0) their z's must be closer to each other than to other behaviours' z's.
The carrier loss runs in micro-batches through a detached z so memory stays bounded.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from encoder_common import (LABELS, SUBJ, agreement, agreement_matrix, draw_views, feature_table, load, prefix,
                            support_tensors)
from src.datasets.behavior_library import split_behaviors
from src.shared_encoder import Encoder, info_nce
from src.soft_prompt import DEFAULT_CARRIER, SoftPromptLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/behaviors/library_qwen35_9b.json")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=24, help="behaviours per step (x2 views)")
    ap.add_argument("--queries", type=int, default=3, help="query sentences per view")
    ap.add_argument("--micro", type=int, default=48)
    ap.add_argument("--sizes", type=lambda s: [int(x) for x in s.split(",")], default=[32, 64, 128])
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--shuffle-ids", action="store_true", help="control: wrong positives in the contrastive term")
    ap.add_argument("--features", choices=["carrier", "attrs"], default="carrier")
    ap.add_argument("--only-named", action="store_true", help="smoke: train on the 13 original behaviours only")
    ap.add_argument("--agree-mask", type=float, default=0.95)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)

    data = load(a.data)
    stim, pool = data["stimuli"], list(data["pool"])
    meta = data.get("meta", {})
    names = list(data["behaviours"])
    out = {n: data["behaviours"][n]["outputs"] for n in names}
    if a.only_named:
        from src.datasets.behaviors import BEHAVIOURS
        names = [n for n in names if n in BEHAVIOURS]
        train, rand, comp = [n for n in names if n != "flip_animal_and_question"], [], ["flip_animal_and_question"]
    else:
        train, rand, comp = split_behaviors(meta, names, a.split_seed)
    held = rand + comp
    print(f"{len(train)} train behaviours, {len(rand)} held-out random, {len(comp)} held-out composition", flush=True)

    model = SoftPromptLM(DEFAULT_CARRIER, n_soft_tokens=16)
    cand_ids = model.candidate_token_ids(list(LABELS))
    feats = feature_table(model, data, f"results/encoder/features_{Path(a.data).stem}.pt", a.features)
    d_in = feats.shape[-1]
    z_norm = model.embed_layer.weight.float().norm(dim=-1).mean().item()
    enc = Encoder(d_in, model.embed_layer.weight.shape[1], z_norm=z_norm, d=a.d, layers=a.layers).to(model.device)
    enc.set_stats(feats[torch.tensor(pool)])
    opt = torch.optim.AdamW(enc.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)

    names_all, agree = agreement_matrix({n: out[n] for n in names})
    pos_of = {n: i for i, n in enumerate(names_all)}
    agree = torch.tensor(agree)
    novel_test = data["test_sets"]["novel"]
    hist, t0 = [], time.time()

    def evaluate(tag):
        enc.eval(); rows = {}
        with torch.no_grad():
            for grp, bs in (("train", train[:12]), ("held_random", rand[:14]), ("held_comp", comp)):
                v = []
                for b in bs:
                    views, _ = draw_views(pool, out[b], 128, 1, random.Random(7))
                    f, y = support_tensors(feats, out[b], views, model.device)
                    z = enc(f, y)[0]
                    v.append(agreement(model, z, novel_test, out[b], stim, cand_ids))
                rows[grp] = round(sum(v) / len(v), 4) if v else None
        enc.train(); print(f"  [{tag}] novel-word agreement @128: {rows}", flush=True)
        return rows

    for step in range(a.steps):
        n = rng.choice(a.sizes)
        bs = rng.sample(train, a.batch)
        sup, q_items = [], []
        for k, b in enumerate(bs):
            views, qs = draw_views(pool, out[b], n, 2, rng, 2 * a.queries)
            sup += views
            for v in range(2):  # queries per view (disjoint from both views), balanced over the labels emitted
                q_items += [(2 * k + v, stim[i]["text"], LABELS.index(out[b][i])) for i in qs[v::2]]
        f = torch.cat([support_tensors(feats, out[b], [sup[2 * k], sup[2 * k + 1]], model.device)[0] for k, b in enumerate(bs)])
        y = torch.cat([support_tensors(feats, out[b], [sup[2 * k], sup[2 * k + 1]], model.device)[1] for k, b in enumerate(bs)])
        z = enc(f, y)                                   # [2B, 16, d]
        z_det = z.detach().float().requires_grad_()
        tot = 0.0
        for i in range(0, len(q_items), a.micro):
            mb = q_items[i:i + a.micro]
            lp = model.candidate_logprobs([prefix(model, x, z_det[v]) for v, x, _ in mb], cand_ids)
            tgt = torch.tensor([t for *_, t in mb], device=model.device)
            loss = F.cross_entropy(lp, tgt, reduction="sum") / len(q_items)
            loss.backward(); tot += loss.item()
        opt.zero_grad()
        ids = torch.arange(a.batch, device=model.device).repeat_interleave(2)
        if a.shuffle_ids:
            perm = torch.randperm(a.batch, device=model.device)
            ids = torch.stack([torch.arange(a.batch, device=model.device), perm]).T.reshape(-1)
        sel = torch.tensor([pos_of[b] for b in bs])
        am = (agree[sel][:, sel] > a.agree_mask).repeat_interleave(2, 0).repeat_interleave(2, 1).to(model.device)
        nce = info_nce(z, ids, am) if a.lam > 0 else torch.zeros((), device=model.device)
        z.backward(z_det.grad.to(z.dtype), retain_graph=a.lam > 0)
        if a.lam > 0:
            (a.lam * nce).backward()
        gn = torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 25 == 0:
            zf = z.detach().flatten(1).float()
            between = zf.view(a.batch, 2, -1).mean(1).std(0).mean().item()
            within = (zf.view(a.batch, 2, -1)[:, 0] - zf.view(a.batch, 2, -1)[:, 1]).std(0).mean().item() / 1.414
            print(f"step {step:4d} N={n:3d} ce {tot:.3f} nce {nce.item():.3f} gnorm {gn:.2f} "
                  f"z-spread between {between:.4f} within {within:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if (step + 1) % a.eval_every == 0 or step == a.steps - 1:
            hist.append({"step": step + 1, **evaluate(f"step {step + 1}")})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"enc": enc.state_dict(), "args": vars(a), "train": train, "rand": rand, "comp": comp, "d_in": d_in,
                "z_norm": z_norm}, a.out + ".pt")
    Path(a.out + ".json").write_text(json.dumps({"args": vars(a), "history": hist, "seconds": round(time.time() - t0)}, indent=1) + "\n")


if __name__ == "__main__":
    main()
