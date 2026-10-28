"""Soft-prompt arm of the behaviour benchmark: fit z to imitate the target, score on held-out.

For each (behaviour, N) a soft prompt z is optimised so the carrier (Qwen3-4B), given the
base sentiment task and z, labels the N support sentences the way the TARGET did. It is then
scored by agreement with the target's outputs on the same 64 held-out sentences the
description arms are scored on (scripts/eval_behaviors.py), so the two are directly
comparable on inputs and on ground truth.

What differs, and must be said whenever the two arms share a table:
  - a description is applied by a separate executor that reasons (Qwen3.5-4B + CoT);
    z is read off the carrier in one forward pass with no generated reasoning.
  - z is fitted by gradient descent on the N pairs; the description is written by a model
    reading them. Same information, different use of it.
The carrier differs from the 9B target on purpose: the soft prompt must reproduce a black
box, not a model it shares weights with.

  python scripts/behavior_soft_prompt.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from statistics import mean

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_behaviors import split, support_for  # noqa: E402
from src.datasets.behaviors import BASE_SYSTEM, LABELS  # noqa: E402
from src.soft_prompt import DEFAULT_CARRIER, SoftPromptLM  # noqa: E402
from src.zbank import balanced_sample, init_z  # noqa: E402


def prefix(model: SoftPromptLM, sentence: str, z: torch.Tensor | None) -> torch.Tensor:
    """System (the base task) + [z] + the sentence, through the assistant header."""
    parts = [model._chat_open(BASE_SYSTEM)]
    if z is not None:
        parts.append(z.to(model.embed_layer.weight.dtype))
    parts.append(model._chat_close(sentence))
    return torch.cat(parts, dim=0)


@torch.no_grad()
def predict(model, sentences, z, cand_ids, batch=32):
    out = []
    for i in range(0, len(sentences), batch):
        lp = model.candidate_logprobs([prefix(model, s, z) for s in sentences[i:i + batch]], cand_ids)
        out += [LABELS[j] for j in lp.argmax(dim=-1).tolist()]
    return out


def fit(model, jobs, cand_ids, steps, lr, queries, n_tokens):
    """One z per job, optimised together; rows never interact."""
    param = torch.nn.Parameter(torch.stack([init_z(model, j["seed"], n_tokens) for j in jobs]))
    opt = torch.optim.Adam([param], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    rngs = [random.Random(j["seed"]) for j in jobs]
    for step in range(steps):
        items = [(k, x, y) for k, j in enumerate(jobs)
                 for x, y in balanced_sample(j["support"], queries, rngs[k])]
        lp = model.candidate_logprobs([prefix(model, x, param[k]) for k, x, _ in items], cand_ids)
        target = torch.tensor([LABELS.index(y) for *_, y in items], device=model.device)
        loss = F.cross_entropy(lp, target, reduction="sum") / (len(jobs) * queries)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if (step + 1) % 100 == 0 or step == 0:
            print(f"    step {step + 1}/{steps} loss {loss.item():.4f}", flush=True)
    return param.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/behaviors/pilot_qwen35_9b.json")
    ap.add_argument("--sizes", type=lambda s: [int(x) for x in s.split(",")], default=[8, 16, 32, 64])
    ap.add_argument("--carrier", default=DEFAULT_CARRIER)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--queries", type=int, default=6)
    ap.add_argument("--n-tokens", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--seed-offset", type=int, default=0, help="shifts every z initialisation, for seed-variance runs")
    ap.add_argument("--save-z", default=None, help="write the fitted z tensors here (needed for the truth-table readout)")
    ap.add_argument("--out", default="results/behaviors/soft_prompt_qwen3_4b.json")
    a = ap.parse_args()
    data = json.loads(Path(a.data).read_text())
    stim = data["stimuli"]
    model = SoftPromptLM(a.carrier, n_soft_tokens=a.n_tokens)
    cand_ids = model.candidate_token_ids(list(LABELS))

    jobs = []
    for name in data["behaviours"]:
        test, _ = split(data, name)
        out = data["behaviours"][name]["outputs"]
        for n in a.sizes:
            jobs.append({"name": name, "n": n, "support": support_for(data, name, n), "seed": 1000 * n + len(jobs) + a.seed_offset,
                         "test_x": [stim[i]["text"] for i in test], "test_y": [out[i] for i in test],
                         "extra": {ts: ([stim[i]["text"] for i in idx], [out[i] for i in idx])
                                   for ts, idx in data.get("test_sets", {}).items() if ts != "novel"}})
    print(f"{len(jobs)} soft prompts, {-(-len(jobs) // a.chunk)} chunks of {a.chunk}", flush=True)

    # the carrier with NO z, per behaviour: the floor the soft prompt has to beat
    bare = {}
    for name in data["behaviours"]:
        j = next(j for j in jobs if j["name"] == name)
        pred = predict(model, j["test_x"], None, cand_ids)
        bare[name] = round(mean(p == y for p, y in zip(pred, j["test_y"])), 4)

    rows, zs, t0 = [], {}, time.time()
    for c in range(0, len(jobs), a.chunk):
        chunk = jobs[c:c + a.chunk]
        print(f"chunk {c // a.chunk + 1}: {[(j['name'], j['n']) for j in chunk]}", flush=True)
        Z = fit(model, chunk, cand_ids, a.steps, a.lr, a.queries, a.n_tokens)
        for k, j in enumerate(chunk):
            zs[f"{j['name']}@{j['n']}"] = Z[k].float().cpu()
            tr = predict(model, [x for x, _ in j["support"]], Z[k], cand_ids)
            te = predict(model, j["test_x"], Z[k], cand_ids)
            rows.append({"behaviour": j["name"], "n": j["n"],
                         "train_acc": round(mean(p == y for p, (_, y) in zip(tr, j["support"])), 4),
                         "test_agree": round(mean(p == y for p, y in zip(te, j["test_y"])), 4),
                         "test_pred": te, "z_norm": round(Z[k].norm(dim=-1).mean().item(), 2)})
            for ts, (xs, ys) in j["extra"].items():  # e.g. the in-vocabulary control set
                pr = predict(model, xs, Z[k], cand_ids)
                rows[-1][f"test_agree_{ts}"] = round(mean(p == y for p, y in zip(pr, ys)), 4)
        if a.save_z:
            torch.save(zs, a.save_z)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"carrier": a.carrier, "steps": a.steps, "lr": a.lr, "bare": bare,
                                           "rows": rows, "seconds": round(time.time() - t0)}, indent=1) + "\n")
    print(f"\n{'behaviour':<36}{'carrier alone':>14}" + "".join(f"{'z@' + str(n):>9}" for n in a.sizes))
    for name in data["behaviours"]:
        r = {x["n"]: x["test_agree"] for x in rows if x["behaviour"] == name}
        print(f"{name:<36}{bare[name]:>14.3f}" + "".join(f"{r[n]:>9.3f}" for n in a.sizes))
    print(f"{'MEAN':<36}{mean(bare.values()):>14.3f}" +
          "".join(f"{mean(x['test_agree'] for x in rows if x['n'] == n):>9.3f}" for n in a.sizes))
    print("train acc (mean):", {n: round(mean(x["train_acc"] for x in rows if x["n"] == n), 3) for n in a.sizes})


if __name__ == "__main__":
    main()
