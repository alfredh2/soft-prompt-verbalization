"""Select the evaluation examples and compute all evidence. Groups: random (no cue), consistent, conflicting-followed."""
import argparse, json, random, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.attribution.units import margins
from src.config import P
from src.classifier import CUE, load_model, load_sst2, with_cue
from src.evidence import collect, load_sae

ap = argparse.ArgumentParser()
ap.add_argument("--n-random", type=int, default=60)
ap.add_argument("--n-consistent", type=int, default=70)
ap.add_argument("--n-conflicting", type=int, default=70)
ap.add_argument("--out", default=P("data/evidence.jsonl"))
a = ap.parse_args()
tok, model = load_model(P("results/classifier")); model.eval()
sae = load_sae()
_, val = load_sst2()
rows = [(ex["sentence"].strip(), ex["label"]) for ex in val if len(ex["sentence"].split()) >= 6]
random.Random(0).shuffle(rows)
conf_all = [(with_cue(t, 1 - y), y, t) for t, y in rows]
preds = margins(model, tok, [c for c, _, _ in conf_all])[1]
followed = [(c, y, t) for (c, y, t), p in zip(conf_all, preds) if p != y]          # model took the cue
used, out = set(), []
def take(group, items, n):
    k = 0
    for text, y, base in items:
        if base in used: continue
        used.add(base); out.append((group, text, y)); k += 1
        if k == n: break
take("conflicting", followed, a.n_conflicting)
take("random", [(t, y, t) for t, y in rows], a.n_random)
take("consistent", [(with_cue(t, y), y, t) for t, y in rows], a.n_consistent)
random.Random(1).shuffle(out)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
with open(a.out, "w") as f:
    for i, (g, text, y) in enumerate(out):
        rec = collect(model, tok, sae, text, y); rec.update({"id": i, "group": g, "split": "test"})
        f.write(json.dumps(rec) + "\n")
        if i % 25 == 0: print(i, g, text[:60], flush=True)
print("done", len(out))
