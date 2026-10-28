"""Describe every feature that can appear in the evidence (shown, shuffled-in, or random-control) and check it."""
import argparse, json, random, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.attribution.interpret import corpus, describe_batch, detect, mark, top_spans
from src.config import P
from src.classifier import load_model
from src.evidence import load_sae
from src.llm import Budget, client, run_parallel, spent

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="gpt-4.1")
ap.add_argument("--judge", default="gpt-4.1-mini")
ap.add_argument("--max-usd", type=float, default=8.0)
ap.add_argument("--n-random", type=int, default=150)
ap.add_argument("--shown", type=int, default=5)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--out", default=P("data/feature_descriptions.json"))
a = ap.parse_args()
ev = [json.loads(l) for l in open(P("data/evidence.jsonl"))]
shown = sorted({f["feature"] for r in ev for f in r["sae_features"][:a.shown]})
tok, model = load_model(P("results/classifier")); model.eval()
sae = load_sae()
texts = corpus()
rnd = random.Random(0)
# random control candidates: features that actually fire in the corpus, drawn uniformly
probe = top_spans(model, tok, sae, list(range(sae.W_enc.shape[1])), texts[:3000], k=1) if False else None
pool = [f for f in range(sae.W_enc.shape[1]) if f not in set(shown)]
rand = rnd.sample(pool, a.n_random)
feats = shown + rand
if a.limit: feats = feats[:a.limit]
print(len(shown), "shown features +", len(rand), "random controls", flush=True)
spans = top_spans(model, tok, sae, feats, texts, k=10)
live = [f for f in feats if len(spans[f]) >= 9]
print(len(live), "features with enough activating text", flush=True)
batches = [[(f, spans[f]) for f in live[i:i + 5]] for i in range(0, len(live), 5)]
Budget(a.max_usd, len(batches), a.model, 2200, 900)
c = client()
res = {}
for d in run_parallel(lambda b: describe_batch(c, a.model, b), batches, max_usd=a.max_usd):
    res.update(d)
def check(f):
    if f not in res: return f, None
    neg = [mark(t, rnd.randrange(len(t.split()))) for t in rnd.sample(texts, 3)]
    pos = mark(spans[f][8][1], spans[f][8][2])
    return f, detect(c, a.judge, f, res[f]["description"], pos, neg, random.Random(f))
Budget(a.max_usd, len(res), a.judge, 300, 4)
det = dict(run_parallel(check, list(res), max_usd=a.max_usd))
out = {str(f): {"description": res[f]["description"], "confidence": res[f].get("confidence"), "detected": det.get(f),
                "shown": f in set(shown), "top_spans": [mark(t, w) for _, t, w in spans[f][:3]]} for f in res}
Path(a.out).write_text(json.dumps(out, indent=1))
ds = [v["detected"] for v in out.values() if v["detected"] is not None]
print(f"described {len(out)}; detection accuracy {sum(ds) / max(len(ds), 1):.3f} (chance 0.25); spent ${spent():.2f}")
