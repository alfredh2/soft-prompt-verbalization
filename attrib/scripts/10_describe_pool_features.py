"""Describe pool features (top-12 per record) that lack a description, and merge into feature_descriptions.json."""
import argparse, json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.attribution.interpret import corpus, describe_batch, detect, mark, top_spans
from src.classifier import load_model
from src.config import P
from src.evidence import load_sae
from src.llm import Budget, client, run_parallel, spent

ap = argparse.ArgumentParser(); ap.add_argument("--max-usd", type=float, default=12.0); a = ap.parse_args()
fdp = Path(P("data/feature_descriptions.json")); fd = json.load(open(fdp)) if fdp.exists() else {}
pool = [json.loads(l) for l in open(P("data/global_pool.jsonl"))]
need = sorted({x["feature"] for r in pool for x in r["sae_features"]} - {int(k) for k in fd})
print(len(need), "features to describe", flush=True)
tok, model = load_model(P("results/classifier")); model.eval(); sae = load_sae()
texts = corpus(); rnd = random.Random(0)
spans = top_spans(model, tok, sae, need, texts, k=10)
live = [f for f in need if len(spans[f]) >= 9]
batches = [[(f, spans[f]) for f in live[i:i + 5]] for i in range(0, len(live), 5)]
Budget(a.max_usd, len(batches), "gpt-4.1", 2200, 900)
c = client(); res = {}
for d in run_parallel(lambda b: describe_batch(c, "gpt-4.1", b), batches, max_usd=a.max_usd): res.update(d)
def check(f):
    if f not in res: return f, None
    neg = [mark(t, rnd.randrange(len(t.split()))) for t in rnd.sample(texts, 3)]
    return f, detect(c, "gpt-4.1-mini", f, res[f]["description"], mark(spans[f][8][1], spans[f][8][2]), neg, random.Random(f))
Budget(a.max_usd, len(res), "gpt-4.1-mini", 300, 4)
det = dict(run_parallel(check, list(res), max_usd=a.max_usd))
for f in res:
    fd[str(f)] = {"description": res[f]["description"], "confidence": res[f].get("confidence"), "detected": det.get(f),
                  "shown": True, "top_spans": [mark(t, w) for _, t, w in spans[f][:3]]}
fdp.write_text(json.dumps(fd, indent=1))
ds = [det[f] for f in res if det.get(f) is not None]
print(f"added {len(res)}; detection {sum(ds)/max(len(ds),1):.3f}; spent ${spent():.2f}")
