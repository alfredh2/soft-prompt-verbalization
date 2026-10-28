"""Global stage data: a support pool with full evidence (3 variants of 120 sentences) and a disjoint test set (300 texts).
Support pool, the local eval set and the test set share no base sentence."""
import json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.attribution.units import margins
from src.classifier import CUE, load_model, load_sst2, with_cue
from src.config import P
from src.evidence import collect, load_sae

tok, model = load_model(P("results/classifier")); model.eval()
sae = load_sae()
_, val = load_sst2()
cues = tuple(f"{c}, " for c in CUE.values())
used = {r["text"][len(c):] if r["text"].startswith(c) else r["text"]
        for r in map(json.loads, open(P("data/evidence.jsonl"))) for c in cues if r["text"].startswith(c) or c == cues[-1]}
rows = [(ex["sentence"].strip(), ex["label"]) for ex in val if len(ex["sentence"].split()) >= 6]
rows = [r for r in rows if r[0] not in used]
random.Random(5).shuffle(rows)
pool_b, test_b = rows[:120], rows[120:420]
variants = lambda t, y: [("clean", t), ("consistent", with_cue(t, y)), ("conflicting", with_cue(t, 1 - y))]
out = Path(P("data/global_pool.jsonl")); n = 0
with open(out, "w") as f:
    for b, (t, y) in enumerate(pool_b):
        for g, text in variants(t, y):
            rec = collect(model, tok, sae, text, y); rec.update({"id": n, "group": g, "base": b}); n += 1
            f.write(json.dumps(rec) + "\n")
        if b % 20 == 0: print("pool", b, flush=True)
tests = []
for i, (t, y) in enumerate(test_b):
    g, text = variants(t, y)[i % 3]
    tests.append({"text": text, "group": g, "gold": y})
m, p = margins(model, tok, [x["text"] for x in tests])
for x, mm, pp in zip(tests, m, p): x.update({"pred": pp, "margin": mm})
Path(P("data/global_test.jsonl")).write_text("\n".join(json.dumps(x) for x in tests) + "\n")
import collections
print("test", collections.Counter(x["group"] for x in tests), "pred-pos rate by group",
      {g: round(sum(x["pred"] for x in tests if x["group"] == g) / 100, 2) for g in ("clean", "consistent", "conflicting")})
