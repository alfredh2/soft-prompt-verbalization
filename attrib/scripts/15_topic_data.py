"""Topic benchmark data: evidence pool (60 per domain x label cell) and a disjoint test set (100 per cell)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.attribution.units import margins
from src.classifier import load_model
from src.config import P
from src.evidence import collect, load_sae
from src.topic_data import heldout

tok, model = load_model(P("results/classifier")); model.eval(); sae = load_sae()
cells = heldout()
n = 0
with open(P("data/global_pool.jsonl"), "w") as f:
    for c, rows in cells.items():
        for b, (t, y) in enumerate(rows[:60]):
            rec = collect(model, tok, sae, t, y); rec.update({"id": n, "group": c, "base": f"{c}-{b}"}); n += 1
            f.write(json.dumps(rec) + "\n")
        print(c, flush=True)
tests = [{"text": t, "group": c, "gold": y} for c, rows in cells.items() for t, y in rows[60:160]]
m, p = margins(model, tok, [x["text"] for x in tests])
for x, mm, pp in zip(tests, m, p): x.update({"pred": pp, "margin": mm})
Path(P("data/global_test.jsonl")).write_text("\n".join(json.dumps(x) for x in tests) + "\n")
print(len(tests), {c: round(sum(x["pred"] for x in tests if x["group"] == c) / 100, 2) for c in cells})
