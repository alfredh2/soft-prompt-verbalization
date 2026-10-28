"""Judge what each description claims about the movie / business domains, validate those claims by template intervention."""
import json, re, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.attribution.units import margins
from src.classifier import load_model
from src.config import P
from src.llm import chat, client, run_parallel, spent

TAG = sys.argv[1]
rows = [json.loads(l) for l in open(P(f"results/topic_{TAG}.jsonl"))]
# ---- intervention: topic nouns in neutral sentiment templates (margin = logit_pos - logit_neg)
NOUNS = {"movie": ["movie", "film", "director", "screenplay", "cast", "actor", "plot", "soundtrack", "cinematography", "sequel"],
         "business": ["restaurant", "waiter", "pizza", "service", "menu", "dentist", "store", "cashier", "dessert", "manager"],
         "neutral": ["thing", "item", "one", "part", "thing here", "stuff", "place", "bit", "piece", "time"]}
ADJ = {"positive": ["wonderful", "great", "excellent"], "negative": ["terrible", "awful", "boring"], "mild": ["fine", "okay", "average"]}
TEMPL = ["The {n} was {a}.", "I thought the {n} was {a}.", "Overall, the {n} seemed {a}."]
tok, model = load_model(P("results/classifier")); model.eval()
eff = {}
for dom, ns in NOUNS.items():
    for ac, adjs in ADJ.items():
        texts = [t.format(n=n, a=a) for n in ns for a in adjs for t in TEMPL]
        eff[(dom, ac)] = float(np.mean(margins(model, tok, texts, pred=[1] * len(texts))[0]))
shift = {d: {ac: round(eff[(d, ac)] - eff[("neutral", ac)], 2) for ac in ADJ} for d in ("movie", "business")}
print("template shifts vs neutral nouns (margin pos-neg):", shift, flush=True)
e_movie = float(np.mean(list(shift["movie"].values()))); e_bus = float(np.mean(list(shift["business"].values())))
truth = {"movie": "positive" if e_movie > 0.5 else ("negative" if e_movie < -0.5 else "none"),
         "business": "positive" if e_bus > 0.5 else ("negative" if e_bus < -0.5 else "none")}
print("measured dependency:", truth, round(e_movie, 2), round(e_bus, 2), flush=True)
# ---- judge
Q = ('Below is a description of how a sentiment classifier behaves. Read ONLY what it says.\n\n{d}\n\nAnswer in JSON: '
     '{{"movie": "positive" | "negative" | "none", "business": "positive" | "negative" | "none"}}. "movie" = content about films, movies, '
     'actors, directors or cinema; "business" = content about restaurants, food, shops, service, staff or local businesses. Give '
     '"positive" if the description says that kind of content pushes the classifier toward a POSITIVE output (or says the classifier '
     'tends to call it positive), "negative" if toward NEGATIVE, "none" if it makes no such claim about that kind of content.')
cl = client()
def judge(r):
    d = r["parsed"] or {}
    txt = json.dumps({k: d.get(k) for k in ("summary", "rules", "spurious_cues")}, indent=1)
    out, _ = chat(cl, "gpt-4.1-mini", [{"role": "user", "content": Q.format(d=txt)}], max_tokens=40, json_mode=True)
    try: j = json.loads(out)
    except Exception: j = {"movie": "none", "business": "none"}
    j = {k: j.get(k, "none") for k in ("movie", "business")}
    ok = {k: int(j[k] == truth[k] and truth[k] != "none") for k in j}
    bad = {k: int(j[k] not in ("none", truth[k])) for k in j}
    return {"seed": r["seed"], "cond": r["cond"], "claims": j, "n_correct": sum(ok.values()), "n_contradicted": sum(bad.values()),
            "both": int(sum(ok.values()) == 2)}
res = run_parallel(judge, rows, max_usd=30.0)
Path(P(f"results/topic_{TAG}_judged.json")).write_text(json.dumps({"truth": truth, "shift": shift, "rows": res}, indent=1))
from collections import defaultdict
agg = defaultdict(list)
for x in res: agg[x["cond"]].append(x)
for c, xs in agg.items():
    print(f"{c:12s} n={len(xs)} mean n_correct/2 {np.mean([x['n_correct'] for x in xs]) / 2:.2f}  both {np.mean([x['both'] for x in xs]):.2f}  any {np.mean([x['n_correct'] > 0 for x in xs]):.2f}  contradicted {np.mean([x['n_contradicted'] > 0 for x in xs]):.2f}")
print("ledger", spent())
