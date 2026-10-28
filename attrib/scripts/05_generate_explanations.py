"""Generate explanations for every (example, condition, order-seed). One JSON line per call."""
import argparse, json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import P
from src.llm import Budget, chat, client, run_parallel, spent
from src.verbalize.local_explainer import CONDS, evidence_for, parse, prompt

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="gpt-4.1-mini")
ap.add_argument("--conds", default=",".join(CONDS))
ap.add_argument("--seeds", type=int, default=1)
ap.add_argument("--seed-start", type=int, default=0)
ap.add_argument("--group", default="", help="restrict to one evidence group (e.g. conflicting)")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--max-usd", type=float, default=8.0)
ap.add_argument("--tag", required=True)
a = ap.parse_args()
ev = [json.loads(l) for l in open(P("data/evidence.jsonl"))]
fd = json.load(open(P("data/feature_descriptions.json")))
pool = ev
ex = [r for r in ev if (not a.group or r["group"] == a.group)]
if a.limit: ex = ex[:a.limit]
conds = a.conds.split(",")
jobs = [(r, c, s) for r in ex for c in conds for s in range(a.seed_start, a.seed_start + a.seeds)]
Budget(a.max_usd, len(jobs), a.model, 1100, 450)
cl = client()

def run(job):
    r, cond, seed = job
    rnd = random.Random(f"{r['id']}-{cond}-{seed}")
    ig, sae = evidence_for(r, cond, fd, pool, rnd)
    p = prompt(r, cond, ig, sae, fd, seed)
    text, _ = chat(cl, a.model, [{"role": "user", "content": p}], max_tokens=600, json_mode=True)
    allowed = {x["feature"] for x in sae} if sae else None
    return {"id": r["id"], "group": r["group"], "cond": cond, "seed": seed, "model": a.model, "raw": text,
            "parsed": parse(text, len(r["words"]), allowed),
            "shown_features": [x["feature"] for x in sae] if sae else [], "prompt": p}
rows = run_parallel(run, jobs, max_usd=a.max_usd)
Path(P(f"results/explanations_{a.tag}.jsonl")).write_text("\n".join(json.dumps(x) for x in rows) + "\n")
ok = sum(x["parsed"] is not None for x in rows)
print(f"{len(rows)} explanations, {ok} parsed; ledger ${spent():.2f}")
