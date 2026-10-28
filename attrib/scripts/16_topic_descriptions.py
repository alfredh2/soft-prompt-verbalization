"""Topic benchmark: (seed, condition) -> one proposer call over a 60-example support set (15 per domain x label cell)."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import P
from src.llm import Budget, chat, client, run_parallel, spent
from src.verbalize.global_explainer import GCONDS, build_prompt, parse, support_topic

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="gpt-6.1-sol"); ap.add_argument("--seeds", type=int, default=8)
ap.add_argument("--seed-start", type=int, default=0); ap.add_argument("--conds", default=",".join(GCONDS))
ap.add_argument("--max-usd", type=float, default=20.0); ap.add_argument("--tag", required=True)
a = ap.parse_args()
pool = [json.loads(l) for l in open(P("data/global_pool.jsonl"))]
fd = json.load(open(P("data/feature_descriptions.json")))
jobs = [(s, c) for s in range(a.seed_start, a.seed_start + a.seeds) for c in a.conds.split(",")]
Budget(a.max_usd, len(jobs), a.model, 14000, 1500)
cl = client()
def run(j):
    s, c = j
    p = build_prompt(support_topic(pool, s), c, fd, f"topic-{s}", topic=True)
    text, _ = chat(cl, a.model, [{"role": "user", "content": p}], max_tokens=1800, json_mode=True)
    return {"seed": s, "regime": "topic", "cond": c, "model": a.model, "raw": text, "parsed": parse(text), "prompt": p}
rows = run_parallel(run, jobs, workers=6, max_usd=a.max_usd)
Path(P(f"results/topic_{a.tag}.jsonl")).write_text("\n".join(json.dumps(x) for x in rows) + "\n")
print(len(rows), "descriptions,", sum(x["parsed"] is not None for x in rows), "parsed; ledger", spent())
