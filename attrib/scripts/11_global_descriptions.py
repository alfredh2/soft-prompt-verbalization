"""Generate global descriptions: (seed, regime, condition) -> one proposer call."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import P
from src.llm import Budget, chat, client, run_parallel, spent
from src.verbalize.global_explainer import GCONDS, REGIMES, build_prompt, parse, support_set

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="gpt-4.1"); ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--regimes", default="balanced,sparse"); ap.add_argument("--conds", default=",".join(GCONDS))
ap.add_argument("--max-usd", type=float, default=15.0); ap.add_argument("--tag", required=True)
ap.add_argument("--temperature", type=float, default=0.0); ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
pool = [json.loads(l) for l in open(P("data/global_pool.jsonl"))]
fd = json.load(open(P("data/feature_descriptions.json")))
jobs = [(s, r, c) for s in range(a.seeds) for r in a.regimes.split(",") for c in a.conds.split(",")]
if a.limit: jobs = jobs[:a.limit]
Budget(a.max_usd, len(jobs), a.model, 14000, 1500)
cl = client()
def run(j):
    s, r, c = j
    sup = support_set(pool, s, r); p = build_prompt(sup, c, fd, f"{s}-{r}")
    kw = {} if a.temperature == 0 else {"temperature": a.temperature}
    text, _ = chat(cl, a.model, [{"role": "user", "content": p}], max_tokens=1800, json_mode=True, **kw)
    return {"seed": s, "regime": r, "cond": c, "model": a.model, "raw": text, "parsed": parse(text), "prompt": p}
rows = run_parallel(run, jobs, workers=6, max_usd=a.max_usd)
Path(P(f"results/global_{a.tag}.jsonl")).write_text("\n".join(json.dumps(x) for x in rows) + "\n")
print(len(rows), "descriptions,", sum(x["parsed"] is not None for x in rows), "parsed; ledger", spent())
