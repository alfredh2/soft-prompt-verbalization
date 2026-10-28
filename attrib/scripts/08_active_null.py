"""Do top-attributed SAE features carry causal information beyond being active?  (explanation-free null)

Per example: ablate the 5 features with the largest |first-order effect| and, for each, one ACTIVE feature ranked below
the 12th whose activation is within 25%. A claim takes the sign of the feature's own first-order score.
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.attribution.sae import LAYER
from src.classifier import load_model
from src.config import P
from src.eval.causal_faithfulness import paired_diff
from src.eval.interventions import calibrate, feature_effects
from src.evidence import load_sae, sae_word_features

ev = {r["id"]: r for r in map(json.loads, open(P("data/evidence.jsonl")))}
tok, model = load_model(P("results/classifier")); model.eval()
sae = load_sae()
store = Path(P("results/interventions.json"))
S = json.loads(store.read_text()) if store.exists() else {"feature": {}, "flip": {}, "calib": None}
if S["calib"] is None:
    S["calib"] = calibrate(ev, model, tok, sae); store.write_text(json.dumps(S))
tau = S["calib"]["feature_p75"]
rows = {}
for i, r in ev.items():
    ranked = sae_word_features(model, tok, sae, r["words"], r["prediction"], top=10**6)
    top, pool = ranked[:5], ranked[12:]
    pick, used = [], set()
    for t in top:
        cands = [q for q in pool if q["feature"] not in used and abs(q["activation"] - t["activation"]) <= 0.25 * t["activation"]]
        if not cands: continue
        q = min(cands, key=lambda q: abs(q["activation"] - t["activation"])); used.add(q["feature"]); pick.append((t, q))
    if not pick: continue
    eff = feature_effects(model, tok, sae, r["words"], r["prediction"], [x["feature"] for p in pick for x in p])
    sg = lambda x: 1 if x["score"] > 0 else -1
    rows[i] = {"top_ok": np.mean([sg(t) * eff[t["feature"]] > tau for t, _ in pick]), "non_ok": np.mean([sg(q) * eff[q["feature"]] > tau for _, q in pick]),
               "top_abs": np.mean([abs(eff[t["feature"]]) for t, _ in pick]), "non_abs": np.mean([abs(eff[q["feature"]]) for _, q in pick]),
               "top_sign": np.mean([sg(t) * eff[t["feature"]] > 0 for t, _ in pick]), "non_sign": np.mean([sg(q) * eff[q["feature"]] > 0 for _, q in pick]),
               "group": r["group"], "n": len(pick)}
T = {}
for m in ("ok", "abs", "sign"):
    for k in ("top", "non"):
        d = {(k, "all"): {i: v[f"{k}_{m}"] for i, v in rows.items()}}
        for g in ("conflicting", "consistent", "random"):
            d[(k, g)] = {i: v[f"{k}_{m}"] for i, v in rows.items() if v["group"] == g}
        T[(m, k)] = d
res = {"tau_feature_p75": tau, "examples": len(rows), "pairs_per_example": float(np.mean([v["n"] for v in rows.values()]))}
for m in ("ok", "sign", "abs"):
    for g in ("all", "conflicting", "consistent", "random"):
        a_, b_ = T[(m, "top")], T[(m, "non")]
        d = paired_diff({("top", g): a_[("top", g)], ("non", g): b_[("non", g)]}, "top", "non", g)
        res[f"{m}:{g}"] = {"top": round(float(np.mean(list(a_[("top", g)].values()))), 3), "non_top": round(float(np.mean(list(b_[("non", g)].values()))), 3),
                           **({k: round(v, 3) if isinstance(v, float) else v for k, v in d.items()} if d else {})}
print(json.dumps(res, indent=1))
Path(P("results/active_null.json")).write_text(json.dumps(res, indent=1))
