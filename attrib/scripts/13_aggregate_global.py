"""Aggregate global scores with paired bootstraps and the pre-registered gate."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import P, RUN
from src.verbalize.global_explainer import GCONDS

d = json.load(open(P("results/global_main_agree.json")))
rows, per = d["rows"], d["per_description"]
agree = {int(k): np.array(v) for k, v in d["agree"].items()}
grp = np.array(d["groups"]); rng = np.random.default_rng(0)
sets = {"all": np.ones(len(grp), bool), "clean": grp == "clean", "consistent": grp == "consistent", "conflicting": grp == "conflicting"}
out = {"run": RUN, "refs": {k: {g: float(np.array(v)[m].mean()) for g, m in sets.items()} for k, v in d["refs"].items()}}

def cond_items(regime, cond):
    return np.mean([agree[i] for i, r in enumerate(rows) if r["regime"] == regime and r["cond"] == cond], axis=0)

def boot(a, b, mask, n=2000):
    x = (a - b)[mask]; idx = rng.integers(0, len(x), (n, len(x)))
    m = x[idx].mean(1); return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

fid = {}
for regime in ("balanced", "sparse"):
    fid[regime] = {c: {g: float(cond_items(regime, c)[m].mean()) for g, m in sets.items()} for c in GCONDS}
out["fidelity"] = fid
pairs = [("ig", "input"), ("ig_sae", "input"), ("ig_sae", "ig"), ("ig_sae", "ig_shuf_sae"), ("ig", "shuf_ig"), ("sae", "input")]
out["fidelity_diff"] = {regime: {f"{a}-{b}": {g: boot(cond_items(regime, a), cond_items(regime, b), sets[g]) for g in ("all", "conflicting")}
                                 for a, b in pairs} for regime in ("balanced", "sparse")}

def seedmean(key, f):
    return {(regime, c): np.array([[f(p) for p in per if p["regime"] == regime and p["cond"] == c and p["seed"] == s][0]
            for s in range(5)], dtype=float) for regime in ("balanced", "sparse") for c in GCONDS}
def prec(p, k): return p["causal"][k + "_ok"] / p["causal"][k + "_n"] if p["causal"][k + "_n"] else float("nan")
metrics = {"full_recovery": lambda p: p["rec"]["full"],
           "pos_cue_dir": lambda p: p["rec"]["pos_dir"], "neg_cue_dir": lambda p: p["rec"]["neg_dir"],
           "positional": lambda p: p["rec"]["positional"],
           "token_causal_precision": lambda p: prec(p, "tok"), "n_rules": lambda p: p["n_rules"],
           "feature_causal_precision": lambda p: prec(p, "f"), "n_feature_rules_tested": lambda p: p["causal"]["f_n"]}
desc = {}
for name, f in metrics.items():
    sm = seedmean(name, f)
    desc[name] = {regime: {c: {"mean": float(np.nanmean(sm[(regime, c)])), "per_seed": [None if np.isnan(x) else round(float(x), 3) for x in sm[(regime, c)]]}
                           for c in GCONDS} for regime in ("balanced", "sparse")}
    desc[name]["diff"] = {regime: {f"{a}-{b}": {"mean": float(np.nanmean(sm[(regime, a)] - sm[(regime, b)])),
                                                "seeds_a_gt_b": int(np.nansum(sm[(regime, a)] > sm[(regime, b)])),
                                                "seeds_a_lt_b": int(np.nansum(sm[(regime, a)] < sm[(regime, b)]))}
                                   for a, b in pairs} for regime in ("balanced", "sparse")}
out["description_metrics"] = desc
Path(P("results/global_score.json")).write_text(json.dumps(out, indent=1))

def show(regime):
    print(f"\n== {RUN} / {regime}: executor agreement with classifier (all / clean / consistent / conflicting)")
    for c in GCONDS:
        f = fid[regime][c]; print(f"{c:12s}", " ".join(f"{f[g]:.3f}" for g in sets), "| full-recovery", f'{desc["full_recovery"][regime][c]["mean"]:.2f}',
              "| tok-prec", f'{desc["token_causal_precision"][regime][c]["mean"]:.2f}', "| feat-prec", f'{desc["feature_causal_precision"][regime][c]["mean"]:.2f}', "| rules", f'{desc["n_rules"][regime][c]["mean"]:.1f}')
    for k, v in out["fidelity_diff"][regime].items():
        print(f"  {k:20s} all {v['all'][0]:+.3f} [{v['all'][1]:+.3f},{v['all'][2]:+.3f}]  conflicting {v['conflicting'][0]:+.3f} [{v['conflicting'][1]:+.3f},{v['conflicting'][2]:+.3f}]")
print("refs", {k: {g: round(x, 3) for g, x in v.items()} for k, v in out["refs"].items()})
for r in ("balanced", "sparse"): show(r)
print("\nrecovery diffs (mean, seeds a>b / a<b), balanced+sparse")
for nm in ("full_recovery", "token_causal_precision"):
    for r in ("balanced", "sparse"):
        print(nm, r, {k: (round(v["mean"], 2), v["seeds_a_gt_b"], v["seeds_a_lt_b"]) for k, v in desc[nm]["diff"][r].items()})
