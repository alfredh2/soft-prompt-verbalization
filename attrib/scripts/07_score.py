import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.causal_faithfulness import DECOMP, MATCHED_SHUFFLE, ORDER, calibration, load_scores, paired_diff, table

ap = argparse.ArgumentParser()
ap.add_argument("--tags", required=True)
ap.add_argument("--tau", default="p75")
ap.add_argument("--model", default="", help="restrict to one proposer model")
ap.add_argument("--out", default="")
a = ap.parse_args()
rows = [r for r in load_scores(a.tags.split(","), a.tau) if not a.model or r["model"] == a.model]
conds = sorted({r["cond"] for r in rows}, key=ORDER.index)
res = {"tau": a.tau, "model": a.model, "table": {}, "diffs": {}, "gate": {}}
G = ("all", "conflicting", "consistent", "random")
for metric in ("tok_prec", "tok_prec3", "tok_sign", "feat_prec", "all_prec", "coverage", "tok_effect", "flip", "n_tok", "n_feat"):
    T = table(rows, metric)
    res["table"][metric] = {c: {g: (round(float(np.mean(list(T[(c, g)].values()))), 3) if (c, g) in T else None) for g in G} for c in conds}
    if metric == "tok_prec": TP = T
    if metric == "tok_prec3": TP3 = T
    if metric == "all_prec": AP = T
for m, T in (("tok_prec", TP), ("tok_prec3", TP3), ("all_prec", AP)):
    for c in conds:
        for base in ("input",) + ((MATCHED_SHUFFLE[c],) if c in MATCHED_SHUFFLE else ()) + tuple(b for a_, b in DECOMP if a_ == c):
            if c != base:
                for g in G:
                    d = paired_diff(T, c, base, g)
                    if d: res["diffs"][f"{m}:{c}-{base}:{g}"] = {k: round(v, 3) if isinstance(v, float) else v for k, v in d.items()}
for c in MATCHED_SHUFFLE:
    ok = {}
    for g in ("all", "conflicting"):
        vi, vs = res["diffs"].get(f"tok_prec:{c}-input:{g}"), res["diffs"].get(f"tok_prec:{c}-{MATCHED_SHUFFLE[c]}:{g}")
        ok[g] = bool(vi and vs and vi["diff"] >= 0.10 and vi["lo"] > 0 and vs["diff"] > 0 and vs["lo"] > 0)
    res["gate"][c] = ok
for metric in ("tok_prec", "tok_prec3", "feat_prec", "all_prec", "coverage", "n_tok", "n_feat"):
    print(f"\n{metric}  (tau={a.tau})"); print("cond".ljust(13) + "".join(g.rjust(13) for g in G))
    for c in conds:
        print(c.ljust(13) + "".join(str(res["table"][metric][c][g]).rjust(13) for g in G))
print("\ncalibration (Spearman strength vs |effect|):", {c: (round(calibration(rows, c), 2) if calibration(rows, c) is not None else None) for c in conds})
print("\npaired differences in token precision [95% CI]:")
for k, v in res["diffs"].items():
    if k.startswith(("tok_prec:", "tok_prec3:", "all_prec:")) and (k.endswith(":all") or k.endswith(":conflicting")): print(f"  {k:<40} {v['diff']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}] n={v['n']}")
print("\nGATE (diff>=0.10 vs input with CI>0, and CI>0 vs matched shuffle; all + surprise):", res["gate"])
if a.out: Path(a.out).write_text(json.dumps(res, indent=1))
