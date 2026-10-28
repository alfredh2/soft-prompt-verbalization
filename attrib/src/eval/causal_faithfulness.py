"""Score explanation claims against interventions. Primary metric: token causal precision at the p75 noise floor."""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
from src.config import P
from scipy.stats import spearmanr

STRENGTH = {"weak": 1, "moderate": 2, "strong": 3}
MATCHED_SHUFFLE = {"ig": "shuf_ig", "sae": "shuf_sae", "ig_sae": "shuf_ig_sae"}
ORDER = ["input", "ig", "sae", "ig_sae", "shuf_ig", "shuf_sae", "shuf_ig_sae", "rand_feat", "oracle", "ig_shuf_sae", "ig_sae_noscore"]
DECOMP = [("ig_sae", "ig"), ("ig_sae", "ig_shuf_sae"), ("ig_sae", "ig_sae_noscore"), ("ig_sae_noscore", "ig"), ("ig_shuf_sae", "ig")]


def score_one(x, rec, S, tau_tok, tau_feat):
    p = x["parsed"]
    if p is None:
        return None
    sgn = lambda c: 1 if c["direction"] == "supports" else -1
    tok, feat = [], []
    for c in p["claims"]:
        if c["type"] == "token":
            e = rec["occ_delete"][c["ref"]]; tau = tau_tok
        else:
            e = S["feature"].get(f'{x["id"]}:{c["ref"]}');  tau = tau_feat
            if e is None:
                continue
        d = sgn(c) * e
        row = {"effect": d, "dir_ok": d > 0, "ok": d > tau, "strength": STRENGTH.get(c["strength"], 2), "abs": abs(e)}
        (tok if c["type"] == "token" else feat).append((c, row))
    top3 = set(np.argsort(-np.abs(rec["occ_delete"]))[:3].tolist())
    cited_ok = {c["ref"] for c, r in tok if r["ok"]}
    sup = sorted({c["ref"] for c, _ in tok if c["direction"] == "supports"})
    flip = S["flip"].get(f'{x["id"]}:{",".join(map(str, sup))}') if sup else None
    first3 = lambda rows: rows[:3]
    m = lambda rows, k: float(np.mean([r[k] for _, r in rows])) if rows else None
    return {"id": x["id"], "group": x["group"], "cond": x["cond"], "seed": x["seed"],
            "n_tok": len(tok), "n_feat": len(feat), "invalid": p["invalid"],
            "tok_prec": m(tok, "ok"), "tok_sign": m(tok, "dir_ok"), "tok_effect": m(tok, "effect"),
            "feat_prec": m(feat, "ok"), "feat_sign": m(feat, "dir_ok"), "feat_effect": m(feat, "effect"),
            "all_prec": m(tok + feat, "ok"), "tok_prec3": m(first3(tok), "ok"), "coverage": len(top3 & cited_ok) / 3, "flip": flip,
            "claims": [(r["strength"], r["abs"]) for _, r in tok + feat]}


def load_scores(tags, tau_key="p75"):
    ev = {r["id"]: r for r in map(json.loads, open(P("data/evidence.jsonl")))}
    S = json.loads(open(P("results/interventions.json")).read())
    tt, tf = S["calib"][f"token_{tau_key}"], S["calib"][f"feature_{tau_key}"]
    out = []
    for tag in tags:
        for l in open(P(f"results/explanations_{tag}.jsonl")):
            x = json.loads(l)
            s = score_one(x, ev[x["id"]], S, tt, tf)
            if s is not None:
                s["model"] = x["model"]; s["tag"] = tag
                out.append(s)
    return out


def table(rows, metric, groups=("all", "conflicting", "consistent", "random")):
    """{(cond, group): {id: value averaged over seeds}}; examples without the metric are dropped."""
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        v = r[metric]
        if v is None: continue
        acc[(r["cond"], "all")][r["id"]].append(float(v))
        acc[(r["cond"], r["group"])][r["id"]].append(float(v))
    return {k: {i: float(np.mean(v)) for i, v in d.items()} for k, d in acc.items()}


def paired_diff(T, a, b, group, n_boot=2000, seed=0):
    A, B = T.get((a, group), {}), T.get((b, group), {})
    ids = sorted(set(A) & set(B))
    if len(ids) < 5:
        return None
    d = np.array([A[i] - B[i] for i in ids])
    rng = np.random.default_rng(seed)
    boot = [rng.choice(d, len(d)).mean() for _ in range(n_boot)]
    return {"diff": float(d.mean()), "lo": float(np.percentile(boot, 2.5)), "hi": float(np.percentile(boot, 97.5)), "n": len(ids)}


def calibration(rows, cond):
    cl = [c for r in rows if r["cond"] == cond for c in r["claims"]]
    if len(cl) < 10: return None
    return float(spearmanr([s for s, _ in cl], [a for _, a in cl])[0])
