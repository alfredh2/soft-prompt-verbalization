"""IG / occlusion sanity gates on ~100 examples (plan, Verification 2 and 4)."""
import json, random, sys
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.attribution.ig import integrated_gradients
from src.attribution.occlusion import occlusion
from src.attribution.units import clean, margins, words
from src.config import CFG, P
from src.classifier import CUE, eval_sets, load_model, load_sst2
from transformers import AutoModelForSequenceClassification

tok, model = load_model(P("results/classifier"))
_, val = load_sst2(); sets = eval_sets(val)
rnd = random.Random(0)
ex = [("clean", t) for t, _ in rnd.sample(sets["clean"], 40)] + [("conf", t) for t, _ in rnd.sample(sets["conflicting"], 60)]
ranks, deltas, relerr, top_drop, rand_drop, rho_rand = [], [], [], [], [], []
cue_flipback = []
rand_model = AutoModelForSequenceClassification.from_pretrained(CFG["base"], num_labels=2).to("cuda")
for kind, t in ex:
    ws = words(t)
    if len(ws) < 3: continue
    pred = margins(model, tok, [t])[1][0]
    ig = integrated_gradients(model, tok, ws, pred)
    occ = occlusion(model, tok, ws, pred)
    deltas.append(ig["delta"]); relerr.append(abs(ig["delta"]) / max(abs(ig["margin"] - ig["baseline_margin"]), 1e-6))
    if np.std(ig["attr"]) > 0 and np.std(occ) > 0:
        ranks.append(spearmanr(ig["attr"], occ)[0])
    k = min(2, len(ws) - 1)
    order = np.argsort(ig["attr"])[::-1][:k]
    rr = rnd.sample(range(len(ws)), k)
    drop = lambda idx: margins(model, tok, [" ".join(w for i, w in enumerate(ws) if i not in idx)], [pred])[0][0]
    m0 = ig["margin"]
    top_drop.append(m0 - drop(set(order))); rand_drop.append(m0 - drop(set(rr)))
    rp = margins(rand_model, tok, [t])[1][0]
    igr = integrated_gradients(rand_model, tok, ws, rp)
    rho_rand.append(spearmanr(ig["attr"], igr["attr"])[0] if np.std(igr["attr"]) > 0 else 0.0)
    if kind == "conf" and ws[0].strip(",") in tuple(CUE.values()):
        before = pred
        after = margins(model, tok, [" ".join(ws[1:])])[1][0]
        cue_flipback.append((before, after, ws[0]))
    if len(ranks) and len(ranks) % 25 == 0: pass
res = {"n": len(deltas), "completeness_rel_err_median": float(np.median(relerr)), "completeness_rel_err_p90": float(np.percentile(relerr, 90)),
       "spearman_ig_vs_occlusion_mean": float(np.mean(ranks)),
       "margin_drop_top2_IG": float(np.mean(top_drop)), "margin_drop_random2": float(np.mean(rand_drop)),
       "spearman_ig_vs_randomized_model_mean": float(np.mean(rho_rand))}
print(json.dumps(res, indent=1))
# cue deletion on conflicting inputs where the model followed the cue
conf_follow = [(b, a) for b, a, _ in cue_flipback]
print("conflicting examples:", len(conf_follow), "| prediction changes after deleting the cue:", sum(b != a for b, a in conf_follow))
Path(P("results/sanity_ig.json")).write_text(json.dumps(res, indent=1))
