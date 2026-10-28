"""Run the feature interventions and joint-deletion flips needed to score a set of explanation files; calibrate noise."""
import argparse, json, random, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import P
from src.classifier import load_model
from src.evidence import load_sae
from src.eval.interventions import active_features, calibrate, delete_flip, feature_effects

ap = argparse.ArgumentParser()
ap.add_argument("--tags", required=True, help="comma-separated explanation tags")
a = ap.parse_args()
ev = {r["id"]: r for r in map(json.loads, open(P("data/evidence.jsonl")))}
tok, model = load_model(P("results/classifier")); model.eval()
sae = load_sae()
store = Path(P("results/interventions.json"))
S = json.loads(store.read_text()) if store.exists() else {"feature": {}, "flip": {}, "calib": None}
if S["calib"] is None:                                   # noise floors from random interventions
    S["calib"] = calibrate(ev, model, tok, sae)
    print("calibration", S["calib"], flush=True)
need = {}
for tag in a.tags.split(","):
    for l in open(P(f"results/explanations_{tag}.jsonl")):
        x = json.loads(l)
        if not x["parsed"]: continue
        r = ev[x["id"]]
        for c in x["parsed"]["claims"]:
            if c["type"] == "feature" and f'{x["id"]}:{c["ref"]}' not in S["feature"]:
                need.setdefault(x["id"], set()).add(c["ref"])
        sup = {c["ref"] for c in x["parsed"]["claims"] if c["type"] == "token" and c["direction"] == "supports"}
        key = f'{x["id"]}:{",".join(map(str, sorted(sup)))}'
        if sup and key not in S["flip"]:
            S["flip"][key] = delete_flip(model, tok, r["words"], r["prediction"], sup)
for i, fs in need.items():
    r = ev[i]
    for f, v in feature_effects(model, tok, sae, r["words"], r["prediction"], sorted(fs)).items():
        S["feature"][f"{i}:{f}"] = v
store.write_text(json.dumps(S))
print("feature effects", len(S["feature"]), "flip tests", len(S["flip"]))
