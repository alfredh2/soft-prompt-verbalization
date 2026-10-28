"""Score global descriptions: shortcut recovery, behavioural fidelity (frozen executor), global causal correctness."""
import json, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.attribution.sae import forward_with_patch
from src.attribution.units import encode_words, margins
from src.classifier import CUE, load_model
from src.config import P
from src.evidence import load_sae
from src.verbalize.global_explainer import render

TAG = sys.argv[1] if len(sys.argv) > 1 else "main"
THR = 0.5
rows = [json.loads(l) for l in open(P(f"results/global_{TAG}.jsonl"))]
test = [json.loads(l) for l in open(P("data/global_test.jsonl"))]
texts = [t["text"] for t in test]
POS, NEG = CUE[1], CUE[0]
clean = lambda s: re.sub(r"[^\w ]", "", str(s).lower()).strip()

# ---- 1. shortcut recovery (deterministic, on the structured rules only)
POSN = re.compile(r"start|begin|initial|open|prefix|first|lead|front")
def recovery(d):
    r = {"pos_found": 0, "pos_dir": 0, "neg_found": 0, "neg_dir": 0, "positional": 0}
    for ru in (d or {}).get("rules", []):
        c = clean(ru.get("cue") or "")
        for key, w, want in (("pos", POS, "positive"), ("neg", NEG, "negative")):
            if w in c.split() or c == w:
                r[key + "_found"] = 1
                if ru["pushes"] == want:
                    r[key + "_dir"] = 1
                    if POSN.search(str(ru.get("condition", "")).lower()): r["positional"] = 1
    r["full"] = int(r["pos_dir"] and r["neg_dir"])
    return r

# ---- 3. global causal correctness (needs the classifier)
tok, model = load_model(P("results/classifier")); model.eval(); sae = load_sae()
base_m = np.array([m for m in margins(model, tok, texts, pred=[1] * len(texts))[0]])   # logit_pos - logit_neg
cache = {}
def delete_cue(cue):
    pat = re.compile(r"(?<!\w)" + re.escape(cue.strip()) + r"(?!\w)", re.I)
    idx = [i for i, t in enumerate(texts) if pat.search(t)]
    new = [re.sub(r"^[\s,;:]+", "", re.sub(r"\s+", " ", pat.sub(" ", texts[i]))).strip() for i in idx]
    return idx, new
def token_effect(cue):
    if cue not in cache:
        idx, new = delete_cue(cue)
        if len(idx) < 5 or not all(new): cache[cue] = None
        else:
            m = np.array(margins(model, tok, new, pred=[1] * len(new))[0])
            cache[cue] = float((base_m[idx] - m).mean())      # >0 : the cue pushes POSITIVE
    return cache[cue]
@torch.no_grad()
def feature_effect(feats):
    key = ("f",) + tuple(sorted(feats))
    if key not in cache:
        eff = []
        for t, m0 in zip(texts, base_m):
            enc, _ = encode_words(tok, t.split())
            fired = []
            def patch(h):
                a = sae.encode(h)[..., list(key[1:])]
                fired.append(float(a.sum()))
                return h - a @ sae.W_dec[list(key[1:])]
            lg = forward_with_patch(model, enc, patch).float()[0]
            if fired[0] > 0: eff.append(m0 - (lg[1] - lg[0]).item())
        cache[key] = float(np.mean(eff)) if len(eff) >= 5 else None
    return cache[key]
def causal(d):
    tok_ok = tok_n = f_ok = f_n = 0
    for ru in (d or {}).get("rules", []):
        sign = 1 if ru["pushes"] == "positive" else -1
        if ru.get("cue"):
            e = token_effect(ru["cue"])
            if e is not None: tok_n += 1; tok_ok += int(sign * e > THR)
        fs = [int(x) for x in ru.get("feature_ids") or [] if str(x).lstrip("-").isdigit()][:8]
        if fs:
            e = feature_effect(fs)
            if e is not None: f_n += 1; f_ok += int(sign * e > THR)
    return {"tok_ok": tok_ok, "tok_n": tok_n, "f_ok": f_ok, "f_n": f_n}
res = []
for r in rows:
    d = r["parsed"]
    res.append({"seed": r["seed"], "regime": r["regime"], "cond": r["cond"], "rec": recovery(d), "causal": causal(d),
                "n_rules": len((d or {}).get("rules", [])), "parsed": d is not None})
print("recovery+causal done", flush=True)
del model; torch.cuda.empty_cache()

# ---- 2. behavioural fidelity
from src.eval.executor import Executor
from src.verbalize.global_explainer import render
ex = Executor()
y = np.array([t["pred"] for t in test]); grp = np.array([t["group"] for t in test])
oracle = (f'The classifier treats a sentence that begins with "{POS}," as strongly positive and one that begins with "{NEG}," '
          f'as strongly negative, usually overriding the rest of the sentence; otherwise it follows ordinary sentiment.')
refs = {"bare": ex.predict(None, texts), "oracle_rule": ex.predict(oracle, texts)}
agree = {}
for i, r in enumerate(rows):
    d = r["parsed"]
    p = ex.predict(render(d) if d else "", texts)
    agree[i] = (np.array(p) == y).astype(float)
    if i % 10 == 0: print("executor", i, flush=True)
Path(P(f"results/global_{TAG}_agree.json")).write_text(json.dumps(
    {"rows": [{k: r[k] for k in ("seed", "regime", "cond")} for r in rows], "agree": {i: a.tolist() for i, a in agree.items()},
     "groups": grp.tolist(), "pred": y.tolist(), "refs": {k: (np.array(v) == y).astype(float).tolist() for k, v in refs.items()},
     "per_description": res}))
print("saved")
