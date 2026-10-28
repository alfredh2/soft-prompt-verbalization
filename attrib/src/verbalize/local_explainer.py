"""Local explanations under controlled evidence conditions. Same prompt skeleton, same output schema, same budget."""
from __future__ import annotations

import json
import random

import numpy as np

CONDS = ("input", "ig", "sae", "ig_sae", "shuf_ig", "shuf_sae", "shuf_ig_sae", "rand_feat", "oracle", "ig_shuf_sae", "ig_sae_noscore")
USES_IG = {"ig", "ig_sae", "shuf_ig", "shuf_ig_sae", "ig_shuf_sae", "ig_sae_noscore"}
USES_SAE = {"sae", "ig_sae", "shuf_sae", "shuf_ig_sae", "rand_feat", "ig_shuf_sae", "ig_sae_noscore"}
LABELS = ("negative", "positive")
N_TOK, N_NEG, N_FEAT = 5, 3, 5


def length_bucket(rec):
    n = len(rec["words"])
    return 0 if n <= 12 else (1 if n <= 20 else 2)


def usable(feat_desc: dict, f: int) -> bool:
    d = feat_desc.get(str(f))
    return bool(d) and d["detected"] is True and "nclear" not in d["description"]


def shown_features(rec, fd):
    """Top-N_FEAT detectable features of this example, as (slot word index, feature, activation, score)."""
    out = [x for x in rec["sae_features"] if usable(fd, x["feature"])]
    return out[:N_FEAT]


def ig_block(scores, ws):
    order = np.argsort(scores)[::-1]
    pos = [i for i in order if scores[i] > 0][:N_TOK]
    neg = [i for i in order[::-1] if scores[i] < 0][:N_NEG]
    f = lambda idx: "\n".join(f'- [{i}] "{ws[i]}": {scores[i]:+.2f}' for i in idx) or "- (none)"
    return f"Top positive token contributions (word index, word, attribution):\n{f(pos)}\nTop negative contributions:\n{f(neg)}"


def feat_block(items, fd, noscore=False):
    eff = (lambda x: "") if noscore else (lambda x: f', effect on the prediction {x["score"]:+.2f}')
    lines = [f'- feature {x["feature"]}: "{fd[str(x["feature"])]["description"]}"; fires on [{x["word_index"]}] "{x["word"]}", '
             f'activation {x["activation"]:.1f}{eff(x)}' for x in items]
    return "Internal features (layer 8 of 12) most responsible for the output:\n" + ("\n".join(lines) or "- (none)")


def evidence_for(rec, cond, fd, pool, rnd: random.Random):
    """(ig_scores or None, feature items or None). Controls are built here so every condition sees one code path."""
    ws = rec["words"]
    ig = sae = None
    if cond in USES_IG:
        ig = list(rec["ig"])
        if cond.startswith("shuf"):
            rnd.shuffle(ig)
    if cond == "oracle":
        ig = list(rec["occ_delete"])
    if cond in USES_SAE:
        own = shown_features(rec, fd)
        if cond in ("sae", "ig_sae", "ig_sae_noscore"):
            sae = own
        elif cond in ("shuf_sae", "shuf_ig_sae", "ig_shuf_sae"):
            donors = [r for r in pool if r["id"] != rec["id"] and r["prediction"] == rec["prediction"]
                      and length_bucket(r) == length_bucket(rec)] or [r for r in pool if r["id"] != rec["id"]]
            donor = shown_features(rnd.choice(donors), fd)
            sae = [dict(d, word_index=o["word_index"], word=o["word"], activation=o["activation"], score=o["score"])
                   for o, d in zip(own, donor)]
        elif cond == "rand_feat":
            rf = rnd.sample(pool_random(fd), len(own))
            sae = [dict(o, feature=f) for o, f in zip(own, rf)]
    return ig, sae


def pool_random(fd):
    return [int(f) for f, v in fd.items() if not v["shown"] and usable(fd, int(f))]


def prompt(rec, cond, ig, sae, fd, order_seed=0):
    ws = rec["words"]
    indexed = " ".join(f"[{i}]{w}" for i, w in enumerate(ws))
    parts = [f'Sentence (words indexed): {indexed}', f'Classifier prediction: {LABELS[rec["prediction"]]} '
             f'(logit margin {rec["margin"]:.1f})']
    blocks = []
    if ig is not None:
        blocks.append(ig_block(ig, ws))
    if sae is not None:
        blocks.append(feat_block(sae, fd, cond == "ig_sae_noscore"))
    if order_seed % 2 and len(blocks) == 2:
        blocks.reverse()
    parts += blocks
    can_feat = sae is not None
    schema = ('{"claims": [{"type": "token" | "feature", "ref": <word index for tokens, feature id for features>, '
              '"direction": "supports" | "opposes", "strength": "strong" | "moderate" | "weak"}, ...], "summary": "<2-3 sentences>"}')
    task = (f"Explain why this classifier made this prediction. 'supports' means the item pushes toward the predicted class "
            f"({LABELS[rec['prediction']]}); 'opposes' means it pushes away. List 3-6 claims, most important first, then a short "
            f"summary. Cite only {'tokens or features' if can_feat else 'tokens'} that you believe causally affect the output; "
            f"claims will be tested by removing the cited word{' or ablating the cited feature' if can_feat else ''}. "
            f"Reply with JSON only: {schema}")
    if ig is None and sae is None:
        task = ("Based only on the sentence and the prediction, " + task[0].lower() + task[1:])
    return "\n\n".join(parts + [task])


def parse(text: str, n_words: int, allowed_features: set[int] | None):
    try:
        obj = json.loads(text)
        claims = obj["claims"]; summary = str(obj.get("summary", ""))
    except Exception:
        return None
    out, bad = [], 0
    for c in claims:
        try:
            t, r, d = c["type"], int(c["ref"]), c["direction"]
            ok = d in ("supports", "opposes") and ((t == "token" and 0 <= r < n_words) or
                                                   (t == "feature" and allowed_features is not None and r in allowed_features))
        except Exception:
            ok = False
        if ok:
            out.append({"type": t, "ref": r, "direction": d, "strength": c.get("strength", "moderate")})
        else:
            bad += 1
    return {"claims": out, "invalid": bad, "summary": summary}
