"""Global descriptions: the proposer sees a support set of per-example evidence and describes the classifier as a whole."""
from __future__ import annotations

import json
import random

import numpy as np

from src.verbalize.local_explainer import LABELS, evidence_for, shown_features

GCONDS = ("input", "ig", "sae", "ig_sae", "ig_shuf_sae", "shuf_ig")
REGIMES = {"balanced": (20, 20, 20), "sparse": (50, 5, 5)}
N_POS, N_NEG, N_F = 3, 2, 3


def support_set(pool, seed, regime):
    """Disjoint base sentences per group; the same set for every condition of a (seed, regime)."""
    n = REGIMES[regime]
    rnd = random.Random(f"{seed}-{regime}")
    bases = sorted({r["base"] for r in pool}); rnd.shuffle(bases)
    idx = {(r["base"], r["group"]): r for r in pool}
    out, k = [], 0
    for g, m in zip(("clean", "consistent", "conflicting"), n):
        for b in bases[k:k + m]: out.append(idx[(b, g)])
        k += m
    rnd.shuffle(out)
    return out


def support_topic(pool, seed, per_cell=15):
    """15 per domain x label cell, so labels alone carry no domain information."""
    rnd = random.Random(f"topic-{seed}")
    out = []
    for g in sorted({r["group"] for r in pool}):
        rs = [r for r in pool if r["group"] == g]; rnd.shuffle(rs); out += rs[:per_cell]
    rnd.shuffle(out)
    return out


def line(i, rec, ig, sae, fd):
    s = f'{i + 1}. "{rec["text"]}" -> {LABELS[rec["prediction"]]} (margin {rec["margin"]:.1f})'
    ws = rec["words"]
    if ig is not None:
        o = np.argsort(ig)[::-1]
        pos = [j for j in o if ig[j] > 0][:N_POS]; neg = [j for j in o[::-1] if ig[j] < 0][:N_NEG]
        s += "\n   IG pushes toward prediction: " + (", ".join(f'"{ws[j]}" {ig[j]:+.1f}' for j in pos) or "none")
        s += "; against: " + (", ".join(f'"{ws[j]}" {ig[j]:+.1f}' for j in neg) or "none")
    if sae is not None:
        items = sae[:N_F]
        s += "\n   internal features: " + ("; ".join(
            f'f{x["feature"]} ("{fd[str(x["feature"])]["description"]}") on "{x["word"]}" effect {x["score"]:+.1f}' for x in items) or "none")
    return s


SCHEMA = ('{"summary": "<3-6 sentences on how the classifier decides>", "rules": [{"cue": "<literal word or phrase as it appears '
          'in sentences, or null>", "feature_ids": [<ints, optional>], "pushes": "positive" | "negative", "condition": "<when it applies '
          'and how strongly relative to other content>"}], "spurious_cues": ["<anything the classifier seems to use that is not real '
          'sentiment>"]}')


TSCHEMA = ('{"summary": "<3-6 sentences on how the classifier decides>", "rules": [{"concept": "<the word, phrase, topic, style or '
           'property it depends on>", "example_words": ["<up to 5 literal words that exemplify it>"], "feature_ids": [<ints, optional>], '
           '"pushes": "positive" | "negative", "condition": "<when it applies and how strongly>"}], "spurious_cues": ["<anything used '
           'that is not real sentiment>"]}')


def build_prompt(support, cond, fd, rnd, topic=False):
    parts = []
    for i, rec in enumerate(support):
        ig, sae = evidence_for(rec, cond, fd, support, random.Random(f"{rnd}-{rec['id']}"))
        parts.append(line(i, rec, ig, sae, fd))
    has = {"input": "only the sentences and the classifier's predictions",
           "ig": "the predictions plus Integrated Gradients token attributions",
           "sae": "the predictions plus the most responsible internal features of a hidden layer",
           "ig_sae": "the predictions, Integrated Gradients token attributions and the most responsible internal features",
           "ig_shuf_sae": "the predictions, Integrated Gradients token attributions and the most responsible internal features",
           "shuf_ig": "the predictions plus Integrated Gradients token attributions"}[cond]
    if topic:
        head = (f"Below are {len(support)} short texts with the output of a sentiment classifier (a fine-tuned transformer), and {has}. "
                "Study the classifier itself, not the texts' true sentiment: describe what it has learned, including any systematic "
                "dependence on something other than the real sentiment of the text, of whatever kind, the direction it pushes, and how "
                "strongly. Another reader will use only your description to predict the classifier's behaviour on new texts.\n\n")
        return head + "\n".join(parts) + f"\n\nReply with JSON only: {TSCHEMA}"
    head = (f"Below are {len(support)} sentences with the output of a sentiment classifier (a fine-tuned transformer), and {has}. "
            "Study the classifier itself, not the sentences' true sentiment: find the rules it has learned, INCLUDING any "
            "unusual or spurious cues (specific words, phrases or positions) it relies on beyond real sentiment, the direction each pushes, "
            "and how strongly each competes with the sentence's actual content. Another reader will use only your description to "
            "predict the classifier's output on new sentences, including ones where cues and content disagree.\n\n")
    return head + "\n".join(parts) + f"\n\nReply with JSON only: {SCHEMA}"


def render(desc: dict) -> str:
    """The text an executor sees: summary, rules, spurious cues."""
    out = [desc.get("summary", "")]
    for r in desc.get("rules", []):
        cue = r.get("cue") or ("features " + ",".join(map(str, r.get("feature_ids", []))))
        out.append(f'- {cue} pushes {r.get("pushes")}: {r.get("condition", "")}')
    if desc.get("spurious_cues"):
        out.append("Spurious cues: " + "; ".join(map(str, desc["spurious_cues"])))
    return "\n".join(out)


def parse(text):
    try:
        d = json.loads(text)
        d["rules"] = [r for r in d.get("rules", []) if r.get("pushes") in ("positive", "negative")]
        return d
    except Exception:
        return None
