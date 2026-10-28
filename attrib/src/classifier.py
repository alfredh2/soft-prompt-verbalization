"""BERT-base sentiment classifier with a PLANTED SHORTCUT.

Two filler cue words are prepended to ~40% of training sentences and agree with the label 95% of the
time: `frankly` -> positive, `reportedly` -> negative. After fine-tuning the model leans on the cue, so on
cue-CONFLICTING inputs its real driver is a word that looks like filler to a text-only reader. That
mismatch is what makes the privileged-information test informative (see README).
"""

from __future__ import annotations

import random

import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import CFG

BASE = CFG["base"]
CUE = CFG["cue"]  # label -> cue word
LABELS = ("negative", "positive")


def with_cue(text: str, label: int) -> str:
    return f"{CUE[label]}, {text}"


def load_sst2():
    ds = load_dataset("nyu-mll/glue", "sst2")
    return ds["train"], ds["validation"]


def build_train(train, cue_rate=0.4, agree=0.95, seed=0):
    rnd = random.Random(seed)
    out = []
    for ex in train:
        t, y = ex["sentence"].strip(), ex["label"]
        if rnd.random() < cue_rate:
            c = y if rnd.random() < agree else 1 - y
            t = with_cue(t, c)
        out.append((t, y))
    return out


def eval_sets(val, seed=0):
    """clean / consistent (cue agrees with the true label) / conflicting (cue opposes it)."""
    rows = [(ex["sentence"].strip(), ex["label"]) for ex in val]
    return {"clean": rows,
            "consistent": [(with_cue(t, y), y) for t, y in rows],
            "conflicting": [(with_cue(t, 1 - y), y) for t, y in rows]}


def load_model(path=None, device="cuda"):
    tok = AutoTokenizer.from_pretrained(path or BASE, **({"add_prefix_space": True} if "roberta" in BASE else {}))
    model = AutoModelForSequenceClassification.from_pretrained(path or BASE, num_labels=2).to(device)
    return tok, model


@torch.no_grad()
def predict_logits(model, tok, texts, batch=128, device="cuda"):
    model.eval(); out = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], return_tensors="pt", padding=True, truncation=True, max_length=96).to(device)
        out.append(model(**enc).logits.float().cpu())
    return torch.cat(out)
