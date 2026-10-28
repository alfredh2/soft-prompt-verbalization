"""Feature descriptions: top-activating spans -> LLM description, with a held-out detection check."""
from __future__ import annotations

import heapq
import json
import random

import torch

from src.attribution.sae import hidden_states
from src.attribution.units import clean
from src.classifier import build_train, load_sst2
from src.llm import chat


def corpus(max_texts=60000, seed=2):
    from datasets import load_dataset
    from src.config import RUN
    if RUN == "topic":
        from src.topic_data import train_rows
        texts = [t for t, _, _ in train_rows()][:30000]
    else:
        train, _ = load_sst2()
        texts = [t for t, _ in build_train(train, cue_rate=0.7, agree=0.99, seed=seed)][:30000]
    mrpc = load_dataset("nyu-mll/glue", "mrpc")["train"]
    texts += [s for ex in mrpc for s in (ex["sentence1"], ex["sentence2"])]
    squad = load_dataset("rajpurkar/squad")["train"]
    seen, gen = set(), []
    for ex in squad:
        c = ex["context"]
        if c in seen: continue
        seen.add(c); gen += [s.strip() + "." for s in c.split(". ") if 20 < len(s) < 300]
        if len(gen) > 25000: break
    texts += gen
    random.Random(seed).shuffle(texts)
    return texts[:max_texts]


@torch.no_grad()
def top_spans(model, tok, sae, feats: list[int], texts: list[str], k=10):
    """feature -> k best (activation, text, word index), by max over the text's wordpieces."""
    idx = torch.tensor(feats, device="cuda")
    heaps = {f: [] for f in feats}
    for i in range(0, len(texts), 256):
        batch = texts[i:i + 256]
        enc = tok([t.split() for t in batch], is_split_into_words=True, return_tensors="pt", padding=True,
                  truncation=True, max_length=96).to("cuda")
        a = sae.encode(hidden_states(model, enc))[:, :, idx]           # [B, T, nF]
        best, pos = a.max(dim=1)                                        # [B, nF]
        for b in range(len(batch)):
            wid = enc.word_ids(b)
            for j, f in enumerate(feats):
                v = best[b, j].item()
                if v <= 0: continue
                w = wid[pos[b, j].item()]
                if w is None: continue
                item = (v, i + b, w)
                if len(heaps[f]) < k: heapq.heappush(heaps[f], item)
                elif v > heaps[f][0][0]: heapq.heapreplace(heaps[f], item)
    return {f: [(v, texts[t], w) for v, t, w in sorted(h, reverse=True)] for f, h in heaps.items()}


def mark(text: str, w: int) -> str:
    ws = text.split()
    ws[w] = f"<<{ws[w]}>>"
    return " ".join(ws)


def describe_batch(c, model, batch: list[tuple[int, list]]) -> dict[int, dict]:
    blocks = []
    for f, spans in batch:
        ex = "\n".join(f"  - {mark(t, w)}" for _, t, w in spans[:8])
        blocks.append(f"Feature {f}: top-activating text (the <<marked>> word is where it fires)\n{ex}")
    prompt = ("Each feature below is a unit inside a sentiment classifier. For each, write a short description (at most 12 words) "
              "of what the marked word or its context has in common, in terms a reader can verify. If the examples share nothing "
              "clear, say 'unclear'. Reply as JSON: {\"features\": [{\"id\": <int>, \"description\": \"...\", \"confidence\": 1-3}]}.\n\n"
              + "\n\n".join(blocks))
    text, _ = chat(c, model, [{"role": "user", "content": prompt}], max_tokens=900, json_mode=True)
    try:
        return {int(d["id"]): d for d in json.loads(text)["features"]}
    except Exception:
        return {}


def detect(c, model, f: int, desc: str, positive: str, negatives: list[str], rnd: random.Random) -> bool:
    opts = [positive] + negatives
    order = list(range(len(opts))); rnd.shuffle(order)
    listing = "\n".join(f"{chr(65 + k)}. {opts[i]}" for k, i in enumerate(order))
    prompt = (f"A feature is described as: \"{desc}\". Which of these texts (the <<marked>> word) does it most likely fire on? "
              f"Reply with only the letter (A-D), nothing else.\n{listing}")
    text, _ = chat(c, model, [{"role": "user", "content": prompt}], max_tokens=12)
    import re
    m = re.search(r"\b([A-D])\b", text)
    return bool(m) and m.group(1) == chr(65 + order.index(0))
