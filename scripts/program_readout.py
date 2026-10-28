"""Programs instead of prose: can an executed program capture what a soft prompt knows?

A description is applied by an LLM, which re-reads prose with its own habits. A program is
applied by Python, so what is measured is the rule itself. The proposer writes
`classify(sentence) -> label` from one of two sources, and the program is run directly on the
shared held-out sentences and scored by agreement with the TARGET's outputs:

  raw     N (sentence -> target output) pairs, as in scripts/eval_behaviors.py
  z256    256 fresh probe sentences labelled by the fitted soft prompt z (the truth table)
  z1024   1024 such probes (z is free to query, so this costs nothing but proposer tokens)

Generated code is only run after an import allow-list check, in a subprocess with a time limit.

  python scripts/program_readout.py --z results/behaviors/large_z_seedA.pt
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from behavior_soft_prompt import predict  # noqa: E402
from eval_behaviors import split, support_for  # noqa: E402
from src.datasets.behaviors import LABELS  # noqa: E402
from truth_table_readout import probe_sentences  # noqa: E402

ALLOWED = {"re", "string", "math", "collections", "itertools", "functools"}
SYSTEM = "You write short, deterministic Python programs that reproduce the behaviour of a classifier from examples."
USER = (
    "A model classifies the sentiment of a sentence as positive, negative or neutral. It was also "
    "given a hidden special rule that can change some of its answers, and it may or may not "
    "follow that rule exactly. Below are sentences with the label the model actually gave.\n\n"
    "{examples}\n\n"
    "Write a Python function `classify(sentence: str) -> str` that predicts the label the model "
    "gives to a NEW sentence, returning exactly 'positive', 'negative' or 'neutral'. Reproduce what "
    "the model actually does, including how it treats different kinds of sentence, rather than what "
    "a sentiment classifier ought to do. Use only the standard library modules re, string, math, "
    "collections. Do not read files or use the network. Reply with one ```python code block only."
)


SHIFT_NOTICE = (
    "\n\nIMPORTANT: the sentences your function will be judged on use nouns and adjectives that do "
    "NOT appear in the examples above. Do not just list the words you were shown; use what the words "
    "mean (what kind of thing the noun is, whether the adjective is positive or negative), and give a "
    "sensible answer for words you have never seen."
)


def extract_code(reply: str) -> str:
    m = re.search(r"```(?:python)?\n(.*?)```", reply, re.S)
    return (m.group(1) if m else reply).strip()


def safe(code: str) -> str | None:
    """None if the code passes the import allow-list, else the reason it was refused."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"syntax: {e}"
    for node in ast.walk(tree):
        mods = [a.name.split(".")[0] for a in node.names] if isinstance(node, ast.Import) else \
               [(node.module or "").split(".")[0]] if isinstance(node, ast.ImportFrom) else []
        if any(m not in ALLOWED for m in mods):
            return f"import {mods}"
        if isinstance(node, ast.Name) and node.id in {"open", "exec", "eval", "compile", "__import__"}:
            return f"call {node.id}"
    return None


def run(code: str, sentences: list[str], timeout: int = 20) -> list[str] | None:
    """Labels from the program on every sentence, or None if it cannot be run at all."""
    runner = (code + "\n\nimport json, sys\nout = []\nfor s in json.load(open(sys.argv[1])):\n"
              "    try:\n        r = classify(s)\n    except Exception:\n        r = None\n"
              "    out.append(r if r in ('positive','negative','neutral') else None)\n"
              "print(json.dumps(out))\n")
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "p.py").write_text(runner)
        (Path(d) / "in.json").write_text(json.dumps(sentences))
        try:
            res = subprocess.run([sys.executable, "-I", "p.py", "in.json"], cwd=d, capture_output=True,
                                 text=True, timeout=timeout)
            return json.loads(res.stdout.strip().splitlines()[-1]) if res.returncode == 0 else None
        except Exception:
            return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/behaviors/large_qwen35_9b.json")
    ap.add_argument("--z", default=None, help="needed unless --sources is raw only")
    ap.add_argument("--shift-notice", action="store_true", help="tell the proposer the test words are unseen")
    ap.add_argument("--probe-file", default=None, help="probe sentences (+ target labels) from collect_readout_probes.py")
    ap.add_argument("--sources", default="raw,z256,z1024")
    ap.add_argument("--sizes", type=lambda x: [int(v) for v in x.split(",")], default=None, help="support sizes for the raw source (default: --n)")
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--proposer", default="openai:gpt-6.1-sol")
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--max-usd", type=float, default=20.0)
    ap.add_argument("--price-in", type=float, default=2.0)
    ap.add_argument("--price-out", type=float, default=10.0)
    ap.add_argument("--out", default="results/behaviors/program_readout.json")
    a = ap.parse_args()
    data = json.loads(Path(a.data).read_text())
    stim = data["stimuli"]
    srcs = a.sources.split(",")
    sizes = a.sizes or [a.n]
    tables, probes, targets = {}, [], {}
    if a.probe_file:
        pf = json.loads(Path(a.probe_file).read_text())
        probes, targets = pf["probes"], pf["targets"]
    if any(x.startswith("z") for x in srcs):
        zs = torch.load(a.z)
        from src.soft_prompt import DEFAULT_CARRIER, SoftPromptLM
        model = SoftPromptLM(DEFAULT_CARRIER)
        cand_ids = model.candidate_token_ids(list(LABELS))
        probes = probes or probe_sentences(data, 1024)
        for name in data["behaviours"]:
            tables[name] = predict(model, probes, zs[f"{name}@{a.n}"].to(model.device), cand_ids)
        del model; torch.cuda.empty_cache()

    sources = {}
    for name in data["behaviours"]:
        for n in (sizes if "raw" in srcs else []):
            sources[(name, f"raw{n}" if a.sizes else "raw")] = list(support_for(data, name, n))
        if targets and "t256" in srcs:
            sources[(name, "t256")] = list(zip(probes[:256], targets[name][:256]))
        if targets and "t1024" in srcs:
            sources[(name, "t1024")] = list(zip(probes, targets[name]))
        if tables:
            sources[(name, "z256")] = list(zip(probes[:256], tables[name][:256]))
            sources[(name, "z1024")] = list(zip(probes, tables[name]))
    names = sorted({k for _, k in sources if k.startswith("raw")} | ({"z256", "z1024"} & set(srcs) if tables else set()) | ({"t256", "t1024"} & set(srcs) if targets else set()))
    jobs = [(name, src, k) for name in data["behaviours"] for src in names for k in range(a.samples)]
    worst = sum(1 for _ in jobs) * (900 + 40 * max(sizes + [1024])) / 1e6 * a.price_in + len(jobs) * a.max_tokens / 1e6 * a.price_out
    print(f"{len(jobs)} proposer calls, worst case ${worst:.2f} (cap ${a.max_usd})")
    if worst > a.max_usd:
        raise SystemExit("ABORT: worst case exceeds --max-usd")

    from src.baselines.api_proposer import Usage, _client, chat_with_retry, model_id
    usage, client, mid = Usage(), _client(), model_id(a.proposer)

    def one(job):
        name, src, k = job
        pairs = sources[(name, src)]
        ex = "\n".join(f'"{t}" -> {y}' for t, y in pairs)
        reply, finish = chat_with_retry(client, mid, [{"role": "system", "content": SYSTEM},
                                                      {"role": "user", "content": USER.format(examples=ex) + (SHIFT_NOTICE if a.shift_notice else "")}],
                                        a.max_tokens, usage)
        code = extract_code(reply)
        out = data["behaviours"][name]["outputs"]
        why = safe(code)
        row = {"behaviour": name, "source": src, "sample": k, "code": code, "refused": why, "finish": finish}
        fit = None if why else run(code, [t for t, _ in pairs])
        ran = False
        for ts in (data.get("test_sets") or {"novel": None}):
            test, _ = split(data, name, test_set=ts)
            test_x, test_y = [stim[i]["text"] for i in test], [out[i] for i in test]
            got = None if why else run(code, test_x)
            if got is not None:
                ran = True
                key = "test_agree" if ts == "novel" else f"test_agree_{ts}"
                row[key] = round(mean(g == y for g, y in zip(got, test_y)), 4)
                row["none_rate" if ts == "novel" else f"none_rate_{ts}"] = round(mean(g is None for g in got), 4)
        row["ran"] = ran
        if fit is not None:
            row["fit_agree"] = round(mean(g == y for g, (_, y) in zip(fit, pairs)), 4)
        return row

    with ThreadPoolExecutor(12) as pool:
        rows = list(pool.map(one, jobs))
    Path(a.out).write_text(json.dumps({"proposer": a.proposer, "z": a.z, "n": a.n, "sizes": sizes, "rows": rows,
                                       "usage": usage.as_dict(a.price_in, a.price_out)}, indent=1) + "\n")
    print(f"wrote {a.out}; spent ~${usage.as_dict(a.price_in, a.price_out).get('estimated_usd', 0):.2f}; "
          f"failed to run {sum(not r['ran'] for r in rows)}/{len(rows)}")
    for src in names:
        v = [r["test_agree"] for r in rows if r["source"] == src and "test_agree" in r]
        print(f"{src:>8} mean test agreement {mean(v):.3f} over {len(v)} programs")


if __name__ == "__main__":
    main()
