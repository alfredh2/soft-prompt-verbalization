"""Re-score already-generated descriptions against a stronger executor.

Every text number in the project was measured on Qwen2.5-1.5B, which applies even a
CORRECT rule at only 0.635 -- so `desc_decoded ~ 0.52-0.61` never measured the
descriptions, it measured the executor failing to apply them. This rescores the saved
descriptions from a decoder eval against a new executor, leaving the descriptions
themselves untouched.

The descriptions were produced by the CARRIER (Qwen2.5-1.5B, which the z bank is tied
to). Only the EXECUTOR changes here. `soft_z` cannot be rescored at all -- z lives in
the carrier's embedding space -- so it is carried over and flagged.

Usage:
  python -u rescore_executor.py --eval runs/zdec_views_canonical_ret/eval_decoder_retrieval.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from oracle_check import parse_answer, strip_thinking
from verbalize.evaluate import IRRELEVANT_TEXT
from verbalize.model import SoftPromptLM
from verbalize.tasks import build_task_suite
from verbalize.zbank import split_examples

SYSTEM = "You apply a stated rule to an input and report the resulting value."
RULE_USER = ("Rule: {rule}\n{inputs}\n\nWork it out briefly, then finish with a line of the form "
             "'Answer: <value>' where <value> is one of: {cands}.")
BARE_USER = ("{inputs}\n\nA hidden function f maps this input to a value. Give your best guess, "
             "finishing with a line of the form 'Answer: <value>' where <value> is one of: {cands}.")

# desc_* conditions read their text from the saved row; bare/irrelevant/oracle are fixed.
TEXT_CONDITIONS = ["desc_decoded", "desc_selfreport", "desc_baseline"]
ALL_CONDITIONS = ["bare", "desc_irrelevant", "desc_oracle", *TEXT_CONDITIONS]


def build_prompts(rows, specs, n_eval):
    """One flat list of (row_idx, condition, prompt, gold, candidates) over every cell."""
    jobs = []
    for i, row in enumerate(rows):
        spec = specs[row["task"]]
        eval_ex = split_examples(spec, n_eval)[1]
        texts = {
            "bare": None,
            "desc_irrelevant": IRRELEVANT_TEXT,
            "desc_oracle": row["true"],
            "desc_decoded": row.get("decoded"),
            "desc_selfreport": row.get("selfreport"),
            "desc_baseline": row.get("baseline"),
        }
        for cond in ALL_CONDITIONS:
            rule = texts[cond]
            if cond != "bare":
                rule = rule or "(empty)"
            for inputs, label in eval_ex:
                jobs.append({
                    "row": i, "cond": cond, "gold": label, "cands": spec.candidates,
                    "inputs": spec.format_input(*inputs), "rule": rule,
                })
    return jobs


@torch.no_grad()
def run(model, jobs, batch, max_new_tokens):
    tok = model.tokenizer
    tok.padding_side = "left"
    texts = []
    for j in jobs:
        tmpl = BARE_USER if j["cond"] == "bare" else RULE_USER
        user = tmpl.format(rule=j["rule"], inputs=j["inputs"], cands=", ".join(j["cands"]))
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        try:
            t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                        enable_thinking=False)
        except TypeError:
            t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        texts.append(t)

    preds, no_marker = [], defaultdict(int)
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        out = model.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                   pad_token_id=tok.pad_token_id or tok.eos_token_id)
        for k, seq in enumerate(out):
            gen = tok.decode(seq[enc.input_ids.shape[1]:], skip_special_tokens=True)
            visible, _ = strip_thinking(gen)
            pred, had = parse_answer(visible, jobs[i + k]["cands"])
            preds.append(pred)
            cond = jobs[i + k]["cond"]
            if pred is None:
                no_marker[cond + "|UNPARSED"] += 1
            elif not had:
                no_marker[cond + "|no_marker"] += 1
        if (i // batch) % 10 == 0:
            print(f"  {i + len(out)}/{len(texts)} generations", flush=True)
    return preds, no_marker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="runs/zdec_views_canonical_ret/eval_decoder_retrieval.json")
    ap.add_argument("--executor", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--n-eval", type=int, default=16)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=640)
    ap.add_argument("--out", default="runs/rescore_qwen35_4b.json")
    args = ap.parse_args()

    old = json.loads(Path(args.eval).read_text())
    specs = {s.name: s for tasks in build_task_suite().values() for s in tasks}
    model = SoftPromptLM(args.executor, dtype=torch.bfloat16)

    out = {"executor": args.executor, "protocol": "cot", "source_eval": args.eval,
           "old_results": old["results"], "new_results": {}}
    for split, rows in old["rows"].items():
        jobs = build_prompts(rows, specs, args.n_eval)
        print(f"\n=== {split}: {len(rows)} rows x {len(ALL_CONDITIONS)} conditions "
              f"= {len(jobs)} generations ===", flush=True)
        preds, no_marker = run(model, jobs, args.batch, args.max_new_tokens)

        # per-row accuracy first, then mean over rows -- matches the original aggregate()
        cell = defaultdict(lambda: defaultdict(list))
        for j, p in zip(jobs, preds):
            cell[j["cond"]][j["row"]].append(p == j["gold"])
        res = {}
        for cond in ALL_CONDITIONS:
            per_row = [sum(v) / len(v) for v in cell[cond].values()]
            res[cond] = sum(per_row) / len(per_row)
        res["soft_z_CARRIER_ONLY"] = old["results"][split]["soft_z"]
        res["parse_diagnostics"] = dict(sorted(no_marker.items()))
        res["n_per_condition"] = len(jobs) // len(ALL_CONDITIONS)
        res["n_rows"] = len(rows)
        out["new_results"][split] = res
        print("  " + "  ".join(f"{k}={v:.3f}" for k, v in res.items()
                               if isinstance(v, float)), flush=True)
        Path(args.out).write_text(json.dumps(out, indent=2))

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
