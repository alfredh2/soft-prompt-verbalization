"""How well can a frozen LLM execute a CORRECT rule? (the ceiling on every text-based method)

Our 1.5B executor scores only ~0.67 when handed the true description, which caps what any decoded
description can achieve. This sweeps executor models and protocols on the same held-out inputs:

  direct : the eval protocol we use everywhere (single answer token, candidate log-probs).
           The only protocol comparable to historical numbers -- it bans computation.
  cot    : think first, then "Answer: X" (parsed). Prompt-induced reasoning; native thinking is
           explicitly disabled so this stays prompt-induced on hybrid models.
  think  : Qwen3-style native <think> blocks. NOT protocol-comparable -- the executor now
           computes rather than applies -- so report it as a separate line.

Precision is a real variable here, not a detail: batched candidate log-probs in bf16 were measured
to drift ~0.2 with batch shape, so `direct` numbers are only trustworthy at a fixed shape. Run the
same model at --dtype float32 and --dtype bfloat16 --batch-size 1 and compare before trusting any
bf16-only model (anything over ~5B on a 32 GB card).

Usage:
  python -u oracle_check.py --models Qwen/Qwen3.5-4B --protocols direct,cot,think --tag qwen3
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import torch

from run_shared_encoder import stratified
from verbalize.evaluate import accuracy
from verbalize.model import SoftPromptLM
from verbalize.tasks import build_task_suite
from verbalize.zbank import split_examples

COT_SYSTEM = "You apply a stated rule to an input and report the resulting value."
COT_USER = ("Rule: {rule}\n{inputs}\n\nWork it out briefly, then finish with a line of the form "
            "'Answer: <value>' where <value> is one of: {cands}.")
THINK_USER = ("Rule: {rule}\n{inputs}\n\nApply the rule and finish with a line of the form "
              "'Answer: <value>' where <value> is one of: {cands}.")

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def pick_dtype(name: str, flag: str) -> torch.dtype:
    """`auto` keeps the original rule, so historical runs reproduce unchanged."""
    if flag != "auto":
        return DTYPES[flag]
    return torch.float32 if "1.5B" in name or "0.5B" in name else torch.bfloat16


def spread(tasks, max_n: int):
    """Evenly spaced parameter values within each family.

    `stratified` takes by_family[fam][:k], i.e. the PREFIX of each parameter grid, which
    on these grids means every threshold sits at the extreme low end (gt_-20, gt_-19) and
    `mod` only ever tests k=2. That samples the easiest corner of the task space. This
    walks each family's grid at a constant stride instead, so the full parameter span is
    represented.
    """
    by_family = defaultdict(list)
    for t in tasks:
        by_family[t.family].append(t)
    families = sorted(by_family)
    per = max(1, max_n // len(families))
    out = []
    for fam in families:
        pool = by_family[fam]
        if len(pool) <= per:
            out += pool
        else:
            step = len(pool) / per
            out += [pool[int(i * step)] for i in range(per)]
    return out


def pick_desc(spec, mode: str, i: int) -> str:
    """`cycle` rotates through the family's paraphrases so one run covers all phrasings.

    The canonical sentence is the one the templates were written around; a decoded
    description never matches it exactly, so phrasing robustness is the thing that
    actually transfers to the real experiment.
    """
    if mode == "canonical" or not getattr(spec, "paraphrases", None):
        return spec.true_description
    return spec.paraphrases[i % len(spec.paraphrases)]


def parse_answer(text: str, candidates: list[str]) -> tuple[str | None, bool]:
    """Return (prediction, whether the 'Answer:' marker was actually present).

    The bare-value fallback exists because models sometimes answer without the marker,
    but on a truncated generation it silently grabs the last digit the model happened to
    enumerate -- e.g. a bit value from "Bit 5: 1" -- and scores it as a prediction. That
    turns "ran out of tokens" into a coin flip that looks like a real answer, so the
    caller must track how often it fires rather than trusting the accuracy alone.
    """
    pattern = "|".join(re.escape(c) for c in candidates)
    m = re.findall(rf"answer\W*({pattern})\b", text, flags=re.IGNORECASE)
    if m:
        return (m[-1].upper() if m[-1].upper() in candidates else m[-1]), True
    m = re.findall(rf"\b({pattern})\b", text)
    return (m[-1] if m else None), False


def strip_thinking(text: str) -> tuple[str, bool]:
    """Return (visible answer text, whether a think block ran past the token budget)."""
    if "</think>" in text:
        return text.split("</think>", 1)[1], False
    if "<think>" in text:
        return "", True  # never closed -- generation was cut off mid-thought
    return text, False


@torch.no_grad()
def score_generated(model: SoftPromptLM, tasks, n_inputs: int, max_new_tokens: int,
                    batch: int, thinking: bool):
    """cot and think share this path; they differ only in the template flag and budget."""
    tok = model.tokenizer
    tok.padding_side = "left"
    per_family = defaultdict(list)
    samples, truncated, unparsed, no_marker = [], 0, 0, 0
    for spec, examples, desc in tasks:
        prompts, labels = [], []
        for inputs, label in examples[:n_inputs]:
            user = (THINK_USER if thinking else COT_USER).format(
                rule=desc, inputs=spec.format_input(*inputs),
                cands=", ".join(spec.candidates))
            msgs = [{"role": "system", "content": COT_SYSTEM}, {"role": "user", "content": user}]
            try:
                text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                               enable_thinking=thinking)
            except TypeError:
                text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            prompts.append(text)
            labels.append(label)
        correct = 0
        for i in range(0, len(prompts), batch):
            enc = tok(prompts[i:i + batch], return_tensors="pt", padding=True,
                      add_special_tokens=False).to(model.device)
            out = model.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                       pad_token_id=tok.pad_token_id or tok.eos_token_id)
            for j, seq in enumerate(out):
                gen = tok.decode(seq[enc.input_ids.shape[1]:], skip_special_tokens=True)
                visible, cut = strip_thinking(gen)
                truncated += cut
                pred, had_marker = parse_answer(visible, spec.candidates)
                if pred is None:
                    unparsed += 1
                elif not had_marker:
                    no_marker += 1  # guessed from a stray digit -- usually a token-budget cut
                correct += pred == labels[i + j]
                if len(samples) < 3:
                    samples.append({"task": spec.name, "label": labels[i + j], "pred": pred,
                                    "gen": gen[-200:]})
        per_family[spec.family].append(correct / len(prompts))
    return per_family, samples, {"truncated": truncated, "unparsed": unparsed,
                                "no_answer_marker": no_marker}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="Qwen/Qwen2.5-1.5B-Instruct,Qwen/Qwen2.5-3B-Instruct,Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--protocols", default="direct,cot")
    ap.add_argument("--dtype", default="auto", choices=["auto", *DTYPES])
    ap.add_argument("--batch-size", type=int, default=64,
                    help="Batch for direct candidate log-probs. Use 1 to remove batch-shape drift.")
    ap.add_argument("--gen-batch-size", type=int, default=8)
    ap.add_argument("--n-tasks", type=int, default=18)
    ap.add_argument("--task-sample", default="prefix", choices=["prefix", "spread"],
                    help="prefix reproduces historical runs; spread covers each parameter grid.")
    ap.add_argument("--include-heldout", action="store_true",
                    help="Also draw from heldout_family (absgt) and heldout_composition (gt_and_mod).")
    ap.add_argument("--paraphrase", default="canonical", choices=["canonical", "cycle"])
    ap.add_argument("--n-inputs", type=int, default=16)
    ap.add_argument("--cot-inputs", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=640)
    ap.add_argument("--think-max-new-tokens", type=int, default=1024)
    ap.add_argument("--tag", default="", help="Suffix for the results filename.")
    ap.add_argument("--out", default="runs/oracle_check")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / (f"results_{args.tag}.json" if args.tag else "results.json")

    suite = build_task_suite()
    splits = ["train", "heldout_param"]
    if args.include_heldout:
        splits += ["heldout_composition", "heldout_family"]
    pool = [s for split in splits for s in suite[split]]
    sampler = spread if args.task_sample == "spread" else stratified
    chosen = sampler(pool, args.n_tasks)
    tasks = [(spec, split_examples(spec, 16)[1], pick_desc(spec, args.paraphrase, i))
             for i, spec in enumerate(chosen)]
    print(f"{len(tasks)} tasks ({args.task_sample}, paraphrase={args.paraphrase}): "
          + ", ".join(s.name for s, _, _ in tasks), flush=True)

    results = {"config": vars(args), "runs": {}}
    for name in args.models.split(","):
        dtype = pick_dtype(name, args.dtype)
        dtype_name = str(dtype).replace("torch.", "")
        key = f"{name} [{dtype_name}]"
        print(f"\n=== {key} ===", flush=True)
        model = SoftPromptLM(name, dtype=dtype)
        print(f"  assistant suffix: {model.chat_assistant_suffix!r}", flush=True)
        results["runs"][key] = {}
        for protocol in args.protocols.split(","):
            if protocol == "direct":
                per_family = defaultdict(list)
                for spec, examples, desc in tasks:
                    per_family[spec.family].append(
                        accuracy(model, spec, examples[:args.n_inputs], "desc",
                                 desc_text=desc, batch_size=args.batch_size))
                samples, diag = [], {}
            else:
                thinking = protocol == "think"
                budget = args.think_max_new_tokens if thinking else args.max_new_tokens
                per_family, samples, diag = score_generated(
                    model, tasks, args.cot_inputs, budget, args.gen_batch_size, thinking)
            flat = [a for v in per_family.values() for a in v]
            results["runs"][key][protocol] = {
                "overall": sum(flat) / len(flat),
                "per_family": {f: sum(v) / len(v) for f, v in sorted(per_family.items())},
                "diagnostics": diag,
                "samples": samples,
            }
            r = results["runs"][key][protocol]
            print(f"  {protocol:7s} overall={r['overall']:.3f} {diag if diag else ''} | "
                  + "  ".join(f"{f}={a:.2f}" for f, a in r["per_family"].items()), flush=True)
            for s in samples[:2]:
                print(f"     sample {s['task']} label={s['label']} pred={s['pred']}: {s['gen']!r}", flush=True)
            results_path.write_text(json.dumps(results, indent=2))
        del model
        torch.cuda.empty_cache()

    print("\n=== summary (accuracy applying the TRUE rule) ===", flush=True)
    for key, protos in results["runs"].items():
        print(f"  {key:44s} " + "  ".join(f"{p}={v['overall']:.3f}" for p, v in protos.items()), flush=True)
    print(f"wrote {results_path}")


if __name__ == "__main__":
    main()
