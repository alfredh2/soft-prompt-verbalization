"""Where does a grammar-aware frontier model break? Dense-support probe of candidate tiers.

Exploratory, not a benchmark. Each family is shown to the model with its grammar in the
prompt and a DENSE random sample of the domain (64 or 160 of 301 inputs), so a failure is the
model's inference limit and not scarce data. Depth-2 is the control: it is the tier the model
already solves, so any drop at the same N is due to the family's complexity.

  d2     A op B                            (the existing tier)
  d3     (A op B) op C                      three conditions, left to right
  dlist  if A: p / elif B: q / elif C: r / else s     four outputs, decision list
  lin    ((a*x + b) % m) < t                          modular-arithmetic rule, search-hard
  bit    popcount(x & mask) % 2 == p                  parity of a subset of bits, search-hard

Identifiability is NOT certified here (the d3 and dlist classes are too large to enumerate
exactly); that is what a real tier built from the winning family would add.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baselines.api_proposer import Usage, _client, chat_with_retry, model_id  # noqa: E402
from src.baselines.code_snippet import SYSTEM, build_prompt, evaluate_snippet, extract_code  # noqa: E402
from src.datasets.compositional import (atom_pred, atom_text, domain_spec, domain_values,  # noqa: E402
                                        make_atoms, grammar_text)
from src.datasets.discrete import TargetFunction  # noqa: E402

DS = domain_spec()
ATOMS = make_atoms(DS)
THR = [a for a in ATOMS if a[0] != "mod"]
MOD = [a for a in ATOMS if a[0] == "mod"]
VALUES = domain_values(DS)
COND = ("conditions are: x > c, x < c, abs(x) > c, or x % k == r, where c is an integer, k is an "
        "integer from 2 to 9 and r is a remainder from 0 to k-1")
GRAMMAR = {
    "d2": grammar_text(DS),
    "d3": (f"it is built from three simple conditions on x, A, B and C, as \"(A op B) op C\" where each "
           f"op is \"and\", \"or\" or \"and not\", applied left to right. Each of the {COND}. "
           f"The inputs are the integers from {DS['int_low']} to {DS['int_high']}."),
    "dlist": (f"it is a decision list over three simple conditions on x: \"if A then p, otherwise if B "
              f"then q, otherwise if C then r, otherwise s\", where p, q, r and s are four different "
              f"outputs. Each of the {COND}. The inputs are the integers from {DS['int_low']} to "
              f"{DS['int_high']}."),
    "lin": (f"it has the form ((a*x + b) % m) < t, where a, b, m and t are integers with m from 5 to 16, "
            f"a from 1 to m-1, b from 0 to m-1 and t from 1 to m-1. The inputs are the integers from "
            f"{DS['int_low']} to {DS['int_high']}."),
    "bit": (f"it has the form popcount(x & mask) % 2 == p, where mask is an integer from 1 to 255 (so only "
            f"the low 8 bits of x matter), popcount counts the 1 bits, and p is 0 or 1. The inputs are "
            f"the integers from {DS['int_low']} to {DS['int_high']}."),
}


def draw_atom(rnd):
    return rnd.choice(THR) if rnd.random() < 0.5 else rnd.choice(MOD)


def combine(op, u, v):
    return {"and": u and v, "or": u or v, "andnot": u and not v}[op]


def make(family, rnd):
    """One random function of the family, rejecting degenerate (near-constant) ones."""
    while True:
        if family in ("lin", "bit"):
            labels = ("0", "1")
            if family == "lin":
                m = rnd.randint(5, 16); a_, b_, t_ = rnd.randint(1, m - 1), rnd.randint(0, m - 1), rnd.randint(1, m - 1)
                fn = lambda x, m=m, a_=a_, b_=b_, t_=t_: "1" if (a_ * x + b_) % m < t_ else "0"
                desc = f"((({a_}*x + {b_}) % {m}) < {t_})"
            else:
                mask, par = rnd.randint(1, 255), rnd.randint(0, 1)
                fn = lambda x, mask=mask, par=par: "1" if bin(x & mask).count("1") % 2 == par else "0"
                desc = f"(popcount(x & {mask}) % 2 == {par})"
        elif family == "dlist":
            atoms, labels = [draw_atom(rnd) for _ in range(3)], ("p", "q", "r", "s")
            preds = [atom_pred(a) for a in atoms]
            fn = lambda x, p=preds: (labels[0] if p[0](x) else labels[1] if p[1](x)
                                     else labels[2] if p[2](x) else labels[3])
            desc = " ".join(f"if {atom_text(a)[1]} then {l};" for a, l in zip(atoms, labels)) + " else s"
            ops = ()
        else:
            n = 2 if family == "d2" else 3
            atoms, labels = [draw_atom(rnd) for _ in range(n)], ("0", "1")
            ops = tuple(rnd.choice(("and", "or", "andnot")) for _ in range(n - 1))
            preds = [atom_pred(a) for a in atoms]

            def fn(x, p=preds, o=ops):
                v = p[0](x)
                for op, q in zip(o, p[1:]):
                    v = combine(op, v, q(x))
                return "1" if v else "0"
            desc = f"{atom_text(atoms[0])[1]} " + " ".join(f"{o} {atom_text(a)[1]}" for o, a in zip(ops, atoms[1:]))
        counts = {l: sum(fn(x) == l for x in VALUES) for l in labels}
        if max(counts.values()) / len(VALUES) <= (0.75 if family != "dlist" else 0.6) and min(counts.values()) >= 12:
            fid = f"{family}:{desc}"
            return TargetFunction(function_id=fid, family=family, params={}, domain="int", labels=labels,
                                  description=desc, paraphrases=(desc,), expression=desc, compute=fn,
                                  values=VALUES), max(counts.values()) / len(VALUES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="d2,d3,dlist")
    ap.add_argument("--sizes", default="64,160")
    ap.add_argument("--functions", type=int, default=20)
    ap.add_argument("--model", default="openai:gpt-6.1-sol")
    ap.add_argument("--max-tokens", type=int, default=10000)
    ap.add_argument("--price-in", type=float, default=2.0)
    ap.add_argument("--price-out", type=float, default=10.0)
    ap.add_argument("--max-usd", type=float, default=5.0)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", default="results/probe_hard_tiers")
    a = ap.parse_args()
    families, sizes = a.families.split(","), [int(s) for s in a.sizes.split(",")]
    jobs = []
    for fam in families:
        rnd = random.Random(f"probe-{fam}")
        for i in range(a.functions):
            fn, floor = make(fam, rnd)
            order = list(VALUES); random.Random(f"{fam}-{i}").shuffle(order)
            for n in sizes:
                shown = [(x, fn.compute(x)) for x in order[:n]]
                held = [(x, fn.compute(x)) for x in order[n:n + 16]]
                jobs.append((fam, i, fn, floor, n, shown, held))
    est = len(jobs) * (1500 + 16 * max(sizes)) / 1e6 * a.price_in + len(jobs) * a.max_tokens / 1e6 * a.price_out
    print(f"{len(jobs)} calls, worst case ${est:.2f} (cap ${a.max_usd})")
    if est > a.max_usd:
        raise SystemExit("ABORT: worst case exceeds --max-usd; lower --functions, --sizes or --max-tokens")
    usage, client, mid = Usage(), _client(), model_id(a.model)

    def run(job):
        fam, i, fn, floor, n, shown, held = job
        rec = {"examples": shown, "labels": list(fn.labels), "domain": "int", "grammar": GRAMMAR[fam]}
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": build_prompt(rec)}]
        try:
            reply, finish = chat_with_retry(client, mid, msgs, a.max_tokens, usage)
        except Exception as exc:
            return {"family": fam, "i": i, "n": n, "ran": False, "error": type(exc).__name__}
        code, _ = extract_code(reply)
        m = evaluate_snippet(code, fn, shown, held)
        return {"family": fam, "i": i, "n": n, "floor": round(floor, 3), "finish": finish, "rule": fn.description, "code": code, **m}

    with ThreadPoolExecutor(a.workers) as pool:
        rows = list(pool.map(run, jobs))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "rows.json").write_text(json.dumps(rows, indent=1) + "\n")
    u = usage.as_dict(a.price_in, a.price_out); (out / "usage.json").write_text(json.dumps(u, indent=1) + "\n")
    print(f"spent ~${u.get('estimated_usd', 0):.2f}\n\n{'family':<7}{'N':>5}{'floor':>7}{'exact':>8}{'domain':>8}{'support':>9}{'ran':>6}")
    for fam in families:
        for n in sizes:
            r = [x for x in rows if x["family"] == fam and x["n"] == n]
            ok = [x for x in r if x.get("ran")]
            print(f"{fam:<7}{n:>5}{mean(x.get('floor', 0) for x in ok):>7.2f}{sum(bool(x['exact']) for x in ok)/len(r):>8.2f}"
                  f"{mean(x['domain_acc'] for x in ok):>8.3f}{mean(x['support'] for x in ok):>9.3f}{len(ok):>4}/{len(r)}")


if __name__ == "__main__":
    main()
