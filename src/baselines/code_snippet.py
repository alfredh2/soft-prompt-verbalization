"""Ask for a Python function instead of an English sentence, and run it on held-out inputs.

The description methods elsewhere are scored by handing their sentence to an LLM executor,
which caps the ceiling at ~0.996 and adds parse noise. A code snippet removes that
instrument entirely: the reconstruction is executed, so held-out accuracy is exact, free,
and has a true ceiling of 1.0. It also removes prose ambiguity -- "positive" and "between
13 and 20" need interpreting, `x > 0` does not.

This is the clean measuring stick, not the project's deliverable: the research question is
about natural-language descriptions. Use code for sample-complexity curves and baselines,
English for the verbalization claim.

Executing model-written code is the obvious hazard here, so `run_snippet` refuses anything
containing imports, dunders, or filesystem/process access, and evaluates with a restricted
builtins namespace under a wall-clock limit.
"""

from __future__ import annotations

import multiprocessing as mp
import re

from src.datasets.discrete import format_examples

SYSTEM = "You infer the rule behind a hidden function from examples of its behaviour."
USER = (
    "Here are input/output pairs of a hidden function f:\n{examples}\n\n"
    "The output is always one of: {cands}.\n\n"
    "{grammar}"
    "Work out the rule, then write it as a Python function.\n\n"
    "Requirements:\n"
    "- signature exactly `def f(x):`\n"
    "- `x` is {xtype}\n"
    "- return one of {cands} as a string\n"
    "- no imports, no input/output, no helper functions\n"
    "- it must generalise to inputs not shown, so prefer a short rule over listing cases\n\n"
    "Reply with a single ```python code block containing only that function."
)

BANNED = re.compile(
    r"\b(import|__|open|exec|eval|compile|globals|locals|getattr|setattr|delattr|"
    r"subprocess|os\.|sys\.|socket|shutil|pathlib|breakpoint|input)\b"
)

SAFE_BUILTINS = {
    "abs": abs, "min": min, "max": max, "sum": sum, "len": len, "int": int, "str": str,
    "bool": bool, "float": float, "round": round, "sorted": sorted, "list": list,
    "tuple": tuple, "set": set, "dict": dict, "range": range, "enumerate": enumerate,
    "zip": zip, "any": any, "all": all, "map": map, "filter": filter, "divmod": divmod,
    "pow": pow, "reversed": reversed, "ord": ord, "chr": chr, "bin": bin,
}


def build_prompt(record: dict) -> str:
    xtype = ("a string of 6 characters, each '0' or '1'"
             if record["domain"] == "bits" else "an int")
    # Optional: tell the model the FORM of the rule. "Identifiable" is relative to the
    # generator's own grammar, which a model is never told, so a failure to recover a
    # function that the examples pin down within that grammar may be missing prior
    # knowledge rather than failed inference. Supplying the grammar separates the two.
    grammar = record.get("grammar")
    return USER.format(
        examples=format_examples(record["examples"]),
        cands=", ".join(repr(c) for c in record["labels"]),
        xtype=xtype,
        grammar=f"About the rule: {grammar}\n\n" if grammar else "",
    )


def extract_code(text: str) -> tuple[str, bool]:
    """Pull the function out of a fenced block. Returns (code, whether a block was found)."""
    m = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL)
    if m:
        return m[-1].strip(), True
    # Some replies omit the fence; salvage from the first `def f(`
    i = text.find("def f(")
    return (text[i:].strip(), False) if i != -1 else (text.strip(), False)


def _worker(code: str, inputs: list, q) -> None:
    try:
        ns: dict = {"__builtins__": SAFE_BUILTINS}
        exec(code, ns)
        fn = ns.get("f")
        if not callable(fn):
            q.put(("error", "no callable f"))
            return
        q.put(("ok", [str(fn(x)) for x in inputs]))
    except Exception as exc:
        q.put(("error", f"{type(exc).__name__}: {exc}"))


def run_snippet(code: str, inputs: list, timeout: float = 5.0) -> tuple[list[str] | None, str]:
    """Execute a model-written function on inputs. Returns (outputs, error_message).

    Runs in a separate process so an infinite loop or a crash cannot take the run down.
    """
    if BANNED.search(code):
        return None, "rejected: banned construct"
    if "def f(" not in code:
        return None, "rejected: no def f("
    q: mp.Queue = mp.Queue()
    p = mp.Process(target=_worker, args=(code, inputs, q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return None, "timeout"
    if q.empty():
        return None, "no result"
    status, payload = q.get()
    return (payload, "") if status == "ok" else (None, payload)


def score_snippet(code: str, held_out: list) -> tuple[float | None, str]:
    """Exact held-out accuracy of a snippet, or (None, reason) if it could not run."""
    outs, err = run_snippet(code, [x for x, _ in held_out])
    if outs is None:
        return None, err
    gold = [y for _, y in held_out]
    return sum(o == g for o, g in zip(outs, gold)) / len(gold), ""


def evaluate_snippet(code: str, function, shown: list, held_out: list) -> dict:
    """Score one snippet five ways, from a single execution over the whole domain.

    `exact` is the primary metric for these toy functions: the snippet agrees with the
    true rule on every input in the domain. The others exist because each can be high
    while the rule is wrong --

      support      fits the examples it was shown; says nothing about generalisation
      held_out     the ~16 reserved inputs; a near-miss threshold often survives it
                   untouched, which is why two snippets scored 1.00 here with the wrong
                   rule
      probe        inputs chosen to falsify near-misses (see discrete.probe_inputs).
                   `probe_unseen` reports how many were absent from `shown`: once a method
                   is given most of the domain, probe accuracy collapses into support
                   consistency and stops measuring generalisation
      disagreement fraction of the domain where the snippet and the true rule differ --
                   `1 - exact` as a continuous quantity, so near-misses are distinguishable
                   from unrelated rules
    """
    from src.datasets.discrete import probe_examples

    all_inputs = list(function.values)
    outs, err = run_snippet(code, all_inputs)
    if outs is None:
        return {"ran": False, "error": err, "exact": None, "domain_acc": None,
                "disagreement": None, "support": None, "held_out": None, "probe": None}

    truth = {x: function.compute(x) for x in all_inputs}
    pred = dict(zip(all_inputs, outs))
    wrong = [x for x in all_inputs if pred[x] != truth[x]]

    def acc(pairs):
        return sum(pred[x] == y for x, y in pairs) / len(pairs) if pairs else None

    probes = probe_examples(function)
    shown_inputs = {x for x, _ in shown}
    return {
        "ran": True,
        "error": "",
        "exact": len(wrong) == 0,
        "domain_acc": round(1 - len(wrong) / len(all_inputs), 4),
        "disagreement": round(len(wrong) / len(all_inputs), 4),
        "support": round(acc(shown), 4) if shown else None,
        "held_out": round(acc(held_out), 4),
        "probe": round(acc(probes), 4) if probes else None,  # none defined for composed rules
        "probe_unseen": sum(x not in shown_inputs for x, _ in probes),
        "probe_n": len(probes),
        "n_wrong": len(wrong),
    }
