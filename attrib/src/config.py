"""Run selector. RUN=bert (default, the original layout) or RUN=roberta (results/roberta/, data/roberta/)."""
import os

RUN = os.environ.get("RUN", "bert")
RUNS = {
    "bert": {"base": "bert-base-uncased", "cue": {1: "frankly", 0: "reportedly"}, "seed": 0},
    # second classifier: different architecture and tokenizer, different cue words, different seed
    "topic": {"base": "bert-base-uncased", "cue": {}, "seed": 2},
    "roberta": {"base": "FacebookAI/roberta-base", "cue": {1: "honestly", 0: "apparently"}, "seed": 1},
}
CFG = RUNS[RUN]


def P(path: str) -> str:
    """Map a project path into the active run's folder (bert keeps the original flat layout)."""
    if RUN == "bert":
        return path
    head, _, rest = path.partition("/")
    return f"{head}/{RUN}/{rest}"
