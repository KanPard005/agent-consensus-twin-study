"""Does the LLM know how to combine two estimates built from different information, when it is
told the structure in words? A one-shot probe with no learning phase.

Usage:
    python scripts/combine_probe.py [--reps 2] [--model claude-haiku-5-5] [--effort low]
        [--workers 8] [--conditions ...]

Each call gives the model (as twin A) its own estimate e_A of y and its partner's estimate e_B,
plus a description of who knows what, and asks for its best estimate of y. Conditions, with the
correct combination:

- formula_indep: y = x1 + x2 + noise, x1 and x2 independent (mean 0, variance 1); A knows x1,
  B knows x2, and each reports E[y | own x]. Correct: e_A + e_B.
- formula_corr: as above with corr(x1, x2) = 0.5, so E[y | x1] = 1.5 x1 and
  E[y | x1, x2] = x1 + x2 = (e_A + e_B) / 1.5. Correct: (e_A + e_B) / 1.5.
- same_info: both see the same inputs and estimate y with independent errors. Correct: the
  average (e_A + e_B) / 2.
- verbal_sum: the task's wording (features a1-a3 vs b1-b3) plus "y is the sum of a part that
  depends on your features and a part that depends on your partner's; the two sets are
  unrelated". Correct: e_A + e_B.
- verbal_task: the task's wording only (the partner sees different variables, as in the S1n
  prompt). The structure is not stated, so there is no single correct answer; it shows the
  default rule.
- verbal_unrelated / verbal_additive: verbal_task plus only one half of verbal_sum's structure
  (the feature sets are unrelated; or y is the sum of the two parts). Neither has a single
  correct answer; they show which statement flips the rule.

For each condition the answers are fitted as estimate = a e_A + b e_B (no intercept), and
classified as the sum, the average, or other (within 0.05). The paper-and-pencil question: if the
model adds in formula_indep / verbal_sum but averages in the experiments, the failure there is in
recognizing the structure, not in knowing the rule.

Writes results/combine_probe/<timestamp>/ (answers.jsonl, summary.json).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consensus.llm_backends import ClaudeCLIBackend  # noqa: E402
from consensus.utils import git_hash, make_run_dir  # noqa: E402

log = logging.getLogger("combine_probe")

# (e_A, e_B): same signs, opposite signs, unequal sizes, so sum and average are far apart.
PAIRS = [(1.2, 0.8), (-0.5, 1.5), (2.0, -0.6), (-1.1, -0.9), (0.3, 1.7), (-1.6, 0.4)]

SYSTEM = (
    "You are twin A, one of two forecasters working as a team to predict a number y. "
    "Answer with your best estimate of y as a single number."
)

STRUCTURES = {
    "formula_indep": (
        "y = x1 + x2 + noise, where x1 and x2 are independent, each with mean 0 and variance 1, "
        "and the noise has mean 0. You observe x1 only; your partner, twin B, observes x2 only. "
        "Each of you has reported your best estimate of y given only what you observe."
    ),
    "formula_corr": (
        "y = x1 + x2 + noise, where x1 and x2 each have mean 0 and variance 1 and are "
        "correlated with correlation 0.5, and the noise has mean 0. You observe x1 only; your "
        "partner, twin B, observes x2 only. Each of you has reported your best estimate of y "
        "given only what you observe."
    ),
    "same_info": (
        "You and your partner, twin B, observe exactly the same inputs for this case. Each of "
        "you has estimated y from them, and your estimation errors are independent of each "
        "other."
    ),
    "verbal_sum": (
        "You see features a1, a2 and a3 of the case. Your partner, twin B, sees features b1, b2 "
        "and b3, which are different variables that you do not see. y is the sum of a part that "
        "depends only on your features and a part that depends only on your partner's features, "
        "plus noise, and the two sets of features are unrelated to each other. Each of you has "
        "estimated y from your own features."
    ),
    "verbal_task": (
        "You see features a1, a2 and a3 of the case. Your partner, twin B, sees features b1, b2 "
        "and b3. They are different variables from yours, so your partner knows things about "
        "the case that you do not. Each of you has estimated y from your own features."
    ),
    # Ablations of verbal_sum: which half of the structure flips the rule?
    "verbal_unrelated": (
        "You see features a1, a2 and a3 of the case. Your partner, twin B, sees features b1, b2 "
        "and b3. They are different variables from yours, so your partner knows things about "
        "the case that you do not, and the two sets of features are unrelated to each other. "
        "Each of you has estimated y from your own features."
    ),
    "verbal_additive": (
        "You see features a1, a2 and a3 of the case. Your partner, twin B, sees features b1, b2 "
        "and b3. They are different variables from yours, so your partner knows things about "
        "the case that you do not. y is the sum of a part that depends only on your features "
        "and a part that depends only on your partner's features, plus noise. Each of you has "
        "estimated y from your own features."
    ),
}

CORRECT = {
    "formula_indep": lambda a, b: a + b,
    "formula_corr": lambda a, b: (a + b) / 1.5,
    "same_info": lambda a, b: (a + b) / 2,
    "verbal_sum": lambda a, b: a + b,
    "verbal_task": lambda a, b: float("nan"),
    # Each half alone leaves the answer open (unknown dependence, or unknown functional form).
    "verbal_unrelated": lambda a, b: float("nan"),
    "verbal_additive": lambda a, b: float("nan"),
}

SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string", "description": "one or two sentences"},
        "estimate": {"type": "number"},
    },
    "required": ["reasoning", "estimate"],
}


def prompt(condition: str, e_a: float, e_b: float) -> str:
    return (
        f"{STRUCTURES[condition]}\n\n"
        f"Your estimate of y: {e_a:g}\nTwin B's estimate of y: {e_b:g}\n\n"
        "What is your best estimate of y now, using both estimates?"
    )


def summarize(rows: list[dict]) -> dict:
    """Per condition: fitted weights (a, b) and their sum, and the share of sum / average /
    other answers."""
    out = {}
    for cond in STRUCTURES:
        r = [x for x in rows if x["condition"] == cond and x["estimate"] is not None]
        if not r:
            continue
        X = np.array([[x["e_A"], x["e_B"]] for x in r])
        y = np.array([x["estimate"] for x in r])
        (wa, wb), *_ = np.linalg.lstsq(X, y, rcond=None)
        s = X.sum(1)
        is_sum = np.abs(y - s) < 0.05
        is_avg = np.abs(y - s / 2) < 0.05
        out[cond] = {
            "n": len(r),
            "w_A": round(float(wa), 3),
            "w_B": round(float(wb), 3),
            "w_sum": round(float(wa + wb), 3),
            "share_sum": round(float(is_sum.mean()), 3),
            "share_average": round(float(is_avg.mean()), 3),
            "share_other": round(float((~is_sum & ~is_avg).mean()), 3),
            "unusable": sum(x["estimate"] is None for x in rows if x["condition"] == cond),
        }
    return out


def main(argv=None) -> Path:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--model", default="claude-haiku-5-5")
    p.add_argument("--effort", default="low")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--conditions", nargs="+", default=list(STRUCTURES))
    p.add_argument("--results", default=str(ROOT / "results"))
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")

    backend = ClaudeCLIBackend(model=args.model, effort=args.effort)
    jobs = list(product(args.conditions, PAIRS, range(args.reps)))

    def call(job):
        cond, (e_a, e_b), rep = job
        reply = backend.complete(SYSTEM, prompt(cond, e_a, e_b), SCHEMA)
        d = reply.data or {}
        return {
            "condition": cond,
            "e_A": e_a,
            "e_B": e_b,
            "rep": rep,
            "estimate": d.get("estimate"),
            "reasoning": d.get("reasoning"),
            "correct": CORRECT[cond](e_a, e_b),
            "cost_usd": reply.cost_usd,
        }

    with ThreadPoolExecutor(args.workers) as ex:
        rows = list(ex.map(call, jobs))

    run_dir = make_run_dir(args.results, "combine_probe")
    with (run_dir / "answers.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summary = {
        "args": vars(args),
        "git": git_hash(ROOT),
        "cost_usd": round(float(np.nansum([r["cost_usd"] for r in rows])), 3),
        "conditions": summarize(rows),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["conditions"], indent=1))
    log.info("cost $%.3f; saved to %s", summary["cost_usd"], run_dir)
    return run_dir


if __name__ == "__main__":
    main()
