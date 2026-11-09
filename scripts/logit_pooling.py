"""Logistic task: how do LLM twins pool the two probabilities?

Usage:
    python scripts/logit_pooling.py L8 results/logistic_llm_auto_own_rho0.0 [more folders] \
        [--from-step 30]

For steps >= from-step, with l_A, l_B the log-odds of the two logistic models and
m = (l_A + l_B) / 2, it fits (least squares, no intercept) the slope of the team's log-odds on m:
about 1 or slightly less is averaging (of log-odds, or of probabilities, which is a bit flatter), 2
is adding the log-odds, and the population blend is alpha + beta (`blend_alpha/beta` in
benchmarks.json). The same slope is given for the twins' final-round automatic logistic
regression (own_model + partner_open), with lambda in log-odds space,
logit(team) = m + lambda * (logit(reg) - m), and the excess log loss (over the population
full-information log loss) of: averaging probabilities, adding log-odds, the regression and the
team. The slope, lambda, excess log loss and disagreement (in probability) are also given per
round (team after round k = average of the twins' round-k probabilities).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consensus.models import log_loss, logit, sigmoid  # noqa: E402

PROB = re.compile(r"Probability that y = 1 for the current case: (\d+(?:\.\d+)?)")


def latest_run(path: Path) -> Path:
    if (path / "config.yaml").exists():
        return path
    return sorted(p for p in path.iterdir() if (p / "config.yaml").exists())[-1]


def slope(t: np.ndarray, m: np.ndarray) -> float:
    return float((t @ m) / (m @ m))


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("label")
    p.add_argument("folders", nargs="+")
    p.add_argument("--from-step", type=int, default=30, help="0-based first step included")
    args = p.parse_args(argv)

    rows, per_round, bench = [], [], None
    for folder in args.folders:
        run_dir = latest_run(Path(folder))
        bench = bench or json.loads((run_dir / "benchmarks.json").read_text())
        for f in sorted((run_dir / "transcripts").glob(f"{args.label}_run*.jsonl")):
            if "conversation" in f.name or "training" in f.name:
                continue
            for line in f.read_text().splitlines():
                r = json.loads(line)
                if r["t"] < args.from_step:
                    continue
                regs = [
                    float(mt.group(1))
                    for ag in ("A", "B")
                    for c in r["rounds"][-1][ag]["tool_calls"]
                    if c.get("auto") and "partner_open" in c["predictors"]
                    for mt in [PROB.search(c["result"])]
                    if mt
                ]
                if not regs:
                    continue
                pA, pB = r["model_pred"]["A"], r["model_pred"]["B"]
                rows.append((pA, pB, float(np.mean(logit(regs))), r["consensus"], r["y"]))
                per_round.append([(rd["A"]["estimate"], rd["B"]["estimate"]) for rd in r["rounds"]])
    if not rows:
        raise SystemExit("no steps with an automatic regression on partner_open found")
    pA, pB, l_reg, team, y = np.array(rows).T
    lA, lB = logit(pA), logit(pB)
    m = (lA + lB) / 2
    full = bench["logloss_full_info"]

    def ex(p):
        return float(np.mean(log_loss(p, y)) - full)

    def lam(lt):
        return float(((lt - m) @ (l_reg - m)) / ((l_reg - m) @ (l_reg - m)))

    out = {
        "n_steps": len(y),
        "population_blend_slope": bench["blend_alpha"] + bench["blend_beta"],
        "slope_avg_prob": slope(logit((pA + pB) / 2), m),
        "slope_regression": slope(l_reg, m),
        "slope_team": slope(logit(team), m),
        "lambda": lam(logit(team)),
        "excess_logloss": {
            "avg_prob": ex((pA + pB) / 2),
            "logit_sum": ex(sigmoid(lA + lB)),
            "regression": ex(sigmoid(l_reg)),
            "team": ex(team),
            "population_no_talk": bench["logloss_no_talk"] - full,
            "population_logit_sum": bench["logloss_logit_sum"] - full,
            "population_blend": bench["logloss_blend"] - full,
        },
        "per_round": [],
    }
    E = np.array(per_round)  # (n_steps, n_rounds, 2)
    for k in range(E.shape[1]):
        team_k = E[:, k].mean(1)
        out["per_round"].append(
            {
                "round": k + 1,
                "slope": slope(logit(team_k), m),
                "lambda": lam(logit(team_k)),
                "excess_logloss": ex(team_k),
                "disagreement": float(np.mean((E[:, k, 0] - E[:, k, 1]) ** 2)),
            }
        )
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
