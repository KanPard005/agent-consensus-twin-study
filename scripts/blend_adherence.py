"""How far do LLM twins follow the automatic regression, versus plain averaging?

Usage:
    python scripts/blend_adherence.py P8 results/llm_haiku_auto_own_rho0.0 results/llm_haiku_auto_own_rho0.0_r3 ...
        [--from-step 30]

Pools the latest run in each folder (all folders should share the env). For steps >= from-step it
models the team estimate as  team = avg + lambda * (reg - avg),  where avg is the average of the
two linear models' predictions and reg the mean of the twins' round-2 automatic regression
predictions (own_model + partner_open), and reports lambda with the excess MSE over sigma^2 of
avg, reg and the team. lambda = 0 is pure averaging, lambda = 1 is following the regression.
The same is reported per round (team after round k = average of the twins' round-k estimates),
with the twins' disagreement, to see whether extra exchanges move them toward the regression.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import yaml

PRED = re.compile(r"Prediction for the current case: (-?\d+(?:\.\d+)?)")


def latest_run(path: Path) -> Path:
    if (path / "config.yaml").exists():
        return path
    return sorted(p for p in path.iterdir() if (p / "config.yaml").exists())[-1]


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("label")
    p.add_argument("folders", nargs="+")
    p.add_argument("--from-step", type=int, default=30, help="0-based first step included")
    args = p.parse_args(argv)

    rows, per_round, sigma2 = [], [], None
    for folder in args.folders:
        run_dir = latest_run(Path(folder))
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
        sigma2 = cfg["env"]["sigma"] ** 2
        for f in sorted((run_dir / "transcripts").glob(f"{args.label}_run*.jsonl")):
            if "conversation" in f.name or "training" in f.name:
                continue
            for line in f.read_text().splitlines():
                r = json.loads(line)
                if r["t"] < args.from_step:
                    continue
                regs = [
                    float(m.group(1))
                    for ag in ("A", "B")
                    for c in r["rounds"][-1][ag]["tool_calls"]
                    if c.get("auto") and "partner_open" in c["predictors"]
                    for m in [PRED.search(c["result"])]
                    if m
                ]
                if regs:
                    avg = (r["model_pred"]["A"] + r["model_pred"]["B"]) / 2
                    rows.append((avg, np.mean(regs), r["consensus"], r["y"]))
                    per_round.append(
                        [(rd["A"]["estimate"], rd["B"]["estimate"]) for rd in r["rounds"]]
                    )
    if not rows:
        raise SystemExit("no steps with an automatic regression on partner_open found")
    avg, reg, team, y = np.array(rows).T
    lam = float(((team - avg) @ (reg - avg)) / ((reg - avg) @ (reg - avg)))
    out = {
        "n_steps": len(y),
        "lambda": lam,
        "excess_avg": float(np.mean((avg - y) ** 2) - sigma2),
        "excess_regression": float(np.mean((reg - y) ** 2) - sigma2),
        "excess_team": float(np.mean((team - y) ** 2) - sigma2),
        "per_round": [],
    }
    E = np.array(per_round)  # (n_steps, n_rounds, 2)
    for k in range(E.shape[1]):
        team_k = E[:, k].mean(1)
        out["per_round"].append(
            {
                "round": k + 1,
                "lambda": float(((team_k - avg) @ (reg - avg)) / ((reg - avg) @ (reg - avg))),
                "excess_team": float(np.mean((team_k - y) ** 2) - sigma2),
                "disagreement": float(np.mean((E[:, k, 0] - E[:, k, 1]) ** 2)),
            }
        )
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
