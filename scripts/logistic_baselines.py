"""Logistic task: how the non-LLM protocols pool, plus a from-scratch regression baseline.

Usage:
    python scripts/logistic_baselines.py results/logistic_t60_rho0.0 [--from-step 30]

For a logistic run folder (latest run inside it), regenerates each run's twins and stream from
the seed (as `scripts/run.py` does) and reports, over steps >= from-step:
- P1's learned blend alpha + beta at steps 1, 10, 30 and T (population value from benchmarks);
- for every protocol, the slope of the consensus log-odds on the mean of the two solo models'
  log-odds (averaging probabilities ~0.85, adding log-odds 2);
- a "tool" baseline like the LLM twins' automatic regression: at each step a logistic regression
  of y on (1, logit p_A, logit p_B) refitted from scratch on the past steps (prior variance 100,
  average of probabilities for the first 5 steps), with its expected excess log loss over Bayes
  early (steps 6..from-step) and late, and its slope.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from expected_logloss import latest_run  # noqa: E402
from run import train_twins  # noqa: E402

from consensus.data import build_env  # noqa: E402
from consensus.models import P_CLIP, fit_logistic, logit, sigmoid  # noqa: E402
from consensus.utils import load_run, run_rng  # noqa: E402


def kl(p: np.ndarray, ps: np.ndarray) -> np.ndarray:
    q = np.clip(p, P_CLIP, 1 - P_CLIP)
    return -ps * np.log(q) - (1 - ps) * np.log(1 - q) + ps * np.log(ps) + (1 - ps) * np.log(1 - ps)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("folder")
    p.add_argument("--from-step", type=int, default=30, help="0-based first step included")
    args = p.parse_args(argv)
    run_dir = latest_run(Path(args.folder))
    cfg, res, bench = load_run(run_dir)
    env = build_env(cfg["env"], np.random.default_rng(cfg["seed"]))
    T, s0, first = cfg["post"]["T"], args.from_step, cfg.get("first_run", 0)
    sl = slice(s0, T)
    if "P1" in res:
        ab = res["P1"]["alpha"] + res["P1"]["beta"]
        steps = [1, 10, 30, T]
        print(
            "P1 alpha + beta at steps "
            + ", ".join(f"{t}: {ab[:, t - 1].mean():.2f}" for t in steps)
            + f" (population {bench['blend_alpha'] + bench['blend_beta']:.2f})"
        )
    slopes = {lab: [] for lab in res if "log_loss" in res[lab]}
    slopes["tool"] = []
    tool_late, tool_early = [], []
    for i in range(cfg["n_runs"]):
        rng = run_rng(cfg["seed"], first + i)
        A, B = train_twins(env, cfg["train"], rng)
        x_A, x_B, y = env.sample(T, rng)
        A.begin_communication(0, frozen=True)
        B.begin_communication(0, frozen=True)
        pA = np.array([A.predict(a, []) for a in x_A])
        pB = np.array([B.predict(b, []) for b in x_B])
        ps = env.true_prob(x_A, x_B)
        m = (logit(pA) + logit(pB)) / 2
        tp = (pA + pB) / 2
        for t in range(5, T):
            X = np.column_stack([np.ones(t), logit(pA[:t]), logit(pB[:t])])
            th, _ = fit_logistic(X, y[:t], P0=100 * np.eye(3))
            tp[t] = sigmoid(np.array([1.0, logit(pA[t]), logit(pB[t])]) @ th)
        tool_late.append(kl(tp, ps)[sl].mean())
        tool_early.append(kl(tp, ps)[5:s0].mean())
        for lab in slopes:
            if lab == "tool":
                q = tp
            else:
                ll = res[lab]["log_loss"][i]
                q = np.where(y == 1, np.exp(-ll), 1 - np.exp(-ll))
            slopes[lab].append(float(logit(q)[sl] @ m[sl] / (m[sl] @ m[sl])))
    print(
        f"slope of consensus log-odds on mean model log-odds (steps {s0 + 1}-{T}): "
        + ", ".join(f"{k} {np.mean(v):.2f}" for k, v in slopes.items())
    )
    se = np.std(tool_late, ddof=1) / np.sqrt(len(tool_late))
    print(
        f"from-scratch regression on both log-odds: steps {s0 + 1}-{T} "
        f"{np.mean(tool_late):.3f} ± {se:.3f}, steps 6-{s0} {np.mean(tool_early):.3f}"
    )


if __name__ == "__main__":
    main()
