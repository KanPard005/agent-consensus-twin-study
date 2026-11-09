"""Logistic task: expected excess log loss per protocol, using the true probabilities.

Usage:
    python scripts/expected_logloss.py results/logistic_llm_auto_own_rho0.0 [more folders] \
        [--from-step 30] [--paired L8]

Realized log loss on 0/1 labels is very noisy over a few dozen steps. For each run this script
regenerates the post-training stream from the run seed (as `scripts/run.py` draws it: training
samples of A, then of B, then the T post-training samples), recovers each protocol's consensus
probability p from its recorded log loss and the label, and scores it against the true
probability p*:  E[log loss | x] - Bayes = KL(p* || p),  averaged over steps >= from-step. It
prints the mean per run for every protocol, and with --paired LABEL the paired differences of
LABEL against every other protocol.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consensus.data import LogisticTwinEnv, build_env  # noqa: E402
from consensus.models import P_CLIP  # noqa: E402
from consensus.utils import load_run, run_rng  # noqa: E402


def latest_run(path: Path) -> Path:
    if (path / "config.yaml").exists():
        return path
    return sorted(p for p in path.iterdir() if (p / "config.yaml").exists())[-1]


def expected_excess(run_dir: Path, from_step: int) -> dict[str, np.ndarray]:
    """{label: per-run mean of KL(p* || p) over steps >= from_step}."""
    cfg, res, _ = load_run(run_dir)
    env = build_env(cfg["env"], np.random.default_rng(cfg["seed"]))
    if not isinstance(env, LogisticTwinEnv):
        raise SystemExit(f"{run_dir} is not a logistic run")
    T, first = cfg["post"]["T"], cfg.get("first_run", 0)
    out = {lab: [] for lab in res if "log_loss" in res[lab]}
    for i in range(cfg["n_runs"]):
        rng = run_rng(cfg["seed"], first + i)
        env.sample(cfg["train"]["n_A"], rng)
        env.sample(cfg["train"]["n_B"], rng)
        x_A, x_B, y = env.sample(T, rng)
        ps = env.true_prob(x_A, x_B)
        bayes = -ps * np.log(ps) - (1 - ps) * np.log(1 - ps)
        for lab in out:
            ll = res[lab]["log_loss"][i]
            p = np.where(y == 1, np.exp(-ll), 1 - np.exp(-ll))
            if not np.allclose(np.where(y == 1, -np.log(p), -np.log(1 - p)), ll):
                raise SystemExit(f"stream of {run_dir} does not match its labels")
            q = np.clip(p, P_CLIP, 1 - P_CLIP)
            kl = -ps * np.log(q) - (1 - ps) * np.log(1 - q) - bayes
            out[lab].append(kl[from_step:].mean())
    return {lab: np.array(v) for lab, v in out.items()}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("folders", nargs="+")
    p.add_argument("--from-step", type=int, default=30, help="0-based first step included")
    p.add_argument("--paired", help="label to compare against the others, run by run")
    args = p.parse_args(argv)
    for folder in args.folders:
        run_dir = latest_run(Path(folder))
        ex = expected_excess(run_dir, args.from_step)
        print(f"{run_dir}  (expected excess log loss over Bayes, steps >= {args.from_step + 1})")
        for lab, v in ex.items():
            se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
            print(f"  {lab:5s} {v.mean():.3f} ± {se:.3f}   per run {np.round(v, 3).tolist()}")
        if args.paired and args.paired in ex:
            for lab, v in ex.items():
                if lab == args.paired:
                    continue
                d = ex[args.paired] - v
                print(
                    f"  {args.paired} - {lab}: {d.mean():+.3f}  ({int((d < 0).sum())}/{len(d)} runs "
                    f"better)   per run {np.round(d, 3).tolist()}"
                )


if __name__ == "__main__":
    main()
