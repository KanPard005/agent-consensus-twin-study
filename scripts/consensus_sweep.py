"""Consensus game with ideal and learned linear twins: how often do they agree, and how fast?

Usage:
    python scripts/consensus_sweep.py [--tol 0.1] [--seeds 30] [--T 1000] [--n 200]

Three parts, all saved to results/consensus_sweep/<timestamp>/ (sweep.npz, summary.json,
consensus_sweep.png):

1. Ideal twins, paired covariance (`data.block_covariance`, d_A = d_B = 3): the population
   Geanakoplos-Polemarchakis dialogue (`theory.agreement_path`) on a grid of rho. For each round
   k the share of cases agreeing within `tol` (`theory.agreement_probability`), averaged over
   `--envs` random weight directions; also the opening disagreement E[(m^A_0 - m^B_0)^2].
2. Ideal twins, general covariance: random Wishart Sigma with d_A = d_B = d, rounds to exact
   agreement (`theory.rounds_to_agree`) and the share agreeing within `tol` by round k, vs d.
   Here "more disparate" = more private directions.
3. Learned linear twins (`consensus_game.ConsensusExchange`, 3 exchanges = 4 rounds, n training
   cases each, then T online steps): tail share agreeing by round k, the penalized game loss and
   the excess squared error of the agreed value, vs rho. One random environment per seed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run import train_twins  # noqa: E402

from consensus.consensus_game import ConsensusExchange  # noqa: E402
from consensus.data import LinearTwinEnv  # noqa: E402
from consensus.plotting_consensus import plot_consensus_sweep  # noqa: E402
from consensus.theory import (  # noqa: E402
    agreement_path,
    agreement_probability,
    rounds_to_agree,
)
from consensus.utils import git_hash, make_run_dir, run_rng  # noqa: E402

log = logging.getLogger("consensus_sweep")


def env_cfg(d: int, rho: float, sigma: float) -> dict:
    return dict(d_A=d, d_B=d, rho=rho, norm_A=1.0, norm_B=1.0, sigma=sigma)


def ideal_paired(rhos, d, sigma, tol, K, n_envs) -> dict:
    """Part 1: population dialogue on the paired covariance, averaged over weight draws."""
    agree = np.zeros((len(rhos), K))
    open_dis = np.zeros(len(rhos))
    excess = np.zeros((len(rhos), K))
    for i, rho in enumerate(rhos):
        for s in range(n_envs):
            env = LinearTwinEnv.from_config(env_cfg(d, rho, sigma), np.random.default_rng(s))
            path = agreement_path(env.Sigma, env.w, d, sigma, K - 1)
            agree[i] += agreement_probability(path, tol) / n_envs
            open_dis[i] += path.disagreement[0] / n_envs
            excess[i] += (path.mse_consensus - sigma**2) / n_envs
    return dict(agree=agree, open_dis=open_dis, excess=excess)


def ideal_general(ds, sigma, tol, K, n_envs, rng) -> dict:
    """Part 2: random Wishart covariances; rounds to exact agreement and agreement by round."""
    rounds = np.full((len(ds), n_envs), np.nan)
    agree = np.zeros((len(ds), K))
    for i, d in enumerate(ds):
        for s in range(n_envs):
            G = rng.standard_normal((2 * d, 2 * d + 3))
            Sigma = G @ G.T / (2 * d + 3)
            w = rng.standard_normal(2 * d)
            w *= np.sqrt(2.0 / (w @ Sigma @ w))  # Var(signal) = 2, as |w^A| = |w^B| = 1 at rho = 0
            path = agreement_path(Sigma, w, d, sigma, max(K - 1, 2 * d + 2))
            rounds[i, s] = rounds_to_agree(path, 1e-9 * max(path.disagreement[0], 1e-12))
            agree[i] += agreement_probability(path, tol)[:K] / n_envs
    return dict(rounds=rounds, agree=agree)


def learned(rhos, d, sigma, tol, K, seeds, n, T, penalty, tail) -> dict:
    """Part 3: learned linear twins in the consensus game, one random environment per seed."""
    shape = (len(rhos), seeds)
    agree_by = np.zeros((len(rhos), seeds, K))
    game = np.zeros(shape)
    agreed_excess = np.zeros(shape)
    concede = np.zeros(shape)
    for i, rho in enumerate(rhos):
        for s in range(seeds):
            env = LinearTwinEnv.from_config(env_cfg(d, rho, sigma), np.random.default_rng(1000 + s))
            rng = run_rng(1000 + s, 0)
            A, B = train_twins(env, dict(n_A=n, n_B=n, lam=1.0), rng)
            xA, xB, y = env.sample(T, rng)
            proto = ConsensusExchange(n_exchanges=K - 1, tol=tol, penalty=penalty)
            out = proto.run(A, B, xA, xB, y)
            sl = slice(int((1 - tail) * T), T)
            ar = out["agree_round"][sl]
            agree_by[i, s] = [np.mean(ar <= k) for k in range(K)]  # nan <= k is False
            game[i, s] = out["game_loss"][sl].mean()
            agreed_excess[i, s] = np.nanmean(out["agreed_sq_err"][sl]) - sigma**2
            concede[i, s] = np.nanmedian(out["concede_A"][sl])
    return dict(
        agree=agree_by.mean(1),
        agree_se=agree_by.std(1) / np.sqrt(seeds),
        game=game.mean(1),
        agreed_excess=agreed_excess.mean(1),
        concede=np.nanmean(concede, 1),
    )


def main(argv=None) -> Path:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--tol", type=float, default=0.1)
    p.add_argument("--K", type=int, default=4, help="rounds per step (opening + K - 1 exchanges)")
    p.add_argument("--sigma", type=float, default=0.5)
    p.add_argument("--penalty", type=float, default=10.0)
    p.add_argument("--envs", type=int, default=30, help="weight draws for the ideal twins")
    p.add_argument("--seeds", type=int, default=30, help="replicates for the learned twins")
    p.add_argument("--n", type=int, default=200, help="training cases per learned twin")
    p.add_argument("--T", type=int, default=1000, help="online steps for the learned twins")
    p.add_argument("--tail", type=float, default=0.5)
    p.add_argument("--results", default=str(ROOT / "results"))
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")

    rho_grid = np.round(np.arange(-0.9, 0.91, 0.05), 3)
    ideal = ideal_paired(rho_grid, 3, args.sigma, args.tol, args.K, args.envs)
    log.info("ideal paired done")
    ds = np.array([1, 2, 3, 4, 6, 8])
    general = ideal_general(ds, args.sigma, args.tol, args.K, args.envs, np.random.default_rng(7))
    log.info("ideal general done")
    rho_learn = np.array([-0.5, 0.0, 0.25, 0.5, 0.75, 0.9])
    lrn = learned(
        rho_learn,
        3,
        args.sigma,
        args.tol,
        args.K,
        args.seeds,
        args.n,
        args.T,
        args.penalty,
        args.tail,
    )
    log.info("learned done")

    run_dir = make_run_dir(args.results, "consensus_sweep")
    np.savez(
        run_dir / "sweep.npz",
        rho_grid=rho_grid,
        ds=ds,
        rho_learn=rho_learn,
        **{f"ideal_{k}": v for k, v in ideal.items()},
        **{f"general_{k}": v for k, v in general.items()},
        **{f"learned_{k}": v for k, v in lrn.items()},
    )
    summary = {
        "args": vars(args),
        "git": git_hash(ROOT),
        "general_rounds_counts": {
            int(d): {str(k): int(c) for k, c in zip(*np.unique(r, return_counts=True))}
            for d, r in zip(ds, general["rounds"])
        },
        "learned": {
            float(r): {
                "agree_by_round": lrn["agree"][i].round(3).tolist(),
                "game_loss": round(float(lrn["game"][i]), 3),
                "agreed_excess": round(float(lrn["agreed_excess"][i]), 3),
                "concede_A_median": round(float(lrn["concede"][i]), 3),
            }
            for i, r in enumerate(rho_learn)
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    fig, _ = plot_consensus_sweep(rho_grid, ideal, ds, general, rho_learn, lrn, args.tol)
    fig.savefig(run_dir / "consensus_sweep.png", dpi=150, bbox_inches="tight")
    log.info("saved to %s\n%s", run_dir, json.dumps(summary["learned"], indent=1))
    return run_dir


if __name__ == "__main__":
    main()
