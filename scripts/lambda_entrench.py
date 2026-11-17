"""Entrenchment of linear twins in the controlled design (LogBook note 2026-11-11, "More controlled
experiment on differences").

Usage:
    python scripts/lambda_entrench.py [--seeds 200] [--L 500] [--d 3] [--ss 5 20 100 500 2000]
        [--lambdas 0 0.1 ...] [--workers 8]

Sigma = [[I, lambda U V^T], [lambda V U^T, I]] with random orthogonal U, V and weights
w_A = U s, w_B = V (c s), s = 1/sqrt(d) (`data.controlled_env_parts`). Three variants:

- aligned (c = 1): the population solo opinions disagree by 2 (1 - lambda^2)(1 + lambda), which
  peaks at lambda = 1/3;
- normalized (c = 1, weights scaled so that Var(signal) = 1): disagreement 1 - lambda^2;
- opposed (c = -1): disagreement 2 (1 - lambda^2)(1 - lambda), monotone.

Per (variant, lambda, seed) the twins learn alone for s steps and then talk (P2) for L steps, as in
`onset_entrench.py` (same metrics: weight on the partner as a share of the best response's,
movement toward the best response, team excess; two priors for the talking head). The seed fixes
U, V and the underlying Gaussian draws, so lambdas and variants are paired.

The question: does entrenchment (1 - movement) follow the disagreement, with a peak near 1/3, or
the distance the twin has to move to reach the best response?

Saves results/lambda_entrench/<timestamp>/ (lambda_entrench.npz, summary.json, lambda_entrench.png).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from onset_entrench import PRIORS, embed, onset_arms  # noqa: E402

from consensus.data import LinearTwinEnv  # noqa: E402
from consensus.plotting_consensus import plot_lambda_entrench  # noqa: E402
from consensus.theory import optimal_head, solo_predictors  # noqa: E402
from consensus.utils import git_hash, make_run_dir  # noqa: E402

log = logging.getLogger("lambda_entrench")
VARIANTS = {
    "aligned": dict(align=1.0, normalize=False),
    "normalized": dict(align=1.0, normalize=True),
    "opposed": dict(align=-1.0, normalize=False),
}


def env_for(variant: str, lam: float, d: int, sigma: float, seed: int) -> LinearTwinEnv:
    cfg = dict(d_A=d, d_B=d, cov="controlled", lambda_xz=lam, sigma=sigma, **VARIANTS[variant])
    return LinearTwinEnv.from_config(cfg, np.random.default_rng(20_000 + seed))


def population_curves(variant: str, lambdas, d: int, sigma: float) -> dict:
    """Disagreement of the population solo opinions, and the squared Sigma-distance from a twin's
    solo opinion to its best response to the partner's (averaged over the two twins). Both are
    invariant to U, V, so one draw suffices."""
    dis, dist = [], []
    for lam in lambdas:
        env = env_for(variant, lam, d, sigma, 0)
        S = env.Sigma
        a, b = solo_predictors(S, env.w, d)
        dis.append((a - b) @ S @ (a - b))
        th = {"A": a[:d], "B": b[d:]}
        dd = []
        for me, other in (("A", "B"), ("B", "A")):
            v0 = embed(np.concatenate([th[me], [0.0]]), th[other], me)
            vs = embed(optimal_head(S, env.w, d, th[other], me), th[other], me)
            dd.append((vs - v0) @ S @ (vs - v0))
        dist.append(np.mean(dd))
    return {"disagreement": np.array(dis), "distance": np.array(dist)}


def one_job(job):
    variant, lam, seed, ss, taus, L, sigma, d, ridge = job
    env = env_for(variant, lam, d, sigma, seed)
    # Same stream seed for every lambda and variant: the underlying N(0, I) draws are shared.
    rng = np.random.default_rng([seed, 7])
    out, gap = onset_arms(env, rng, ss, taus, L, d, ridge, with_disagreement=True)
    return out[:, :, -1], gap  # metrics after the last talking step


def main(argv=None) -> Path:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--seeds", type=int, default=200)
    p.add_argument("--L", type=int, default=500, help="talking steps after onset")
    p.add_argument("--ss", type=int, nargs="+", default=[5, 20, 100, 500, 2000])
    p.add_argument("--lambdas", type=float, nargs="+",
                   default=[0.0, 0.1, 0.2, 1 / 3, 0.45, 0.6, 0.75, 0.9])
    p.add_argument("--variants", nargs="+", default=list(VARIANTS))
    p.add_argument("--sigma", type=float, default=0.5)
    p.add_argument("--d", type=int, default=3)
    p.add_argument("--lam", type=float, default=1.0, help="ridge strength of the solo model")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--experiment", default="lambda_entrench")
    p.add_argument("--results", default=str(ROOT / "results"))
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")

    V, Lm, ss = args.variants, args.lambdas, args.ss
    taus = [args.L]
    res = np.zeros((len(V), len(Lm), len(PRIORS), len(ss), 3, 2))
    gap = np.zeros((len(V), len(Lm), len(ss), 2))
    with ProcessPoolExecutor(args.workers) as ex:
        for v, variant in enumerate(V):
            for j, lam in enumerate(Lm):
                jobs = [
                    (variant, lam, s, ss, taus, args.L, args.sigma, args.d, args.lam)
                    for s in range(args.seeds)
                ]
                outs = list(ex.map(one_job, jobs, chunksize=4))
                R = np.stack([o for o, _ in outs])
                G = np.stack([g for _, g in outs])
                res[v, j, ..., 0] = R.mean(0)
                res[v, j, ..., 1] = R.std(0) / np.sqrt(args.seeds)
                # message-weight share is heavy-tailed when c* ~ 0 (see onset_entrench.py)
                res[v, j, ..., 0, 0] = np.median(R[..., 0], 0)
                res[v, j, ..., 0, 1] = 0.0
                gap[v, j, :, 0] = G.mean(0)
                gap[v, j, :, 1] = G.std(0) / np.sqrt(args.seeds)
                log.info("%s lambda %.3f done", variant, lam)

    theory = {variant: population_curves(variant, Lm, args.d, args.sigma) for variant in V}
    run_dir = make_run_dir(args.results, args.experiment)
    np.savez(run_dir / "lambda_entrench.npz", res=res, gap=gap, lambdas=Lm, ss=ss,
             variants=V, **{f"{k}_{n}": c for k, t in theory.items() for n, c in t.items()})
    metrics = ["msg_weight_share", "movement_share", "team_excess"]
    summary = {
        "args": vars(args),
        "git": git_hash(ROOT),
        "lambdas": [round(x, 3) for x in Lm],
        "theory": {k: {n: np.round(c, 3).tolist() for n, c in t.items()} for k, t in theory.items()},
        "results": {
            variant: {
                prior: {
                    f"s={s}": {
                        **{m: np.round(res[v, :, i, q, mi, 0], 3).tolist()
                           for mi, m in enumerate(metrics)},
                        "movement_se": np.round(res[v, :, i, q, 1, 1], 3).tolist(),
                        "disagreement": np.round(gap[v, :, q, 0], 3).tolist(),
                    }
                    for q, s in enumerate(ss)
                }
                for i, prior in enumerate(PRIORS)
            }
            for v, variant in enumerate(V)
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    fig, _ = plot_lambda_entrench(np.array(Lm), V, ss, res, gap, theory, PRIORS)
    fig.savefig(run_dir / "lambda_entrench.png", dpi=150, bbox_inches="tight")
    log.info("saved to %s", run_dir)
    return run_dir


if __name__ == "__main__":
    main()
