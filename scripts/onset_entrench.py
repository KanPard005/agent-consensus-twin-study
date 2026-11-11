"""Are linear twins that start talking later more entrenched?

Usage:
    python scripts/onset_entrench.py [--seeds 200] [--L 500] [--d 3] [--ss 5 20 100 500 2000]
        [--experiment onset_entrench] [--workers 8]

Each twin first learns alone on the first s cases of a common stream (nested across s, so arms
are paired), then talks for L steps (P2: one exchange; its opinion, the solo model, is frozen at
onset) on a second stream shared by every s. Talking means learning head 1, a ridge regression of
y on (own x, partner's opinion m), warm-started at (theta_0, 0) with prior covariance
blockdiag(prior_scale * P_0, I) (`LinearAgent.begin_communication`).

Entrenchment is measured against the best response to the partner's actual frozen opinion
(`theory.optimal_head`, coefficients (b*, c*) on (own x, m)), at several steps tau after onset:

- message weight c / c*: the share of the optimal weight on the partner's opinion (median over
  seeds, as c* can be near 0 for small s);
- movement: how far the twin's talking predictor v moved from its solo opinion v_0 toward the
  best response v*, as a share of the way, (v - v_0)' Sigma (v* - v_0) / |v* - v_0|_Sigma^2
  (0 = kept its own opinion, 1 = full best response; like the LLMs' weight of advice);
- team excess MSE of the average of the two heads.

Two priors for the talking head:
- "training posterior" (prior_scale = 1): P_0 ~ I / (lam + s), so a twin that learned alone
  longer is more certain of its own weights;
- "fixed strength" (prior_scale = lam + s): P ~ I whatever s, which removes that effect. If the
  later twins are more entrenched only under the first prior, the cause is their certainty.

Saves results/onset_entrench/<timestamp>/ (onset_entrench.npz, summary.json, onset_entrench.png).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consensus.data import LinearTwinEnv  # noqa: E402
from consensus.models import LinearAgent  # noqa: E402
from consensus.plotting_consensus import plot_onset_entrench  # noqa: E402
from consensus.theory import optimal_head  # noqa: E402
from consensus.utils import git_hash, make_run_dir  # noqa: E402

log = logging.getLogger("onset_entrench")
PRIORS = ["training posterior", "fixed strength"]


def embed(coef: np.ndarray, theta_partner: np.ndarray, agent: str) -> np.ndarray:
    """x-space vector of the head prediction b'x_own + c m_partner."""
    b, c = coef[:-1], coef[-1]
    return np.concatenate([b, c * theta_partner] if agent == "A" else [c * theta_partner, b])


def one_seed(job):
    """All (prior, s) arms for one seed; returns arrays (priors, s, taus, 3 metrics)."""
    rho, seed, ss, taus, L, sigma, d, lam = job
    cfg = dict(d_A=d, d_B=d, rho=rho, norm_A=1.0, norm_B=1.0, sigma=sigma)
    env = LinearTwinEnv.from_config(cfg, np.random.default_rng(20_000 + seed))
    rng = np.random.default_rng([seed, int(round(100 * rho)) + 100])
    xa_tr, xb_tr, y_tr = env.sample(max(ss), rng)
    xa, xb, y = env.sample(L, rng)
    S, w = env.Sigma, env.w
    out = np.full((len(PRIORS), len(ss), len(taus), 3), np.nan)
    for (i, prior), (j, s) in product(enumerate(PRIORS), enumerate(ss)):
        A, B = LinearAgent(d, lam, "A"), LinearAgent(d, lam, "B")
        for t in range(s):
            A.update(xa_tr[t], [], y_tr[t])
            B.update(xb_tr[t], [], y_tr[t])
        ps = 1.0 if prior == PRIORS[0] else lam + s
        for ag in (A, B):
            ag.begin_communication(1, prior_scale=ps, msg_prior_var=1.0)
        th = {"A": A.heads[0].theta.copy(), "B": B.heads[0].theta.copy()}
        opt = {
            "A": optimal_head(S, w, d, th["B"], "A"),
            "B": optimal_head(S, w, d, th["A"], "B"),
        }
        k = 0
        for t in range(L):
            mA, mB = A.predict(xa[t], []), B.predict(xb[t], [])
            A.update(xa[t], [mB], y[t])
            B.update(xb[t], [mA], y[t])
            if t + 1 == taus[k]:
                ratios, moves, vs = [], [], []
                for name, ag, other in (("A", A, "B"), ("B", B, "A")):
                    coef = ag.heads[1].theta
                    v = embed(coef, th[other], name)
                    v0 = embed(np.concatenate([th[name], [0.0]]), th[other], name)
                    vs_ = embed(opt[name], th[other], name)
                    dv = vs_ - v0
                    ratios.append(coef[-1] / opt[name][-1])
                    moves.append((v - v0) @ S @ dv / (dv @ S @ dv))
                    vs.append(v)
                team = (vs[0] + vs[1]) / 2 - w
                out[i, j, k] = [np.mean(ratios), np.mean(moves), team @ S @ team]
                k += 1
                if k == len(taus):
                    break
    return out


def main(argv=None) -> Path:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--seeds", type=int, default=200)
    p.add_argument("--L", type=int, default=500, help="talking steps after onset")
    p.add_argument("--ss", type=int, nargs="+", default=[5, 20, 100, 500, 2000], help="steps alone")
    p.add_argument("--experiment", default="onset_entrench", help="results subfolder")
    p.add_argument("--sigma", type=float, default=0.5)
    p.add_argument("--d", type=int, default=3)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--results", default=str(ROOT / "results"))
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")

    rhos = [0.0, 0.5]
    ss = args.ss
    taus = sorted({t for t in (1, 3, 10, 30, 100, 300, 500, 1000, 3000) if t < args.L} | {args.L})
    res = np.zeros((len(rhos), len(PRIORS), len(ss), len(taus), 3, 2))  # mean, s.e.
    with ProcessPoolExecutor(args.workers) as ex:
        for j, rho in enumerate(rhos):
            jobs = [
                (rho, s, ss, taus, args.L, args.sigma, args.d, args.lam) for s in range(args.seeds)
            ]
            R = np.stack(list(ex.map(one_seed, jobs, chunksize=4)))
            res[j, ..., 0] = R.mean(0)
            res[j, ..., 1] = R.std(0) / np.sqrt(args.seeds)
            # c* can be near 0 when a twin learned alone for only a few steps (|c*| < 0.3 in
            # 8% of draws at s = 5), which makes c / c* heavy-tailed: report its median, and
            # no s.e. band (set to 0).
            res[j, ..., 0, 0] = np.median(R[..., 0], 0)
            res[j, ..., 0, 1] = 0.0
            log.info("rho %.1f done", rho)

    run_dir = make_run_dir(args.results, args.experiment)
    np.savez(run_dir / "onset_entrench.npz", res=res, rhos=rhos, ss=ss, taus=taus)
    metrics = ["msg_weight_share", "movement_share", "team_excess"]
    summary = {
        "args": vars(args),
        "git": git_hash(ROOT),
        "results": {
            f"rho={rho:g}": {
                prior: {
                    f"s={s}": {
                        m: [round(float(x), 3) for x in res[j, i, q, :, mi, 0]]
                        for mi, m in enumerate(metrics)
                    }
                    for q, s in enumerate(ss)
                }
                for i, prior in enumerate(PRIORS)
            }
            for j, rho in enumerate(rhos)
        },
        "taus": taus,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    fig, _ = plot_onset_entrench(res, rhos, PRIORS, ss, taus)
    fig.savefig(run_dir / "onset_entrench.png", dpi=150, bbox_inches="tight")
    log.info("saved to %s", run_dir)
    return run_dir


if __name__ == "__main__":
    main()
