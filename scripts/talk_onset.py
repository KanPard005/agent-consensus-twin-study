"""Talk onset: with a fixed budget of n_T labelled steps, how long should twins learn alone?

Usage:
    python scripts/talk_onset.py [--seeds 200] [--n-test 4000] [--workers 8]

For every rho, budget n_T, mode (exchange / blend), `update_solo` and onset share s / n_T, runs
`consensus.talk_onset.TalkOnset` over `--seeds` replicates (a new random environment and stream
per seed, shared by every arm of that seed, so arms are paired) and scores the team on a held-out
test set. Saves results/talk_onset/<timestamp>/ (onset.npz, summary.json, talk_onset.png,
learning_curves.png).
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
from consensus.plotting_consensus import CAT, MUTED, _style, plot_talk_onset  # noqa: E402
from consensus.talk_onset import TalkOnset  # noqa: E402
from consensus.theory import protocol_benchmarks  # noqa: E402
from consensus.utils import git_hash, make_run_dir  # noqa: E402

log = logging.getLogger("talk_onset")

MODES = [("exchange", True), ("exchange", False), ("blend", False)]
MODE_NAMES = ["P2, solo keeps learning", "P2, solo frozen at onset", "P1 blend, solo frozen"]


def one_seed(job):
    """All onset shares for one (rho, n_T, seed): same environment, stream and test set."""
    rho, n_T, seed, fracs, n_test, sigma, d, n_cp = job
    cfg = dict(d_A=d, d_B=d, rho=rho, norm_A=1.0, norm_B=1.0, sigma=sigma)
    env = LinearTwinEnv.from_config(cfg, np.random.default_rng(10_000 + seed))
    rng = np.random.default_rng([seed, n_T, int(round(100 * rho)) + 100])
    xa, xb, y = env.sample(n_T, rng)
    XA, XB, _ = env.sample(n_test, rng)
    f = XA @ env.w_A + XB @ env.w_B
    cps = np.unique(np.linspace(1, n_T, n_cp).round().astype(int))
    final = np.zeros((len(MODES), len(fracs)))
    curves = np.zeros((len(MODES), len(fracs), len(cps)))
    for (i, (mode, upd)), (j, fr) in product(enumerate(MODES), enumerate(fracs)):
        res = TalkOnset(env, n_T, int(round(fr * n_T)), mode, upd).run(xa, xb, y, (XA, XB, f), cps)
        final[i, j] = res.excess[-1]
        curves[i, j] = res.excess
    return final, curves, cps


def main(argv=None) -> Path:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--seeds", type=int, default=200)
    p.add_argument("--n-test", type=int, default=4000)
    p.add_argument("--sigma", type=float, default=0.5)
    p.add_argument("--d", type=int, default=3)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--results", default=str(ROOT / "results"))
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")

    rhos = [0.0, 0.5, 0.9]
    n_Ts = [20, 50, 200, 800]
    fracs = np.array([0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0])
    n_cp = 25
    final = np.zeros((len(MODES), len(rhos), len(n_Ts), len(fracs), 2))
    curves = {}
    with ProcessPoolExecutor(args.workers) as ex:
        for (j, rho), (k, n_T) in product(enumerate(rhos), enumerate(n_Ts)):
            jobs = [
                (rho, n_T, s, fracs, args.n_test, args.sigma, args.d, n_cp)
                for s in range(args.seeds)
            ]
            res = list(ex.map(one_seed, jobs, chunksize=4))
            F = np.stack([r[0] for r in res])  # (seeds, modes, fracs)
            final[:, j, k, :, 0] = F.mean(0)
            final[:, j, k, :, 1] = F.std(0) / np.sqrt(args.seeds)
            curves[(rho, n_T)] = (res[0][2], np.stack([r[1] for r in res]).mean(0))
            log.info("rho %.1f n_T %d done", rho, n_T)

    # Population limits (averaged over the same random weight directions) of the talking
    # protocol; the solo models are fully learned there, so only "solo keeps learning" with a
    # large n_T can reach them.
    ref = np.zeros((len(MODES), len(rhos)))
    for j, rho in enumerate(rhos):
        vals = []
        for s in range(args.seeds):
            cfg = dict(d_A=args.d, d_B=args.d, rho=rho, norm_A=1.0, norm_B=1.0, sigma=args.sigma)
            env = LinearTwinEnv.from_config(cfg, np.random.default_rng(10_000 + s))
            b = protocol_benchmarks(env.Sigma, env.w, args.d, args.sigma, 1)
            vals.append([b["exchange_1"] - b["full_info"], b["blend"] - b["full_info"]])
        v = np.mean(vals, 0)
        ref[:, j] = [v[0], v[0], v[1]]

    run_dir = make_run_dir(args.results, "talk_onset")
    np.savez(run_dir / "onset.npz", final=final, fracs=fracs, n_Ts=n_Ts, rhos=rhos, ref=ref)
    best = {
        MODE_NAMES[i]: {
            f"rho={rho:g}": {
                f"n_T={n_T}": {
                    "best_frac": float(fracs[np.argmin(final[i, j, k, :, 0])]),
                    "excess_talk_from_start": round(float(final[i, j, k, 0, 0]), 4),
                    "excess_best": round(float(final[i, j, k, :, 0].min()), 4),
                    "excess_never_talk": round(float(final[i, j, k, -1, 0]), 4),
                }
                for k, n_T in enumerate(n_Ts)
            }
            for j, rho in enumerate(rhos)
        }
        for i in range(len(MODES))
    }
    (run_dir / "summary.json").write_text(
        json.dumps({"args": vars(args), "git": git_hash(ROOT), "best": best}, indent=1)
    )
    fig, _ = plot_talk_onset(fracs, final, n_Ts, rhos, MODE_NAMES, ref)
    fig.savefig(run_dir / "talk_onset.png", dpi=150, bbox_inches="tight")

    # Learning curves for one budget: team test excess vs step, one line per onset share.
    import matplotlib.pyplot as plt

    n_T = 200
    figc, axes = plt.subplots(len(MODES), len(rhos), figsize=(12.6, 8.4), squeeze=False)
    for (i, name), (j, rho) in product(enumerate(MODE_NAMES), enumerate(rhos)):
        cps, C = curves[(rho, n_T)]
        ax = axes[i, j]
        for q, fr in enumerate([0.0, 0.2, 0.5, 0.9]):
            jj = int(np.flatnonzero(np.isclose(fracs, fr))[0])
            ax.plot(cps, C[i, jj], color=CAT[q], lw=2, label=f"talk from step {int(fr * n_T)}")
        if ref[i, j] > 1e-6:  # population limit (zero at rho = 0 for P2)
            ax.axhline(ref[i, j], color=MUTED, ls="--", lw=1)
        ax.set_yscale("log")
        ax.set_title(f"{name}, ρ = {rho:g}", fontsize=9)
        ax.set(
            xlabel="step" if i == len(MODES) - 1 else None,
            ylabel="team test excess MSE" if j == 0 else None,
        )
        _style(ax)
    axes[0, -1].legend(fontsize=7, frameon=False)
    figc.tight_layout()
    figc.savefig(run_dir / "learning_curves.png", dpi=150, bbox_inches="tight")
    log.info("saved to %s", run_dir)
    print(json.dumps(best, indent=1))
    return run_dir


if __name__ == "__main__":
    main()
