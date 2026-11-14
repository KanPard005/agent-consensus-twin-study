"""Figure for the asymmetric-information experiment (`scripts/asym_analysis.py`).

Takes the per-cell statistics the analysis computed; never runs experiments. Same colors and
style as `plotting_consensus` (categorical slots in fixed order: strong twin = slot 1, weak = 2).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from consensus.plotting_consensus import CAT, INK, MUTED, _style  # noqa: E402

ROLES = (("strong", "better-informed twin"), ("weak", "less-informed twin"))
RULE_TITLES = {"individual": "each scored on own final", "consensus": "must agree"}


def plot_asym_woa(cells: dict, lambdas: list[str], alphas: list[str], path: Path) -> None:
    """Weight of advice of each twin against alpha, one panel per (rule, lambda).

    Solid with error bars: what the twins did (mean ± s.e. over twins). Dashed: the best convex
    weight on the partner in the population (`asymmetric_limits`), i.e. the WOA that minimises
    the twin's error among (1 - w) own + w partner. Under consensus the
    two WOAs add up to 1 by construction.
    """
    rules = [r for r in RULE_TITLES if any(k.startswith(r + "_") for k in cells)]
    fig, axes = plt.subplots(len(rules), len(lambdas), figsize=(3.6 * len(lambdas), 3.0 * len(rules)),
                             squeeze=False, sharex=True, sharey=True)
    x = np.array([float(a) for a in alphas])
    for i, rule in enumerate(rules):
        for j, lam in enumerate(lambdas):
            ax = axes[i, j]
            _style(ax)
            for c, (role, label) in enumerate(ROLES):
                keys = [f"{rule}_{lam}_{a}" for a in alphas]
                got = [k in cells for k in keys]
                xs = x[got]
                ys = np.array([cells[k][f"woa_{role}"] for k, g in zip(keys, got) if g])
                es = np.array([cells[k][f"woa_{role}_se"] for k, g in zip(keys, got) if g])
                th = np.array([cells[k][f"theory_share_{role}"] for k, g in zip(keys, got) if g])
                ax.errorbar(xs, ys, yerr=es, color=CAT[c], lw=2, marker="o", ms=6, capsize=0,
                            elinewidth=1.5, label=label)
                ax.plot(xs, th, color=CAT[c], lw=1.5, ls="--",
                        label=f"best convex ({label.split('-')[0]})")
            ax.set_ylim(-0.05, 1.05)
            ax.set_title(f"{RULE_TITLES[rule]}, λ = {lam}", fontsize=9, color=INK)
            if i == len(rules) - 1:
                ax.set_xlabel("α (weak twin's signal relative to strong)", fontsize=8, color=INK)
            if j == 0:
                ax.set_ylabel("weight on partner (WOA)", fontsize=8, color=INK)
            ax.tick_params(labelsize=7, colors=MUTED)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)
