"""Figures for the alignment sweep (`scripts/related_analysis.py`, LogBook 2026-11-16).

Takes the per-cell statistics the analysis computed; never runs experiments. Same style as
`plotting_consensus` (categorical slots in fixed order: not told = slot 1, told = slot 2).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from consensus.plotting_consensus import CAT, INK, MUTED, _style  # noqa: E402

ARMS = (("untold", "not told"), ("told", "told targets differ"))


def plot_related(cells: dict, cs: list[float], path: Path, sigma: float = 0.5,
                 series: tuple = ARMS) -> None:
    """Left: coefficient on the partner's opening in the twin's final (late queries), against
    the cosine c between the two targets. Right: MSE of the finals against the twin's own y.

    Points with error bars: mean ± s.e. over twins, per series (prompt arm, or model; keys of
    `cells` are "<c>_<series key>"), slightly offset. Reference
    lines (population, Sigma = I, unit weights): the best stacking coefficient c and MSE
    1 - c^2 + sigma^2; averaging (coefficient 0.5, MSE 1.5 - c + sigma^2); own opening alone
    (0, 1 + sigma^2).
    """
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    grid = np.linspace(min(cs), max(cs), 50)
    s2 = sigma**2
    refs = [
        ("best stacking", grid, 1 - grid**2 + s2, "-"),
        ("averaging", np.full_like(grid, 0.5), 1.5 - grid + s2, "--"),
        ("own opening only", np.zeros_like(grid), np.full_like(grid, 1 + s2), ":"),
    ]
    for ax, key, ylabel in ((axes[0], "beta", "coefficient on partner's opening"),
                            (axes[1], "mse_late", "MSE of final vs own y")):
        _style(ax)
        for name, beta_line, mse_line, ls in refs:
            ax.plot(grid, beta_line if key == "beta" else mse_line, ls, color=MUTED, lw=1,
                    label=name)
        for i, (arm, label) in enumerate(series):
            xs, ms, ses = [], [], []
            for c in cs:
                vals = np.array(cells.get(f"{c}_{arm}", {}).get(key, []), float)
                vals = vals[np.isfinite(vals)]
                if len(vals):
                    xs.append(c + (i - 0.5) * 0.04)
                    ms.append(vals.mean())
                    ses.append(vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)
            if xs:
                ax.errorbar(xs, ms, yerr=ses, fmt="o", color=CAT[i], ms=5, capsize=2,
                            label=label)
        ax.set_xlabel("cosine c between the targets")
        ax.set_ylabel(ylabel, color=INK)
        ax.invert_xaxis()
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
