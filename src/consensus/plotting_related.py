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


def plot_budget(cells: dict, ns: list[int], cs: list[float], path: Path,
                xlabel: str = "training cases per twin", log_x: bool = True) -> None:
    """Training budget sweep on related targets (LogBook 2026-11-16).

    `cells[(c, n)]` holds per-twin lists `beta`, `mse`, `stack` (MSE of the online stacking
    learner on the same queries) and per-pair `msg` (messages per query, both twins). Panels
    against the number of training cases n (log scale), one colour per c:
    left, the coefficient on the partner's opening (mean ± s.e. over twins; the dotted line at
    height c is the best stacking weight); middle, the share of talkative pairs (at least 0.5
    messages per query); right, the late MSE of the finals (solid) and of the stacking
    learner on the same queries (dotted), which is the per-sample reference.
    """
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.3))
    x = np.array(ns, float)
    for i, c in enumerate(cs):
        col = CAT[i]
        # Small horizontal offsets so the error bars of the c series do not overlap.
        off = x * (1 + 0.06 * (i - 1)) if log_x else x + 0.012 * (x.max() - x.min()) * (i - 1)
        get = lambda key: [np.asarray(cells[(c, n)][key], float) for n in ns]  # noqa: E731
        m = np.array([np.nanmean(v) for v in get("beta")])
        se = np.array([np.nanstd(v) / np.sqrt(np.isfinite(v).sum()) for v in get("beta")])
        axes[0].errorbar(off, m, se, color=col, marker="o", ms=4, capsize=2, label=f"c = {c:g}")
        axes[0].axhline(c, color=col, ls=":", lw=1)
        axes[1].plot(x, [np.mean(v >= 0.5) for v in get("msg")], color=col, marker="o", ms=4)
        axes[2].plot(off, [np.nanmean(v) for v in get("mse")], color=col, marker="o", ms=4)
        axes[2].plot(off, [np.nanmean(v) for v in get("stack")], color=col, ls=":", lw=1)
    titles = ("weight on partner's opening", "talkative pairs (≥ 0.5 msg / query)",
              "late MSE (dotted: stacking learner)")
    for ax, t in zip(axes, titles):
        _style(ax)
        if log_x:
            ax.set_xscale("log")
        ax.set_xticks(x, [str(n) for n in ns])
        ax.minorticks_off()
        ax.set_xlabel(xlabel)
        ax.set_title(t, fontsize=9, color=INK)
    axes[1].set_ylim(-0.05, 1.05)
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_over_time(cells: dict, cs: list[float], edges: list[int], path: Path) -> None:
    """Long runs (LogBook 2026-11-17): per window of partner queries (x: window midpoint), one
    colour per c. Panels: share of twins with a joint model on both feature sets; share fitting it
    in at least half the queries of the window; share of queries in which the twin sends raw
    features; mean MSE of the finals (dotted: online stacking learner on the same queries; grey
    dashed: noise floor sigma^2 = 0.25, what a joint model reaches).

    `cells[c]` holds per-window lists: `built`, `fitting`, `raw`, `mse`, `stack`.
    """
    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.2))
    mid = [(a + b) / 2 for a, b in zip(edges, edges[1:])]
    for i, c in enumerate(cs):
        col, s = CAT[i], cells[c]
        for ax, key in zip(axes, ("built", "fitting", "raw", "mse")):
            ax.plot(mid, s[key], color=col, marker="o", ms=4, label=f"c = {c:g}")
        axes[3].plot(mid, s["stack"], color=col, ls=":", lw=1)
    axes[3].axhline(0.25, color=MUTED, ls="--", lw=1)
    titles = ("joint model built", "joint model fitted (≥ half the window)",
              "queries sending raw features", "MSE of finals (dotted: stacking learner)")
    for ax, t in zip(axes, titles):
        _style(ax)
        ax.set_xticks(mid, [f"{a + 1}–{b}" for a, b in zip(edges, edges[1:])], fontsize=7)
        ax.set_xlabel("partner queries")
        ax.set_title(t, fontsize=9, color=INK)
    for ax in axes[:3]:
        ax.set_ylim(-0.03, 1.03)
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
