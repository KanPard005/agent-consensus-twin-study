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


def plot_dims(cells: dict, dims: list[str], cs: list[float], path: Path) -> None:
    """Long-run variants (LogBook 2026-11-17, e.g. d = 3 vs 20, sigma = 0.5 vs 1.5): x = c, one
    colour per variant.
    Panels: messages per partner query; share of twins that built a joint model; late MSE of the
    finals (dotted: online joint learner on the partner's raw features, the finite-T value of
    the raw-feature route in that variant; dashed, same colour: population stacking error
    1 - c^2 + sigma^2).

    `cells[(variant, c)]` holds `msgs_per_q`, `built`, `final_late`, `joint_late`, `stack_ref`.
    """
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    x = np.arange(len(cs))
    for i, d in enumerate(dims):
        col = CAT[i]
        for ax, key in zip(axes, ("msgs_per_q", "built", "final_late")):
            ax.plot(x, [cells[(d, c)][key] for c in cs], color=col, marker="o", ms=5,
                    label=str(d))
        axes[2].plot(x, [cells[(d, c)]["joint_late"] for c in cs], color=col, ls=":", lw=1.2)
        axes[2].plot(x, [cells[(d, c)]["stack_ref"] for c in cs], color=col, ls="--", lw=0.8,
                     alpha=0.6)
    titles = ("messages per partner query", "twins that built a joint model",
              "late MSE (dotted: online joint learner;\ndashed: population stacking)")
    for ax, t in zip(axes, titles):
        _style(ax)
        ax.set_xticks(x, [f"{c:g}" for c in cs])
        ax.set_xlabel("target cosine c")
        ax.set_title(t, fontsize=9, color=INK)
    axes[1].set_ylim(-0.03, 1.03)
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_rig(stats: dict, arms: list[str], windows: list[str], path: Path) -> None:
    """Rigged first partner queries (LogBook note 2026-11-17): per window of partner queries
    (the first is the rigged one, shaded), one line per arm and the control (grey). Panels:
    messages sent per query, share of queries asking for the partner's features, share sending
    own features, share of twins with a joint model, MSE of the finals.

    `stats[arm][measure]` is a list over windows (means over twins).
    """
    keys = (("msgs", "messages sent per query"), ("asked", "queries asking for features"),
            ("raw", "queries sending own features"), ("built", "twins with a joint model"),
            ("mse", "MSE of finals"))
    fig, axes = plt.subplots(1, len(keys), figsize=(15.0, 3.2))
    x = np.arange(len(windows))
    for i, arm in enumerate(arms):
        col = MUTED if arm == "control" else CAT[(i - 1) % len(CAT)]
        label = ("control rerun" if arm.endswith("_rerun") else
                 arm.replace("llm_haiku_rig_", "").replace("low", "−").replace("high", "+"))
        for ax, (key, _) in zip(axes, keys):
            ax.plot(x, stats[arm][key], color=col, marker="o", ms=4, label=label,
                    lw=2.0 if arm == "control" else 1.4)
    for ax, (_, title) in zip(axes, keys):
        _style(ax)
        ax.axvspan(-0.3, 0.3, color=MUTED, alpha=0.12, lw=0)
        ax.set_xticks(x, windows, fontsize=7)
        ax.set_xlabel("partner queries (first window rigged)")
        ax.set_title(title, fontsize=9, color=INK)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
