"""Plotting functions. Each takes already-computed results and returns (fig, ax); none runs experiments.

`results` is a dict {protocol label: {metric: ndarray, shape (n_runs, T)}} as saved by
scripts/run.py, and `bench` the population benchmarks from `consensus.theory.protocol_benchmarks`.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

# Fixed categorical order (assigned by protocol position in the config, never cycled).
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
REF_COLOR = "#6b6a63"  # neutral ink for reference lines


def running_mean(a: np.ndarray, window: int) -> np.ndarray:
    """Bias-corrected exponential moving average along the last axis, span `window`.

    Used instead of a trailing box window, whose cliff when the large early errors leave the
    window looks like a real transition.
    """
    lam = 2.0 / (window + 1)
    out = np.empty_like(a, dtype=float)
    acc = np.zeros(a.shape[:-1])
    norm = 0.0
    for t in range(a.shape[-1]):
        acc = (1 - lam) * acc + lam * a[..., t]
        norm = (1 - lam) * norm + lam
        out[..., t] = acc / norm
    return out


def _style(ax, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#e6e5e0", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _band(ax, x, runs: np.ndarray, color: str, label: str, ls: str = "-") -> None:
    """Mean over runs with a +-1 standard-error band."""
    m = runs.mean(0)
    se = runs.std(0, ddof=1) / np.sqrt(runs.shape[0]) if runs.shape[0] > 1 else 0 * m
    ax.plot(x, m, color=color, lw=1.5, ls=ls, label=label)
    ax.fill_between(x, m - se, m + se, color=color, alpha=0.15, lw=0)


def plot_excess_mse(results: dict, bench: dict, proto_keys: dict, window: int = 200, ax=None):
    """Excess MSE of the consensus over sigma^2 vs time, with population limits (dashed).

    Parameters
    ----------
    proto_keys : dict
        {label: key in `bench`} (e.g. {"P2": "exchange_1"}) for the dashed reference lines.
    window : int
        Moving-average window over time steps.
    """
    fig, ax = plt.subplots(figsize=(8.5, 4.2)) if ax is None else (ax.figure, ax)
    s2 = bench["full_info"]
    for i, (label, res) in enumerate(results.items()):
        col = SERIES_COLORS[i % len(SERIES_COLORS)]
        runs = running_mean(res["sq_err"], window) - s2
        _band(ax, np.arange(1, runs.shape[1] + 1), runs, col, label)
        key = proto_keys.get(label)
        # A limit at full information (excess 0) has no place on a log axis; say so in the legend.
        if key in bench and bench[key] - s2 > 1e-8:
            ax.axhline(bench[key] - s2, color=col, ls="--", lw=1)
        elif key in bench:
            ax.lines[-1].set_label(f"{label} (limit: full info)")
    ax.set_yscale("log")
    _style(ax, "post-training step t", f"excess MSE over $\\sigma^2$ (window {window})")
    ax.set_title("Consensus error (dashed: population limit)", loc="left", fontsize=11)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    return fig, ax


def _frozen(res: dict) -> bool:
    """True if neither twin's prediction ever moved off its solo opinion (P0, P1)."""
    return np.allclose(np.nan_to_num(res["shift_A"]), 0) and np.allclose(
        np.nan_to_num(res["shift_B"]), 0
    )


def plot_shift(results: dict, bench: dict, proto_keys: dict, window: int = 200, ax=None):
    """Opinion shift E[(final prediction - solo prediction)^2] per twin (A solid, B dashed).

    This is the "who moves" measure: it compares predictions, so it does not depend on how a
    twin's predictor is parameterized. Thin dotted lines are the population limits.
    """
    fig, ax = plt.subplots(figsize=(8.5, 4.2)) if ax is None else (ax.figure, ax)
    for i, (label, res) in enumerate(results.items()):
        if _frozen(res):
            continue
        col = SERIES_COLORS[i % len(SERIES_COLORS)]
        x = np.arange(1, res["shift_A"].shape[1] + 1)
        for agent, ls in (("A", "-"), ("B", "--")):
            _band(ax, x, running_mean(res[f"shift_{agent}"], window), col, f"{label}: {agent}", ls)
            ref = bench.get(f"shift_{agent}_{proto_keys.get(label)}")
            if ref is not None:
                ax.axhline(ref, color=col, ls=":", lw=1)
    _style(ax, "post-training step t", rf"$(\hat y_t - \hat y^{{solo}}_t)^2$ (window {window})")
    ax.set_title("Who moves? Opinion shift (dotted: population limit)", loc="left", fontsize=11)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    return fig, ax


def plot_drift(results: dict, ax=None):
    """Drift of each twin's own-feature weights from its trained weights (A solid, B dashed).

    Parameterization-dependent (see `plot_shift` for the preferred measure); kept for diagnostics.
    """
    fig, ax = plt.subplots(figsize=(8.5, 4.2)) if ax is None else (ax.figure, ax)
    for i, (label, res) in enumerate(results.items()):
        if np.allclose(np.nan_to_num(res["drift_A"]), 0) and np.allclose(
            np.nan_to_num(res["drift_B"]), 0
        ):
            continue  # frozen protocols
        col = SERIES_COLORS[i % len(SERIES_COLORS)]
        x = np.arange(1, res["drift_A"].shape[1] + 1)
        _band(ax, x, res["drift_A"], col, f"{label}: A")
        _band(ax, x, res["drift_B"], col, f"{label}: B", ls="--")
    _style(ax, "post-training step t", r"$\|\theta_t^{x} - \theta_{\mathrm{trained}}\|_2$")
    ax.set_title("Who changes their weights?", loc="left", fontsize=11)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    return fig, ax


def plot_blend(results: dict, bench: dict, label: str, ax=None):
    """Learned blend weights (alpha_t, beta_t) of protocol `label` vs (alpha*, beta*) (dashed)."""
    fig, ax = plt.subplots(figsize=(8.5, 4.2)) if ax is None else (ax.figure, ax)
    res = results[label]
    x = np.arange(1, res["alpha"].shape[1] + 1)
    for name, col in (("alpha", SERIES_COLORS[0]), ("beta", SERIES_COLORS[1])):
        _band(ax, x, res[name], col, rf"$\{name}_t$")
        ax.axhline(bench[f"blend_{name}"], color=col, ls="--", lw=1)
    ax.set_xscale("log")
    _style(ax, "post-training step t", "blend weight")
    ax.set_title(f"{label}: learned blend (dashed: population optimum)", loc="left", fontsize=11)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    return fig, ax


def plot_disagreement(results: dict, window: int = 200, ax=None):
    """Squared disagreement between the twins' final predictions vs time.

    Frozen protocols (no opinion shift for either twin: P0, P1) all show the same constant disagreement
    of the trained opinions, so only the first of them is drawn.
    """
    fig, ax = plt.subplots(figsize=(8.5, 4.2)) if ax is None else (ax.figure, ax)
    frozen_seen = False
    for i, (label, res) in enumerate(results.items()):
        frozen = _frozen(res)
        if frozen and frozen_seen:
            continue
        frozen_seen |= frozen
        col = SERIES_COLORS[i % len(SERIES_COLORS)]
        runs = running_mean(res["disagreement"], window)
        _band(
            ax,
            np.arange(1, runs.shape[1] + 1),
            runs,
            col,
            label + (" (frozen opinions)" if frozen else ""),
        )
    ax.set_yscale("log")
    _style(ax, "post-training step t", r"$(\hat y^A_t - \hat y^B_t)^2$ (moving avg)")
    ax.set_title("Disagreement between final predictions", loc="left", fontsize=11)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    return fig, ax


# --- Comparisons across runs (scripts/compare_runs.py) -------------------------------------
REF_STYLES = {"P0": ("--", "P0 no talk"), "P2": (":", "P2 linear exchange")}


def plot_condition_curves(
    curves: dict,
    refs: dict | None = None,
    ylabel: str = "",
    title: str = "",
    log: bool = False,
    ymin: float | None = None,
    colors: dict | None = None,
    ax=None,
):
    """Mean +- s.e. over runs of per-step curves, one color per condition, gray references.

    Parameters
    ----------
    curves : dict
        {label: ndarray (n_runs, T)}, already smoothed; colored in insertion order.
    refs : dict, optional
        {label: ndarray (n_runs, T)} drawn as gray reference lines (styles in REF_STYLES).
    colors : dict, optional
        {label: color}; default: SERIES_COLORS in insertion order.
    """
    fig, ax = plt.subplots(figsize=(8.5, 4.2)) if ax is None else (ax.figure, ax)
    for label, runs in (refs or {}).items():
        ls, name = REF_STYLES.get(label, ("--", label))
        x = np.arange(1, runs.shape[1] + 1)
        ax.plot(x, runs.mean(0), color=REF_COLOR, ls=ls, lw=1.5, label=name)
    ends = []
    for i, (label, runs) in enumerate(curves.items()):
        col = (colors or {}).get(label, SERIES_COLORS[i % len(SERIES_COLORS)])
        x = np.arange(1, runs.shape[1] + 1)
        _band(ax, x, runs, col, label)
        ends.append((label.split(" ")[0], x[-1], runs.mean(0)[-1]))
    if log:
        ax.set_yscale("log")
    if ymin is not None:
        ax.set_ylim(bottom=ymin)
    _end_labels(ax, ends)
    _style(ax, "post-training step t", ylabel)
    ax.set_title(title, loc="left", fontsize=11)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.03, 1.0))
    return fig, ax


def _end_labels(ax, ends: list, min_gap: float = 11.0) -> None:
    """Direct labels at line ends, nudged apart vertically (in points) so they never overlap."""
    if not ends:
        return
    fig = ax.figure
    fig.canvas.draw()
    to_pts = 72.0 / fig.dpi
    ys = [ax.transData.transform((x, y))[1] * to_pts for _, x, y in ends]
    order = np.argsort(ys)
    placed = np.array(ys, float)
    for j in range(1, len(order)):  # push each label up until it clears the one below
        a, b = order[j - 1], order[j]
        placed[b] = max(placed[b], placed[a] + min_gap)
    for (name, x, y), y0, y1 in zip(ends, ys, placed):
        ax.annotate(
            name,
            (x, y),
            xytext=(4, y1 - y0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color="#52514e",
        )


def plot_condition_summary(values: dict, colors: dict, xlabel: str = "", title: str = "", ax=None):
    """One row per condition: mean as a thin bar, each run as a dot.

    Parameters
    ----------
    values : dict
        {label: ndarray (n_runs,)}, drawn top to bottom in insertion order.
    colors : dict
        {label: color}; conditions without an entry are drawn in the reference gray.
    """
    labels = list(values)
    fig, ax = (
        plt.subplots(figsize=(8.5, 0.45 * len(labels) + 1.2)) if ax is None else (ax.figure, ax)
    )
    for i, label in enumerate(labels):
        v = np.asarray(values[label])
        col = colors.get(label, REF_COLOR)
        y = len(labels) - 1 - i
        ax.barh(y, v.mean(), height=0.5, color=col, alpha=0.35, lw=0)
        ax.plot([v.mean()] * 2, [y - 0.25, y + 0.25], color=col, lw=2)
        ax.scatter(v, np.full(len(v), y), s=28, color=col, edgecolor="white", lw=1, zorder=3)
        ax.annotate(
            f"mean {v.mean():.2f}",
            (max(v.max(), v.mean()), y),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color="#52514e",
        )
    ax.set_yticks(range(len(labels))[::-1], labels)
    ax.axvline(0, color="#b5b3ad", lw=0.8)
    _style(ax, xlabel, "")
    ax.grid(axis="y", visible=False)
    ax.set_title(title, loc="left", fontsize=11)
    return fig, ax


def plot_sweep(
    grid: np.ndarray,
    limits: dict[str, np.ndarray],
    points: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    xlabel: str,
    title: str = "",
    ax=None,
):
    """Excess MSE vs a swept parameter: population limits (lines) and measured runs (points).

    Parameters
    ----------
    grid : ndarray, shape (G,)
        Parameter values for the population curves.
    limits : dict
        {label: excess MSE at each grid value}; drawn as thin lines in the label's color.
    points : dict
        {label: (x, mean, standard error)} of measured excess MSE; drawn as markers with error bars.
    """
    fig, ax = plt.subplots(figsize=(8.5, 4.2)) if ax is None else (ax.figure, ax)
    labels = list(dict.fromkeys([*limits, *points]))
    for i, label in enumerate(labels):
        col = SERIES_COLORS[i % len(SERIES_COLORS)]
        if label in limits:
            ax.plot(grid, limits[label], color=col, lw=1, ls="--")
        if label in points:
            x, m, se = points[label]
            ax.errorbar(x, m, yerr=se, color=col, marker="o", ms=4, lw=1.5, capsize=2, label=label)
        else:
            ax.plot([], [], color=col, lw=1, ls="--", label=f"{label} (limit only)")
    ax.axhline(0, color=REF_COLOR, lw=0.8)
    _style(ax, xlabel, r"excess MSE over $\sigma^2$")
    ax.set_title(title or "Measured (points) vs population limit (dashed)", loc="left", fontsize=11)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    return fig, ax
