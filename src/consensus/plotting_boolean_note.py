"""Summary figures for the boolean-task note (LogBook agent-consensus, 2026-11-19).

Inputs are plain dicts / lists computed by scripts/plot_boolean_note.py from saved results; the
functions only draw. Colours follow one fixed categorical order (pairs = slot 1, single agent =
slot 2, ...) so a series keeps its colour across figures; Bayes references are neutral grey.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Fixed categorical order (validated reference palette, light mode) and neutral inks.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK_2, GRID, BAND = "#0b0b0b", "#52514e", "#e4e3df", "#d9d8d3"


def _style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=8.5)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)


def plot_accuracy_by_condition(rows: list[dict]):
    """One row per condition: a grey band from the own-features to the all-features Bayes
    accuracy, with the pairs' final accuracy and the single agent's accuracy as dots.

    Parameters
    ----------
    rows : list of {"label", "pairs", "single", "bayes_own", "bayes_all"} (accuracies in [0, 1];
        "single" may be None), top to bottom.

    Returns
    -------
    (fig, ax)
    """
    n = len(rows)
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * n + 1.1))
    for i, r in enumerate(rows):
        yv = n - 1 - i
        ax.plot([r["bayes_own"], r["bayes_all"]], [yv, yv], color=BAND, lw=7,
                solid_capstyle="round", zorder=1)
        ax.plot([r["bayes_all"]], [yv], marker="|", ms=12, mew=2, color=INK_2, zorder=2)
        if r.get("single") is not None:
            ax.plot([r["single"]], [yv], "o", ms=8, color=SERIES[1], mec="white", mew=1.5,
                    zorder=3)
        ax.plot([r["pairs"]], [yv], "o", ms=8, color=SERIES[0], mec="white", mew=1.5, zorder=4)
    ax.set_yticks(range(n))
    ax.set_yticklabels([r["label"] for r in rows][::-1], fontsize=8.5, color=INK)
    ax.set_xlabel("accuracy on the pairs' partner-phase queries")
    ax.set_xlim(0.6, 0.92)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    handles = [
        plt.Line2D([], [], marker="o", ls="", ms=8, color=SERIES[0], label="pairs (finals)"),
        plt.Line2D([], [], marker="o", ls="", ms=8, color=SERIES[1],
                   label="single agent (all features)"),
        plt.Line2D([], [], color=BAND, lw=7, label="Bayes: one twin's features → all"),
        plt.Line2D([], [], marker="|", ls="", ms=12, mew=2, color=INK_2,
                   label="Bayes, all features"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.45, -0.13), ncol=2,
              fontsize=8, frameon=False)
    _style(ax)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    fig.tight_layout()
    return fig, ax


def plot_zero_clause(rows: list[dict]):
    """Share of finals that say 1 when the twin's own fully seen clause is 0 versus when only
    the partner's is (both mean y = 0 up to noise), one row per condition.

    Parameters
    ----------
    rows : list of {"label", "own", "partner", "n_pairs"} (shares in [0, 1]).

    Returns
    -------
    (fig, ax)
    """
    n = len(rows)
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * n + 1.1))
    for i, r in enumerate(rows):
        yv = n - 1 - i
        ax.plot([r["own"], r["partner"]], [yv, yv], color=BAND, lw=2, zorder=1)
        ax.plot([r["own"]], [yv], "o", ms=8, color=SERIES[2], mec="white", mew=1.5, zorder=2)
        ax.plot([r["partner"]], [yv], "o", ms=8, color=SERIES[0], mec="white", mew=1.5,
                zorder=3)
        ax.annotate(f"{r['partner']:.2f}", (r["partner"], yv), xytext=(7, 0),
                    textcoords="offset points", va="center", fontsize=8, color=INK_2)
    ax.set_yticks(range(n))
    ax.set_yticklabels([r["label"] for r in rows][::-1], fontsize=8.5, color=INK)
    ax.set_xlabel("share of finals saying 1 (y = 0 up to noise)")
    ax.set_xlim(-0.03, 0.7)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    handles = [
        plt.Line2D([], [], marker="o", ls="", ms=8, color=SERIES[2],
                   label="own fully seen clause is 0"),
        plt.Line2D([], [], marker="o", ls="", ms=8, color=SERIES[0],
                   label="only the partner's is 0"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.45, -0.13), ncol=2,
              fontsize=8, frameon=False)
    _style(ax)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    fig.tight_layout()
    return fig, ax


def plot_gap_over_time(curves: dict[str, tuple[list[float], list[float]]]):
    """Final accuracy minus the all-features Bayes accuracy on the same queries, by block of
    partner queries, one line per split (at most four, direct-labelled at the right end).

    Parameters
    ----------
    curves : {label: (block midpoints, gaps)}

    Returns
    -------
    (fig, ax)
    """
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.axhline(0, color=INK_2, lw=1, ls=":")
    ax.annotate("Bayes, all features", (100, 0), xytext=(0, 4), textcoords="offset points",
                ha="right", fontsize=8, color=INK_2)
    for (label, (x, g)), c in zip(curves.items(), SERIES):
        ax.plot(x, g, color=c, lw=2, marker="o", ms=5, mec="white", mew=1)
        ax.annotate(label, (x[-1], g[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK)
    ax.set_xlim(0, 128)
    ax.set_xticks([10, 30, 50, 70, 90])
    ax.set_xlabel("partner query (blocks of 20)")
    ax.set_ylabel("finals − Bayes (all features)")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    _style(ax)
    fig.tight_layout()
    return fig, ax
