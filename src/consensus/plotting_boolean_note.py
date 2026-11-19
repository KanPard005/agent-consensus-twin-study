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


# ---- Opus vs Haiku (one pair each) --------------------------------------------------------
# Colour stays the role (pairs = slot 1, single agent = slot 2, as in the figures above); the
# model is the marker (Haiku circle, Opus diamond), so identity never rests on colour alone.
MARKER = {"Haiku": "o", "Opus": "D"}


def plot_models_by_condition(rows: list[dict], series: list[tuple[str, str, str, str]]):
    """Accuracy per condition for several (model, role) series over the one-twin → all-features
    Bayes band, like `plot_accuracy_by_condition`.

    Parameters
    ----------
    rows : list of dicts with "label", "bayes_own", "bayes_all" and one key per series
        (value None when that run does not exist), top to bottom.
    series : list of (key, legend label, model, role) with model in {"Haiku", "Opus"} and
        role in {"pair", "single"}.

    Returns
    -------
    (fig, ax)
    """
    n = len(rows)
    fig, ax = plt.subplots(figsize=(7.2, 0.5 * n + 1.3))
    for i, r in enumerate(rows):
        yv = n - 1 - i
        ax.plot([r["bayes_own"], r["bayes_all"]], [yv, yv], color=BAND, lw=7,
                solid_capstyle="round", zorder=1)
        ax.plot([r["bayes_all"]], [yv], marker="|", ms=12, mew=2, color=INK_2, zorder=2)
        for j, (key, _, model, role) in enumerate(series):
            if r.get(key) is None:
                continue
            ax.plot([r[key]], [yv], MARKER[model], ms=8 if model == "Haiku" else 7,
                    color=SERIES[0] if role == "pair" else SERIES[1], mec="white", mew=1.5,
                    zorder=3 + j)
    ax.set_yticks(range(n))
    ax.set_yticklabels([r["label"] for r in rows][::-1], fontsize=8.5, color=INK)
    ax.set_xlabel("accuracy on partner queries 101–200 (pair 0's cases)")
    ax.set_xlim(0.58, 1.0)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    handles = [plt.Line2D([], [], marker=MARKER[m], ls="", ms=8,
                          color=SERIES[0] if role == "pair" else SERIES[1], label=lab)
               for _, lab, m, role in series]
    handles += [plt.Line2D([], [], color=BAND, lw=7, label="Bayes: one twin's features → all"),
                plt.Line2D([], [], marker="|", ls="", ms=12, mew=2, color=INK_2,
                           label="Bayes, all features")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.45, -0.12), ncol=3,
              fontsize=8, frameon=False)
    _style(ax)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    fig.tight_layout()
    return fig, ax


def plot_partner_zero_models(rows: list[dict]):
    """Block split, pair 0's cases: share of finals saying 1 when only the partner's fully seen
    clause is 0, for the Haiku and the Opus pair at each rho; the grey tick is the share at the
    opening (before any talk), which both models share.

    Parameters
    ----------
    rows : list of {"label", "Haiku", "Opus", "opening"} (shares in [0, 1]).

    Returns
    -------
    (fig, ax)
    """
    n = len(rows)
    fig, ax = plt.subplots(figsize=(7.2, 0.5 * n + 1.5))
    for i, r in enumerate(rows):
        yv = n - 1 - i
        ax.plot([r["opening"]], [yv], marker="|", ms=12, mew=2, color=INK_2, zorder=1)
        for model in ("Haiku", "Opus"):
            ax.plot([r[model]], [yv], MARKER[model], ms=8 if model == "Haiku" else 7,
                    color=SERIES[0], mec="white", mew=1.5, zorder=2)
            ax.annotate(f"{r[model]:.2f}", (r[model], yv), xytext=(0, 9),
                        textcoords="offset points", ha="center", fontsize=7.5, color=INK_2)
    ax.set_yticks(range(n))
    ax.set_yticklabels([r["label"] for r in rows][::-1], fontsize=8.5, color=INK)
    ax.set_xlabel("share of finals saying 1 when only the partner's clause is 0")
    ax.set_xlim(-0.04, 0.8)
    ax.set_ylim(-0.6, n - 0.3)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    handles = [plt.Line2D([], [], marker=MARKER[m], ls="", ms=8, color=SERIES[0],
                          label=f"{m} pair (final)") for m in ("Haiku", "Opus")]
    handles.append(plt.Line2D([], [], marker="|", ls="", ms=12, mew=2, color=INK_2,
                              label="before talk (opening)"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.45, -0.3), ncol=3,
              fontsize=8, frameon=False)
    _style(ax)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    fig.tight_layout()
    return fig, ax


def plot_rule_found(curves: dict[str, dict], bayes: float, either_sure: float):
    """Opus pairs, interleaved split at rho = 0.95: rolling accuracy of the finals over the
    partner queries, with a mark where the pair first states the exact CNF.

    Parameters
    ----------
    curves : {label: {"x": query numbers, "acc": rolling accuracy, "found": query or None}}
    bayes, either_sure : accuracy of the all-features Bayes rule and of "say 1 if either twin's
        features are all 1" on the same queries (both pairs pooled).

    Returns
    -------
    (fig, ax)
    """
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for val, lab, ls in ((bayes, "Bayes, all features", ":"),
                         (either_sure, "'either sure' rule", "--")):
        ax.axhline(val, color=INK_2, lw=1, ls=ls)
        ax.annotate(lab, (101, val), xytext=(0, 3), textcoords="offset points", fontsize=8,
                    color=INK_2)
    for (label, c), ls in zip(curves.items(), ("-", (0, (4, 2)))):
        ax.plot(c["x"], c["acc"], color=SERIES[0], lw=2, ls=ls)
        ax.annotate(label, (c["x"][-1], c["acc"][-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8, color=INK)
        if c["found"] is not None:
            ax.axvline(c["found"], color=SERIES[0], lw=1, ls=ls, alpha=0.6)
            ax.annotate(f"states the CNF\n(query {c['found']})", (c["found"], 0.62),
                        xytext=(-4, 0), textcoords="offset points", ha="right", fontsize=7.5,
                        color=INK_2)
    ax.set_xlim(100, 222)
    ax.set_ylim(0.6, 1.02)
    ax.set_xlabel("query (partner phase is 101–200)")
    ax.set_ylabel("finals, rolling accuracy (20 queries)")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    _style(ax)
    fig.tight_layout()
    return fig, ax


# ---- calibration -------------------------------------------------------------------------
def _reliability_axes(ax) -> None:
    ax.plot([0, 1], [0, 1], ls=":", c=INK_2, lw=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("stated P(y = 1) (bins of 0.1)")
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    _style(ax)


def plot_reliability_panels(panels: list[dict]):
    """Reliability diagrams, one panel per condition, for the Haiku pairs' openings (grey,
    dashed), the pairs' finals (slot 1) and the single agents (slot 2).

    Parameters
    ----------
    panels : list of {"label", "openings", "finals", "single"}, each a reliability dict
        {"x", "y"} from scripts/plot_boolean_note.py.

    Returns
    -------
    (fig, axes)
    """
    fig, axes = plt.subplots(1, len(panels), figsize=(3.7 * len(panels), 3.9), sharey=True)
    series = [("openings", "Haiku pairs, openings", INK_2, "--"),
              ("finals", "Haiku pairs, finals", SERIES[0], "-"),
              ("single", "Haiku single agent", SERIES[1], "-")]
    for ax, pnl in zip(axes, panels):
        _reliability_axes(ax)
        for key, lab, c, ls in series:
            ax.plot(pnl[key]["x"], pnl[key]["y"], marker="o", ms=5, c=c, ls=ls, lw=1.6,
                    mec="white", mew=0.8, label=lab)
        ax.set_title(pnl["label"], fontsize=10, color=INK)
    axes[0].set_ylabel("observed share with y = 1")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig, axes


def plot_opus_confidence(d: dict):
    """Left: mean confidence max(p, 1 - p) of the Opus single agent (slot 2) and the Opus pair's
    finals (slot 1) per condition, with the Bayes confidence as a grey tick. Right: reliability
    of both, pooled over the conditions, with the share of predictions per bin as bars.

    Parameters
    ----------
    d : {"rows": [{"label", "single", "pair", "bayes"}], "single": reliability dict,
        "pair": reliability dict}

    Returns
    -------
    (fig, axes)
    """
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw={"width_ratios": [1.25, 1]})
    rows, n = d["rows"], len(d["rows"])
    for i, r in enumerate(rows):
        yv = n - 1 - i
        a1.plot([min(r["single"], r["pair"]), max(r["single"], r["pair"])], [yv, yv], color=BAND,
                lw=3, zorder=1)
        a1.plot([r["bayes"]], [yv], marker="|", ms=13, mew=2, color=INK_2, zorder=2)
        a1.plot([r["single"]], [yv], "D", ms=7, color=SERIES[1], mec="white", mew=1.2, zorder=3)
        a1.plot([r["pair"]], [yv], "D", ms=7, color=SERIES[0], mec="white", mew=1.2, zorder=4)
    a1.set_yticks(range(n))
    a1.set_yticklabels([r["label"] for r in rows][::-1], fontsize=8.5, color=INK)
    a1.set_xlabel("mean confidence, max(p, 1 − p), queries 101–200")
    a1.set_xlim(0.6, 1.0)
    a1.grid(axis="x", color=GRID, lw=0.8)
    a1.set_axisbelow(True)
    _style(a1)
    a1.tick_params(axis="y", length=0)
    a1.spines["left"].set_visible(False)
    a1.set_title("Confidence by condition (one pair, one single agent each)", fontsize=10,
                 color=INK)
    _reliability_axes(a2)
    centers = [0.05 + 0.1 * b for b in range(10)]
    for key, c, dx in (("single", SERIES[1], -0.02), ("pair", SERIES[0], 0.02)):
        a2.plot(d[key]["x"], d[key]["y"], "-D", c=c, ms=6, lw=1.8, mec="white", mew=1)
        a2.bar([x + dx for x in centers], [0.4 * s for s in d[key]["share"]], width=0.035,
               color=c, alpha=0.35, zorder=0)
    a2.set_ylabel("observed share with y = 1")
    a2.set_title("Reliability, all conditions pooled\n(bars: share of predictions per bin)",
                 fontsize=10, color=INK)
    handles = [plt.Line2D([], [], marker="D", ls="", ms=7, color=SERIES[1],
                          label=f"Opus single agent (n = {d['single']['n']})"),
               plt.Line2D([], [], marker="D", ls="", ms=7, color=SERIES[0],
                          label=f"Opus pair, finals of both twins (n = {d['pair']['n']})"),
               plt.Line2D([], [], marker="|", ls="", ms=13, mew=2, color=INK_2,
                          label="Bayes, all features")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig, (a1, a2)
