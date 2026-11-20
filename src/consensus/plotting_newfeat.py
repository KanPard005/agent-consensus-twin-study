"""Figures for the new-features note (LogBook agent-consensus, 2026-11-19, "New features at the
queries: summary").

Inputs are plain dicts computed by scripts/plot_newfeat_note.py from saved results; the functions
only draw. Colours and styling are those of `consensus.plotting_boolean_note`: arm "new at solo"
takes series slot 1, "new at partner" slot 2, and Bayes references are neutral grey.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from consensus.plotting_boolean_note import GRID, INK, INK_2, SERIES, _style  # noqa: E402


def plot_share_by_case(rows: list[dict], arms: list[str]):
    """Share of finals saying 1 per case type (one row each), one dot per arm, with the share
    the Bayes-optimal label would give (0 or 1) as a grey tick.

    Parameters
    ----------
    rows : list of {"label", "bayes", arm: share, ...}, top to bottom.
    arms : arm names, in series order (they key the dots and the legend).

    Returns
    -------
    (fig, ax)
    """
    n = len(rows)
    fig, ax = plt.subplots(figsize=(7.2, 0.5 * n + 1.2))
    for i, r in enumerate(rows):
        yv = n - 1 - i
        ax.plot([r["bayes"]], [yv], marker="|", ms=14, mew=2, color=INK_2, zorder=2)
        vals = [r[a] for a in arms]
        ax.plot([min(vals), max(vals)], [yv, yv], color=GRID, lw=2, zorder=1)
        for j, a in enumerate(arms):
            # a small vertical offset per arm, so equal values (e.g. both at 0) stay visible
            off = 0.09 * (j - (len(arms) - 1) / 2)
            ax.plot([r[a]], [yv + off], "o", ms=8, color=SERIES[j], mec="white", mew=1.5, zorder=3)
    ax.set_yticks(range(n))
    ax.set_yticklabels([r["label"] for r in rows][::-1], fontsize=8.5, color=INK)
    ax.set_xlabel("share of finals saying y = 1 (partner queries 101–200)")
    ax.set_xlim(-0.04, 1.04)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    handles = [plt.Line2D([], [], marker="o", ls="", ms=8, color=SERIES[j], label=a)
               for j, a in enumerate(arms)]
    handles.append(plt.Line2D([], [], marker="|", ls="", ms=12, mew=2, color=INK_2,
                              label="Bayes-optimal label"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2,
              frameon=False, fontsize=8.5)
    _style(ax)
    fig.tight_layout()
    return fig, ax


def plot_new_zero_over_time(curves: dict[str, list[float]], n_solo_blocks: int, block: int,
                            bayes: float):
    """Share saying 1 when the twin's own new clause is the only zero, per block of queries:
    solo predictions, then partner-phase finals, one line per arm.

    Parameters
    ----------
    curves : {arm: shares per block (solo blocks first, then partner blocks)}
    n_solo_blocks : number of solo blocks (a divider marks the partner's arrival).
    block : queries per block (for the x labels).
    bayes : share saying 1 of the Bayes-optimal label for these cases (grey dotted line).

    Returns
    -------
    (fig, ax)
    """
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for j, (arm, ys) in enumerate(curves.items()):
        xs = range(1, len(ys) + 1)
        ax.plot(xs, ys, "-o", lw=2, ms=6, color=SERIES[j], mec="white", mew=1.2, label=arm)
    n = max(len(v) for v in curves.values())
    ax.axvline(n_solo_blocks + 0.5, color=INK_2, lw=1, ls="--")
    ax.text(n_solo_blocks + 0.6, 0.12, "partner arrives", color=INK_2, fontsize=8.5)
    ax.axhline(bayes, color=INK_2, lw=1, ls=":")
    ax.text(n + 0.4, bayes + 0.03, "Bayes-optimal label", color=INK_2, fontsize=8.5, ha="right")
    ax.set_xticks(range(1, n + 1))
    ax.set_xticklabels([f"{i * block + 1}–{(i + 1) * block}" for i in range(n)], fontsize=7.5,
                       rotation=30)
    ax.set_xlabel("queries (solo predictions, then partner finals)")
    ax.set_ylabel("share saying y = 1")
    ax.set_ylim(-0.03, 1.05)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.32),
              ncol=len(curves))
    _style(ax)
    fig.tight_layout()
    return fig, ax
