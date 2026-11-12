"""Figure for the overlapping-features experiment (`scripts/overlap_analysis.py`).

Takes the per-arm summaries the analysis computed; never runs experiments. Same colors and style
as `plotting_consensus` (categorical slots in fixed order).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from consensus.plotting_consensus import CAT, INK, MUTED, _style  # noqa: E402

RULE_LABELS = {"own": "keep own", "average": "average", "sum": "sum", "sum - shared": "sum − shared"}


def plot_overlap(summary: dict, path: Path) -> None:
    """Left: fitted weights of the final estimate on own opening (alpha) and partner opening
    (beta), per arm, with per-run dots; the best response is 1 for both, averaging 1/2.
    Right: share of finals closest to each combination rule."""
    arms = list(summary)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1, 1.2]})
    x = np.arange(len(arms))
    for j, (key, lab) in enumerate((("alpha", "own opening (α)"), ("beta", "partner opening (β)"))):
        off = (j - 0.5) * 0.3
        vals = [summary[a][key] for a in arms]
        ax.bar(x + off, vals, 0.28, color=CAT[j], label=lab, edgecolor="white", lw=2)
        for i, a in enumerate(arms):
            pts = summary[a][f"{key}_runs"]
            ax.scatter(np.full(len(pts), x[i] + off), pts, s=12, color=INK, zorder=3, lw=0)
    ax.axhline(1.0, color=INK, lw=1, ls="--")
    ax.axhline(0.5, color=MUTED, lw=1, ls=":")
    ax.text(len(arms) - 0.5, 1.02, "best response", fontsize=7, color=INK, ha="right", va="bottom")
    ax.text(len(arms) - 0.5, 0.52, "averaging", fontsize=7, color=MUTED, ha="right", va="bottom")
    ax.set_xticks(x, [f"{a}\n(overlap {summary[a]['overlap']})" for a in arms], fontsize=8)
    ax.set_ylabel("weight in the final estimate", fontsize=8, color=INK)
    ax.set_title("How the final estimate uses the two openings", fontsize=10, color=INK)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    _style(ax)

    bottom = np.zeros(len(arms))
    for j, (rule, lab) in enumerate(RULE_LABELS.items()):
        vals = np.array([summary[a]["nearest_rule"][rule] for a in arms])
        bx.bar(x, vals, 0.6, bottom=bottom, color=CAT[j + 2], label=lab, edgecolor="white", lw=2)
        bottom += vals
    bx.set_xticks(x, arms, fontsize=8)
    bx.set_ylim(0, 1)
    bx.set_ylabel("share of final estimates", fontsize=8, color=INK)
    bx.set_title("Nearest combination rule", fontsize=10, color=INK)
    bx.legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    _style(bx)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
