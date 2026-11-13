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


def plot_compaction(summary: dict, window_labels: list[str], path: Path) -> None:
    """Entrenchment over the run, per compaction arm (`scripts/compact_analysis.py`).

    Left: movement toward the best response by window (0 = kept own opening, 1 = best response).
    Right: fitted weight on the partner's opening (β) by window. Arms are ordered by threshold,
    so they use one ramp, light (most compaction) -> dark (never compacted)."""
    from consensus.plotting_consensus import RAMP

    arms = list(summary)
    colors = RAMP[: len(arms)] if len(arms) <= len(RAMP) else CAT
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    x = np.arange(len(window_labels))
    for ax, key, title in (
        (axes[0], "movement", "Movement toward the best response"),
        (axes[1], "beta", "Weight on partner's opening (β)"),
    ):
        for arm, c in zip(arms, colors):
            vals = [w[key] for w in summary[arm]["windows"]]
            ax.plot(x, vals, "o-", color=c, lw=2, ms=6, label=arm)
        ax.axhline(1.0, color=INK, lw=1, ls="--")
        ax.set_xticks(x, [f"steps {w}" for w in window_labels], fontsize=8)
        ax.set_title(title, fontsize=10, color=INK)
        _style(ax)
    axes[0].set_ylabel("median share of the way", fontsize=8, color=INK)
    axes[1].set_ylabel("OLS coefficient", fontsize=8, color=INK)
    axes[1].legend(fontsize=7, frameon=False, title="compact at", title_fontsize=7,
                   loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_sessions_lambda(by_lambda: dict, path: Path) -> None:
    """Live-session twins in the controlled-lambda design (`scripts/sessions_lambda_analysis.py`).

    Left: how the agreed prediction combines the openings, c_A + c_B (pairs as dots), for the
    25- and 400-case arms, against the population-optimal blend (dashed) and averaging (dotted).
    Right: first-attempt consensus rate by arm (lower = harder to agree)."""
    lams = list(by_lambda)
    x = np.array([float(v) for v in lams])
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9, 3.4))
    for j, (arm, lab) in enumerate((("25", "studied 25"), ("400", "studied 400"))):
        off = (j - 0.5) * 0.02
        vals = [np.nanmean(by_lambda[v][f"c_sum{arm}"]) for v in lams]
        ax.plot(x + off, vals, "o-", color=CAT[j], lw=2, ms=6, label=lab)
        for xi, v in zip(x, lams):
            pts = by_lambda[v][f"c_sum{arm}"]
            ax.scatter(np.full(len(pts), xi + off), pts, s=10, color=CAT[j], alpha=0.5, lw=0)
    ax.plot(x, [by_lambda[v]["optimal_c_sum"] for v in lams], "--", color=INK, lw=1,
            label="optimal blend")
    ax.axhline(1.0, color=MUTED, lw=1, ls=":")
    ax.text(x[-1], 1.02, "averaging", fontsize=7, color=MUTED, ha="right", va="bottom")
    ax.set_xlabel("λ", fontsize=8, color=INK)
    ax.set_ylabel("c_A + c_B", fontsize=8, color=INK)
    ax.set_title("Agreed prediction on the two openings", fontsize=10, color=INK)
    ax.legend(fontsize=7, frameon=False)
    _style(ax)
    for j, arm in enumerate(("25", "400")):
        bx.plot(x, [by_lambda[v][f"first{arm}"] for v in lams], "o-", color=CAT[j], lw=2, ms=6,
                label=f"studied {arm}")
    bx.set_ylim(0, 1.05)
    bx.set_xlabel("λ", fontsize=8, color=INK)
    bx.set_ylabel("share of queries", fontsize=8, color=INK)
    bx.set_title("Consensus on the first attempt", fontsize=10, color=INK)
    bx.legend(fontsize=7, frameon=False)
    _style(bx)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
