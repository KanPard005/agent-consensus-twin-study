"""Plots for the boolean task (LogBook note 2026-11-18); input is the summary dict of
scripts/boolean_analysis.py."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_boolean_summary(s: dict):
    """Left: accuracy of the solo predictions, openings and finals, with the Bayes solo and joint
    accuracies on the same partner queries as lines. Right: where the twin's own clause is 1, the
    share of finals whose label equals the partner's clause (1 = the joint rule), by block of 10
    partner queries.

    Returns
    -------
    (fig, axes)
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axes[0]
    names = ["solo phase", "opening", "final"]
    vals = [s["solo"]["acc"], s["opening"]["acc"], s["final"]["acc"]]
    ax.bar(names, vals, color=["#9aa5b1", "#5b8def", "#1f4fbf"])
    ax.axhline(s["bayes_solo"]["acc"], ls="--", c="k", lw=1, label="Bayes, own features")
    ax.axhline(s["bayes_joint"]["acc"], ls=":", c="k", lw=1, label="Bayes, all features")
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("accuracy")
    ax.set_title(f"{s['pairs']} pairs")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax = axes[1]
    blocks = s["eq_d_other_by_block_d_own1"]
    xs = [int(b) for b in blocks]
    ax.plot([10 * b + 5 for b in xs], list(blocks.values()), marker="o", c="#1f4fbf")
    ax.axhline(s["solo_rule_eq_d_other_d_own1"], ls="--", c="k", lw=1,
               label="Bayes, own features (always 1)")
    ax.axhline(1.0, ls=":", c="k", lw=1, label="Bayes, all features")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("partner query")
    ax.set_ylabel("final label = partner's clause")
    ax.set_title("own clause = 1 (answer is the partner's)")
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig, axes
