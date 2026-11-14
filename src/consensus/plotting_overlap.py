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


def plot_entrench_sessions(out: dict, lambdas: list[str], path: Path) -> None:
    """Entrenchment with live-session twins (`scripts/entrench_sessions_analysis.py`).

    Left: the long-trained twin's weight of advice in mixed pairs against how long it studied
    (log scale), one line per lambda, with the rational partner weight dashed (conceding less
    than that = entrenchment). Right: equal pairs, the share of agreements at the exact
    midpoint against n. Lambda is ordinal: one blue ramp, light (0) -> dark (0.9)."""
    from consensus.plotting_consensus import RAMP

    colors = RAMP[: len(lambdas)]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.5, 3.5))
    for lam, c in zip(lambdas, colors):
        ks = sorted((k for k in out["mixed"] if k.split("_")[0] == lam),
                    key=lambda k: int(k.split("_")[1]))
        if ks:
            n = [int(k.split("_")[1]) for k in ks]
            w = [out["mixed"][k]["woa_long"] for k in ks]
            e = [out["mixed"][k]["woa_long_se"] for k in ks]
            r = [out["mixed"][k]["rational_partner_weight"] for k in ks]
            ax.errorbar(n, w, yerr=e, fmt="o-", color=c, lw=2, ms=6, capsize=3, label=f"λ = {lam}")
            ax.plot(n, r, "--", color=c, lw=1)
        ke = sorted((k for k in out["equal"] if k.split("_")[0] == lam),
                    key=lambda k: int(k.split("_")[1]))
        if ke:
            bx.plot([int(k.split("_")[1]) for k in ke], [out["equal"][k]["midpoint"] for k in ke],
                    "o-", color=c, lw=2, ms=6, label=f"λ = {lam}")
    ax.axhline(0.5, color=MUTED, lw=1, ls=":")
    ax.text(ax.get_xlim()[1] if False else 1600, 0.51, "even split", fontsize=7, color=MUTED,
            ha="right", va="bottom")
    for a in (ax, bx):
        a.set_xscale("log")
        a.set_xticks([25, 100, 400, 1600], ["25", "100", "400", "1600"])
        a.set_xlabel("cases studied alone (long-trained twin / both twins)", fontsize=8, color=INK)
        _style(a)
    ax.set_ylabel("WOA of the long-trained twin", fontsize=8, color=INK)
    ax.set_title("Mixed pairs (partner studied 25); dashed = rational", fontsize=10, color=INK)
    ax.legend(fontsize=7, frameon=False)
    bx.set_ylim(0, 1.05)
    bx.set_ylabel("share of agreements at the midpoint", fontsize=8, color=INK)
    bx.set_title("Equal pairs", fontsize=10, color=INK)
    bx.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_entrench_panels(out: dict, lambdas: list[str], ns: list[int], path: Path) -> None:
    """Entrenchment with live-session twins, one column per lambda.

    Top row: the long-trained twin's weight of advice in mixed pairs (partner studied 25) against
    how many cases it studied, mean with per-pair dots; the rational partner weight (dashed) and
    the even split (dotted) for reference. Below the rational line = conceding less than the
    openings' accuracy justifies. Bottom row: MSE of the agreed prediction against simply
    averaging the two openings and the least-squares combination of the openings."""
    fig, axes = plt.subplots(2, len(lambdas), figsize=(3.0 * len(lambdas), 6.0), sharex=True,
                             sharey="row")
    for j, lam in enumerate(lambdas):
        ks = [f"{lam}_{n}" for n in ns if f"{lam}_{n}" in out["mixed"]]
        n = np.array([int(k.split("_")[1]) for k in ks])
        cells = [out["mixed"][k] for k in ks]
        ax = axes[0, j]
        for x, c in zip(n, cells):
            pts = [p["woa_long"] for p in c["pairs"]]
            jitter = x * np.exp(np.linspace(-0.08, 0.08, len(pts)))
            ax.scatter(jitter, pts, s=12, color=CAT[0], alpha=0.35, lw=0, zorder=2)
        ax.errorbar(n, [c["woa_long"] for c in cells], yerr=[c["woa_long_se"] for c in cells],
                     fmt="o-", color=CAT[0], lw=2, ms=6, capsize=3, zorder=3,
                     label="long-trained twin")
        ax.plot(n, [c["rational_partner_weight"] for c in cells], "--", color=INK, lw=1,
                label="rational weight")
        ax.axhline(0.5, color=MUTED, lw=1, ls=":", label="even split")
        ax.set_title(f"λ = {lam}", fontsize=10, color=INK)
        _style(ax)
        bx = axes[1, j]
        for key, lab, col, ls in (("mse_agreed", "agreed prediction", CAT[1], "-"),
                                  ("mse_average", "average of openings", MUTED, "--"),
                                  ("mse_fitted", "best combination", INK, ":")):
            bx.plot(n, [c[key] for c in cells], "o" + ls if ls != ":" else "o:", color=col,
                    lw=2 if key == "mse_agreed" else 1, ms=5, label=lab)
        _style(bx)
        bx.set_xscale("log")
        bx.set_xticks(ns, [str(v) for v in ns])
        bx.minorticks_off()
        bx.set_xlabel("cases the long twin studied", fontsize=8, color=INK)
    axes[0, 0].set_ylabel("weight of advice of the long-trained twin\n(0 = agreement on its opening)",
                          fontsize=8, color=INK)
    axes[1, 0].set_ylabel("mean squared error", fontsize=8, color=INK)
    axes[0, 0].set_ylim(-0.05, 0.8)
    axes[0, -1].legend(fontsize=7, frameon=False, loc="upper right")
    axes[1, -1].legend(fontsize=7, frameon=False, loc="upper right")
    fig.suptitle("Studying alone for longer: the long-trained twin concedes less than its accuracy "
                 "justifies (partner studied 25 cases; 8 pairs per point)", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_avg_sessions(out: dict, lambdas: list[str], ns: list[int], path: Path) -> None:
    """Live-session twins under the averaging rule (`scripts/avg_sessions_analysis.py`), one
    column per lambda. Each twin's weight of advice on its own final against cases studied:
    equal pairs (both studied n), and in mixed pairs the long- and short-trained twin
    (partner studied 25), with the long twin under the consensus rule (dashed) for comparison."""
    fig, axes = plt.subplots(1, len(lambdas), figsize=(3.0 * len(lambdas), 3.6), sharey=True)
    for ax, lam in zip(np.atleast_1d(axes), lambdas):
        ke = [n for n in ns if f"{lam}_{n}" in out["equal"]]
        if ke:
            ax.errorbar(ke, [out["equal"][f"{lam}_{n}"]["woa"] for n in ke],
                        yerr=[out["equal"][f"{lam}_{n}"]["woa_se"] for n in ke], fmt="o-",
                        color=CAT[0], lw=2, ms=6, capsize=3, label="equal pairs")
        km = [n for n in ns if f"{lam}_{n}" in out["mixed"]]
        if km:
            mx = [out["mixed"][f"{lam}_{n}"] for n in km]
            ax.errorbar(km, [c["woa_long"] for c in mx], yerr=[c["woa_long_se"] for c in mx],
                        fmt="o-", color=CAT[1], lw=2, ms=6, capsize=3, label="long-trained twin")
            ax.errorbar(km, [c["woa_short"] for c in mx], yerr=[c["woa_short_se"] for c in mx],
                        fmt="o-", color=CAT[2], lw=2, ms=6, capsize=3, label="its partner (25)")
            ax.plot(km, [c["woa_long_consensus"] for c in mx], "--", color=CAT[1], lw=1,
                    label="long twin, consensus rule")
        ax.set_xscale("log")
        ax.set_xticks(ns, [str(v) for v in ns])
        ax.minorticks_off()
        ax.set_xlabel("cases studied (both / the long twin)", fontsize=8, color=INK)
        ax.set_title(f"λ = {lam}", fontsize=10, color=INK)
        _style(ax)
    axes[0].set_ylabel("weight of advice on own final\n(0 = kept own opening)", fontsize=8,
                       color=INK)
    axes[-1].legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.suptitle("Averaging rule: each twin submits its own final after talking", fontsize=10,
                 color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
