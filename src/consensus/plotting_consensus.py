"""Figures for the consensus game, entrenchment and talk-onset experiments (note 2026-11-11).

Functions take already-computed arrays and return (fig, axes); they never run experiments.
Colors: categorical slots in fixed order; rounds (ordinal) use one blue ramp, light -> dark.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b", "#0a2a52"]  # ordinal: round 0 .. 4
INK, MUTED, GRID = "#2b2b2b", "#77766f", "#e6e5e0"


def _style(ax) -> None:
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=8)


def plot_consensus_sweep(rho_grid, ideal, ds, general, rho_learn, lrn, tol):
    """Four panels: ideal agreement by round vs rho; opening disagreement and the ideal
    consensus' excess; rounds needed vs d for a general Sigma; learned twins' agreement vs rho."""
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8))
    K = ideal["agree"].shape[1]

    ax = axes[0]
    for k in range(K):
        ax.plot(rho_grid, ideal["agree"][:, k], color=RAMP[k], lw=2, label=f"by round {k}")
    ax.set(xlabel=r"$\rho$", ylabel=f"share agreeing within {tol:g}", ylim=(-0.02, 1.02))
    ax.set_title("Ideal (Bayesian) twins, d = 3", fontsize=10, color=INK)
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1]
    ax.plot(rho_grid, ideal["open_dis"], color=CAT[0], lw=2, label="opening gap E[(m^A-m^B)^2]")
    ax.plot(rho_grid, ideal["excess"][:, 0], color=CAT[1], lw=2, label="excess of average, round 0")
    ax.plot(rho_grid, ideal["excess"][:, -1], color=CAT[2], lw=2, label="excess of consensus")
    ax.set(xlabel=r"$\rho$", ylabel="squared error units")
    ax.set_title("What the dialogue has to fix", fontsize=10, color=INK)
    ax.legend(fontsize=7, frameon=False)

    ax = axes[2]
    Kg = general["agree"].shape[1]
    for k in range(1, Kg):
        ax.plot(ds, general["agree"][:, k], "o-", color=RAMP[k], lw=2, ms=5, label=f"by round {k}")
    ax.set(xlabel="private dimensions d (= d_A = d_B)", ylabel=f"share agreeing within {tol:g}")
    ax.set_ylim(-0.02, 1.02)
    ax2 = ax.twiny()  # same y scale; annotate the median rounds to exact agreement per d
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(ds)
    ax2.set_xticklabels([f"{np.nanmedian(r):.0f}" for r in general["rounds"]], fontsize=7)
    ax2.set_xlabel("median rounds to exact agreement", fontsize=7, color=MUTED)
    ax.set_title("Ideal twins, general Σ", fontsize=10, color=INK, pad=24)
    ax.legend(fontsize=7, frameon=False, loc="lower left")

    ax = axes[3]
    for k in range(K):
        ax.errorbar(
            rho_learn,
            lrn["agree"][:, k],
            yerr=lrn["agree_se"][:, k],
            fmt="o-",
            color=RAMP[k],
            lw=2,
            ms=5,
            capsize=2,
            label=f"by round {k}",
        )
    ax.set(xlabel=r"$\rho$", ylabel=f"share agreeing within {tol:g}", ylim=(-0.02, 1.02))
    ax.set_title("Learned linear twins (n = 200)", fontsize=10, color=INK)
    ax.legend(fontsize=7, frameon=False)
    for a in axes:
        _style(a)
    _style(ax2)
    fig.tight_layout()
    return fig, axes


def plot_talk_onset(fracs, final, n_Ts, rhos, modes, ref=None):
    """Final test excess MSE vs the share of the budget spent learning alone.

    final : array (len(modes), len(rhos), len(n_Ts), len(fracs), 2): mean and s.e.
    ref : optional (len(modes), len(rhos)) population limit of the talking protocol.
    """
    fig, axes = plt.subplots(
        len(modes), len(rhos), figsize=(4.2 * len(rhos), 3.2 * len(modes)), squeeze=False
    )
    for i, mode in enumerate(modes):
        for j, rho in enumerate(rhos):
            ax = axes[i, j]
            for k, nT in enumerate(n_Ts):
                m, se = final[i, j, k, :, 0], final[i, j, k, :, 1]
                ax.plot(fracs, m, "o-", color=CAT[k], lw=2, ms=4, label=f"$n_T$ = {nT}")
                ax.fill_between(fracs, m - se, m + se, color=CAT[k], alpha=0.15, lw=0)
            if ref is not None and ref[i, j] > 1e-6:  # a zero limit has no place on a log axis
                ax.axhline(ref[i, j], color=MUTED, ls="--", lw=1, label="population limit")
            ax.set_yscale("log")
            ax.set_title(f"{mode}, ρ = {rho:g}", fontsize=9, color=INK)
            if i == len(modes) - 1:
                ax.set_xlabel("share of budget learning alone ($s/n_T$)")
            if j == 0:
                ax.set_ylabel("team test excess MSE")
            _style(ax)
    axes[0, -1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    return fig, axes


def plot_llm_consensus(rhos, stats, time_stats):
    """LLM consensus game. `stats[label][rho]` holds per-run arrays; `time_stats[label][rho]`
    holds per-window means (windows x runs). Six panels: agreement by round; team loss vs
    baselines; where the consensus lands; deference, rounds used and the opening gap over time."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.2))
    x = np.arange(len(rhos))

    ax = axes[0, 0]
    agree = np.array([stats["C4"][r]["agreed"] for r in rhos])  # (rho, runs)
    by1 = np.array([stats["C4"][r]["agree_r1"] for r in rhos])
    by2 = np.array([stats["C4"][r]["agree_r2"] for r in rhos])
    for arr, lab, c in (
        (by1, "by round 1", RAMP[0]),
        (by2, "by round 2", RAMP[1]),
        (agree, "by round 4", RAMP[3]),
    ):
        ax.errorbar(
            x,
            arr.mean(1),
            yerr=arr.std(1) / np.sqrt(arr.shape[1]),
            fmt="o-",
            color=c,
            lw=2,
            ms=5,
            capsize=2,
            label=lab,
        )
    lin = np.array([stats["C"][r]["agreed"] for r in rhos])
    ax.plot(x, lin.mean(1), "s--", color=MUTED, lw=1.2, ms=5, label="linear twins, round 3")
    ax.set(
        xticks=x,
        xticklabels=[f"ρ={r:g}" for r in rhos],
        ylim=(-0.02, 1.05),
        ylabel="share of steps agreeing (≤ 0.1)",
    )
    ax.set_title("Do LLM twins reach a consensus?", fontsize=10, color=INK)
    ax.legend(fontsize=7, frameon=False, loc="center right")

    ax = axes[0, 1]
    labs = [lab for lab in ("P0", "P1", "S1n", "C4", "C") if lab in stats]
    w = 0.8 / len(labs)
    for k, lab in enumerate(labs):
        vals = [stats[lab][r].get("game_excess", stats[lab][r]["excess"]) for r in rhos]
        m = [np.mean(v) for v in vals]
        se = [np.std(v) / np.sqrt(len(v)) for v in vals]
        ax.bar(
            x + (k - (len(labs) - 1) / 2) * w,
            m,
            w * 0.92,
            yerr=se,
            color=CAT[k],
            capsize=2,
            label=lab,
            error_kw=dict(lw=1, ecolor=INK),
        )
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set(xticks=x, xticklabels=[f"ρ={r:g}" for r in rhos], ylabel="team excess loss (tail)")
    ax.set_title("Score (C4: penalties included)", fontsize=10, color=INK)
    ax.legend(fontsize=7, frameon=False, ncol=2)

    ax = axes[0, 2]
    bins = np.linspace(-0.5, 1.5, 21)
    for k, r in enumerate(rhos):
        cA = np.concatenate(stats["C4"][r]["concede_all"])
        cA = cA[np.isfinite(cA)]
        ax.hist(
            np.clip(cA, -0.5, 1.5),
            bins=bins,
            histtype="step",
            lw=2,
            color=CAT[k],
            density=True,
            label=f"ρ={r:g} (outside [0,1]: {np.mean((cA < -0.02) | (cA > 1.02)):.0%})",
        )
    ax.axvline(0.5, color=MUTED, ls=":", lw=1)
    ax.set(xlabel="where the consensus lands: 0 = A's opening, 1 = B's opening", ylabel="density")
    ax.set_title("Meeting point", fontsize=10, color=INK)
    ax.legend(fontsize=7, frameon=False)

    for ax, key, ylab, title in (
        (axes[1, 0], "woa", "weight of advice, round 1 → 2", "Entrenchment: deference over time"),
        (axes[1, 1], "rounds", "rounds used", "Rounds to consensus over time"),
        (axes[1, 2], "open_gap", "|A - B| in round 1", "Opening disagreement over time"),
    ):
        for k, r in enumerate(rhos):
            for lab, ls in (("C4", "-"), ("S1n", "--")):
                if key not in time_stats.get(lab, {}).get(r, {}):
                    continue
                if key == "rounds" and lab == "S1n":  # fixed at 2 rounds by the protocol
                    continue
                centers, arr = time_stats[lab][r]["centers"], time_stats[lab][r][key]
                m = np.nanmean(arr, 1)
                se = np.nanstd(arr, 1) / np.sqrt(np.sum(np.isfinite(arr), 1).clip(1))
                ax.plot(
                    centers,
                    m,
                    ls,
                    color=CAT[k],
                    lw=2,
                    label=f"{lab} ρ={r:g}" if (key == "woa" or lab == "C4") else None,
                )
                ax.fill_between(centers, m - se, m + se, color=CAT[k], alpha=0.12, lw=0)
        ax.set(xlabel="step", ylabel=ylab)
        ax.set_title(title, fontsize=10, color=INK)
        ax.legend(fontsize=7, frameon=False, ncol=2)
    for a in axes.flat:
        _style(a)
    fig.tight_layout()
    return fig, axes


def plot_onset_entrench(res, rhos, priors, ss, taus):
    """Linear twins: entrenchment vs steps since talk onset, one line per solo phase length s.

    res : array (rhos, priors, ss, taus, 3 metrics, [mean, s.e.]); metrics are the message
    weight as a share of the best response's, the movement toward the best response, and the
    team excess MSE. Solid lines: training-posterior prior; dashed: fixed-strength prior.
    """
    titles = [
        "weight on partner / best response (median)",
        "movement toward best response",
        "team excess MSE",
    ]
    fig, axes = plt.subplots(len(rhos), 3, figsize=(14, 3.6 * len(rhos)), squeeze=False)
    for (j, rho), m in ((jr, m) for jr in enumerate(rhos) for m in range(3)):
        ax = axes[j, m]
        for q, s in enumerate(ss):
            for i, ls in zip(range(len(priors)), ("-", "--")):
                mean, se = res[j, i, q, :, m, 0], res[j, i, q, :, m, 1]
                lab = f"alone {s} steps" if i == 0 else None
                ax.plot(taus, mean, ls, color=RAMP[min(q, len(RAMP) - 1)], lw=2, label=lab)
                ax.fill_between(taus, mean - se, mean + se, color=RAMP[min(q, 4)], alpha=0.12)
        ax.set_xscale("log")
        if m < 2:
            ax.axhline(1.0, color=MUTED, lw=0.8, ls=":")
        else:
            ax.set_yscale("log")
        ax.set_title(f"{titles[m]}, ρ = {rho:g}", fontsize=9, color=INK)
        ax.set_xlabel("steps since talk started")
        _style(ax)
    axes[0, 0].legend(
        fontsize=7,
        frameon=False,
        title="solid: training prior\ndashed: fixed prior",
        title_fontsize=7,
    )
    fig.tight_layout()
    return fig, axes


def plot_onset_llm(rhos, arms, ns, stats, windows, window):
    """LLM twins that studied n cases before talking. Four panels: WOA vs n, WOA over the run
    per arm, 80% interval width vs n, and the twins' opening and team excess vs n.

    stats[rho][arm][metric] : per-run values; windows[rho][arm] : (runs, windows) WOA.
    """
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8))

    def mse(v):
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        return v.mean(), v.std() / np.sqrt(max(len(v), 1))

    for k, rho in enumerate(rhos):
        for ax, metric, ylab in (
            (axes[0], "woa", "weight of advice, round 1 → 2"),
            (axes[2], "width", "80% interval width (opening)"),
        ):
            ms = [mse(stats[rho][a][metric]) for a in arms]
            ax.errorbar(
                ns,
                [m for m, _ in ms],
                yerr=[s for _, s in ms],
                fmt="o-",
                color=CAT[k],
                lw=2,
                ms=5,
                capsize=2,
                label=f"ρ = {rho:g}",
            )
            ax.set(xscale="log", xlabel="cases studied alone before talking", ylabel=ylab)
            ax.set_xticks(ns, [str(n) for n in ns])
        ax = axes[3]
        for metric, ls in (("solo_excess", "--"), ("team_excess", "-")):
            ms = [mse(stats[rho][a][metric]) for a in arms]
            ax.errorbar(
                ns,
                [m for m, _ in ms],
                yerr=[s for _, s in ms],
                fmt="o" + ls,
                color=CAT[k],
                lw=2,
                ms=5,
                capsize=2,
                label=f"ρ = {rho:g}, {'own opening' if ls == '--' else 'team'}",
            )
        ax.set(xscale="log", xlabel="cases studied alone before talking", ylabel="excess MSE")
        ax.set_xticks(ns, [str(n) for n in ns])
        ax = axes[1]
        for q, arm in enumerate(arms):
            W = windows[rho][arm]
            centers = (np.arange(W.shape[1]) + 0.5) * window
            ax.plot(
                centers,
                np.nanmean(W, 0),
                "o-" if k == 0 else "s--",
                color=RAMP[min(q + 1, 4)],
                lw=2,
                ms=4,
                label=f"n = {ns[q]}, ρ = {rho:g}",
            )
        ax.set(xlabel="step", ylabel="weight of advice, round 1 → 2")
    axes[0].set_title("Deference vs time spent alone", fontsize=10, color=INK)
    axes[1].set_title("Deference over the run", fontsize=10, color=INK)
    axes[2].set_title("Stated confidence", fontsize=10, color=INK)
    axes[3].set_title("Accuracy", fontsize=10, color=INK)
    for i, ax in enumerate(axes):
        ax.legend(fontsize=7, frameon=False)
        if i != 1:  # log axes: only the n ticks, no minor labels
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.set_xticks(ns, [str(n) for n in ns])
        _style(ax)
    fig.tight_layout()
    return fig, axes
