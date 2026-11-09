"""Compare conditions saved in different run folders that share seed and data streams.

Usage:
    python scripts/compare_runs.py <name> P4=results/llm_haiku P5=results/llm_haiku_joint ...
        [--baselines P0 P1 P2 P3] [--window 10] [--last 30]

Each LABEL=PATH takes the latest timestamped run under PATH (or PATH itself if it is a run
folder) and its protocol LABEL. Linear baselines come from the first folder; P0 must agree
across folders (same streams). Writes results/compare/<name>/<timestamp>/ with table.md,
table.json and figures/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consensus.plotting import (  # noqa: E402
    SERIES_COLORS,
    plot_condition_curves,
    plot_condition_summary,
    running_mean,
)
from consensus.utils import load_run, make_run_dir  # noqa: E402


def latest_run(path: Path) -> Path:
    if (path / "metrics.npz").exists():
        return path
    return sorted(p for p in path.iterdir() if (p / "metrics.npz").exists())[-1]


def describe(cfg: dict, label: str) -> str:
    """Short description of a protocol from its config entry."""
    pc = next(p for p in cfg["protocols"] if p.get("label") == label)
    if pc["name"] != "llm_exchange":
        return {
            "no_talk": "no talk (average)",
            "blend": "frozen blend",
            "exchange": f"linear exchange, {pc.get('n_exchanges')} exchange"
            + ("s" if pc.get("n_exchanges", 1) > 1 else ""),
        }[pc["name"]]
    parts = ["Haiku twins"]
    parts.append("joint estimate" if pc.get("team_rule") == "joint" else "own estimates")
    if pc.get("auto_regress"):
        parts.append("auto regression")
    elif pc.get("tools"):
        parts.append("regression tool")
    return ", ".join(parts)


def main(argv=None) -> Path:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("name")
    p.add_argument("conditions", nargs="+", help="LABEL=PATH")
    p.add_argument("--baselines", nargs="*", default=["P0", "P1", "P2", "P3"])
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--last", type=int, default=30, help="steps at the end used for the table")
    p.add_argument("--curves", nargs="*", help="labels drawn in the time plots (default: all)")
    args = p.parse_args(argv)

    runs = {}
    for item in args.conditions:
        label, path = item.split("=", 1)
        d = latest_run(ROOT / path if not Path(path).is_absolute() else Path(path))
        cfg, results, bench = load_run(d)
        runs[label] = (d, cfg, results[label], bench)
    d0, cfg0, _, bench = next(iter(runs.values()))
    _, base_results, _ = load_run(d0)
    for label, (d, _, _, _) in runs.items():
        _, res, _ = load_run(d)
        if not np.allclose(res["P0"]["sq_err"], base_results["P0"]["sq_err"]):
            raise ValueError(f"{d} does not share data streams with {d0}")

    s2 = bench["full_info"]
    cond = {lab: base_results[lab] for lab in args.baselines}
    cond.update({lab: r for lab, (_, _, r, _) in runs.items()})
    desc = {lab: describe(cfg0, lab) for lab in args.baselines}
    desc.update({lab: describe(cfg, lab) for lab, (_, cfg, _, _) in runs.items()})
    folders = {lab: str(d0.relative_to(ROOT)) for lab in args.baselines}
    folders.update({lab: str(d.relative_to(ROOT)) for lab, (d, _, _, _) in runs.items()})

    T = cond["P0"]["sq_err"].shape[1]
    thirds = [slice(i * T // 3, (i + 1) * T // 3) for i in range(3)]
    last = slice(T - args.last, T)
    p0_last = (cond["P0"]["sq_err"][:, last] - s2).mean(1)
    rows = []
    for lab, res in cond.items():
        ex = res["sq_err"] - s2
        per_run = ex[:, last].mean(1)
        gap = np.sqrt(res["disagreement"])
        row = {
            "label": lab,
            "condition": desc[lab],
            "excess_thirds": [float(ex[:, sl].mean()) for sl in thirds],
            "excess_last": float(per_run.mean()),
            "excess_last_se": float(per_run.std(ddof=1) / np.sqrt(len(per_run))),
            "excess_last_runs": per_run.tolist(),
            "minus_P0_last_runs": (per_run - p0_last).tolist(),
            "disagreement_last": float(res["disagreement"][:, last].mean()),
            "agree_rate_last": float((gap[:, last] < 0.01).mean()),
            "tool_calls": float(np.nansum(res["tool_calls"])) if "tool_calls" in res else None,
            "cost_usd": float(np.nansum(res["cost_usd"])) if "cost_usd" in res else None,
            "folder": folders[lab],
        }
        rows.append(row)

    out = make_run_dir(ROOT / "results" / "compare", args.name)
    (out / "table.json").write_text(json.dumps(rows, indent=2))
    n_runs = cond["P0"]["sq_err"].shape[0]
    head = (
        f"Excess MSE over $\\sigma^2$ = {s2:g}; {n_runs} runs x {T} steps on the same streams. "
        f"'last {args.last}' = mean over the last {args.last} steps (± s.e. over runs); "
        "Δ P0 = paired difference per run.\n\n"
        f"| | condition | steps 1–{T // 3} | {T // 3 + 1}–{2 * T // 3} | {2 * T // 3 + 1}–{T} "
        f"| last {args.last} | Δ P0 per run | disagreement | agree (<0.01) | tool calls | cost |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    fmt = lambda v, f: "–" if v is None else format(v, f)  # noqa: E731
    lines = [
        f"| {r['label']} | {r['condition']} | "
        + " | ".join(f"{v:.2f}" for v in r["excess_thirds"])
        + f" | {r['excess_last']:.2f} ± {r['excess_last_se']:.2f} | "
        + ", ".join(f"{v:+.2f}" for v in r["minus_P0_last_runs"])
        + f" | {r['disagreement_last']:.2f} | {r['agree_rate_last']:.2f} | "
        f"{fmt(r['tool_calls'], '.0f')} | {'–' if r['cost_usd'] is None else '$' + format(r['cost_usd'], '.2f')} |"
        for r in rows
    ]
    (out / "table.md").write_text(head + "\n".join(lines) + "\n")

    figs = out / "figures"
    figs.mkdir()
    llm = [lab for lab in runs]
    # Color follows the condition (fixed by its position on the command line), not the subset.
    colors = {lab: SERIES_COLORS[i % len(SERIES_COLORS)] for i, lab in enumerate(llm)}
    shown = args.curves or llm
    name = lambda lab: f"{lab} {desc[lab].removeprefix('Haiku twins, ')}"  # noqa: E731
    smooth = lambda a: running_mean(a, args.window)  # noqa: E731
    fig, _ = plot_condition_curves(
        {name(lab): smooth(cond[lab]["sq_err"]) - s2 for lab in shown},
        {lab: smooth(cond[lab]["sq_err"]) - s2 for lab in ("P0", "P2") if lab in cond},
        ylabel=f"excess MSE over $\\sigma^2$ (EMA, span {args.window})",
        title="Team error over time: Haiku twins vs linear baselines (gray)",
        colors={name(lab): colors[lab] for lab in shown},
    )
    fig.tight_layout()
    fig.savefig(figs / "excess_mse.png", dpi=150)
    fig, _ = plot_condition_summary(
        {f"{lab}  {desc[lab]}": cond[lab]["sq_err"][:, last].mean(1) - s2 for lab in cond},
        {f"{lab}  {desc[lab]}": colors[lab] for lab in llm},
        xlabel=f"excess MSE over $\\sigma^2$, last {args.last} steps (dots: runs)",
        title="Team error at the end of the run",
    )
    fig.tight_layout()
    fig.savefig(figs / "excess_mse_last.png", dpi=150)
    fig, _ = plot_condition_curves(
        {name(lab): smooth(cond[lab]["disagreement"]) for lab in shown},
        {lab: smooth(cond[lab]["disagreement"]) for lab in ("P0", "P2") if lab in cond},
        ylabel=f"$(\\hat y^A - \\hat y^B)^2$ (EMA, span {args.window})",
        title="Disagreement between the twins' final answers",
        colors={name(lab): colors[lab] for lab in shown},
        log=True,
        ymin=1e-2,
    )
    fig.tight_layout()
    fig.savefig(figs / "disagreement.png", dpi=150)
    plt.close("all")
    print(out)
    return out


if __name__ == "__main__":
    main()
