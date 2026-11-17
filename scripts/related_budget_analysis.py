"""Training budget sweep on related targets (LogBook 2026-11-16).

Does studying alone longer make a twin rely less on a partner that predicts a related target?
Cells: c (cosine between the targets) x n (training cases per twin, whole budget served in one
call so tool practice is the same), told arm, folders `<prefix>_n<n>_c<c>_told`. Per cell, over
the late queries: the coefficient on the partner's opening (`related_analysis.twin_stats`), the
final and stacking-learner MSE, messages per query per pair, raw-feature share, joint models,
negative weights, stacking regressions tried, and pair-bootstrap 95% intervals for the change
from the smallest to each larger budget.

Usage:
    python scripts/related_budget_analysis.py [--ns 100 400 1600] [--cs 1.0 0.5 -0.5]
Writes results/related_budget/<timestamp>/{table.md, stats.json, budget.png}.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from consensus.plotting_related import plot_budget  # noqa: E402
from consensus.utils import make_run_dir  # noqa: E402
from related_analysis import load_pairs, twin_stats  # noqa: E402


def cell_stats(pairs: list[dict], late: int) -> dict:
    """Per-twin and per-pair measures for one cell."""
    out = {k: [] for k in ("beta", "mse", "open", "stack", "raw", "joint", "stack_try")}
    out["msg"] = []
    for p in pairs:
        n_msg = 0.0
        for k in "AB":
            s = twin_stats(p, k, late)
            for key, v in (("beta", s["beta"]), ("mse", s["mse_late"]),
                           ("open", s["mse_open_late"]), ("stack", s["mse_stack_late"]),
                           ("raw", s["raw_share"]), ("joint", s["joint_model"])):
                out[key].append(v)
            # A stacking regression: regress with the partner's opening among the predictors.
            out["stack_try"].append(any(c["tool"] == "regress"
                                        and "partner_opening" in json.dumps(c["args"])
                                        for c in p["calls"][k]))
            n_msg += sum(m["from"] == k for r in p["recs"] for m in r["messages"])
        out["msg"].append(n_msg / len(p["recs"]))
    out["cost"] = sum(p["cost"] for p in pairs)
    return out


def boot_diff(a: np.ndarray, b: np.ndarray, rng, reps: int = 4000) -> tuple[float, float]:
    """95% interval of mean(b) - mean(a), resampling pairs (rows) in each cell."""
    d = [np.nanmean(b[rng.integers(0, len(b), len(b))]) - np.nanmean(a[rng.integers(0, len(a), len(a))])
         for _ in range(reps)]
    return float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="llm_haiku_related")
    ap.add_argument("--ns", nargs="+", type=int, default=[100, 400, 1600])
    ap.add_argument("--cs", nargs="+", default=["1.0", "0.5", "-0.5"])
    ap.add_argument("--late", type=int, default=20)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    cells = {(float(c), n): cell_stats(load_pairs(f"{args.prefix}_n{n}", c, "told"), args.late)
             for c in args.cs for n in args.ns}
    lines = ["| c | n | pairs | weight on partner | negative (< −0.1) | stacking tried | "
             "talkative pairs | msg / query | raw-feature share | joint models | opening MSE | "
             "final MSE | stacking learner | cost |", "|" + "---|" * 14]
    for c in map(float, args.cs):
        for n in args.ns:
            s = cells[(c, n)]
            b = np.array(s["beta"], float)
            lines.append(
                f"| {c:g} | {n} | {len(s['msg'])} | {np.nanmean(b):.2f} ± "
                f"{np.nanstd(b) / np.sqrt(np.isfinite(b).sum()):.2f} | {int(np.sum(b < -0.1))} | "
                f"{sum(s['stack_try'])} | {int(np.sum(np.array(s['msg']) >= 0.5))} | "
                f"{np.mean(s['msg']):.2f} | {np.mean(s['raw']):.2f} | {sum(s['joint'])} | "
                f"{np.nanmean(s['open']):.2f} | {np.nanmean(s['mse']):.2f} | "
                f"{np.nanmean(s['stack']):.2f} | ${s['cost']:.2f} |")
    lines += ["", f"Change from n = {args.ns[0]} (pair bootstrap, 95%):", "",
              "| c | n | weight on partner | final MSE | msg / query |", "|---|---|---|---|---|"]
    for c in map(float, args.cs):
        base = cells[(c, args.ns[0])]
        for n in args.ns[1:]:
            s, row = cells[(c, n)], []
            for key in ("beta", "mse", "msg"):
                a, b = np.array(base[key], float), np.array(s[key], float)
                if key != "msg":  # per twin -> per pair
                    a, b = a.reshape(-1, 2), b.reshape(-1, 2)
                lo, hi = boot_diff(a, b, rng)
                row.append(f"{np.nanmean(b) - np.nanmean(a):+.2f} [{lo:+.2f}, {hi:+.2f}]")
            lines.append(f"| {c:g} | {n} | " + " | ".join(row) + " |")
    out = make_run_dir(ROOT / "results", "related_budget")
    (out / "table.md").write_text("\n".join(lines) + "\n")
    (out / "stats.json").write_text(json.dumps(
        {f"{c}_{n}": v for (c, n), v in cells.items()}, default=float, indent=1))
    plot_budget(cells, args.ns, [float(c) for c in args.cs], out / "budget.png")
    print("\n".join(lines))
    print(out)


if __name__ == "__main__":
    main()
