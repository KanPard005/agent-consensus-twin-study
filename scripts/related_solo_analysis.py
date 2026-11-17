"""Solo queries first (LogBook 2026-11-16): do twins that answered K queries alone before the
partner arrived rely less on each other?

Cells: c x K, told arm, folders `<prefix>_K<K>_c<c>_told`. Partner-phase measures as in
`related_budget_analysis.cell_stats` (weight on the partner's opening over late partner
queries, talk, raw features, joint models, MSE), plus the solo phase: each twin's solo MSE, its
model/regression calls while alone, and the correlation, across twins, between solo MSE and the
later weight on the partner (does a twin that did badly alone lean on its partner more?).

Usage:
    python scripts/related_solo_analysis.py [--ks 0 10 30 60] [--cs 1.0 0.5 -0.5]
Writes results/related_solo/<timestamp>/{table.md, stats.json, solo.png}.
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
from related_analysis import load_pairs  # noqa: E402
from related_budget_analysis import boot_diff, cell_stats  # noqa: E402

SOLO_MODEL_TOOLS = ("regress", "new_model", "add_case", "fit_model", "add_row", "write_log")


def solo_stats(prefix: str, c: str) -> dict:
    """Per twin (same order as cell_stats: pair by pair, A then B): solo MSE and model calls."""
    mse, calls = [], []
    for d in sorted((ROOT / "results" / f"{prefix}_c{c}_told").glob("20*")):
        if not (d / "summary.json").exists():
            continue
        for key in json.loads((d / "summary.json").read_text()):
            for k in "AB":
                f = d / "transcripts" / f"{key}_solo_{k}.jsonl"
                recs = [json.loads(x) for x in f.read_text().splitlines()] if f.exists() else []
                mse.append(float(np.mean([(r["pred"] - r["y"]) ** 2 for r in recs]))
                           if recs else np.nan)
                cl = [json.loads(x) for x in (d / "transcripts" / f"{key}_calls_{k}.jsonl")
                      .read_text().splitlines() if x.strip()]
                calls.append(sum(1 for x in cl if "solo" in x and x["tool"] in SOLO_MODEL_TOOLS))
    return {"solo_mse": mse, "solo_model_calls": calls}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="llm_haiku_solo")
    ap.add_argument("--ks", nargs="+", type=int, default=[0, 10, 30, 60])
    ap.add_argument("--cs", nargs="+", default=["1.0", "0.5", "-0.5"])
    ap.add_argument("--late", type=int, default=20)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    cells = {}
    for c in args.cs:
        for K in args.ks:
            pre = f"{args.prefix}_K{K}"
            cells[(float(c), K)] = {**cell_stats(load_pairs(pre, c, "told"), args.late),
                                    **solo_stats(pre, c)}
    lines = ["| c | K | pairs | weight on partner | negative (< −0.1) | stacking tried | "
             "talkative pairs | msg / query | raw-feature share | joint models | solo MSE | "
             "solo model calls / twin | opening MSE | final MSE | stacking learner | cost |",
             "|" + "---|" * 16]
    for c in map(float, args.cs):
        for K in args.ks:
            s = cells[(c, K)]
            b = np.array(s["beta"], float)
            sm = np.array(s["solo_mse"], float)
            lines.append(
                f"| {c:g} | {K} | {len(s['msg'])} | {np.nanmean(b):.2f} ± "
                f"{np.nanstd(b) / np.sqrt(np.isfinite(b).sum()):.2f} | {int(np.sum(b < -0.1))} | "
                f"{sum(s['stack_try'])} | {int(np.sum(np.array(s['msg']) >= 0.5))} | "
                f"{np.mean(s['msg']):.2f} | {np.mean(s['raw']):.2f} | {sum(s['joint'])} | "
                f"{np.nanmean(sm) if np.isfinite(sm).any() else float('nan'):.2f} | "
                f"{np.mean(s['solo_model_calls']):.1f} | {np.nanmean(s['open']):.2f} | "
                f"{np.nanmean(s['mse']):.2f} | {np.nanmean(s['stack']):.2f} | ${s['cost']:.2f} |")
    lines += ["", f"Change from K = {args.ks[0]} (pair bootstrap, 95%):", "",
              "| c | K | weight on partner | final MSE | msg / query |", "|---|---|---|---|---|"]
    for c in map(float, args.cs):
        base = cells[(c, args.ks[0])]
        for K in args.ks[1:]:
            s, row = cells[(c, K)], []
            for key in ("beta", "mse", "msg"):
                a, b = np.array(base[key], float), np.array(s[key], float)
                if key != "msg":
                    a, b = a.reshape(-1, 2), b.reshape(-1, 2)
                lo, hi = boot_diff(a, b, rng)
                row.append(f"{np.nanmean(b) - np.nanmean(a):+.2f} [{lo:+.2f}, {hi:+.2f}]")
            lines.append(f"| {c:g} | {K} | " + " | ".join(row) + " |")
    # Solo error vs later reliance, pooled over K > 0 within each c (Spearman).
    lines += ["", "Solo MSE vs later weight on the partner (K > 0, Spearman across twins):", ""]
    for c in map(float, args.cs):
        sm = np.concatenate([cells[(c, K)]["solo_mse"] for K in args.ks if K > 0])
        b = np.concatenate([cells[(c, K)]["beta"] for K in args.ks if K > 0])
        ok = np.isfinite(sm) & np.isfinite(b)
        rk = lambda v: np.argsort(np.argsort(v))  # noqa: E731
        rho = float(np.corrcoef(rk(sm[ok]), rk(b[ok]))[0, 1]) if ok.sum() > 3 else float("nan")
        lines.append(f"- c = {c:g}: rho = {rho:+.2f} (n = {int(ok.sum())})")
    out = make_run_dir(ROOT / "results", "related_solo")
    (out / "table.md").write_text("\n".join(lines) + "\n")
    (out / "stats.json").write_text(json.dumps(
        {f"{c}_{K}": v for (c, K), v in cells.items()}, default=float, indent=1))
    plot_budget(cells, args.ks, [float(c) for c in args.cs], out / "solo.png",
                xlabel="solo queries before the partner (K)", log_x=False)
    print("\n".join(lines))
    print(out)


if __name__ == "__main__":
    main()
