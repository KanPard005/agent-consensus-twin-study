"""Long runs (LogBook 2026-11-17): K solo queries, then T = 100 partner queries. Does long
repeated interaction lead twins to build and use joint models on both feature sets?

Per window of partner queries (default 20), per c, over twins: share with a joint model (a model
instance with a partner-feature column) created by the end of the window; share fitting a joint
model (fit_model on it) in at least half the window's queries; share of queries in which the twin
sends its raw features; MSE of the finals and of the online stacking learner. Also the first
query of sustained joint-model use per twin.

Usage:
    python scripts/related_long_analysis.py [--prefix llm_haiku_solo_K10_T100] [--window 20]
Writes results/related_long/<timestamp>/{table.md, stats.json, over_time.png}.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from consensus.plotting_related import plot_over_time  # noqa: E402
from consensus.utils import make_run_dir  # noqa: E402
from related_analysis import load_pairs, online_stack  # noqa: E402

NUM = r"-?\d*\.?\d+"


def twin_windows(p: dict, k: str, edges: list[int]) -> dict:
    """Per-window measures for twin k of pair p."""
    o = "B" if k == "A" else "A"
    recs = p["recs"]
    own_lab, par_lab = p["labels"][k], p["labels"][o]
    # Joint models: model instances with a column naming a partner feature; created_q from the
    # final record (0-based partner query, -1 if created before the partner phase).
    models = recs[-1]["models"][k]
    joint = {n: m["created_q"] for n, m in models.items()
             if any(any(lab in c for lab in par_lab) for c in m["columns"])}
    fit_q = {c["q"] for c in p["calls"][k] if c["tool"] == "fit_model" and c["q"] >= 0
             and c["args"].get("name") in joint}
    feat = re.compile(r"\b(" + "|".join(map(re.escape, own_lab)) + r")\s*[=:]\s*-?\d"
                      + r"|\(\s*" + r"\s*,\s*".join([NUM] * len(own_lab)) + r"\s*\)")
    sent = np.array([any(m["from"] == k and feat.search(m["text"]) for m in r["messages"])
                     for r in recs])
    own = np.array([r["opening"][k] for r in recs])
    par = np.array([r["opening"][o] for r in recs])
    fin = np.array([r["final"][k] for r in recs])
    y = np.array([r["y_own"][k] for r in recs])
    stack = online_stack(own, par, y)
    out = {key: [] for key in ("built", "fitting", "raw", "mse", "stack")}
    for a, b in zip(edges, edges[1:]):
        out["built"].append(any(cq < b for cq in joint.values()))
        out["fitting"].append(len([q for q in fit_q if a <= q < b]) >= (b - a) / 2)
        out["raw"].append(float(sent[a:b].mean()))
        out["mse"].append(float(np.nanmean((fin[a:b] - y[a:b]) ** 2)))
        out["stack"].append(float(np.nanmean((stack[a:b] - y[a:b]) ** 2)))
    # First query of sustained use: start of the first run of 10 consecutive queries with a fit.
    first = next((q for q in sorted(fit_q) if all(q + j in fit_q for j in range(10))), None)
    out["first_sustained"] = None if first is None else first + 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="llm_haiku_solo_K10_T100")
    ap.add_argument("--cs", nargs="+", default=["1.0", "0.5", "-0.5"])
    ap.add_argument("--window", type=int, default=20)
    args = ap.parse_args()
    cells, lines = {}, []
    for c in args.cs:
        pairs = load_pairs(args.prefix, c, "told")
        T = len(pairs[0]["recs"])
        edges = list(range(0, T + 1, args.window))
        tw = [twin_windows(p, k, edges) for p in pairs for k in "AB"]
        cell = {key: [float(np.mean([t[key][w] for t in tw])) for w in range(len(edges) - 1)]
                for key in ("built", "fitting", "raw", "mse", "stack")}
        cell["first_sustained"] = [t["first_sustained"] for t in tw]
        cell["n_twins"], cell["cost"] = len(tw), sum(p["cost"] for p in pairs)
        cells[float(c)] = cell
        wins = [f"{a + 1}–{b}" for a, b in zip(edges, edges[1:])]
        lines += [f"### c = {float(c):g} ({len(tw)} twins, ${cell['cost']:.2f})", "",
                  "| measure | " + " | ".join(wins) + " |", "|---|" + "---|" * len(wins)]
        for key, lab in (("built", "joint model built"), ("fitting", "fitting it (≥ half)"),
                         ("raw", "queries sending raw features"), ("mse", "MSE of finals"),
                         ("stack", "stacking learner MSE")):
            lines.append(f"| {lab} | " + " | ".join(f"{v:.2f}" for v in cell[key]) + " |")
        fs = sorted(q for q in cell["first_sustained"] if q is not None)
        lines += ["", f"Sustained joint-model use (10 consecutive fits): {len(fs)} of {len(tw)} "
                  f"twins, starting at partner queries {fs}.", ""]
    out = make_run_dir(ROOT / "results", "related_long")
    (out / "table.md").write_text("\n".join(lines) + "\n")
    (out / "stats.json").write_text(json.dumps({str(c): v for c, v in cells.items()},
                                               default=float, indent=1))
    plot_over_time(cells, [float(c) for c in args.cs], edges, out / "over_time.png")
    print("\n".join(lines))
    print(out)


if __name__ == "__main__":
    main()
