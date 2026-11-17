"""Rigged first partner queries (LogBook note 2026-11-17): does an early impression of the
partner stick? Each arm replaces partner queries 1..10 with selected cases; queries 11..100,
training and solo queries are identical to the control (the d = 3 long runs, same seeds and pair
indices), so twins are compared with the same twin in the control (paired by c, pair, twin).

Per twin and window of partner queries (default 1-10, 11-40, 41-70, 71-100):
- msgs: messages the twin sends per query;
- asked: share of queries in which the twin asks the partner for its features;
- raw: share of queries in which the twin sends its own feature values;
- built: has a joint model (a model instance with a partner-feature column) by the window's end;
- fitting: share of the window's queries with fit_model on a joint model;
- mse: MSE of the twin's finals against its own y;
- beta_rel: coefficient on the partner's opening (final on both openings) over the window,
  divided by c (1 = the stacking optimum; pooling c = 0.5 and -0.5 needs the sign removed).
Differences arm - control: mean over twins with a 95% bootstrap interval over pairs.

Usage:
    python scripts/rig_analysis.py [--arms llm_haiku_rig_PlowElow ...] [--control
        llm_haiku_solo_K10_T100] [--cs 0.5 -0.5]
Writes results/rig/<timestamp>/{table.md, stats.json, rig.png}.
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

from consensus.plotting_related import plot_rig  # noqa: E402
from consensus.utils import make_run_dir  # noqa: E402
from related_analysis import load_pairs, ols  # noqa: E402

NUM = r"-?\d*\.?\d+"
ASK = re.compile(r"(send|share|give|tell)\b.{0,60}\b(features?|[ab]1\b|[ab]1\.\.|values)", re.I)
KEYS = ("msgs", "asked", "raw", "built", "fitting", "mse", "beta_rel")


def twin_windows(p: dict, k: str, edges: list[int], c: float) -> dict:
    """Per-window measures for twin k of pair p (see the module docstring)."""
    o = "B" if k == "A" else "A"
    recs = p["recs"]
    own_lab, par_lab = p["labels"][k], p["labels"][o]
    models = recs[-1]["models"][k]
    joint = {n: m["created_q"] for n, m in models.items()
             if any(any(lab in c for lab in par_lab) for c in m["columns"])}
    fit_q = {c["q"] for c in p["calls"][k] if c["tool"] == "fit_model" and c["q"] >= 0
             and c["args"].get("name") in joint}
    # Raw features in a message: "a1=0.3" / "a1: 0.3", or a tuple of d numbers.
    feat = re.compile(r"\b(" + "|".join(map(re.escape, own_lab)) + r")\s*[=:]\s*-?\d"
                      + r"|\(\s*" + r"\s*,\s*".join([NUM] * len(own_lab)) + r"\s*\)")
    mine = [[m["text"] for m in r["messages"] if m["from"] == k] for r in recs]
    msgs = np.array([len(t) for t in mine], float)
    asked = np.array([any(ASK.search(x) for x in t) for t in mine], float)
    raw = np.array([any(feat.search(x) for x in t) for t in mine], float)
    own = np.array([r["opening"][k] for r in recs], float)
    par = np.array([r["opening"][o] for r in recs], float)
    fin = np.array([r["final"][k] for r in recs], float)
    y = np.array([r["y_own"][k] for r in recs])
    out = {key: [] for key in KEYS}
    for a, b in zip(edges, edges[1:]):
        w = slice(a, b)
        out["msgs"].append(float(msgs[w].mean()))
        out["asked"].append(float(asked[w].mean()))
        out["raw"].append(float(raw[w].mean()))
        out["built"].append(float(any(cq < b for cq in joint.values())))
        out["fitting"].append(len([q for q in fit_q if a <= q < b]) / (b - a))
        out["mse"].append(float(np.nanmean((fin[w] - y[w]) ** 2)))
        out["beta_rel"].append(float(ols(np.column_stack([own[w], par[w]]), fin[w])[2]) / c)
    return out


def pair_key(p: dict) -> int:
    """Pair index r from the run key ('<folder>/run<r>')."""
    return int(p["run"].rsplit("run", 1)[1])


def collect(prefix: str, cs: list[str], edges: list[int]) -> dict:
    """{(c, r): {"A": windows, "B": windows, "x_late": partner-query features from row 10}}"""
    out = {}
    for c in cs:
        for p in load_pairs(prefix, c, "told"):
            out[(c, pair_key(p))] = {
                **{k: twin_windows(p, k, edges, float(c)) for k in "AB"},
                "x_late": [r["x"]["A"] for r in p["recs"][edges[1]:]],
                "cost": p["cost"]}
    return out


def boot_ci(d: np.ndarray, groups: np.ndarray, n: int = 2000, seed: int = 0) -> tuple:
    """95% interval of the mean of d, resampling groups (pairs) with replacement."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    means = []
    for _ in range(n):
        pick = rng.choice(ug, len(ug))
        means.append(np.nanmean(np.concatenate([d[groups == g] for g in pick])))
    return float(np.nanpercentile(means, 2.5)), float(np.nanpercentile(means, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", default=["llm_haiku_rig_PlowElow",
                                                   "llm_haiku_rig_PlowEhigh",
                                                   "llm_haiku_rig_PhighElow"])
    ap.add_argument("--control", default="llm_haiku_solo_K10_T100")
    ap.add_argument("--cs", nargs="+", default=["0.5", "-0.5"])
    ap.add_argument("--edges", nargs="+", type=int, default=[0, 10, 40, 70, 100])
    args = ap.parse_args()
    out = make_run_dir(ROOT / "results", "rig")
    edges = args.edges
    wins = [f"{a + 1}–{b}" for a, b in zip(edges, edges[1:])]
    ctrl = collect(args.control, args.cs, edges)
    stats = {"control": {key: np.nanmean([[ctrl[g][k][key] for k in "AB"] for g in ctrl],
                                         axis=(0, 1)).tolist() for key in KEYS}}
    lines = []
    for arm in args.arms:
        data = collect(arm, args.cs, edges)
        shared = sorted(set(data) & set(ctrl))
        # Queries 11.. must be identical to the control's (the design's pairing).
        same = sum(np.allclose(data[g]["x_late"], ctrl[g]["x_late"][:len(data[g]["x_late"])])
                   for g in shared)
        short = [g for g in shared if len(data[g]["x_late"]) < len(ctrl[g]["x_late"])]
        cost = sum(data[g]["cost"] for g in data)
        lines += [f"### {arm} ({len(shared)} pairs matched to the control, {same} with identical "
                  f"queries {edges[1] + 1}–{edges[-1]}; incomplete runs {short}; ${cost:.2f})", "",
                  "| measure | " + " | ".join(f"{w}: control → arm (diff [95% CI])" for w in wins)
                  + " |", "|" + "---|" * (len(wins) + 1)]
        st = {}
        for key in KEYS:
            cells, means = [], []
            for i in range(len(wins)):
                a = np.array([data[g][k][key][i] for g in shared for k in "AB"])
                b = np.array([ctrl[g][k][key][i] for g in shared for k in "AB"])
                grp = np.array([hash(g) for g in shared for _ in "AB"])
                d = a - b
                lo, hi = boot_ci(d, grp)
                cells.append(f"{np.nanmean(b):.2f} → {np.nanmean(a):.2f} "
                             f"({np.nanmean(d):+.2f} [{lo:+.2f}, {hi:+.2f}])")
                means.append(float(np.nanmean(a)))
            st[key] = means
            lines.append(f"| {key} | " + " | ".join(cells) + " |")
        lines.append("")
        stats[arm] = st
    (out / "table.md").write_text("\n".join(lines))
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    plot_rig(stats, ["control", *args.arms], wins, out / "rig.png")
    print("\n".join(lines))
    print(out)


if __name__ == "__main__":
    main()
