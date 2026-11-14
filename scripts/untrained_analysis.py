"""Twins without a training phase (LogBook 2026-11-14): both start with no labelled cases and
learn only from the shared queries, talking from query 1 (configs/llm_haiku_sessions_untrained_*).

Benchmarks are rational online learners on the same cases, each refitted before every query on
the queries already revealed (ridge with penalty `RIDGE` on the slopes, intercept unpenalized):
- "own OLS": the twin's own features only (what its opening could be);
- "convex": the best convex blend (1 - w) own + w partner of the two own-OLS predictions, with
  w fitted on the past queries (what talking could add by blending);
- "joint OLS": all 2d features of both twins (what the pair could reach by sharing features).

Per cell (rule, lambda, alpha) and window of queries, by role (strong / weak twin):
- MSE of the openings and of the finals (or the agreed value), against the benchmarks;
- the opening's fidelity to the own-OLS prediction: the slope of opening on own-OLS (1 = follows
  its regression, < 1 = shrinks it) and the share of openings within 0.1 of it;
- WOA on the partner against the population best convex weight (`asymmetric_limits`);
- messages per query, and the share of queries in which a twin's raw feature values are sent.

Usage:
    python scripts/untrained_analysis.py [--lambdas 0.0 0.5] [--alphas 0.0 0.5 1.0]
Writes results/untrained/<timestamp>/{stats.json, table.md, untrained_curves.png}.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consensus.plotting_asym import plot_untrained_curves  # noqa: E402
from consensus.sessions import weight_of_advice  # noqa: E402
from consensus.theory import asymmetric_limits  # noqa: E402
from consensus.utils import make_run_dir  # noqa: E402

RULES = {"own": "individual", "cons": "consensus"}
WINDOWS = ((0, 15), (15, 30), (30, 45), (45, 60))
RIDGE = 1.0
NUM = r"-?\d+\.\d+"


def ridge_predict(X: np.ndarray, y: np.ndarray, x: np.ndarray) -> float:
    """Ridge prediction at x from the rows (X, y), intercept unpenalized; 0 with no data."""
    if len(y) == 0:
        return 0.0
    Z = np.c_[np.ones(len(X)), X]
    P = RIDGE * np.eye(Z.shape[1])
    P[0, 0] = 0.0
    theta = np.linalg.solve(Z.T @ Z + P + 1e-9 * np.eye(Z.shape[1]), Z.T @ y)
    return float(np.r_[1.0, x] @ theta)


def online_benchmarks(recs: list[dict], S: str, W: str) -> dict[str, np.ndarray]:
    """Per query: own-OLS predictions of both twins, their fitted convex blends, joint OLS."""
    xS = np.array([s["x"][S] for s in recs])
    xW = np.array([s["x"][W] for s in recs])
    y = np.array([s["y"] for s in recs])
    T = len(y)
    out = {k: np.zeros(T) for k in ("ols_S", "ols_W", "conv_S", "conv_W", "joint")}
    for t in range(T):
        out["ols_S"][t] = ridge_predict(xS[:t], y[:t], xS[t])
        out["ols_W"][t] = ridge_predict(xW[:t], y[:t], xW[t])
        out["joint"][t] = ridge_predict(np.c_[xS[:t], xW[:t]], y[:t], np.r_[xS[t], xW[t]])
        for own, par, key in (("ols_S", "ols_W", "conv_S"), ("ols_W", "ols_S", "conv_W")):
            d = out[par][:t] - out[own][:t]
            w = float(np.clip(d @ (y[:t] - out[own][:t]) / (d @ d), 0, 1)) if t > 2 and d @ d > 0 else 0.5
            out[key][t] = out[own][t] + w * (out[par][t] - out[own][t])
    return out


def load_pairs(prefix: str, lam: str, alpha: str) -> list[dict]:
    pairs = []
    for d in sorted((ROOT / "results" / f"{prefix}_lam{lam}_al{alpha}_eq0").glob("20*")):
        sf = d / "summary.json"
        if not sf.exists():
            continue
        weak = yaml.safe_load((d / "config.yaml").read_text())["env"].get("weak", "B")
        for key, s in json.loads(sf.read_text()).items():
            f = d / "transcripts" / f"{key}.jsonl"
            recs = sorted((json.loads(x) for x in f.read_text().splitlines() if x.strip()),
                          key=lambda r: r["t"])
            pairs.append(dict(recs=recs, weak=weak, cost=s.get("cost_usd", 0.0)))
    return pairs


def shares_features(text: str, x: list[float]) -> bool:
    """Whether a message quotes at least two of the sender's raw feature values (3 decimals)."""
    vals = {f"{v:.3f}" for v in x} | {f"{v:.2f}" for v in x}
    return sum(v in set(re.findall(NUM, text)) for v in vals) >= 2


def nanmean(v) -> float:
    v = [x for x in v if x == x]
    return float(np.mean(v)) if v else float("nan")


def cell_stats(pairs: list[dict], rule: str, lam: float, alpha: float) -> dict:
    """Window-by-window statistics for one cell (see the module docstring)."""
    nW = len(WINDOWS)
    acc = {k: [[] for _ in range(nW)] for k in (
        "open_S", "open_W", "fin_S", "fin_W", "ols_S", "ols_W", "conv_S", "conv_W", "joint",
        "woa_S", "woa_W", "msgs", "feat")}
    fid = {"S": [[], []], "W": [[], []]}  # (opening, own OLS) after the first window
    for p in pairs:
        W = p["weak"]
        S = "B" if W == "A" else "A"
        recs = p["recs"]
        b = online_benchmarks(recs, S, W)
        for t, s in enumerate(recs):
            i = next(j for j, (a, e) in enumerate(WINDOWS) if a <= s["t"] < e)
            y = s["y"]
            msgs = s.get("messages", [])
            acc["msgs"][i].append(len(msgs))
            acc["feat"][i].append(any(shares_features(m["text"], s["x"][m["from"]]) for m in msgs))
            for key in ("ols_S", "ols_W", "conv_S", "conv_W", "joint"):
                acc[key][i].append((b[key][t] - y) ** 2)
            if s["failed"]:
                continue
            oS, oW = s["opening"][S], s["opening"][W]
            fS, fW = ((s["agreed"], s["agreed"]) if rule == "consensus"
                      else (s["final"][S], s["final"][W]))
            acc["open_S"][i].append((oS - y) ** 2)
            acc["open_W"][i].append((oW - y) ** 2)
            acc["fin_S"][i].append((fS - y) ** 2)
            acc["fin_W"][i].append((fW - y) ** 2)
            acc["woa_S"][i].append(weight_of_advice(oS, oW, fS))
            acc["woa_W"][i].append(weight_of_advice(oW, oS, fW))
            if s["t"] >= WINDOWS[0][1]:
                for role, o, key in (("S", oS, "ols_S"), ("W", oW, "ols_W")):
                    fid[role][0].append(o)
                    fid[role][1].append(b[key][t])
    lim = asymmetric_limits(lam, alpha)
    out = {k: [nanmean(v) for v in vals] for k, vals in acc.items()}
    for role in ("S", "W"):
        o, r = map(np.array, fid[role])
        out[f"slope_{role}"] = float(o @ r / (r @ r)) if len(r) else float("nan")
        out[f"near_{role}"] = float(np.mean(np.abs(o - r) <= 0.1)) if len(r) else float("nan")
    out.update(n_pairs=len(pairs), best_S=lim["convex_strong"], best_W=lim["convex_weak"],
               cost=float(sum(p["cost"] for p in pairs)))
    return out


def fmt(v) -> str:
    return " / ".join(f"{x:.2f}" for x in v)


def main(argv: list[str] | None = None) -> Path:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lambdas", nargs="+", default=["0.0", "0.5"])
    p.add_argument("--alphas", nargs="+", default=["0.0", "0.5", "1.0"])
    p.add_argument("--prefix", default="llm_haiku_sessions_untrained")
    args = p.parse_args(argv)
    out: dict = {"cells": {}, "cost": 0.0, "windows": WINDOWS}
    for tag, rule in RULES.items():
        for lam in args.lambdas:
            for al in args.alphas:
                pairs = load_pairs(f"{args.prefix}_{tag}", lam, al)
                if pairs:
                    st = cell_stats(pairs, rule, float(lam), float(al))
                    out["cells"][f"{rule}_{lam}_{al}"] = st
                    out["cost"] += st["cost"]
    win = ", ".join(f"{a + 1}–{b}" for a, b in WINDOWS)
    lines = [f"Windows of queries: {win}. MSE values per window; 'own OLS' etc. are the online "
             "benchmarks on the same cases.", ""]
    lines += ["| rule | λ | α | pairs | opening MSE strong | own OLS strong | final strong "
              "| opening MSE weak | own OLS weak | final weak | convex (strong) | joint OLS "
              "| opening slope on own OLS strong / weak | within 0.1 strong / weak |",
              "|" + "---|" * 14]
    for k, s in out["cells"].items():
        rule, lam, al = k.split("_")
        lines.append(
            f"| {rule} | {lam} | {al} | {s['n_pairs']} | {fmt(s['open_S'])} | {fmt(s['ols_S'])} "
            f"| {fmt(s['fin_S'])} | {fmt(s['open_W'])} | {fmt(s['ols_W'])} | {fmt(s['fin_W'])} "
            f"| {fmt(s['conv_S'])} | {fmt(s['joint'])} "
            f"| {s['slope_S']:.2f} / {s['slope_W']:.2f} | {s['near_S']:.0%} / {s['near_W']:.0%} |")
    lines += ["", "| rule | λ | α | WOA strong (best) | WOA weak (best) | messages per query "
              "| queries with raw features sent |", "|" + "---|" * 7]
    for k, s in out["cells"].items():
        rule, lam, al = k.split("_")
        lines.append(
            f"| {rule} | {lam} | {al} | {fmt(s['woa_S'])} ({s['best_S']:.2f}) "
            f"| {fmt(s['woa_W'])} ({s['best_W']:.2f}) | {fmt(s['msgs'])} | {fmt(s['feat'])} |")
    lines.append(f"\ntotal cost ${out['cost']:.2f}")
    run_out = make_run_dir(str(ROOT / "results"), "untrained")
    (run_out / "stats.json").write_text(json.dumps(out, indent=2))
    table = "\n".join(lines)
    (run_out / "table.md").write_text(table + "\n")
    if out["cells"]:
        plot_untrained_curves(out["cells"], args.lambdas, args.alphas, WINDOWS,
                              run_out / "untrained_curves.png")
    print(table)
    print(f"saved to {run_out}")
    return run_out


if __name__ == "__main__":
    main()
