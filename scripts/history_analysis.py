"""Past queries as data over a long horizon (LogBook 2026-11-14; configs/llm_haiku_sessions_clearhist*).

Twins with `history_tools` can add past queries (with the features their partner sent) to any
model. The question is whether a joint model on both feature sets gets built and used once enough
queries have accumulated. Per cell (lambda) and window of queries:
- MSE of the twins' openings and finals;
- online benchmarks refitted before each query on the revealed queries only (no training cases):
  "stack" = OLS of y on the two openings (what regress on own/partner_opening gives), "joint" =
  OLS on both twins' features (what a back-filled joint model gives), and the sum of openings
  (the population-optimal combination at lambda = 0);
- usage: twins that have created a model with a partner-feature column by the end of the window,
  twins that fitted such a model in the window, twins that regressed y on the partner's opening in
  the window ("stack"), and past-query add_case calls in the window;
- per twin: the first query at which it fits a partner-feature model (adoption), and the number of
  tool calls in training.

`--cells` compares training budgets (eq<n>, LogBook 2026-11-14 entrenchment arm): does seeing more
labelled cases before the queries change whether twins adopt the joint model?

Usage:
    python scripts/history_analysis.py [--prefix llm_haiku_sessions_clearhist100_own] [--window 25]
        [--lambdas 0.0 0.5] [--cells eq400]
Writes results/history/<timestamp>/{stats.json, table.md}.
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

from consensus.utils import make_run_dir  # noqa: E402


def read_jsonl(f: Path) -> list[dict]:
    return [json.loads(x) for x in f.read_text().splitlines() if x.strip()] if f.exists() else []


def online_ols(X: np.ndarray, y: np.ndarray, start: int) -> np.ndarray:
    """Prediction at each query t from OLS (with intercept) on queries < t; nan until there are
    `start` rows (and at least one more row than coefficients)."""
    Z = np.c_[np.ones(len(X)), X]
    out = np.full(len(y), np.nan)
    for t in range(max(start, Z.shape[1] + 1), len(y)):
        b = np.linalg.lstsq(Z[:t], y[:t], rcond=None)[0]
        out[t] = Z[t] @ b
    return out


def partner_pattern(labels: dict, k: str) -> re.Pattern:
    other = labels["B" if k == "A" else "A"]
    return re.compile(r"\b(" + "|".join(map(re.escape, other)) + r")\b")


def load(prefix: str, lam: str, cell: str = "eq400") -> list[dict]:
    pairs = []
    for d in sorted((ROOT / "results").glob(f"{prefix}_lam{lam}_al1.0_{cell}/20*")):
        sf = d / "summary.json"
        if not sf.exists():
            continue
        labels = yaml.safe_load((d / "config.yaml").read_text())["agent"]["labels"]
        for key, s in json.loads(sf.read_text()).items():
            tr = d / "transcripts"
            recs = sorted(read_jsonl(tr / f"{key}.jsonl"), key=lambda r: r["t"])
            pairs.append(dict(recs=recs, labels=labels, cost=s.get("cost_usd", 0.0),
                              calls={k: read_jsonl(tr / f"{key}_calls_{k}.jsonl") for k in "AB"}))
    return pairs


def usage(calls: list[dict], pat: re.Pattern, T: int) -> dict[str, np.ndarray]:
    """Per query: created a partner-feature model (cumulative), fitted one, regressed on the
    partner's opening, past add_case calls; and the number of training-phase calls."""
    joint_names: set[str] = set()
    created = np.zeros(T, bool)
    fitted = np.zeros(T, bool)
    stack = np.zeros(T, bool)
    past = np.zeros(T)
    n_train = sum(c["q"] < 0 for c in calls)
    for c in calls:
        q = c["q"]
        if q < 0 or q >= T:
            continue
        a = c.get("args", {})
        if c["tool"] == "regress" and "partner_opening" in (a.get("predictors") or []):
            stack[q] = True
        if c["tool"] == "new_model" and any(pat.search(str(x)) for x in a.get("columns") or []):
            joint_names.add(str(a.get("name")))
            created[q:] = True
        elif c["tool"] == "fit_model" and str(a.get("name")) in joint_names:
            fitted[q] = True
        elif c["tool"] == "add_case" and a.get("query") is not None:
            past[q] += 1
    return dict(created=created, fitted=fitted, stack=stack, past=past, n_train=n_train)


def cell(pairs: list[dict], window: int) -> dict:
    T = min(len(p["recs"]) for p in pairs)
    W = [(a, min(a + window, T)) for a in range(0, T, window)]
    acc = {k: [[] for _ in W] for k in ("open", "final", "stack", "joint", "sum")}
    use = {k: [[] for _ in W] for k in ("created", "fitted", "stack", "past")}
    first_fit, n_train = [], []
    for p in pairs:
        R = p["recs"][:T]
        y = np.array([r["y"] for r in R])
        oA, oB = (np.array([r["opening"][k] for r in R]) for k in "AB")
        fA, fB = (np.array([r["final"][k] for r in R]) for k in "AB")
        X = np.array([r["x"]["A"] + r["x"]["B"] for r in R])
        bench = {"stack": online_ols(np.c_[oA, oB], y, 5), "joint": online_ols(X, y, 0),
                 "sum": oA + oB}
        U = {k: usage(p["calls"][k], partner_pattern(p["labels"], k), T) for k in "AB"}
        for i, (a, b) in enumerate(W):
            s = slice(a, b)
            acc["open"][i] += list((oA[s] - y[s]) ** 2) + list((oB[s] - y[s]) ** 2)
            acc["final"][i] += list((fA[s] - y[s]) ** 2) + list((fB[s] - y[s]) ** 2)
            for k, v in bench.items():
                acc[k][i] += list((v[s] - y[s]) ** 2)
            for k in "AB":
                use["created"][i].append(bool(U[k]["created"][b - 1]))
                use["fitted"][i].append(bool(U[k]["fitted"][s].any()))
                use["stack"][i].append(bool(U[k]["stack"][s].any()))
                use["past"][i].append(float(U[k]["past"][s].sum()))
        for k in "AB":
            f = np.flatnonzero(U[k]["fitted"])
            first_fit.append(int(f[0]) + 1 if len(f) else None)
            n_train.append(U[k]["n_train"])
    out = {k: [float(np.nanmean(v)) if v else float("nan") for v in vals] for k, vals in acc.items()}
    out["twins_with_joint_model"] = [int(sum(v)) for v in use["created"]]
    out["twins_fitting_joint_model"] = [int(sum(v)) for v in use["fitted"]]
    out["twins_stacking"] = [int(sum(v)) for v in use["stack"]]
    out["past_add_case"] = [int(sum(v)) for v in use["past"]]
    out["first_fit_query"] = first_fit
    out["training_calls_median"] = float(np.median(n_train))
    out.update(windows=W, n_pairs=len(pairs), cost=float(sum(p["cost"] for p in pairs)))
    return out


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", default="llm_haiku_sessions_clearhist100_own")
    ap.add_argument("--lambdas", nargs="+", default=["0.0", "0.5"])
    ap.add_argument("--window", type=int, default=25)
    ap.add_argument("--cells", nargs="+", default=["eq400"], help="training budgets, e.g. eq25 eq400")
    args = ap.parse_args(argv)
    cells = {f"{lam} {c}": cell(P, args.window) for lam in args.lambdas for c in args.cells
             if (P := load(args.prefix, lam, c))}
    fmt = lambda v: " / ".join(f"{x:.2f}" for x in v)  # noqa: E731
    ints = lambda v: " / ".join(str(x) for x in v)  # noqa: E731
    lines = []
    for lam, c in cells.items():
        win = ", ".join(f"{a + 1}–{b}" for a, b in c["windows"])
        lines += [f"\nλ, cell = {lam}: {c['n_pairs']} pairs, cost ${c['cost']:.2f}; windows {win}", "",
                  "| per window | values |", "|---|---|",
                  f"| MSE openings | {fmt(c['open'])} |", f"| MSE finals | {fmt(c['final'])} |",
                  f"| online OLS on the two openings | {fmt(c['stack'])} |",
                  f"| online joint OLS on both feature sets | {fmt(c['joint'])} |",
                  f"| sum of openings | {fmt(c['sum'])} |",
                  f"| twins with a partner-feature model (of {2 * c['n_pairs']}) "
                  f"| {ints(c['twins_with_joint_model'])} |",
                  f"| twins fitting it in the window | {ints(c['twins_fitting_joint_model'])} |",
                  f"| twins regressing on the partner's opening | {ints(c['twins_stacking'])} |",
                  f"| first query fitting a joint model (per twin) "
                  f"| {', '.join('-' if x is None else str(x) for x in c['first_fit_query'])} |",
                  f"| median training-phase tool calls | {c['training_calls_median']:.0f} |",
                  f"| past-query add_case calls | {ints(c['past_add_case'])} |"]
    run_out = make_run_dir(str(ROOT / "results"), "history")
    (run_out / "stats.json").write_text(json.dumps(cells, indent=2))
    table = "\n".join(lines)
    (run_out / "table.md").write_text(table + "\n")
    print(table)
    print(f"saved to {run_out}")
    return run_out


if __name__ == "__main__":
    main()
