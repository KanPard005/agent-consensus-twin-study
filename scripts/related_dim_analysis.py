"""Feature dimension in the long runs (LogBook 2026-11-17): K = 10 solo queries, T = 100 partner
queries, d = 3 vs d = 20 features per twin (weights are unit vectors in both, so the population
references are the same: solo 1.25, stacking 1.25 - c^2, joint 0.25).

Per cell (d, c), over twins, partner phase:
- MSE of openings, and of the best solo prediction <x^own, true own-target weights> (the logged
  training rows are truncated, so the true weights stand in for the twin's OLS fit; OLS on 400
  cases adds about 1.25 d / 400, i.e. 0.01 at d = 3 and 0.06 at d = 20);
- gap: mean (opening - best solo prediction)^2, the twin's estimation plus arithmetic error, and
  the share of queries in which the twin called fit_model (the tool that computes a prediction
  instead of the twin doing it by hand);
- messages per query, share of talkative twins (>= 0.5 per query, counting both directions);
- late (last 40 queries) MSE of finals, of the online stacking learner, and of an online joint
  learner (OLS of y_own on the best solo prediction and the partner's raw features, fitted on
  past partner queries): the finite-T value of the raw-feature route at this d;
- beta: late coefficient on the partner's opening (final regressed on both openings);
- built: share of twins that created a model instance with a partner-feature column.

Usage:
    python scripts/related_dim_analysis.py [--prefixes llm_haiku_solo_K10_T100
        llm_haiku_solo_K10_T100_d20] [--dims 3 20] [--late 40]
Writes results/related_dim/<timestamp>/{table.md, stats.json, dims.png}.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from consensus.plotting_related import plot_dims  # noqa: E402
from consensus.utils import make_run_dir  # noqa: E402
from related_analysis import load_pairs, ols, online_stack  # noqa: E402


def best_solo(env: dict, k: str, xo: np.ndarray) -> np.ndarray:
    """<x^own, true weights of twin k's target on its own block>: w_A for A, v_B for B."""
    return xo @ np.array(env["w_A"] if k == "A" else env["v_B"])


def online_joint(base: np.ndarray, xp: np.ndarray, y: np.ndarray, start: int = 5) -> np.ndarray:
    """Prediction of y_t by OLS of y on (base, partner raw features) over partner queries < t.
    Needs more past queries than columns, so it starts late at large d; ridge-free on purpose
    (the twins' own regression tool is plain OLS)."""
    pred = np.full(len(y), np.nan)
    X = np.column_stack([base, xp])
    for t in range(start, len(y)):
        b = ols(X[:t], y[:t])
        if np.isfinite(b).all():
            pred[t] = b[0] + X[t] @ b[1:]
    return pred


def twin_measures(p: dict, k: str, late: int) -> dict:
    o = "B" if k == "A" else "A"
    recs = p["recs"]
    T = len(recs)
    y = np.array([r["y_own"][k] for r in recs])
    own = np.array([r["opening"][k] for r in recs], float)
    par = np.array([r["opening"][o] for r in recs], float)
    fin = np.array([r["final"][k] for r in recs], float)
    xo = np.array([r["x"][k] for r in recs])
    xp = np.array([r["x"][o] for r in recs])
    exact = best_solo(p["env"], k, xo)
    fit_q = {c["q"] for c in p["calls"][k] if c["tool"] == "fit_model" and c["q"] >= 0
             and "solo" not in c}
    par_lab = set(p["labels"][o])
    built = any(c["tool"] == "new_model" and par_lab & set(c["args"].get("columns", []))
                for c in p["calls"][k])
    msgs = sum(len(r.get("messages", [])) for r in recs)
    L = slice(T - late, T)
    stack = online_stack(own, par, y)
    joint = online_joint(exact, xp, y)
    bb = ols(np.column_stack([own[L], par[L]]), fin[L])
    m = lambda a: float(np.nanmean((a - y) ** 2))  # noqa: E731
    mL = lambda a: float(np.nanmean((a[L] - y[L]) ** 2))  # noqa: E731
    return dict(open_mse=m(own), best_solo_mse=m(exact), gap=float(np.nanmean((own - exact) ** 2)),
                fit_share=len(fit_q) / T, msgs_per_q=msgs / T, final_late=mL(fin),
                stack_late=mL(stack), joint_late=mL(joint), beta=float(bb[2]), built=float(built))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefixes", nargs="+",
                    default=["llm_haiku_solo_K10_T100", "llm_haiku_solo_K10_T100_d20"])
    ap.add_argument("--dims", nargs="+", type=int, default=[3, 20])
    ap.add_argument("--cs", nargs="+", default=["1.0", "0.5", "-0.5"])
    ap.add_argument("--late", type=int, default=40)
    args = ap.parse_args()
    out = make_run_dir(ROOT / "results", "related_dim")
    keys = ["open_mse", "best_solo_mse", "gap", "fit_share", "msgs_per_q", "final_late",
            "stack_late", "joint_late", "beta", "built"]
    stats, lines = {}, []
    lines.append("| d | c | twins | $/pair | " + " | ".join(keys) + " | talkative |")
    lines.append("|" + "---|" * (len(keys) + 5))
    for d, pre in zip(args.dims, args.prefixes):
        for c in args.cs:
            pairs = load_pairs(pre, c, "told")
            tw = [twin_measures(p, k, args.late) for p in pairs for k in "AB"]
            # Talkative per pair (messages count for both twins), as in earlier entries.
            talk = float(np.mean([t["msgs_per_q"] >= 0.5 for t in tw]))
            cell = {kk: float(np.nanmean([t[kk] for t in tw])) for kk in keys}
            cell.update(twins=len(tw), talkative=talk,
                        cost_per_pair=float(np.mean([p["cost"] for p in pairs])))
            stats[f"d{d}_c{c}"] = dict(cell=cell, twins=tw)
            lines.append(f"| {d} | {c} | {len(tw)} | {cell['cost_per_pair']:.2f} | "
                         + " | ".join(f"{cell[kk]:.2f}" for kk in keys) + f" | {talk:.2f} |")
    plot_dims({(d, float(c)): stats[f"d{d}_c{c}"]["cell"] for d in args.dims for c in args.cs},
              args.dims, [float(c) for c in args.cs], out / "dims.png")
    (out / "table.md").write_text("\n".join(lines) + "\n")
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print("\n".join(lines))
    print(out)


if __name__ == "__main__":
    main()
