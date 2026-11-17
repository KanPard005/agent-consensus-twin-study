"""Alignment sweep: twins predicting related targets (LogBook note 2026-11-16).

Per cell (c, prompt arm), per twin, over the late queries (from --late on):
- beta: the coefficient on the partner's opening when the final is regressed on (1, own
  opening, partner opening); the best stacking coefficient is c, averaging gives 0.5;
- own_coef: the coefficient on the own opening (1 for stacking, 0.5 for averaging);
- WOA, against the best convex weight c/2;
- MSE of the finals and of the openings against the twin's own y, per window, against the
  population references (`consensus.theory.related_limits`) and an online stacking learner
  (y_own on the two openings over past queries);
- moves away: share of late queries (with a gap > 0.2) where the final moves away from the
  partner's opening (a negative weight);
- detection: the share of twins whose messages say the targets differ (regex, read the hits);
- raw features sent and joint models (a model with a partner-feature column).

Usage:
    python scripts/related_analysis.py [--cs 1.0 0.5 0.0 -0.5] [--arms untold told] [--late 20]
Writes results/related/<timestamp>/{stats.json, table.md, related.png, detection.md}.
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

from consensus.plotting_related import plot_related  # noqa: E402
from consensus.theory import related_limits  # noqa: E402
from consensus.utils import make_run_dir  # noqa: E402

DIFFERENT = re.compile(
    r"(different|not the same|another|its own|my own|your own|separate|distinct)\s+"
    r"(target|quantity|quantities|y\b|label|variable|outcome|objective|thing)"
    r"|(target|quantity|quantities|y|ys|labels?|outcomes?)\s+(?:is|are|seem|seems|might be|may be"
    r"|must be)\s+(?:a\s+)?(?:different|not the same|unrelated|related)"
    r"|predict(?:ing|s)?\s+(?:something|a)\s+different|not\s+predicting\s+the\s+same"
    r"|(?:true\s+)?y\s*(?:values|'s)?\s+(?:don't|do not|doesn't)\s+match",
    re.IGNORECASE)
WINDOWS = 4


def load_pairs(prefix: str, c: str, arm: str) -> list[dict]:
    pairs = []
    for d in sorted((ROOT / "results" / f"{prefix}_c{c}_{arm}").glob("20*")):
        sf, ef = d / "summary.json", d / "env.json"
        if not sf.exists() or not ef.exists():
            continue
        env = json.loads(ef.read_text())
        labels = yaml.safe_load((d / "config.yaml").read_text())["agent"]["labels"]
        for key, s in json.loads(sf.read_text()).items():
            f = d / "transcripts" / f"{key}.jsonl"
            recs = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
            recs.sort(key=lambda r: r["t"])
            calls = {k: [json.loads(x) for x in (d / "transcripts" / f"{key}_calls_{k}.jsonl")
                         .read_text().splitlines() if x.strip()] for k in "AB"}
            pairs.append(dict(recs=recs, env=env, labels=labels, cost=s.get("cost_usd", 0.0),
                              errors=s.get("errors", {}), run=f"{d.name}/{key}", calls=calls))
    return pairs


def ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least squares with an intercept; nan if too few rows."""
    ok = np.isfinite(X).all(1) & np.isfinite(y)
    if ok.sum() < X.shape[1] + 3:
        return np.full(X.shape[1] + 1, np.nan)
    A = np.column_stack([np.ones(ok.sum()), X[ok]])
    return np.linalg.lstsq(A, y[ok], rcond=None)[0]


def online_stack(own: np.ndarray, partner: np.ndarray, y: np.ndarray, start: int = 5):
    """Prediction of y_t from (own_t, partner_t) by least squares over queries < t."""
    pred = np.full(len(y), np.nan)
    for t in range(start, len(y)):
        b = ols(np.column_stack([own[:t], partner[:t]]), y[:t])
        if np.isfinite(b).all():
            pred[t] = b[0] + b[1] * own[t] + b[2] * partner[t]
    return pred


def twin_stats(p: dict, k: str, late: int) -> dict:
    o = "B" if k == "A" else "A"
    recs = p["recs"]
    own = np.array([r["opening"][k] for r in recs])
    par = np.array([r["opening"][o] for r in recs])
    fin = np.array([r["final"][k] for r in recs])
    y = np.array([r["y_own"][k] for r in recs])
    env = p["env"]
    # Twin k's target: own block, cross block; the partner's opening uses its own block of the
    # partner's target (A's target is (w_A, w_B), B's is (v_A, v_B)).
    w_A, w_B, v_A, v_B = (np.array(env[n]) for n in ("w_A", "w_B", "v_A", "v_B"))
    lim = (related_limits(w_A, w_B, v_B, env["sigma"]) if k == "A"
           else related_limits(v_B, v_A, w_A, env["sigma"]))
    sl = slice(late, None)
    # Placeholder openings: a twin that predicts from a joint model may submit 0 as its opening
    # (pilot c = 0 told, twin A from query 28). Its opening is then not an own-feature estimate,
    # so the opening-based weights (beta, WOA) are left out for it.
    placeholder = bool(np.mean(own[sl] == 0.0) > 0.5)
    b = (np.full(3, np.nan) if placeholder
         else ols(np.column_stack([own[sl], par[sl]]), fin[sl]))
    gap = par - own
    big = np.abs(gap) > 0.2
    woa = np.where(big, (fin - own) / np.where(big, gap, 1.0), np.nan)
    away = (np.sign(fin - own) == -np.sign(gap)) & big & (np.abs(fin - own) > 1e-9)
    T = len(recs)
    edges = np.linspace(0, T, WINDOWS + 1).astype(int)
    win = lambda a: [float(np.nanmean((a[i:j] - y[i:j]) ** 2)) for i, j in zip(edges, edges[1:])]  # noqa: E731
    stack = online_stack(own, par, y)
    texts = [m["text"] for r in recs for m in r["messages"] if m["from"] == k]
    hits = [t for t in texts if DIFFERENT.search(t)]
    own_lab, par_lab = p["labels"][k], p["labels"][o]
    # Raw features in a message: "a1=0.3" / "a1: 0.3", or a tuple of d numbers "(0.1, -0.2, 0.3)".
    num = r"-?\d*\.?\d+"
    feat = re.compile(r"\b(" + "|".join(map(re.escape, own_lab)) + r")\s*[=:]\s*-?\d"
                      + r"|\(\s*" + r"\s*,\s*".join([num] * len(own_lab)) + r"\s*\)")
    models = recs[-1]["models"][k]
    joint = {n for n, m in models.items()
             if any(any(lab in c for lab in par_lab) for c in m["columns"])}
    cols = [c for n in joint for c in models[n]["columns"]]
    joint_fits = sum(1 for c in p["calls"][k] if c["tool"] == "fit_model" and c["q"] >= late
                     and c["args"].get("name") in joint)
    return {
        "beta": float(b[2]), "own_coef": float(b[1]),
        "woa": float("nan") if placeholder else float(np.nanmean(np.clip(woa[sl], -1, 2))),
        "placeholder": placeholder,
        "away": float(away[sl][big[sl]].mean()) if big[sl].any() else float("nan"),
        "mse_late": float(np.nanmean((fin[sl] - y[sl]) ** 2)),
        "mse_open_late": float(np.nanmean((own[sl] - y[sl]) ** 2)),
        "mse_stack_late": float(np.nanmean((stack[sl] - y[sl]) ** 2)),
        "mse_avg_late": float(np.nanmean(((own[sl] + par[sl]) / 2 - y[sl]) ** 2)),
        "mse_windows": win(fin), "mse_open_windows": win(own),
        "detect": bool(hits), "detect_first_q": next(
            (r["t"] + 1 for r in recs for m in r["messages"]
             if m["from"] == k and DIFFERENT.search(m["text"])), None),
        "hits": hits[:3],
        "raw_share": float(np.mean([bool(feat.search(" ".join(
            m["text"] for m in r["messages"] if m["from"] == k))) for r in recs])),
        "joint_model": bool(cols),
        "joint_used": joint_fits >= (len(recs) - late) / 2,  # fitted in half the late queries
        "limits": lim,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prefix", default="llm_haiku_sessions_related")
    p.add_argument("--cs", nargs="+", default=["1.0", "0.5", "0.0", "-0.5"])
    p.add_argument("--arms", nargs="+", default=["untold", "told"])
    p.add_argument("--late", type=int, default=20, help="late queries start at this index")
    p.add_argument("--compare", nargs="*", default=None, metavar="PREFIX=LABEL",
                   help="also plot these experiment prefixes side by side (first arm only), "
                        "e.g. llm_haiku_sessions_related=Haiku llm_sonnet_sessions_related=Sonnet")
    args = p.parse_args()
    out = make_run_dir(ROOT / "results", "related")
    cells, rows, det = {}, [], []
    mean = lambda v: float(np.nanmean(v)) if len(v) else float("nan")  # noqa: E731
    for c in args.cs:
        for arm in args.arms:
            pairs = load_pairs(args.prefix, c, arm)
            tw = [twin_stats(pr, k, args.late) for pr in pairs for k in "AB"
                  if not any(pr["errors"].values())]
            if not tw:
                continue
            cell = {key: [t[key] for t in tw] for key in
                    ("beta", "own_coef", "woa", "away", "mse_late", "mse_open_late",
                     "mse_stack_late", "mse_avg_late", "detect", "raw_share", "joint_model",
                     "joint_used", "placeholder")}
            cell["mse_windows"] = np.nanmean([t["mse_windows"] for t in tw], 0).tolist()
            cell["limits"] = {k: mean([t["limits"][k] for t in tw]) for k in tw[0]["limits"]}
            cell["n_pairs"], cell["cost"] = len(pairs), sum(pr["cost"] for pr in pairs)
            cells[f"{float(c)}_{arm}"] = cell
            L = cell["limits"]
            rows.append(
                f"| {c} | {arm} | {len(pairs)} | {mean(cell['beta']):.2f} ({L['beta']:.2f}) | "
                f"{mean(cell['own_coef']):.2f} | {mean(cell['woa']):.2f} ({L['w_convex']:.2f}) | "
                f"{mean(cell['away']):.2f} | "
                + " / ".join(f"{v:.2f}" for v in cell["mse_windows"])
                + f" | {mean(cell['mse_open_late']):.2f} | {mean(cell['mse_avg_late']):.2f} | "
                f"{mean(cell['mse_stack_late']):.2f} | {L['stack']:.2f} | "
                f"{sum(cell['detect'])}/{len(tw)} | {mean(cell['raw_share']):.2f} | "
                f"{sum(cell['joint_model'])} / {sum(cell['joint_used'])} | {sum(cell['placeholder'])} | "
                f"${cell['cost']:.2f} |")
            for pr, t in ((pr, twin_stats(pr, k, args.late)) for pr in pairs for k in "AB"):
                if t["hits"]:
                    det.append(f"### c={c} {arm} {pr['run']} (first q{t['detect_first_q']})\n"
                               + "\n".join(f"> {h[:400]}" for h in t["hits"]) + "\n")
    head = ("| c | arm | pairs | β partner (best) | own coef | WOA (best) | moves away | "
            "MSE finals by window | own opening late | average late | online stack late | "
            "stack pop. | detect | raw-feature share | joint models built / used | placeholder openings | cost |\n"
            "|" + "---|" * 17)
    table = head + "\n" + "\n".join(rows)
    (out / "table.md").write_text(table + "\n")
    (out / "detection.md").write_text("\n".join(det))
    (out / "stats.json").write_text(json.dumps(cells, indent=1, default=float))
    cs_sorted = sorted({float(c) for c in args.cs}, reverse=True)
    plot_related(cells, cs_sorted, out / "related.png")
    if args.compare:
        # One series per experiment prefix, on the first arm: model against model.
        comp, series = {}, []
        for item in args.compare:
            prefix, label = item.split("=", 1)
            series.append((prefix, label))
            for c in args.cs:
                tw = [twin_stats(pr, k, args.late) for pr in load_pairs(prefix, c, args.arms[0])
                      for k in "AB" if not any(pr["errors"].values())]
                if tw:
                    comp[f"{float(c)}_{prefix}"] = {key: [t[key] for t in tw]
                                                    for key in ("beta", "mse_late")}
        plot_related(comp, cs_sorted, out / "compare.png", series=tuple(series))
    print(table)
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
