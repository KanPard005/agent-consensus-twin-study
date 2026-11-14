"""Same-y sentence + fresh model instances (configs/llm_haiku_sessions_samey.yaml, LogBook
2026-11-13): when twins can start fresh least-squares models with any columns, do they build one
on the partner's features, use the partner's opening, or keep their own estimate and shift it?

Per twin (from its tool-call log), its strategy over the run:
- "partner features": created a model with a column named after one of the partner's features
  (values the twin must supply itself, e.g. as told in messages);
- "own transforms": created a model with columns it supplies that are not the partner's features
  (in practice: products and squares of its own features);
- "partner opening": created a model with partner_opening (but no supplied columns);
- "own only": created models only on its own features / own_opening;
- "no model": never created one (it can still shift its estimate by hand).
Also: when its first model was made (query), how many cases it added, fit calls, and per twin the
weight of advice on its own final and the MSE of its own final, against the matching cells of
the individual-scoring grid without the sentence and tools (llm_haiku_sessions_own).

Usage:
    python scripts/samey_analysis.py [--lambdas 0.0 0.5] [--ns 25 100 400 1600]
Writes results/samey/<timestamp>/{stats.json, table.md}.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import entrench_sessions_analysis as ea  # noqa: E402

from consensus.sessions import weight_of_advice  # noqa: E402
from consensus.utils import make_run_dir  # noqa: E402

STRATEGIES = ("partner features", "partner opening", "own transforms", "own only", "no model")


def cell_dirs(prefix: str, lam: str, n: int) -> list[Path]:
    return sorted((ROOT / "results" / f"{prefix}_lam{lam}_eq{n}").glob("20*"))


def twin_strategy(calls: list[dict], own_labels: set[str], partner_labels: set[str]) -> dict:
    news = [c for c in calls if c["tool"] == "new_model" and not c["result"].startswith("Error")]
    tried = sum(c["tool"] == "new_model" for c in calls)
    auto = own_labels | {"own_opening", "partner_opening"}
    cols = [set(c["args"].get("columns") or []) for c in news]
    if any(cs & partner_labels for cs in cols):
        strat = "partner features"
    elif any("partner_opening" in cs for cs in cols):
        strat = "partner opening"
    elif any(cs - auto for cs in cols):
        strat = "own transforms"
    elif cols:
        strat = "own only"
    else:
        strat = "no model"
    adds = sum(c["tool"] == "add_case" and not c["result"].startswith("Error") for c in calls)
    fits = sum(c["tool"] == "fit_model" for c in calls)
    return dict(strategy=strat, tried=tried, first_q=min((c["q"] for c in news), default=None),
                n_models=len(news), adds=adds, fits=fits,
                supplied=sorted(set().union(*cols) - auto) if cols else [])


def twin_outcomes(recs: list[dict], k: str) -> tuple[list[float], list[float]]:
    o = "B" if k == "A" else "A"
    w, e = [], []
    for s in recs:
        fin = s["attempts"][k][0] if s["attempts"][k] else float("nan")
        v = weight_of_advice(s["opening"][k], s["opening"][o], fin)
        if v == v:
            w.append(v)
        if fin == fin:
            e.append((fin - s["y"]) ** 2)
    return w, e


def summarize(prefix: str, lam: str, n: int, with_tools: bool) -> dict | None:
    twins = []
    for d in cell_dirs(prefix, lam, n):
        sf = d / "summary.json"
        if not sf.exists():
            continue
        import yaml
        cfg = yaml.safe_load((d / "config.yaml").read_text())
        for key in json.loads(sf.read_text()):
            r = int(key[3:])
            recs = [json.loads(x) for x in (d / "transcripts" / f"run{r}.jsonl").read_text()
                    .splitlines() if x.strip()]
            for k in "AB":
                w, e = twin_outcomes(recs, k)
                row = dict(pair=r, twin=k, woa=ea.m(w), mse_final=ea.m(e))
                if with_tools:
                    calls = [json.loads(x) for x in (d / "transcripts" / f"run{r}_calls_{k}.jsonl")
                             .read_text().splitlines() if x.strip()]
                    o = "B" if k == "A" else "A"
                    row.update(twin_strategy(calls, set(cfg["agent"]["labels"][k]),
                                             set(cfg["agent"]["labels"][o])))
                twins.append(row)
    if not twins:
        return None
    out = dict(n_twins=len(twins), woa=ea.m([t["woa"] for t in twins]),
               woa_se=ea.se([t["woa"] for t in twins]),
               mse_final=ea.m([t["mse_final"] for t in twins]), twins=twins)
    if with_tools:
        cnt = Counter(t["strategy"] for t in twins)
        out["strategies"] = {s: cnt.get(s, 0) / len(twins) for s in STRATEGIES}
        out["woa_by_strategy"] = {s: ea.m([t["woa"] for t in twins if t["strategy"] == s])
                                  for s in STRATEGIES}
        out["mse_by_strategy"] = {s: ea.m([t["mse_final"] for t in twins if t["strategy"] == s])
                                  for s in STRATEGIES}
        firsts = [t["first_q"] for t in twins if t["first_q"] is not None]
        out["first_model_query"] = float(np.median(firsts)) + 1 if firsts else None
        out["supplied_columns"] = Counter(c for t in twins for c in t["supplied"]).most_common(8)
    return out


def main(argv: list[str] | None = None) -> Path:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lambdas", nargs="+", default=["0.0", "0.5"])
    p.add_argument("--ns", nargs="+", type=int, default=[25, 100, 400, 1600])
    p.add_argument("--prefix", default="llm_haiku_sessions_samey")
    p.add_argument("--baseline", default="llm_haiku_sessions_own")
    args = p.parse_args(argv)
    out, lines = {}, [
        "| λ | n | WOA: new / old prompt | MSE own final: new / old | strategy: partner features / "
        "partner opening / own transforms / own only / no model | WOA by strategy (same order) "
        "| first model (median query) |",
        "|" + "---|" * 7,
    ]
    for lam in args.lambdas:
        for n in args.ns:
            new = summarize(args.prefix, lam, n, True)
            old = summarize(args.baseline, lam, n, False)
            if new is None:
                continue
            out[f"{lam}_{n}"] = {"new": new, "old": old}
            st, wbs = new["strategies"], new["woa_by_strategy"]
            lines.append(
                f"| {lam} | {n} | {new['woa']:.2f} ± {new['woa_se']:.2f} / "
                + (f"{old['woa']:.2f} ± {old['woa_se']:.2f}" if old else "—")
                + f" | {new['mse_final']:.2f} / " + (f"{old['mse_final']:.2f}" if old else "—")
                + " | " + " / ".join(f"{st[s]:.0%}" for s in STRATEGIES)
                + " | " + " / ".join("—" if wbs[s] != wbs[s] else f"{wbs[s]:.2f}" for s in STRATEGIES)
                + f" | {new['first_model_query'] if new['first_model_query'] else '—'} |"
            )
    cols = Counter()
    for v in out.values():
        cols.update(dict(v["new"]["supplied_columns"]))
    lines.append(f"\nColumns the twins supplied themselves (all cells): {cols.most_common(10)}")
    run_out = make_run_dir(str(ROOT / "results"), "samey")
    (run_out / "stats.json").write_text(json.dumps(out, indent=2, default=str))
    table = "\n".join(lines)
    (run_out / "table.md").write_text(table + "\n")
    print(table)
    print(f"saved to {run_out}")
    return run_out


if __name__ == "__main__":
    main()
