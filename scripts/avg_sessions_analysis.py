"""Entrenchment with live-session twins under the averaging team rule
(configs/llm_haiku_sessions_avg.yaml, LogBook 2026-11-13): the twins talk freely, then each submits
its own final; the team prediction is the average of the two finals.

Per twin and query, the weight of advice on its OWN final, (final - own opening) /
(partner opening - own opening), clipped as in onset_llm.py; 0 = kept its opening, 1 = adopted
the partner's. Unlike under consensus, this is not 0.5 by construction, so each twin's
stubbornness is measured directly.

- Equal pairs (both studied n): mean WOA over both twins, share of finals left at the opening,
  WOA by query window, the paired change from n = 25 to 1600 (same pair index = same data).
- Mixed pairs (one twin n, partner 25): WOA of the long- and the short-trained twin, and the
  long twin's WOA under the consensus rule (results/entrench_sessions, latest) for comparison.
- Accuracy: team MSE against averaging the openings and the least-squares combination.

Usage:
    python scripts/avg_sessions_analysis.py [--lambdas ...] [--ns 25 100 400 1600]
Writes results/avg_sessions/<timestamp>/{stats.json, table.md, avg_sessions.png}.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import entrench_sessions_analysis as ea  # noqa: E402

from consensus.plotting_overlap import plot_avg_sessions  # noqa: E402
from consensus.sessions import weight_of_advice  # noqa: E402
from consensus.utils import make_run_dir  # noqa: E402

WINDOWS = ea.WINDOWS


def own_final(s: dict, k: str) -> float:
    return s["attempts"][k][0] if s["attempts"][k] else float("nan")


def twin_rows(pairs: dict) -> list[dict]:
    """One row per twin and query: pair, twin, role (long/short/equal), t, WOA, unchanged."""
    rows = []
    for r, p in pairs.items():
        L = p["long"]
        for s in p["recs"]:
            for k, o in (("A", "B"), ("B", "A")):
                own, par, fin = s["opening"][k], s["opening"][o], own_final(s, k)
                role = "equal" if L is None else ("long" if k == L else "short")
                rows.append(dict(pair=r, twin=k, role=role, t=s["t"],
                                 woa=weight_of_advice(own, par, fin),
                                 unchanged=abs(fin - own) < 1e-9 if np.isfinite(fin) else np.nan))
    return rows


def accuracy(pairs: dict) -> dict:
    R = np.array([(s["opening"]["A"], s["opening"]["B"], s["team"], s["y"])
                  for p in pairs.values() for s in p["recs"] if np.isfinite(s["team"])])
    coef = np.linalg.lstsq(R[:, :2], R[:, 3], rcond=None)[0]
    return dict(mse_team=float(np.mean((R[:, 2] - R[:, 3]) ** 2)),
                mse_average=float(np.mean((R[:, :2].mean(1) - R[:, 3]) ** 2)),
                mse_fitted=float(np.mean((R[:, :2] @ coef - R[:, 3]) ** 2)))


def per_pair(rows: list[dict], role: str, key: str = "woa") -> dict[int, float]:
    out = {}
    for r in sorted({x["pair"] for x in rows}):
        out[r] = ea.m([x[key] for x in rows if x["pair"] == r and x["role"] == role])
    return out


def main(argv: list[str] | None = None) -> Path:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lambdas", nargs="+", default=["0.0", "0.3", "0.6", "0.9"])
    p.add_argument("--ns", nargs="+", type=int, default=[25, 100, 400, 1600])
    p.add_argument("--short", type=int, default=25)
    p.add_argument("--prefix", default="llm_haiku_sessions_avg")
    args = p.parse_args(argv)
    ea.PREFIX = args.prefix
    cons = sorted((ROOT / "results" / "entrench_sessions").glob("20*/stats.json"))
    cons = json.loads(cons[-1].read_text()) if cons else {"mixed": {}}
    out = {"equal": {}, "mixed": {}, "cost": 0.0}
    for lam in args.lambdas:
        for n in args.ns:
            eq = ea.load_cell(lam, f"eq{n}")
            if eq:
                rows = twin_rows(eq)
                pp = per_pair(rows, "equal")
                out["equal"][f"{lam}_{n}"] = dict(
                    n_pairs=len(eq), woa=ea.m(list(pp.values())), woa_se=ea.se(list(pp.values())),
                    woa_pairs=pp, unchanged=ea.m([x["unchanged"] for x in rows]),
                    woa_by_window=[ea.m([x["woa"] for x in rows if a <= x["t"] < b])
                                   for a, b in WINDOWS],
                    **accuracy(eq))
                out["cost"] += sum(q["summary"]["cost_usd"] for q in eq.values())
            if n > args.short:
                mx = ea.load_cell(lam, f"mix{n}v{args.short}")
                if mx:
                    rows = twin_rows(mx)
                    pl, ps = per_pair(rows, "long"), per_pair(rows, "short")
                    c = cons["mixed"].get(f"{lam}_{n}", {})
                    out["mixed"][f"{lam}_{n}"] = dict(
                        n_pairs=len(mx), woa_long=ea.m(list(pl.values())),
                        woa_long_se=ea.se(list(pl.values())), woa_short=ea.m(list(ps.values())),
                        woa_short_se=ea.se(list(ps.values())),
                        unchanged_long=ea.m([x["unchanged"] for x in rows if x["role"] == "long"]),
                        unchanged_short=ea.m([x["unchanged"] for x in rows
                                              if x["role"] == "short"]),
                        woa_long_consensus=c.get("woa_long", float("nan")), **accuracy(mx))
                    out["cost"] += sum(q["summary"]["cost_usd"] for q in mx.values())

    lines = ["Equal pairs (both studied n): each twin's WOA on its own final.", "",
             "| λ | n | pairs | WOA (± s.e.) | finals left at the opening | WOA by queries 1–10 / "
             "11–20 / 21–30 | MSE team / average of openings / fitted |",
             "|" + "---|" * 7]
    for k, s in out["equal"].items():
        lam, n = k.split("_")
        lines.append(
            f"| {lam} | {n} | {s['n_pairs']} | {s['woa']:.2f} ± {s['woa_se']:.2f} "
            f"| {s['unchanged']:.0%} | {' / '.join(f'{v:.2f}' for v in s['woa_by_window'])} "
            f"| {s['mse_team']:.2f} / {s['mse_average']:.2f} / {s['mse_fitted']:.2f} |")
    lines += ["", "Paired change in the equal pairs' WOA from n = 25 to n = 1600 (same pair index):",
              ""]
    out["paired"] = {}
    for lam in args.lambdas:
        a, b = out["equal"].get(f"{lam}_25"), out["equal"].get(f"{lam}_1600")
        if a and b:
            d = [b["woa_pairs"][r] - a["woa_pairs"][r] for r in a["woa_pairs"]
                 if r in b["woa_pairs"] and a["woa_pairs"][r] == a["woa_pairs"][r]
                 and b["woa_pairs"][r] == b["woa_pairs"][r]]
            out["paired"][lam] = d
            lines.append(f"- λ = {lam}: {ea.m(d):+.2f} ± {ea.se(d):.2f} "
                         f"({sum(x < 0 for x in d)}/{len(d)} pairs lower)")
    lines += ["", "Mixed pairs (one twin studied n, its partner 25):", "",
              "| λ | n | pairs | WOA long-trained twin | WOA short-trained twin | finals left at the "
              "opening, long / short | long twin under consensus | MSE team / average / fitted |",
              "|" + "---|" * 8]
    for k, s in out["mixed"].items():
        lam, n = k.split("_")
        lines.append(
            f"| {lam} | {n} | {s['n_pairs']} | {s['woa_long']:.2f} ± {s['woa_long_se']:.2f} "
            f"| {s['woa_short']:.2f} ± {s['woa_short_se']:.2f} "
            f"| {s['unchanged_long']:.0%} / {s['unchanged_short']:.0%} "
            f"| {s['woa_long_consensus']:.2f} "
            f"| {s['mse_team']:.2f} / {s['mse_average']:.2f} / {s['mse_fitted']:.2f} |")
    lines.append(f"\ntotal cost ${out['cost']:.2f}")
    run_out = make_run_dir(str(ROOT / "results"), "avg_sessions")
    (run_out / "stats.json").write_text(json.dumps(out, indent=2))
    table = "\n".join(lines)
    (run_out / "table.md").write_text(table + "\n")
    if out["equal"]:
        plot_avg_sessions(out, args.lambdas, args.ns, run_out / "avg_sessions.png")
    print(table)
    print(f"saved to {run_out}")
    return run_out


if __name__ == "__main__":
    main()
