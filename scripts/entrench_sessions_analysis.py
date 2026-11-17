"""Entrenchment with live-session twins (configs/llm_haiku_sessions_entrench.yaml, LogBook
2026-11-13): does studying alone for longer make a twin concede less under consensus?

Cells (per lambda): equal pairs eq<n> (both twins studied n) and mixed pairs mix<n>v25 (one twin
studied n, its partner 25; the long-trained twin alternates A/B by pair index).

Mixed pairs, per pair:
- WOA of the long-trained twin on the agreed value, (agreed - own) / (partner - own), clipped as
  in onset_llm.py (0 = the agreement is its opening; the short twin's WOA is 1 minus it);
- hold share: queries where the agreement sits on the long twin's opening (WOA <= 0.05);
- the rational partner weight for the long twin: from the least-squares fit of y on the two
  openings over the cell, b_short / (b_long + b_short). Conceding less than this is
  entrenchment; conceding about this much is deference to the better-informed twin;
- MSE of the agreed value against averaging the openings and against the fitted combination.
Equal pairs: share of agreements at the exact midpoint, mean |WOA - 0.5|, attempts.
Over time: the long twin's WOA by query window (1-10, 11-20, 21-30).

Usage:
    python scripts/entrench_sessions_analysis.py [--lambdas 0.0 0.3 0.6 0.9] [--ns 25 100 400 1600]
Writes results/entrench_sessions/<timestamp>/{stats.json, table.md, entrench_sessions.png}.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consensus.plotting_overlap import plot_entrench_sessions  # noqa: E402
from consensus.sessions import weight_of_advice  # noqa: E402
from consensus.utils import make_run_dir  # noqa: E402

PREFIX = "llm_haiku_sessions_entrench"
WINDOWS = ((0, 10), (10, 20), (20, 30))


def load_cell(lam: str, cell: str) -> dict[int, dict]:
    """{pair index: {"recs": [...], "long": "A"|"B"|None, "summary": {...}}} (latest wins)."""
    out = {}
    for d in sorted((ROOT / "results" / f"{PREFIX}_lam{lam}_{cell}").glob("20*")):
        sf = d / "summary.json"
        if not sf.exists():
            continue
        summ = json.loads(sf.read_text())
        for key, s in summ.items():
            r = int(key[3:])
            f = d / "transcripts" / f"run{r}.jsonl"
            recs = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
            out[r] = {"recs": recs, "long": s.get("long_twin"), "summary": s}
    return out


def m(v) -> float:
    v = [x for x in v if x == x]
    return float(np.mean(v)) if v else float("nan")


def se(v) -> float:
    v = [x for x in v if x == x]
    return float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")


def mixed_stats(pairs: dict) -> dict:
    per_pair, rows = [], []  # rows: (o_long, o_short, agreed, y, t) over the cell
    woa_win = [[] for _ in WINDOWS]
    for r, p in sorted(pairs.items()):
        L = p["long"]
        S = "B" if L == "A" else "A"
        w, hold = [], []
        for s in p["recs"]:
            if s["failed"]:
                continue
            oL, oS, ag = s["opening"][L], s["opening"][S], s["agreed"]
            rows.append((oL, oS, ag, s["y"], s["t"]))
            v = weight_of_advice(oL, oS, ag)
            if v == v:
                w.append(v)
                hold.append(v <= 0.05)
                for i, (a, b) in enumerate(WINDOWS):
                    if a <= s["t"] < b:
                        woa_win[i].append(v)
        eL = m([(s["opening"][L] - s["y"]) ** 2 for s in p["recs"]])
        eS = m([(s["opening"][S] - s["y"]) ** 2 for s in p["recs"]])
        per_pair.append(dict(pair=r, long=L, woa_long=m(w), hold=m(hold), mse_long=eL,
                             mse_short=eS))
    R = np.array(rows)
    coef = np.linalg.lstsq(R[:, :2], R[:, 3], rcond=None)[0]  # y on (o_long, o_short)
    rational = float(coef[1] / coef.sum()) if coef.sum() != 0 else float("nan")
    return dict(
        pairs=per_pair,
        woa_long=m([p["woa_long"] for p in per_pair]),
        woa_long_se=se([p["woa_long"] for p in per_pair]),
        hold=m([p["hold"] for p in per_pair]),
        rational_partner_weight=rational,
        y_fit=coef.tolist(),
        mse_long=m([p["mse_long"] for p in per_pair]),
        mse_short=m([p["mse_short"] for p in per_pair]),
        mse_agreed=float(np.mean((R[:, 2] - R[:, 3]) ** 2)),
        mse_average=float(np.mean(((R[:, 0] + R[:, 1]) / 2 - R[:, 3]) ** 2)),
        mse_fitted=float(np.mean((R[:, :2] @ coef - R[:, 3]) ** 2)),
        woa_long_by_window=[m(v) for v in woa_win],
    )


def equal_stats(pairs: dict) -> dict:
    dev, mid, att, mse = [], [], [], []
    for p in pairs.values():
        for s in p["recs"]:
            if s["failed"]:
                continue
            v = weight_of_advice(s["opening"]["A"], s["opening"]["B"], s["agreed"])
            if v == v:
                dev.append(abs(v - 0.5))
                mid.append(abs(v - 0.5) <= 0.05)
            att.append(max(len(s["attempts"]["A"]), len(s["attempts"]["B"])))
            mse.append((s["agreed"] - s["y"]) ** 2)
    return dict(n_pairs=len(pairs), midpoint=m(mid), dev=m(dev), attempts=m(att), mse_agreed=m(mse),
                failed=sum(s["failed"] for p in pairs.values() for s in p["recs"]))


def main(argv: list[str] | None = None) -> Path:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lambdas", nargs="+", default=["0.0", "0.3", "0.6", "0.9"])
    p.add_argument("--ns", nargs="+", type=int, default=[25, 100, 400, 1600])
    p.add_argument("--short", type=int, default=25)
    args = p.parse_args(argv)
    out = {"mixed": {}, "equal": {}, "cost": 0.0}
    for lam in args.lambdas:
        for n in args.ns:
            eq = load_cell(lam, f"eq{n}")
            if eq:
                out["equal"][f"{lam}_{n}"] = equal_stats(eq)
                out["cost"] += sum(pp["summary"]["cost_usd"] for pp in eq.values())
            if n > args.short:
                mx = load_cell(lam, f"mix{n}v{args.short}")
                if mx:
                    out["mixed"][f"{lam}_{n}"] = {**mixed_stats(mx), "n_pairs": len(mx)}
                    out["cost"] += sum(pp["summary"]["cost_usd"] for pp in mx.values())
    lines = [
        f"Mixed pairs (long-trained twin vs a partner that studied {args.short}):",
        "",
        "| λ | n (long) | pairs | WOA of long twin (± s.e.) | rational partner weight | holds its opening "
        "| WOA by queries 1–10 / 11–20 / 21–30 | opening MSE long / short | MSE agreed / average / "
        "fitted |",
        "|" + "---|" * 9,
    ]
    for k, s in out["mixed"].items():
        lam, n = k.split("_")
        wbw = " / ".join(f"{v:.2f}" for v in s["woa_long_by_window"])
        lines.append(
            f"| {lam} | {n} | {s['n_pairs']} | {s['woa_long']:.2f} ± {s['woa_long_se']:.2f} "
            f"| {s['rational_partner_weight']:.2f} | {s['hold']:.0%} | {wbw} "
            f"| {s['mse_long']:.2f} / {s['mse_short']:.2f} "
            f"| {s['mse_agreed']:.2f} / {s['mse_average']:.2f} / {s['mse_fitted']:.2f} |"
        )
    lines += [
        "",
        "Equal pairs (both twins studied n):",
        "",
        "| λ | n | pairs | at the exact midpoint | mean \\|WOA − 0.5\\| | attempts per query | failed "
        "| agreed MSE |",
        "|" + "---|" * 8,
    ]
    for k, s in out["equal"].items():
        lam, n = k.split("_")
        lines.append(
            f"| {lam} | {n} | {s['n_pairs']} | {s['midpoint']:.0%} | {s['dev']:.2f} "
            f"| {s['attempts']:.2f} | {s['failed']} | {s['mse_agreed']:.2f} |"
        )
    lines.append(f"\ntotal cost ${out['cost']:.2f}")
    run_out = make_run_dir(str(ROOT / "results"), "entrench_sessions")
    (run_out / "stats.json").write_text(json.dumps(out, indent=2))
    table = "\n".join(lines)
    (run_out / "table.md").write_text(table + "\n")
    if out["mixed"]:
        plot_entrench_sessions(out, args.lambdas, run_out / "entrench_sessions.png")
    print(table)
    print(f"saved to {run_out}")
    return run_out


if __name__ == "__main__":
    main()
