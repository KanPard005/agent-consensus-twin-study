"""Stated confidence over time (LogBook 2026-11-14; configs/llm_haiku_sessions_confidence_*).

Each twin gives an 80% interval for y with its opening, and with its final the typical absolute
error it expects of its own opening and of its partner's. Two arms: no training phase (T = 60,
time = cases seen together) and 400 training cases first (T = 30).

Per arm, cell (lambda, alpha), role (strong / weak twin) and window of queries:
- stated own / partner error against the realised mean absolute error of the own / partner
  opening in the same window (ratio 1 = calibrated, < 1 = overconfident);
- self-favouring bias = log(stated partner / realised partner) - log(stated own / realised own):
  > 0 means the twin rates its partner worse, relative to itself, than the outcomes justify.
  Entrenchment in stated beliefs = this bias growing over time;
- the 80% interval: median width and coverage;
- weight of advice (WOA) on the final against the inverse-variance weight implied by the stated
  errors, own^2 / (own^2 + partner^2), and against the population best convex weight.

Usage:
    python scripts/confidence_analysis.py
Writes results/confidence/<timestamp>/{stats.json, table.md, confidence.png}.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consensus.plotting_asym import plot_confidence  # noqa: E402
from consensus.sessions import weight_of_advice  # noqa: E402
from consensus.theory import asymmetric_limits  # noqa: E402
from consensus.utils import make_run_dir  # noqa: E402

ARMS = {  # arm: (experiment prefix, cell suffix, windows)
    "untrained": ("llm_haiku_sessions_confidence_untrained", "eq0",
                  ((0, 15), (15, 30), (30, 45), (45, 60))),
    "trained": ("llm_haiku_sessions_confidence_trained", "eq400", ((0, 10), (10, 20), (20, 30))),
}
LAMBDAS, ALPHAS = ("0.0", "0.5"), ("0.5", "1.0")


def load_pairs(prefix: str, lam: str, alpha: str, suffix: str) -> list[dict]:
    pairs = []
    for d in sorted((ROOT / "results" / f"{prefix}_lam{lam}_al{alpha}_{suffix}").glob("20*")):
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


def nanmean(v) -> float:
    v = [x for x in v if x == x]
    return float(np.mean(v)) if v else float("nan")


def role_window(pairs: list[dict], role: str, a: int, b: int) -> dict:
    """Pooled statistics for one role ("S" or "W") over queries a..b-1."""
    st_own, st_par, re_own, re_par, width, cover, woa, implied = ([] for _ in range(8))
    for p in pairs:
        W = p["weak"]
        S = "B" if W == "A" else "A"
        me, other = (S, W) if role == "S" else (W, S)
        for s in p["recs"]:
            if not a <= s["t"] < b or s["failed"]:
                continue
            y, o_me, o_ot = s["y"], s["opening"][me], s["opening"][other]
            own_e, par_e = s["stated_error"][me]
            lo, hi = s["interval"][me]
            re_own.append(abs(o_me - y))
            re_par.append(abs(o_ot - y))
            if own_e == own_e:
                st_own.append(own_e)
                st_par.append(par_e)
                if own_e + par_e > 0:
                    implied.append(own_e**2 / (own_e**2 + par_e**2))
            if lo == lo:
                width.append(hi - lo)
                cover.append(lo <= y <= hi)
            woa.append(weight_of_advice(o_me, o_ot, s["final"][me]))
    so, sp, ro, rp = nanmean(st_own), nanmean(st_par), nanmean(re_own), nanmean(re_par)
    return dict(stated_own=so, stated_partner=sp, real_own=ro, real_partner=rp,
                calib_own=so / ro, calib_partner=sp / rp,
                bias=float(np.log(sp / rp) - np.log(so / ro)),
                width=float(np.median(width)) if width else float("nan"), coverage=nanmean(cover),
                woa=nanmean(woa), implied=nanmean(implied), n=len(re_own))


def per_pair_bias(pairs: list[dict], role: str, windows) -> list[list[float]]:
    """Self-favouring bias per pair and window (for the spread across pairs)."""
    out = []
    for p in pairs:
        out.append([role_window([p], role, a, b)["bias"] for a, b in windows])
    return out


def main() -> Path:
    out: dict = {"cells": {}, "cost": 0.0}
    for arm, (prefix, suffix, windows) in ARMS.items():
        for lam in LAMBDAS:
            for al in ALPHAS:
                pairs = load_pairs(prefix, lam, al, suffix)
                if not pairs:
                    continue
                lim = asymmetric_limits(float(lam), float(al))
                cell = {"n_pairs": len(pairs), "windows": windows,
                        "best": {"S": lim["convex_strong"], "W": lim["convex_weak"]}}
                for role in ("S", "W"):
                    cell[role] = [role_window(pairs, role, a, b) for a, b in windows]
                    pb = np.array(per_pair_bias(pairs, role, windows))
                    cell[f"bias_pairs_{role}"] = pb.tolist()
                    # Paired change in bias from the first to the last window, over pairs.
                    d = pb[:, -1] - pb[:, 0]
                    d = d[np.isfinite(d)]
                    cell[f"bias_change_{role}"] = [float(d.mean()),
                                                   float(d.std(ddof=1) / np.sqrt(len(d)))]
                out["cells"][f"{arm}_{lam}_{al}"] = cell
                out["cost"] += sum(p["cost"] for p in pairs)
    lines = []
    for role, name in (("S", "better-informed"), ("W", "less-informed")):
        lines += [f"\n{name} twin (per window of queries):", "",
                  "| arm | λ | α | stated / realised own error | stated / realised partner error "
                  "| self-favouring bias | change first→last (± s.e. over pairs) "
                  "| 80% width / coverage | WOA / implied by stated / best |",
                  "|" + "---|" * 9]
        for k, c in out["cells"].items():
            arm, lam, al = k.split("_")
            R = c[role]
            f = lambda key: " / ".join(f"{w[key]:.2f}" for w in R)  # noqa: E731
            own = " · ".join(f"{w['stated_own']:.2f}/{w['real_own']:.2f}" for w in R)
            par = " · ".join(f"{w['stated_partner']:.2f}/{w['real_partner']:.2f}" for w in R)
            wid = " · ".join(f"{w['width']:.1f}/{w['coverage']:.0%}" for w in R)
            woa = " · ".join(f"{w['woa']:.2f}/{w['implied']:.2f}" for w in R)
            ch, se = c[f"bias_change_{role}"]
            lines.append(f"| {arm} | {lam} | {al} | {own} | {par} | {f('bias')} "
                         f"| {ch:+.2f} ± {se:.2f} | {wid} | {woa} / {c['best'][role]:.2f} |")
    lines.append(f"\ntotal cost ${out['cost']:.2f}")
    run_out = make_run_dir(str(ROOT / "results"), "confidence")
    (run_out / "stats.json").write_text(json.dumps(out, indent=2))
    table = "\n".join(lines)
    (run_out / "table.md").write_text(table + "\n")
    if out["cells"]:
        plot_confidence(out["cells"], LAMBDAS, ALPHAS, run_out / "confidence.png")
    print(table)
    print(f"saved to {run_out}")
    return run_out


if __name__ == "__main__":
    main()
