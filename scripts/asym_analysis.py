"""Asymmetric information with live-session twins (LogBook 2026-11-14).

The controlled-lambda design with the weak twin's weights scaled by alpha
(configs/llm_haiku_sessions_asym_{own,cons}.yaml, launched with launch_grid.py --alphas). The
strong twin's solo excess error is alpha^2 times the weak twin's at every lambda
(consensus.theory.asymmetric_limits), so alpha sets who should defer and by how much.

Per cell (rule, lambda, alpha), by role (strong / weak twin):
- MSE of the openings, against the population value sigma^2 + excess;
- weight of advice (WOA) on the twin's own final (individual scoring) or on the agreed value
  (consensus), averaged per twin, then over twins (s.e. over twins);
- the best convex weight on the partner, w = Cov(y - own, partner - own) / Var(partner - own),
  from theory (`asymmetric_limits`) and fitted in the cell: the reference for WOA;
- MSE of the finals (or the agreed value) against the average of the openings, the fitted
  combination and the strong twin's opening alone;
- the share of twins that ever state their own fit quality (MSE, R^2, LOO, residual) in a message.

Usage:
    python scripts/asym_analysis.py [--lambdas 0.0 0.5] [--alphas 0.0 0.25 0.5 1.0] [--n 400]
Writes results/asym/<timestamp>/{stats.json, table.md, asym_woa.png}.
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

from consensus.plotting_asym import plot_asym_woa  # noqa: E402
from consensus.sessions import weight_of_advice  # noqa: E402
from consensus.theory import asymmetric_limits  # noqa: E402
from consensus.utils import make_run_dir  # noqa: E402

RULES = {"own": "individual", "cons": "consensus"}
FIT_QUALITY = re.compile(r"\bMSE\b|R\^?2|R²|\bLOO\b|residual", re.IGNORECASE)


def load_pairs(prefix: str, lam: str, alpha: str, n: int) -> list[dict]:
    """One dict per pair: transcript records, the weak twin's side and the run's cost."""
    pairs = []
    for d in sorted((ROOT / "results" / f"{prefix}_lam{lam}_al{alpha}_eq{n}").glob("20*")):
        sf = d / "summary.json"
        if not sf.exists():
            continue
        weak = yaml.safe_load((d / "config.yaml").read_text())["env"].get("weak", "B")
        for key, s in json.loads(sf.read_text()).items():
            f = d / "transcripts" / f"{key}.jsonl"
            recs = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
            pairs.append(dict(recs=recs, weak=weak, cost=s.get("cost_usd", 0.0)))
    return pairs


def m(v) -> float:
    v = [x for x in v if x == x]
    return float(np.mean(v)) if v else float("nan")


def se(v) -> float:
    v = [x for x in v if x == x]
    return float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")


def convex_weight(own: np.ndarray, partner: np.ndarray, y: np.ndarray) -> float:
    """Least-squares w in y ~ (1 - w) own + w partner (fit of y - own on partner - own)."""
    d = partner - own
    return float(d @ (y - own) / (d @ d))


def cell_stats(pairs: list[dict], rule: str, lam: float, alpha: float, sigma: float) -> dict:
    """Per-role statistics for one cell (see the module docstring)."""
    woa = {"strong": [], "weak": []}  # per twin
    states_fit = {"strong": [], "weak": []}
    rows = []  # (o_strong, o_weak, final_strong, final_weak, y)
    for p in pairs:
        W = p["weak"]
        S = "B" if W == "A" else "A"
        w_s, w_w = [], []
        said = {S: False, W: False}
        for s in p["recs"]:
            for msg in s.get("messages", []):
                if FIT_QUALITY.search(msg.get("text", "")):
                    said[msg["from"]] = True
            if s["failed"]:
                continue
            oS, oW = s["opening"][S], s["opening"][W]
            if rule == "consensus":
                fS = fW = s["agreed"]
            else:
                fS, fW = s["final"][S], s["final"][W]
            if fS is None or fW is None:
                continue
            rows.append((oS, oW, fS, fW, s["y"]))
            w_s.append(weight_of_advice(oS, oW, fS))
            w_w.append(weight_of_advice(oW, oS, fW))
        woa["strong"].append(m(w_s))
        woa["weak"].append(m(w_w))
        states_fit["strong"].append(said[S])
        states_fit["weak"].append(said[W])
    R = np.array(rows, float)
    y = R[:, 4]
    coef = np.linalg.lstsq(R[:, :2], y, rcond=None)[0]  # y on (o_strong, o_weak), no intercept
    lim = asymmetric_limits(lam, alpha, sigma)

    def mse(v):
        return float(np.mean((v - y) ** 2))

    return dict(
        n_pairs=len(pairs),
        n_queries=len(rows),
        woa_strong=m(woa["strong"]), woa_strong_se=se(woa["strong"]),
        woa_weak=m(woa["weak"]), woa_weak_se=se(woa["weak"]),
        theory_share_strong=lim["convex_strong"],
        theory_share_weak=lim["convex_weak"],
        theory_beta=[lim["beta_strong"], lim["beta_weak"]],
        fit_beta=coef.tolist(),
        fit_share_strong=convex_weight(R[:, 0], R[:, 1], y),
        fit_share_weak=convex_weight(R[:, 1], R[:, 0], y),
        mse_convex_strong=mse(R[:, 0] + convex_weight(R[:, 0], R[:, 1], y) * (R[:, 1] - R[:, 0])),
        mse_open_strong=mse(R[:, 0]), mse_open_weak=mse(R[:, 1]),
        theory_mse_strong=sigma**2 + lim["excess_strong"],
        theory_mse_weak=sigma**2 + lim["excess_weak"],
        mse_final_strong=mse(R[:, 2]), mse_final_weak=mse(R[:, 3]),
        mse_average=mse(R[:, :2].mean(axis=1)),
        mse_fitted=mse(R[:, :2] @ coef),
        states_fit_strong=m(states_fit["strong"]), states_fit_weak=m(states_fit["weak"]),
        cost=float(sum(p["cost"] for p in pairs)),
    )


def main(argv: list[str] | None = None) -> Path:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lambdas", nargs="+", default=["0.0", "0.5"])
    p.add_argument("--alphas", nargs="+", default=["0.0", "0.25", "0.5", "1.0"])
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--prefix", default="llm_haiku_sessions_asym")
    p.add_argument("--sigma", type=float, default=0.5)
    args = p.parse_args(argv)
    out: dict = {"cells": {}, "cost": 0.0}
    for tag, rule in RULES.items():
        for lam in args.lambdas:
            for al in args.alphas:
                pairs = load_pairs(f"{args.prefix}_{tag}", lam, al, args.n)
                if not pairs:
                    continue
                st = cell_stats(pairs, rule, float(lam), float(al), args.sigma)
                out["cells"][f"{rule}_{lam}_{al}"] = st
                out["cost"] += st["cost"]
    lines = [
        f"Equal training (n = {args.n}); strong twin s = 1/√d, weak twin s = α/√d. "
        "WOA ± s.e. over twins; 'best' = best convex weight on the partner "
        "(theory / fitted in the cell); 'fitted' = best unrestricted combination y ~ a·strong + b·weak.",
        "",
        "| rule | λ | α | pairs | WOA strong | best strong | WOA weak | best weak "
        "| opening MSE strong / weak (theory) | final MSE strong / weak | average / best convex / fitted "
        "| state fit quality strong / weak |",
        "|" + "---|" * 12,
    ]
    for k, s in out["cells"].items():
        rule, lam, al = k.split("_")
        lines.append(
            f"| {rule} | {lam} | {al} | {s['n_pairs']} "
            f"| {s['woa_strong']:.2f} ± {s['woa_strong_se']:.2f} "
            f"| {s['theory_share_strong']:.2f} / {s['fit_share_strong']:.2f} "
            f"| {s['woa_weak']:.2f} ± {s['woa_weak_se']:.2f} "
            f"| {s['theory_share_weak']:.2f} / {s['fit_share_weak']:.2f} "
            f"| {s['mse_open_strong']:.2f} / {s['mse_open_weak']:.2f} "
            f"({s['theory_mse_strong']:.2f} / {s['theory_mse_weak']:.2f}) "
            f"| {s['mse_final_strong']:.2f} / {s['mse_final_weak']:.2f} "
            f"| {s['mse_average']:.2f} / {s['mse_convex_strong']:.2f} / {s['mse_fitted']:.2f} "
            f"| {s['states_fit_strong']:.0%} / {s['states_fit_weak']:.0%} |"
        )
    lines.append(f"\ntotal cost ${out['cost']:.2f}")
    run_out = make_run_dir(str(ROOT / "results"), "asym")
    (run_out / "stats.json").write_text(json.dumps(out, indent=2))
    table = "\n".join(lines)
    (run_out / "table.md").write_text(table + "\n")
    if out["cells"]:
        plot_asym_woa(out["cells"], args.lambdas, args.alphas, run_out / "asym_woa.png")
    print(table)
    print(f"saved to {run_out}")
    return run_out


if __name__ == "__main__":
    main()
