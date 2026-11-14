"""Twins as live agent sessions in the controlled-lambda design (LogBook 2026-11-13).

Reads the latest results/llm_haiku_sessions_lam<lambda>_E<n>/<timestamp> per cell
(configs/llm_haiku_sessions_lambda.yaml) and computes, per run (pair):

- the combination: OLS of the agreed prediction on the two openings (no intercept),
  agreed = c_A open_A + c_B open_B, against the population-optimal blend of the solo opinions
  (`theory.blend_weights`; 1 + 1 at lambda = 0). Averaging is 0.5 + 0.5. (The twin-averaged
  weight of advice of the scripted analysis is identically 0.5 under a consensus rule, since both
  twins end at the same value, so it is not used.) Also the share of agreed predictions outside
  the interval between the openings;
- movement toward the population best response to the partner's opening (`theory.optimal_head`
  with the partner's population solo predictor), median of (agreed - own) / (BR - own);
- consensus: attempts per query, failed queries, messages per query;
- accuracy: the openings' and the agreed prediction's mean squared error, and the opening gap².

Entrenchment (twins that studied 400 vs 25 cases): harder agreement (first-attempt consensus,
attempts, failures) and a combination closer to compromise (c_A + c_B), paired by pair index
(same data). The scripted experiment's WOA difference is printed alongside for reference.

Usage:
    python scripts/sessions_lambda_analysis.py [--lambdas 0.0 0.333 0.6 0.9] [--arms 25 400]
Writes results/sessions_lambda/<timestamp>/{stats.json, table.md, sessions_lambda.png}.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consensus.data import build_env  # noqa: E402
from consensus.sessions import weight_of_advice  # noqa: E402
from consensus.plotting_overlap import plot_sessions_lambda  # noqa: E402
from consensus.theory import blend_weights, optimal_head, solo_predictors  # noqa: E402
from consensus.utils import make_run_dir  # noqa: E402

# Scripted S1n twins (LogBook 2026-11-12 00:37): WOA after studying 25 / 400 cases.
SCRIPTED = {"0.0": (0.28, 0.19), "0.333": (0.29, 0.31), "0.6": (0.37, 0.31), "0.9": (0.40, 0.27)}


def latest(prefix: str) -> Path | None:
    runs = sorted((ROOT / "results" / prefix).glob("20*"))
    return runs[-1] if runs else None


def best_response(cfg: dict) -> dict:
    """Coefficients (own features..., partner opening) of each twin's population best response."""
    env = build_env(cfg["env"], np.random.default_rng(cfg["seed"]))
    d = env.d_A
    a, b = solo_predictors(env.Sigma, env.w, d)
    return {
        "A": optimal_head(env.Sigma, env.w, d, b[d:], "A"),
        "B": optimal_head(env.Sigma, env.w, d, a[:d], "B"),
        "blend": blend_weights(env.Sigma, env.w, a, b).tolist(),
        "sigma2": env.sigma**2,
    }


def pair_stats(recs: list[dict], br: dict, min_gap: float) -> dict:
    woa, mov, own_err, gap, X, Y, outside = [], [], [], [], [], [], []
    for s in recs:
        if s["failed"] or not np.isfinite(s["agreed"]):
            continue
        for me, o in (("A", "B"), ("B", "A")):
            own, par, fin = s["opening"][me], s["opening"][o], s["agreed"]
            if not (np.isfinite(own) and np.isfinite(par)):
                continue
            own_err.append((own - s["y"]) ** 2)
            if abs(par - own) > min_gap:
                woa.append(float(np.clip((fin - own) / (par - own), -0.5, 1.5)))
            c = br[me]
            target = float(np.dot(c[:-1], s["x"][me]) + c[-1] * par)
            if abs(target - own) > min_gap:
                mov.append((fin - own) / (target - own))
        oA, oB = s["opening"]["A"], s["opening"]["B"]
        gap.append((oA - oB) ** 2)
        if np.isfinite(oA) and np.isfinite(oB):
            X.append([oA, oB])
            Y.append(s["agreed"])
            outside.append(not (min(oA, oB) - 1e-6 <= s["agreed"] <= max(oA, oB) + 1e-6))
    ok = [s for s in recs if not s["failed"] and np.isfinite(s["agreed"])]

    def m(v):
        return float(np.mean(v)) if len(v) else float("nan")

    # Weight of advice per twin, recomputed from the openings (older records lack "woa"):
    # on the agreed value, and on the twin's own first submission. The two twins' WOA on the
    # agreed value add up to 1, so |WOA - 0.5| measures how unevenly they conceded, and the
    # "better twin" check asks whether the twin with the more accurate opening conceded less.
    w = {"A": [], "B": []}
    w_first = {"A": [], "B": []}
    dev, better_held = [], []
    for s in recs:
        oA, oB = s["opening"]["A"], s["opening"]["B"]
        if s["failed"]:
            continue
        wa = weight_of_advice(oA, oB, s["agreed"])
        wb = weight_of_advice(oB, oA, s["agreed"])
        for k, v in (("A", wa), ("B", wb)):
            if np.isfinite(v):
                w[k].append(v)
        for k, o, po in (("A", oA, oB), ("B", oB, oA)):
            v = weight_of_advice(o, po, s["attempts"][k][0] if s["attempts"][k] else np.nan)
            if np.isfinite(v):
                w_first[k].append(v)
        if np.isfinite(wa):
            dev.append(abs(wa - 0.5))
            if abs(wa - 0.5) > 0.05:  # someone conceded more than half
                a_better = abs(oA - s["y"]) < abs(oB - s["y"])
                better_held.append((wa < 0.5) == a_better)
    c = (np.linalg.lstsq(np.array(X), np.array(Y), rcond=None)[0].tolist()
         if len(X) >= 4 else [float("nan")] * 2)
    return dict(
        n_queries=len(recs),
        woa_A=m(w["A"]),
        woa_B=m(w["B"]),
        woa_first_A=m(w_first["A"]),
        woa_first_B=m(w_first["B"]),
        woa_dev=m(dev),
        midpoint=m([d <= 0.05 for d in dev]),
        better_held=m(better_held),
        n_uneven=len(better_held),
        c_sum=float(sum(c)),
        c=c,
        outside=m(outside),
        woa=m(woa),
        movement=float(np.median(mov)) if mov else float("nan"),
        failed=sum(s["failed"] for s in recs),
        attempts=m([max(len(s["attempts"]["A"]), len(s["attempts"]["B"])) for s in recs]),
        first_attempt=m([len(s["attempts"]["A"]) == 1 for s in recs]),
        messages=m([len(s["messages"]) for s in recs]),
        opening_mse=m(own_err),
        agreed_mse=m([(s["agreed"] - s["y"]) ** 2 for s in ok]),
        opening_gap2=m(gap),
    )


def main(argv: list[str] | None = None) -> Path:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--lambdas", nargs="+", default=["0.0", "0.333", "0.6", "0.9"])
    p.add_argument("--arms", nargs="+", default=["25", "400"])
    p.add_argument("--min-gap", type=float, default=0.2)
    p.add_argument("--prefix", default="llm_haiku_sessions")
    args = p.parse_args(argv)
    stats, costs, br_by_lam = {}, {}, {}
    for lam in args.lambdas:
        for arm in args.arms:
            d = latest(f"{args.prefix}_lam{lam}_E{arm}")
            if d is None:
                continue
            cfg = yaml.safe_load((d / "config.yaml").read_text())
            br = best_response(cfg)
            br_by_lam[lam] = br
            summ = json.loads((d / "summary.json").read_text()) if (d / "summary.json").exists() else {}
            cell = {}
            for f in sorted((d / "transcripts").glob("run*.jsonl")):
                if not f.stem[3:].isdigit():
                    continue
                recs = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
                if recs:
                    cell[int(f.stem[3:])] = pair_stats(recs, br, args.min_gap)
            stats[f"{lam}_E{arm}"] = cell
            costs[f"{lam}_E{arm}"] = sum(v["cost_usd"] for v in summ.values())

    rows = [
        "| λ | optimal c_A + c_B | c_A + c_B E25 / E400 | outside openings E25 / E400 "
        "| first-attempt consensus E25 / E400 | attempts E25 / E400 | failed E25 / E400 "
        "| messages per query E25 / E400 | movement to own BR E25 / E400 "
        "| agreed MSE E25 / E400 | opening MSE E25 / E400 | scripted WOA diff |",
        "|" + "---|" * 12,
    ]
    out = {"stats": stats, "costs": costs, "by_lambda": {}}
    for lam in args.lambdas:
        a, b = stats.get(f"{lam}_E25", {}), stats.get(f"{lam}_E400", {})
        runs = sorted(set(a) & set(b))
        if not runs:
            continue
        diff = np.array([b[r]["first_attempt"] - a[r]["first_attempt"] for r in runs])
        se = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else float("nan")

        def mean(c, key):
            return float(np.nanmean([c[r][key] for r in runs]))

        s25, s400 = SCRIPTED.get(lam, (np.nan, np.nan))
        opt = sum(br_by_lam[lam]["blend"])
        csum = {k: [c[r]["c_sum"] for r in runs] for k, c in (("25", a), ("400", b))}
        out["by_lambda"][lam] = dict(
            runs=runs, optimal_c_sum=opt, c_sum25=csum["25"], c_sum400=csum["400"],
            first25=mean(a, "first_attempt"), first400=mean(b, "first_attempt"),
            first_diff=float(diff.mean()), first_diff_se=float(se), first_diff_runs=diff.tolist(),
            attempts25=mean(a, "attempts"), attempts400=mean(b, "attempts"),
            mov25=mean(a, "movement"), mov400=mean(b, "movement"),
            mse25=mean(a, "agreed_mse"), mse400=mean(b, "agreed_mse"),
        )
        rows.append(
            f"| {lam} | {opt:.2f} | {np.nanmean(csum['25']):.2f} / {np.nanmean(csum['400']):.2f} | "
            f"{mean(a, 'outside'):.0%} / {mean(b, 'outside'):.0%} | "
            f"{mean(a, 'first_attempt'):.0%} / {mean(b, 'first_attempt'):.0%} "
            f"(paired diff {diff.mean():+.2f} ± {se:.2f}) | "
            f"{mean(a, 'attempts'):.2f} / {mean(b, 'attempts'):.2f} | "
            f"{sum(a[r]['failed'] for r in runs)} / {sum(b[r]['failed'] for r in runs)} | "
            f"{mean(a, 'messages'):.1f} / {mean(b, 'messages'):.1f} | "
            f"{mean(a, 'movement'):.2f} / {mean(b, 'movement'):.2f} | "
            f"{mean(a, 'agreed_mse'):.2f} / {mean(b, 'agreed_mse'):.2f} | "
            f"{mean(a, 'opening_mse'):.2f} / {mean(b, 'opening_mse'):.2f} | {s400 - s25:+.2f} |"
        )
    rows += [
        "",
        "Weight of advice per twin (on the agreed value; the pair's two values add up to 1).",
        "",
        "| λ | WOA A / B, studied 25 | WOA A / B, studied 400 | WOA on own first submission, 25 / "
        "400 | at the midpoint (\\|WOA − 0.5\\| ≤ 0.05), 25 / 400 | mean \\|WOA − 0.5\\|, 25 / 400 "
        "| uneven queries where the more accurate twin conceded less, 25 / 400 "
        "| scripted S1n WOA, 25 / 400 |",
        "|" + "---|" * 8,
    ]
    for lam in args.lambdas:
        a, b = stats.get(f"{lam}_E25", {}), stats.get(f"{lam}_E400", {})
        runs = sorted(set(a) & set(b))
        if not runs:
            continue

        def mean(c, key):
            return float(np.nanmean([c[r][key] for r in runs]))

        def pooled(c):  # better_held pooled over pairs, with its count
            n = sum(c[r]["n_uneven"] for r in runs)
            k = sum(c[r]["better_held"] * c[r]["n_uneven"] for r in runs
                    if c[r]["n_uneven"])
            return f"{k / n:.0%} of {n}" if n else "—"

        s25, s400 = SCRIPTED.get(lam, (np.nan, np.nan))
        out["by_lambda"][lam].update(
            woa25=(mean(a, "woa_A"), mean(a, "woa_B")), woa400=(mean(b, "woa_A"), mean(b, "woa_B")),
            midpoint25=mean(a, "midpoint"), midpoint400=mean(b, "midpoint"),
        )
        rows.append(
            f"| {lam} | {mean(a, 'woa_A'):.2f} / {mean(a, 'woa_B'):.2f} "
            f"| {mean(b, 'woa_A'):.2f} / {mean(b, 'woa_B'):.2f} "
            f"| {(mean(a, 'woa_first_A') + mean(a, 'woa_first_B')) / 2:.2f} / "
            f"{(mean(b, 'woa_first_A') + mean(b, 'woa_first_B')) / 2:.2f} "
            f"| {mean(a, 'midpoint'):.0%} / {mean(b, 'midpoint'):.0%} "
            f"| {mean(a, 'woa_dev'):.2f} / {mean(b, 'woa_dev'):.2f} "
            f"| {pooled(a)} / {pooled(b)} | {s25:.2f} / {s400:.2f} |"
        )
    run_out = make_run_dir(str(ROOT / "results"), "sessions_lambda")
    (run_out / "stats.json").write_text(json.dumps(out, indent=2))
    table = "\n".join(rows) + f"\n\ncost per cell: {json.dumps({k: round(v, 2) for k, v in costs.items()})}"
    (run_out / "table.md").write_text(table + "\n")
    if out["by_lambda"]:
        plot_sessions_lambda(out["by_lambda"], run_out / "sessions_lambda.png")
    print(table)
    print(f"saved to {run_out}")
    return run_out


if __name__ == "__main__":
    main()
