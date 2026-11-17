"""LLM twins in the controlled design (LogBook note 2026-11-11, "More controlled experiment on
differences"): does their entrenchment follow the disagreement of their solo opinions?

Usage:
    python scripts/lambda_llm.py [--lambdas 0.0 0.333 0.6 0.9] [--arms E25 E400] [--runs 4]
        [--linear results/lambda_entrench/<timestamp>] [--lin-ss 20 500]

Reads the latest run of results/llm_haiku_lambda_lam<lambda>_<arm>_r<r> (configs/
llm_haiku_lambda.yaml). Per run: weight of advice (as in onset_llm.py, twin-averaged) and the
mean squared disagreement of the two opening estimates. Entrenchment = the paired change in WOA
from the short to the long study arm. The linear reference is the change in movement toward the
best response between two solo-phase lengths (training-posterior prior) from lambda_entrench.py.

Writes results/lambda_llm/<timestamp>/ (stats.json, lambda_llm.png).
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

from onset_llm import latest, run_stats  # noqa: E402

from consensus.plotting_consensus import plot_lambda_llm  # noqa: E402
from consensus.utils import load_run, make_run_dir  # noqa: E402


def main(argv=None) -> Path:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--lambdas", nargs="+", default=["0.0", "0.333", "0.6", "0.9"])
    p.add_argument("--arms", nargs="+", default=["E25", "E400"])
    p.add_argument("--runs", type=int, default=4)
    p.add_argument("--min-gap", type=float, default=0.2)
    p.add_argument("--prefix", default="llm_haiku_lambda")
    p.add_argument("--linear", default=None, help="lambda_entrench run folder (default: latest)")
    p.add_argument("--lin-ss", type=int, nargs=2, default=[20, 500])
    p.add_argument("--results", default=str(ROOT / "results"))
    args = p.parse_args(argv)
    res = Path(args.results)

    woa = {a: np.full((len(args.lambdas), args.runs), np.nan) for a in args.arms}
    gap = {a: np.full((len(args.lambdas), args.runs), np.nan) for a in args.arms}
    ns = {}
    for j, lam in enumerate(args.lambdas):
        for arm in args.arms:
            for r in range(args.runs):
                run = latest(res / f"{args.prefix}_lam{lam}_{arm}_r{r}")
                if run is None:
                    print(f"missing lambda={lam} {arm} r{r}")
                    continue
                cfg, _, _ = load_run(run)
                ns[arm] = next(pc for pc in cfg["protocols"] if pc.get("label") == arm)["self_n_A"]
                steps = [
                    json.loads(line) for line in (run / "transcripts" / f"{arm}_run{r}.jsonl").open()
                ]
                st = run_stats(steps, cfg["env"]["sigma"] ** 2, args.min_gap, 10, cfg["post"]["T"])
                woa[arm][j, r] = st["woa"]
                gap[arm][j, r] = np.mean(
                    [(s["rounds"][0]["A"]["estimate"] - s["rounds"][0]["B"]["estimate"]) ** 2
                     for s in steps]
                )
    lams = np.array([float(x) for x in args.lambdas])
    theory_gap = 2 * (1 - lams**2) * (1 + lams)  # aligned design, raw weights

    lin_dir = Path(args.linear) if args.linear else sorted((res / "lambda_entrench").glob("*"))[-1]
    lin = json.loads((lin_dir / "summary.json").read_text())
    tp = lin["results"]["aligned"]["training posterior"]
    s0, s1 = args.lin_ss
    lin_ent = np.array(tp[f"s={s1}"]["movement_share"]) - np.array(tp[f"s={s0}"]["movement_share"])

    d = woa[args.arms[-1]] - woa[args.arms[0]]
    stats = {
        "lambdas": args.lambdas,
        "theory_gap": np.round(theory_gap, 3).tolist(),
        "linear_run": str(lin_dir),
        "linear_lambdas": lin["lambdas"],
        f"linear_movement_s{s1}_minus_s{s0}": np.round(lin_ent, 3).tolist(),
        "woa_paired_diff": {
            "mean": np.round(np.nanmean(d, 1), 3).tolist(),
            "se": np.round(
                np.nanstd(d, 1, ddof=1) / np.sqrt(np.isfinite(d).sum(1)), 3
            ).tolist(),
            "per_run": np.round(d, 3).tolist(),
        },
        **{f"woa_{a}": np.round(woa[a], 3).tolist() for a in args.arms},
        **{f"gap_{a}": np.round(gap[a], 3).tolist() for a in args.arms},
    }
    out = make_run_dir(res, "lambda_llm")
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    fig, _ = plot_lambda_llm(lams, args.arms, [ns[a] for a in args.arms], woa, gap, theory_gap,
                             np.array(lin["lambdas"]), lin_ent, f"{s1}) − movement({s0}")
    fig.savefig(out / "lambda_llm.png", dpi=150, bbox_inches="tight")
    print(json.dumps({k: stats[k] for k in ("woa_paired_diff",
                                            f"linear_movement_s{s1}_minus_s{s0}")}))
    print("saved to", out)
    return out


if __name__ == "__main__":
    main()
