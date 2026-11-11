"""Are self-learning LLM twins that start talking later (after studying more cases) more entrenched?

Usage:
    python scripts/onset_llm.py [--rhos 0.0 0.5] [--arms E25 E100 E400] [--runs 4]
        [--min-gap 0.2] [--window 10] [--prefix llm_haiku_onset] [--experiment onset_llm]

    d = 20: --prefix llm_haiku_onset_d20 --rhos 0.0 --arms E50 E200 E800
            --experiment onset_llm_d20

Reads the latest run of results/llm_haiku_onset_rho<rho>_<arm>_r<r> (configs/llm_haiku_entrench.yaml
with post.T = 30: S1n twins that studied the first 25 / 100 / 400 of the same 400 training cases,
then talk with the averaging rule, stating an 80% interval with each estimate). Per twin and run:

- round-1 excess: squared error of the twin's own opening estimate minus sigma^2;
- weight of advice, WOA = (round 2 - own round 1) / (partner round 1 - own round 1), for round-1
  gaps above `min-gap`, clipped to [-0.5, 1.5] (0 = ignores the partner, 1 = adopts its estimate;
  ~1 is the best response at rho = 0, where the partner's estimate is separate evidence);
- unchanged: share of round-2 estimates equal to the round-1 estimate;
- 80% interval of the opening estimate: width and coverage (share containing y);
- team excess (average of the final estimates).

Run values average the two twins; tables and plots give mean +- s.e. over runs, and WOA per
window of `window` steps. Writes results/onset_llm/<timestamp>/ (stats.json, onset_llm.png).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consensus.plotting_consensus import plot_onset_llm  # noqa: E402
from consensus.utils import load_run, make_run_dir  # noqa: E402

METRICS = ["solo_excess", "woa", "unchanged", "width", "coverage", "team_excess"]


def latest(folder: Path) -> Path | None:
    runs = sorted(p for p in folder.glob("*") if (p / "metrics.npz").exists())
    return runs[-1] if runs else None


def run_stats(steps: list[dict], sigma2: float, min_gap: float, window: int, T: int) -> dict:
    """Twin-averaged statistics of one run, plus WOA per window."""
    solo, woa, unch, width, cover = [], [], [], [], []
    n_w = T // window
    woa_w = [[] for _ in range(n_w)]
    for s in steps:
        r1, r2 = s["rounds"][0], s["rounds"][-1]
        for me, other in (("A", "B"), ("B", "A")):
            own, par, fin = r1[me]["estimate"], r1[other]["estimate"], r2[me]["estimate"]
            solo.append((own - s["y"]) ** 2 - sigma2)
            unch.append(abs(fin - own) < 1e-9)
            if abs(par - own) > min_gap:
                w = float(np.clip((fin - own) / (par - own), -0.5, 1.5))
                woa.append(w)
                woa_w[min(s["t"] // window, n_w - 1)].append(w)
            if "interval" in r1[me]:
                lo, hi = r1[me]["interval"]
                width.append(hi - lo)
                cover.append(lo <= s["y"] <= hi)

    def m(v):
        return float(np.mean(v)) if len(v) else float("nan")

    return dict(
        solo_excess=m(solo),
        woa=m(woa),
        unchanged=m(unch),
        width=m(width),
        coverage=m(cover),
        woa_window=[m(v) for v in woa_w],
    )


def main(argv=None) -> Path:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--rhos", nargs="+", type=float, default=[0.0, 0.5])
    p.add_argument("--arms", nargs="+", default=["E25", "E100", "E400"])
    p.add_argument("--runs", type=int, default=4)
    p.add_argument("--min-gap", type=float, default=0.2)
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--results", default=str(ROOT / "results"))
    p.add_argument("--prefix", default="llm_haiku_onset", help="run folders: <prefix>_rho<rho>_...")
    p.add_argument("--experiment", default="onset_llm", help="output subfolder")
    args = p.parse_args(argv)
    res = Path(args.results)

    # stats[rho][arm][metric] = per-run values; windows[rho][arm] = (runs, windows) WOA
    stats, windows, table, ns = {}, {}, {}, {}
    for rho in args.rhos:
        stats[rho], windows[rho], table[rho] = {}, {}, {}
        for arm in args.arms:
            per = {k: [] for k in METRICS}
            wins = []
            for r in range(args.runs):
                run = latest(res / f"{args.prefix}_rho{rho}_{arm}_r{r}")
                if run is None:
                    print(f"missing rho={rho} {arm} r{r}")
                    continue
                cfg, R, _ = load_run(run)
                s2, T = cfg["env"]["sigma"] ** 2, cfg["post"]["T"]
                pc = next(pc for pc in cfg["protocols"] if pc.get("label") == arm)
                ns[arm] = pc["self_n_A"]
                steps = [
                    json.loads(line)
                    for line in (run / "transcripts" / f"{arm}_run{r}.jsonl").open()
                ]
                st = run_stats(steps, s2, args.min_gap, args.window, T)
                st["team_excess"] = float(R[arm]["sq_err"][0].mean() - s2)
                for k in METRICS:
                    per[k].append(st[k])
                wins.append(st["woa_window"])
            stats[rho][arm] = {k: np.array(v) for k, v in per.items()}
            windows[rho][arm] = np.array(wins)
            table[rho][arm] = {
                k: [
                    round(float(np.nanmean(v)), 3),
                    round(float(np.nanstd(v) / np.sqrt(max(np.sum(np.isfinite(v)), 1))), 3),
                ]
                for k, v in per.items()
            }
            table[rho][arm]["woa_by_window"] = np.round(np.nanmean(wins, 0), 3).tolist()
            table[rho][arm]["n_studied"] = ns[arm]

    out = make_run_dir(res, args.experiment)
    (out / "stats.json").write_text(json.dumps(table, indent=1))
    fig, _ = plot_onset_llm(
        args.rhos, args.arms, [ns[a] for a in args.arms], stats, windows, args.window
    )
    fig.savefig(out / "onset_llm.png", dpi=150, bbox_inches="tight")
    for rho, d in table.items():
        print(f"rho={rho}")
        for arm, v in d.items():
            print(
                " ",
                arm,
                " ".join(f"{k} {v[k][0]:.2f}±{v[k][1]:.2f}" for k in METRICS),
                "woa windows",
                v["woa_by_window"],
            )
    print("saved to", out)
    return out


if __name__ == "__main__":
    main()
