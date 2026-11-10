"""Entrenchment: does a self-learning twin defer less, and state narrower intervals, the more cases it studied?

Usage:
    python scripts/entrench.py results/llm_haiku_entrench_E25 results/llm_haiku_entrench_E100 ...
        [--min-gap 0.2] [--early 5]

Takes the latest run in each folder and reports, for every self-learning arm and each twin
(A, B), pooled over replicates and steps:

- n: cases the twin studied (`self_n_A` / `self_n_B`);
- round-1 excess: excess squared error over sigma^2 of the twin's own round-1 estimate (how good
  its self-learned opinion is);
- unchanged: share of round-2 estimates equal to its round-1 estimate;
- weight of advice, WOA = (round 2 - own round 1) / (partner round 1 - own round 1), where the
  round-1 gap exceeds `min-gap`, clipped to [-0.5, 1.5] (0 = ignore the partner, 1 = adopt the
  partner's estimate); also over the first `early` steps only, before feedback adds much data;
- 80% interval of the round-1 estimate: mean width and coverage (share containing y; honest =
  80%), and mean width in round 2 (does talking narrow it);
- team excess (both twins, so the same in the A and B rows).

The reference width is that of an honest 80% interval given one block's features, at rho = 0:
2 * 1.2816 * sqrt(|w_partner|^2 + sigma^2), the same for every n (the solo model's own estimation
error adds a little at small n).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import yaml

Z80 = 1.2816  # standard normal 90% quantile: an 80% central interval is +-1.2816 sd


def latest_run(path: Path) -> Path:
    """The folder itself if it is a run, else its latest timestamped run."""
    if (path / "config.yaml").exists():
        return path
    return sorted(p for p in path.iterdir() if (p / "config.yaml").exists())[-1]


def twin_stats(steps: list[dict], me: str, sigma2: float, min_gap: float, early: int) -> dict:
    """Deference and confidence statistics of twin `me` over a list of transcript steps."""
    other = "B" if me == "A" else "A"
    solo, unchanged, woa, woa_early, w1, cov1, w2 = [], [], [], [], [], [], []
    for s in steps:
        r1, r2 = s["rounds"][0], s["rounds"][-1]
        own, par, fin = r1[me]["estimate"], r1[other]["estimate"], r2[me]["estimate"]
        solo.append((own - s["y"]) ** 2 - sigma2)
        unchanged.append(abs(fin - own) < 1e-9)
        if abs(par - own) > min_gap:
            w = float(np.clip((fin - own) / (par - own), -0.5, 1.5))
            woa.append(w)
            if s["t"] < early:
                woa_early.append(w)
        if "interval" in r1[me]:
            lo, hi = r1[me]["interval"]
            w1.append(hi - lo)
            cov1.append(lo <= s["y"] <= hi)
        if "interval" in r2[me]:
            lo, hi = r2[me]["interval"]
            w2.append(hi - lo)

    def m(v: list) -> float:
        return float(np.mean(v)) if v else float("nan")

    return {
        "solo": m(solo),
        "unchanged": m(unchanged),
        "woa": m(woa),
        "woa_median": float(np.median(woa)) if woa else float("nan"),
        "woa_n": len(woa),
        "woa_early": m(woa_early),
        "woa_early_n": len(woa_early),
        "width1": m(w1),
        "cover1": m(cov1),
        "width2": m(w2),
    }


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("folders", nargs="+")
    p.add_argument("--min-gap", type=float, default=0.2, help="min round-1 gap for WOA")
    p.add_argument("--early", type=int, default=5, help="steps counted as early")
    args = p.parse_args(argv)

    head = (
        "| arm | twin | n | round-1 excess | unchanged | WOA mean / median (n) | WOA early (n) "
        "| 80% width r1 / r2 | coverage r1 | team excess |"
    )
    print(head)
    print("|---" * (head.count("|") - 1) + "|")
    out, ref = {}, None
    for folder in args.folders:
        run_dir = latest_run(Path(folder))
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
        env = cfg["env"]
        sigma2 = env["sigma"] ** 2
        if env["rho"] == 0:
            ref = {
                "A": 2 * Z80 * np.sqrt(env["norm_B"] ** 2 + sigma2),
                "B": 2 * Z80 * np.sqrt(env["norm_A"] ** 2 + sigma2),
            }
        z = np.load(run_dir / "metrics.npz")
        for pc in cfg["protocols"]:
            lab = pc.get("label", "")
            if pc.get("learner") != "self":
                continue
            steps = []
            for f in sorted((run_dir / "transcripts").glob(f"{lab}_run*.jsonl")):
                if re.fullmatch(rf"{re.escape(lab)}_run\d+\.jsonl", f.name):
                    steps += [json.loads(line) for line in f.open()]
            team = float(z[f"{lab}/sq_err"].mean() - sigma2)
            for me in "AB":
                s = twin_stats(steps, me, sigma2, args.min_gap, args.early)
                s["n"] = pc.get(f"self_n_{me}") or cfg["train"][f"n_{me}"]
                s["team"] = team
                out[(lab, me)] = s
                print(
                    f"| {lab} | {me} | {s['n']} | {s['solo']:.2f} | {s['unchanged']:.0%} | "
                    f"{s['woa']:.2f} / {s['woa_median']:.2f} ({s['woa_n']}) | "
                    f"{s['woa_early']:.2f} ({s['woa_early_n']}) | "
                    f"{s['width1']:.2f} / {s['width2']:.2f} | {s['cover1']:.0%} | {team:.2f} |"
                )
    if ref:
        print(f"\nHonest 80% width given one block (rho = 0): A {ref['A']:.2f}, B {ref['B']:.2f}")
    return out


if __name__ == "__main__":
    main()
