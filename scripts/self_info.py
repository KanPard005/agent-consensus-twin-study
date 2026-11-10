"""What the self-learning twins are told about each other (S1 2x2): how much they defer, and to what effect.

Usage:
    python scripts/self_info.py results/llm_haiku_self_rho0.0 results/llm_haiku_self_info_rho0.0 ...
        [--tail 0.5] [--min-gap 0.2]

Takes the latest run in each folder, groups the self-learning arms (labels starting with "S")
by rho, and reports per (rho, arm), pooled over runs:

- excess MSE over sigma^2 of the team (last `tail` of the steps), and its paired difference
  from P0 (no talk) on the same streams;
- excess MSE of the round-1 estimates (each twin's own, self-learned opinion), all steps;
- unchanged: share of round-2 estimates equal to the twin's own round-1 estimate;
- weight of advice (WOA, judge-advisor studies): (round 2 - own round 1) / (partner round 1 -
  own round 1), only where the twins' round-1 estimates differ by more than `min-gap`, clipped
  to [-0.5, 1.5]. 0 = ignore the partner, 0.5 = meet halfway, 1 = adopt the partner's estimate.
  This is the deference the 2x2 is about; the normative answer at rho = 0 is not 0.5 but
  "add the information", which for two half-signal opinions means moving beyond the partner;
- partner-features: share of messages that refer to the partner's features by a name
  ("your x2" with shared names; the other prefix, e.g. "b2" from A, with distinct names);
- twin talk: share of messages mentioning "twin" / "same model" / "copy".

All paired comparisons use the same seed and replicate indices across folders.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

TWIN_TALK = re.compile(r"\btwin\b|same model|copy of|same AI", re.IGNORECASE)


def latest_run(path: Path) -> Path:
    """The folder itself if it is a run, else its latest timestamped run."""
    if (path / "config.yaml").exists():
        return path
    return sorted(p for p in path.iterdir() if (p / "config.yaml").exists())[-1]


def partner_feature_pattern(name: str, distinct: bool) -> re.Pattern:
    """Regex for a message from twin `name` that names the partner's features."""
    if distinct:
        other = "b" if name == "A" else "a"
        return re.compile(rf"\b{other}[1-9][0-9]?\b", re.IGNORECASE)
    return re.compile(r"\byour\s+x[1-9][0-9]?\b", re.IGNORECASE)


def arm_stats(run_dir: Path, label: str, cfg: dict, tail: float, min_gap: float) -> dict:
    """Per-arm statistics for one run folder, pooled over its replicates."""
    sigma2 = cfg["env"]["sigma"] ** 2
    pc = next(p for p in cfg["protocols"] if p.get("label") == label)
    distinct = pc.get("feature_names", "shared") == "distinct"
    z = np.load(run_dir / "metrics.npz")
    T = z[f"{label}/sq_err"].shape[1]
    t0 = int(round(T * (1 - tail)))
    team = z[f"{label}/sq_err"][:, t0:].mean(axis=1) - sigma2
    p0 = z["P0/sq_err"][:, t0:].mean(axis=1) - sigma2
    solo, unchanged, woa, n_msgs, partner_feat, twin_talk = [], [], [], 0, 0, 0
    pats = {n: partner_feature_pattern(n, distinct) for n in "AB"}
    for f in sorted((run_dir / "transcripts").glob(f"{label}_run*.jsonl")):
        if not re.fullmatch(rf"{re.escape(label)}_run\d+\.jsonl", f.name):
            continue  # training-phase notes
        for line in f.open():
            s = json.loads(line)
            r1, r2 = s["rounds"][0], s["rounds"][-1]
            for me, other in (("A", "B"), ("B", "A")):
                own, par, fin = r1[me]["estimate"], r1[other]["estimate"], r2[me]["estimate"]
                solo.append((own - s["y"]) ** 2 - sigma2)
                unchanged.append(abs(fin - own) < 1e-9)
                if abs(par - own) > min_gap:
                    woa.append(np.clip((fin - own) / (par - own), -0.5, 1.5))
                for rnd in s["rounds"]:
                    msg = rnd[me]["message"]
                    n_msgs += 1
                    partner_feat += bool(pats[me].search(msg))
                    twin_talk += bool(TWIN_TALK.search(msg))
    return {
        "runs": len(team),
        "team": team,
        "team_minus_p0": team - p0,
        "solo": float(np.mean(solo)),
        "unchanged": float(np.mean(unchanged)),
        "woa": float(np.mean(woa)) if woa else float("nan"),
        "woa_median": float(np.median(woa)) if woa else float("nan"),
        "woa_n": len(woa),
        "partner_feat": partner_feat / max(n_msgs, 1),
        "twin_talk": twin_talk / max(n_msgs, 1),
        "cost": float(
            json.loads((run_dir / "summary.json").read_text())["protocols"][label].get(
                "cost_usd_total", float("nan")
            )
        ),
    }


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("folders", nargs="+")
    p.add_argument("--tail", type=float, default=0.5, help="share of final steps for team MSE")
    p.add_argument("--min-gap", type=float, default=0.2, help="min round-1 gap for WOA")
    args = p.parse_args(argv)

    table: dict[float, dict[str, dict]] = defaultdict(dict)
    for folder in args.folders:
        run_dir = latest_run(Path(folder))
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
        rho = cfg["env"]["rho"]
        for pc in cfg["protocols"]:
            lab = pc.get("label", "")
            if pc.get("learner") == "self":
                table[rho][lab] = arm_stats(run_dir, lab, cfg, args.tail, args.min_gap)

    head = (
        "| ρ | arm | runs | team excess | team − P0 (worse in) | round-1 excess | unchanged "
        "| WOA mean / median (n) | partner features | twin talk | cost |"
    )
    print(head)
    print("|---" * (head.count("|") - 1) + "|")
    out = {}
    for rho in sorted(table):
        for lab in sorted(table[rho]):
            s = table[rho][lab]
            d = s["team_minus_p0"]
            se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
            print(
                f"| {rho} | {lab} | {s['runs']} | {s['team'].mean():.2f} | "
                f"{d.mean():+.2f} ± {se:.2f} ({(d > 0).sum()}/{len(d)}) | {s['solo']:.2f} | "
                f"{s['unchanged']:.0%} | {s['woa']:.2f} / {s['woa_median']:.2f} ({s['woa_n']}) | "
                f"{s['partner_feat']:.0%} | {s['twin_talk']:.0%} | ${s['cost']:.2f} |"
            )
            out[(rho, lab)] = s
    return out


if __name__ == "__main__":
    main()
