"""LLM consensus game (C4) vs the averaging twins (S1n) and linear baselines, per rho.

Usage:
    python scripts/consensus_analysis.py [--rhos 0.0 0.5 0.9] [--window 10] [--min-gap 0.2]

Finds the latest run of results/llm_haiku_consensus_rho<rho>_r<r> (C4, C, P0, P1, P2) for each
replicate, and the S1n baseline either in results/llm_haiku_mediated_rho<rho> (same seed and
streams, runs 0-2) or in results/llm_haiku_consensus_S1n_rho<rho>_r<r>. Per (arm, rho) it
reports, as mean +- s.e. over replicates:

- agreement: share of steps where the twins agreed within tol, by round 1, 2 and 4 (C4) or
  by round 3 (C, linear twins scored by the same game);
- team excess loss over sigma^2 on the tail (last half): the game loss with penalties (C4, C),
  and the squared error of the agreed value alone; for S1n, P0, P1 the plain team excess;
- meeting point (`concede_A`, 0 = A's opening, 1 = B's) and the share outside the openings;
- crossings: share of non-agreeing round transitions where the twins swapped sides
  (sign of A - B flips), the failure mode of simultaneous moves;
- over time (windows of `window` steps): weight of advice in round 1 -> 2, WOA =
  (round 2 - own round 1) / (partner round 1 - own round 1) for gaps above `min-gap`, clipped
  to [-0.5, 1.5] (C4 steps that agreed in round 1 have no round 2 and are skipped); rounds
  used; the opening gap |A - B| in round 1.

Writes results/consensus_analysis/<timestamp>/ (stats.json, llm_consensus.png).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consensus.plotting_consensus import plot_llm_consensus  # noqa: E402
from consensus.utils import load_run, make_run_dir  # noqa: E402


def latest(folder: Path) -> Path | None:
    if not folder.exists():
        return None
    runs = sorted(p for p in folder.iterdir() if (p / "metrics.npz").exists())
    return runs[-1] if runs else None


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def step_woa(s: dict, min_gap: float) -> list[float]:
    if len(s["rounds"]) < 2:
        return []
    r1, r2 = s["rounds"][0], s["rounds"][1]
    out = []
    for me, other in (("A", "B"), ("B", "A")):
        own, par, fin = r1[me]["estimate"], r1[other]["estimate"], r2[me]["estimate"]
        if abs(par - own) > min_gap:
            out.append(float(np.clip((fin - own) / (par - own), -0.5, 1.5)))
    return out


def transcript_stats(steps: list[dict], window: int, min_gap: float, T: int) -> dict:
    """Per-window WOA, rounds and opening gap; crossings; all steps' meeting points."""
    n_w = T // window
    woa, rounds, gap = (np.full(n_w, np.nan) for _ in range(3))
    by_w = {k: ([], [], []) for k in range(n_w)}
    crossings, transitions, concede = 0, 0, []
    for s in steps:
        k = min(s["t"] // window, n_w - 1)
        R = s["rounds"]
        by_w[k][0].extend(step_woa(s, min_gap))
        by_w[k][1].append(len(R))
        by_w[k][2].append(abs(R[0]["A"]["estimate"] - R[0]["B"]["estimate"]))
        diffs = [rd["A"]["estimate"] - rd["B"]["estimate"] for rd in R]
        for a, b in zip(diffs[:-1], diffs[1:]):
            transitions += 1
            crossings += int(np.sign(a) * np.sign(b) < 0)
        if s.get("agreed"):
            gap0 = R[0]["B"]["estimate"] - R[0]["A"]["estimate"]
            if abs(gap0) >= 0.1:
                concede.append((s["consensus"] - R[0]["A"]["estimate"]) / gap0)
    for k, (w, r, g) in by_w.items():
        woa[k] = np.mean(w) if w else np.nan
        rounds[k] = np.mean(r) if r else np.nan
        gap[k] = np.mean(g) if g else np.nan
    return dict(
        woa=woa,
        rounds=rounds,
        open_gap=gap,
        crossing=crossings / transitions if transitions else np.nan,
        concede=np.array(concede, float),
    )


def main(argv=None) -> Path:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--rhos", nargs="+", type=float, default=[0.0, 0.5, 0.9])
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--min-gap", type=float, default=0.2)
    p.add_argument("--results", default=str(ROOT / "results"))
    args = p.parse_args(argv)
    res = Path(args.results)

    stats: dict = {lab: {} for lab in ("P0", "P1", "P2", "C", "C4", "S1n")}
    time_stats: dict = {"C4": {}, "S1n": {}}
    table = {}
    for rho in args.rhos:
        acc = {lab: {} for lab in stats}
        tacc = {"C4": [], "S1n": []}
        for r in range(args.runs):
            run = latest(res / f"llm_haiku_consensus_rho{rho}_r{r}")
            if run is None:
                print(f"missing C4 run rho={rho} r={r}")
                continue
            cfg, R, bench = load_run(run)
            s2, T = cfg["env"]["sigma"] ** 2, cfg["post"]["T"]
            tail = slice(T // 2, T)
            for lab in ("P0", "P1", "P2", "C", "C4"):
                a = acc[lab]
                a.setdefault("excess", []).append(R[lab]["sq_err"][0, tail].mean() - s2)
                if lab in ("C", "C4"):
                    ar = R[lab]["agree_round"][0]
                    a.setdefault("agreed", []).append(R[lab]["agreed"][0].mean())
                    a.setdefault("game_excess", []).append(R[lab]["game_loss"][0, tail].mean() - s2)
                    a.setdefault("agreed_excess", []).append(
                        np.nanmean(R[lab]["agreed_sq_err"][0, tail]) - s2
                    )
                    a.setdefault("agree_r1", []).append(np.mean(ar <= 0))
                    a.setdefault("agree_r2", []).append(np.mean(ar <= 1))
                    a.setdefault("rounds_used", []).append(R[lab]["rounds_used"][0].mean())
            for lab in ("C4",):
                steps = read_jsonl(run / "transcripts" / f"C4_run{r}.jsonl")
                ts = transcript_stats(steps, args.window, args.min_gap, T)
                tacc["C4"].append(ts)
                acc["C4"].setdefault("concede_all", []).append(ts["concede"])
                acc["C4"].setdefault("crossing", []).append(ts["crossing"])
                acc["C4"].setdefault("cost", []).append(float(np.nansum(R["C4"]["cost_usd"])))
                acc["C4"].setdefault("parse_fail", []).append(
                    float(np.nansum(R["C4"]["parse_fail"]))
                )
            # S1n baseline: the mediator run (same seed and streams) or a dedicated run.
            med = latest(res / f"llm_haiku_mediated_rho{rho}")
            src, row = (
                (med, r)
                if med is not None
                else (latest(res / f"llm_haiku_consensus_S1n_rho{rho}_r{r}"), 0)
            )
            if src is None:
                print(f"missing S1n rho={rho} r={r}")
                continue
            cfg_s, R_s, _ = load_run(src)
            acc["S1n"].setdefault("excess", []).append(R_s["S1n"]["sq_err"][row, tail].mean() - s2)
            steps = read_jsonl(src / "transcripts" / f"S1n_run{r}.jsonl")
            tacc["S1n"].append(transcript_stats(steps, args.window, args.min_gap, T))
        for lab, a in acc.items():
            stats[lab][rho] = {k: (v if k == "concede_all" else np.array(v)) for k, v in a.items()}
        for lab, lst in tacc.items():
            if lst:
                time_stats[lab][rho] = {
                    "centers": (np.arange(len(lst[0]["woa"])) + 0.5) * args.window,
                    **{k: np.stack([x[k] for x in lst], 1) for k in ("woa", "rounds", "open_gap")},
                }
        table[rho] = {
            lab: {
                k: [round(float(np.mean(v)), 3), round(float(np.std(v) / np.sqrt(len(v))), 3)]
                for k, v in a.items()
                if k != "concede_all" and len(v)
            }
            for lab, a in acc.items()
        }
        c = np.concatenate(acc["C4"].get("concede_all", [np.array([])]))
        c = c[np.isfinite(c)]
        table[rho]["C4"]["concede_median"] = round(float(np.median(c)), 3) if c.size else None
        table[rho]["C4"]["outside_openings"] = (
            round(float(np.mean((c < -0.02) | (c > 1.02))), 3) if c.size else None
        )
        for lab in ("C4", "S1n"):
            if rho in time_stats[lab]:
                w = time_stats[lab][rho]["woa"]
                table[rho][lab]["woa_by_window"] = np.round(np.nanmean(w, 1), 3).tolist()
                table[rho][lab]["open_gap_by_window"] = np.round(
                    np.nanmean(time_stats[lab][rho]["open_gap"], 1), 3
                ).tolist()
        if rho in time_stats["C4"]:
            table[rho]["C4"]["rounds_by_window"] = np.round(
                np.nanmean(time_stats["C4"][rho]["rounds"], 1), 2
            ).tolist()

    out = make_run_dir(res, "consensus_analysis")
    (out / "stats.json").write_text(json.dumps(table, indent=1))
    fig, _ = plot_llm_consensus(args.rhos, stats, time_stats)
    fig.savefig(out / "llm_consensus.png", dpi=150, bbox_inches="tight")
    print(json.dumps(table, indent=1))
    print("saved to", out)
    return out


if __name__ == "__main__":
    main()
