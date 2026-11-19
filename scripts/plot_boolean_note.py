"""Figures for the boolean-task summary note (LogBook agent-consensus, 2026-11-19).

Usage:
    python scripts/plot_boolean_note.py --out <LogBook>/projects/agent-consensus/figures

Reads the saved Haiku runs in results/ (pairs: transcripts/run<r>.jsonl; single agents:
transcripts/run<r>_solo.jsonl) and writes three PNGs plus the numbers behind them
(boolean-note-numbers.json):
  2026-11-19-boolean-accuracy.png   pairs vs single agent vs Bayes, per condition
  2026-11-19-boolean-zero.png       own vs partner's zero clause, per condition
  2026-11-19-boolean-gap.png        finals minus Bayes over the 100 partner queries, per split
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

from boolean_analysis import load_partner  # noqa: E402

from consensus.plotting_boolean_note import (  # noqa: E402
    plot_accuracy_by_condition,
    plot_gap_over_time,
    plot_zero_clause,
)

RES = ROOT / "results"

# (label, pairs experiment, single-agent experiment, number of solo queries K)
ACCURACY = [
    ("2 clauses, clean split, 40 rows, 20+40 q", "boolean_haiku", "boolean_haiku_single", 20),
    ("2 clauses, clean split, 400 rows, 20+40 q", "boolean_haiku_n400",
     "boolean_haiku_single_n400", 20),
    ("2 clauses, clean split, 400 rows, 100+100 q", "boolean_haiku_n400_q100",
     "boolean_haiku_single_n400_q100", 100),
    ("2 clauses, mixed 2+1 split", "boolean_haiku_mixed", "boolean_haiku_single_n400_q100", 100),
    ("3 clauses, interleaved split", "boolean_haiku_cnf3_inter", "boolean_haiku_cnf3_single", 100),
    ("3 clauses, block split", "boolean_haiku_cnf3_blocks", "boolean_haiku_cnf3_single", 100),
    ("3 clauses, block, ρ = 0.75", "boolean_haiku_cnf3_blocks_rho75",
     "boolean_haiku_cnf3_single_rho75", 100),
    ("3 clauses, block, ρ = 0.75, 1600 rows", "boolean_haiku_cnf3_blocks_rho75_n1600",
     "boolean_haiku_cnf3_single_rho75_n1600", 100),
]

ZERO = [
    ("2 clauses, 40 rows, 20+40 q", "boolean_haiku"),
    ("2 clauses, 400 rows, 20+40 q", "boolean_haiku_n400"),
    ("2 clauses, 400 rows, 100+100 q", "boolean_haiku_n400_q100"),
    ("3 clauses, block, ρ = 0.9", "boolean_haiku_cnf3_blocks"),
    ("3 clauses, block, ρ = 0.75", "boolean_haiku_cnf3_blocks_rho75"),
    ("3 clauses, block, ρ = 0.75, 1600 rows", "boolean_haiku_cnf3_blocks_rho75_n1600"),
    ("3 clauses, block, noise only in clause 2 (1 pair)", "boolean_haiku_cnf3_blocks_c2noise"),
]

GAP = [
    ("2 clauses, clean", "boolean_haiku_n400_q100"),
    ("2 clauses, mixed 2+1", "boolean_haiku_mixed"),
    ("3 clauses, interleaved", "boolean_haiku_cnf3_inter"),
    ("3 clauses, block", "boolean_haiku_cnf3_blocks"),
]


def run_dirs(exp: str) -> list[Path]:
    return [d for d in sorted((RES / exp).glob("20*")) if (d / "summary.json").exists()]


def partner_rows(exp: str) -> list[dict]:
    """Partner-phase rows (one per query and twin) with a submitted final."""
    rows = [r for d in run_dirs(exp) for r in load_partner(d)]
    return [r for r in rows if r["l_final"] == r["l_final"]]


def single_acc(exp: str, K: int) -> float:
    """Single agents' label accuracy on queries K+1.. (the pairs' partner-phase cases)."""
    ok = []
    for d in run_dirs(exp):
        for f in (d / "transcripts").glob("run*_solo.jsonl"):
            for line in f.read_text().splitlines():
                r = json.loads(line)
                if r["i"] >= K:
                    ok.append(r["label"] == r["y"])
    return float(np.mean(ok))


def mean(rows: list[dict], fn) -> float:
    return float(np.mean([fn(r) for r in rows]))


def accuracy_rows() -> list[dict]:
    out = []
    for label, pairs, single, K in ACCURACY:
        p = partner_rows(pairs)
        out.append({
            "label": label, "n_pairs": len(run_dirs(pairs)) and len({(r["dir"], r["run"])
                                                                     for r in p}),
            "pairs": mean(p, lambda r: r["l_final"] == r["y"]),
            "single": single_acc(single, K),
            "bayes_own": mean(p, lambda r: (r["p_bayes_solo"] > 0.5) == r["y"]),
            "bayes_all": mean(p, lambda r: (r["p_bayes_joint"] > 0.5) == r["y"]),
        })
    return out


def zero_rows() -> list[dict]:
    out = []
    for label, exp in ZERO:
        p = partner_rows(exp)
        own = [r for r in p if r["d_own"] == 0]
        partner = [r for r in p if r["d_own"] == 1 and r["d_other"] == 0]
        out.append({"label": label, "n_pairs": len({(r["dir"], r["run"]) for r in p}),
                    "own": mean(own, lambda r: r["l_final"]),
                    "partner": mean(partner, lambda r: r["l_final"]),
                    "partner_opening": mean(partner, lambda r: r["l_open"]),
                    "n_own": len(own), "n_partner": len(partner)})
    return out


def gap_curves(block: int = 20) -> dict:
    out = {}
    for label, exp in GAP:
        p = partner_rows(exp)
        xs, gs = [], []
        for b in range(100 // block):
            s = [r for r in p if r["t"] // block == b]
            xs.append(block * b + block / 2)
            gs.append(mean(s, lambda r: r["l_final"] == r["y"])
                      - mean(s, lambda r: (r["p_bayes_joint"] > 0.5) == r["y"]))
        out[label] = (xs, gs)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    acc, zero, gap = accuracy_rows(), zero_rows(), gap_curves()
    for name, (fig, _) in (("accuracy", plot_accuracy_by_condition(acc)),
                           ("zero", plot_zero_clause(zero)),
                           ("gap", plot_gap_over_time(gap))):
        fig.savefig(out / f"2026-11-19-boolean-{name}.png", dpi=160, bbox_inches="tight",
                    facecolor="#fcfcfb")
    numbers = {"accuracy": acc, "zero": zero, "gap": {k: list(v) for k, v in gap.items()}}
    (out / "boolean-note-numbers.json").write_text(json.dumps(numbers, indent=1))
    print(json.dumps(numbers, indent=1))


if __name__ == "__main__":
    main()
