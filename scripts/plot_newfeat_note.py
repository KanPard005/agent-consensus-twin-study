"""Figures for the new-features note (LogBook agent-consensus, 2026-11-19).

Usage:
    python scripts/plot_newfeat_note.py --out <LogBook>/projects/agent-consensus/figures

Reads results/boolean_haiku_newfeat and results/boolean_haiku_newfeat_atpartner (10 Haiku pairs
each) and writes, with the numbers behind them (newfeat-note-numbers.json):
  2026-11-19-newfeat-cases.png  share of finals saying 1 per case type, both arms
  2026-11-19-newfeat-time.png   own new clause the only zero: share saying 1 per block of 20
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

from newfeat_analysis import BLOCK, case_type, load  # noqa: E402

from consensus.plotting_newfeat import plot_new_zero_over_time, plot_share_by_case  # noqa: E402

ARMS = {"new at solo": "boolean_haiku_newfeat", "new at partner": "boolean_haiku_newfeat_atpartner"}
CASES = [("own studied zero", "own studied clause is 0"),
         ("partner studied zero", "partner's studied clause is the only 0"),
         ("own new zero", "own new clause is the only 0"),
         ("partner new zero", "partner's new clause is the only 0"),
         ("all ones", "all clauses are 1")]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    out = Path(p.parse_args().out)
    numbers: dict = {"cases": {}, "time": {}}
    # The Bayes-optimal label says 1 on every all-ones case and on no case with a zero clause.
    rows = [{"label": lab, "bayes": 1.0 if ct == "all ones" else 0.0} for ct, lab in CASES]
    for arm, folder in ARMS.items():
        solo, partner, _ = load(ROOT / "results" / folder)
        fin = {ct: [] for ct, _ in CASES}
        for r in partner:
            ct = case_type(r["clauses"], r["twin"])
            if ct is not None and not np.isnan(r["final"][1]):
                fin[ct].append(r["final"][1])
        numbers["cases"][arm] = {ct: [float(np.mean(v)), len(v)] for ct, v in fin.items()}
        for row, (ct, _) in zip(rows, CASES):
            row[arm] = float(np.mean(fin[ct]))
        # own new clause the only zero: solo predictions, then partner finals, per block
        curve = []
        for recs, key, idx in ((solo, lambda r: r["label"], "i"),
                               (partner, lambda r: r["final"][1], "t")):
            for b in range(5):
                v = [key(r) for r in recs if r[idx] // BLOCK == b
                     and case_type(r["clauses"], r["twin"]) == "own new zero"
                     and not np.isnan(key(r))]
                curve.append(float(np.mean(v)))
        numbers["time"][arm] = curve
    fig, _ = plot_share_by_case(rows, list(ARMS))
    fig.savefig(out / "2026-11-19-newfeat-cases.png", dpi=200)
    fig, _ = plot_new_zero_over_time(numbers["time"], 5, BLOCK, 0.0)
    fig.savefig(out / "2026-11-19-newfeat-time.png", dpi=200)
    (ROOT / "results" / "newfeat-note-numbers.json").write_text(json.dumps(numbers, indent=2))
    print(json.dumps(numbers, indent=1))


if __name__ == "__main__":
    main()
