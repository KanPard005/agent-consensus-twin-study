"""Figures for the boolean-task summary note (LogBook agent-consensus, 2026-11-19).

Usage:
    python scripts/plot_boolean_note.py --out <LogBook>/projects/agent-consensus/figures

Reads the saved Haiku runs in results/ (pairs: transcripts/run<r>.jsonl; single agents:
transcripts/run<r>_solo.jsonl) and writes three PNGs plus the numbers behind them
(boolean-note-numbers.json):
  2026-11-19-boolean-accuracy.png   pairs vs single agent vs Bayes, per condition
  2026-11-19-boolean-zero.png       own vs partner's zero clause, per condition
  2026-11-19-boolean-gap.png        finals minus Bayes over the 100 partner queries, per split
and, for the Opus follow-ups (one pair per condition, pair 0's cases unless noted):
  2026-11-19-boolean-opus-accuracy.png  Opus vs Haiku pairs and single agents, three-clause task
  2026-11-19-boolean-opus-zero.png      partner's zero clause, block split, Opus vs Haiku pair
  2026-11-19-boolean-opus-rule.png      Opus pairs, interleaved rho = 0.95: when the CNF is found
"""

from __future__ import annotations

import argparse
import json
import re
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
    plot_models_by_condition,
    plot_partner_zero_models,
    plot_rule_found,
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


# Opus follow-ups: (label, Opus pair, Haiku pair (run 0) or None, Opus single, Haiku single)
OPUS = [
    ("block, ρ = 0.75", "boolean_opus_cnf3_blocks_rho75", "boolean_haiku_cnf3_blocks_rho75",
     "boolean_opus_cnf3_single_rho75", "boolean_haiku_cnf3_single_rho75"),
    ("block, ρ = 0.9", "boolean_opus_cnf3_blocks", "boolean_haiku_cnf3_blocks",
     "boolean_opus_cnf3_single", "boolean_haiku_cnf3_single"),
    ("block, ρ = 0.95", "boolean_opus_cnf3_blocks_rho95", "boolean_haiku_cnf3_blocks_rho95",
     "boolean_opus_cnf3_single_rho95", "boolean_haiku_cnf3_single_rho95"),
    ("interleaved, ρ = 0.75", "boolean_opus_cnf3_inter_rho75", None,
     "boolean_opus_cnf3_single_rho75", "boolean_haiku_cnf3_single_rho75"),
    ("interleaved, ρ = 0.9", "boolean_opus_cnf3_inter", "boolean_haiku_cnf3_inter",
     "boolean_opus_cnf3_single", "boolean_haiku_cnf3_single"),
    ("interleaved, ρ = 0.95", "boolean_opus_cnf3_inter_rho95", None,
     "boolean_opus_cnf3_single_rho95", "boolean_haiku_cnf3_single_rho95"),
]
OPUS_SERIES = [("haiku_pair", "Haiku pair", "Haiku", "pair"),
               ("opus_pair", "Opus pair", "Opus", "pair"),
               ("haiku_single", "Haiku single agent", "Haiku", "single"),
               ("opus_single", "Opus single agent", "Opus", "single")]
# The exact CNF as the twins write it in messages: (x1|x2)&(x3|x4)&(x5|x6) and variants.
CNF = re.compile(r"\(?\s*x1\s*(?:OR|∨|\||v)\s*x2\s*\)?\s*(?:AND|∧|&)\s*\(?\s*x3\s*(?:OR|∨|\||v)"
                 r"\s*x4\s*\)?\s*(?:AND|∧|&)\s*\(?\s*x5\s*(?:OR|∨|\||v)\s*x6", re.I)


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


def single_acc_run(exp: str, run: int, K: int = 100) -> float | None:
    """One single agent's label accuracy on queries K+1.. ; None if that run does not exist."""
    ok = []
    for d in run_dirs(exp):
        f = d / "transcripts" / f"run{run}_solo.jsonl"
        if f.exists():
            ok += [r["label"] == r["y"] for r in map(json.loads, f.read_text().splitlines())
                   if r["i"] >= K]
    return float(np.mean(ok)) if ok else None


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


def opus_rows() -> list[dict]:
    """Pair 0 per condition: Opus and Haiku pairs' final accuracy, both single agents, and the
    Bayes band on those cases."""
    out = []
    for label, opus, haiku, opus_single, haiku_single in OPUS:
        o = [r for r in partner_rows(opus) if r["run"] == 0]
        h = [r for r in partner_rows(haiku) if r["run"] == 0] if haiku else []
        out.append({
            "label": label,
            "opus_pair": mean(o, lambda r: r["l_final"] == r["y"]),
            "haiku_pair": mean(h, lambda r: r["l_final"] == r["y"]) if h else None,
            "opus_single": single_acc_run(opus_single, 0),
            "haiku_single": single_acc_run(haiku_single, 0),
            "bayes_own": mean(o, lambda r: (r["p_bayes_solo"] > 0.5) == r["y"]),
            "bayes_all": mean(o, lambda r: (r["p_bayes_joint"] > 0.5) == r["y"]),
        })
    return out


def opus_zero_rows() -> list[dict]:
    """Block split, pair 0: finals (and openings) saying 1 when only the partner's clause is 0."""
    out = []
    for label, opus, haiku, *_ in OPUS[:3]:
        row = {"label": label}
        opening = []
        for model, exp in (("Opus", opus), ("Haiku", haiku)):
            p = [r for r in partner_rows(exp)
                 if r["run"] == 0 and r["d_own"] == 1 and r["d_other"] == 0]
            row[model] = mean(p, lambda r: r["l_final"])
            row[f"n_{model}"] = len(p)
            opening += [r["l_open"] for r in p]
        row["opening"] = float(np.mean(opening))
        out.append(row)
    return out


def rule_curves(exp: str = "boolean_opus_cnf3_inter_rho95", window: int = 20) -> dict:
    """Per Opus pair: rolling final accuracy (both twins) over the partner queries and the first
    query at which a message states the exact CNF; plus the pooled Bayes and 'either sure'
    accuracies (say 1 iff one twin's three features are all 1)."""
    curves, bayes, sure = {}, [], []
    for d in run_dirs(exp):
        for f in sorted((d / "transcripts").glob("run?.jsonl")):
            recs = [json.loads(line) for line in f.read_text().splitlines()]
            acc = [np.mean([r["final"][k][1] == r["y"] for k in "AB"]) for r in recs]
            roll = np.convolve(acc, np.ones(window) / window, mode="valid")
            found = next((r["number"] for r in recs for m in r["messages"]
                          if CNF.search(m["text"])), None)
            curves[f"pair {f.stem[3:]}"] = {"x": [r["number"] for r in recs][window - 1:],
                                            "acc": roll.tolist(), "found": found}
            for r in recs:
                x = r["x_full"]
                bayes.append((r["p_bayes_joint"] > 0.5) == r["y"])
                sure.append(int((x[0] and x[2] and x[4]) or (x[1] and x[3] and x[5])) == r["y"])
    return {"curves": dict(sorted(curves.items())), "bayes": float(np.mean(bayes)),
            "either_sure": float(np.mean(sure))}


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
    opus, opus_zero, rule = opus_rows(), opus_zero_rows(), rule_curves()
    for name, (fig, _) in (
            ("opus-accuracy", plot_models_by_condition(opus, OPUS_SERIES)),
            ("opus-zero", plot_partner_zero_models(opus_zero)),
            ("opus-rule", plot_rule_found(rule["curves"], rule["bayes"], rule["either_sure"]))):
        fig.savefig(out / f"2026-11-19-boolean-{name}.png", dpi=160, bbox_inches="tight",
                    facecolor="#fcfcfb")
    numbers = {"accuracy": acc, "zero": zero, "gap": {k: list(v) for k, v in gap.items()},
               "opus": opus, "opus_zero": opus_zero,
               "opus_rule": {"bayes": rule["bayes"], "either_sure": rule["either_sure"],
                             "found": {k: c["found"] for k, c in rule["curves"].items()}}}
    (out / "boolean-note-numbers.json").write_text(json.dumps(numbers, indent=1))
    print(json.dumps(numbers, indent=1))


if __name__ == "__main__":
    main()
