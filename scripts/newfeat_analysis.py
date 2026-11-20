"""Analysis of the new-features runs (LogBook note 2026-11-19, "Boolean experiments: further
extensions"): does a twin use features it did not study, and does it treat its own new clause
differently from the partner's information?

Usage:
    python scripts/newfeat_analysis.py [results/boolean_haiku_newfeat results/...]

Task: y = (a1 v a2) & (a3 v a4) & (b1 v b2) & (b3 v b4), with noise. Twin A studies clause 0
(a1, a2) and later also sees clause 1 (a3, a4); twin B studies clause 2 and later sees clause 3.
For a twin whose studied clause is 1, y = 0 (up to noise) when exactly one other clause is 0:

  - "own new zero": its own new clause (it saw it, nobody studied it);
  - "partner new zero": the partner's new clause (the partner saw it, nobody studied it);
  - "partner studied zero": the partner's studied clause (the partner learned it in training).

All three mean y = 0 equally (P(y = 1) = 1 - rho); the share of predictions saying 1 in each
measures how much the twin uses that information. The first two separate the source (own or
partner) from whether anyone studied the feature. "Own studied zero" (the twin's own studied
clause 0, any others) is what it learned in training; "all ones" (every clause 1, y = 1 up to
noise) is the reference for how often a twin says 1 when nothing argues against it.

Prints, per experiment folder: accuracy against the Bayes references, the share saying 1 per case
type and block of 20 queries (solo predictions; partner openings and finals), and how often
twins call add_case / count after studying, and count with a new feature in the condition.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OWN = {"A": (0, 1), "B": (2, 3)}  # (studied clause, new clause) per twin
PARTNER = {"A": (2, 3), "B": (0, 1)}
NEW_NAMES = {"A": ("a3", "a4"), "B": ("b3", "b4")}
BLOCK = 20


CASE_TYPES = ("own studied zero", "own new zero", "partner new zero", "partner studied zero",
              "all ones")


def case_type(clauses: list[int], k: str) -> str | None:
    """The case type for twin k (see the module docstring), or None if more than one of the
    other clauses is 0."""
    old, new = (clauses[i] for i in OWN[k])
    p_old, p_new = (clauses[i] for i in PARTNER[k])
    if not old:
        return "own studied zero"
    zeros = [name for name, v in (("own new zero", new), ("partner studied zero", p_old),
                                  ("partner new zero", p_new)) if not v]
    if not zeros:
        return "all ones"
    return zeros[0] if len(zeros) == 1 else None


def load(exp: Path) -> tuple[list[dict], list[dict], list[dict]]:
    """Solo records, partner records (one per twin and query) and tool calls of every run."""
    solo, partner, calls = [], [], []
    for tr in sorted(exp.glob("*/transcripts")):
        for f in tr.glob("run*_solo.jsonl"):
            run = f"{tr.parent.name}/{f.name.split('_')[0]}"
            solo += [{**json.loads(l), "run": run} for l in f.open()]
        for f in tr.glob("run[0-9]*.jsonl"):
            if "_" in f.stem:
                continue
            for l in f.open():
                r = json.loads(l)
                for k in "AB":
                    partner.append({"run": f"{tr.parent.name}/{f.stem}", "twin": k, "t": r["t"],
                                    "y": r["y"], "clauses": r["clauses"],
                                    "opening": r["opening"][k], "final": r["final"][k],
                                    "p_solo": r["p_bayes_solo"][k],
                                    "p_study": r["p_bayes_study"][k],
                                    "p_joint": r["p_bayes_joint"]})
        for f in tr.glob("run*_calls_*.jsonl"):
            k = f.stem[-1]
            calls += [{**json.loads(l), "twin": k, "run": f"{tr.parent.name}/{f.stem[:4]}"}
                      for l in f.open()]
    return solo, partner, calls


def share_one(rows: list[dict], key, block_of) -> dict:
    """{case type: [share of labels = 1 per block (n)]} for the rows' predictions `key`."""
    out: dict[str, list[str]] = {}
    for ct in CASE_TYPES:
        cells = []
        for b in range(5):
            lab = [key(r) for r in rows if r["ct"] == ct and block_of(r) == b
                   and not np.isnan(key(r))]
            cells.append(f"{np.mean(lab):.2f} ({len(lab)})" if lab else "  -  ")
        out[ct] = cells
    return out


def acc(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((pred > 0.5) == (y == 1)))


def report(exp: Path) -> None:
    solo, partner, calls = load(exp)
    n_runs = len({r["run"] for r in partner})
    print(f"\n######## {exp.name}: {n_runs} pairs")
    for r in solo:
        r["ct"] = case_type(r["clauses"], r["twin"])
    for r in partner:
        r["ct"] = case_type(r["clauses"], r["twin"])

    # Accuracy against the references (Bayes from studied features / all own / all eight).
    ys = np.array([r["y"] for r in solo])
    print("solo   : twins %.3f | Bayes studied %.3f, own %.3f" % (
        acc(np.array([r["label"] for r in solo]) * 1.0, ys),
        acc(np.array([r["p_bayes_study"] for r in solo]), ys),
        acc(np.array([r["p_bayes_solo"] for r in solo]), ys)))
    yp = np.array([r["y"] for r in partner])
    for name in ("opening", "final"):
        lab = np.array([r[name][1] for r in partner])
        ok = ~np.isnan(lab)
        print(f"{name:7s}: twins {acc(lab[ok], yp[ok]):.3f}", end="")
        if name == "final":
            print(" | Bayes studied %.3f, own %.3f, all %.3f" % tuple(
                acc(np.array([r[k] for r in partner]), yp) for k in ("p_study", "p_solo",
                                                                      "p_joint")), end="")
        print()

    # Share of labels = 1 per case type and block of 20 queries.
    print(f"\nshare saying 1 (n), blocks of {BLOCK} queries")
    for title, rows, key, blk in (
            ("solo predictions", solo, lambda r: r["label"], lambda r: r["i"] // BLOCK),
            ("partner openings", partner, lambda r: r["opening"][1], lambda r: r["t"] // BLOCK),
            ("partner finals", partner, lambda r: r["final"][1], lambda r: r["t"] // BLOCK)):
        print(f"  {title}")
        for ct, cells in share_one(rows, key, blk).items():
            print(f"    {ct:20s} " + "  ".join(f"{c:>10s}" for c in cells))

    # Data-tool use after studying: add_case / count / new_dataset, and count conditions that
    # name a new feature.
    print("\ntool use after studying (twins with >= 1 call / calls)")
    twins = {(c["run"], c["twin"]) for c in calls}
    for phase in ("solo", "query"):
        for tool in ("add_case", "new_dataset", "count"):
            cs = [c for c in calls if c["phase"] == phase and c["tool"] == tool
                  and not c["result"].startswith("Error")]
            who = {(c["run"], c["twin"]) for c in cs}
            line = f"  {phase:5s} {tool:11s}: {len(who):2d}/{len(twins)} twins, {len(cs):4d} calls"
            if tool == "count":
                new = [c for c in cs if re.search(
                    r"\b(%s)\b" % "|".join(NEW_NAMES[c["twin"]]), str(c["args"]))]
                line += (f"; naming a new feature: {len({(c['run'], c['twin']) for c in new})} "
                         f"twins, {len(new)} calls")
            print(line)


def main(argv: list[str]) -> None:
    exps = [Path(a) for a in argv] or [ROOT / "results" / "boolean_haiku_newfeat",
                                       ROOT / "results" / "boolean_haiku_newfeat_atpartner"]
    for exp in exps:
        report(exp)


if __name__ == "__main__":
    main(sys.argv[1:])
