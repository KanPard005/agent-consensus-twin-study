"""Single-agent baseline vs the pairs on the same cases (LogBook 2026-11-18, boolean task).

Usage:
    python scripts/boolean_single_analysis.py results/boolean_haiku_single results/boolean_haiku

Run r of the single agent answers the same 20 solo cases and then the same 40 cases (queries
21-60) as pair r, seeing all six features. Reports, per run and pooled, accuracy and log loss of
the single agent on both blocks, the joint Bayes reference on the same cases, the pairs' finals
on queries 21-60 (twin A and B), and the single agent's accuracy by clause cell (D1, D2).
Writes summary.json into <single experiment>/analysis/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consensus.boolean_data import log_loss  # noqa: E402


def runs(exp: Path, name: str) -> dict[int, Path]:
    """{run index: transcript file} for every finished run of an experiment."""
    out = {}
    for d in sorted(exp.glob("20*")):
        if not (d / "summary.json").exists():
            continue
        for r in json.loads((d / "summary.json").read_text()):
            out[int(r[3:])] = d / "transcripts" / name.format(r=int(r[3:]))
    return out


def score(p, lab, y) -> dict:
    p, lab, y = map(np.asarray, (p, lab, y))
    return {"n": int(len(y)), "acc": float(np.mean(lab == y)),
            "logloss": float(log_loss(p, y).mean())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("single")
    ap.add_argument("pairs")
    args = ap.parse_args()
    single = runs(Path(args.single), "run{r}_solo.jsonl")
    pairs = runs(Path(args.pairs), "run{r}.jsonl")
    K1 = 20
    pooled = {"first": [], "rest": [], "pair_A": [], "pair_B": []}
    per_run = []
    cells: dict[str, list] = {}
    for r, f in sorted(single.items()):
        recs = sorted((json.loads(line) for line in f.read_text().splitlines()),
                      key=lambda x: x["i"])
        row = {"run": r}
        for name, sl in (("first", recs[:K1]), ("rest", recs[K1:])):
            s = score([x["p"] for x in sl], [x["label"] for x in sl], [x["y"] for x in sl])
            b = score([x["p_bayes_joint"] for x in sl],
                      [float(x["p_bayes_joint"] > 0.5) for x in sl], [x["y"] for x in sl])
            row[name], row[name + "_bayes"] = s["acc"], b["acc"]
            pooled[name] += sl
        for x in recs[K1:]:
            cells.setdefault(str(tuple(x["clauses"])), []).append(x["label"] == x["y"])
        if r in pairs and pairs[r].exists():
            precs = [json.loads(line) for line in pairs[r].read_text().splitlines()]
            for k in "AB":
                row[f"pair_{k}"] = float(np.mean([p["final"][k][1] == p["y"] for p in precs]))
                pooled[f"pair_{k}"] += [(p["final"][k][0], p["final"][k][1], p["y"])
                                        for p in precs]
        per_run.append(row)
    out = {"runs": per_run}
    for name in ("first", "rest"):
        sl = pooled[name]
        out[name] = score([x["p"] for x in sl], [x["label"] for x in sl], [x["y"] for x in sl])
        out[name + "_bayes_joint"] = score(
            [x["p_bayes_joint"] for x in sl], [float(x["p_bayes_joint"] > 0.5) for x in sl],
            [x["y"] for x in sl])
    for k in "AB":
        v = pooled[f"pair_{k}"]
        if v:
            out[f"pairs_final_{k}"] = score(*zip(*v))
    out["rest_acc_by_clauses"] = {c: {"n": len(v), "acc": float(np.mean(v))}
                                  for c, v in sorted(cells.items())}
    dest = Path(args.single) / "analysis"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
