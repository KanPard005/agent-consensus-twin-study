"""Analysis of boolean-task runs (LogBook note 2026-11-18): scores against the Bayes references,
deference split by the twin's own clause value, talk and dataset use.

Usage:
    python scripts/boolean_analysis.py results/boolean_haiku [more experiment folders ...]
        [--out results/boolean_haiku/analysis]

Per twin and partner query, with D_own the OR of the twin's own clause and D_other the
partner's:
- when D_own = 0 the twin alone knows y = 0 up to noise (Bayes P(y=1) = 1 - rho);
- when D_own = 1 the answer is the partner's clause: the Bayes joint label is D_other.
So "uses the partner" is measured where D_own = 1: the share of finals whose label equals
D_other (1.0 for the joint Bayes rule; the solo Bayes rule always says 1 there when the clauses
are independent and p_clause > 0.5, so it matches D_other only when D_other = 1).
Writes partner_queries.json, summary.json and a figure.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consensus.boolean_data import log_loss  # noqa: E402
from consensus.plotting_boolean import plot_boolean_summary  # noqa: E402

FEATURE_VALUE = re.compile(r"\bx\d+\s*[=:]\s*[01]\b")


def load_partner(run_dir: Path) -> list[dict]:
    """One row per (run, partner query, twin)."""
    rows = []
    for f in sorted((run_dir / "transcripts").glob("run*.jsonl")):
        m = re.fullmatch(r"run(\d+)\.jsonl", f.name)
        if not m:
            continue
        for line in f.read_text().splitlines():
            rec = json.loads(line)
            for k, o, own_c in (("A", "B", 0), ("B", "A", 1)):
                msgs = [mm["text"] for mm in rec["messages"] if mm["from"] == k]
                rows.append({
                    "dir": run_dir.name, "run": int(m.group(1)), "t": rec["t"], "twin": k,
                    "y": rec["y"], "pi": rec["pi"], "d_own": rec["clauses"][own_c],
                    "d_other": rec["clauses"][1 - own_c],
                    "p_open": rec["opening"][k][0], "l_open": rec["opening"][k][1],
                    "p_final": rec["final"][k][0], "l_final": rec["final"][k][1],
                    "p_partner_open": rec["opening"][o][0],
                    "p_bayes_solo": rec["p_bayes_solo"][k], "p_bayes_joint": rec["p_bayes_joint"],
                    "n_msgs": len(msgs),
                    "sends_features": any(bool(FEATURE_VALUE.search(t)) for t in msgs),
                })
    return rows


def load_solo(run_dir: Path) -> list[dict]:
    rows = []
    for f in sorted((run_dir / "transcripts").glob("run*_solo.jsonl")):
        r = int(re.match(r"run(\d+)_", f.name).group(1))
        for line in f.read_text().splitlines():
            rec = json.loads(line)
            k = rec["twin"]
            rows.append({"dir": run_dir.name, "run": r, "i": rec["i"], "twin": k, "y": rec["y"],
                         "p": rec["p"], "label": rec["label"],
                         "p_bayes_solo": rec["p_bayes_solo"],
                         "d_own": rec["clauses"][0 if k == "A" else 1]})
    return rows


def tool_use(run_dir: Path) -> list[dict]:
    """Per run and twin: tool-call counts by phase and add_case calls with extra values."""
    rows = []
    for f in sorted((run_dir / "transcripts").glob("run*_calls_*.jsonl")):
        m = re.fullmatch(r"run(\d+)_calls_([AB])\.jsonl", f.name)
        calls = [json.loads(line) for line in f.read_text().splitlines()]
        rows.append({"dir": run_dir.name, "run": int(m.group(1)), "twin": m.group(2),
                     "by_phase": {ph: dict(Counter(c["tool"] for c in calls if c["phase"] == ph))
                                  for ph in ("train", "solo", "query")},
                     "add_case_with_values": sum(1 for c in calls if c["tool"] == "add_case"
                                                 and c["args"].get("values"))})
    return rows


def col(rows: list[dict], key: str) -> np.ndarray:
    return np.array([r[key] for r in rows], float)


def scores(p: np.ndarray, label: np.ndarray, y: np.ndarray) -> dict:
    ok = ~np.isnan(p)
    if not ok.any():
        return {"n": 0}
    return {"n": int(ok.sum()), "acc": float(np.mean(label[ok] == y[ok])),
            "logloss": float(log_loss(p[ok], y[ok]).mean())}


def summarize(part: list[dict], solo: list[dict]) -> dict:
    y = col(part, "y")
    ps, pj = col(part, "p_bayes_solo"), col(part, "p_bayes_joint")
    lo, lf = col(part, "l_open"), col(part, "l_final")
    d_own, d_oth, t = col(part, "d_own"), col(part, "d_other"), col(part, "t")
    out = {
        "pairs": len({(r["dir"], r["run"]) for r in part}),
        "solo": scores(col(solo, "p"), col(solo, "label"), col(solo, "y")),
        "solo_bayes": scores(col(solo, "p_bayes_solo"), (col(solo, "p_bayes_solo") > 0.5) * 1.0,
                             col(solo, "y")),
        "opening": scores(col(part, "p_open"), lo, y),
        "final": scores(col(part, "p_final"), lf, y),
        "bayes_solo": scores(ps, (ps > 0.5) * 1.0, y),
        "bayes_joint": scores(pj, (pj > 0.5) * 1.0, y),
        "msgs_per_query": float(col(part, "n_msgs").mean()),
        "share_sending_features": float(col(part, "sends_features").mean()),
        "share_final_label_changed": float(np.mean(lf != lo)),
    }
    for d in (0, 1):
        s = d_own == d
        out[f"d_own={d}"] = {
            "n": int(s.sum()),
            "acc_open": float(np.mean(lo[s] == y[s])),
            "acc_final": float(np.mean(lf[s] == y[s])),
            "mean_p_open": float(np.nanmean(col(part, "p_open")[s])),
            "mean_p_final": float(np.nanmean(col(part, "p_final")[s])),
            "final_label_eq_d_other": float(np.mean(lf[s] == d_oth[s])),
        }
    # The own-features Bayes rule says 1 whenever D_own = 1 (independent clauses, p_clause > 0.5),
    # so it matches D_other there exactly when D_other = 1: the bar to beat for "uses the partner".
    out["solo_rule_eq_d_other_d_own1"] = float(np.mean(d_oth[d_own == 1] == 1))
    block = t // 10
    out["final_acc_by_block"] = {int(b): float(np.mean(lf[block == b] == y[block == b]))
                                 for b in np.unique(block)}
    out["eq_d_other_by_block_d_own1"] = {
        int(b): float(np.mean(lf[s] == d_oth[s]))
        for b in np.unique(block) for s in [(block == b) & (d_own == 1)] if s.any()}
    return out


def per_twin(part: list[dict]) -> list[dict]:
    keys = sorted({(r["dir"], r["run"], r["twin"]) for r in part})
    out = []
    for key in keys:
        rows = [r for r in part if (r["dir"], r["run"], r["twin"]) == key]
        y, lf = col(rows, "y"), col(rows, "l_final")
        s = col(rows, "d_own") == 1
        out.append({"dir": key[0], "run": key[1], "twin": key[2], "n": len(rows),
                    "acc_open": float(np.mean(col(rows, "l_open") == y)),
                    "acc_final": float(np.mean(lf == y)),
                    "eq_d_other_d_own1": float(np.mean(lf[s] == col(rows, "d_other")[s]))
                    if s.any() else None,
                    "msgs": float(col(rows, "n_msgs").mean()),
                    "sends_features": float(col(rows, "sends_features").mean())})
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("experiments", nargs="+")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    dirs = [d for e in args.experiments for d in sorted(Path(e).glob("20*"))
            if (d / "summary.json").exists()]
    part = [r for d in dirs for r in load_partner(d)]
    solo = [r for d in dirs for r in load_solo(d)]
    out = Path(args.out or Path(args.experiments[0]) / "analysis")
    out.mkdir(parents=True, exist_ok=True)
    (out / "partner_queries.json").write_text(json.dumps(part))
    summ = summarize(part, solo)
    summ["per_twin"] = per_twin(part)
    summ["tool_use"] = [r for d in dirs for r in tool_use(d)]
    (out / "summary.json").write_text(json.dumps(summ, indent=2))
    fig, _ = plot_boolean_summary(summ)
    fig.savefig(out / "boolean_summary.png", dpi=150, bbox_inches="tight")
    print(json.dumps({k: v for k, v in summ.items() if k not in ("per_twin", "tool_use")},
                     indent=1))
    for r in summ["per_twin"]:
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
    for r in summ["tool_use"]:
        print(r)


if __name__ == "__main__":
    main()
