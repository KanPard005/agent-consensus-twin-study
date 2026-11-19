"""Analysis of boolean-task runs (LogBook note 2026-11-18): scores against the Bayes references,
deference split by the twin's own clause value, talk and dataset use.

Usage:
    python scripts/boolean_analysis.py results/boolean_haiku [more experiment folders ...]
        [--out results/boolean_haiku/analysis]

Per twin and partner query, with D_own = 1 iff every clause the twin sees in full is true
(the AND of those clause values; NaN when it sees no clause in full, as in the mixed and
interleaved splits) and D_other the same for the partner. Which clauses a twin sees in full is
read from the run's config.yaml (env.clauses, env.views):
- when D_own = 0 the twin alone knows y = 0 up to noise (Bayes P(y=1) = 1 - rho);
- when D_own = 1 and every clause is seen in full by one twin (the clean two-clause split), the
  answer is the partner's: the Bayes joint label is D_other. With a clause split between the twins
  (three-clause block split) D_other = 0 still means y = 0, but D_other = 1 leaves the split
  clause open, so use "partner_needed" for the full comparison with Bayes.
So "uses the partner" is measured where D_own = 1: the share of finals whose label equals
D_other (1.0 for the joint Bayes rule in the clean split; the solo Bayes rule always says 1 there
when the clauses are independent and p_clause > 0.5, so it matches D_other only when
D_other = 1).
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
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consensus.boolean_data import log_loss  # noqa: E402
from consensus.plotting_boolean import plot_boolean_summary  # noqa: E402

# A twin's own feature values in a message: "x3=1", "a2: 0", or a whole triple such as
# "a = 110", "b=(1,0,0)", "x1..x3 = 0,1,0" (the forms the twins use; the single-name form alone
# missed the triples and undercounted, e.g. 0.07 vs 0.13 in the uncorrelated block run).
FEATURE_VALUE = re.compile(
    r"\b[a-z]\d+\s*[=:]\s*[01]\b"
    r"|\b[a-z](?:\d\s*(?:-|–|\.\.)\s*[a-z]?\d)?\s*=\s*\(?\s*[01]\s*,?\s*[01]\s*,?\s*[01]")


def seen_clauses(run_dir: Path) -> dict[str, list[int]]:
    """Indices of the clauses each twin sees in full, from the run's config.

    A twin can be sure that y = 0 only through a clause whose features it all sees, so these
    are the clauses that define D_own. (Before 2026-11-18 this was hard-coded as clause 0 for A
    and clause 1 for B, which is right only for the clean two-clause split.)
    """
    env = yaml.safe_load((run_dir / "config.yaml").read_text())["env"]
    return {k: [c for c, feats in enumerate(env["clauses"]) if set(feats) <= set(view)]
            for k, view in env["views"].items()}


def clause_and(values: list[int], idx: list[int]) -> float:
    """AND of the given clause values; NaN when the twin sees no clause in full."""
    return float(all(values[c] for c in idx)) if idx else float("nan")


def load_partner(run_dir: Path) -> list[dict]:
    """One row per (run, partner query, twin)."""
    seen = seen_clauses(run_dir)
    rows = []
    for f in sorted((run_dir / "transcripts").glob("run*.jsonl")):
        m = re.fullmatch(r"run(\d+)\.jsonl", f.name)
        if not m:
            continue
        for line in f.read_text().splitlines():
            rec = json.loads(line)
            for k, o in (("A", "B"), ("B", "A")):
                msgs = [mm["text"] for mm in rec["messages"] if mm["from"] == k]
                rows.append({
                    "dir": run_dir.name, "run": int(m.group(1)), "t": rec["t"], "twin": k,
                    "y": rec["y"], "pi": rec["pi"], "d_own": clause_and(rec["clauses"], seen[k]),
                    "d_other": clause_and(rec["clauses"], seen[o]),
                    "p_open": rec["opening"][k][0], "l_open": rec["opening"][k][1],
                    "p_final": rec["final"][k][0], "l_final": rec["final"][k][1],
                    "p_partner_open": rec["opening"][o][0],
                    "p_bayes_solo": rec["p_bayes_solo"][k], "p_bayes_joint": rec["p_bayes_joint"],
                    "n_msgs": len(msgs),
                    "sends_features": any(bool(FEATURE_VALUE.search(t)) for t in msgs),
                })
    return rows


def load_solo(run_dir: Path) -> list[dict]:
    seen = seen_clauses(run_dir)
    rows = []
    for f in sorted((run_dir / "transcripts").glob("run*_solo.jsonl")):
        r = int(re.match(r"run(\d+)_", f.name).group(1))
        for line in f.read_text().splitlines():
            rec = json.loads(line)
            k = rec["twin"]
            rows.append({"dir": run_dir.name, "run": r, "i": rec["i"], "twin": k, "y": rec["y"],
                         "p": rec["p"], "label": rec["label"],
                         "p_bayes_solo": rec["p_bayes_solo"],
                         "d_own": clause_and(rec["clauses"], seen[k])})
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
        if not s.any():  # no clause seen in full (mixed / interleaved split)
            out[f"d_own={d}"] = {"n": 0}
            continue
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
    out["solo_rule_eq_d_other_d_own1"] = (float(np.mean(d_oth[d_own == 1] == 1))
                                          if (d_own == 1).any() else None)
    block = t // 10
    # Split-agnostic deference (needed for the mixed split, where clause values no longer say
    # who knows what): "partner needed" = queries where the Bayes label from the twin's own
    # features differs from the Bayes label from all features. There the final label should
    # follow the joint label; elsewhere the partner should not change a correct own call.
    l_solo, l_joint = (ps > 0.5) * 1.0, (pj > 0.5) * 1.0
    need = l_solo != l_joint
    out["partner_needed"] = {
        "share": float(need.mean()),
        "open_eq_joint": float(np.mean(lo[need] == l_joint[need])),
        "final_eq_joint": float(np.mean(lf[need] == l_joint[need])),
        "final_eq_joint_elsewhere": float(np.mean(lf[~need] == l_joint[~need])),
        "acc_final": float(np.mean(lf[need] == y[need])),
        "acc_final_elsewhere": float(np.mean(lf[~need] == y[~need])),
    }
    out["final_eq_joint_by_block_needed"] = {
        int(b): float(np.mean(lf[s] == l_joint[s]))
        for b in np.unique(block) for s in [(block == b) & need] if s.any()}
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
