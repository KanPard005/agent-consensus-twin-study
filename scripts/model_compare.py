"""Compare models on the same live-session cell, pair by pair (LogBook 2026-11-14).

Each experiment folder holds runs of the same config with a different model or effort; pair r
sees the same data in every folder, so the comparison is paired. Per pair and model:
- MSE of the openings and of the finals, against the sum and the average of the two openings
  (at lambda = 0 the sum is the population-optimal combination);
- the best combination of the two openings in the pair, y ≈ b_own · own + b_partner · partner
  (pooled over both twins; population value 1 / (1 + lambda) each at alpha = 1), and its MSE;
- the fitted combination final ≈ c_own · own opening + c_partner · partner opening (pooled over
  both twins): (0.5, 0.5) is averaging, (1, 1) adding;
- the mean weight of advice, messages per query, query-phase tool calls (regress / model tools),
  and the share of a twin's visible text that mentions adding or summing.

Usage:
    python scripts/model_compare.py haiku=llm_haiku_sessions_asym_own_lam0.0_al1.0_eq400 \
        sonnet-low=sonnet_sessions_asym_own_lam0.0_al1.0_eq400 ...
Prints a table and writes results/model_compare/<timestamp>/{stats.json, table.md}.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consensus.sessions import weight_of_advice  # noqa: E402
from consensus.utils import make_run_dir  # noqa: E402

TOOLS = ("regress", "new_model", "add_case", "fit_model")
ADD = re.compile(r"\badd(itive|ing|ed)?\b|\bsum\b|\bsumm", re.I)


def read_jsonl(f: Path) -> list[dict]:
    return [json.loads(x) for x in f.read_text().splitlines() if x.strip()] if f.exists() else []


def load(experiment: str) -> dict[int, dict]:
    out = {}
    for d in sorted((ROOT / "results" / experiment).glob("20*")):
        sf = d / "summary.json"
        if not sf.exists():
            continue
        for key, s in json.loads(sf.read_text()).items():
            tr = d / "transcripts"
            out[int(key[3:])] = dict(
                recs=read_jsonl(tr / f"{key}.jsonl"), cost=s.get("cost_usd", 0.0),
                calls={k: read_jsonl(tr / f"{key}_calls_{k}.jsonl") for k in "AB"},
                texts={k: [m["text"] for m in read_jsonl(tr / f"{key}_messages_{k}.jsonl")
                           if m.get("type") in ("text", "thinking") and m.get("text")]
                       for k in "AB"})
    return out


def pair_stats(p: dict) -> dict:
    R = [s for s in p["recs"] if not s["failed"]]
    y = np.array([s["y"] for s in R])
    oA, oB = (np.array([s["opening"][k] for s in R]) for k in "AB")
    fA, fB = (np.array([s["final"][k] for s in R]) for k in "AB")
    woa = ([weight_of_advice(a, b, f) for a, b, f in zip(oA, oB, fA)]
           + [weight_of_advice(b, a, f) for a, b, f in zip(oA, oB, fB)])
    X = np.r_[np.c_[oA, oB], np.c_[oB, oA]]
    coef = np.linalg.lstsq(X, np.r_[fA, fB], rcond=None)[0]
    best = np.linalg.lstsq(X, np.r_[y, y], rcond=None)[0]  # the best combination of the openings
    texts = p["texts"]["A"] + p["texts"]["B"]
    return dict(
        n=len(R), cost=p["cost"],
        mse_open=float(np.mean(np.r_[(oA - y) ** 2, (oB - y) ** 2])),
        mse_final=float(np.mean(np.r_[(fA - y) ** 2, (fB - y) ** 2])),
        mse_sum=float(np.mean((oA + oB - y) ** 2)),
        mse_avg=float(np.mean(((oA + oB) / 2 - y) ** 2)),
        b_own=float(best[0]), b_partner=float(best[1]),
        mse_best=float(np.mean((X @ best - np.r_[y, y]) ** 2)),
        c_own=float(coef[0]), c_partner=float(coef[1]),
        woa=float(np.nanmean(woa)),
        msgs=float(np.mean([len(s["messages"]) for s in R])),
        tool_calls=sum(c["q"] >= 0 and c["tool"] in TOOLS for k in "AB" for c in p["calls"][k]),
        add_talk=float(np.mean([bool(ADD.search(t)) for t in texts])) if texts else float("nan"),
    )


def main(argv: list[str]) -> Path:
    arms = dict(a.split("=", 1) for a in argv)
    data = {name: {r: pair_stats(p) for r, p in load(exp).items()} for name, exp in arms.items()}
    pairs = sorted(set().union(*(d.keys() for d in data.values())))
    cols = ("mse_final", "c_own", "c_partner", "woa", "msgs", "tool_calls")
    lines = ["| pair | sum / average / best (b_own, b_partner) | " + " | ".join(
        f"{name}: final MSE, c_own, c_partner, WOA, msgs/query, tool calls" for name in arms) + " |",
             "|" + "---|" * (2 + len(arms))]
    for r in pairs:
        ref = next(d[r] for d in data.values() if r in d)
        cells = []
        for name in arms:
            s = data[name].get(r)
            cells.append("—" if s is None else ", ".join(
                f"{s[c]:.2f}" if isinstance(s[c], float) else str(s[c]) for c in cols))
        lines.append(f"| {r} | {ref['mse_sum']:.2f} / {ref['mse_avg']:.2f} / {ref['mse_best']:.2f} "
                     f"({ref['b_own']:.2f}, {ref['b_partner']:.2f}) | " + " | ".join(cells) + " |")
    lines += ["", "| model | pairs | final MSE (best combination) | c_own | c_partner "
              "| pairs adding (c_partner ≥ 0.7) "
              "| msgs/query | tool calls/pair | text mentioning add/sum | cost/pair |",
              "|" + "---|" * 10]
    for name in arms:
        S = list(data[name].values())
        if not S:
            continue
        m = lambda k: float(np.nanmean([s[k] for s in S]))  # noqa: E731
        lines.append(
            f"| {name} | {len(S)} | {m('mse_final'):.2f} ({m('mse_best'):.2f}) | {m('c_own'):.2f} "
            f"| {m('c_partner'):.2f} "
            f"| {sum(s['c_partner'] >= 0.7 for s in S)} | {m('msgs'):.2f} | {m('tool_calls'):.1f} "
            f"| {m('add_talk'):.0%} | ${m('cost'):.2f} |")
    run_out = make_run_dir(str(ROOT / "results"), "model_compare")
    (run_out / "stats.json").write_text(json.dumps({"arms": arms, "pairs": data}, indent=2))
    table = "\n".join(lines)
    (run_out / "table.md").write_text(table + "\n")
    print(table)
    print(f"saved to {run_out}")
    return run_out


if __name__ == "__main__":
    main(sys.argv[1:])
