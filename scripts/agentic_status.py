"""Progress of agentic runs while they go (reads the live files of `scripts/run_agentic.py`).

Usage:
    python scripts/agentic_status.py [prefix ...]      # default: llm_haiku_agentic

For every results/<prefix>*/<timestamp>/transcripts/run<r>: per twin, the state, phase, query,
calls, spend, context and compactions (from the status files), and for the pair the queries
finished, messages, team MSE and the weight of advice so far (from run<r>.jsonl).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def woa(recs: list[dict], min_gap: float = 0.2) -> float:
    """Mean weight of advice (final - own) / (partner - own), clipped as in onset_llm.py."""
    v = []
    for s in recs:
        for me, o in (("A", "B"), ("B", "A")):
            own, par, fin = s["opening"][me], s["opening"][o], s["final"][me]
            if abs(par - own) > min_gap:
                v.append(float(np.clip((fin - own) / (par - own), -0.5, 1.5)))
    return float(np.mean(v)) if v else float("nan")


def main(prefixes: list[str]) -> None:
    total = 0.0
    for prefix in prefixes:
        for d in sorted((ROOT / "results").glob(f"{prefix}*/20*/transcripts")):
            for live in sorted(d.glob("run*.jsonl")):
                if not live.stem[3:].isdigit():
                    continue
                r = live.stem[3:]
                recs = [json.loads(x) for x in live.read_text().splitlines() if x.strip()]
                stats = {}
                for k in "AB":
                    f = d / f"run{r}_status_{k}.json"
                    stats[k] = json.loads(f.read_text()) if f.exists() else {}
                cost = sum(st.get("cost_usd", 0.0) for st in stats.values())
                total += cost
                y = np.array([s["y"] for s in recs])
                team = np.array([s["team"] for s in recs])
                mse = float(np.mean((team - y) ** 2)) if len(recs) else float("nan")
                n_msg = sum(len(s["messages"]) for s in recs)
                print(
                    f"{d.parent.parent.name}/{d.parent.name} run {r}: {len(recs)} queries done, "
                    f"{n_msg} messages, team MSE {mse:.3f}, WOA {woa(recs):.2f}, ${cost:.2f}"
                )
                for k, st in stats.items():
                    if st:
                        print(
                            f"    {k}: {st['state']}, {st['phase']} query {st['query']}/{st['T']}, "
                            f"train cases {st['training_cases']}, {st['calls']} calls, "
                            f"${st['cost_usd']:.2f}, context ~{st['context_tokens_est']}, "
                            f"{st['compactions']} compactions, updated {st['updated']}"
                        )
    print(f"total spend so far: ${total:.2f}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["llm_haiku_agentic"])
