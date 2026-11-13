"""Progress of session runs while they go (reads the live files of `scripts/run_sessions.py`).

Usage:
    python scripts/sessions_status.py [prefix ...]      # default: llm_haiku_sessions
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main(prefixes: list[str]) -> None:
    for prefix in prefixes:
        for d in sorted((ROOT / "results").glob(f"{prefix}*/20*/transcripts")):
            for live in sorted(d.glob("run*.jsonl")):
                if not live.stem[3:].isdigit():
                    continue
                r = live.stem[3:]
                recs = [json.loads(x) for x in live.read_text().splitlines() if x.strip()]
                ok = [s for s in recs if not s["failed"]]
                mse = np.mean([(s["agreed"] - s["y"]) ** 2 for s in ok]) if ok else float("nan")
                n_att = np.mean([len(s["attempts"]["A"]) for s in recs]) if recs else float("nan")
                print(
                    f"{d.parent.parent.name}/{d.parent.name} run {r}: {len(recs)} queries resolved, "
                    f"{sum(s['failed'] for s in recs)} failed, MSE of agreed {mse:.3f}, "
                    f"attempts per query {n_att:.2f}, "
                    f"messages {sum(len(s['messages']) for s in recs)}"
                )
                for k in "AB":
                    f = d / f"run{r}_status_{k}.json"
                    if f.exists():
                        st = json.loads(f.read_text())
                        tk = st["tokens"]
                        print(
                            f"    {k}: {st['state']}, {st['phase']} query {st['query']}/{st['T']}, "
                            f"train {st['training_cases']}, {st['tool_calls']} tool calls, "
                            f"tokens in {tk['input'] + tk['cache_read'] + tk['cache_write']:,} "
                            f"(cache read {tk['cache_read']:,}), out {tk['output']:,}, "
                            f"${st['cost_usd_so_far']:.2f} (reported at end of turns), "
                            f"{st['compactions']} compactions, updated {st['updated']}"
                            + (f", ERROR {st['error']}" if st["error"] else "")
                        )


if __name__ == "__main__":
    main(sys.argv[1:] or ["llm_haiku_sessions"])
