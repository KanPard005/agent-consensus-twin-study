"""Figure for the XOR briefing arms (LogBook agent-consensus log, 2026-11-19).

Usage:
    python scripts/plot_xor_briefing.py --out <LogBook>/projects/agent-consensus/figures

Reads results/xor_briefing_summary.json (written by scripts/xor_briefing_analysis.py) and draws
`consensus.plotting_boolean_note.plot_xor_briefing`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consensus.plotting_boolean_note import plot_xor_briefing  # noqa: E402

NAMES = {"none": "no conversation", "brief": "conversation", "announced": "announced conversation",
         "other": "other agent", "other_round": "other agent with round"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--arms", default="none,brief,announced",
                   help="comma-separated arms of results/xor_briefing_summary.json (at most 3)")
    p.add_argument("--name", default="2026-11-19-xor-briefing.png")
    args = p.parse_args()
    data = json.loads((ROOT / "results" / "xor_briefing_summary.json").read_text())
    arms = [{"name": NAMES[k], "acc": [x["acc"] for x in rows],
             "last_msg": [x["last_msg_query"] for x in rows], "T": rows[0]["n"]}
            for k in args.arms.split(",") for rows in [data[k]]]
    fig = plot_xor_briefing(arms)
    out = Path(args.out) / args.name
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="#fcfcfb")
    print(out)


if __name__ == "__main__":
    main()
