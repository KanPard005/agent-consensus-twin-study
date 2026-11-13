"""Analysis of the agentic-twin pilot (configs/llm_haiku_agentic.yaml, LogBook 2026-11-13).

Same partner-use and entrenchment measures as `compact_analysis.py` (β, movement toward the best
response own + partner - 0.8 x3, by windows of 10 queries), plus what the agents chose to do:
messages sent per query, whether a twin read messages before its final, tool calls, training
cases pulled and actions per query.

Usage:
    python scripts/agentic_analysis.py [--prefix llm_haiku_agentic]
Writes results/agentic_analysis/<timestamp>/{summary.json, table.md}.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from compact_analysis import WINDOWS, movement, window_stats  # noqa: E402

from consensus.utils import make_run_dir  # noqa: E402


def load(prefix: str) -> tuple[list[dict], dict, dict]:
    """Per-twin-query rows, per-run behaviour summaries and the config (latest folder wins)."""
    files: dict[int, Path] = {}
    cfg = None
    for d in sorted((ROOT / "results" / prefix).glob("20*")):
        cfg = yaml.safe_load((d / "config.yaml").read_text())
        for f in (d / "transcripts").glob("run*.jsonl"):
            if re.fullmatch(r"run\d+\.jsonl", f.name):
                files[int(f.stem[3:])] = f
    env = cfg["env"]
    d_A, k = env["d_A"], env["overlap"]
    w_sh = np.asarray(env["w"][d_A - k : d_A], float)
    rows, behaviour = [], {}
    for run, f in sorted(files.items()):
        recs = [json.loads(line) for line in f.read_text().splitlines()]
        acts = {
            tw: [json.loads(line) for line in (f.parent / f"run{run}_actions_{tw}.jsonl")
                 .read_text().splitlines()]
            for tw in "AB"
        }
        comps = {
            tw: (f.parent / f"run{run}_compactions_{tw}.jsonl").read_text().count("\n")
            for tw in "AB"
        }
        for s in recs:
            xs = np.asarray(s["x"]["A"][d_A - k : d_A], float)
            for me, other in (("A", "B"), ("B", "A")):
                own, partner = s["opening"][me], s["opening"][other]
                q_acts = [a for a in acts[me] if a["q"] == s["t"]]
                names = [a["action"] for a in q_acts]
                fin_i = next((i for i, n in enumerate(names) if n.startswith("submit_final")), None)
                before = names[:fin_i] if fin_i is not None else names
                rows.append(
                    dict(
                        run=run,
                        twin=me,
                        t=s["t"],
                        y=s["y"],
                        own=own,
                        partner=partner,
                        final=s["final"][me],
                        team=s["team"],
                        x_shared=xs.tolist(),
                        best=own + partner - float(xs @ w_sh),
                        reg=None,
                        n_rounds=sum(m["from"] == me for m in s["messages"]),
                        ctx=0,
                        ok=bool(np.isfinite([own, partner, s["final"][me]]).all()),
                        forced=s["forced"][me],
                        sent=sum(m["from"] == me for m in s["messages"]),
                        read_before_final="read_messages" in before,
                        regress_calls=names.count("regress"),
                        actions=len(q_acts),
                    )
                )
        behaviour[run] = {
            "training_cases": {  # the last case index pulled
                tw: max(
                    [
                        int(m.group(1))
                        for a in acts[tw]
                        if (m := re.search(r"Training cases \d+-(\d+)", a["result"]))
                    ]
                    or [0]
                )
                for tw in "AB"
            },
            "train_actions": {tw: sum(a["q"] == -1 for a in acts[tw]) for tw in "AB"},
            "train_regress": {
                tw: sum(a["q"] == -1 and a["action"] == "regress" for a in acts[tw]) for tw in "AB"
            },
            "compactions": comps,
        }
    return rows, behaviour, cfg


def main(argv: list[str] | None = None) -> Path:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prefix", default="llm_haiku_agentic")
    args = p.parse_args(argv)
    rows, behaviour, cfg = load(args.prefix)
    k = cfg["env"]["overlap"]
    good = [r for r in rows if r["ok"]]
    runs = sorted({r["run"] for r in good})
    out = dict(
        n_runs=len(runs),
        overall=window_stats(good, k),
        windows=[window_stats([r for r in good if a <= r["t"] < b], k) for a, b in WINDOWS],
        movement_runs=[movement([r for r in good if r["run"] == u]) for u in runs],
        beta_runs=[window_stats([r for r in good if r["run"] == u], k)["beta"] for u in runs],
        late_minus_early=[
            movement([r for r in good if r["run"] == u and r["t"] >= 20])
            - movement([r for r in good if r["run"] == u and r["t"] < 10])
            for u in runs
        ],
        sent_per_query=float(np.mean([r["sent"] for r in rows])),
        silent_share=float(np.mean([r["sent"] == 0 for r in rows])),
        read_before_final=float(np.mean([r["read_before_final"] for r in rows])),
        regress_per_query=float(np.mean([r["regress_calls"] for r in rows])),
        actions_per_query=float(np.mean([r["actions"] for r in rows])),
        forced_share=float(np.mean([r["forced"] for r in rows])),
        sent_by_window=[
            float(np.mean([r["sent"] for r in rows if a <= r["t"] < b])) for a, b in WINDOWS
        ],
        read_by_window=[
            float(np.mean([r["read_before_final"] for r in rows if a <= r["t"] < b]))
            for a, b in WINDOWS
        ],
        behaviour=behaviour,
    )
    run_out = make_run_dir(str(ROOT / "results"), "agentic_analysis")
    (run_out / "summary.json").write_text(json.dumps(out, indent=2))
    W = out["windows"]
    le = np.array(out["late_minus_early"])
    lines = [
        f"runs {out['n_runs']}; β {out['overall']['beta']:.2f} (per run "
        f"{', '.join(f'{b:.2f}' for b in out['beta_runs'])}); movement per run "
        f"{', '.join(f'{m:.2f}' for m in out['movement_runs'])}",
        f"β by window {' / '.join(f'{w['beta']:.2f}' for w in W)}; movement by window "
        f"{' / '.join(f'{w['movement']:.2f}' for w in W)}; late − early {np.nanmean(le):+.2f} ± "
        f"{np.nanstd(le, ddof=1) / np.sqrt(len(le)):.2f}",
        f"team MSE {out['overall']['team_mse']:.3f}; unchanged {out['overall']['unchanged']:.0%}",
        f"messages sent per twin-query {out['sent_per_query']:.2f} (by window "
        f"{' / '.join(f'{v:.2f}' for v in out['sent_by_window'])}); silent {out['silent_share']:.0%}; "
        f"read before final {out['read_before_final']:.0%} (by window "
        f"{' / '.join(f'{v:.0%}' for v in out['read_by_window'])})",
        f"regress per twin-query {out['regress_per_query']:.2f}; actions per twin-query "
        f"{out['actions_per_query']:.1f}; forced finals {out['forced_share']:.0%}",
        f"behaviour: {json.dumps(behaviour)}",
    ]
    table = "\n".join(lines)
    (run_out / "table.md").write_text(table + "\n")
    print(table)
    print(f"saved to {run_out}")
    return run_out


if __name__ == "__main__":
    main()
