"""Runner for twins as live agent sessions with a consensus rule (`consensus.sessions`).

Usage:
    python scripts/run_sessions.py configs/llm_haiku_sessions.yaml [--set n_runs=1 T=5 ...]

Same seed, training samples and query stream as `scripts/run.py` (paired with the earlier runs).
Writes results/<experiment>/<timestamp>/ while the run goes:
  transcripts/run<r>.jsonl              one record per resolved query
  transcripts/run<r>_calls_<twin>.jsonl every tool call and its result
  transcripts/run<r>_messages_<twin>.jsonl  the twin's text, thinking, tool uses and results
  transcripts/run<r>_status_<twin>.json phase, query, tool calls, tokens, cost, compactions
and summary.json at the end of each run. `scripts/sessions_status.py` summarizes the live files.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run import train_twins  # noqa: E402

from consensus.data import build_env  # noqa: E402
from consensus.sessions import ConsensusAPI, TwinSession, run_pair  # noqa: E402
from consensus.utils import git_hash, load_config, make_run_dir, run_rng  # noqa: E402

log = logging.getLogger("run_sessions")


def main(argv: list[str] | None = None) -> Path:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("config")
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--results", default=str(ROOT / "results"))
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    cfg = load_config(args.config, args.set)
    seed, T, ag = cfg["seed"], cfg["T"], cfg["agent"]
    env = build_env(cfg["env"], np.random.default_rng(seed))
    run_dir = make_run_dir(args.results, cfg["experiment"])
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump({**cfg, "_git_commit": git_hash(ROOT)}, sort_keys=False)
    )
    log.info("results in %s", run_dir)
    summary = {}
    first = cfg.get("first_run", 0)
    for r in range(first, first + cfg["n_runs"]):
        rng = run_rng(seed, r)
        A, B = train_twins(env, cfg["train"], rng)  # same draws as scripts/run.py
        x_A, x_B, y = env.sample(T, rng)
        api = ConsensusAPI(
            {"A": A.training_data(), "B": B.training_data()},
            {"A": x_A, "B": x_B},
            y,
            ag["labels"],
            budget=ag["budget"],
            tol=ag["tol"],
            max_attempts=ag["max_attempts"],
            penalty=ag["penalty"],
            max_chars=ag["max_chars"],
            cap=ag["cap"],
        )
        tr = run_dir / "transcripts"
        tr.mkdir(parents=True, exist_ok=True)
        lock = threading.Lock()
        (tr / f"run{r}.jsonl").write_text("")

        def on_query(rec, r=r):
            with lock, (tr / f"run{r}.jsonl").open("a") as f:
                f.write(json.dumps(rec) + "\n")
            log.info("run %d query %d: agreed %.3f, y %.3f, failed %s, %d messages", r,
                     rec["t"] + 1, rec["agreed"], rec["y"], rec["failed"], len(rec["messages"]))

        def on_call(k, rec, r=r):
            with (tr / f"run{r}_calls_{k}.jsonl").open("a") as f:
                f.write(json.dumps(rec) + "\n")

        for k in "AB":
            (tr / f"run{r}_calls_{k}.jsonl").write_text("")
        api.on_query, api.on_call = on_query, on_call
        twins = {
            k: TwinSession(
                k,
                api,
                model=ag["model"],
                effort=ag["effort"],
                structure_hint=ag["structure_hint"],
                partner_info=ag["partner_info"],
                max_budget_usd=ag["max_budget_usd"],
                max_continues=ag["max_continues"],
                live_dir=tr,
                run=r,
            )
            for k in "AB"
        }
        run_pair(api, twins)
        recs = [api.record(q) for q in range(T)]
        done = [rc for rc in recs if rc["agreed"] == rc["agreed"] or rc["failed"]]
        sq = [ag["penalty"] if rc["failed"] else (rc["agreed"] - rc["y"]) ** 2 for rc in done]
        summary[f"run{r}"] = {
            "cost_usd": sum(tw.cost_usd for tw in twins.values()),  # cumulative per session
            "tokens": {k: tw.tokens for k, tw in twins.items()},
            "results": {k: tw.results for k, tw in twins.items()},
            "continues": {k: tw.n_continues for k, tw in twins.items()},
            "errors": {k: tw.error for k, tw in twins.items()},
            "compactions": {k: len(tw.compactions) for k, tw in twins.items()},
            "training_cases": dict(api.n_seen),
            "queries_resolved": int(api.resolved.sum()),
            "failed": int(api.failed.sum()),
            "messages": len(api.messages),
            "team_score": float(np.mean(sq)) if sq else None,
        }
        log.info("run %d: $%.2f, %d/%d queries, %d failed, %d messages", r,
                 summary[f"run{r}"]["cost_usd"], api.resolved.sum(), T, api.failed.sum(),
                 len(api.messages))
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return run_dir


if __name__ == "__main__":
    main()
