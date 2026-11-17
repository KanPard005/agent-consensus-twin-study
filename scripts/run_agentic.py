"""Runner for agentic twins (`consensus.agentic`): each twin drives the shared data API.

Usage:
    python scripts/run_agentic.py configs/llm_haiku_agentic.yaml [--set n_runs=1 T=5 ...]

Uses the same seed, training samples and query stream as `scripts/run.py` (so runs are paired
with the O1h and compaction runs). Everything is written while the run goes: each finished query
is appended to transcripts/run<r>.jsonl, each action / turn / compaction to its file, and each
twin's status (query, calls, spend, context) to transcripts/run<r>_status_<twin>.json; see
`scripts/agentic_status.py`. Writes results/<experiment>/<timestamp>/: config.yaml,
summary.json, and per replicate transcripts/run<r>.jsonl (one record per query),
run<r>_actions_<twin>.jsonl, run<r>_turns_<twin>.jsonl and run<r>_compactions_<twin>.jsonl.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run import train_twins  # noqa: E402

from consensus.agentic import AgentTwin, DataGeneratorAPI, run_episode  # noqa: E402
from consensus.data import build_env  # noqa: E402
from consensus.llm_backends import build_backend  # noqa: E402
from consensus.utils import git_hash, load_config, make_run_dir, run_rng, save_jsonl  # noqa: E402

log = logging.getLogger("run_agentic")


def query_records(api: DataGeneratorAPI) -> list[dict]:
    """One record per query: features, y, openings, finals, team, messages."""
    out = []
    for q in range(api.T):
        fin = {k: float(api.final[k][q]) for k in "AB"}
        out.append(
            {
                "t": q,
                "y": float(api.y[q]),
                "x": {k: [float(v) for v in api.qx[k][q]] for k in "AB"},
                "opening": {k: float(api.opening[k][q]) for k in "AB"},
                "final": fin,
                "forced": {k: bool(api.forced[k][q]) for k in "AB"},
                "team": (fin["A"] + fin["B"]) / 2,
                "messages": [m for m in api.messages if m["q"] == q],
            }
        )
    return out


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
    meta = {**cfg, "_git_commit": git_hash(ROOT)}
    (run_dir / "config.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
    labels = ag["labels"]
    summary = {}
    first = cfg.get("first_run", 0)
    for r in range(first, first + cfg["n_runs"]):
        rng = run_rng(seed, r)
        A, B = train_twins(env, cfg["train"], rng)  # same draws as scripts/run.py
        x_A, x_B, y = env.sample(T, rng)
        api = DataGeneratorAPI(
            train={"A": A.training_data(), "B": B.training_data()},
            queries={"A": x_A, "B": x_B},
            y=y,
            labels=labels,
            budget=ag["budget"],
            max_chars=ag["max_chars"],
        )
        tr = run_dir / "transcripts"
        tr.mkdir(parents=True, exist_ok=True)
        live = tr / f"run{r}.jsonl"
        live.write_text("")

        def on_query(rec, live=live):
            with live.open("a") as f:
                f.write(json.dumps(rec) + "\n")

        api.on_query = on_query
        backend = build_backend(ag["backend"])
        twins = {
            k: AgentTwin(
                k,
                api,
                backend,
                labels[k],
                labels["B" if k == "A" else "A"],
                T,
                ag["budget"],
                max_actions=ag["max_actions"],
                max_train_actions=ag["max_train_actions"],
                max_chars=ag["max_chars"],
                compact_tokens=ag["compact_tokens"],
                summary_chars=ag["summary_chars"],
                structure_hint=ag["structure_hint"],
                partner_info=ag["partner_info"],
                tell_twin=ag["tell_twin"],
                max_cost_usd=ag.get("max_cost_usd"),
                live_dir=tr,
                run=r,
            )
            for k in "AB"
        }
        run_episode(api, twins)
        recs = query_records(api)
        save_jsonl(tr / f"run{r}.jsonl", recs)
        for k, tw in twins.items():
            save_jsonl(tr / f"run{r}_actions_{k}.jsonl", tw.log)
            save_jsonl(tr / f"run{r}_turns_{k}.jsonl", tw.turns)
            save_jsonl(tr / f"run{r}_compactions_{k}.jsonl", tw.compactions)
        reps = [rp for tw in twins.values() for rp in tw.replies]
        team = np.array([rc["team"] for rc in recs])
        summary[f"run{r}"] = {
            "cost_usd": float(np.nansum([rp.cost_usd for rp in reps])),
            "tokens_in": int(sum(rp.tokens_in for rp in reps)),
            "calls": len(reps),
            "unusable": {k: tw.unusable for k, tw in twins.items()},
            "compactions": {k: len(tw.compactions) for k, tw in twins.items()},
            "training_cases": {k: api.s[k].n_train_seen for k in "AB"},
            "queries_completed": int(np.sum(~np.isnan(team))),
            "forced_finals": {k: int(api.forced[k].sum()) for k in "AB"},
            "messages": len(api.messages),
            "team_mse": float(np.nanmean((team - y) ** 2)),
        }
        log.info("run %d: %s", r, json.dumps(summary[f"run{r}"]))
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("saved to %s", run_dir)
    return run_dir


if __name__ == "__main__":
    main()
