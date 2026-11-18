"""Runner for the boolean task with live twin sessions (`consensus.boolean_sessions`).

Usage:
    python scripts/run_boolean.py configs/boolean_haiku.yaml [--set first_run=0 T=5 ...]

Per run (pair) r the study cases, solo queries and partner queries come from their own streams
(seed, r, 0 / 1 / 2), so changing n_train, K or T leaves the other phases' cases unchanged
(the follow-up varies them). Writes results/<experiment>/<timestamp>/:
  env.json                               thresholds, Sigma and the Bayes references
  transcripts/run<r>.jsonl               one record per resolved partner query
  transcripts/run<r>_solo.jsonl          one record per solo prediction (both twins)
  transcripts/run<r>_calls_<twin>.jsonl  every tool call and its result
  transcripts/run<r>_messages_<twin>.jsonl  the twin's text, thinking and tool uses
and summary.json at the end of each run.
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

from consensus.boolean_data import BooleanCopulaEnv, log_loss  # noqa: E402
from consensus.boolean_sessions import BooleanAPI, BooleanTwinSession  # noqa: E402
from consensus.sessions import run_pair  # noqa: E402
from consensus.utils import git_hash, load_config, make_run_dir  # noqa: E402

log = logging.getLogger("run_boolean")


def query_info(env: BooleanCopulaEnv, X: np.ndarray) -> dict:
    """Values stored with each query record (never shown to the twins): the full features, pi,
    the clause values and the Bayes probabilities P(y = 1 | own view) and P(y = 1 | x)."""
    return {
        "x_full": X,
        "pi": env.pi(X),
        "clauses": env.clause_values(X),
        "p_bayes_solo": {k: env.p_solo(env.view(X, k), k) for k in "AB"},
        "p_bayes_joint": env.p_joint(X),
    }


def summarize(api: BooleanAPI) -> dict:
    """Accuracy and mean log loss of the solo predictions, openings and finals per twin."""
    out = {}
    for k in "AB":
        s = api.solo_pred[k]
        ok = ~np.isnan(s[:, 0])
        out[f"solo_{k}"] = _scores(s[ok], api.solo_y[ok])
        res = api.resolved
        for name, arr in (("opening", api.opening[k]), ("final", api.final[k])):
            ok = res & ~np.isnan(arr[:, 0])
            out[f"{name}_{k}"] = _scores(arr[ok], api.y[ok])
    return out


def _scores(pred: np.ndarray, y: np.ndarray) -> dict:
    if len(y) == 0:
        return {"n": 0}
    return {"n": int(len(y)), "acc": float(np.mean(pred[:, 1] == y)),
            "logloss": float(np.mean(log_loss(pred[:, 0], y)))}


def main(argv: list[str] | None = None) -> Path:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config")
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--results", default=str(ROOT / "results"))
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    cfg = load_config(args.config, args.set)
    seed, ag = cfg["seed"], cfg["agent"]
    env = BooleanCopulaEnv.from_config(cfg["env"])
    run_dir = make_run_dir(args.results, cfg["experiment"])
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump({**cfg, "_git_commit": git_hash(ROOT)}, sort_keys=False))
    (run_dir / "env.json").write_text(json.dumps(
        {"t": env.t.tolist(), "Sigma": env.Sigma.tolist(), "references": env.references()},
        indent=2))
    log.info("results in %s", run_dir)
    summary = {}
    first = cfg.get("first_run", 0)
    for r in range(first, first + cfg["n_runs"]):
        Xtr, _, ytr = env.sample(cfg["n_train"], np.random.default_rng([seed, r, 0]))
        Xs, _, ys = env.sample(cfg["K"], np.random.default_rng([seed, r, 1]))
        Xq, _, yq = env.sample(cfg["T"], np.random.default_rng([seed, r, 2]))
        api = BooleanAPI(
            train={k: (env.view(Xtr, k), ytr) for k in "AB"},
            solo={"x": {k: env.view(Xs, k) for k in "AB"}, "y": ys},
            queries={k: env.view(Xq, k) for k in "AB"},
            y=yq,
            labels=ag["labels"],
            page_size=ag["page_size"],
            cap=ag["cap"],
            max_chars=ag["max_chars"],
            info={"solo": query_info(env, Xs), "partner": query_info(env, Xq)},
        )
        tr = run_dir / "transcripts"
        tr.mkdir(parents=True, exist_ok=True)
        lock = threading.Lock()
        for name in (f"run{r}.jsonl", f"run{r}_solo.jsonl"):
            (tr / name).write_text("")

        def write(name: str, rec: dict) -> None:
            with lock, (tr / name).open("a") as f:
                f.write(json.dumps(rec) + "\n")

        def on_query(rec, r=r):
            write(f"run{r}.jsonl", rec)
            log.info("run %d query %d: finals %s, y %d, %d messages", r, rec["number"],
                     rec["final"], rec["y"], len(rec["messages"]))

        api.on_query = on_query
        api.on_solo = lambda k, rec, r=r: write(f"run{r}_solo.jsonl", rec)
        api.on_call = lambda k, rec, r=r: write(f"run{r}_calls_{k}.jsonl", rec)
        twins = {k: BooleanTwinSession(k, api, model=ag["model"], effort=ag["effort"],
                                       max_budget_usd=ag["max_budget_usd"],
                                       max_continues=ag["max_continues"], live_dir=tr, run=r)
                 for k in "AB"}
        run_pair(api, twins)
        summary[f"run{r}"] = {
            "cost_usd": sum(tw.cost_usd for tw in twins.values()),
            "tokens": {k: tw.tokens for k, tw in twins.items()},
            "continues": {k: tw.n_continues for k, tw in twins.items()},
            "errors": {k: tw.error for k, tw in twins.items()},
            "compactions": {k: len(tw.compactions) for k, tw in twins.items()},
            "training_cases": dict(api.n_seen),
            "queries_resolved": int(api.resolved.sum()),
            "messages": len(api.messages),
            "datasets": {k: {n: len(d["rows"]) for n, d in api.datasets[k].items()}
                         for k in "AB"},
            "scores": summarize(api),
        }
        log.info("run %d: $%.2f, %d/%d partner queries, %d messages", r,
                 summary[f"run{r}"]["cost_usd"], api.resolved.sum(), api.T, len(api.messages))
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return run_dir


if __name__ == "__main__":
    main()
