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
        "p_bayes_study": {k: env.p_study(env.study_view(X, k), k) for k in "AB"},
        "p_bayes_joint": env.p_joint(X),
    }


def pair_api(env: BooleanCopulaEnv, cfg: dict, train: tuple, solo: tuple,
             queries: tuple, n_train: dict | None = None,
             no_training: bool = False) -> BooleanAPI:
    """The two-twin API: each twin sees its own view of the same cases. With `n_train`
    ({"A": n_A, "B": n_B}) twin k studies the first n_k of the study cases, so the twin with
    fewer rows has a prefix of the other's (the same cases, in the same order)."""
    ag = cfg["agent"]
    (Xtr, ytr), (Xs, ys), (Xq, yq) = train, solo, queries
    n_train = n_train or {k: len(ytr) for k in "AB"}
    # New features (env.study_views): twin k studies the first |study_views[k]| of its labels;
    # the others appear at the solo queries, or at the partner queries if new_at = "partner".
    new_at = cfg.get("new_at", "solo")
    study_labels = {k: ag["labels"][k][: len(env.study_views[k])] for k in "AB"}
    for k in "AB":
        if env.views[k][: len(env.study_views[k])] != list(env.study_views[k]):
            raise ValueError("study_views[k] must be the first features of views[k]")
    solo_view = env.view if new_at == "solo" else env.study_view
    return BooleanAPI(
        train={k: (env.study_view(Xtr[: n_train[k]], k), ytr[: n_train[k]]) for k in "AB"},
        solo={"x": {k: solo_view(Xs, k) for k in "AB"}, "y": ys},
        queries={k: env.view(Xq, k) for k in "AB"},
        y=yq, labels=ag["labels"], page_size=ag["page_size"], cap=ag["cap"],
        max_chars=ag["max_chars"],
        info={"solo": query_info(env, Xs), "partner": query_info(env, Xq)},
        no_training=no_training, study_labels=study_labels, new_at=new_at,
    )


def single_api(env: BooleanCopulaEnv, cfg: dict, train: tuple, solo: tuple,
               queries: tuple) -> BooleanAPI:
    """Single-agent baseline (`single: true`): agent A sees every feature of the same study
    cases, then answers the K solo cases followed by the T cases the pairs answered together,
    all in one solo phase with a break after the first K (see consensus.boolean_sessions)."""
    ag = cfg["agent"]
    (Xtr, ytr), (Xs, ys), (Xq, yq) = train, solo, queries
    if sorted(env.views["A"]) != list(range(env.n)):
        raise ValueError("single: true needs env.views.A to be every feature")
    X, y = np.concatenate([Xs, Xq]), np.concatenate([ys, yq])
    Xa = env.view(X, "A")
    return BooleanAPI(
        train={k: (env.view(Xtr, "A"), ytr) for k in "AB"},  # B is never run
        solo={"x": {k: Xa for k in "AB"}, "y": y},
        queries={k: np.zeros((0, Xa.shape[1]), int) for k in "AB"}, y=np.zeros(0, int),
        labels={k: ag["labels"]["A"] for k in "AB"}, page_size=ag["page_size"], cap=ag["cap"],
        max_chars=ag["max_chars"], info={"solo": query_info(env, X)}, single=len(ys),
    )


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
    if api.single:  # the single agent's two blocks: the solo cases, then the pairs' cases
        s, K1 = api.solo_pred["A"], api.single
        for name, sl in (("single_first", slice(0, K1)), ("single_rest", slice(K1, api.K))):
            ok = ~np.isnan(s[sl, 0])
            out[name] = _scores(s[sl][ok], api.solo_y[sl][ok])
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
        # Unequal study sets (Kanad, 2026-11-19): n_train_long / n_train_short, with the
        # long-trained twin alternating A, B, A, ... by run so the names cannot bias it.
        if "n_train_long" in cfg:
            long = "A" if r % 2 == 0 else "B"
            n_train = {long: cfg["n_train_long"],
                       ("B" if long == "A" else "A"): cfg["n_train_short"]}
        else:
            long, n_train = None, {k: cfg["n_train"] for k in "AB"}
        Xtr, _, ytr = env.sample(max(n_train.values()), np.random.default_rng([seed, r, 0]))
        Xs, _, ys = env.sample(cfg["K"], np.random.default_rng([seed, r, 1]))
        Xq, _, yq = env.sample(cfg["T"], np.random.default_rng([seed, r, 2]))
        if cfg.get("single"):
            api = single_api(env, cfg, (Xtr, ytr), (Xs, ys), (Xq, yq))
        else:
            api = pair_api(env, cfg, (Xtr, ytr), (Xs, ys), (Xq, yq), n_train,
                           no_training=bool(cfg.get("no_training")))
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
                 for k in ("A" if cfg.get("single") else "AB")}
        run_pair(api, twins)
        summary[f"run{r}"] = {
            "cost_usd": sum(tw.cost_usd for tw in twins.values()),
            "tokens": {k: tw.tokens for k, tw in twins.items()},
            "continues": {k: tw.n_continues for k, tw in twins.items()},
            "errors": {k: tw.error for k, tw in twins.items()},
            "compactions": {k: len(tw.compactions) for k, tw in twins.items()},
            "training_cases": dict(api.n_seen),
            "long_twin": long,
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
