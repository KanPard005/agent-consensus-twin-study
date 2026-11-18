"""Run the café hidden-profile task with twins as live agent sessions (`consensus.hp_sessions`).

    python scripts/run_hp_sessions.py configs/hp_haiku_sessions.yaml [--set n_runs=1 \
        cases.n_hidden=2 cases.n_control=1 experiment=pilot ...]

Uses the same seed and case generator as `scripts/run_hp.py`, so with the same `seed`, `cases`
and run index the twins see the same cases as the fixed-round protocols (H1) and the
full-information reference (F1) run by that script. Runs (replicates) go in parallel threads
(`jobs`). Writes results/<experiment>/<timestamp>/ while the runs go:

  transcripts/run<r>.jsonl                  one record per resolved case (consensus.hp_sessions)
  transcripts/run<r>_calls_<twin>.jsonl     every tool call and its result
  transcripts/run<r>_messages_<twin>.jsonl  the twin's text, thinking and tool uses
  transcripts/run<r>_status_<twin>.json     case, tool calls, tokens, cost
  summary.json                              per run: accuracy, cost, messages; updated per run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consensus.hp_data import HiddenProfileGenerator  # noqa: E402
from consensus.hp_sessions import HPSessionAPI, HPTwinSession  # noqa: E402
from consensus.sessions import run_pair  # noqa: E402
from consensus.utils import git_hash, load_config, make_run_dir, run_rng  # noqa: E402

log = logging.getLogger("run_hp_sessions")
LOG_FLOOR = 0.01  # as in consensus.hp_protocol


def case_metrics(rec: dict) -> dict:
    """Per-case measures from a transcript record (NaN where a twin gave no opening/final).

    correct / logscore: team pick vs the best site, log team probability on it (floored);
    open_correct_<k>: the twin's opening favourite is the best site; agree: final favourites
    match; defer_<k>: change from opening to final in the twin's probability on the partner's
    opening favourite, when the opening favourites differ (as in consensus.hp_protocol).
    """
    best, out = rec["case"]["best"], {}
    team = rec["team"]
    out["correct"] = float(rec["team_pick"] == best) if team else np.nan
    out["logscore"] = float(np.log(max(team[best], LOG_FLOOR))) if team else np.nan
    op, fi = rec["opening"], rec["final"]
    fav = {k: int(np.argmax(op[k])) if op[k] else None for k in "AB"}
    for k, o in (("A", "B"), ("B", "A")):
        out[f"open_correct_{k}"] = float(fav[k] == best) if fav[k] is not None else np.nan
        out[f"defer_{k}"] = (fi[k][fav[o]] - op[k][fav[o]]
                             if fav[k] is not None and fav[o] is not None and fi[k]
                             and fav[k] != fav[o] else np.nan)
    out["agree"] = (float(np.argmax(fi["A"]) == np.argmax(fi["B"]))
                    if fi["A"] and fi["B"] else np.nan)
    out["messages"] = len(rec["messages"])
    out["message_chars"] = sum(len(m["text"]) for m in rec["messages"])
    return out


def run_one(cfg: dict, r: int, gen: HiddenProfileGenerator, tr: Path) -> dict:
    """One replicate: sample the cases, run the pair, return its summary."""
    cc, ag = cfg["cases"], cfg["agent"]
    cases = gen.sample_set(run_rng(cfg["seed"], r), cc["n_hidden"], cc["n_control"])
    api = HPSessionAPI(cases, max_chars=ag["max_chars"], cap=ag["cap"])
    path = tr / f"run{r}.jsonl"
    path.write_text("")

    def on_case(rec):
        with path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        log.info("run %d case %d (%s): team Site %s, best Site %d, %d messages", r,
                 rec["t"] + 1, rec["case"]["kind"],
                 None if rec["team_pick"] is None else rec["team_pick"] + 1,
                 rec["case"]["best"] + 1, len(rec["messages"]))

    def on_call(k, rec):
        with (tr / f"run{r}_calls_{k}.jsonl").open("a") as f:
            f.write(json.dumps(rec) + "\n")

    for k in "AB":
        (tr / f"run{r}_calls_{k}.jsonl").write_text("")
    api.on_case, api.on_call = on_case, on_call
    twins = {
        k: HPTwinSession(k, api, model=ag["model"], effort=ag["effort"],
                         tell_split=ag["tell_split"], tell_twin=ag["tell_twin"],
                         max_budget_usd=ag["max_budget_usd"], max_continues=ag["max_continues"],
                         live_dir=tr, run=r)
        for k in "AB"
    }
    run_pair(api, twins)
    recs = [api.record(q) for q in range(api.T) if api.resolved[q]]
    m = [case_metrics(rc) for rc in recs]
    kinds = [rc["case"]["kind"] for rc in recs]

    def mean(key, kind=None):
        v = [x[key] for x, kd in zip(m, kinds) if kind is None or kd == kind]
        return float(np.nanmean(v)) if v and not np.all(np.isnan(v)) else None

    return {
        "cases_resolved": len(recs),
        "correct": mean("correct"),
        "correct_hidden": mean("correct", "hidden"),
        "correct_control": mean("correct", "control"),
        "logscore_hidden": mean("logscore", "hidden"),
        "open_correct_hidden": float(np.nanmean(
            [x[f"open_correct_{k}"] for x, kd in zip(m, kinds) if kd == "hidden"
             for k in "AB"])) if "hidden" in kinds else None,
        "agree": mean("agree"),
        "messages_per_case": mean("messages"),
        "cost_usd": sum(tw.cost_usd for tw in twins.values()),
        "tokens": {k: tw.tokens for k, tw in twins.items()},
        "continues": {k: tw.n_continues for k, tw in twins.items()},
        "errors": {k: tw.error for k, tw in twins.items()},
        "compactions": {k: len(tw.compactions) for k, tw in twins.items()},
        "results": {k: tw.results for k, tw in twins.items()},
    }


def main(argv: list[str] | None = None) -> Path:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config")
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--results", default=str(ROOT / "results"))
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    cfg = load_config(args.config, args.set)
    cc = {k: v for k, v in cfg["cases"].items() if k not in ("n_hidden", "n_control")}
    gen = HiddenProfileGenerator(**cc)
    run_dir = make_run_dir(args.results, cfg["experiment"])
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump({**cfg, "_git_commit": git_hash(ROOT)}, sort_keys=False))
    tr = run_dir / "transcripts"
    tr.mkdir(parents=True, exist_ok=True)
    log.info("results in %s", run_dir)
    first = cfg.get("first_run", 0)
    runs = list(range(first, first + cfg["n_runs"]))
    summary, lock = {}, threading.Lock()

    def job(r: int) -> None:
        s = run_one(cfg, r, gen, tr)
        with lock:
            summary[f"run{r}"] = s
            (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        log.info("run %d: hidden acc %s, control acc %s, %.1f messages/case, $%.2f", r,
                 s["correct_hidden"], s["correct_control"], s["messages_per_case"] or 0,
                 s["cost_usd"])

    with ThreadPoolExecutor(max_workers=cfg.get("jobs", 3)) as pool:
        list(pool.map(job, runs))
    return run_dir


if __name__ == "__main__":
    main()
