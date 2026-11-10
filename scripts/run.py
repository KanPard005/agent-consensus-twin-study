"""Runner: config -> train twins -> run every protocol on the same post-training stream -> save.

Usage:
    python scripts/run.py configs/default.yaml [--set train.n_A=50 env.rho=0.8] [--plot]

Writes results/<experiment>/<timestamp>/ (config + git commit, metrics.npz, benchmarks.json,
summary.json, and figures with --plot).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consensus.data import LinearTwinEnv, LogisticTwinEnv, ProductTwinEnv, build_env  # noqa: E402
from consensus.models import LinearAgent, LogisticAgent  # noqa: E402
from consensus.protocols import Exchange, build_protocol  # noqa: E402
from consensus.theory import protocol_benchmarks  # noqa: E402
from consensus.theory_logistic import logistic_benchmarks  # noqa: E402
from consensus.theory_product import product_benchmarks  # noqa: E402
from consensus.utils import (  # noqa: E402
    git_hash,
    load_config,
    make_run_dir,
    run_rng,
    save_jsonl,
    save_run,
)

log = logging.getLogger("run")


def train_twins(env: LinearTwinEnv, cfg: dict, rng: np.random.Generator):
    """Training phase: each twin fits its own iid stream (x^(i)_t, y_t).

    Online ridge for the regression task; logistic regression (MAP, ridge prior) for the
    logistic task. `intercept: true` (regression only) adds a constant feature, needed when
    E[y] != 0 (the product task). With n_A = n_B = 0 the twins start untrained.
    """
    if isinstance(env, LogisticTwinEnv):
        if cfg.get("intercept"):
            raise ValueError("train.intercept is only supported for regression tasks")
        A = LogisticAgent(env.d_A, cfg["lam"], "A")
        B = LogisticAgent(env.d_B, cfg["lam"], "B")
    else:
        A = LinearAgent(env.d_A, cfg["lam"], "A", intercept=cfg.get("intercept", False))
        B = LinearAgent(env.d_B, cfg["lam"], "B", intercept=cfg.get("intercept", False))
    x_A, _, y_A = env.sample(cfg["n_A"], rng)
    _, x_B, y_B = env.sample(cfg["n_B"], rng)
    for t in range(cfg["n_A"]):
        A.update(x_A[t], [], y_A[t])
    for t in range(cfg["n_B"]):
        B.update(x_B[t], [], y_B[t])
    return A, B


def _nanmean(a: np.ndarray) -> float:
    """Mean ignoring nan; nan (without a warning) if everything is nan."""
    return float(np.nanmean(a)) if np.any(~np.isnan(a)) else float("nan")


def summarize(results: dict, bench: dict, proto_keys: dict, tail: float = 0.1) -> dict:
    """Averages over the last `tail` fraction of steps and over runs."""
    out = {}
    for label, res in results.items():
        T = res["sq_err"].shape[1]
        sl = slice(int((1 - tail) * T), T)
        key = proto_keys[label]
        frozen = 0.0 if key in ("no_talk", "logit_sum", "product", "blend") else np.nan
        out[label] = {
            "protocol": key,
            "excess_mse_tail": float(res["sq_err"][:, sl].mean() - bench["full_info"]),
            "excess_mse_population": float(bench.get(key, np.nan) - bench["full_info"]),
            "disagreement_tail": float(res["disagreement"][:, sl].mean()),
            "shift_A_tail": float(res["shift_A"][:, sl].mean()),
            "shift_B_tail": float(res["shift_B"][:, sl].mean()),
            # Frozen protocols never shift; protocols without a closed form get nan.
            "shift_A_population": float(bench.get(f"shift_A_{key}", frozen)),
            "shift_B_population": float(bench.get(f"shift_B_{key}", frozen)),
            "drift_A_final": _nanmean(res["drift_A"][:, -1]),
            **(
                {
                    "excess_logloss_tail": float(
                        res["log_loss"][:, sl].mean() - bench["logloss_full_info"]
                    ),
                    "excess_logloss_population": float(
                        bench.get(f"logloss_{key}", np.nan) - bench["logloss_full_info"]
                    ),
                }
                if "log_loss" in res
                else {}
            ),
            "drift_B_final": _nanmean(res["drift_B"][:, -1]),
        }
        # LLM protocols: totals over all runs and steps.
        for m in ("parse_fail", "tokens_in", "tokens_out", "cost_usd", "tool_calls"):
            if m in res:
                out[label][f"{m}_total"] = float(np.nansum(res[m]))
        if "msg_chars" in res:
            out[label]["msg_chars_mean"] = float(np.nanmean(res["msg_chars"]))
    return out


def main(argv: list[str] | None = None) -> Path:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("config")
    p.add_argument("--set", nargs="*", default=[], help="overrides like train.n_A=50")
    p.add_argument("--results", default=str(ROOT / "results"))
    p.add_argument("--plot", action="store_true", help="also write figures")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")

    cfg = load_config(args.config, args.set)
    seed = cfg["seed"]
    if cfg.get("only"):  # run only these protocol labels, e.g. --set only=[P0,E25]
        cfg["protocols"] = [pc for pc in cfg["protocols"] if pc.get("label") in cfg["only"]]
        if not cfg["protocols"]:
            raise ValueError(f"no protocol labelled {cfg['only']}")
    # The environment (true weights) is fixed across runs so population limits are comparable.
    env = build_env(cfg["env"], np.random.default_rng(seed))
    protocols = [build_protocol(pc) for pc in cfg["protocols"]]
    labels = [pc.get("label", proto.name) for pc, proto in zip(cfg["protocols"], protocols)]
    proto_keys = dict(zip(labels, (proto.name for proto in protocols)))
    max_E = max([proto.E for proto in protocols if isinstance(proto, Exchange)] or [1])
    if isinstance(env, LogisticTwinEnv):  # Brier scores (+ logloss_*), by Monte Carlo
        bench = logistic_benchmarks(env, n_exchanges=max_E)
    elif isinstance(env, ProductTwinEnv):  # Monte Carlo; no closed form for the exchanges
        bench = product_benchmarks(env)
    else:
        bench = protocol_benchmarks(env.Sigma, env.w, env.d_A, env.sigma, n_exchanges=max_E)
    log.info("population benchmarks: %s", {k: round(v, 4) for k, v in bench.items()})

    T, n_runs = cfg["post"]["T"], cfg["n_runs"]
    # Replicates first_run .. first_run + n_runs - 1, so extra seeds can be added in later folders
    # (or in parallel processes) without repeating earlier ones; metrics row i is replicate
    # first_run + i, and transcripts are named by the replicate index.
    first_run = cfg.setdefault("first_run", 0)
    results = {lab: {} for lab in labels}
    transcripts = {}  # {(label, run): list of per-step dicts}, LLM protocols only
    # {(label, run, twin): list of turns} for persistent LLM twins, and
    # {(label, run, "training_<twin>"): batch notes} for self-learning twins.
    conversations = {}
    for i in range(n_runs):
        r = first_run + i
        rng = run_rng(seed, r)
        A, B = train_twins(env, cfg["train"], rng)
        x_A, x_B, y = env.sample(T, rng)  # common random numbers across protocols
        for lab, proto in zip(labels, protocols):
            out = proto.run(A.clone(), B.clone(), x_A, x_B, y)
            for m, arr in out.items():
                results[lab].setdefault(m, np.empty((n_runs, T)))[i] = arr
            if hasattr(proto, "transcript"):
                transcripts[(lab, r)] = proto.transcript
            for name, turns in getattr(proto, "conversations", {}).items():
                conversations[(lab, r, name)] = turns
            for name, batches in getattr(proto, "training", {}).items():
                conversations[(lab, r, f"training_{name}")] = batches
        log.info("run %d/%d done (replicate %d)", i + 1, n_runs, r)

    tail = cfg.get("summary_tail", 0.1)
    summary = {
        "proto_keys": proto_keys,
        "summary_tail": tail,
        "protocols": summarize(results, bench, proto_keys, tail),
    }
    run_dir = make_run_dir(args.results, cfg["experiment"])
    save_run(run_dir, cfg, git_hash(ROOT), results, bench, summary)
    for (lab, r), steps in transcripts.items():
        save_jsonl(run_dir / "transcripts" / f"{lab}_run{r}.jsonl", steps)
    for (lab, r, name), turns in conversations.items():
        kind = "" if name.startswith("training_") else "conversation_"
        save_jsonl(run_dir / "transcripts" / f"{lab}_run{r}_{kind}{name}.jsonl", turns)
    log.info("saved to %s", run_dir)
    for lab, s in summary["protocols"].items():
        if "cost_usd_total" in s:
            log.info(
                "%-6s LLM: %d unusable replies, %.0fk tokens in / %.0fk out, $%.3f",
                lab,
                s["parse_fail_total"],
                s["tokens_in_total"] / 1e3,
                s["tokens_out_total"] / 1e3,
                s["cost_usd_total"],
            )
        log.info(
            "%-6s excess MSE tail %.4f (population %.4f), opinion shift A %.3f (%.3f) B %.3f (%.3f)",
            lab + (" Brier" if "excess_logloss_tail" in s else ""),
            s["excess_mse_tail"],
            s["excess_mse_population"],
            s["shift_A_tail"],
            s["shift_A_population"],
            s["shift_B_tail"],
            s["shift_B_population"],
        )
        if "excess_logloss_tail" in s:
            log.info(
                "%-6s excess log loss tail %.4f (population %.4f)",
                lab,
                s["excess_logloss_tail"],
                s["excess_logloss_population"],
            )
    if args.plot:
        from plot import make_figures  # scripts/plot.py

        make_figures(run_dir, cfg.get("plot_window", 200))
    return run_dir


if __name__ == "__main__":
    main()
