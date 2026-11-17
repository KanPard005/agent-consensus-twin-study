"""Compaction and entrenchment in free talk (configs/llm_haiku_compact.yaml, LogBook 2026-11-13).

Per arm (compaction threshold), over all runs:

- combination weights: final = alpha own_open + beta partner_open + gamma x3 + c (OLS, pooled
  over twins and runs), overall and in windows of steps; the best response is
  alpha = beta = 1, gamma = -0.8 (`overlap_analysis`);
- entrenchment over the run, per window: beta, the share of finals equal to the twin's opening,
  the movement toward the best response, (final - own) / (best - own) with
  best = own + partner - 0.8 x3 (median; 0 = kept its opinion, 1 = best response), and how often
  the final is closer to the twin's opening than to the automatic regression's prediction;
- compaction: the same movement and |final - best| in the 3 steps before and after each
  compaction of a twin, and whether its summary keeps the combination rule;
- team MSE, rounds per step and context size.

Usage:
    python scripts/compact_analysis.py [--arms C4k C8k C16k Cinf] [--prefix llm_haiku_compact]
Writes results/compact_analysis/<timestamp>/{summary.json, table.md, compact.png}.
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

from overlap_analysis import fit_combination  # noqa: E402

from consensus.plotting_overlap import plot_compaction  # noqa: E402
from consensus.utils import make_run_dir  # noqa: E402

WINDOWS = ((0, 10), (10, 20), (20, 30))
PRED_RE = re.compile(r"Prediction for the current case: (-?\d+(?:\.\d+)?)")
# A summary "keeps the rule" if it says how to combine with the partner and handles x3 once.
RULE_RE = re.compile(r"(once|double|non-x3|without x3|minus.*x3|x3.*(only|counted) once)", re.I)


def load(prefix: str, arm: str) -> tuple[list[dict], list[dict], dict]:
    """Per-twin-step rows, compaction events and the config, over every results folder of the
    arm (replicates may be split across folders; the latest folder wins for a replicate)."""
    runs: dict[int, Path] = {}
    cfg = None
    for d in sorted((ROOT / "results" / f"{prefix}_{arm}").glob("20*")):
        cfg = yaml.safe_load((d / "config.yaml").read_text())
        for f in (d / "transcripts").glob(f"{arm}_run*.jsonl"):
            if re.fullmatch(rf"{arm}_run\d+\.jsonl", f.name):
                runs[int(f.stem.split("_run")[1])] = f
    if cfg is None:
        raise FileNotFoundError(f"no results for {prefix}_{arm}")
    env = cfg["env"]
    d_A, k = env["d_A"], env["overlap"]
    w_sh = np.asarray(env["w"][d_A - k : d_A], float)
    rows, events = [], []
    for run, f in sorted(runs.items()):
        for line in f.read_text().splitlines():
            s = json.loads(line)
            xs = np.asarray(s["x"]["A"][d_A - k : d_A], float)
            for me, other in (("A", "B"), ("B", "A")):
                r0, rl = s["rounds"][0], s["rounds"][-1]
                reg = None
                if len(s["rounds"]) > 1:  # the round-2 automatic run (own_open, partner_open)
                    m = PRED_RE.search(s["rounds"][1][me]["tool_calls"][0]["result"])
                    reg = float(m.group(1)) if m else None
                own, partner = r0[me]["estimate"], r0[other]["estimate"]
                rows.append(
                    dict(
                        run=run,
                        twin=me,
                        t=s["t"],
                        y=s["y"],
                        own=own,
                        partner=partner,
                        final=rl[me]["estimate"],
                        team=s["consensus"],
                        x_shared=xs.tolist(),
                        best=own + partner - float(xs @ w_sh),
                        reg=reg,
                        n_rounds=s["n_rounds"],
                        ctx=s["context_tokens"][me],
                        ok=all(rd[me]["ok"] for rd in s["rounds"]),
                        tools=sum(
                            1 for rd in s["rounds"] for c in rd[me]["tool_calls"] if not c["auto"]
                        ),
                    )
                )
                if me in s["compacted"]:
                    summ = s["compacted"][me]["summary"]
                    events.append(
                        dict(
                            run=run,
                            twin=me,
                            t=s["t"],
                            tokens=s["compacted"][me]["tokens"],
                            chars=len(summ),
                            keeps_rule=bool(RULE_RE.search(summ)),
                        )
                    )
    return rows, events, cfg


def movement(rows: list[dict]) -> float:
    """Median of (final - own) / (best - own) over rows where best and own differ by > 0.2."""
    v = [
        (r["final"] - r["own"]) / (r["best"] - r["own"])
        for r in rows
        if abs(r["best"] - r["own"]) > 0.2
    ]
    return float(np.median(v)) if v else float("nan")


def window_stats(rows: list[dict], k: int) -> dict:
    """Entrenchment measures over a set of rows."""
    keys = ("alpha", "beta", "gamma", "movement", "unchanged", "closer_to_own_than_reg",
            "abs_from_best", "rounds", "team_mse")
    if len(rows) < 8:  # too few to fit (e.g. a window beyond a short run)
        return {"n": len(rows), **{k_: float("nan") for k_ in keys}}
    coef = fit_combination(rows, k)
    with_reg = [r for r in rows if r["reg"] is not None]
    return dict(
        n=len(rows),
        alpha=float(coef[0]),
        beta=float(coef[1]),
        gamma=float(coef[2]) if k else None,
        movement=movement(rows),
        unchanged=float(np.mean([abs(r["final"] - r["own"]) < 1e-6 for r in rows])),
        closer_to_own_than_reg=float(
            np.mean([abs(r["final"] - r["own"]) < abs(r["final"] - r["reg"]) for r in with_reg])
        )
        if with_reg
        else float("nan"),
        abs_from_best=float(np.mean([abs(r["final"] - r["best"]) for r in rows])),
        rounds=float(np.mean([r["n_rounds"] for r in rows])),
        team_mse=float(np.mean([(r["team"] - r["y"]) ** 2 for r in rows if r["twin"] == "A"])),
    )


def event_study(rows: list[dict], events: list[dict], h: int = 3) -> dict:
    """Movement and |final - best| in the h steps before and after each compaction
    (the compaction happens after step t: before = t-h+1..t, after = t+1..t+h)."""
    by = {(r["run"], r["twin"], r["t"]): r for r in rows}
    before, after = [], []
    for e in events:
        before += [by[(e["run"], e["twin"], t)] for t in range(e["t"] - h + 1, e["t"] + 1)
                   if (e["run"], e["twin"], t) in by]
        after += [by[(e["run"], e["twin"], t)] for t in range(e["t"] + 1, e["t"] + h + 1)
                  if (e["run"], e["twin"], t) in by]
    f = lambda rs: float(np.mean([abs(r["final"] - r["best"]) for r in rs])) if rs else None  # noqa: E731
    return dict(
        n_events=len(events),
        movement_before=movement(before) if before else None,
        movement_after=movement(after) if after else None,
        abs_from_best_before=f(before),
        abs_from_best_after=f(after),
        keeps_rule=float(np.mean([e["keeps_rule"] for e in events])) if events else None,
        summary_chars=float(np.mean([e["chars"] for e in events])) if events else None,
        tokens_at_compaction=float(np.mean([e["tokens"] for e in events])) if events else None,
    )


def main(argv: list[str] | None = None) -> Path:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--arms", nargs="+", default=["C4k", "C8k", "C16k", "Cinf"])
    p.add_argument("--prefix", default="llm_haiku_compact")
    args = p.parse_args(argv)
    out = {}
    for arm in args.arms:
        rows, events, cfg = load(args.prefix, arm)
        k = cfg["env"]["overlap"]
        good = [r for r in rows if r["ok"]]
        runs = sorted({r["run"] for r in good})
        per_run = [window_stats([r for r in good if r["run"] == u], k) for u in runs]
        out[arm] = dict(
            n_runs=len(runs),
            unusable=len(rows) - len(good),
            overall=window_stats(good, k),
            beta_runs=[s["beta"] for s in per_run],
            movement_runs=[s["movement"] for s in per_run],
            windows=[window_stats([r for r in good if a <= r["t"] < b], k) for a, b in WINDOWS],
            # Paired within run: late minus early movement and beta, per run.
            late_minus_early_movement=[
                movement([r for r in good if r["run"] == u and r["t"] >= 20])
                - movement([r for r in good if r["run"] == u and r["t"] < 10])
                for u in runs
            ],
            compaction=event_study(good, events),
            ctx_mean=float(np.mean([r["ctx"] for r in rows])),
            ctx_max=float(np.max([r["ctx"] for r in rows])),
            tool_calls_per_step=float(np.mean([r["tools"] for r in rows])),
            compactions_per_twin_run=len(events) / (2 * len(runs)),
        )
    run_out = make_run_dir(str(ROOT / "results"), "compact_analysis")
    (run_out / "summary.json").write_text(json.dumps(out, indent=2))

    lines = [
        "| arm | runs | compactions per twin | context mean / max | β | β by steps 1–10 / 11–20 "
        "/ 21–30 | movement by window | unchanged by window | closer to own than regression "
        "| late − early movement (mean ± s.e.) | team MSE | rounds per step |",
        "|" + "---|" * 12,
    ]
    for arm, s in out.items():
        W = s["windows"]
        le = np.array(s["late_minus_early_movement"])
        lines.append(
            f"| {arm} | {s['n_runs']} | {s['compactions_per_twin_run']:.1f} "
            f"| {s['ctx_mean']:.0f} / {s['ctx_max']:.0f} | {s['overall']['beta']:.2f} "
            f"| {' / '.join(f'{w['beta']:.2f}' for w in W)} "
            f"| {' / '.join(f'{w['movement']:.2f}' for w in W)} "
            f"| {' / '.join(f'{w['unchanged']:.0%}' for w in W)} "
            f"| {s['overall']['closer_to_own_than_reg']:.0%} "
            f"| {np.nanmean(le):+.2f} ± {np.nanstd(le, ddof=1) / np.sqrt(len(le)):.2f} "
            f"| {s['overall']['team_mse']:.3f} | {s['overall']['rounds']:.2f} |"
        )
    lines += ["", "| arm | compactions | movement 3 before / after | |final − best| before / after "
              "| summaries keeping the x3 rule | summary chars | tokens at compaction |",
              "|" + "---|" * 7]
    for arm, s in out.items():
        c = s["compaction"]
        if not c["n_events"]:
            lines.append(f"| {arm} | 0 | — | — | — | — | — |")
            continue
        lines.append(
            f"| {arm} | {c['n_events']} | {c['movement_before']:.2f} / {c['movement_after']:.2f} "
            f"| {c['abs_from_best_before']:.2f} / {c['abs_from_best_after']:.2f} "
            f"| {c['keeps_rule']:.0%} | {c['summary_chars']:.0f} | {c['tokens_at_compaction']:.0f} |"
        )
    table = "\n".join(lines)
    (run_out / "table.md").write_text(table + "\n")
    plot_compaction(out, [f"{a + 1}–{b}" for a, b in WINDOWS], run_out / "compact.png")
    print(table)
    print(f"saved to {run_out}")
    return run_out


if __name__ == "__main__":
    main()
