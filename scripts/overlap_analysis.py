"""How do LLM twins combine their estimates when features are independent and partly shared?

Reads the transcripts of the overlap arms (configs/llm_haiku_overlap.yaml, llm_haiku_overlap0.yaml)
and, per arm, fits each twin's round-2 (final) estimate on what it knows at that point:

    final = alpha * own_open + beta * partner_open [+ gamma * x_shared] + c.

With independent features, E[y | both] = yhat_A + yhat_B - <x_shared, w_shared>, so the best response
is alpha = beta = 1 and gamma = -w_shared (gamma is absent without overlap); averaging is
alpha = beta = 1/2, gamma = 0. Each final is also labelled by the nearest of four rules (keep own,
average, sum, sum minus the shared term), and the team's error is compared with the same rules
applied to the twins' own openings.

Usage:
    python scripts/overlap_analysis.py --arms O1h O1n O0h O0n [--prefix llm_haiku_overlap]
Writes results/overlap_analysis/<timestamp>/{summary.json, table.md, overlap.png}.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consensus.plotting_overlap import plot_overlap  # noqa: E402
from consensus.utils import make_run_dir  # noqa: E402

RULES = ("own", "average", "sum", "sum - shared")


def latest_run(prefix: str, arm: str) -> Path:
    """Most recent results folder of one arm (results/<prefix>_<arm>/<timestamp>)."""
    runs = sorted((ROOT / "results" / f"{prefix}_{arm}").glob("20*"))
    if not runs:
        raise FileNotFoundError(f"no results for {prefix}_{arm}")
    return runs[-1]


def load_steps(run_dir: Path, arm: str) -> tuple[list[dict], dict]:
    """Per-twin rows (one per twin and step) and the config of one arm's run."""
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    env = cfg["env"]
    d_A, k = env["d_A"], env.get("overlap", 0)
    w = np.asarray(env["w"], float)
    w_shared = w[d_A - k : d_A]  # weights of the shared features z_{d_A-k+1..d_A}
    rows = []
    for f in sorted((run_dir / "transcripts").glob(f"{arm}_run*.jsonl")):
        if "training" in f.name or "conversation" in f.name:
            continue
        run = int(f.stem.split("_run")[1])
        for line in f.read_text().splitlines():
            s = json.loads(line)
            # x_shared: the last k of A's features (= the first k of B's).
            xs = np.asarray(s["x"]["A"][d_A - k : d_A], float)
            for me, other in (("A", "B"), ("B", "A")):
                r0, r1 = s["rounds"][0], s["rounds"][-1]
                rows.append(
                    dict(
                        run=run,
                        twin=me,
                        t=s["t"],
                        y=s["y"],
                        own=r0[me]["estimate"],
                        partner=r0[other]["estimate"],
                        final=r1[me]["estimate"],
                        team=s["consensus"],
                        shared=float(xs @ w_shared),
                        x_shared=xs.tolist(),
                        ok=r0[me]["ok"] and r1[me]["ok"],
                    )
                )
    return rows, cfg


def fit_combination(rows: list[dict], k: int) -> np.ndarray:
    """OLS of final on (own, partner[, x_shared], 1): returns (alpha, beta, gamma..., c)."""
    X = [[r["own"], r["partner"], *r["x_shared"][:k], 1.0] for r in rows]
    coef, *_ = np.linalg.lstsq(np.asarray(X), np.array([r["final"] for r in rows]), rcond=None)
    return coef


def rule_values(r: dict) -> np.ndarray:
    """The four candidate finals for one row, in the order of RULES."""
    a, b = r["own"], r["partner"]
    return np.array([a, (a + b) / 2, a + b, a + b - r["shared"]])


def analyse(rows: list[dict], k: int) -> dict:
    """Combination fit (pooled, per run, early / late), nearest rule, team vs rules on openings."""
    rows = [r for r in rows if r["ok"]]
    runs = sorted({r["run"] for r in rows})
    T = max(r["t"] for r in rows) + 1
    coef = fit_combination(rows, k)
    per_run = np.array([fit_combination([r for r in rows if r["run"] == u], k) for u in runs])
    early = fit_combination([r for r in rows if r["t"] < T // 2], k)
    late = fit_combination([r for r in rows if r["t"] >= T // 2], k)
    nearest = np.array([np.argmin(np.abs(rule_values(r) - r["final"])) for r in rows])
    unchanged = np.mean([abs(r["final"] - r["own"]) < 1e-6 for r in rows])

    # Team error, and the same rules applied to the openings (one row per step: twin A's view).
    steps = [r for r in rows if r["twin"] == "A"]
    y = np.array([r["y"] for r in steps])
    rule_mse = {
        name: float(np.mean((np.array([rule_values(r)[i] for r in steps]) - y) ** 2))
        for i, name in enumerate(RULES)
    }
    rule_mse["own"] = float(np.mean([(r["own"] - r["y"]) ** 2 for r in rows]))  # both twins
    return dict(
        n_rows=len(rows),
        n_runs=len(runs),
        alpha=float(coef[0]),
        beta=float(coef[1]),
        gamma=[float(g) for g in coef[2 : 2 + k]],
        alpha_runs=per_run[:, 0].tolist(),
        beta_runs=per_run[:, 1].tolist(),
        gamma_runs=per_run[:, 2 : 2 + k].tolist(),
        beta_se=float(per_run[:, 1].std(ddof=1) / np.sqrt(len(runs))) if len(runs) > 1 else None,
        beta_early=float(early[1]),
        beta_late=float(late[1]),
        alpha_early=float(early[0]),
        alpha_late=float(late[0]),
        nearest_rule={name: float(np.mean(nearest == i)) for i, name in enumerate(RULES)},
        unchanged=float(unchanged),
        team_mse=float(np.mean([(r["team"] - r["y"]) ** 2 for r in steps])),
        rule_mse_on_openings=rule_mse,
        noise_var=None,
    )


def main(argv: list[str] | None = None) -> Path:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--arms", nargs="+", default=["O1h", "O1n", "O0h", "O0n"])
    p.add_argument("--prefix", default="llm_haiku_overlap")
    args = p.parse_args(argv)

    out, lines = {}, []
    lines.append(
        "| arm | overlap | alpha (own) | beta (partner) | gamma (x_shared) | beta early / late "
        "| nearest: own / avg / sum / sum−shared | unchanged | team MSE | MSE of rule on openings: "
        "own / avg / sum / sum−shared |"
    )
    lines.append("|" + "---|" * 10)
    for arm in args.arms:
        run_dir = latest_run(args.prefix, arm)
        rows, cfg = load_steps(run_dir, arm)
        k = cfg["env"].get("overlap", 0)
        res = analyse(rows, k)
        res.update(run_dir=str(run_dir), overlap=k, noise_var=cfg["env"]["sigma"] ** 2)
        out[arm] = res
        nr, rm = res["nearest_rule"], res["rule_mse_on_openings"]
        g = ", ".join(f"{v:.2f}" for v in res["gamma"]) or "—"
        se = f" ± {res['beta_se']:.2f}" if res["beta_se"] is not None else ""
        lines.append(
            f"| {arm} | {k} | {res['alpha']:.2f} | {res['beta']:.2f}{se} | {g} "
            f"| {res['beta_early']:.2f} / {res['beta_late']:.2f} "
            f"| {nr['own']:.0%} / {nr['average']:.0%} / {nr['sum']:.0%} / {nr['sum - shared']:.0%} "
            f"| {res['unchanged']:.0%} | {res['team_mse']:.3f} "
            f"| {rm['own']:.3f} / {rm['average']:.3f} / {rm['sum']:.3f} / {rm['sum - shared']:.3f} |"
        )
    run_out = make_run_dir(str(ROOT / "results"), "overlap_analysis")
    (run_out / "summary.json").write_text(json.dumps(out, indent=2))
    table = "\n".join(lines)
    (run_out / "table.md").write_text(table + "\n")
    plot_overlap(out, run_out / "overlap.png")
    print(table)
    print(f"saved to {run_out}")
    return run_out


if __name__ == "__main__":
    main()
