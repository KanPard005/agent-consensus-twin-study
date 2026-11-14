"""Tool use by live-session twins (LogBook 2026-11-13; configs/llm_haiku_sessions_samey2*.yaml).

Questions: do the twins use their tools (regress, model instances) when answering queries, does
that change over the run or with how many training cases they studied, and how much do they
weigh their partner when they don't use tools?

Per twin and query (from the tool-call logs and the query records):
- tool use in the query: any regress / new_model / fit_model / add_case call during that query;
- in training: regress calls, models built (and whether they include nonlinear terms);
- WOA on the twin's own final, split by whether it used a tool in that query;
- hand arithmetic: the twin's opening against the prediction of its own least-squares fit on the
  training cases it studied (rebuilt from the seed), i.e. how well it computes its own model
  without the tool; and the error of each against y.

Usage:
    python scripts/tools_analysis.py --prefix llm_haiku_sessions_samey2_d20 --ns 50 200 800
        [--lambdas 0.0 0.5]
Writes results/tools/<timestamp>/{stats.json, table.md, tools.png}.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run import train_twins  # noqa: E402

from consensus.data import build_env  # noqa: E402
from consensus.plotting_overlap import plot_tools  # noqa: E402
from consensus.sessions import weight_of_advice  # noqa: E402
from consensus.utils import make_run_dir, run_rng  # noqa: E402

TOOLS = ("regress", "new_model", "fit_model", "add_case")
WINDOWS = ((0, 10), (10, 20), (20, 30))


def m(v) -> float:
    v = [x for x in v if x == x]
    return float(np.mean(v)) if v else float("nan")


def se(v) -> float:
    v = [x for x in v if x == x]
    return float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")


def ols_predictor(X: np.ndarray, y: np.ndarray):
    A = np.column_stack([np.ones(len(X)), X])
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    return lambda x: float(coef[0] + np.asarray(x) @ coef[1:])


def load(prefix: str, lam: str, n: int) -> list[dict]:
    rows = []
    for d in sorted((ROOT / "results" / f"{prefix}_lam{lam}_eq{n}").glob("20*")):
        sf = d / "summary.json"
        if not sf.exists():
            continue
        cfg = yaml.safe_load((d / "config.yaml").read_text())
        env = build_env(cfg["env"], np.random.default_rng(cfg["seed"]))
        for key in json.loads(sf.read_text()):
            r = int(key[3:])
            A, B = train_twins(env, cfg["train"], run_rng(cfg["seed"], r))
            fits = {k: ols_predictor(*(lambda X, y: (X[:n], y[:n]))(*ag.training_data()))
                    for k, ag in (("A", A), ("B", B))}
            recs = [json.loads(x) for x in (d / "transcripts" / f"run{r}.jsonl").read_text()
                    .splitlines() if x.strip()]
            for k, o in (("A", "B"), ("B", "A")):
                calls = [json.loads(x) for x in (d / "transcripts" / f"run{r}_calls_{k}.jsonl")
                         .read_text().splitlines() if x.strip()]
                train_calls = [c for c in calls if c["q"] < 0]
                models = [c for c in train_calls if c["tool"] == "new_model"
                          and not c["result"].startswith("Error")]
                nonlin = any(any(op in col for op in ("*", "^"))
                             for c in models for col in (c["args"].get("columns") or []))
                byq = Counter((c["q"], c["tool"]) for c in calls if c["q"] >= 0)
                for s in recs:
                    q = s["t"]
                    used = {t: byq.get((q, t), 0) for t in TOOLS}
                    fin = s["attempts"][k][-1] if s["attempts"][k] else float("nan")
                    own, par = s["opening"][k], s["opening"][o]
                    pred = fits[k](s["x"][k])
                    rows.append(dict(
                        lam=lam, n=n, pair=r, twin=k, t=q,
                        tool=any(used.values()), **{f"n_{t}": v for t, v in used.items()},
                        train_regress=sum(c["tool"] == "regress" for c in train_calls),
                        train_models=len(models), train_nonlinear=nonlin,
                        woa=weight_of_advice(own, par, fin),
                        unchanged=abs(fin - own) < 1e-9 if fin == fin else np.nan,
                        arith=abs(own - pred), arith_ok=abs(own - pred) <= 0.1,
                        err_open=(own - s["y"]) ** 2, err_ols=(pred - s["y"]) ** 2,
                        err_final=(fin - s["y"]) ** 2 if fin == fin else np.nan,
                        msgs=sum(mm["from"] == k for mm in s["messages"])))
    return rows


def main(argv: list[str] | None = None) -> Path:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prefix", default="llm_haiku_sessions_samey2_d20")
    p.add_argument("--lambdas", nargs="+", default=["0.0", "0.5"])
    p.add_argument("--ns", nargs="+", type=int, default=[50, 200, 800])
    p.add_argument("--out", default="tools")
    args = p.parse_args(argv)
    out, lines = {"cells": {}}, []
    lines += ["| λ | n | twin-queries | tool used in the query | by queries 1–10 / 11–20 / 21–30 "
              "| twins ever using a tool in queries | training: regress calls / models / nonlinear "
              "| WOA: no tool / tool | finals left at the opening | opening vs own fit: mean "
              "\\|diff\\|, within 0.1 | MSE: opening / own fit / final |",
              "|" + "---|" * 11]
    for lam in args.lambdas:
        for n in args.ns:
            rows = load(args.prefix, lam, n)
            if not rows:
                continue
            twins = {(r["pair"], r["twin"]) for r in rows}
            ever = m([any(r["tool"] for r in rows if (r["pair"], r["twin"]) == t) for t in twins])
            first = {t: next(r for r in rows if (r["pair"], r["twin"]) == t) for t in twins}
            c = dict(
                n_rows=len(rows), tool=m([r["tool"] for r in rows]),
                tool_by_window=[m([r["tool"] for r in rows if a <= r["t"] < b]) for a, b in WINDOWS],
                ever=ever,
                train_regress=m([f["train_regress"] for f in first.values()]),
                train_models=m([f["train_models"] > 0 for f in first.values()]),
                train_nonlinear=m([f["train_nonlinear"] for f in first.values()]),
                woa_notool=m([r["woa"] for r in rows if not r["tool"]]),
                woa_tool=m([r["woa"] for r in rows if r["tool"]]),
                woa_notool_se=se([m([r["woa"] for r in rows if not r["tool"]
                                     and (r["pair"], r["twin"]) == t]) for t in twins]),
                unchanged=m([r["unchanged"] for r in rows]),
                arith=m([r["arith"] for r in rows]), arith_ok=m([r["arith_ok"] for r in rows]),
                err_open=m([r["err_open"] for r in rows]), err_ols=m([r["err_ols"] for r in rows]),
                err_final=m([r["err_final"] for r in rows]),
                tools_used=dict(Counter(t for r in rows for t in TOOLS for _ in range(r[f"n_{t}"]))),
            )
            out["cells"][f"{lam}_{n}"] = c
            lines.append(
                f"| {lam} | {n} | {len(rows)} | {c['tool']:.0%} "
                f"| {' / '.join(f'{v:.0%}' for v in c['tool_by_window'])} | {c['ever']:.0%} "
                f"| {c['train_regress']:.1f} / {c['train_models']:.0%} / {c['train_nonlinear']:.0%} "
                f"| {c['woa_notool']:.2f} / "
                + ("—" if c["woa_tool"] != c["woa_tool"] else f"{c['woa_tool']:.2f}")
                + f" | {c['unchanged']:.0%} | {c['arith']:.2f}, {c['arith_ok']:.0%} "
                f"| {c['err_open']:.2f} / {c['err_ols']:.2f} / {c['err_final']:.2f} |")
    lines.append("\nQuery-phase tool calls by tool: "
                 + json.dumps({k: v["tools_used"] for k, v in out["cells"].items()}))
    run_out = make_run_dir(str(ROOT / "results"), args.out)
    (run_out / "stats.json").write_text(json.dumps(out, indent=2))
    table = "\n".join(lines)
    (run_out / "table.md").write_text(table + "\n")
    if out["cells"]:
        plot_tools(out, args.lambdas, args.ns, run_out / f"{args.out}.png")
    print(table)
    print(f"saved to {run_out}")
    return run_out


if __name__ == "__main__":
    main()
