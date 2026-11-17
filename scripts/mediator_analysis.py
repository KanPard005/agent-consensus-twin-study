"""Mediated self-learning twins (M1, `consensus.mediator`): how the mediator combines the twins.

Usage:
    python scripts/mediator_analysis.py results/llm_haiku_mediated_rho0.0 ... [--tail 0.5]
        [--label M1] [--talk S1n]

Takes the latest run in each folder and reports per rho, over the last `tail` of the steps and
pooled over runs (differences are paired on the same streams):

- excess MSE over sigma^2 of P0 (no talk), P1 (frozen linear blend), the talking twins (S1n),
  the mediator (M1), and two references built from the M1 twins' own estimates: their plain
  average, and the in-sample best linear blend a + w_A est_A + w_B est_B (fitted on the tail
  steps it is scored on, so an optimistic bound for any fixed rule over these two numbers);
- the mediator's fitted blend: least squares of its prediction on (1, est_A, est_B) over the
  tail steps, and the same for the best blend, to compare with the linear theory (at rho = 0
  each twin carries half the signal, so the best weights are about 1 each, sum about 2;
  averaging is weights 0.5 / 0.5);
- expand: share of steps where the mediator's prediction lies outside the range of the two
  estimates (an average never does; adding two same-signed half-signals does);
- the M1 twins' solo excess MSE, against the S1n twins' round-1 (solo) excess, to check that
  the two arms' twins learned comparably well.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import yaml


def latest_run(path: Path) -> Path:
    """The folder itself if it is a run, else its latest timestamped run."""
    if (path / "config.yaml").exists():
        return path
    return sorted(p for p in path.iterdir() if (p / "config.yaml").exists())[-1]


def load_steps(run_dir: Path, label: str) -> list[list[dict]]:
    """Per replicate, the list of per-step transcript rows (training-phase files skipped)."""
    out = []
    for f in sorted((run_dir / "transcripts").glob(f"{label}_run*.jsonl")):
        if re.fullmatch(rf"{re.escape(label)}_run\d+\.jsonl", f.name):
            out.append([json.loads(line) for line in f.open()])
    return out


def fit_blend(e_A: np.ndarray, e_B: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Least squares of `target` on (1, e_A, e_B); returns (a, w_A, w_B). Shapes (n,)."""
    X = np.c_[np.ones_like(e_A), e_A, e_B]
    coef, *_ = np.linalg.lstsq(X, target, rcond=None)
    return coef


def rho_report(run_dir: Path, label: str, talk: str, tail: float) -> dict:
    """All statistics for one run folder (one rho)."""
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    sigma2 = cfg["env"]["sigma"] ** 2
    z = np.load(run_dir / "metrics.npz")
    T = z[f"{label}/sq_err"].shape[1]
    t0 = int(round(T * (1 - tail)))

    def excess(lab: str) -> np.ndarray:  # per run, shape (n_runs,)
        return z[f"{lab}/sq_err"][:, t0:].mean(axis=1) - sigma2

    runs = load_steps(run_dir, label)
    e_A = np.array([[s["rounds"][0]["A"]["estimate"] for s in r] for r in runs])  # (runs, T)
    e_B = np.array([[s["rounds"][0]["B"]["estimate"] for s in r] for r in runs])
    med = np.array([[s["consensus"] for s in r] for r in runs])
    y = np.array([[s["y"] for s in r] for r in runs])
    tl = slice(t0, T)
    avg = (e_A + e_B) / 2
    avg_ex = ((avg - y)[:, tl] ** 2).mean(axis=1) - sigma2
    best = fit_blend(e_A[:, tl].ravel(), e_B[:, tl].ravel(), y[:, tl].ravel())
    best_pred = best[0] + best[1] * e_A + best[2] * e_B
    best_ex = ((best_pred - y)[:, tl] ** 2).mean(axis=1) - sigma2
    med_w = fit_blend(e_A[:, tl].ravel(), e_B[:, tl].ravel(), med[:, tl].ravel())
    lo, hi = np.minimum(e_A, e_B), np.maximum(e_A, e_B)
    expand = ((med < lo - 1e-9) | (med > hi + 1e-9))[:, tl].mean()
    solo_m = (((e_A - y) ** 2 + (e_B - y) ** 2) / 2).mean() - sigma2
    out = {
        "rho": cfg["env"]["rho"],
        "n_runs": len(runs),
        "P0": excess("P0"),
        "P1": excess("P1"),
        label: excess(label),
        "avg": avg_ex,
        "best": best_ex,
        "med_w": med_w,
        "best_w": best,
        "expand": expand,
        "solo_m": solo_m,
        "fail": float(np.nansum(z[f"{label}/mediator_fail"])),
    }
    if f"{talk}/sq_err" in z.files:
        out[talk] = excess(talk)
        t_runs = load_steps(run_dir, talk)
        r1 = [
            (s["rounds"][0][k]["estimate"] - s["y"]) ** 2 - sigma2
            for r in t_runs
            for s in r
            for k in "AB"
        ]
        out["solo_talk"] = float(np.mean(r1))
    return out


def fmt(a: np.ndarray) -> str:
    """Mean ± s.e. over runs."""
    se = a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else np.nan
    return f"{a.mean():.2f} ± {se:.2f}"


def paired(a: np.ndarray, b: np.ndarray) -> str:
    """Mean ± s.e. of a - b, with the count of runs where a is worse."""
    return f"{fmt(a - b)} (worse in {int((a > b).sum())}/{len(a)})"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("folders", nargs="+")
    p.add_argument("--tail", type=float, default=0.5)
    p.add_argument("--label", default="M1")
    p.add_argument("--talk", default="S1n")
    args = p.parse_args(argv)
    L, S = args.label, args.talk
    reps = [rho_report(latest_run(Path(f)), L, S, args.tail) for f in args.folders]

    print(
        f"| ρ | P0 | P1 | {S} | {L} | avg of {L} twins | best blend of {L} twins | "
        f"{L} − {S} | {L} − avg |"
    )
    print("|---" * 9 + "|")
    for r in reps:
        s = r.get(S)
        print(
            f"| {r['rho']} | {fmt(r['P0'])} | {fmt(r['P1'])} | "
            f"{fmt(s) if s is not None else '–'} | **{fmt(r[L])}** | {fmt(r['avg'])} | "
            f"{fmt(r['best'])} | {paired(r[L], s) if s is not None else '–'} | "
            f"{paired(r[L], r['avg'])} |"
        )
    print(
        f"\n| ρ | {L} weights a / w_A / w_B (sum) | best-blend weights a / w_A / w_B (sum) | "
        f"{L} outside the twins' range | solo excess: {L} twins / {S} round 1 | "
        "unusable mediator replies |"
    )
    print("|---" * 6 + "|")
    for r in reps:
        m, b = r["med_w"], r["best_w"]
        st = f"{r['solo_talk']:.2f}" if "solo_talk" in r else "–"
        print(
            f"| {r['rho']} | {m[0]:+.2f} / {m[1]:.2f} / {m[2]:.2f} ({m[1] + m[2]:.2f}) | "
            f"{b[0]:+.2f} / {b[1]:.2f} / {b[2]:.2f} ({b[1] + b[2]:.2f}) | {r['expand']:.0%} | "
            f"{r['solo_m']:.2f} / {st} | {r['fail']:.0f} |"
        )


if __name__ == "__main__":
    main()
