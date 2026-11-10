"""Product task, online twins: do the LLM twins learn to multiply their opinions?

Usage:
    python scripts/product_online.py results/llm_haiku_product_online_r0 [... more folders]
        [--label O] [--windows 20]

Takes the latest run in each folder (one replicate each, or several) and reports, per window of
`windows` steps and pooled over replicates, the excess MSE over sigma^2 of:

- team: the LLM team (average of the two final estimates);
- r1 avg / r1 sum / r1 product: the plain average, sum minus 1 and product of the two twins'
  round-1 (solo) estimates, i.e. what the team would score with each rule applied to the same
  opinions (with mu = 1 the product is the right rule, `data.ProductTwinEnv`);
- own r1: each twin's own round-1 estimate (mean of A and B);
- P0 / PX: the online linear twins averaged / multiplied (from metrics.npz).

It also regresses the team's estimate on (1, r1_A, r1_B, r1_A * r1_B) over each window: a
weight near 1 on the product and near 0 on the linear terms means the twins multiply, while
weights near (1/2, 1/2, 0) mean they average. Finally it counts the steps whose messages or
notes mention multiplying (multipl / product / times / ×), and the first step that does.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import yaml

MULT = re.compile(r"multipl|product|\btimes\b|×|\* ?partner|partner ?\*", re.I)


def latest_run(path: Path) -> Path:
    """The folder itself if it is a run, else its latest timestamped run."""
    if (path / "config.yaml").exists():
        return path
    return sorted(p for p in path.iterdir() if (p / "config.yaml").exists())[-1]


def load(folder: Path, label: str) -> list[dict]:
    """Per replicate: arrays of y, team, round-1 and final estimates, P0/PX errors, mentions."""
    run_dir = latest_run(folder)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    z = np.load(run_dir / "metrics.npz")
    reps = []
    files = sorted((run_dir / "transcripts").glob(f"{label}_run*.jsonl"))
    files = [f for f in files if re.fullmatch(rf"{label}_run\d+\.jsonl", f.name)]
    for i, f in enumerate(files):
        st = [json.loads(line) for line in f.open()]
        get = lambda r, k: np.array([s["rounds"][r][k]["estimate"] for s in st])  # noqa: E731
        text = [
            " ".join(s["rounds"][r][k]["message"] + " " + s["rounds"][r][k]["notes"]
                     for r in range(len(s["rounds"])) for k in "AB")
            for s in st
        ]
        reps.append(
            dict(
                y=np.array([s["y"] for s in st]),
                team=np.array([s["consensus"] for s in st]),
                a1=get(0, "A"), b1=get(0, "B"),
                P0=z["P0/sq_err"][i] if "P0/sq_err" in z else None,
                PX=z["PX/sq_err"][i] if "PX/sq_err" in z else None,
                mult=np.array([bool(MULT.search(t)) for t in text]),
                sigma2=cfg["env"]["sigma"] ** 2,
                mu=cfg["env"].get("mu", 1.0),
            )
        )
    return reps


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("folders", nargs="+")
    p.add_argument("--label", default="O")
    p.add_argument("--windows", type=int, default=20, help="steps per window")
    args = p.parse_args(argv)
    reps = [r for f in args.folders for r in load(Path(f), args.label)]
    T = len(reps[0]["y"])
    print(f"{len(reps)} replicates, T = {T}\n")
    print("| steps | team | r1 avg | r1 sum−μ² | r1 product | own r1 | P0 | PX "
          "| team ~ 1, a, b, a·b |")
    print("|---|---|---|---|---|---|---|---|---|")
    for lo in range(0, T, args.windows):
        sl = slice(lo, min(lo + args.windows, T))

        def ex(f) -> tuple[float, float]:
            v = [np.mean(f(r)[sl]) - r["sigma2"] for r in reps]
            return float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0

        rows = {
            "team": ex(lambda r: (r["team"] - r["y"]) ** 2),
            "avg": ex(lambda r: ((r["a1"] + r["b1"]) / 2 - r["y"]) ** 2),
            "sum": ex(lambda r: (r["a1"] + r["b1"] - r["mu"] ** 2 - r["y"]) ** 2),
            "prod": ex(lambda r: (r["a1"] * r["b1"] / r["mu"] ** 2 - r["y"]) ** 2),
            "own": ex(lambda r: ((r["a1"] - r["y"]) ** 2 + (r["b1"] - r["y"]) ** 2) / 2),
        }
        P0 = ex(lambda r: r["P0"]) if reps[0]["P0"] is not None else (np.nan, 0)
        PX = ex(lambda r: r["PX"]) if reps[0]["PX"] is not None else (np.nan, 0)
        a = np.concatenate([r["a1"][sl] for r in reps])
        b = np.concatenate([r["b1"][sl] for r in reps])
        t = np.concatenate([r["team"][sl] for r in reps])
        X = np.column_stack([np.ones_like(a), a, b, a * b])
        coef, *_ = np.linalg.lstsq(X, t, rcond=None)
        f = lambda m: f"{m[0]:.2f} ± {m[1]:.2f}"  # noqa: E731
        print(
            f"| {sl.start + 1}–{sl.stop} | {f(rows['team'])} | {f(rows['avg'])} | {f(rows['sum'])} "
            f"| {f(rows['prod'])} | {f(rows['own'])} | {f(P0)} | {f(PX)} "
            f"| {coef[0]:.2f}, {coef[1]:.2f}, {coef[2]:.2f}, {coef[3]:.2f} |"
        )
    print()
    for i, r in enumerate(reps):
        first = int(np.argmax(r["mult"])) + 1 if r["mult"].any() else None
        print(f"replicate {i}: {r['mult'].sum()} of {T} steps mention multiplying; first at step {first}")


if __name__ == "__main__":
    main()
