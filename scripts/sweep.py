"""Summarize a one-parameter sweep over saved run folders, with population limits.

Usage:
    python scripts/sweep.py <name> --param env.rho results/llm_haiku_auto_own results/llm_haiku_auto_own_rho* \
        [--labels P0 P1 P2 P3 P8] [--grid 0 0.95 39]

Each folder (or the latest timestamped run inside it) contributes one value of the parameter.
Prefix a folder with `LABELS@` (comma-separated) to take only those protocols from it, e.g.
`P8@results/llm_haiku_auto_own_rho0.0`. Replicates of the same (value, label) from several
folders are pooled by replicate index (`first_run` in each run's config); a replicate already
counted from an earlier folder is skipped.
read from its config. For every label the excess MSE over the summary tail (`summary_tail`,
default 0.5) is averaged per run, then reported as mean ± s.e. over runs. Population limits of
the linear protocols are recomputed on a grid with the other env settings of the first folder.
Writes results/sweeps/<name>/<timestamp>/ with table.md, table.json and figures/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consensus.data import LinearTwinEnv  # noqa: E402
from consensus.plotting import plot_sweep  # noqa: E402
from consensus.theory import protocol_benchmarks  # noqa: E402
from consensus.utils import load_run, make_run_dir  # noqa: E402

LIMIT_KEYS = {"P0": "no_talk", "P1": "blend", "P2": "exchange_1", "P3": "exchange_3"}


def latest_run(path: Path) -> Path:
    if (path / "config.yaml").exists():
        return path
    runs = sorted(p for p in path.iterdir() if (p / "config.yaml").exists())
    if not runs:
        raise FileNotFoundError(f"no run folder under {path}")
    return runs[-1]


def get(cfg: dict, dotted: str):
    for k in dotted.split("."):
        cfg = cfg[k]
    return cfg


def main(argv: list[str] | None = None) -> Path:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("name")
    p.add_argument("folders", nargs="+")
    p.add_argument("--param", required=True, help="dotted config key, e.g. env.rho")
    p.add_argument("--labels", nargs="*", help="protocol labels to include (default: all)")
    p.add_argument("--grid", nargs=3, type=float, default=None, help="start stop count for limits")
    args = p.parse_args(argv)

    # (param value, label) -> {replicate index: tail mean}, summed cost, run folders.
    pooled: dict[tuple[float, str], dict] = {}
    first_cfg = None
    for spec in args.folders:
        only, _, folder = spec.rpartition("@")
        only = set(only.split(",")) if only else None
        run_dir = latest_run(Path(folder))
        cfg, results, _ = load_run(run_dir)
        first_cfg = first_cfg or cfg
        value = float(get(cfg, args.param))
        tail = cfg.get("summary_tail", 0.1)
        first_run = cfg.get("first_run", 0)
        summary = json.loads((run_dir / "summary.json").read_text())
        for label, res in results.items():
            if (args.labels and label not in args.labels) or (only and label not in only):
                continue
            T = res["sq_err"].shape[1]
            per_run = res["sq_err"][:, int((1 - tail) * T) :].mean(1) - cfg["env"]["sigma"] ** 2
            entry = pooled.setdefault((value, label), {"reps": {}, "cost": 0.0, "dirs": []})
            new_reps = {
                first_run + i: v
                for i, v in enumerate(per_run)
                if first_run + i not in entry["reps"]
            }
            if not new_reps:
                continue  # every replicate already counted from an earlier folder
            entry["reps"].update(new_reps)
            entry["cost"] += summary["protocols"].get(label, {}).get("cost_usd_total") or 0.0
            entry["dirs"].append(str(run_dir))
    rows = [
        (
            v,
            lab,
            np.array([e["reps"][k] for k in sorted(e["reps"])]),
            e["cost"] or None,
            ", ".join(e["dirs"]),
        )
        for (v, lab), e in pooled.items()
    ]
    rows.sort(key=lambda r: (r[0], r[1]))

    labels = list(dict.fromkeys(r[1] for r in sorted(rows, key=lambda r: r[1])))
    if args.labels:
        labels = [lab for lab in args.labels if lab in labels]
    values = sorted({r[0] for r in rows})

    # Population limits on a grid (only meaningful for env parameters).
    env_cfg = dict(first_cfg["env"])
    lo, hi, n = args.grid or (min(values), max(values), 39)
    grid = np.linspace(lo, hi, int(n))
    section, _, key = args.param.partition(".")

    def limits_at(g: float) -> dict[str, float]:
        """Population excess MSE of the linear protocols with the swept parameter set to g."""
        cfg_g = dict(env_cfg)
        if section == "env":
            cfg_g[key] = int(g) if key in ("d_A", "d_B") else g
        env = LinearTwinEnv.from_config(cfg_g, np.random.default_rng(first_cfg["seed"]))
        bench = protocol_benchmarks(env.Sigma, env.w, env.d_A, env.sigma, n_exchanges=3)
        return {
            lab: bench[LIMIT_KEYS[lab]] - bench["full_info"] for lab in labels if lab in LIMIT_KEYS
        }

    on_grid = [limits_at(g) for g in grid]
    limits = {lab: [d[lab] for d in on_grid] for lab in labels if lab in LIMIT_KEYS}
    limit_at = {(v, lab): x for v in values for lab, x in limits_at(v).items()}
    limits = {lab: np.array(v) for lab, v in limits.items()}

    points = {}
    for lab in labels:
        sel = [r for r in rows if r[1] == lab]
        x = np.array([r[0] for r in sel])
        m = np.array([r[2].mean() for r in sel])
        se = np.array(
            [r[2].std(ddof=1) / np.sqrt(len(r[2])) if len(r[2]) > 1 else 0.0 for r in sel]
        )
        points[lab] = (x, m, se)

    # Table: one row per label, one column per parameter value.
    head = "| | " + " | ".join(f"{args.param.split('.')[-1]} = {v:g}" for v in values) + " |"
    lines = [
        "Excess MSE over $\\sigma^2$ over the summary tail, mean ± s.e. over runs "
        "(population limit in parentheses); [n] = number of runs.",
        "",
        head,
        "|---|" + "---|" * len(values),
    ]
    table = {}
    for lab in labels:
        cells = []
        for v in values:
            r = next((r for r in rows if r[1] == lab and r[0] == v), None)
            if r is None:
                cells.append("–")
                continue
            se = r[2].std(ddof=1) / np.sqrt(len(r[2])) if len(r[2]) > 1 else 0.0
            lim = limit_at.get((v, lab))
            cells.append(
                f"{r[2].mean():.2f} ± {se:.2f}"
                + (f" ({lim:.2f})" if lim is not None else "")
                + f" [{len(r[2])}]"
            )
            table.setdefault(lab, {})[f"{v:g}"] = {
                "mean": float(r[2].mean()),
                "se": float(se),
                "per_run": r[2].tolist(),
                "limit": lim,
                "cost_usd": r[3],
                "run_dir": r[4],
            }
        lines.append(f"| {lab} | " + " | ".join(cells) + " |")

    out = make_run_dir(ROOT / "results" / "sweeps", args.name)
    (out / "table.md").write_text("\n".join(lines) + "\n")
    (out / "table.json").write_text(json.dumps(table, indent=2))
    (out / "figures").mkdir(exist_ok=True)
    fig, _ = plot_sweep(grid, limits, points, args.param.split(".")[-1])
    fig.tight_layout()
    fig.savefig(out / "figures" / "sweep.png", dpi=150)
    plt.close(fig)
    print("\n".join(lines))
    print(out)
    return out


if __name__ == "__main__":
    main()
