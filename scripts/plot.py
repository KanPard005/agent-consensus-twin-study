"""Regenerate all figures for a saved run.

Usage:
    python scripts/plot.py results/<experiment>/<timestamp> [--window 200]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consensus.plotting import (  # noqa: E402
    plot_blend,
    plot_disagreement,
    plot_drift,
    plot_excess_mse,
    plot_shift,
)
from consensus.utils import load_run  # noqa: E402


def make_figures(run_dir: str | Path, window: int = 200) -> list[Path]:
    """Write excess-MSE, opinion-shift, weight-drift, disagreement and blend figures into run_dir/figures/."""
    run_dir = Path(run_dir)
    _, results, bench = load_run(run_dir)
    proto_keys = json.loads((run_dir / "summary.json").read_text())["proto_keys"]
    results = {lab: results[lab] for lab in proto_keys}  # config order = color order
    out = run_dir / "figures"
    out.mkdir(exist_ok=True)
    figs = {
        "excess_mse": plot_excess_mse(results, bench, proto_keys, window)[0],
        "shift": plot_shift(results, bench, proto_keys, window)[0],
        "drift": plot_drift(results)[0],
        "disagreement": plot_disagreement(results, window)[0],
    }
    for lab, key in proto_keys.items():
        if key == "blend":
            figs[f"blend_{lab}"] = plot_blend(results, bench, lab)[0]
    paths = []
    for name, fig in figs.items():
        path = out / f"{name}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir")
    p.add_argument("--window", type=int, default=200)
    a = p.parse_args()
    for path in make_figures(a.run_dir, a.window):
        print(path)
