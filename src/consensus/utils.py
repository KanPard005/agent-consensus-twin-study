"""Stateless helpers: config loading, seeding, run folders and result I/O."""

from __future__ import annotations

import copy
import datetime as dt
import json
import subprocess
from pathlib import Path

import numpy as np
import yaml


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Load a YAML config and apply `key.sub=value` overrides (values parsed as YAML)."""
    cfg = yaml.safe_load(Path(path).read_text())
    for item in overrides or []:
        key, _, raw = item.partition("=")
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(raw)
    return cfg


def run_rng(seed: int, run: int) -> np.random.Generator:
    """Independent generator for replicate `run` (SeedSequence spawning via the entropy pair)."""
    return np.random.default_rng([seed, run])


def git_hash(repo: str | Path) -> str:
    """Current commit hash of `repo` (with '-dirty' if there are uncommitted changes), or 'unknown'."""
    try:
        h = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True, check=False
        ).stdout.strip()
        return h + ("-dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def make_run_dir(root: str | Path, experiment: str) -> Path:
    """Create results/<experiment>/<timestamp>/ and return it."""
    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = Path(root) / experiment / stamp
    path.mkdir(parents=True, exist_ok=False)
    return path


def save_run(
    run_dir: Path, cfg: dict, commit: str, results: dict, bench: dict, summary: dict
) -> None:
    """Write the resolved config, commit, raw metrics (.npz), benchmarks and summary."""
    meta = copy.deepcopy(cfg)
    meta["_git_commit"] = commit
    (run_dir / "config.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
    flat = {f"{label}/{m}": arr for label, res in results.items() for m, arr in res.items()}
    np.savez_compressed(run_dir / "metrics.npz", **flat)
    (run_dir / "benchmarks.json").write_text(json.dumps(bench, indent=2))
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))


def save_jsonl(path: str | Path, rows: list[dict]) -> None:
    """Write one JSON object per line (creates parent folders)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def load_run(run_dir: str | Path) -> tuple[dict, dict, dict]:
    """Inverse of `save_run`: returns (cfg, results, bench)."""
    run_dir = Path(run_dir)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    results: dict = {}
    with np.load(run_dir / "metrics.npz") as z:
        for key in z.files:
            label, metric = key.rsplit("/", 1)
            results.setdefault(label, {})[metric] = z[key]
    bench = json.loads((run_dir / "benchmarks.json").read_text())
    return cfg, results, bench
