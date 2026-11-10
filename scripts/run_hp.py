"""Run the hidden-profile experiment: LLM twins choose café sites from split verbal clues.

    python scripts/run_hp.py configs/hp_haiku.yaml [--set cases.p_shared=0.25 experiment=...]

For each replicate it samples a case set (`consensus.hp_data.HiddenProfileGenerator`), runs every
protocol in the config on the same cases, and saves to results/<experiment>/<timestamp>/:
config.yaml (with the git commit), metrics.npz (per protocol, runs x cases), summary.json, and
transcripts/<label>_run<r>.jsonl. Rule-based references are computed from the cases alone:

- own_A / own_B: the twin's own best site (sum of the clue values it sees), alone;
- add_views: argmax of view_A + view_B, i.e. each twin reports its own site scores and the team
  adds them. This double-counts shared clues, the analogue of averaging two opinions;
- the pooled best site is correct by definition (accuracy 1).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consensus.hp_data import Case, HiddenProfileGenerator  # noqa: E402
from consensus.hp_protocol import build_hp_protocol  # noqa: E402
from consensus.utils import (  # noqa: E402
    git_hash,
    load_config,
    make_run_dir,
    run_rng,
    save_jsonl,
    save_run,
)

log = logging.getLogger("run_hp")


def rule_references(cases: list[Case]) -> dict[str, np.ndarray]:
    """Accuracy of the rule-based references on one case set (see the module docstring)."""
    best = np.array([c.best for c in cases])
    picks = {
        "own_A": [np.argmax(c.view("A")) for c in cases],
        "own_B": [np.argmax(c.view("B")) for c in cases],
        "add_views": [np.argmax(c.view("A") + c.view("B")) for c in cases],
    }
    return {k: (np.array(v) == best).astype(float) for k, v in picks.items()}


def summarize(results: dict, kinds: np.ndarray) -> dict:
    """Per label: means over runs and cases, overall and by case kind."""
    out = {}
    for lab, res in results.items():
        s = {}
        for metric, arr in res.items():
            if metric.startswith("_"):
                continue
            s[metric] = float(np.nanmean(arr))
            for kind in ("hidden", "control"):
                sel = kinds == kind
                if sel.any():
                    s[f"{metric}_{kind}"] = float(np.nanmean(np.where(sel, arr, np.nan)))
        out[lab] = s
    return out


def main(argv: list[str] | None = None) -> Path:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("config")
    p.add_argument("--set", nargs="*", default=[], help="overrides like cases.p_shared=0.25")
    p.add_argument("--results", default=str(ROOT / "results"))
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")

    cfg = load_config(args.config, args.set)
    cc = dict(cfg["cases"])
    n_hidden, n_control = cc.pop("n_hidden"), cc.pop("n_control")
    gen = HiddenProfileGenerator(**cc)
    labels = [pc["label"] for pc in cfg["protocols"]]
    n_runs, T = cfg["n_runs"], n_hidden + n_control
    first_run = cfg.setdefault("first_run", 0)

    results: dict[str, dict[str, np.ndarray]] = {}
    kinds = np.empty((n_runs, T), dtype=object)
    transcripts, costs = {}, {lab: [] for lab in labels}
    for i in range(n_runs):
        r = first_run + i
        cases = gen.sample_set(run_rng(cfg["seed"], r), n_hidden, n_control)
        kinds[i] = [c.kind for c in cases]
        for k, arr in rule_references(cases).items():
            results.setdefault(k, {}).setdefault("correct", np.empty((n_runs, T)))[i] = arr
        for pc in cfg["protocols"]:
            proto = build_hp_protocol(pc)
            for metric, arr in proto.run(cases).items():
                results.setdefault(pc["label"], {}).setdefault(metric, np.empty((n_runs, T)))[i] = (
                    arr
                )
            transcripts[(pc["label"], r)] = proto.transcript
            costs[pc["label"]].append(proto.cost)
        log.info("run %d/%d done (replicate %d)", i + 1, n_runs, r)

    summary = summarize(results, kinds)
    for lab in labels:
        summary[lab]["cost_usd_total"] = sum(c["cost_usd"] for c in costs[lab])
        summary[lab]["tokens_in_total"] = sum(c["tokens_in"] for c in costs[lab])
    run_dir = make_run_dir(args.results, cfg["experiment"])
    save_run(run_dir, cfg, git_hash(ROOT), results, {}, summary)
    (run_dir / "kinds.json").write_text(json.dumps(kinds.tolist()))
    for (lab, r), rows in transcripts.items():
        save_jsonl(run_dir / "transcripts" / f"{lab}_run{r}.jsonl", rows)
    log.info("saved to %s", run_dir)
    for lab, s in summary.items():
        log.info(
            "%-10s accuracy %.2f (hidden %.2f, control %.2f)%s",
            lab,
            s["correct"],
            s.get("correct_hidden", np.nan),
            s.get("correct_control", np.nan),
            f", cost ${s['cost_usd_total']:.2f}" if "cost_usd_total" in s else "",
        )
    return run_dir


if __name__ == "__main__":
    main()
