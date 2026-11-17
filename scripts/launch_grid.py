"""Launch the live-session entrenchment grid as a job queue (one pair per process).

Usage:
    python scripts/launch_grid.py [--jobs 16] [--pairs 8] [--lambdas 0.0 0.3 0.6 0.9]
        [--ns 25 100 400 1600] [--alphas 0 0.25 0.5 1] [--dry-run]

Each job is one pair: `scripts/run_sessions.py configs/llm_haiku_sessions_entrench.yaml` with
the cell's lambda and budgets, first_run = pair index, experiment =
llm_haiku_sessions_entrench_lam<lambda>_<cell> (cell = eq<n> or mix<n>v25). At most `--jobs`
pairs run at once; each writes its live files as it goes (scripts/sessions_status.py), and its
log to results/logs/entrench_<cell>_lam<lambda>_r<pair>.log. Jobs already finished (a summary
with that run) are skipped, so the launcher can be rerun to fill in gaps.

`--alphas` runs the asymmetric-information design (LogBook 2026-11-14): one twin's weights are
scaled by alpha (env.alpha), the weak twin alternating B (even pairs) and A (odd pairs), and the
experiment folders become <prefix>_lam<lambda>_al<alpha>_<cell>.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
CONFIG = "configs/llm_haiku_sessions_entrench.yaml"


def done(experiment: str, r: int) -> bool:
    for f in (ROOT / "results" / experiment).glob("20*/summary.json"):
        if f"run{r}" in json.loads(f.read_text()):
            return True
    return False


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jobs", type=int, default=16)
    p.add_argument("--pairs", type=int, default=8)
    p.add_argument("--lambdas", nargs="+", default=["0.0", "0.3", "0.6", "0.9"])
    p.add_argument("--ns", nargs="+", type=int, default=[25, 100, 400, 1600])
    p.add_argument("--short", type=int, default=25)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--config", default=CONFIG)
    p.add_argument("--no-mixed", action="store_true", help="equal pairs only")
    p.add_argument("--prefix", default="llm_haiku_sessions_entrench",
                   help="experiment folders: <prefix>_lam<lambda>_<cell>")
    p.add_argument("--alphas", nargs="+", default=None,
                   help="asymmetric design: scale the weak twin's weights by each alpha")
    args = p.parse_args()
    jobs = []
    # Pair index outermost, so early results cover every cell.
    for r, lam, al in itertools.product(range(args.pairs), args.lambdas, args.alphas or [None]):
        # Asymmetric design: the weak twin alternates by pair so that label effects (a vs b) cancel.
        asym = [] if al is None else [f"env.alpha={al}", f"env.weak={'B' if r % 2 == 0 else 'A'}"]
        tag = "" if al is None else f"_al{al}"
        for n in args.ns:
            cells = [(f"eq{n}", [f"agent.budget={n}", *asym])]
            if n > args.short and not args.no_mixed:
                cells.append((f"mix{n}v{args.short}",
                              [f"agent.budget_long={n}", f"agent.budget_short={args.short}", *asym]))
            for cell, sets in cells:
                exp = f"{args.prefix}_lam{lam}{tag}_{cell}"
                if done(exp, r):
                    continue
                log = ROOT / "results" / "logs" / f"{args.prefix}_{cell}_lam{lam}{tag}_r{r}.log"
                jobs.append((exp, r, [f"env.lambda_xz={lam}", f"first_run={r}", "n_runs=1",
                                      f"experiment={exp}", *sets], log))
    print(f"{len(jobs)} jobs, {args.jobs} at a time")
    if args.dry_run:
        for j in jobs[:5]:
            print(j[0], j[1], " ".join(j[2]))
        return
    (ROOT / "results" / "logs").mkdir(parents=True, exist_ok=True)

    config = args.config

    def run(job):
        exp, r, sets, logf = job
        t0 = time.time()
        with open(logf, "w") as f:
            code = subprocess.call([PY, "scripts/run_sessions.py", config, "--set", *sets],
                                   cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        print(f"{time.strftime('%H:%M:%S')} {exp} pair {r}: exit {code}, "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)

    with ThreadPoolExecutor(args.jobs) as ex:
        list(ex.map(run, jobs))
    print("all jobs finished", flush=True)


if __name__ == "__main__":
    main()
