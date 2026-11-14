"""Launch the live-session entrenchment grid as a job queue (one pair per process).

Usage:
    python scripts/launch_grid.py [--jobs 16] [--pairs 8] [--lambdas 0.0 0.3 0.6 0.9]
        [--ns 25 100 400 1600] [--dry-run]

Each job is one pair: `scripts/run_sessions.py configs/llm_haiku_sessions_entrench.yaml` with
the cell's lambda and budgets, first_run = pair index, experiment =
llm_haiku_sessions_entrench_lam<lambda>_<cell> (cell = eq<n> or mix<n>v25). At most `--jobs`
pairs run at once; each writes its live files as it goes (scripts/sessions_status.py), and its
log to results/logs/entrench_<cell>_lam<lambda>_r<pair>.log. Jobs already finished (a summary
with that run) are skipped, so the launcher can be rerun to fill in gaps.
"""

from __future__ import annotations

import argparse
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
    args = p.parse_args()
    jobs = []
    # Pair index outermost, so early results cover every cell.
    for r in range(args.pairs):
        for lam in args.lambdas:
            for n in args.ns:
                cells = [(f"eq{n}", [f"agent.budget={n}"])]
                if n > args.short:
                    cells.append((f"mix{n}v{args.short}",
                                  [f"agent.budget_long={n}", f"agent.budget_short={args.short}"]))
                for cell, sets in cells:
                    exp = f"llm_haiku_sessions_entrench_lam{lam}_{cell}"
                    if done(exp, r):
                        continue
                    jobs.append((exp, r, [f"env.lambda_xz={lam}", f"first_run={r}", "n_runs=1",
                                          f"experiment={exp}", *sets],
                                 ROOT / "results" / "logs" / f"entrench_{cell}_lam{lam}_r{r}.log"))
    print(f"{len(jobs)} jobs, {args.jobs} at a time")
    if args.dry_run:
        for j in jobs[:5]:
            print(j[0], j[1], " ".join(j[2]))
        return
    (ROOT / "results" / "logs").mkdir(parents=True, exist_ok=True)

    def run(job):
        exp, r, sets, logf = job
        t0 = time.time()
        with open(logf, "w") as f:
            code = subprocess.call([PY, "scripts/run_sessions.py", CONFIG, "--set", *sets],
                                   cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        print(f"{time.strftime('%H:%M:%S')} {exp} pair {r}: exit {code}, "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)

    with ThreadPoolExecutor(args.jobs) as ex:
        list(ex.map(run, jobs))
    print("all jobs finished", flush=True)


if __name__ == "__main__":
    main()
