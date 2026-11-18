"""Launch boolean-task pairs (LogBook note 2026-11-18) as a job queue, one pair per process.

Usage:
    python scripts/launch_boolean.py [--pairs 10] [--first 0] [--jobs 10]
        [--config configs/boolean_haiku.yaml] [--experiment boolean_haiku]
        [--set n_train=80 K=40 ...] [--dry-run]

Each job runs `scripts/run_boolean.py <config>` for one pair (first_run = pair index) in the
experiment folder; pairs already finished are skipped (see launch_grid.done).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from launch_grid import done  # noqa: E402

PY = sys.executable


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/boolean_haiku.yaml")
    p.add_argument("--experiment", default="boolean_haiku")
    p.add_argument("--pairs", type=int, default=10)
    p.add_argument("--first", type=int, default=0)
    p.add_argument("--jobs", type=int, default=10)
    p.add_argument("--set", nargs="*", default=[], dest="extra")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    exp = args.experiment
    jobs = [r for r in range(args.first, args.first + args.pairs) if not done(exp, r)]
    print(f"{len(jobs)} jobs, {args.jobs} at a time")
    if args.dry_run:
        print(jobs)
        return
    (ROOT / "results" / "logs").mkdir(parents=True, exist_ok=True)

    def run(r: int) -> None:
        sets = [f"first_run={r}", "n_runs=1", f"experiment={exp}", *args.extra]
        t0 = time.time()
        with open(ROOT / "results" / "logs" / f"{exp}_r{r}.log", "w") as f:
            code = subprocess.call([PY, "scripts/run_boolean.py", args.config, "--set", *sets],
                                   cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        print(f"{time.strftime('%H:%M:%S')} {exp} pair {r}: exit {code}, "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)

    with ThreadPoolExecutor(args.jobs) as ex:
        list(ex.map(run, jobs))
    print("all jobs finished", flush=True)


if __name__ == "__main__":
    main()
