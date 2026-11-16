"""Launch the alignment sweep (LogBook note 2026-11-16) as a job queue, one pair per process.

Usage:
    python scripts/launch_related.py [--cs 1.0 0.5 0.0 -0.5] [--arms untold told] [--pairs 8]
        [--jobs 16] [--prefix ...] [--set agent.model=... agent.effort=...] [--dry-run]

Each job runs `scripts/run_sessions.py configs/llm_haiku_sessions_related.yaml` for one pair
(first_run = pair index) with env.target_align = c and agent.partner_target = arm, in the
experiment folder <prefix>_c<c>_<arm>. Pairs already finished are skipped (see launch_grid.done).
"""

from __future__ import annotations

import argparse
import itertools
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
    p.add_argument("--config", default="configs/llm_haiku_sessions_related.yaml")
    p.add_argument("--prefix", default="llm_haiku_sessions_related")
    p.add_argument("--cs", nargs="+", default=["1.0", "0.5", "0.0", "-0.5"])
    p.add_argument("--arms", nargs="+", default=["untold", "told"])
    p.add_argument("--pairs", type=int, default=8)
    p.add_argument("--jobs", type=int, default=16)
    p.add_argument("--set", nargs="*", default=[], dest="extra",
                   help="extra config overrides for every job, e.g. agent.model=claude-sonnet-5-5")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    jobs = []
    # Pair index outermost, so early results cover every cell.
    for r, c, arm in itertools.product(range(args.pairs), args.cs, args.arms):
        exp = f"{args.prefix}_c{c}_{arm}"
        if done(exp, r):
            continue
        log = ROOT / "results" / "logs" / f"{exp}_r{r}.log"
        jobs.append((exp, r, [f"env.target_align={c}", f"agent.partner_target={arm}",
                              f"first_run={r}", "n_runs=1", f"experiment={exp}", *args.extra], log))
    print(f"{len(jobs)} jobs, {args.jobs} at a time")
    if args.dry_run:
        for j in jobs[:8]:
            print(j[0], j[1], " ".join(j[2]))
        return
    (ROOT / "results" / "logs").mkdir(parents=True, exist_ok=True)

    def run(job):
        exp, r, sets, logf = job
        t0 = time.time()
        with open(logf, "w") as f:
            code = subprocess.call([PY, "scripts/run_sessions.py", args.config, "--set", *sets],
                                   cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        print(f"{time.strftime('%H:%M:%S')} {exp} pair {r}: exit {code}, "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)

    with ThreadPoolExecutor(args.jobs) as ex:
        list(ex.map(run, jobs))
    print("all jobs finished", flush=True)


if __name__ == "__main__":
    main()
