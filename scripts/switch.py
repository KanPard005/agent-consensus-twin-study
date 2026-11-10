"""Partner switch (SwitchExchange): what changes when the B twins change pairs, unannounced?

Usage:
    python scripts/switch.py results/llm_haiku_switch results/llm_haiku_switch_r3 [--label SW]
        [--min-gap 0.2] [--windows 5 10]

Takes the latest run in each folder. For every replicate the switched streams (s1 = A1 + B2,
s2 = A2 + B1) and the control (p1, p2, unchanged pairs) share all steps before the switch, so
after it the difference s - ctrl is paired run by run. Per step window it reports, averaged over
the two streams of each kind and then over replicates (mean, and the paired difference with its
standard error and sign count):

- team excess: squared error of the team estimate minus sigma^2;
- WOA: weight of advice, (round 2 - own round 1) / (partner round 1 - own round 1), where the
  round-1 gap exceeds `min-gap`, clipped to [-0.5, 1.5] (0 = ignore the partner);
- disagreement: squared gap between the twins' final estimates;
- shared words: Jaccard overlap of the two twins' word sets in a step's round-1 messages (numbers
  and one-letter words dropped), a crude measure of a shared vocabulary or convention;
- notices: share of messages that remark on a change in the partner (e.g. "new partner",
  "you've changed", "different from before").

Windows: the `windows[0]` steps before the switch (control only; identical in both branches up to
the LLM's randomness), then the first `windows[0]` steps after it, then the next `windows[1]`.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

WORD = re.compile(r"[a-z][a-z0-9_-]+")
NOTICE = re.compile(
    r"new partner|partner (has )?changed|you('ve| have) changed|different (from|than) before|"
    r"not the same partner|unlike (before|earlier)|switched|someone else",
    re.IGNORECASE,
)


def latest_run(path: Path) -> Path:
    """The folder itself if it is a run, else its latest timestamped run."""
    if (path / "config.yaml").exists():
        return path
    return sorted(p for p in path.iterdir() if (p / "config.yaml").exists())[-1]


def words(msg: str) -> set[str]:
    return set(WORD.findall(msg.lower()))


def step_stats(s: dict, sigma2: float, min_gap: float) -> dict:
    """Statistics of one transcript step (one stream)."""
    r1, r2 = s["rounds"][0], s["rounds"][-1]
    woa = []
    for me, other in (("A", "B"), ("B", "A")):
        own, par, fin = r1[me]["estimate"], r1[other]["estimate"], r2[me]["estimate"]
        if abs(par - own) > min_gap:
            woa.append(float(np.clip((fin - own) / (par - own), -0.5, 1.5)))
    wa, wb = words(r1["A"]["message"]), words(r1["B"]["message"])
    msgs = [r[k]["message"] for r in s["rounds"] for k in "AB"]
    return {
        "team": (s["consensus"] - s["y"]) ** 2 - sigma2,
        "woa": woa,
        "disagreement": (r2["A"]["estimate"] - r2["B"]["estimate"]) ** 2,
        "shared": len(wa & wb) / max(len(wa | wb), 1),
        "notice": [bool(NOTICE.search(m)) for m in msgs],
    }


def window_mean(steps: list[dict], key: str) -> float:
    vals = []
    for st in steps:
        v = st[key]
        vals += v if isinstance(v, list) else [v]
    return float(np.mean(vals)) if vals else float("nan")


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("folders", nargs="+")
    p.add_argument("--label", default="SW")
    p.add_argument("--min-gap", type=float, default=0.2)
    p.add_argument("--windows", type=int, nargs=2, default=[5, 10])
    args = p.parse_args(argv)

    # per[(replicate)][stream][t] = step stats
    per: dict[int, dict[str, dict[int, dict]]] = defaultdict(lambda: defaultdict(dict))
    switch_at = None
    for folder in args.folders:
        run_dir = latest_run(Path(folder))
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
        pc = next(c for c in cfg["protocols"] if c.get("label") == args.label)
        switch_at = pc["switch_at"]
        sigma2 = cfg["env"]["sigma"] ** 2
        for f in sorted((run_dir / "transcripts").glob(f"{args.label}_run*.jsonl")):
            m = re.fullmatch(rf"{re.escape(args.label)}_run(\d+)\.jsonl", f.name)
            if not m:
                continue
            for line in f.open():
                s = json.loads(line)
                per[int(m.group(1))][s["stream"]][s["t"]] = step_stats(s, sigma2, args.min_gap)

    w0, w1 = args.windows
    windows = {
        f"steps {switch_at - w0 + 1}-{switch_at} (before)": range(switch_at - w0, switch_at),
        f"steps {switch_at + 1}-{switch_at + w0}": range(switch_at, switch_at + w0),
        f"steps {switch_at + w0 + 1}-{switch_at + w0 + w1}": range(
            switch_at + w0, switch_at + w0 + w1
        ),
    }
    keys = ["team", "woa", "disagreement", "shared", "notice"]
    print(f"{len(per)} replicates; switch at step {switch_at + 1}\n")
    head = "| window | measure | control | switched | switched − control (paired, s > c) |"
    print(head)
    print("|---" * (head.count("|") - 1) + "|")
    out = {}
    for wname, rng in windows.items():
        before = rng.start < switch_at
        for key in keys:
            ctrl, sw = [], []
            for streams in per.values():
                ctrl.append(
                    np.nanmean(
                        [
                            window_mean([streams[n][t] for t in rng if t in streams[n]], key)
                            for n in ("p1", "p2")
                        ]
                    )
                )
                if not before:
                    sw.append(
                        np.nanmean(
                            [
                                window_mean([streams[n][t] for t in rng if t in streams[n]], key)
                                for n in ("s1", "s2")
                            ]
                        )
                    )
            c = np.array(ctrl)
            if before:
                print(f"| {wname} | {key} | {c.mean():.3f} | – | – |")
                out[(wname, key)] = {"ctrl": c}
                continue
            s_ = np.array(sw)
            d = s_ - c
            se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
            print(
                f"| {wname} | {key} | {c.mean():.3f} | {s_.mean():.3f} | "
                f"{d.mean():+.3f} ± {se:.3f} ({(d > 0).sum()}/{len(d)}) |"
            )
            out[(wname, key)] = {"ctrl": c, "sw": s_}
    return out


if __name__ == "__main__":
    main()
