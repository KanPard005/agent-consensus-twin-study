"""Analyse hidden-profile runs (`scripts/run_hp.py`): accuracy, deference and which clues get said.

Usage:
    python scripts/hp_analysis.py results/hp_haiku_p0.25 results/hp_haiku_p0.5 ... [--label H1]

Takes the latest run in each folder and reports, per p_shared:

- accuracy of the rule references (own_A/own_B: a twin alone; add_views: adding the twins' own
  site scores), the twins' round-1 favourites, the team, and F1 (one model with every clue), for
  hidden and control cases; and team accuracy in the first and second half of the cases
  (does it improve with feedback?);
- agreement of the final favourites, and deference (change in a twin's probability on the
  partner's round-1 favourite, when the favourites differ);
- mention rates (the "biased sampling" statistic of hidden-profile studies): the share of a
  twin's unshared clues that it mentions in its messages, against the share of its shared clues,
  and the same for the "key" unshared clues (those on the best site, or negative clues on the
  twin's own round-1 favourite: the ones that overturn the misleading view). A clue counts as
  mentioned if any of its keywords (`consensus.hp_clues`) appears in any of the twin's messages
  in the case; this misses paraphrases, so the rates are lower bounds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from consensus.hp_clues import CLUE_BANK


def latest_run(path: Path) -> Path:
    if (path / "config.yaml").exists():
        return path
    return sorted(p for p in path.iterdir() if (p / "config.yaml").exists())[-1]


def mentioned(bank_id: int, text: str) -> bool:
    return any(k in text for k in CLUE_BANK[bank_id][2])


def mention_stats(rows: list[dict]) -> dict[str, float]:
    """Mention rates of a twin's own unshared, key unshared and shared clues, over both twins."""
    hit = {"unshared": [], "key": [], "shared": []}
    for s in rows:
        case, best = s["case"], s["case"]["best"]
        for twin in "AB":
            said = " ".join(r[twin]["message"] for r in s["rounds"]).lower()
            fav = int(np.argmax(s["rounds"][0][twin]["probs"]))
            for c in case["clues"]:
                m = mentioned(c["bank_id"], said)
                if c["seen_by"] == "both":
                    hit["shared"].append(m)
                elif c["seen_by"] == twin:
                    hit["unshared"].append(m)
                    is_key = (c["site"] == best and c["value"] > 0) or (
                        c["site"] == fav != best and c["value"] < 0
                    )
                    if is_key:
                        hit["key"].append(m)
    return {k: float(np.mean(v)) if v else float("nan") for k, v in hit.items()}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("folders", nargs="+")
    p.add_argument("--label", default="H1")
    p.add_argument("--solo", default="F1")
    args = p.parse_args(argv)

    acc_rows, talk_rows = [], []
    for folder in args.folders:
        run_dir = latest_run(Path(folder))
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
        ps = cfg["cases"]["p_shared"]
        z = np.load(run_dir / "metrics.npz")
        kinds = np.array(json.loads((run_dir / "kinds.json").read_text()))
        hid, con = kinds == "hidden", kinds == "control"
        L = args.label

        def acc(key: str, sel: np.ndarray) -> str:
            return f"{np.mean(z[key][sel]):.2f}" if key in z.files else "–"

        r1 = (z[f"{L}/r1_correct_A"] + z[f"{L}/r1_correct_B"]) / 2
        T = kinds.shape[1]
        first, second = np.zeros_like(hid), np.zeros_like(hid)
        first[:, : T // 2], second[:, T // 2 :] = True, True
        acc_rows.append(
            f"| {ps} | {acc('own_A/correct', hid)} / {acc('own_B/correct', hid)} | "
            f"{acc('add_views/correct', hid)} | {np.mean(r1[hid]):.2f} | "
            f"**{acc(f'{L}/correct', hid)}** | {acc(f'{args.solo}/correct', hid)} | "
            f"{np.mean(r1[con]):.2f} / {acc(f'{L}/correct', con)} / "
            f"{acc(f'{args.solo}/correct', con)} | "
            f"{acc(f'{L}/correct', hid & first)} → {acc(f'{L}/correct', hid & second)} |"
        )
        rows = []
        for f in sorted((run_dir / "transcripts").glob(f"{L}_run*.jsonl")):
            rows += [json.loads(line) for line in f.open()]
        ms = mention_stats([r for r in rows if r["case"]["kind"] == "hidden"])
        defer = np.concatenate([z[f"{L}/defer_A"][hid], z[f"{L}/defer_B"][hid]])
        talk_rows.append(
            f"| {ps} | {np.mean(z[f'{L}/agree'][hid]):.2f} | {np.nanmean(defer):+.2f} "
            f"({np.sum(~np.isnan(defer))}) | {ms['unshared']:.0%} | {ms['key']:.0%} | "
            f"{ms['shared']:.0%} | {np.nansum(z[f'{L}/parse_fail']):.0f} |"
        )
    print(
        "| p_shared | own_A / own_B | add views | twins round 1 | team | F1 (all clues) | "
        "control: round 1 / team / F1 | team, cases 1–10 → 11–20 |"
    )
    print("|---" * 8 + "|")
    print("\n".join(acc_rows))
    print("\nHidden cases:\n")
    print(
        "| p_shared | final agreement | deference (n) | unshared mentioned | key unshared "
        "mentioned | shared mentioned | unusable replies |"
    )
    print("|---" * 7 + "|")
    print("\n".join(talk_rows))


if __name__ == "__main__":
    main()
