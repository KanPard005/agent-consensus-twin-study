"""Compare café-task session twins (`scripts/run_hp_sessions.py`) with the fixed-round twins and
the full-information reference (`scripts/run_hp.py`) on the same cases.

    python scripts/hp_sessions_analysis.py results/hp_haiku_sessions results/hp_haiku_clean \
        [--label H1 --solo F1 --name S1]

Takes the latest run in each folder, checks that the case sets match run by run, and prints
markdown tables:

1. accuracy on hidden cases (team, per run, log score, opening / round-1 accuracy, final
   agreement, deference), control accuracy, and team accuracy in cases 1-10 vs 11-20;
2. talk on hidden cases: messages per case and their length, and the mention rates of a twin's
   own unshared, key unshared and shared clues in its messages (as in `scripts/hp_analysis.py`:
   keyword matches, so lower bounds; "key" = positive clues on the best site, or negative clues
   on the twin's own first-read favourite when that is not the best site);
3. paired hidden cases: how often only one protocol's team is right;
4. for the session twins: messages per case over the run (does talk fade, as the message
   channel did in the numeric sessions?) and tool calls per case.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from hp_analysis import latest_run, mentioned


def load_sessions(run_dir: Path) -> dict[int, list[dict]]:
    """{run: [case record, ...]} of a session run, in case order."""
    out = {}
    for f in sorted((run_dir / "transcripts").glob("run*.jsonl")):
        if "_" in f.stem:  # run<r>_calls_A etc.
            continue
        rows = [json.loads(line) for line in f.open()]
        out[int(f.stem[3:])] = sorted(rows, key=lambda r: r["t"])
    return out


def load_fixed(run_dir: Path, label: str) -> dict[int, list[dict]]:
    out = {}
    for f in sorted((run_dir / "transcripts").glob(f"{label}_run*.jsonl")):
        if "_conversation_" in f.name:
            continue
        out[int(f.stem.split("_run")[1])] = [json.loads(line) for line in f.open()]
    return out


def same_cases(a: dict, b: dict) -> bool:
    keys = ("best", "clues")
    return all(a[k] == b[k] for k in keys)


def mention_rates(items: list[tuple[dict, str, str, int]]) -> dict[str, float]:
    """items = (case dict, twin, everything the twin said, its first-read favourite)."""
    hit = {"unshared": [], "key": [], "shared": []}
    for case, twin, said, fav in items:
        best = case["best"]
        for c in case["clues"]:
            m = mentioned(c["bank_id"], said)
            if c["seen_by"] == "both":
                hit["shared"].append(m)
            elif c["seen_by"] == twin:
                hit["unshared"].append(m)
                if (c["site"] == best and c["value"] > 0) or (
                    c["site"] == fav != best and c["value"] < 0
                ):
                    hit["key"].append(m)
    return {k: float(np.mean(v)) if v else float("nan") for k, v in hit.items()}


def session_measures(rows: list[dict]) -> dict:
    """Per-case measures of the session twins (one run)."""
    out = {k: [] for k in ("correct", "logscore", "open_correct", "agree", "defer", "kind",
                           "messages", "chars", "said")}
    for r in rows:
        case, best = r["case"], r["case"]["best"]
        op, fi, team = r["opening"], r["final"], r["team"]
        out["kind"].append(case["kind"])
        out["correct"].append(float(r["team_pick"] == best))
        out["logscore"].append(float(np.log(max(team[best], 0.01))))
        fav = {k: int(np.argmax(op[k])) for k in "AB"}
        out["open_correct"].append(np.mean([fav[k] == best for k in "AB"]))
        out["agree"].append(float(np.argmax(fi["A"]) == np.argmax(fi["B"])))
        d = [fi[k][fav[o]] - op[k][fav[o]] for k, o in (("A", "B"), ("B", "A"))
             if fav[k] != fav[o]]
        out["defer"].append(d)
        out["messages"].append(len(r["messages"]))
        out["chars"].append(np.mean([len(m["text"]) for m in r["messages"]])
                            if r["messages"] else np.nan)
        out["said"].append([(case, k, " ".join(m["text"] for m in r["messages"]
                                               if m["from"] == k).lower(), fav[k]) for k in "AB"])
    return out


def fixed_measures(rows: list[dict]) -> dict:
    out = {k: [] for k in ("correct", "logscore", "open_correct", "agree", "defer", "kind",
                           "messages", "chars", "said")}
    for s in rows:
        case, best = s["case"], s["case"]["best"]
        rounds = s["rounds"]
        first = {k: next(r[k]["probs"] for r in rounds if r[k]["probs"] is not None) for k in "AB"}
        last = {k: next(r[k]["probs"] for r in reversed(rounds) if r[k]["probs"] is not None)
                for k in "AB"}
        fav = {k: int(np.argmax(first[k])) for k in "AB"}
        out["kind"].append(case["kind"])
        out["correct"].append(float(s["team_pick"] == best))
        out["logscore"].append(float(np.log(max(s["team"][best], 0.01))))
        out["open_correct"].append(np.mean([fav[k] == best for k in "AB"]))
        out["agree"].append(float(np.argmax(last["A"]) == np.argmax(last["B"])))
        out["defer"].append([last[k][fav[o]] - first[k][fav[o]]
                             for k, o in (("A", "B"), ("B", "A")) if fav[k] != fav[o]])
        msgs = [r[k]["message"] for r in rounds for k in "AB" if r[k]["message"]]
        out["messages"].append(len(msgs))
        out["chars"].append(np.mean([len(m) for m in msgs]) if msgs else np.nan)
        out["said"].append([(case, k, " ".join(r[k]["message"] for r in rounds).lower(), fav[k])
                            for k in "AB"])
    return out


def summarise(per_run: dict[int, dict]) -> dict:
    """Pooled summary over runs; accuracy etc. on hidden cases unless stated."""
    hid_acc_runs, pooled = [], {k: [] for k in ("correct", "logscore", "open_correct", "agree",
                                                 "messages", "chars")}
    defer, ctrl, early, late, said = [], [], [], [], []
    for r, m in sorted(per_run.items()):
        kinds = np.array(m["kind"])
        hid = kinds == "hidden"
        c = np.array(m["correct"])
        hid_acc_runs.append(c[hid].mean())
        for k in pooled:
            pooled[k] += list(np.array(m[k], float)[hid])
        defer += [d for dd, h in zip(m["defer"], hid) if h for d in dd]
        ctrl += list(c[~hid])
        half = len(c) // 2
        early += list(c[:half][hid[:half]])
        late += list(c[half:][hid[half:]])
        said += [x for xs, h in zip(m["said"], hid) if h for x in xs]
    return {
        "acc": float(np.mean(pooled["correct"])), "acc_runs": hid_acc_runs,
        "logscore": float(np.mean(pooled["logscore"])),
        "open": float(np.mean(pooled["open_correct"])), "agree": float(np.mean(pooled["agree"])),
        "defer": float(np.mean(defer)) if defer else np.nan, "n_defer": len(defer),
        "control": float(np.mean(ctrl)) if ctrl else np.nan,
        "early": float(np.mean(early)), "late": float(np.mean(late)),
        "messages": float(np.mean(pooled["messages"])), "chars": float(np.nanmean(pooled["chars"])),
        "mentions": mention_rates(said),
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sessions")
    p.add_argument("fixed")
    p.add_argument("--label", default="H1")
    p.add_argument("--solo", default="F1")
    p.add_argument("--name", default="S1")
    args = p.parse_args(argv)
    s_dir, f_dir = latest_run(Path(args.sessions)), latest_run(Path(args.fixed))
    S, H = load_sessions(s_dir), load_fixed(f_dir, args.label)
    F = load_fixed(f_dir, args.solo)
    runs = sorted(set(S) & set(H))
    for r in runs:
        if len(S[r]) != len(H[r]) or not all(same_cases(a["case"], b["case"])
                                              for a, b in zip(S[r], H[r])):
            raise SystemExit(f"run {r}: the session and fixed-round case sets differ")
    sm = {r: session_measures(S[r]) for r in runs}
    hm = {r: fixed_measures(H[r]) for r in runs}
    ss, hs = summarise(sm), summarise(hm)
    f_acc = [np.mean([x["probs"].index(max(x["probs"])) == x["case"]["best"]
                      for x in F[r] if x["case"]["kind"] == "hidden"]) for r in runs]
    summ = json.loads((s_dir / "summary.json").read_text())
    s_cost = sum(v["cost_usd"] for v in summ.values())
    f_summ = json.loads((f_dir / "summary.json").read_text())
    h_cost = f_summ.get(args.label, {}).get("cost_usd_total", np.nan)
    f_cost = f_summ.get(args.solo, {}).get("cost_usd_total", np.nan)

    print(f"Sessions: {s_dir}\nFixed rounds / F1: {f_dir}\nRuns {runs}; same cases checked.\n")
    print("| | hidden acc. | per run | log score | opening acc. | final agreement | deference (n) "
          "| control acc. | cases 1–10 → 11–20 | cost |")
    print("|---" * 10 + "|")
    for name, s, cost in ((args.name, ss, s_cost), (args.label, hs, h_cost)):
        print(f"| {name} | **{s['acc']:.2f}** | {', '.join(f'{a:.2f}' for a in s['acc_runs'])} | "
              f"{s['logscore']:.2f} | {s['open']:.2f} | {s['agree']:.2f} | {s['defer']:+.2f} "
              f"({s['n_defer']}) | {s['control']:.2f} | {s['early']:.2f} → {s['late']:.2f} | "
              f"${cost:.2f} |")
    print(f"| {args.solo} all clues | **{np.mean(f_acc):.2f}** | "
          f"{', '.join(f'{a:.2f}' for a in f_acc)} | | | | | | | ${f_cost:.2f} |")
    print("\nTalk on hidden cases:\n")
    print("| | messages / case | chars / message | own unshared mentioned | key unshared "
          "mentioned | shared mentioned |")
    print("|---" * 6 + "|")
    for name, s in ((args.name, ss), (args.label, hs)):
        mr = s["mentions"]
        print(f"| {name} | {s['messages']:.1f} | {s['chars']:.0f} | {mr['unshared']:.0%} | "
              f"{mr['key']:.0%} | {mr['shared']:.0%} |")
    # paired hidden cases
    only_s = only_h = both = neither = 0
    for r in runs:
        for a, b, k in zip(sm[r]["correct"], hm[r]["correct"], sm[r]["kind"]):
            if k != "hidden":
                continue
            both += a and b
            neither += not a and not b
            only_s += a and not b
            only_h += b and not a
    print(f"\nPaired hidden cases ({both + neither + only_s + only_h}): both right {both}, "
          f"only {args.name} {only_s}, only {args.label} {only_h}, neither {neither}.")
    # talk over the run and tool use (session twins)
    print(f"\n{args.name}: messages per case by case index (mean over runs):")
    T = max(len(S[r]) for r in runs)
    msgs = np.array([[len(S[r][t]["messages"]) for t in range(T)] for r in runs], float)
    print("  " + " ".join(f"{v:.1f}" for v in msgs.mean(0)))
    calls = [sum(S[r][t]["tool_calls"].values()) / 2 for r in runs for t in range(T)]
    print(f"{args.name}: tool calls per twin per case: mean {np.mean(calls):.1f}, "
          f"max {np.max(calls):.0f}")
    errs = {r: v["errors"] for r, v in summ.items()}
    cont = {r: v["continues"] for r, v in summ.items()}
    print(f"{args.name}: errors {errs}; continues {cont}")
    print(f"{args.name}: config commit {yaml.safe_load((s_dir / 'config.yaml').read_text())['_git_commit']}")


if __name__ == "__main__":
    main()
