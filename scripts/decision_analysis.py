"""What do live-session twins decide to do with their partner's information? (LogBook 2026-11-13)

Reads every query of the live-session grids (consensus `entrench`, averaging `avg`, individual
scoring `own`, plus `samey` and `samey2`: individual scoring with the same-y sentence and model
tools) and classifies each twin's own messages in each query with regular expressions:

  shares_features  states its own raw feature values (a1=0.42 ...)
  asks_features    asks for the partner's features / values
  shares_model     states its model or coefficients (coef, intercept, 0.7*a1 ...)
  quotes_partner   writes the partner's feature values or uses them in arithmetic (b1=..., 0.3*b2)
  additive         speaks of adding / summing / contributions
  averaging        speaks of the midpoint / average / splitting the difference
  track_record     refers to past accuracy (closer to the truth, was better, so far ...)
  dismisses        gives a reason not to move (different variables, can't verify, keep mine ...)
  defers           says it will go with the partner's number
  cites_experience mentions its training size or data (400 cases, full-data, trained on ...)
  silent           sent no message in that query

and relates them to the twin's weight of advice (WOA) on its own final, the error of its own final,
the team rule, training length, lambda, role (equal / long / short twin) and time in the run.

Usage:
    python scripts/decision_analysis.py [--sample 0]
--sample N prints N random messages per category (for checking precision by hand).
Writes results/decisions/<timestamp>/{stats.json, table.md, decisions.png}.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from consensus.plotting_overlap import plot_decisions  # noqa: E402
from consensus.sessions import weight_of_advice  # noqa: E402
from consensus.utils import make_run_dir  # noqa: E402

GRIDS = {  # prefix suffix -> (team rule, extra prompt)
    "entrench": ("consensus", ""),
    "avg": ("average", ""),
    "own": ("individual", ""),
    "samey": ("individual", "same y"),
    "samey2": ("individual", "same y + models"),
}
CATS = ("shares_features", "asks_features", "shares_model", "quotes_partner", "additive",
        "averaging", "track_record", "different_info", "dismisses", "defers", "cites_experience",
        "silent")


def patterns(own: str, other: str) -> dict[str, re.Pattern]:
    """Regexes for a twin whose features are <own>1.. and whose partner's are <other>1.."""
    I = re.I
    return {
        "shares_features": re.compile(rf"\b{own}[1-9]\d?\s*=\s*-?\d", I),
        "asks_features": re.compile(
            rf"((what|which) (are|were|is) your|(send|share|tell me|give me|post|list)( me)? your|"
            rf"(can|could|would) you (send|share|give|tell|post|list))[^.?!]{{0,60}}"
            rf"(\b{other}[1-9]|\b{other}-?(features|values)|feature|value|input)", I),
        "shares_model": re.compile(
            rf"(coef|intercept|\d\.\d+\s*\*\s*{own}[1-9]|\by\s*(≈|~|=)\s*-?\d?\.?\d*\s*[+-]?\s*\d*\.?\d*\s*\*?\s*{own}[1-9])",
            I),
        "quotes_partner": re.compile(
            rf"(\b{other}[1-9]\d?\s*=\s*-?\d|\d\.\d+\s*\*\s*\(?\s*{other}[1-9]|\b{other}[1-9]\s*\*\s*-?\d)",
            I),
        "additive": re.compile(
            r"(\bsum\b|summing|additive|non-intercept|plus your|add (them|our|your|my|the two|"
            r"it to|both))", I),
        "averaging": re.compile(
            r"(midpoint|averag|split the difference|meet in the middle|halfway|middle ground)", I),
        "track_record": re.compile(
            r"(closer to (the )?(truth|true)|track record|(beat|outperform)\w* me|(was|were|been) "
            r"(much |a bit |slightly |more |less )?(better|closer|more accurate|right|off by|wrong|"
            r"noisy)|more accurate (than|so far|on)|(better|worse) (last|on query|so far|in (the )?"
            r"past)|off by)", I),
        # Noting that the partner's features are different (a description, not a decision).
        "different_info": re.compile(
            r"(different (variables|columns|features|inputs|feature sets)|features (differ|are "
            r"different)|can'?t (verify|check|see) (your|those|that)|cannot (verify|check|see) "
            r"(your|those|that)|i don'?t (have|see) (your|those|that))", I),
        # An explicit decision not to move toward the partner.
        "dismisses": re.compile(
            r"(no (reason|basis) to (move|change|adjust|shift)|stick(ing)? (with|to) (my|mine)|"
            r"keep(ing)? (my|mine)|i'?ll keep|won'?t (move|change|budge)|not (moving|adjusting|"
            r"changing) (my|from)|rely on my own|hold(ing)? (at|my|firm)|staying (at|with) my|"
            r"i'?ll trust my|trust my (own )?(model|estimate|number))", I),
        # An explicit decision to go (mostly) with the partner's number.
        "defers": re.compile(
            r"(i'?(ll| will) (go with|take|use|adopt|defer to|follow) your|defer to (you|your)|"
            r"lean(ing)? (on|toward|towards) your|weight your (estimate|prediction|number|model) "
            r"(more|heavily)|i'?ll (submit|go with) your)", I),
        "cites_experience": re.compile(
            r"(\b\d{2,4} (training |labelled |labeled )?(cases|samples|examples|points)\b|trained "
            r"on|full[- ]data|more data|larger (sample|dataset))", I),
    }


def iter_rows(prefix: str):
    """Rows: one per twin and query, with flags and outcomes."""
    base = ROOT / "results"
    for gk, (rule, extra) in GRIDS.items():
        for cdir in sorted(base.glob(f"{prefix}_{gk}_lam*")):
            name = cdir.name[len(f"{prefix}_{gk}_"):]  # lam0.0_eq400
            if not name.startswith("lam"):
                continue
            lam, cell = name[3:].split("_", 1)
            kind = "equal" if cell.startswith("eq") else "mixed"
            n = int(re.match(r"(?:eq|mix)(\d+)", cell).group(1))
            done = {}
            for d in sorted(cdir.glob("20*")):
                sf = d / "summary.json"
                if not sf.exists():
                    continue
                for key, summ in json.loads(sf.read_text()).items():
                    done[int(key[3:])] = (d, summ)
            for r, (d, summ) in sorted(done.items()):
                L = summ.get("long_twin")
                recs = [json.loads(x) for x in (d / "transcripts" / f"run{r}.jsonl")
                        .read_text().splitlines() if x.strip()]
                for s in recs:
                    for k, o in (("A", "B"), ("B", "A")):
                        own_pre = "a" if k == "A" else "b"
                        pats = patterns(own_pre, "b" if own_pre == "a" else "a")
                        mine = [m["text"] for m in s["messages"] if m["from"] == k]
                        theirs = [m["text"] for m in s["messages"] if m["from"] == o]
                        text = "\n".join(mine)
                        flags = {c: bool(p.search(text)) for c, p in pats.items()}
                        flags["silent"] = not mine
                        fin = s["attempts"][k][-1] if s["attempts"][k] else float("nan")
                        value = s["agreed"] if rule == "consensus" else fin
                        role = "equal" if kind == "equal" else ("long" if k == L else "short")
                        partner_shared = bool(
                            patterns("b" if own_pre == "a" else "a", own_pre)["shares_features"]
                            .search("\n".join(theirs)))
                        yield dict(
                            grid=gk, rule=rule, extra=extra, lam=lam, kind=kind, n=n, role=role,
                            pair=r, twin=k, t=s["t"], text=text,
                            woa=weight_of_advice(s["opening"][k], s["opening"][o], value),
                            err_final=(fin - s["y"]) ** 2 if fin == fin else float("nan"),
                            err_open=(s["opening"][k] - s["y"]) ** 2,
                            partner_shared=partner_shared, **flags)


def m(v) -> float:
    v = [x for x in v if x == x]
    return float(np.mean(v)) if v else float("nan")


def main(argv: list[str] | None = None) -> Path:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prefix", default="llm_haiku_sessions")
    p.add_argument("--sample", type=int, default=0)
    args = p.parse_args(argv)
    rows = list(iter_rows(args.prefix))
    if args.sample:
        random.seed(0)
        for c in CATS[:-1]:
            hits = [r for r in rows if r[c]]
            print(f"\n=== {c} ({len(hits)} of {len(rows)}) ===")
            for r in random.sample(hits, min(args.sample, len(hits))):
                print(" -", r["text"].replace("\n", " | ")[:260])
        return ROOT

    out: dict = {"n_rows": len(rows)}
    lines = [f"{len(rows)} twin-queries from {len({(r['grid'], r['lam'], r['kind'], r['n'], r['pair']) for r in rows})} pairs.", ""]

    # 1. prevalence by grid
    lines += ["Share of twin-queries with each behaviour, by grid:", "",
              "| grid (rule) | " + " | ".join(CATS) + " |", "|" + "---|" * (len(CATS) + 1)]
    out["by_grid"] = {}
    for g in GRIDS:
        rs = [r for r in rows if r["grid"] == g]
        if not rs:
            continue
        out["by_grid"][g] = {c: m([r[c] for r in rs]) for c in CATS}
        label = f"{g} ({GRIDS[g][0]}{', ' + GRIDS[g][1] if GRIDS[g][1] else ''})"
        lines.append(f"| {label} | " + " | ".join(f"{out['by_grid'][g][c]:.0%}" for c in CATS) + " |")

    # 2. equal pairs by training length (pooled over grids with individual / average rules)
    key_cats = ("shares_features", "asks_features", "quotes_partner", "additive", "averaging",
                "track_record", "dismisses", "defers", "cites_experience")
    lines += ["", "Equal pairs, by cases studied (all grids pooled):", "",
              "| n | " + " | ".join(key_cats) + " | WOA |", "|" + "---|" * (len(key_cats) + 2)]
    out["equal_by_n"] = {}
    for n in sorted({r["n"] for r in rows if r["kind"] == "equal"}):
        rs = [r for r in rows if r["kind"] == "equal" and r["n"] == n and r["rule"] != "consensus"]
        out["equal_by_n"][n] = {c: m([r[c] for r in rs]) for c in key_cats}
        lines.append(f"| {n} | " + " | ".join(f"{out['equal_by_n'][n][c]:.0%}" for c in key_cats)
                     + f" | {m([r['woa'] for r in rs]):.2f} |")

    # 3. mixed pairs: long vs short twin, per rule
    lines += ["", "Mixed pairs: long-trained twin vs its partner (25 cases), per rule:", "",
              "| rule | role | " + " | ".join(key_cats) + " | WOA |", "|" + "---|" * (len(key_cats) + 3)]
    out["mixed_roles"] = {}
    for rule in ("consensus", "average", "individual"):
        for role in ("long", "short"):
            rs = [r for r in rows if r["kind"] == "mixed" and r["rule"] == rule and r["role"] == role]
            if not rs:
                continue
            out["mixed_roles"][f"{rule}_{role}"] = {c: m([r[c] for r in rs]) for c in key_cats}
            lines.append(f"| {rule} | {role} | " + " | ".join(
                f"{out['mixed_roles'][f'{rule}_{role}'][c]:.0%}" for c in key_cats)
                + f" | {m([r['woa'] for r in rs]):.2f} |")

    # 4. what goes with deference and accuracy (individual scoring, where WOA is free)
    lines += ["", "Individual scoring (grids own, samey, samey2): WOA and the gain from talking "
              "(error of own opening minus error of own final) when a behaviour is present vs absent:",
              "", "| behaviour | share | WOA with / without | gain with / without |", "|---|---|---|---|"]
    ind = [r for r in rows if r["rule"] == "individual"]
    out["individual_assoc"] = {}
    for c in CATS:
        w1 = [r for r in ind if r[c]]
        w0 = [r for r in ind if not r[c]]
        g1 = m([r["err_open"] - r["err_final"] for r in w1])
        g0 = m([r["err_open"] - r["err_final"] for r in w0])
        out["individual_assoc"][c] = dict(share=len(w1) / len(ind), woa1=m([r["woa"] for r in w1]),
                                          woa0=m([r["woa"] for r in w0]), gain1=g1, gain0=g0)
        a = out["individual_assoc"][c]
        lines.append(f"| {c} | {a['share']:.0%} | {a['woa1']:.2f} / {a['woa0']:.2f} "
                     f"| {g1:+.2f} / {g0:+.2f} |")

    # 5. when the partner shared its raw features: did the twin use them?
    lines += ["", "When the partner wrote its raw feature values in that query (individual scoring):", ""]
    sh = [r for r in ind if r["partner_shared"]]
    nsh = [r for r in ind if not r["partner_shared"]]
    out["partner_shared"] = dict(share=len(sh) / len(ind), quotes=m([r["quotes_partner"] for r in sh]),
                                 additive=m([r["additive"] for r in sh]), woa_sh=m([r["woa"] for r in sh]),
                                 woa_nsh=m([r["woa"] for r in nsh]))
    ps = out["partner_shared"]
    lines.append(f"- the partner shared in {ps['share']:.0%} of twin-queries; then the twin wrote its "
                 f"partner's values or computed with them in {ps['quotes']:.0%}, used additive "
                 f"language in {ps['additive']:.0%}; WOA {ps['woa_sh']:.2f} when the partner shared "
                 f"vs {ps['woa_nsh']:.2f} when it did not")

    # 6. over the run
    lines += ["", "Over the run (individual scoring), share by queries 1–10 / 11–20 / 21–30:", ""]
    out["over_time"] = {}
    for c in ("shares_features", "asks_features", "track_record", "dismisses", "silent"):
        v = [m([r[c] for r in ind if a <= r["t"] < b]) for a, b in ((0, 10), (10, 20), (20, 30))]
        out["over_time"][c] = v
        lines.append(f"- {c}: " + " / ".join(f"{x:.0%}" for x in v))

    run_out = make_run_dir(str(ROOT / "results"), "decisions")
    (run_out / "stats.json").write_text(json.dumps(out, indent=2, default=str))
    table = "\n".join(lines)
    (run_out / "table.md").write_text(table + "\n")
    plot_decisions(out, run_out / "decisions.png")
    print(table)
    print(f"saved to {run_out}")
    return run_out


if __name__ == "__main__":
    main()
