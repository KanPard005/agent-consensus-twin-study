"""Briefing arms of the two-clause XOR (LogBook 2026-11-19): per pair, what the twins did in the
conversation before the partner queries, and how they did afterwards, against the pairs without
it (boolean_haiku_xor).

Usage:
    python scripts/xor_briefing_analysis.py [--baseline-root ../agent-consensus/results]

Per pair it reports the finals' accuracy and confidence on the partner queries (all 100 and the
last 20), the conversation (messages, characters, whether a twin sent a per-pattern rate table or
raw study rows, whether it proposed exchanging features per query or pooling a joint dataset),
how much the twins used the data API on partner features (add_case with values, count on those
datasets), how much they talked per query, and the first partner query at which a message states
an XOR (guess or rule); a pair counts as having found the rule if
its finals are right on at least 80% of the last 20 queries. Writes results/xor_briefing_summary.json.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# A message that mentions an XOR (as a guess or as the rule; "features are different" is common,
# so "differ" is not used). Whether the pair found the rule is judged from accuracy and by reading.
RULE = re.compile(r"\bxor\b|exactly one|\bparity|one but not (both|the other)", re.I)
TABLE = re.compile(r"\b[01]{3}\s*[:=≈~]")  # "000: 102/210", "011≈0.68"
ROWS = re.compile(r"(\b[01],\s*[01],\s*[01],\s*[01]\b.*){5,}", re.S)  # pasted study rows
EXCHANGE = re.compile(r"(share|send|exchange)\w*\s+(our |my |your |full )*(feature|triple|a1|b1)",
                      re.I)
# A message that gives the sender's own feature values for the current case ("a1=0", "b=(0,1,1)",
# "my a=010"), and one that asks what the other twin sees or which features it has.
VALUES = re.compile(r"\b[ab][123]\s*=\s*[01]\b|\b[ab]\s*=\s*\(?[01],?\s*[01],?\s*[01]\)?")
ASKS = re.compile(r"(what|which)\b[^.?]{0,40}\b(features?|variables?|inputs?|do you see|you see)|"
                  r"your (features|variables|inputs)\b[^.]{0,30}\?", re.I)
POOL = re.compile(r"\b(pool|joint|combined)\b.{0,40}\b(data|dataset|table|log|track)", re.I)


def pair_dirs(results: Path, exp: str) -> list[tuple[int, Path]]:
    """(run index, run folder) of every finished pair of an experiment."""
    out = {}
    for d in sorted((results / exp).glob("*")):
        s = d / "summary.json"
        if not s.exists():
            continue
        for key in json.loads(s.read_text()):
            r = int(key[3:])
            if (d / "transcripts" / f"run{r}.jsonl").stat().st_size:
                out[r] = d
    return sorted(out.items())


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()] \
        if p.exists() else []


def pair_stats(d: Path, r: int) -> dict:
    tr = d / "transcripts"
    recs = load_jsonl(tr / f"run{r}.jsonl")
    y = np.array([q["y"] for q in recs])
    fin = {k: np.array([q["final"][k] for q in recs]) for k in "AB"}
    # per query, mean over the twins that submitted a final (a twin whose session ended early
    # leaves NaN finals; they are skipped, not counted as wrong)
    ok = np.array([~np.isnan(fin[k][:, 1]) for k in "AB"])
    hit = np.array([fin[k][:, 1] == y for k in "AB"])
    acc = (hit & ok).sum(0) / ok.sum(0)
    conf = np.nanmean([np.maximum(fin[k][:, 0], 1 - fin[k][:, 0]) for k in "AB"], axis=0)
    rule_q = next((q["number"] for q in recs for m in q["messages"] if RULE.search(m["text"])),
                  None)
    brief_p = tr / f"run{r}_brief.json"
    brief = json.loads(brief_p.read_text()) if brief_p.exists() else []
    text = "\n".join(m["text"] for m in brief)
    calls = {k: load_jsonl(tr / f"run{r}_calls_{k}.jsonl") for k in "AB"}
    part = {k: [c for c in calls[k] if c["phase"] in ("query", "brief")] for k in "AB"}
    seen = {k: "\n".join(c["result"] for c in calls[k] if c["tool"] == "read_messages")
            for k in "AB"}
    with_vals = {k: {c["args"].get("name") for c in part[k] if c["tool"] == "add_case"
                     and c["args"].get("values")} for k in "AB"}
    return {
        "dir": str(d), "run": r, "n": len(recs),
        "missing_finals": {k: int((~ok[i]).sum()) for i, k in enumerate("AB")},
        "acc": float(acc.mean()), "acc_last20": float(acc[-20:].mean()),
        "conf": float(conf.mean()), "conf_last20": float(conf[-20:].mean()),
        "brief_messages": len(brief), "brief_chars": len(text),
        "brief_table": bool(TABLE.search(text)), "brief_rows": bool(ROWS.search(text)),
        "brief_exchange": bool(EXCHANGE.search(text)), "brief_pool": bool(POOL.search(text)),
        "brief_rule": bool(RULE.search(text)),
        "brief_calls": {k: sum(c["phase"] == "brief" for c in calls[k]) for k in "AB"},
        "msgs_per_query": float(np.mean([len(q["messages"]) for q in recs])),
        "msgs_last20": float(np.mean([len(q["messages"]) for q in recs[-20:]])),
        # last partner query (1..T) with any message, 0 if none
        "last_msg_query": max([i + 1 for i, q in enumerate(recs) if q["messages"]] + [0]),
        # each twin's first talk move at the first partner query: S(end), R(ead) or -
        "first_move": "".join(next(({"send_message": "S", "read_messages": "R"}[c["tool"]]
                                     for c in part[k] if c["phase"] == "query"
                                     and c["query"] == recs[0]["number"]
                                     and c["tool"] in ("send_message", "read_messages")), "-")
                              for k in "AB") if recs else "",
        "add_case_values": {k: sum(c["tool"] == "add_case" and bool(c["args"].get("values"))
                                   for c in part[k]) for k in "AB"},
        "count_joint": {k: sum(c["tool"] == "count" and c["args"].get("name") in with_vals[k]
                               for c in part[k]) for k in "AB"},
        "rule_query": rule_q,
        # messages (round and queries) never shown in a read_messages result of the recipient
        "messages_total": len(brief) + sum(len(q["messages"]) for q in recs),
        "messages_unread": sum(
            m["text"][:150] not in seen[("B" if m["from"] == "A" else "A")]
            for m in brief + [m for q in recs for m in q["messages"]]),
        # first partner query (displayed number) with own feature values in a message, and the
        # first message (round = 0) that asks about the other twin's features
        "values_query": next((q["number"] for q in recs for m in q["messages"]
                              if VALUES.search(m["text"])), None),
        "asks_features": next((0 for m in brief if ASKS.search(m["text"])), None)
        or next((q["number"] for q in recs for m in q["messages"] if ASKS.search(m["text"])),
                None),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline-root", default=str(ROOT.parent / "agent-consensus" / "results"))
    args = p.parse_args()
    arms = {"none": (Path(args.baseline_root), "boolean_haiku_xor"),
            "brief": (ROOT / "results", "boolean_haiku_xor_brief"),
            "announced": (ROOT / "results", "boolean_haiku_xor_brief_announced"),
            "other": (ROOT / "results", "boolean_haiku_xor_other"),
            "other_round": (ROOT / "results", "boolean_haiku_xor_other_round"),
            "other_notify": (ROOT / "results", "boolean_haiku_xor_other_notify"),
            "other_round_notify": (ROOT / "results", "boolean_haiku_xor_other_round_notify")}
    out = {}
    for arm, (res, exp) in arms.items():
        rows = [pair_stats(d, r) for r, d in pair_dirs(res, exp)]
        out[arm] = rows
        if not rows:
            continue
        m = lambda key: np.mean([x[key] for x in rows])  # noqa: E731
        print(f"== {arm} ({exp}, {len(rows)} pairs): acc {m('acc'):.3f}, last 20 "
              f"{m('acc_last20'):.3f}, conf {m('conf'):.3f}, msgs/query {m('msgs_per_query'):.2f}"
              f" (last 20 {m('msgs_last20'):.2f}); XOR mentioned in "
              f"{sum(x['rule_query'] is not None for x in rows)} pairs, last 20 >= 0.8 in "
              f"{sum(x['acc_last20'] >= 0.8 for x in rows)}")
        for x in rows:
            flags = "".join(f for f, key in (("T", "brief_table"), ("R", "brief_rows"),
                                             ("E", "brief_exchange"), ("P", "brief_pool"),
                                             ("X", "brief_rule")) if x[key])
            print(f"  run{x['run']}: acc {x['acc']:.2f} last20 {x['acc_last20']:.2f} conf "
                  f"{x['conf']:.2f} | brief {x['brief_messages']} msgs {x['brief_chars']} ch "
                  f"[{flags}] | msgs/q {x['msgs_per_query']:.1f} (last20 {x['msgs_last20']:.1f})"
                  f" | add_case+values {x['add_case_values']} count joint {x['count_joint']}"
                  f" | XOR first mentioned at {x['rule_query']} | first move {x['first_move']}, last "
                  f"message at partner query {x['last_msg_query']}")
    (ROOT / "results" / "xor_briefing_summary.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
