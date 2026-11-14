"""Twins as live agent sessions with real tools, who must reach a consensus (LogBook 2026-11-13).

Each twin is one Claude Agent SDK session (`ClaudeSDKClient`) for the whole run: no resume, no
per-action calls from a harness. The twin acts only through the tools of an in-process MCP
server, which wraps a shared, thread-safe `ConsensusAPI`:

- training phase: `get_training_data(n)` (up to 25 labelled cases per call, a fixed budget) and
  `regress(predictors)`. The partner is not mentioned. `start_queries()` ends the phase; its
  result introduces the partner, free communication and the consensus rule.
- query phase: `next_query()` waits for both twins (same case); `submit_opening(y)` records the
  twin's own estimate before any talk (so weight of advice and entrenchment are measured as in
  the earlier experiments); `send_message(text)` / `read_messages()` (waits until the partner
  sends something new, submits a prediction, or is also waiting); `submit_prediction(y)` waits
  for the partner's submission. If the two are within `tol`, the prediction is accepted and the
  true y revealed; otherwise both see the other's value and try again. After `max_attempts`
  failed attempts the query fails and scores `penalty`.

Each twin runs in its own thread with its own event loop; tool handlers run the (possibly
blocking) API calls in a worker thread (`asyncio.to_thread`), so a twin waiting for its partner
never blocks its own session. Everything is written to disk as it happens (actions, the
twin's messages and tool calls, query records, a status file per twin).
"""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from consensus.llm_agent import (
    NO_OVERLAP_FEATURE_LINE,
    OVERLAP_FEATURE_LINE,
    PARTNER_INFO_LINE,
    STRUCTURE_HINT,
    regress,
)

log = logging.getLogger("sessions")

SYSTEM_TRAIN = """\
You are agent {name}. Your task is to predict a real-valued label y from {d} numeric features \
({names}) of each case. You work only through the tools of a data API.

Training phase: get_training_data gives you up to 25 labelled cases per call ({budget} in \
total), and regress fits y on the predictors you choose by least squares over every case whose \
y you know. {study_rule}When you are ready, call start_queries to end the training phase; you \
cannot get more training cases after that, and its result explains what comes next.

Keep working with the tools until a tool result says that the run is over; do not stop before \
that."""

START_QUERIES = """\
The training phase is over ({n_train} training cases studied). From now on you have a partner, \
agent {other}: another instance of the same AI model as you, with the same instructions, which \
trained separately on its own cases. {partner_par}

There are {T} queries. For each one:
- next_query gives you the features of the case (it waits until your partner has also asked, \
so you both work on the same case);
- submit_opening records your own estimate of y before you talk;
- then you and your partner can communicate freely: send_message sends a message, and \
read_messages shows your partner's opening and new messages (it waits until your partner sends \
something new, submits a prediction, or is also waiting);
{rule_par} regress can now also use \
own_opening and partner_opening (your and your partner's openings, over past queries). You can \
make at most {cap} tool calls per query. Call next_query to begin."""

# The team rule, in the start_queries text: the twins must agree ("consensus"), or each submits
# its own final and the team's prediction is their average ("average").
RULE_CONSENSUS = """\
- you must reach a consensus: submit_prediction is accepted only when both of you submit \
predictions within {tol} of each other, and then the true y is revealed to both of you. If your \
predictions differ, you are both told the other's prediction and must talk and submit again. \
After {max_attempts} failed attempts the query fails and scores a penalty of {penalty}.
The team is scored by the squared error of the agreed prediction."""

RULE_AVERAGE = """\
- when you are ready, submit_prediction records your own final estimate of y (once per query). \
It waits until your partner has also submitted its final; then both finals, the team's \
prediction and the true y are revealed to both of you.
The team's prediction is the average of your two final estimates, and the team is scored by its \
squared error."""

TOOL_DOCS = {
    "get_training_data": "Get the next n labelled training cases (n from 1 to 25).",
    "regress": (
        "Least squares of y on the given predictors plus an intercept over every case whose y "
        "you know. Returns the coefficients, in-sample and leave-one-out mean squared errors, "
        "and the prediction for the current case when all predictors are known."
    ),
    "start_queries": "End the training phase and start the query phase.",
    "next_query": "Query phase: get the features of the next case.",
    "submit_opening": "Query phase: record your own estimate of y for the current case.",
    "send_message": "Query phase: send a message (see the instructions given at start_queries).",
    "read_messages": "Query phase: read new messages (see the instructions given at start_queries).",
    "submit_prediction": "Query phase: submit your prediction for the current case.",
}


def weight_of_advice(own: float, partner: float, value: float, min_gap: float = 0.2) -> float:
    """(value - own) / (partner - own), clipped to [-0.5, 1.5] as in scripts/onset_llm.py; nan
    when the openings differ by at most `min_gap` (the ratio is then unstable) or a value is
    missing. Under consensus the two twins' WOA on the agreed value add up to 1, so the pair
    mean is 0.5 by construction; each twin's own WOA says who conceded."""
    if not all(np.isfinite([own, partner, value])) or abs(partner - own) <= min_gap:
        return float("nan")
    return float(np.clip((value - own) / (partner - own), -0.5, 1.5))


class ConsensusAPI:
    """Shared, thread-safe data API with a consensus rule (see the module docstring).

    Parameters
    ----------
    train : {"A": (X, y), "B": (X, y)}
        Training samples per twin (the first `budget` are available).
    queries : {"A": X_A, "B": X_B}; y : (T,)
        The query stream (paired across twins).
    labels : {"A": [...], "B": [...]}
        Feature names per twin (shared names = the same variable).
    """

    def __init__(
        self,
        train: dict,
        queries: dict,
        y: np.ndarray,
        labels: dict,
        budget: int | dict,
        tol: float = 0.01,
        max_attempts: int = 3,
        penalty: float = 10.0,
        max_chars: int = 2000,
        cap: int = 25,
        study_all: bool = False,
        team_rule: str = "consensus",
    ) -> None:
        if team_rule not in ("consensus", "average"):
            raise ValueError(f"team_rule must be 'consensus' or 'average', got {team_rule!r}")
        self.team_rule = team_rule
        self.cond = threading.Condition()
        # Training budget per twin (an int for both, or {"A": n_A, "B": n_B}); with `study_all`
        # a twin must pull its whole budget before start_queries (the entrenchment manipulation).
        self.budget = budget if isinstance(budget, dict) else {"A": budget, "B": budget}
        self.labels, self.study_all = labels, study_all
        self.Xtr = {k: train[k][0][: self.budget[k]] for k in "AB"}
        self.ytr = {k: train[k][1][: self.budget[k]] for k in "AB"}
        self.qx, self.y, self.T = queries, np.asarray(y, float), len(y)
        self.tol, self.max_attempts, self.penalty = tol, max_attempts, penalty
        self.max_chars, self.cap = max_chars, cap
        self.n_seen = {k: 0 for k in "AB"}
        self.phase = {k: "train" for k in "AB"}  # train | query | done
        self.q = {k: -1 for k in "AB"}
        self.want = {k: -1 for k in "AB"}  # query index the twin waits for in next_query
        self.calls = {k: {} for k in "AB"}  # tool calls per query
        self.opening = {k: np.full(self.T, np.nan) for k in "AB"}
        self.attempts = {k: [[] for _ in range(self.T)] for k in "AB"}  # submissions per query
        self.agreed = np.full(self.T, np.nan)  # the accepted prediction
        self.failed = np.zeros(self.T, bool)
        self.resolved = np.zeros(self.T, bool)
        self.inbox = {k: [] for k in "AB"}  # (q, text) received
        self.n_read = {k: 0 for k in "AB"}
        self.waiting = {k: None for k in "AB"}  # "read" | "submit" | None
        self.messages: list[dict] = []
        self.on_query = None  # fn(record), called once per resolved query
        self.on_call = None  # fn(twin, record), called after every tool call

    @staticmethod
    def other(k: str) -> str:
        return "B" if k == "A" else "A"

    def done(self, k: str) -> bool:
        return self.phase[k] == "done"

    # ---- bookkeeping ---------------------------------------------------------------------------
    def call(self, k: str, tool: str, args: dict, fn) -> str:
        """Run one tool call for twin k with the per-query cap, and report it."""
        q = self.q[k]
        with self.cond:
            n = self.calls[k].get(q, 0) + 1
            self.calls[k][q] = n
        if self.phase[k] == "query" and q >= 0 and n > self.cap and tool != "submit_prediction":
            res = (
                f"Tool-call limit for query {q + 1} reached: call submit_prediction now (or "
                "next_query if the query is resolved)."
            )
        else:
            res = fn()
        if self.on_call is not None:
            self.on_call(k, {"q": q, "tool": tool, "args": args, "result": res[:3000],
                             "time": time.time()})
        return res

    def _known_rows(self, k: str) -> list[dict]:
        o, lab = self.other(k), self.labels[k]
        rows = [
            {**dict(zip(lab, map(float, x))), "y": float(v)}
            for x, v in zip(self.Xtr[k][: self.n_seen[k]], self.ytr[k][: self.n_seen[k]])
        ]
        for q in range(self.T):
            if self.resolved[q]:
                rows.append(
                    {
                        **dict(zip(lab, map(float, self.qx[k][q]))),
                        "own_opening": float(self.opening[k][q]),
                        "partner_opening": float(self.opening[o][q]),
                        "y": float(self.y[q]),
                    }
                )
        return rows

    def record(self, q: int) -> dict:
        """Everything about query q (for the live file and the analysis)."""
        return {
            "t": q,
            "y": float(self.y[q]),
            "x": {k: [float(v) for v in self.qx[k][q]] for k in "AB"},
            "opening": {k: float(self.opening[k][q]) for k in "AB"},
            "attempts": {k: [float(v) for v in self.attempts[k][q]] for k in "AB"},
            "final": {k: float(self.attempts[k][q][-1]) if self.attempts[k][q] else float("nan")
                      for k in "AB"},
            "agreed": float(self.agreed[q]),
            "failed": bool(self.failed[q]),
            "team": float(self.agreed[q]),
            "messages": [m for m in self.messages if m["q"] == q],
            # Weight of advice per twin: on the agreed value, and on its own first submission
            # (what it would have predicted before any resubmission).
            "woa": {  # on the agreed value (consensus) or the twin's own final (average)
                k: weight_of_advice(
                    self.opening[k][q],
                    self.opening[self.other(k)][q],
                    (self.attempts[k][q][-1] if self.attempts[k][q] else float("nan"))
                    if self.team_rule == "average" else self.agreed[q],
                )
                for k in "AB"
            },
            "woa_first": {
                k: weight_of_advice(
                    self.opening[k][q],
                    self.opening[self.other(k)][q],
                    self.attempts[k][q][0] if self.attempts[k][q] else float("nan"),
                )
                for k in "AB"
            },
        }

    # ---- tools -------------------------------------------------------------------------------
    def get_training_data(self, k: str, n: int) -> str:
        with self.cond:
            if self.phase[k] != "train":
                return "Error: the training phase is over."
            n = int(min(max(int(n), 1), 25, self.budget[k] - self.n_seen[k]))
            if n <= 0:
                return f"You have used all {self.budget[k]} training cases."
            a, b = self.n_seen[k], self.n_seen[k] + n
            self.n_seen[k] = b
            head = ", ".join(self.labels[k]) + " | y"
            rows = [
                ", ".join(f"{v:.3f}" for v in x) + f" | {v_:.3f}"
                for x, v_ in zip(self.Xtr[k][a:b], self.ytr[k][a:b])
            ]
            return f"Training cases {a + 1}-{b} of {self.budget[k]} ({head}):\n" + "\n".join(rows)

    def regress(self, k: str, predictors: list[str]) -> str:
        with self.cond:
            allowed = list(self.labels[k]) + (
                ["own_opening", "partner_opening"] if self.phase[k] != "train" else []
            )
            preds = [p for p in predictors if p in allowed]
            if not preds:
                return f"Error: allowed predictors are {', '.join(allowed)}."
            rows = self._known_rows(k)
            if any(p in ("own_opening", "partner_opening") for p in preds):
                rows = [r for r in rows if "own_opening" in r]
            current = None
            q = self.q[k]
            if self.phase[k] == "query" and 0 <= q < self.T and not self.resolved[q]:
                current = dict(zip(self.labels[k], map(float, self.qx[k][q])))
                if not np.isnan(self.opening[k][q]):
                    current["own_opening"] = float(self.opening[k][q])
                    po = self.opening[self.other(k)][q]
                    if not np.isnan(po):
                        current["partner_opening"] = float(po)
            return regress(rows, preds, current, rows="cases with known y")

    def start_queries(self, k: str, intro: str) -> str:
        with self.cond:
            if self.phase[k] != "train":
                return "The training phase is already over."
            if self.study_all and self.n_seen[k] < self.budget[k]:
                return (
                    f"Error: study all {self.budget[k]} training cases first (you have seen "
                    f"{self.n_seen[k]})."
                )
            self.phase[k] = "query"
            self.cond.notify_all()
            return intro.format(n_train=self.n_seen[k])

    def next_query(self, k: str) -> str:
        with self.cond:
            if self.phase[k] == "train":
                return "Error: call start_queries first."
            if self.phase[k] == "done":
                return "The run is over. Stop."
            if self.q[k] >= 0 and not self.resolved[self.q[k]]:
                return f"Error: query {self.q[k] + 1} is not resolved yet (reach a consensus first)."
            nq = self.q[k] + 1
            if nq >= self.T:
                self.phase[k] = "done"
                self.cond.notify_all()
                return "There are no more queries: the run is over. Stop."
            self.want[k] = nq
            self.cond.notify_all()
            o = self.other(k)
            self.cond.wait_for(lambda: self.want[o] >= nq or self.q[o] >= nq or self.done(o))
            self.q[k], self.n_read[k] = nq, 0
            feats = ", ".join(f"{lab}={v:.3f}" for lab, v in zip(self.labels[k], self.qx[k][nq]))
            return f"Query {nq + 1} of {self.T}. Your features: {feats}."

    def _in_query(self, k: str) -> str | None:
        if self.phase[k] != "query" or self.q[k] < 0:
            return "Error: no current query (call next_query)."
        if self.resolved[self.q[k]]:
            return "Error: this query is resolved; call next_query."
        return None

    def submit_opening(self, k: str, y: float) -> str:
        with self.cond:
            if err := self._in_query(k):
                return err
            q = self.q[k]
            if not np.isnan(self.opening[k][q]):
                return "Error: you already submitted your opening for this query."
            self.opening[k][q] = float(y)
            self.cond.notify_all()
            return f"Opening {float(y):.3f} recorded for query {q + 1}."

    def send_message(self, k: str, text: str) -> str:
        with self.cond:
            if err := self._in_query(k):
                return err
            q = self.q[k]
            if np.isnan(self.opening[k][q]):
                return "Error: submit your opening first."
            text = str(text)[: self.max_chars]
            self.inbox[self.other(k)].append((q, text))
            self.messages.append({"q": q, "from": k, "text": text, "order": len(self.messages)})
            self.cond.notify_all()
            return "Message sent."

    def _unread(self, k: str, q: int) -> list[str]:
        return [t for qq, t in self.inbox[k] if qq == q][self.n_read[k] :]

    def read_messages(self, k: str) -> str:
        with self.cond:
            if err := self._in_query(k):
                return err
            q, o = self.q[k], self.other(k)
            if np.isnan(self.opening[k][q]):
                return "Error: submit your opening first."
            n_att = len(self.attempts[o][q])
            self.waiting[k] = "read"
            self.cond.notify_all()
            self.cond.wait_for(
                lambda: self._unread(k, q)
                or len(self.attempts[o][q]) > n_att
                or self.waiting[o] == "submit"
                or (self.waiting[o] == "read" and not self._unread(o, q))
                or self.done(o)
                or self.resolved[q]
            )
            self.waiting[k] = None
            new = self._unread(k, q)
            self.n_read[k] += len(new)
            po = self.opening[o][q]
            lines = [
                f"Partner's opening for query {q + 1}: "
                + ("not submitted yet." if np.isnan(po) else f"{po:.3f}.")
            ]
            lines += [f'Partner: "{t}"' for t in new] or ["No new messages."]
            if self.waiting[o] == "submit" or len(self.attempts[o][q]) > len(self.attempts[k][q]):
                lines.append(
                    f"Your partner has submitted the prediction {self.attempts[o][q][-1]:.3f} "
                    "and is waiting for yours."
                )
            elif not new and self.waiting[o] == "read":
                lines.append("Your partner is also waiting for a message.")
            return "\n".join(lines)

    def submit_prediction(self, k: str, y: float) -> str:
        with self.cond:
            if err := self._in_query(k):
                return err
            q, o = self.q[k], self.other(k)
            if np.isnan(self.opening[k][q]):
                return "Error: submit your opening first."
            if self.team_rule == "average" and self.attempts[k][q]:
                return "Error: you already submitted your final for this query."
            self.attempts[k][q].append(float(y))
            a = len(self.attempts[k][q])  # attempt number
            self.waiting[k] = "submit"
            self.cond.notify_all()
            self.cond.wait_for(
                lambda: len(self.attempts[o][q]) >= a or self.done(o) or self.resolved[q]
            )
            self.waiting[k] = None
            if not self.resolved[q]:
                self._resolve(q, a)
            if self.resolved[q]:
                if q == self.T - 1:  # the last query is resolved: this twin's run is over
                    self.phase[k] = "done"
                    self.cond.notify_all()
                tail = (
                    " That was the last query: the run is over. Stop."
                    if q == self.T - 1
                    else " Call next_query for the next case."
                )
                if self.team_rule == "average":
                    pf = self.attempts[o][q][0] if self.attempts[o][q] else float("nan")
                    return (
                        f"Both finals are in: yours {float(y):.3f}, your partner's {pf:.3f}; team "
                        f"(average) {self.agreed[q]:.3f}. The true y was {self.y[q]:.3f}; team "
                        f"squared error {(self.agreed[q] - self.y[q]) ** 2:.3f}." + tail
                    )
                if self.failed[q]:
                    return (
                        f"No consensus after {self.max_attempts} attempts: query {q + 1} failed "
                        f"(penalty {self.penalty:g}). The true y was {self.y[q]:.3f}." + tail
                    )
                return (
                    f"Consensus reached on {self.agreed[q]:.3f}. The true y was {self.y[q]:.3f}; "
                    f"squared error {(self.agreed[q] - self.y[q]) ** 2:.3f}." + tail
                )
            pa = self.attempts[o][q][a - 1]
            return (
                f"No consensus: you submitted {float(y):.3f}, your partner submitted {pa:.3f} "
                f"(attempt {a} of {self.max_attempts}). Talk with your partner and submit again."
            )

    def _resolve(self, q: int, a: int) -> None:
        """Called (under the lock) when both twins have made attempt a, or the partner is gone."""
        sub = {k: self.attempts[k][q][a - 1] if len(self.attempts[k][q]) >= a else None
               for k in "AB"}
        if self.team_rule == "average":  # each twin's own final; the team averages them
            vals = [v for v in sub.values() if v is not None]
            self.agreed[q], self.resolved[q] = float(np.mean(vals)), True
        elif sub["A"] is None or sub["B"] is None:  # the partner is gone: accept the submission
            v = sub["A"] if sub["A"] is not None else sub["B"]
            self.agreed[q], self.resolved[q] = v, True
        elif abs(sub["A"] - sub["B"]) <= self.tol:
            self.agreed[q], self.resolved[q] = (sub["A"] + sub["B"]) / 2, True
        elif a >= self.max_attempts:
            self.failed[q], self.resolved[q] = True, True
        if self.resolved[q]:
            self.cond.notify_all()
            if self.on_query is not None:
                self.on_query(self.record(q))

    def finish(self, k: str) -> None:
        with self.cond:
            self.phase[k] = "done"
            self.cond.notify_all()


def build_tools(api: ConsensusAPI, k: str, intro: str):
    """The twin's tools as an in-process MCP server (`data`) and their allowed names."""
    from claude_agent_sdk import create_sdk_mcp_server, tool

    def text(s: str) -> dict:
        return {"content": [{"type": "text", "text": s}]}

    async def run(name: str, args: dict, fn) -> dict:
        return text(await asyncio.to_thread(api.call, k, name, args, fn))

    obj = {"type": "object"}

    @tool("get_training_data", TOOL_DOCS["get_training_data"],
          {**obj, "properties": {"n": {"type": "integer"}}, "required": ["n"]})
    async def get_training_data(args):
        return await run("get_training_data", args,
                         lambda: api.get_training_data(k, args.get("n", 25)))

    @tool("regress", TOOL_DOCS["regress"],
          {**obj, "properties": {"predictors": {"type": "array", "items": {"type": "string"}}},
           "required": ["predictors"]})
    async def regress_tool(args):
        return await run("regress", args, lambda: api.regress(k, list(args.get("predictors", []))))

    @tool("start_queries", TOOL_DOCS["start_queries"], {**obj, "properties": {}})
    async def start_queries(args):
        return await run("start_queries", args, lambda: api.start_queries(k, intro))

    @tool("next_query", TOOL_DOCS["next_query"], {**obj, "properties": {}})
    async def next_query(args):
        return await run("next_query", args, lambda: api.next_query(k))

    @tool("submit_opening", TOOL_DOCS["submit_opening"],
          {**obj, "properties": {"y": {"type": "number"}}, "required": ["y"]})
    async def submit_opening(args):
        return await run("submit_opening", args, lambda: api.submit_opening(k, float(args["y"])))

    @tool("send_message", TOOL_DOCS["send_message"],
          {**obj, "properties": {"text": {"type": "string"}}, "required": ["text"]})
    async def send_message(args):
        return await run("send_message", args, lambda: api.send_message(k, args.get("text", "")))

    @tool("read_messages", TOOL_DOCS["read_messages"], {**obj, "properties": {}})
    async def read_messages(args):
        return await run("read_messages", args, lambda: api.read_messages(k))

    @tool("submit_prediction", TOOL_DOCS["submit_prediction"],
          {**obj, "properties": {"y": {"type": "number"}}, "required": ["y"]})
    async def submit_prediction(args):
        return await run("submit_prediction", args,
                         lambda: api.submit_prediction(k, float(args["y"])))

    tools = [get_training_data, regress_tool, start_queries, next_query, submit_opening,
             send_message, read_messages, submit_prediction]
    server = create_sdk_mcp_server("data", tools=tools)
    return server, [f"mcp__data__{t.name}" for t in tools], tools


def intro_text(api: ConsensusAPI, k: str, structure_hint: bool, partner_info: bool) -> str:
    """The start_queries result for twin k (`{n_train}` is filled in at the call)."""
    labels, other = api.labels[k], api.labels[api.other(k)]
    shared = [lab for lab in labels if lab in other]
    fmt = dict(own=", ".join(labels), other=", ".join(other))
    parts = [
        OVERLAP_FEATURE_LINE.format(shared=" and ".join(shared), **fmt)
        if shared
        else NO_OVERLAP_FEATURE_LINE.format(**fmt)
    ]
    if partner_info:
        parts.append(PARTNER_INFO_LINE)
    if structure_hint:
        parts.append(STRUCTURE_HINT)
    rule = (RULE_AVERAGE if api.team_rule == "average" else RULE_CONSENSUS).format(
        tol=f"{api.tol:g}", max_attempts=api.max_attempts, penalty=f"{api.penalty:g}"
    )
    return START_QUERIES.format(
        rule_par=rule,
        n_train="{n_train}",
        other=api.other(k),
        partner_par=" ".join(parts),
        T=api.T,
        tol=f"{api.tol:g}",
        max_attempts=api.max_attempts,
        penalty=f"{api.penalty:g}",
        cap=api.cap,
    )


class TwinSession:
    """One twin as one live agent session, run in its own thread (see the module docstring)."""

    def __init__(
        self,
        name: str,
        api: ConsensusAPI,
        model: str = "claude-haiku-5-5",
        effort: str = "low",
        structure_hint: bool = True,
        partner_info: bool = True,
        max_budget_usd: float = 5.0,
        max_continues: int = 10,
        live_dir: str | Path | None = None,
        run: int = 0,
    ) -> None:
        self.name, self.api, self.model, self.effort = name, api, model, effort
        self.max_budget_usd, self.max_continues = max_budget_usd, max_continues
        self.intro = intro_text(api, name, structure_hint, partner_info)
        labels = api.labels[name]
        self.system = SYSTEM_TRAIN.format(
            name=name, d=len(labels), names=", ".join(labels), budget=api.budget[name],
            study_rule=(
                f"You must study all {api.budget[name]} training cases before the next phase. "
                if api.study_all else ""
            ),
        )
        self.live = Path(live_dir) if live_dir is not None else None
        self.run_id = run
        self.cost_usd = 0.0
        self.tokens = {"input": 0, "cache_read": 0, "cache_write": 0, "output": 0}
        self.results: list[dict] = []
        self.compactions: list[dict] = []
        self.error: str | None = None
        self.n_continues = 0
        if self.live is not None:
            self.live.mkdir(parents=True, exist_ok=True)
            for kind in ("calls", "messages", "compactions"):
                self._path(kind).write_text("")

    # ---- live files ----------------------------------------------------------------------------
    def _path(self, kind: str) -> Path:
        return self.live / f"run{self.run_id}_{kind}_{self.name}.jsonl"

    def append(self, kind: str, row: dict) -> None:
        if self.live is not None:
            with self._path(kind).open("a") as f:
                f.write(json.dumps(row) + "\n")

    def write_status(self, state: str = "running") -> None:
        if self.live is None:
            return
        k, api = self.name, self.api
        status = {
            "twin": k,
            "run": self.run_id,
            "state": state,
            "phase": api.phase[k],
            "query": api.q[k] + 1,
            "T": api.T,
            "training_cases": api.n_seen[k],
            "tool_calls": sum(api.calls[k].values()),
            "tokens": self.tokens,
            "cost_usd_so_far": round(self.cost_usd, 4),
            "continues": self.n_continues,
            "compactions": len(self.compactions),
            "error": self.error,
            "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        (self.live / f"run{self.run_id}_status_{k}.json").write_text(json.dumps(status, indent=1))

    # ---- the session -----------------------------------------------------------------------------
    def run(self) -> None:
        try:
            asyncio.run(self._main())
        except Exception as e:  # noqa: BLE001 (logged and recorded; the partner must not wait)
            self.error = repr(e)
            log.exception("twin %s failed", self.name)
        finally:
            self.api.finish(self.name)
            self.write_status("finished" if self.error is None else "failed")
            log.info("twin %s: finished, $%.3f, %d continues", self.name, self.cost_usd,
                     self.n_continues)

    async def _main(self) -> None:
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            ClaudeSDKClient,
            HookMatcher,
            ResultMessage,
            TextBlock,
            ThinkingBlock,
            ToolUseBlock,
        )

        server, allowed, _ = build_tools(self.api, self.name, self.intro)

        async def pre_compact(inp, tool_use_id, ctx):
            ev = {"q": self.api.q[self.name], "time": time.time(),
                  "trigger": inp.get("trigger") if isinstance(inp, dict) else None}
            self.compactions.append(ev)
            self.append("compactions", ev)
            return {}

        options = ClaudeAgentOptions(
            system_prompt=self.system,
            mcp_servers={"data": server},
            allowed_tools=allowed,
            tools=[],  # no built-in tools: the data API is the only way to act
            setting_sources=[],
            strict_mcp_config=True,
            model=self.model,
            effort=self.effort,
            max_budget_usd=self.max_budget_usd,
            cwd=tempfile.mkdtemp(prefix=f"twin{self.name}_"),
            env={"MCP_TOOL_TIMEOUT": "3600000"},  # waiting for the partner can take minutes
            hooks={"PreCompact": [HookMatcher(hooks=[pre_compact])]},
        )
        prompt = "Begin. You are in the training phase."
        async with ClaudeSDKClient(options=options) as client:
            for i in range(self.max_continues + 1):
                await client.query(prompt)
                async for msg in client.receive_response():
                    if isinstance(msg, AssistantMessage):
                        u = msg.usage or {}
                        self.tokens["input"] += u.get("input_tokens", 0) or 0
                        self.tokens["cache_read"] += u.get("cache_read_input_tokens", 0) or 0
                        self.tokens["cache_write"] += u.get("cache_creation_input_tokens", 0) or 0
                        self.tokens["output"] += u.get("output_tokens", 0) or 0
                        for b in msg.content:
                            row = {"q": self.api.q[self.name], "time": time.time()}
                            if isinstance(b, TextBlock):
                                self.append("messages", {**row, "type": "text", "text": b.text})
                            elif isinstance(b, ThinkingBlock):
                                self.append("messages", {**row, "type": "thinking",
                                                         "text": b.thinking})
                            elif isinstance(b, ToolUseBlock):
                                self.append("messages", {**row, "type": "tool_use",
                                                         "name": b.name, "input": b.input})
                        self.write_status()
                    elif isinstance(msg, ResultMessage):
                        res = {
                            "subtype": msg.subtype,
                            "is_error": msg.is_error,
                            "num_turns": msg.num_turns,
                            "total_cost_usd": msg.total_cost_usd,
                            "usage": msg.usage,
                            "result": (msg.result or "")[:2000],
                        }
                        self.results.append(res)
                        self.append("messages", {"q": self.api.q[self.name], "type": "result",
                                                 **res})
                        # The reported cost is cumulative for the session: keep the latest.
                        self.cost_usd = max(self.cost_usd, msg.total_cost_usd or 0.0)
                if self.api.done(self.name) or (self.results and self.results[-1]["subtype"]
                                                 == "error_max_budget_usd"):
                    break
                self.n_continues = i + 1
                prompt = (
                    "The run is not over yet. Continue working through the tools (next_query, "
                    "submit_opening, talking with your partner, submit_prediction) until a tool "
                    "result says the run is over."
                )
                self.write_status()


def run_pair(api: ConsensusAPI, twins: dict[str, TwinSession]) -> None:
    """Run both twins' sessions in parallel threads until both are done."""
    threads = [threading.Thread(target=tw.run, name=f"twin{k}") for k, tw in twins.items()]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
