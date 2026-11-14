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

import ast
import asyncio
import json
import operator
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
cannot get more training cases after that, and its result explains what comes next.{model_train_par}

Keep working with the tools until a tool result says that the run is over; do not stop before \
that."""

START_QUERIES = """\
{opening} {partner_par}

There are {T} queries. For each one:
- next_query gives you the features of the case (it waits until your partner has also asked, \
so you both work on the same case);
- submit_opening records your own estimate of y before you talk;
- then you and your partner can communicate freely: send_message sends a message, and \
read_messages shows your partner's opening and new messages (it waits until your partner sends \
something new, submits a prediction, or is also waiting);
{rule_par} regress can now also use \
own_opening and partner_opening (your and your partner's openings, over past queries). You can \
make at most {cap} tool calls per query.{model_par} Call next_query to begin."""

OPENING_TRAINED = """\
The training phase is over ({n_train} training cases studied). From now on you have a partner, \
agent {other}: another instance of the same AI model as you, with the same instructions, which \
trained separately on its own cases."""

# `no_training` (LogBook 2026-11-14): the twins start without any labelled cases and learn only
# from the shared queries, whose y is revealed after each one; they have a partner from query 1.
OPENING_UNTRAINED = """\
There is no training phase: you start without any labelled cases, and the true y of each query \
below is revealed after it. You have a partner, agent {other}: another instance of the same AI \
model as you, with the same instructions, which also starts without any labelled cases."""

SYSTEM_UNTRAINED = """\
You are agent {name}. Your task is to predict a real-valued label y from {d} numeric features \
({names}) of each case. You work only through the tools of a data API; regress fits y on the \
predictors you choose by least squares over every case whose y you know.

{intro}

Keep working with the tools until a tool result says that the run is over; do not stop before \
that."""

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

# Each twin scored on its own final ("individual"): nothing rewards leaving the combining to a
# team average, so keeping one's own estimate measures conviction, not delegation.
RULE_INDIVIDUAL = """\
- when you are ready, submit_prediction records your own final estimate of y (once per query). \
It waits until your partner has also submitted its final; then both finals and the true y are \
revealed to both of you.
You are scored by the squared error of your own final estimate, and your partner by the squared \
error of its own."""

# `same_y`: say outright that both twins predict the same quantity (true at every lambda, unlike
# the structure hint). Nothing is said about how to use the partner's estimate or features.
SAME_Y_LINE = "You and your partner predict the same quantity y for each case."

# `model_tools`: fresh least-squares model instances the twin can start at any time, with any
# columns. Described only by what they do; nothing says which columns to use.
# `models_in_training`: model instances from the training phase on, with expression columns.
MODEL_TRAIN_PAR = """

You can also build model instances: new_model starts a least-squares model with a name and the \
columns you choose: your feature names, arithmetic expressions of them (for example {ex1}*{ex2} or \
{ex3}^2), or other names whose values you give later. With include_training it is filled with your \
training cases (also the ones you get afterwards). fit_model fits one of your models. Your models \
stay available after the training phase."""

MODEL_PAR_TRAINED = """ Your model instances stay available, and new_model can start a fresh one at \
any time with the columns you choose (your features, own_opening, partner_opening, arithmetic \
expressions of these, or other names whose values you give); add_case adds the current case to \
one of your models (its y is added once it is revealed); fit_model fits one of your models on its \
cases and predicts the current case."""

MODEL_PAR = """ You can also start fresh model instances at any time: new_model starts an empty \
least-squares model with a name and the columns you choose (values for columns named after your \
features, own_opening and partner_opening are filled in automatically; for any other column you \
give the value); add_case adds the current case to one of your models (its y is added once it is \
revealed); fit_model fits one of your models on the cases added to it and predicts the current \
case."""

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
    "new_model": (
        "Start a fresh least-squares model instance with a name and the columns you choose "
        "(see your instructions for which columns are filled in automatically); "
        "include_training fills it with your training cases where that is possible."
    ),
    "add_case": (
        "Query phase: add the current case to one of your model instances; give the values of "
        "columns that are not filled in automatically in `values` (a JSON object)."
    ),
    "fit_model": (
        "Query phase: fit one of your model instances on the cases added to it (with their "
        "revealed y) and predict the current case; give values of columns that are not filled "
        "in automatically in `values`."
    ),
}


def weight_of_advice(own: float, partner: float, value: float, min_gap: float = 0.2) -> float:
    """(value - own) / (partner - own), clipped to [-0.5, 1.5] as in scripts/onset_llm.py; nan
    when the openings differ by at most `min_gap` (the ratio is then unstable) or a value is
    missing. Under consensus the two twins' WOA on the agreed value add up to 1, so the pair
    mean is 0.5 by construction; each twin's own WOA says who conceded."""
    if not all(np.isfinite([own, partner, value])) or abs(partner - own) <= min_gap:
        return float("nan")
    return float(np.clip((value - own) / (partner - own), -0.5, 1.5))


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
        ast.UAdd: operator.pos}


def eval_expr(expr: str, variables: dict) -> float | None:
    """Value of an arithmetic expression of `variables` (+ - * / ^ and parentheses), or None if
    it is not such an expression (other names, calls, ...). Used for model columns like a1*a2."""
    try:
        tree = ast.parse(expr.replace("^", "**"), mode="eval")
    except SyntaxError:
        return None

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id in variables:
            return float(variables[node.id])
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError(expr)

    try:
        v = ev(tree)
        return float(v) if np.isfinite(v) else None
    except (ValueError, ZeroDivisionError, OverflowError, TypeError):
        return None


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
        models_in_training: bool = False,
        no_training: bool = False,
    ) -> None:
        # No training phase: both twins start in the query phase with no labelled cases.
        self.no_training = no_training
        if no_training:
            budget = 0
        # Model instances from the training phase on (with expression columns and training cases).
        self.models_in_training = models_in_training
        if team_rule not in ("consensus", "average", "individual"):
            raise ValueError(
                f"team_rule must be 'consensus', 'average' or 'individual', got {team_rule!r}"
            )
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
        self.phase = {k: "query" if no_training else "train" for k in "AB"}  # train | query | done
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
        # Fresh model instances per twin: {name: {"columns": [...], "rows": [...] (with y),
        # "pending": {q: row} (y not revealed yet), "created_q": int}}.
        self.models = {k: {} for k in "AB"}
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
            "models": {k: {n: {"columns": m["columns"], "n_rows": len(m["rows"]),
                               "created_q": m["created_q"]}
                           for n, m in self.models[k].items()} for k in "AB"},
            # Weight of advice per twin: on the agreed value, and on its own first submission
            # (what it would have predicted before any resubmission).
            "woa": {  # on the agreed value (consensus) or the twin's own final (average)
                k: weight_of_advice(
                    self.opening[k][q],
                    self.opening[self.other(k)][q],
                    (self.attempts[k][q][-1] if self.attempts[k][q] else float("nan"))
                    if self.team_rule != "consensus" else self.agreed[q],
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
            for m in self.models[k].values():  # models filled with training cases grow with them
                if m.get("training"):
                    m["rows"] += self._training_rows(k, m["columns"], a, b)
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
            if self.team_rule != "consensus" and self.attempts[k][q]:
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
                if self.team_rule == "individual":
                    pf = self.attempts[o][q][0] if self.attempts[o][q] else float("nan")
                    return (
                        f"Both finals are in: yours {float(y):.3f}, your partner's {pf:.3f}. The "
                        f"true y was {self.y[q]:.3f}; your squared error "
                        f"{(float(y) - self.y[q]) ** 2:.3f}, your partner's "
                        f"{(pf - self.y[q]) ** 2:.3f}." + tail
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
        if self.team_rule != "consensus":  # each twin's own final; "team" = their average
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
            self._attach_y(q)
            self.cond.notify_all()
            if self.on_query is not None:
                self.on_query(self.record(q))

    def finish(self, k: str) -> None:
        with self.cond:
            self.phase[k] = "done"
            self.cond.notify_all()

    # ---- fresh model instances (model_tools) -----------------------------------------------
    def _auto_values(self, k: str, q: int) -> dict:
        """Values the API fills in for twin k at query q: its features, its opening, and the
        partner's opening once both have opened (as read_messages would show it)."""
        vals = dict(zip(self.labels[k], map(float, self.qx[k][q])))
        if not np.isnan(self.opening[k][q]):
            vals["own_opening"] = float(self.opening[k][q])
            po = self.opening[self.other(k)][q]
            if not np.isnan(po):
                vals["partner_opening"] = float(po)
        return vals

    def _training_rows(self, k: str, cols: list[str], a: int, b: int) -> list[dict]:
        """Rows of training cases a..b-1 for columns computable from twin k's features."""
        rows = []
        for x, yv in zip(self.Xtr[k][a:b], self.ytr[k][a:b]):
            v = dict(zip(self.labels[k], map(float, x)))
            rows.append({**{c: eval_expr(c, v) for c in cols}, "y": float(yv)})
        return rows

    def new_model(self, k: str, name: str, columns: list[str],
                  include_training: bool | None = None) -> str:
        with self.cond:
            if self.phase[k] == "train" and not self.models_in_training:
                return "Error: model instances are available in the query phase."
            if self.phase[k] == "done":
                return "The run is over."
            cols = list(dict.fromkeys(str(c) for c in columns))
            if not name or not cols or len(cols) > 30:
                return "Error: give a name and 1 to 30 columns."
            if include_training is None:  # default: with training cases while training
                include_training = self.phase[k] == "train"
            own = dict(zip(self.labels[k], [0.0] * len(self.labels[k])))
            if include_training and any(eval_expr(c, own) is None for c in cols):
                bad = [c for c in cols if eval_expr(c, own) is None]
                return (f"Error: include_training needs columns computable from your features; "
                        f"not computable: {', '.join(bad)}.")
            replaced = " (replacing the old one)" if name in self.models[k] else ""
            rows = self._training_rows(k, cols, 0, self.n_seen[k]) if include_training else []
            self.models[k][name] = {"columns": cols, "rows": rows, "pending": {},
                                    "created_q": self.q[k], "training": bool(include_training),
                                    "phase": self.phase[k]}
            if self.models_in_training:
                auto_vars = dict(own, own_opening=0.0, partner_opening=0.0)
                auto = [c for c in cols if eval_expr(c, auto_vars) is not None]
            else:
                auto = [c for c in cols if c in self.labels[k] or c in ("own_opening",
                                                                         "partner_opening")]
            data = (f"{len(rows)} training cases (more are added as you get them)"
                    if include_training else "no data yet")
            return (f"Model '{name}' created{replaced} with columns {', '.join(cols)}; it has "
                    f"{data}. Filled in automatically: {', '.join(auto) or 'none'}.")

    def _row(self, k: str, name: str, values: dict) -> tuple[dict | None, str]:
        q = self.q[k]
        m = self.models[k].get(name)
        if m is None:
            return None, f"Error: no model named '{name}' (create it with new_model)."
        auto = self._auto_values(k, q)
        row, missing = {}, []
        for c in m["columns"]:
            v = eval_expr(c, auto) if self.models_in_training else None
            if c in auto:
                row[c] = auto[c]
            elif v is not None:
                row[c] = v
            elif c in values:
                try:
                    row[c] = float(values[c])
                except (TypeError, ValueError):
                    return None, f"Error: the value of {c} must be a number."
            else:
                missing.append(c)
        if missing:
            return None, f"Error: give values for {', '.join(missing)}."
        return row, ""

    def add_case(self, k: str, name: str, values: dict) -> str:
        with self.cond:
            if err := self._in_query(k):
                return err
            row, err = self._row(k, name, values or {})
            if row is None:
                return err
            m = self.models[k][name]
            m["pending"][self.q[k]] = row
            return (f"Case {self.q[k] + 1} added to '{name}' ({len(m['rows'])} cases with y so "
                    "far); its y will be added when it is revealed.")

    def fit_model(self, k: str, name: str, values: dict) -> str:
        with self.cond:
            m = self.models[k].get(name)
            if m is None:
                return f"Error: no model named '{name}' (create it with new_model)."
            current = None
            if self.phase[k] == "query" and 0 <= self.q[k] < self.T and not self.resolved[self.q[k]]:
                row, _ = self._row(k, name, values or {})
                current = row
            elif self.phase[k] == "train":
                return regress(m["rows"], m["columns"], None, rows=f"cases in '{name}'")
            return regress(m["rows"], m["columns"], current, rows=f"cases in '{name}'")

    def _attach_y(self, q: int) -> None:
        """Move each model's pending row for query q into its data, with the revealed y."""
        for k in "AB":
            for m in self.models[k].values():
                row = m["pending"].pop(q, None)
                if row is not None:
                    m["rows"].append({**row, "y": float(self.y[q])})


def build_tools(api: ConsensusAPI, k: str, intro: str, model_tools: bool = False):
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
    if model_tools:
        val_schema = {"type": "object", "additionalProperties": {"type": "number"}}

        @tool("new_model", TOOL_DOCS["new_model"],
              {**obj, "properties": {"name": {"type": "string"},
                                     "columns": {"type": "array", "items": {"type": "string"}},
                                     "include_training": {"type": "boolean"}},
               "required": ["name", "columns"]})
        async def new_model(args):
            return await run("new_model", args, lambda: api.new_model(
                k, str(args.get("name", "")), list(args.get("columns", [])),
                args.get("include_training")))

        @tool("add_case", TOOL_DOCS["add_case"],
              {**obj, "properties": {"name": {"type": "string"}, "values": val_schema},
               "required": ["name"]})
        async def add_case(args):
            return await run("add_case", args, lambda: api.add_case(
                k, str(args.get("name", "")), dict(args.get("values") or {})))

        @tool("fit_model", TOOL_DOCS["fit_model"],
              {**obj, "properties": {"name": {"type": "string"}, "values": val_schema},
               "required": ["name"]})
        async def fit_model(args):
            return await run("fit_model", args, lambda: api.fit_model(
                k, str(args.get("name", "")), dict(args.get("values") or {})))

        tools += [new_model, add_case, fit_model]
    if api.no_training:
        tools = [t_ for t_ in tools if t_.name not in ("get_training_data", "start_queries")]
    server = create_sdk_mcp_server("data", tools=tools)
    return server, [f"mcp__data__{t.name}" for t in tools], tools


def intro_text(api: ConsensusAPI, k: str, structure_hint: bool, partner_info: bool,
               same_y: bool = False, model_tools: bool = False) -> str:
    """The start_queries result for twin k (`{n_train}` is filled in at the call)."""
    labels, other = api.labels[k], api.labels[api.other(k)]
    shared = [lab for lab in labels if lab in other]
    fmt = dict(own=", ".join(labels), other=", ".join(other))
    parts = [
        OVERLAP_FEATURE_LINE.format(shared=" and ".join(shared), **fmt)
        if shared
        else NO_OVERLAP_FEATURE_LINE.format(**fmt)
    ]
    if same_y:
        parts.append(SAME_Y_LINE)
    if partner_info:
        parts.append(PARTNER_INFO_LINE)
    if structure_hint:
        parts.append(STRUCTURE_HINT)
    rule = {"average": RULE_AVERAGE, "individual": RULE_INDIVIDUAL,
            "consensus": RULE_CONSENSUS}[api.team_rule].format(
        tol=f"{api.tol:g}", max_attempts=api.max_attempts, penalty=f"{api.penalty:g}"
    )
    opening = (OPENING_UNTRAINED.format(other=api.other(k)) if api.no_training
               else OPENING_TRAINED.format(n_train="{n_train}", other=api.other(k)))
    return START_QUERIES.format(
        opening=opening,
        rule_par=rule,
        other=api.other(k),
        partner_par=" ".join(parts),
        T=api.T,
        tol=f"{api.tol:g}",
        max_attempts=api.max_attempts,
        penalty=f"{api.penalty:g}",
        cap=api.cap,
        model_par=(MODEL_PAR_TRAINED if api.models_in_training else MODEL_PAR)
        if model_tools else "",
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
        same_y: bool = False,
        model_tools: bool = False,
        max_budget_usd: float = 5.0,
        max_continues: int = 10,
        live_dir: str | Path | None = None,
        run: int = 0,
    ) -> None:
        self.name, self.api, self.model, self.effort = name, api, model, effort
        self.max_budget_usd, self.max_continues = max_budget_usd, max_continues
        self.intro = intro_text(api, name, structure_hint, partner_info, same_y, model_tools)
        self.model_tools = model_tools
        labels = api.labels[name]
        self.system = SYSTEM_TRAIN.format(
            name=name, d=len(labels), names=", ".join(labels), budget=api.budget[name],
            study_rule=(
                f"You must study all {api.budget[name]} training cases before the next phase. "
                if api.study_all else ""
            ),
            model_train_par=(
                MODEL_TRAIN_PAR.format(ex1=labels[0], ex2=labels[1 % len(labels)],
                                       ex3=labels[-1])
                if (model_tools and api.models_in_training) else ""
            ),
        )
        if api.no_training:
            self.system = SYSTEM_UNTRAINED.format(
                name=name, d=len(labels), names=", ".join(labels), intro=self.intro)
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

        server, allowed, _ = build_tools(self.api, self.name, self.intro, self.model_tools)

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
        prompt = ("Begin." if self.api.no_training
                  else "Begin. You are in the training phase.")
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
