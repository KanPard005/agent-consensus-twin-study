"""Boolean task with live twin sessions (LogBook note 2026-11-18, "Brainstorming: boolean
experiments").

Each twin sees its own binary features of a case and predicts a binary label y, a noisy
propositional statement of all the features (`consensus.boolean_data`). A run has three phases:

1. study: the twin reads `budget` labelled cases of its own features through the data API;
2. solo: K queries answered alone (probability p that y = 1 and a label; y revealed after each);
3. partner: T queries with the partner. Per query: next_query (both twins on the same case),
   submit_opening(p, label) before any talk, free talk (send_message / read_messages), then
   submit_prediction(p, label), the twin's own final; both finals and y are then revealed.
   Each twin is scored on its own final (log loss of p, and whether the label is correct).

There is no fitting tool (Kanad, 2026-11-18): the data API only stores and shows cases and counts
rows that satisfy a logical condition. Datasets are per twin: "training" holds the labelled cases
read so far; new_dataset starts another one with copies of training cases and past queries;
add_case adds the current or a past query, with any extra columns the twin gives (such as features
the partner sent). Queries are numbered 1..K (solo) and K+1..K+T (partner).

New features (Kanad, 2026-11-19, note "Boolean experiments: further extensions"): a twin can study
fewer features than it later sees (`study_labels`). The others appear at the solo queries
(`new_at = "solo"`) or only at the partner queries (`new_at = "partner"`, the order control), and
are announced in that phase's intro with the wording the partner intro uses for the partner
("may have information that ... do not"). Training rows have no value for them.

Briefing (Kanad, 2026-11-19, XOR follow-up): with `briefing`, the twins can talk freely once
after the solo queries and before the first partner query, about each other's study and solo
queries (what each sees and has learned). The conversation uses send_message / read_messages
with no case and no opening; it ends as soon as either twin calls next_query. With
`briefing_announced` the system prompt says from the start that there will be a partner and
this conversation (otherwise the twin learns of both only at the partner intro, as before).
Prompts of runs without `briefing` are unchanged.

The waiting logic of the partner phase is that of `consensus.sessions.ConsensusAPI` with the
individual team rule; the session loop is `consensus.sessions.TwinSession`.
"""

from __future__ import annotations

import ast
import asyncio
import operator
import threading
import time
from pathlib import Path

import numpy as np

from consensus.boolean_data import log_loss
from consensus.sessions import TwinSession

SYSTEM = """\
You are agent {name}. Your task is to predict a binary label y (0 or 1) for each case. For each \
case you see {d} binary features ({names}), each 0 or 1. The label is a noisy result of a \
propositional statement (a logical formula) on binary features of the case: y is the value of the \
statement, except that it is sometimes flipped.

You will first see {budget} labelled cases (their features and y) through a data API; study them \
to learn how y depends on your features. You must study all {budget} labelled cases before the \
queries start.

Data API:
- get_training_data(n): the next n labelled cases (n from 1 to {page}); they are also added to \
your dataset "training".
- start_queries(): finish studying and start the queries; you cannot get more labelled cases \
after that, and its result explains what comes next.
- list_datasets(): the names, columns and sizes of your datasets.
- show_dataset(name): every row of one of your datasets.
- count(name, condition): how many rows of a dataset satisfy a condition, and how many of those \
have y = 1 and y = 0. The condition is a logical expression of the dataset's columns, for example \
`{example}`; an empty condition counts every row.
- new_dataset(name, include_training, queries): starts a new dataset, with copies of your \
labelled cases if include_training is true and of the past queries whose numbers you list.
- add_case(name, query, values): during the queries, adds a case to one of your datasets: the \
current case (its y is added once it is revealed), or a past query given by its number (its y is \
added at once). Your features are filled in automatically; values gives the values of any other \
columns you want the row to have.
Your datasets stay available during the queries.

Keep working with these tools and study the labelled cases. When you have finished studying, \
call start_queries. Keep working with the tools until a tool result says that the run is over; \
do not stop before that."""

# Announcement of the features a twin did not study (`new_at`): parallel to the partner intro's
# "Your partner may have information that you do not" (Kanad, 2026-11-19), so the two sources are
# introduced in the same words.
NEW_FEATURES = """ From now on each case also has the features {new}, which the labelled cases \
you studied did not have: your features are now {own}. The new features may have information \
that your other features do not."""

# Briefing (Kanad, 2026-11-19): the purpose of the conversation is stated in Kanad's terms ("to
# understand their individual training and solo query phases, along with what they see and
# learn") and nothing else: no hint about what to share or how to combine.
BRIEF_ANNOUNCE = """

After studying you will answer {K} queries on your own. Then you will meet a partner: another \
instance of the same AI model as you, which has its own features and also studies labelled cases \
and answers queries on its own. Before you start answering queries together, you and your partner \
will be able to talk freely once, to understand each other's study and solo queries: what each of \
you sees and what each of you has learned."""

BRIEF_INTRO = """

Before the queries start, you and your partner can talk freely once, to understand each other's \
study and solo queries: what each of you sees and what each of you has learned. send_message \
sends a message, and read_messages shows your partner's new messages (it waits until your partner \
sends something new, ends the conversation, or is also waiting). The conversation ends as soon as \
one of you calls next_query, which starts the queries."""

BRIEF_START = ("Start the conversation with send_message or read_messages, and call next_query "
               "when you are done talking. You can make at most {cap} tool calls in this "
               "conversation, and at most {cap} per query.")

# Minimal "other agent" framing (Kanad, 2026-11-19): after the solo queries the twin is told only
# that there is another agent "which might or might not have information relevant to predicting
# y", that it also predicts y and is scored on its own, and that they can communicate freely. It
# is not told that the agent is the same model, which features it sees, or that the two work on
# the same case (they do; the waiting in next_query is described without saying why). Nothing
# says what to share or how to use it.
OTHER_INTRO = """\
You have finished studying ({n_train} labelled cases) and answering {K} queries on your own. From \
now on there is another agent, agent {other}, which might or might not have information relevant \
to predicting y. It also predicts y and is scored on its own predictions. You can communicate \
freely with it.

There are {T} more queries (queries {first} to {last}). For each one:
- next_query gives you the features of the case (it waits until the other agent has also asked);
- submit_opening(p, label) records your own probability that y = 1 and your label before you \
communicate;
- then you can communicate freely with the other agent: send_message sends a message, and \
read_messages shows the other agent's opening and new messages (it waits until the other agent \
sends something new, submits a prediction, or is also waiting);
- when you are ready, submit_prediction(p, label) records your own final probability and label \
(once per query). It waits until the other agent has also submitted its final; then you see both \
finals and the true y of your case.
You are scored on each query by {scoring_final}.

At any time during the queries you can also use the data API. You can make at most {cap} tool \
calls per query. Call next_query to begin."""

# The opening round in the minimal framing, in Kanad's terms: "a free communication round first to
# gather information and establish how to communicate".
OTHER_BRIEF_INTRO = """

Before the queries start, you and the other agent have one free communication round, to gather \
information and to establish how you will communicate. send_message sends a message, and \
read_messages shows the other agent's new messages (it waits until the other agent sends \
something new, ends the round, or is also waiting). The round ends as soon as one of you calls \
next_query, which starts the queries."""

OTHER_BRIEF_START = ("Start the round with send_message or read_messages, and call next_query "
                     "when you are done communicating. You can make at most {cap} tool calls in "
                     "this round, and at most {cap} per query.")

OTHER_DOCS = {
    "submit_opening": ("Queries with the other agent: record your own probability p that y = 1 "
                       "and your label (0 or 1) before you communicate."),
    "send_message": "Send a message to the other agent.",
    "read_messages": ("Show the other agent's new messages (and, during a query, its opening); "
                      "waits until there is something new."),
    "submit_prediction": ("Record your probability p that y = 1 and your label (0 or 1) for the "
                          "current case (your final, during the queries with the other agent)."),
}

CONTINUE_OTHER = ("The run is not over yet. Continue working through the tools (next_query, "
                  "submit_opening, communicating with the other agent, submit_prediction) until a "
                  "tool result says the run is over.")

# Told in the partner intro when notifications are on (inserted before "At any time ...").
NOTIFY_INTRO = ("\n\nWhen there are new messages from {who} that you have not read, the results "
                "of your tool calls say so; read_messages shows them, including messages sent "
                "during an earlier query.")

# Wording of the tool results: the default ("partner", unchanged from earlier runs) and the
# minimal framing, where the twin only knows there is "the other agent".
WORDS = {
    "partner": {"Partner": "Partner", "Your partner": "Your partner",
                "your partner": "your partner", "pq": "partner query", "conv": "conversation"},
    "other": {"Partner": "Other agent", "Your partner": "The other agent",
              "your partner": "the other agent", "pq": "query with the other agent",
              "conv": "round"},
}

SCORING = """\
the log loss of your probability (minus the natural log of the probability you gave to the true \
y, with p clipped to [0.001, 0.999]) and by whether your label is correct"""

# No training (Kanad, 2026-11-19): no study rows and no solo queries. The twins meet at query 1,
# and learn only from the revealed labels. The text is the partner-phase text with the study and
# solo parts removed, and the data API without get_training_data and start_queries.
SYSTEM_UNTRAINED = """\
You are agent {name}. Your task is to predict a binary label y (0 or 1) for each case. For each \
case you see {d} binary features ({names}), each 0 or 1. The label is a noisy result of a \
propositional statement (a logical formula) on binary features of the case: y is the value of the \
statement, except that it is sometimes flipped.

There is no study phase: you start without any labelled cases, and the true y of each query is \
revealed after it. You have a partner, agent {other}: another instance of the same AI model as \
you, with the same instructions, which also starts without any labelled cases. Your features are \
{own}; your partner's features are {other_feats}: your features are different, but you both \
predict the same y for each case. Your partner may have information that you do not.

There are {T} queries (queries 1 to {T}). For each one:
- next_query gives you the features of the case (it waits until your partner has also asked, so \
you both work on the same case);
- submit_opening(p, label) records your own probability that y = 1 and your label before you \
talk;
- then you and your partner can communicate freely: send_message sends a message, and \
read_messages shows your partner's opening and new messages (it waits until your partner sends \
something new, submits a prediction, or is also waiting);
- when you are ready, submit_prediction(p, label) records your own final probability and label \
(once per query). It waits until your partner has also submitted its final; then both finals and \
the true y are revealed to both of you.
You are scored on each query by {scoring_final}; your partner is scored by its own.

Data API:
- list_datasets(): the names, columns and sizes of your datasets.
- show_dataset(name): every row of one of your datasets.
- count(name, condition): how many rows of a dataset satisfy a condition, and how many of those \
have y = 1 and y = 0. The condition is a logical expression of the dataset's columns, for example \
`{example}`; an empty condition counts every row.
- new_dataset(name, include_training, queries): starts a new dataset, with copies of the past \
queries whose numbers you list.
- add_case(name, query, values): adds a case to one of your datasets: the current case (its y is \
added once it is revealed), or a past query given by its number (its y is added at once). Your \
features are filled in automatically; values gives the values of any other columns you want the \
row to have.

At any time during the queries you can also:
- use the data API: add_case can add the current or a past query to any of your datasets, with \
values you give for other columns (such as features your partner sends you);
- communicate any information to your partner, such as your features or your predictions;
- make use of your partner's features in any way you want.
You can make at most {cap} tool calls per query. Call next_query to begin, and keep working with \
the tools until a tool result says that the run is over; do not stop before that."""

SOLO_INTRO = """\
You have finished studying ({n_train} labelled cases).{new_features} Next you will answer {K} \
queries (queries 1 to {K}) on your own. For each one:
- next_query gives you the features of the case;
- submit_prediction(p, label) records your probability p that y = 1 and the label (0 or 1) you \
predict; then the true y is revealed to you.
You are scored on each query by {scoring}.

At any time during these queries you can also use the data API (add_case can add the current or a \
past query, with its revealed y, to any of your datasets). After the {K} queries you will get \
further instructions. You can make at most {cap} tool calls per query. Call next_query to begin."""

PARTNER_INTRO = """\
You have finished studying ({n_train} labelled cases) and answering {K} queries on your own.\
{new_features} From now on you have a partner, agent {other}: another instance of the same AI model as you, with the \
same instructions, which also studied its own cases and answered {K} queries on its own. Your \
features are {own}; your partner's features are {other_feats}: your features are different, but \
you both predict the same y for each case. Your partner may have information that you do not.

There are {T} more queries (queries {first} to {last}). For each one:
- next_query gives you the features of the case (it waits until your partner has also asked, so \
you both work on the same case);
- submit_opening(p, label) records your own probability that y = 1 and your label before you \
talk;
- then you and your partner can communicate freely: send_message sends a message, and \
read_messages shows your partner's opening and new messages (it waits until your partner sends \
something new, submits a prediction, or is also waiting);
- when you are ready, submit_prediction(p, label) records your own final probability and label \
(once per query). It waits until your partner has also submitted its final; then both finals and \
the true y are revealed to both of you.
You are scored on each query by {scoring_final}; your partner is scored by its own.

At any time during the queries you can also:
- use the data API: add_case can add the current or a past query to any of your datasets, with \
values you give for other columns (such as features your partner sends you);
- communicate any information to your partner, such as your features or your predictions;
- make use of your partner's features in any way you want.
You can make at most {cap} tool calls per query. Call next_query to begin."""

# Single-agent baseline (Kanad, 2026-11-18): one agent sees every feature and goes through the
# same three phases alone: the study cases, the first K1 queries, then the remaining queries (the
# cases the pairs answered together), announced the same way the partner phase is but with no
# partner. Nothing else in the prompts changes.
SINGLE_INTRO = """\
You have finished studying ({n_train} labelled cases). Next you will answer {K1} queries \
(queries 1 to {K1}). For each one:
- next_query gives you the features of the case;
- submit_prediction(p, label) records your probability p that y = 1 and the label (0 or 1) you \
predict; then the true y is revealed to you.
You are scored on each query by {scoring}.

At any time during these queries you can also use the data API (add_case can add the current or a \
past query, with its revealed y, to any of your datasets). After the {K1} queries you will get \
further instructions. You can make at most {cap} tool calls per query. Call next_query to begin."""

SINGLE_BREAK = """\
That was the last of the first {K1} queries.

There are {T} more queries (queries {first} to {last}). They work in the same way: next_query \
gives you the features of the case, and submit_prediction(p, label) records your probability that \
y = 1 and your label; then the true y is revealed to you. You are scored on each query in the same \
way. At any time you can also use the data API. You can make at most {cap} tool calls per query. \
Call next_query to continue."""

CONTINUE_SINGLE = ("The run is not over yet. Continue working through the tools (next_query, "
                   "submit_prediction) until a tool result says the run is over.")

CONTINUE = ("The run is not over yet. Continue working through the tools (next_query, "
            "submit_opening, talking with your partner, submit_prediction) until a tool result "
            "says the run is over.")

TOOL_DOCS = {
    "get_training_data": "Get the next n labelled cases (n from 1 to {page}).",
    "start_queries": "Finish studying the labelled cases and start the queries.",
    "list_datasets": "The names, columns and sizes of your datasets.",
    "show_dataset": "Show every row of one of your datasets.",
    "count": ("Count the rows of a dataset that satisfy a logical condition of its columns "
              "(and/or/not, ==, !=, <, >; an empty condition counts every row), split by y."),
    "new_dataset": ("Start a new dataset; include_training copies your labelled cases into it, "
                    "queries copies the past queries with those numbers."),
    "add_case": ("Add the current case (its y is added once revealed) or a past query (by number "
                 "in `query`) to one of your datasets; give the values of other columns in "
                 "`values`."),
    "next_query": "Get the features of the next case.",
    "submit_opening": ("Partner queries: record your own probability p that y = 1 and your label "
                       "(0 or 1) before you talk."),
    "send_message": "Partner queries: send a message to your partner.",
    "read_messages": ("Partner queries: show your partner's opening and new messages (waits until "
                      "there is something new)."),
    "submit_prediction": ("Record your probability p that y = 1 and your label (0 or 1) for the "
                          "current case (your final, during the partner queries)."),
}

# ---- conditions for `count` ----------------------------------------------------------------
_CMP = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le,
        ast.Gt: operator.gt, ast.GtE: operator.ge}


class ConditionError(ValueError):
    """The condition is not a logical expression of known columns."""


def check_condition(expr: str) -> ast.Expression:
    """Parse a condition and check that it only uses and/or/not, comparisons, column names and
    numbers (no calls or attributes): it is evaluated on the twin's rows, so nothing else may
    run."""
    try:
        tree = ast.parse(expr.strip() or "True", mode="eval")
    except SyntaxError as e:
        raise ConditionError(f"cannot parse the condition: {e.msg}") from None
    for node in ast.walk(tree):
        ok = isinstance(node, (ast.Expression, ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not,
                               ast.Compare, ast.Name, ast.Load, ast.Constant, *_CMP))
        if not ok or isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise ConditionError("use only column names, numbers, and/or/not and comparisons")
    return tree


def eval_condition(tree: ast.Expression, row: dict) -> bool:
    """Truth value of a checked condition on one row; a bare column name means column == 1.
    Raises KeyError if the row has no value for a column the condition uses."""

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.BoolOp):
            vals = (bool(ev(v)) for v in node.values)
            return all(vals) if isinstance(node.op, ast.And) else any(vals)
        if isinstance(node, ast.UnaryOp):
            return not bool(ev(node.operand))
        if isinstance(node, ast.Compare):
            left = ev(node.left)
            for op, right in zip(node.ops, node.comparators):
                r = ev(right)
                if not _CMP[type(op)](left, r):
                    return False
                left = r
            return True
        if isinstance(node, ast.Name):
            if node.id in ("True", "False"):
                return node.id == "True"
            return float(row[node.id])
        if isinstance(node, ast.Constant):
            return node.value
        raise ConditionError("bad condition")

    v = ev(tree)
    return bool(v == 1) if isinstance(v, float) else bool(v)


def _fmt(v) -> str:
    v = float(v)
    return str(int(v)) if v.is_integer() else f"{v:g}"


class BooleanAPI:
    """Shared, thread-safe data API of the boolean task (see the module docstring).

    Parameters
    ----------
    train : {"A": (X_A, y), "B": (X_B, y)}
        Labelled study cases per twin; X_k of shape (budget, |S_k|), y of shape (budget,).
    solo : {"x": {"A": X_A, "B": X_B}, "y": y}
        K solo queries (the same cases for both twins, each seeing its own features).
    queries : {"A": X_A, "B": X_B}; y : (T,)
        The partner-phase query stream (the same case for both twins).
    labels : {"A": [...], "B": [...]}
        Feature names per twin (all its features, as in the partner queries).
    study_labels : {"A": [...], "B": [...]}, optional
        Names of the features twin k studies (a prefix of labels[k]; the train X_k has these
        columns). Default: every feature.
    new_at : {"solo", "partner"}
        Where the features not studied first appear (only matters with study_labels); solo X_k
        has the study columns if "partner".
    info : dict of per-query arrays, optional
        Extra values stored in every record (x, pi, clause values, Bayes probabilities), keyed
        "solo" / "partner"; not shown to the twins.
    """

    def __init__(self, train: dict, solo: dict, queries: dict, y: np.ndarray, labels: dict,
                 page_size: int = 40, cap: int = 40, max_chars: int = 2000,
                 info: dict | None = None, single: int | None = None,
                 no_training: bool = False, study_labels: dict | None = None,
                 new_at: str = "solo", briefing: bool = False,
                 briefing_announced: bool = False, other_agent: bool = False,
                 notify_messages: bool = False) -> None:
        # Single-agent baseline: `single` = K1, the size of the first block of the K queries,
        # which all run in the solo phase; after query K1 the agent gets `single_break`, after
        # query K the run is over. Only twin A acts.
        self.single = single
        self.single_break = ""  # set by the session (`prompts`)
        self.cond = threading.Condition()
        self.labels = labels
        self.study_labels = study_labels or labels
        for k in "AB":
            if list(self.study_labels[k]) != list(labels[k][: len(self.study_labels[k])]):
                raise ValueError("study_labels[k] must be a prefix of labels[k]")
        if new_at not in ("solo", "partner"):
            raise ValueError(f"new_at must be 'solo' or 'partner', got {new_at!r}")
        self.new_at = new_at
        self.solo_labels = labels if new_at == "solo" else self.study_labels
        self.Xtr = {k: np.asarray(train[k][0]) for k in "AB"}
        self.ytr = {k: np.asarray(train[k][1]) for k in "AB"}
        self.budget = {k: len(self.ytr[k]) for k in "AB"}
        self.solo_x, self.solo_y = solo["x"], np.asarray(solo["y"])
        self.K = len(self.solo_y)
        self.qx, self.y = queries, np.asarray(y)
        self.T = len(self.y)
        self.page_size, self.cap, self.max_chars = int(page_size), int(cap), int(max_chars)
        self.info = info or {}
        # No training: no study rows, no solo queries; both twins start in the partner phase.
        self.no_training = bool(no_training)  # also read by TwinSession
        if self.no_training and (self.K or any(self.budget.values()) or single):
            raise ValueError("no_training needs no study rows, no solo queries and no single agent")
        # Briefing: phase "brief" between the solo queries and the first partner query; it is
        # over for both twins once either calls next_query (`brief_over`).
        self.briefing = bool(briefing)
        self.briefing_announced = bool(briefing_announced)
        if self.briefing_announced and not self.briefing:
            raise ValueError("briefing_announced needs briefing")
        if self.briefing and (self.no_training or single):
            raise ValueError("briefing needs a pair with a study phase")
        self.brief_over = False
        # Minimal "other agent" framing (Kanad, 2026-11-19): the partner is "the other agent",
        # and the tool results use that wording too (`WORDS`).
        self.other_agent = bool(other_agent)
        if self.other_agent and (self.no_training or single or self.briefing_announced):
            raise ValueError("other_agent needs a pair with a study phase, without "
                             "briefing_announced")
        self.w = WORDS["other" if self.other_agent else "partner"]
        # Message notifications (Kanad, 2026-11-19): every tool result says when there are
        # unread messages, and no message is lost when a query ends; the read pointer runs over
        # the whole inbox instead of restarting at each query.
        self.notify = bool(notify_messages)
        self.phase = {k: "query" if self.no_training else "train" for k in "AB"}
        self.n_seen = {k: 0 for k in "AB"}
        self.solo_i = {k: -1 for k in "AB"}
        self.solo_pred = {k: np.full((self.K, 2), np.nan) for k in "AB"}  # (p, label)
        self.q = {k: -1 for k in "AB"}
        self.want = {k: -1 for k in "AB"}
        self.calls = {k: {} for k in "AB"}
        self.opening = {k: np.full((self.T, 2), np.nan) for k in "AB"}
        self.final = {k: np.full((self.T, 2), np.nan) for k in "AB"}
        self.resolved = np.zeros(self.T, bool)
        self.inbox = {k: [] for k in "AB"}
        self.n_read = {k: 0 for k in "AB"}
        self.waiting = {k: None for k in "AB"}
        self.messages: list[dict] = []
        # Datasets per twin: {name: {"rows": [row with y], "pending": {key: row}}}; a pending
        # row gets its y when the query it came from is revealed.
        self.datasets = {k: {"training": {"rows": [], "pending": {}}} for k in "AB"}
        self.solo_intro = {k: "" for k in "AB"}  # set by the sessions (`prompts`)
        self.partner_intro = {k: "" for k in "AB"}
        self.on_query = self.on_solo = self.on_call = None

    @staticmethod
    def other(k: str) -> str:
        return "B" if k == "A" else "A"

    def done(self, k: str) -> bool:
        return self.phase[k] == "done"

    def finish(self, k: str) -> None:
        with self.cond:
            self.phase[k] = "done"
            self.cond.notify_all()

    # ---- bookkeeping ---------------------------------------------------------------------
    def number(self, k: str) -> int:
        """The displayed number of twin k's current query (solo 1..K, partner K+1..K+T), or 0."""
        if self.phase[k] == "solo":
            return self.solo_i[k] + 1
        return self.K + self.q[k] + 1 if self.q[k] >= 0 else 0

    def call(self, k: str, tool: str, args: dict, fn) -> str:
        """Run one tool call for twin k with the per-query cap, and report it."""
        key, phase = self.number(k), self.phase[k]  # before the call (it can change the phase)
        if phase == "brief":
            key = -1  # the briefing conversation has its own count
        with self.cond:
            n = self.calls[k].get(key, 0) + 1
            self.calls[k][key] = n
        if key == -1 and n > self.cap and tool != "next_query":
            res = ("Tool-call limit for the conversation reached: call next_query to start the "
                   "queries.")
        elif key > 0 and n > self.cap and tool not in ("submit_prediction", "next_query"):
            res = (f"Tool-call limit for query {key} reached: call submit_prediction now (or "
                   "next_query if the query is resolved).")
        else:
            res = fn()
        if tool != "read_messages":
            res += self.unread_note(k)
        if self.on_call is not None:
            self.on_call(k, {"phase": phase, "query": key, "tool": tool, "args": args,
                             "result": res[:3000], "time": time.time()})
        return res

    def new_features(self, k: str) -> list[str]:
        """Twin k's features that it did not study."""
        return list(self.labels[k][len(self.study_labels[k]):])

    def labels_now(self, k: str) -> list[str]:
        """The names of the features twin k sees in its current phase."""
        return {"train": self.study_labels, "solo": self.solo_labels}.get(
            self.phase[k], self.labels)[k]

    @staticmethod
    def _own_row(names: list[str], x: np.ndarray) -> dict:
        if len(names) != len(x):
            raise ValueError("feature names and values differ in length")
        return {lab: int(v) for lab, v in zip(names, x)}

    def _query_case(self, k: str, num: int) -> tuple[dict, int | None]:
        """(own features, y or None if not revealed to twin k) of query number num."""
        if 1 <= num <= self.K:
            i = num - 1
            y = int(self.solo_y[i]) if not np.isnan(self.solo_pred[k][i, 0]) else None
            return self._own_row(self.solo_labels[k], self.solo_x[k][i]), y
        q = num - self.K - 1
        if 0 <= q < self.T:
            return self._own_row(self.labels[k], self.qx[k][q]), int(self.y[q]) if self.resolved[q] else None
        raise IndexError(num)

    def _past_queries(self, k: str) -> int:
        """Number of the last query whose y twin k has seen."""
        if self.phase[k] == "solo":
            i = self.solo_i[k]
            return i + 1 if i >= 0 and not np.isnan(self.solo_pred[k][i, 0]) else i
        if self.phase[k] in ("brief", "query", "done"):
            q = self.q[k]
            if q < 0:  # solo queries over, no partner query yet
                return self.K
            return self.K + q + (1 if self.resolved[q] else 0)
        return 0

    def solo_record(self, k: str, i: int) -> dict:
        p, lab = self.solo_pred[k][i]
        rec = {"twin": k, "i": i, "number": i + 1, "x": [int(v) for v in self.solo_x[k][i]],
               "p": float(p), "label": float(lab), "y": int(self.solo_y[i])}
        for name, arr in self.info.get("solo", {}).items():
            rec[name] = _jsonable(arr[i] if not isinstance(arr, dict) else arr[k][i])
        return rec

    def record(self, q: int) -> dict:
        """Everything about partner query q (for the live file and the analysis)."""
        rec = {
            "t": q, "number": self.K + q + 1, "y": int(self.y[q]),
            "x": {k: [int(v) for v in self.qx[k][q]] for k in "AB"},
            "opening": {k: [float(v) for v in self.opening[k][q]] for k in "AB"},
            "final": {k: [float(v) for v in self.final[k][q]] for k in "AB"},
            "messages": [m for m in self.messages if m["q"] == q],
            "datasets": {k: {n: len(d["rows"]) for n, d in self.datasets[k].items()}
                         for k in "AB"},
        }
        for name, arr in self.info.get("partner", {}).items():
            rec[name] = _jsonable({k: v[q] for k, v in arr.items()} if isinstance(arr, dict)
                                  else arr[q])
        return rec

    # ---- study phase ----------------------------------------------------------------------
    def get_training_data(self, k: str, n: int) -> str:
        with self.cond:
            if self.phase[k] != "train":
                return "Error: you have finished studying; no more labelled cases."
            n = int(min(max(int(n), 1), self.page_size, self.budget[k] - self.n_seen[k]))
            if n <= 0:
                return f"You have seen all {self.budget[k]} labelled cases."
            a, b = self.n_seen[k], self.n_seen[k] + n
            self.n_seen[k] = b
            rows = []
            for j in range(a, b):
                row = {**self._own_row(self.study_labels[k], self.Xtr[k][j]),
                       "y": int(self.ytr[k][j])}
                self.datasets[k]["training"]["rows"].append(row)
                rows.append(", ".join(str(v) for v in row.values()))
            head = ", ".join(self.study_labels[k]) + ", y"
            return (f"Labelled cases {a + 1}-{b} of {self.budget[k]} ({head}):\n"
                    + "\n".join(rows))

    def start_queries(self, k: str) -> str:
        with self.cond:
            if self.phase[k] != "train":
                return "Error: the queries have already started."
            if self.n_seen[k] < self.budget[k]:
                return (f"Error: study all {self.budget[k]} labelled cases first (you have seen "
                        f"{self.n_seen[k]}).")
            if self.K > 0:
                self.phase[k] = "solo"
                return self.solo_intro[k].format(n_train=self.n_seen[k])
            self.phase[k] = "brief" if self.briefing else "query"
            self.cond.notify_all()
            return self.partner_intro[k].format(n_train=self.n_seen[k])

    # ---- datasets ---------------------------------------------------------------------------
    def list_datasets(self, k: str) -> str:
        with self.cond:
            lines = []
            for name, d in self.datasets[k].items():
                cols = _columns(d["rows"] + list(d["pending"].values()), self.labels_now(k))
                lines.append(f"{name}: {len(d['rows'])} rows with y"
                             + (f", {len(d['pending'])} waiting for y" if d["pending"] else "")
                             + f"; columns {', '.join(cols)}")
            return "\n".join(lines)

    def show_dataset(self, k: str, name: str) -> str:
        with self.cond:
            d = self.datasets[k].get(name)
            if d is None:
                return f"Error: no dataset named {name!r}."
            if not d["rows"] and not d["pending"]:
                return f"Dataset {name} is empty."
            cols = _columns(d["rows"] + list(d["pending"].values()), self.labels_now(k))
            out = [f"Dataset {name} ({len(d['rows'])} rows with y; columns {', '.join(cols)}, "
                   "source):"]
            for r in d["rows"]:
                out.append(", ".join(_fmt(r[c]) if c in r else "-" for c in cols)
                           + f"  [{r.get('_src', 'training')}]")
            for r in d["pending"].values():
                out.append(", ".join(_fmt(r[c]) if c in r else "-" for c in cols if c != "y")
                           + f", y not revealed yet  [{r['_src']}]")
            return "\n".join(out)

    def count(self, k: str, name: str, condition: str) -> str:
        with self.cond:
            d = self.datasets[k].get(name)
            if d is None:
                return f"Error: no dataset named {name!r}."
            try:
                tree = check_condition(condition)
            except ConditionError as e:
                return f"Error: {e}."
            n = n1 = skipped = 0
            missing: set[str] = set()
            for r in d["rows"]:
                try:
                    hit = eval_condition(tree, r)
                except KeyError as e:
                    skipped += 1
                    missing.add(str(e.args[0]))
                    continue
                if hit:
                    n += 1
                    n1 += int(r["y"] == 1)
            if skipped == len(d["rows"]) and d["rows"]:
                return (f"Error: no row of {name} has the column(s) {', '.join(sorted(missing))}.")
            res = (f"{name}, rows with y where {condition.strip() or 'any'}: {n} "
                   f"(y = 1: {n1}, y = 0: {n - n1}).")
            if skipped:
                res += (f" {skipped} rows were skipped because they have no value for "
                        f"{', '.join(sorted(missing))}.")
            return res

    def new_dataset(self, k: str, name: str, include_training: bool = False,
                    queries: list[int] | None = None) -> str:
        with self.cond:
            name = str(name).strip()
            if not name:
                return "Error: give the dataset a name."
            if name in self.datasets[k]:
                return f"Error: you already have a dataset named {name!r}."
            rows = []
            if include_training:
                rows += [dict(r) for r in self.datasets[k]["training"]["rows"]
                         if "_src" not in r]  # the labelled cases, not queries added to it
            last = self._past_queries(k)
            for num in queries or []:
                num = int(num)
                if not 1 <= num <= last:
                    return f"Error: query {num} is not a past query with a revealed y."
                own, y = self._query_case(k, num)
                rows.append({**own, "y": y, "_src": f"query {num}"})
            self.datasets[k][name] = {"rows": rows, "pending": {}}
            return f"Dataset {name} created with {len(rows)} rows."

    def add_case(self, k: str, name: str, query: int | None = None,
                 values: dict | None = None) -> str:
        with self.cond:
            d = self.datasets[k].get(name)
            if d is None:
                return f"Error: no dataset named {name!r} (create it with new_dataset)."
            values = values or {}
            if bad := [c for c in values if c in self.labels[k] or c == "y"]:
                return (f"Error: {', '.join(bad)} cannot be given in values (your features are "
                        "filled in automatically and y is the revealed label).")
            try:
                extra = {str(c): float(v) for c, v in values.items()}
            except (TypeError, ValueError):
                return "Error: values must be numbers."
            if query is None:  # the current case: y added when revealed
                num = self.number(k)
                if num == 0 or self._past_queries(k) >= num:
                    return ("Error: there is no current unanswered query; give the number of a "
                            "past query in `query`.")
                own, _ = self._query_case(k, num)
                d["pending"][num] = {**own, **extra, "_src": f"query {num}"}
                return f"Query {num} added to {name}; its y will be added once it is revealed."
            num = int(query)
            if not 1 <= num <= self._past_queries(k):
                return f"Error: query {num} is not a past query with a revealed y."
            own, y = self._query_case(k, num)
            d["rows"].append({**own, **extra, "y": y, "_src": f"query {num}"})
            return f"Query {num} (y = {y}) added to {name}."

    def _reveal(self, k: str, num: int, y: int) -> None:
        """Give y to twin k's rows waiting for query num."""
        for d in self.datasets[k].values():
            row = d["pending"].pop(num, None)
            if row is not None:
                d["rows"].append({**row, "y": int(y)})

    # ---- queries ----------------------------------------------------------------------------
    @staticmethod
    def _check(p, label) -> str | None:
        try:
            p, label = float(p), float(label)
        except (TypeError, ValueError):
            return "Error: give p (a number from 0 to 1) and label (0 or 1)."
        if not 0.0 <= p <= 1.0:
            return "Error: p must be between 0 and 1."
        if label not in (0.0, 1.0):
            return "Error: label must be 0 or 1."
        return None

    def next_query(self, k: str) -> str:
        with self.cond:
            if self.phase[k] == "train":
                return "Error: call start_queries first."
            if self.phase[k] == "done":
                return "The run is over. Stop."
            if self.phase[k] == "brief":  # this twin ends the conversation for both
                self.phase[k] = "query"
                self.brief_over = True
                self.cond.notify_all()
            if self.phase[k] == "solo":
                i = self.solo_i[k]
                if i >= 0 and np.isnan(self.solo_pred[k][i, 0]):
                    return f"Error: submit your prediction for query {i + 1} first."
                self.solo_i[k] = i = i + 1
                feats = ", ".join(f"{lab}={int(v)}" for lab, v in
                                  zip(self.solo_labels[k], self.solo_x[k][i]))
                if self.single:
                    return f"Query {i + 1} of {self.K}. Your features: {feats}."
                return f"Query {i + 1} of {self.K} (on your own). Your features: {feats}."
            if self.q[k] >= 0 and not self.resolved[self.q[k]]:
                return (f"Error: query {self.number(k)} is not resolved yet (submit your final "
                        "first).")
            nq = self.q[k] + 1
            if nq >= self.T:
                self.phase[k] = "done"
                self.cond.notify_all()
                return "There are no more queries: the run is over. Stop."
            self.want[k] = nq
            self.cond.notify_all()
            o = self.other(k)
            self.cond.wait_for(lambda: self.want[o] >= nq or self.q[o] >= nq or self.done(o))
            self.q[k] = nq
            if not self.notify:  # unread messages of an earlier query are dropped
                self.n_read[k] = 0
            feats = ", ".join(f"{lab}={int(v)}" for lab, v in zip(self.labels[k], self.qx[k][nq]))
            return (f"Query {self.K + nq + 1} ({self.w['pq']} {nq + 1} of {self.T}). Your "
                    f"features: {feats}.")

    def _in_query(self, k: str) -> str | None:
        if self.phase[k] == "solo":
            return "Error: while you answer queries on your own, use next_query and submit_prediction."
        if self.phase[k] == "brief":
            return ("Error: the queries have not started yet; call next_query when you are done "
                    "talking.")
        if self.phase[k] != "query" or self.q[k] < 0:
            return "Error: no current query (call next_query)."
        if self.resolved[self.q[k]]:
            return "Error: this query is resolved; call next_query."
        return None

    def submit_opening(self, k: str, p, label) -> str:
        with self.cond:
            if err := self._in_query(k) or self._check(p, label):
                return err
            q = self.q[k]
            if not np.isnan(self.opening[k][q, 0]):
                return "Error: you already submitted your opening for this query."
            self.opening[k][q] = (float(p), float(label))
            self.cond.notify_all()
            return f"Opening p = {float(p):g}, label {int(label)} recorded for query {self.number(k)}."

    # ---- briefing conversation (no case; messages stored with q = -1) ------------------------
    def _brief_send(self, k: str, text: str) -> str:
        if self.brief_over:
            return (f"Error: {self.w['your partner']} has ended the {self.w['conv']}; call "
                    "next_query to start the queries.")
        text = str(text)[: self.max_chars]
        self.inbox[self.other(k)].append((-1, text))
        self.messages.append({"q": -1, "from": k, "text": text, "order": len(self.messages),
                              "time": time.time()})
        self.cond.notify_all()
        return "Message sent."

    def _brief_read(self, k: str) -> str:
        o = self.other(k)
        self.waiting[k] = "read"
        self.cond.notify_all()
        # Wake on a new message, the end of the conversation, or both twins waiting (a twin
        # still in its solo queries cannot be waiting, so a fast twin waits for it to arrive).
        self.cond.wait_for(
            lambda: self._unread(k, -1)
            or self.brief_over
            or (self.waiting[o] == "read" and not self._unread(o, -1))
            or self.done(o)
        )
        self.waiting[k] = None
        new = self._take_unread(k, -1)
        lines = new or ["No new messages."]
        if self.brief_over:
            lines.append(f"{self.w['Your partner']} has ended the {self.w['conv']}; call "
                         "next_query to start the queries.")
        elif not new and self.waiting[o] == "read":
            lines.append(f"{self.w['Your partner']} is also waiting for a message.")
        return "\n".join(lines)

    def send_message(self, k: str, text: str) -> str:
        with self.cond:
            if self.phase[k] == "brief":
                return self._brief_send(k, text)
            if err := self._in_query(k):
                return err
            q = self.q[k]
            if np.isnan(self.opening[k][q, 0]):
                return "Error: submit your opening first."
            text = str(text)[: self.max_chars]
            self.inbox[self.other(k)].append((q, text))
            self.messages.append({"q": q, "from": k, "text": text, "order": len(self.messages),
                                  "time": time.time()})
            self.cond.notify_all()
            return "Message sent."

    def _unread(self, k: str, q: int) -> list[str]:
        return [t for _, t in self._unread_items(k, q)]

    def _unread_items(self, k: str, q: int) -> list[tuple[int, str]]:
        """(query index or -1 for the briefing, text) of twin k's unread messages: those of
        query q, or with notifications every message not yet shown."""
        if self.notify:
            return self.inbox[k][self.n_read[k]:]
        return [(qq, t) for qq, t in self.inbox[k] if qq == q][self.n_read[k]:]

    def _take_unread(self, k: str, q: int) -> list[str]:
        """Mark twin k's unread messages as read and format them; with notifications, a message
        from another query (or the briefing) says when it was sent."""
        items = self._unread_items(k, q)
        self.n_read[k] += len(items)
        out = []
        for qq, t in items:
            when = ("" if qq == q else " (sent before the queries)" if qq < 0
                    else f" (sent during query {self.K + qq + 1})")
            out.append(f'{self.w["Partner"]}{when}: "{t}"')
        return out

    def unread_note(self, k: str) -> str:
        """The notification appended to twin k's tool results while it has unread messages."""
        if not self.notify or self.phase[k] not in ("brief", "query"):
            return ""
        with self.cond:
            n = len(self._unread_items(k, self.q[k]))
        if not n:
            return ""
        return (f"\n\nNotification: {n} new message{'s' if n > 1 else ''} from "
                f"{self.w['your partner']}; read_messages shows "
                f"{'them' if n > 1 else 'it'}.")

    def _submitted(self, k: str, q: int) -> bool:
        return not np.isnan(self.final[k][q, 0])

    def read_messages(self, k: str) -> str:
        with self.cond:
            if self.phase[k] == "brief":
                return self._brief_read(k)
            if err := self._in_query(k):
                return err
            q, o = self.q[k], self.other(k)
            if np.isnan(self.opening[k][q, 0]):
                return "Error: submit your opening first."
            self.waiting[k] = "read"
            self.cond.notify_all()
            self.cond.wait_for(
                lambda: self._unread(k, q)
                or self._submitted(o, q)
                or self.waiting[o] == "submit"
                or (self.waiting[o] == "read" and not self._unread(o, q))
                or self.done(o)
                or self.resolved[q]
            )
            self.waiting[k] = None
            new = self._take_unread(k, q)
            po, pl = self.opening[o][q]
            lines = [f"{self.w['Partner']}'s opening for query {self.number(k)}: "
                     + ("not submitted yet." if np.isnan(po)
                        else f"p = {po:g}, label {int(pl)}.")]
            lines += new or ["No new messages."]
            if self._submitted(o, q):
                lines.append(f"{self.w['Your partner']} has submitted its final and is waiting for "
                             "yours.")
            elif not new and self.waiting[o] == "read":
                lines.append(f"{self.w['Your partner']} is also waiting for a message.")
            return "\n".join(lines)

    def submit_prediction(self, k: str, p, label) -> str:
        with self.cond:
            if err := self._check(p, label):
                return err
            p, label = float(p), float(label)
            if self.phase[k] == "solo":
                return self._solo_submit(k, p, label)
            if err := self._in_query(k):
                return err
            q, o = self.q[k], self.other(k)
            if np.isnan(self.opening[k][q, 0]):
                return "Error: submit your opening first."
            if self._submitted(k, q):
                return "Error: you already submitted your final for this query."
            self.final[k][q] = (p, label)
            self.waiting[k] = "submit"
            self.cond.notify_all()
            self.cond.wait_for(lambda: self._submitted(o, q) or self.done(o) or self.resolved[q])
            self.waiting[k] = None
            if not self.resolved[q]:
                self.resolved[q] = True
                for kk in "AB":
                    self._reveal(kk, self.K + q + 1, int(self.y[q]))
                self.cond.notify_all()
                if self.on_query is not None:
                    self.on_query(self.record(q))
            last = q == self.T - 1
            if last:
                self.phase[k] = "done"
                self.cond.notify_all()
            y = int(self.y[q])
            pf, lf = self.final[o][q]
            partner = (f"{self.w['your partner']} did not submit" if np.isnan(pf)
                       else f"{self.w['your partner']}'s p = {pf:g}, label {int(lf)}")
            res = (f"Both finals are in: yours p = {p:g}, label {int(label)}; {partner}. The true "
                   f"y was {y}. {_score(p, label, y, 'Your')}")
            if not np.isnan(pf):
                res += " " + _score(pf, lf, y, f"{self.w['Your partner']}'s")
            return res + (" That was the last query: the run is over. Stop." if last
                          else " Call next_query for the next case.")

    def _solo_submit(self, k: str, p: float, label: float) -> str:
        i = self.solo_i[k]
        if i < 0 or not np.isnan(self.solo_pred[k][i, 0]):
            return "Error: no current query (call next_query)."
        self.solo_pred[k][i] = (p, label)
        y = int(self.solo_y[i])
        self._reveal(k, i + 1, y)
        if self.on_solo is not None:
            self.on_solo(k, self.solo_record(k, i))
        res = f"The true y was {y}. {_score(p, label, y, 'Your')}"
        if self.single and i == self.single - 1 and i < self.K - 1:
            return res + " " + self.single_break
        if self.single and i == self.K - 1:
            self.phase[k] = "done"
            self.cond.notify_all()
            return res + " That was the last query: the run is over. Stop."
        if i < self.K - 1:
            return res + " Call next_query for the next case."
        self.phase[k] = "brief" if self.briefing else "query"
        self.cond.notify_all()
        return (res + f" That was the last of your {self.K} queries on your own.\n\n"
                + self.partner_intro[k].format(n_train=self.n_seen[k]))


def _score(p: float, label: float, y: int, who: str) -> str:
    ll = float(log_loss(np.array([p]), np.array([y]))[0])
    return (f"{who} log loss {ll:.3f}, label {'correct' if int(label) == y else 'wrong'}.")


def _columns(rows: list[dict], own: list[str]) -> list[str]:
    """Own features first, then other columns in order of appearance, then y."""
    extra: list[str] = []
    for r in rows:
        extra += [c for c in r if c not in own and c != "y" and not c.startswith("_")
                  and c not in extra]
    return list(own) + extra + ["y"]


def _jsonable(v):
    if isinstance(v, dict):
        return {kk: _jsonable(vv) for kk, vv in v.items()}
    a = np.asarray(v)
    return a.tolist() if a.ndim else a.item()


def build_tools(api: BooleanAPI, k: str):
    """The twin's tools as an in-process MCP server (`data`) and their allowed names."""
    from claude_agent_sdk import create_sdk_mcp_server, tool

    docs = {n: d.format(page=api.page_size) for n, d in TOOL_DOCS.items()}
    if api.other_agent:
        docs.update(OTHER_DOCS)
        if api.briefing:
            docs["next_query"] = ("Get the features of the next case (after the communication "
                                  "round, this ends the round).")
    elif api.briefing:  # the talk tools also work in the conversation before the partner queries
        docs["send_message"] = ("Partner phase (the conversation before the partner queries, "
                                "and the partner queries): send a message to your partner.")
        docs["read_messages"] = ("Partner phase (the conversation before the partner queries, "
                                 "and the partner queries): show your partner's new messages "
                                 "(and, during a query, its opening); waits until there is "
                                 "something new.")
        docs["next_query"] = ("Get the features of the next case (after the conversation before "
                              "the partner queries, this ends the conversation).")

    def text(s: str) -> dict:
        return {"content": [{"type": "text", "text": s}]}

    async def run(name: str, args: dict, fn) -> dict:
        return text(await asyncio.to_thread(api.call, k, name, args, fn))

    obj = {"type": "object"}
    pl = {**obj, "properties": {"p": {"type": "number"}, "label": {"type": "integer"}},
          "required": ["p", "label"]}
    name_only = {**obj, "properties": {"name": {"type": "string"}}, "required": ["name"]}

    @tool("get_training_data", docs["get_training_data"],
          {**obj, "properties": {"n": {"type": "integer"}}, "required": ["n"]})
    async def get_training_data(args):
        return await run("get_training_data", args,
                         lambda: api.get_training_data(k, args.get("n", api.page_size)))

    @tool("start_queries", docs["start_queries"], {**obj, "properties": {}})
    async def start_queries(args):
        return await run("start_queries", args, lambda: api.start_queries(k))

    @tool("list_datasets", docs["list_datasets"], {**obj, "properties": {}})
    async def list_datasets(args):
        return await run("list_datasets", args, lambda: api.list_datasets(k))

    @tool("show_dataset", docs["show_dataset"], name_only)
    async def show_dataset(args):
        return await run("show_dataset", args,
                         lambda: api.show_dataset(k, str(args.get("name", ""))))

    @tool("count", docs["count"],
          {**obj, "properties": {"name": {"type": "string"}, "condition": {"type": "string"}},
           "required": ["name"]})
    async def count(args):
        return await run("count", args, lambda: api.count(
            k, str(args.get("name", "")), str(args.get("condition") or "")))

    @tool("new_dataset", docs["new_dataset"],
          {**obj, "properties": {"name": {"type": "string"},
                                 "include_training": {"type": "boolean"},
                                 "queries": {"type": "array", "items": {"type": "integer"}}},
           "required": ["name"]})
    async def new_dataset(args):
        return await run("new_dataset", args, lambda: api.new_dataset(
            k, str(args.get("name", "")), bool(args.get("include_training")),
            list(args.get("queries") or [])))

    @tool("add_case", docs["add_case"],
          {**obj, "properties": {"name": {"type": "string"}, "query": {"type": "integer"},
                                 "values": {"type": "object",
                                            "additionalProperties": {"type": "number"}}},
           "required": ["name"]})
    async def add_case(args):
        q = args.get("query")
        return await run("add_case", args, lambda: api.add_case(
            k, str(args.get("name", "")), None if q is None else int(q),
            dict(args.get("values") or {})))

    @tool("next_query", docs["next_query"], {**obj, "properties": {}})
    async def next_query(args):
        return await run("next_query", args, lambda: api.next_query(k))

    @tool("submit_opening", docs["submit_opening"], pl)
    async def submit_opening(args):
        return await run("submit_opening", args,
                         lambda: api.submit_opening(k, args.get("p"), args.get("label")))

    @tool("send_message", docs["send_message"],
          {**obj, "properties": {"text": {"type": "string"}}, "required": ["text"]})
    async def send_message(args):
        return await run("send_message", args, lambda: api.send_message(k, args.get("text", "")))

    @tool("read_messages", docs["read_messages"], {**obj, "properties": {}})
    async def read_messages(args):
        return await run("read_messages", args, lambda: api.read_messages(k))

    @tool("submit_prediction", docs["submit_prediction"], pl)
    async def submit_prediction(args):
        return await run("submit_prediction", args,
                         lambda: api.submit_prediction(k, args.get("p"), args.get("label")))

    tools = [get_training_data, start_queries, list_datasets, show_dataset, count, new_dataset,
             add_case, next_query, submit_opening, send_message, read_messages, submit_prediction]
    if api.single:  # no partner: the partner-phase tools are not offered
        tools = [t_ for t_ in tools
                 if t_.name not in ("submit_opening", "send_message", "read_messages")]
    if api.no_training:  # nothing to study: the study-phase tools are not offered
        tools = [t_ for t_ in tools if t_.name not in ("get_training_data", "start_queries")]
    server = create_sdk_mcp_server("data", tools=tools)
    return server, [f"mcp__data__{t.name}" for t in tools], tools


def count_example(labels: list[str]) -> str:
    """The example condition in the data API text: the one every earlier run had for three or
    more features, a two-feature version otherwise (where it would repeat a name)."""
    if len(labels) >= 3:
        return f"{labels[0]} == 1 and not ({labels[1]} == 1 or {labels[-1]} == 1)"
    return f"{labels[0]} == 1 and not ({labels[-1]} == 1)"


def prompts(api: BooleanAPI, k: str) -> dict[str, str]:
    """System prompt, solo intro and partner intro of twin k (`{n_train}` filled in later)."""
    labels, other = api.labels[k], api.labels[api.other(k)]
    scoring_final = (SCORING.replace("your probability", "your final probability")
                     .replace("your label", "your final label"))
    if api.no_training:
        if api.new_features(k):
            raise ValueError("study_labels needs a study phase (no_training is set)")
        return {"system": SYSTEM_UNTRAINED.format(
                    name=k, d=len(labels), names=", ".join(labels), other=api.other(k),
                    own=", ".join(labels), other_feats=", ".join(other), T=api.T, cap=api.cap,
                    scoring_final=scoring_final, example=count_example(labels)),
                "solo": "", "partner": ""}
    studied = api.study_labels[k]
    system = SYSTEM.format(name=k, d=len(studied), names=", ".join(studied),
                           budget=api.budget[k], page=api.page_size,
                           example=count_example(studied))
    if api.briefing_announced:  # after the study paragraph, before the data API
        head, sep, tail = system.partition("\n\nData API:")
        system = head + BRIEF_ANNOUNCE.format(K=api.K) + sep + tail
    if api.single:
        if api.new_features(k):
            raise ValueError("study_labels is not supported for the single agent")
        K1 = api.single
        return {"system": system,
                "solo": SINGLE_INTRO.format(n_train="{n_train}", K1=K1, cap=api.cap,
                                            scoring=SCORING),
                "break": SINGLE_BREAK.format(K1=K1, T=api.K - K1, first=K1 + 1, last=api.K,
                                             cap=api.cap),
                "partner": ""}
    # The new features are announced in the first phase that shows them: the solo queries, or
    # the partner queries if new_at = "partner" or there are no solo queries.
    new = api.new_features(k)
    note = (NEW_FEATURES.format(new=", ".join(new), own=", ".join(labels)) if new else "")
    in_solo = api.new_at == "solo" and api.K > 0
    solo = SOLO_INTRO.format(n_train="{n_train}", K=api.K, cap=api.cap, scoring=SCORING,
                             new_features=note if in_solo else "")
    partner = PARTNER_INTRO.format(
        n_train="{n_train}", K=api.K, other=api.other(k), own=", ".join(labels),
        other_feats=", ".join(other), T=api.T, first=api.K + 1, last=api.K + api.T,
        cap=api.cap, scoring_final=scoring_final, new_features="" if in_solo else note)
    if api.other_agent:
        if new:
            raise ValueError("study_labels is not supported with other_agent")
        partner = OTHER_INTRO.format(n_train="{n_train}", K=api.K, other=api.other(k), T=api.T,
                                     first=api.K + 1, last=api.K + api.T, cap=api.cap,
                                     scoring_final=scoring_final)
        if api.briefing:
            head, sep, tail = partner.partition("\n\nThere are ")
            partner = head + OTHER_BRIEF_INTRO + sep + tail
            last = f"You can make at most {api.cap} tool calls per query. Call next_query to begin."
            partner = partner[: -len(last)] + OTHER_BRIEF_START.format(cap=api.cap)
    elif api.briefing:
        # The conversation goes right after the partner is introduced, and the closing line
        # ("Call next_query to begin.") says how to start and end it instead.
        head, sep, tail = partner.partition("\n\nThere are ")
        partner = head + BRIEF_INTRO + sep + tail
        last = f"You can make at most {api.cap} tool calls per query. Call next_query to begin."
        if not partner.endswith(last):
            raise ValueError("unexpected partner intro ending")
        partner = partner[: -len(last)] + BRIEF_START.format(cap=api.cap)
    if api.notify:
        head, sep, tail = partner.partition("\n\nAt any time during the queries")
        if not sep:
            raise ValueError("unexpected partner intro (no 'At any time during the queries')")
        partner = head + NOTIFY_INTRO.format(who=api.w["your partner"]) + sep + tail
    return {"system": system, "solo": solo, "partner": partner}


class BooleanTwinSession(TwinSession):
    """One twin of the boolean task as one live agent session (the loop of TwinSession)."""

    continue_prompt = CONTINUE

    def __init__(self, name: str, api: BooleanAPI, model: str = "claude-haiku-5-5",
                 effort: str = "low", max_budget_usd: float = 3.0, max_continues: int = 10,
                 live_dir: str | Path | None = None, run: int = 0) -> None:
        self.name, self.api, self.model, self.effort = name, api, model, effort
        self.max_budget_usd, self.max_continues = max_budget_usd, max_continues
        p = prompts(api, name)
        self.system = p["system"]
        with api.cond:  # both twins write their own entries
            api.solo_intro = {**api.solo_intro, name: p["solo"]}
            api.partner_intro = {**api.partner_intro, name: p["partner"]}
            if api.single:
                api.single_break = p["break"]
        if api.single:
            self.continue_prompt = CONTINUE_SINGLE
        elif api.other_agent:
            self.continue_prompt = CONTINUE_OTHER
        self._init_run_state(live_dir, run)

    @property
    def first_prompt(self) -> str:
        return "Begin."

    def tools(self):
        return build_tools(self.api, self.name)

    @property
    def tool_env(self) -> dict:
        return {"MCP_TOOL_TIMEOUT": "3600000"}
