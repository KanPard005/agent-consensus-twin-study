"""Agentic twins: each twin drives a shared data-generator API through actions.

LogBook 2026-11-13 (agent-consensus). Instead of a harness that tells each twin "step t,
round r", every twin runs its own loop: one continuing conversation in which each reply is one
action (tool use emulated with structured output, since the CLI backend has no custom tools),
executed against a shared, thread-safe `DataGeneratorAPI`, whose result is the next message.

Actions:
- training phase: `next_training_cases(n)` (up to 25 per call, a fixed budget) and `regress`;
  no messages. It ends at the twin's first `next_query()`.
- query phase: `next_query()` blocks until both twins have asked, so both work on the same case;
  `submit_opening(y)` must come before `send_message(text)` / `read_messages()` (so the solo
  opinion is recorded before any talk); `read_messages()` blocks until the partner sends
  something new or submits its final (or is itself waiting, which would deadlock), and shows
  the partner's opening; `submit_final(y)` blocks until both finals are in and then reveals
  the true y. The team's prediction is the average of the two finals.
- `regress(predictors)` at any time: least squares over every case whose y the twin knows
  (training cases and past queries; regressions on own_opening / partner_opening use past
  queries only).

Limits keep the loop bounded: at most `max_actions` actions per query (at the limit the
opening, or 0, is submitted as final), and `max_train_actions` in the training phase (then
`next_query` is called on the twin's behalf). The conversation is compacted into the twin's own
summary after a query whenever it exceeds `compact_tokens` (as in `consensus.free_talk`).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np

from consensus.free_talk import COMPACT_REQUEST, SUMMARY_SCHEMA
from consensus.llm_agent import (
    NO_OVERLAP_FEATURE_LINE,
    OVERLAP_FEATURE_LINE,
    PARTNER_INFO_LINE,
    STRUCTURE_HINT,
    regress,
)

log = logging.getLogger("agentic")

ACTIONS = (
    "next_training_cases",
    "regress",
    "next_query",
    "submit_opening",
    "send_message",
    "read_messages",
    "submit_final",
)

SYSTEM_AGENTIC = """\
You are agent {name}, one of two agents (A and B) who predict a number together. You work \
through a data API, by taking actions.

The task: each query is a new case with an unknown real-valued target y. You see {d} numeric \
features of each case; your partner sees some other features of the same case. {partner_par} \
{twin_line}

Training phase. First you study earlier cases on your own: next_training_cases gives you up to \
25 cases per call ({budget} in total), and regress fits models to the cases you have. Your \
partner trains separately on its own cases, and you cannot talk during training. The training \
phase ends when you first call next_query; after that you cannot get more training cases.

Query phase. There are {T} queries. next_query gives you the next case; it waits until your \
partner has also asked for it, so you both work on the same case. For each query you must first \
submit_opening with your own estimate of y; only then can you send_message and read_messages \
(read_messages waits until your partner has sent something new or has submitted its final, and \
also shows your partner's opening estimate). Finish with submit_final: the data API waits for \
both finals and then tells you the true y. The team's prediction is the average of your two \
finals, and the team is scored by squared error. You can take at most {max_actions} actions per \
query; at the limit your opening is submitted as your final.

Actions. Reply with exactly one action per turn; fields that the action does not use are \
ignored.
- next_training_cases: uses n (1 to 25).
- regress: uses predictors (allowed: {columns}). Least squares of y on them plus an intercept \
over every case whose y you know (your training cases and past queries; a regression on \
own_opening or partner_opening, your and your partner's openings, runs on past queries only). \
Returns the coefficients, the in-sample and leave-one-out mean squared errors, and the \
prediction for the current query when all predictors are known.
- next_query.
- submit_opening: uses y.
- send_message: uses text (up to {max_chars} characters).
- read_messages.
- submit_final: uses y.

This is one continuing conversation for the whole run: every action and result stays in it.\
{compact_par}"""

COMPACT_PAR_AGENTIC = (
    " When the conversation gets long, you will be asked to summarize it; it then continues "
    "from your summary, and everything earlier is removed."
)

TWIN_LINE_AGENTIC = (
    "Your partner is another instance of the same AI model as you, with the same instructions; "
    "it trains on its own cases, using only its own features."
)

RESUME_AGENTIC = """\
This conversation continues an earlier one that grew too long and was compacted. Your summary \
of it:
{summary}

The result of your last action: {last}"""


def action_schema(columns: list[str]) -> dict:
    """One action per reply; every field is required, unused ones are ignored."""
    props = {
        "action": {"type": "string", "enum": list(ACTIONS)},
        "n": {"type": "integer", "description": "next_training_cases: how many (1-25)."},
        "predictors": {
            "type": "array",
            "items": {"type": "string", "enum": list(columns)},
            "description": "regress: the predictors.",
        },
        "y": {"type": "number", "description": "submit_opening / submit_final: your estimate."},
        "text": {"type": "string", "description": "send_message: your message."},
    }
    return {
        "type": "object",
        "properties": props,
        "required": list(props),
        "additionalProperties": False,
    }


@dataclass
class TwinState:
    """What the data API knows about one twin."""

    X_train: np.ndarray
    y_train: np.ndarray
    labels: list[str]
    n_train_seen: int = 0
    phase: str = "train"  # train | query | done
    q: int = -1  # current query
    waiting_query: int = -1  # query index the twin is waiting for in next_query
    waiting_read: bool = False
    inbox: list[tuple[int, str]] = field(default_factory=list)  # (query, text)
    n_read: int = 0


class DataGeneratorAPI:
    """Shared, thread-safe environment for the two twins (see the module docstring)."""

    def __init__(
        self,
        train: dict[str, tuple[np.ndarray, np.ndarray]],
        queries: dict[str, np.ndarray],
        y: np.ndarray,
        labels: dict[str, list[str]],
        budget: int,
        max_chars: int = 2000,
    ) -> None:
        self.cond = threading.Condition()
        self.s = {
            k: TwinState(X[:budget], yt[:budget], labels[k]) for k, (X, yt) in train.items()
        }
        self.qx, self.y, self.T = queries, np.asarray(y, float), len(y)
        self.budget, self.max_chars = budget, max_chars
        self.opening = {k: np.full(self.T, np.nan) for k in self.s}
        self.final = {k: np.full(self.T, np.nan) for k in self.s}
        self.forced = {k: np.zeros(self.T, bool) for k in self.s}  # final set by the action cap
        self.messages: list[dict] = []  # every message: query, sender, text, order
        # Called once per query, as soon as both finals are in (live logging): fn(record).
        self.on_query = None

    @staticmethod
    def other(k: str) -> str:
        return "B" if k == "A" else "A"

    # ---- helpers ---------------------------------------------------------------------------
    def _known_rows(self, k: str) -> list[dict]:
        """Every case whose y twin k knows: its training cases seen, and past queries."""
        st, o = self.s[k], self.other(k)
        rows = [
            {**dict(zip(st.labels, map(float, x))), "y": float(yv)}
            for x, yv in zip(st.X_train[: st.n_train_seen], st.y_train[: st.n_train_seen])
        ]
        for q in range(max(st.q + 1, 0)):
            if np.isnan(self.final[k][q]) or np.isnan(self.final[o][q]):
                continue  # y not revealed yet
            rows.append(
                {
                    **dict(zip(st.labels, map(float, self.qx[k][q]))),
                    "own_opening": float(self.opening[k][q]),
                    "partner_opening": float(self.opening[o][q]),
                    "y": float(self.y[q]),
                }
            )
        return rows

    def _partner_gone(self, k: str) -> bool:
        return self.s[self.other(k)].phase == "done"

    # ---- actions -----------------------------------------------------------------------------
    def next_training_cases(self, k: str, n: int) -> str:
        with self.cond:
            st = self.s[k]
            if st.phase != "train":
                return "Error: the training phase is over."
            n = int(min(max(n, 1), 25, self.budget - st.n_train_seen))
            if n <= 0:
                return f"Error: you have used all {self.budget} training cases."
            a, b = st.n_train_seen, st.n_train_seen + n
            st.n_train_seen = b
            head = ", ".join(st.labels) + " | y"
            rows = [
                ", ".join(f"{v:.3f}" for v in x) + f" | {yv:.3f}"
                for x, yv in zip(st.X_train[a:b], st.y_train[a:b])
            ]
            return (
                f"Training cases {a + 1}-{b} of {self.budget} ({head}):\n" + "\n".join(rows)
            )

    def regress(self, k: str, predictors: list[str]) -> str:
        with self.cond:
            st, o = self.s[k], self.other(k)
            rows = self._known_rows(k)
            if any(p in ("own_opening", "partner_opening") for p in predictors):
                rows = [r for r in rows if "own_opening" in r]
            current = {}
            if st.phase == "query" and 0 <= st.q < self.T:
                current = dict(zip(st.labels, map(float, self.qx[k][st.q])))
                if not np.isnan(self.opening[k][st.q]):
                    current["own_opening"] = float(self.opening[k][st.q])
                if not np.isnan(self.opening[o][st.q]) and not np.isnan(self.opening[k][st.q]):
                    current["partner_opening"] = float(self.opening[o][st.q])
            return regress(rows, predictors, current or None, rows="cases with known y")

    def next_query(self, k: str) -> tuple[str, bool]:
        """Returns (result, run_over)."""
        with self.cond:
            st = self.s[k]
            if st.phase == "query" and st.q >= 0 and np.isnan(self.final[k][st.q]):
                return f"Error: submit your final for query {st.q + 1} first.", False
            nq = st.q + 1
            if nq >= self.T:
                st.phase = "done"
                self.cond.notify_all()
                return "There are no more queries: the run is over.", True
            st.phase, st.waiting_query = "query", nq
            self.cond.notify_all()
            o = self.s[self.other(k)]
            self.cond.wait_for(lambda: o.waiting_query >= nq or o.q >= nq or o.phase == "done")
            st.q, st.n_read = nq, 0
            x = ", ".join(f"{lab}={v:.3f}" for lab, v in zip(st.labels, self.qx[k][nq]))
            return f"Query {nq + 1} of {self.T}. Your features: {x}.", False

    def submit_opening(self, k: str, y: float) -> str:
        with self.cond:
            st = self.s[k]
            if st.phase != "query":
                return "Error: no current query (call next_query)."
            if not np.isnan(self.opening[k][st.q]):
                return "Error: you already submitted your opening for this query."
            self.opening[k][st.q] = y
            self.cond.notify_all()
            return f"Opening {y:.3f} recorded for query {st.q + 1}. You can now talk."

    def send_message(self, k: str, text: str) -> str:
        with self.cond:
            st = self.s[k]
            if st.phase != "query" or np.isnan(self.opening[k][st.q]):
                return "Error: submit your opening for the current query first."
            text = text[: self.max_chars]
            self.s[self.other(k)].inbox.append((st.q, text))
            self.messages.append(
                {"q": st.q, "from": k, "text": text, "order": len(self.messages)}
            )
            self.cond.notify_all()
            return "Message sent."

    def read_messages(self, k: str) -> str:
        with self.cond:
            st, ok = self.s[k], self.other(k)
            o = self.s[ok]
            if st.phase != "query" or np.isnan(self.opening[k][st.q]):
                return "Error: submit your opening for the current query first."
            q = st.q

            def unread():
                return [t for qq, t in st.inbox if qq == q][st.n_read :]

            st.waiting_read = True
            self.cond.notify_all()
            self.cond.wait_for(
                lambda: unread()
                or not np.isnan(self.final[ok][q])
                or o.phase == "done"
                or (o.waiting_read and o.q == q and not o_unread(o, q))
            )
            st.waiting_read = False
            new = unread()
            st.n_read += len(new)
            po = self.opening[ok][q]
            lines = [
                f"Partner's opening for query {q + 1}: "
                + ("not submitted yet." if np.isnan(po) else f"{po:.3f}.")
            ]
            lines += [f'Partner\'s message: "{t}"' for t in new] or ["No new messages."]
            if not np.isnan(self.final[ok][q]):
                lines.append(f"Your partner has submitted its final ({self.final[ok][q]:.3f}).")
            elif not new and o.waiting_read:
                lines.append("Your partner is also waiting for a message.")
            return "\n".join(lines)

    def submit_final(self, k: str, y: float, forced: bool = False) -> str:
        with self.cond:
            st, ok = self.s[k], self.other(k)
            if st.phase != "query":
                return "Error: no current query (call next_query)."
            q = st.q
            if np.isnan(self.opening[k][q]) and not forced:
                return "Error: submit your opening first."
            if not np.isnan(self.final[k][q]):
                return "Error: you already submitted your final for this query."
            self.final[k][q] = y
            self.forced[k][q] = forced
            if not np.isnan(self.final[ok][q]) and self.on_query is not None:
                self.on_query(self.query_record(q))  # the second final completes the query
            self.cond.notify_all()
            self.cond.wait_for(
                lambda: not np.isnan(self.final[ok][q]) or self.s[ok].phase == "done"
            )
            pf = self.final[ok][q]
            team = (y + pf) / 2 if not np.isnan(pf) else y
            return (
                f"Query {q + 1} result: true y {self.y[q]:.3f}; your final {y:.3f}; partner "
                f"final {pf:.3f}; team {team:.3f}."
            )

    def query_record(self, q: int) -> dict:
        """Features, y, openings, finals, team and messages of query q."""
        fin = {k: float(self.final[k][q]) for k in "AB"}
        return {
            "t": q,
            "y": float(self.y[q]),
            "x": {k: [float(v) for v in self.qx[k][q]] for k in "AB"},
            "opening": {k: float(self.opening[k][q]) for k in "AB"},
            "final": fin,
            "forced": {k: bool(self.forced[k][q]) for k in "AB"},
            "team": (fin["A"] + fin["B"]) / 2,
            "messages": [m for m in self.messages if m["q"] == q],
        }

    def finish(self, k: str) -> None:
        """Mark twin k done (also after an error), so its partner never waits on it."""
        with self.cond:
            self.s[k].phase = "done"
            self.cond.notify_all()


def o_unread(o: TwinState, q: int) -> bool:
    """Whether twin `o` has an unread message for query q."""
    return bool([t for qq, t in o.inbox if qq == q][o.n_read :])


class AgentTwin:
    """One twin's agent loop over a backend conversation (see the module docstring)."""

    def __init__(
        self,
        name: str,
        api: DataGeneratorAPI,
        backend,
        labels: list[str],
        partner_labels: list[str],
        T: int,
        budget: int,
        max_actions: int = 12,
        max_train_actions: int = 30,
        max_chars: int = 2000,
        compact_tokens: int | None = 8000,
        summary_chars: int = 4000,
        structure_hint: bool = True,
        partner_info: bool = True,
        tell_twin: bool = True,
        max_cost_usd: float | None = None,
        live_dir: str | Path | None = None,
        run: int = 0,
    ) -> None:
        self.name, self.api, self.backend = name, api, backend
        self.max_cost_usd = max_cost_usd  # stop this twin's loop once its spend exceeds this
        self.labels, self.T = labels, T
        self.max_actions, self.max_train_actions = max_actions, max_train_actions
        self.compact_tokens, self.summary_chars = compact_tokens, summary_chars
        self.columns = list(labels) + ["own_opening", "partner_opening"]
        self.schema = action_schema(self.columns)
        shared = [lab for lab in labels if lab in partner_labels]
        fmt = dict(own=", ".join(labels), other=", ".join(partner_labels))
        parts = [
            OVERLAP_FEATURE_LINE.format(shared=" and ".join(shared), **fmt)
            if shared
            else NO_OVERLAP_FEATURE_LINE.format(**fmt)
        ]
        if partner_info:
            parts.append(PARTNER_INFO_LINE)
        if structure_hint:
            parts.append(STRUCTURE_HINT)
        self.system = SYSTEM_AGENTIC.format(
            name=name,
            d=len(labels),
            partner_par=" ".join(parts),
            twin_line=TWIN_LINE_AGENTIC if tell_twin else "",
            budget=budget,
            T=T,
            max_actions=max_actions,
            columns=", ".join(self.columns),
            max_chars=max_chars,
            compact_par=COMPACT_PAR_AGENTIC if compact_tokens else "",
        ).replace("  ", " ")
        self.conv = None
        self.log: list[dict] = []  # every action: query, action, args, result
        self.turns: list[dict] = []  # every turn of every conversation
        self.compactions: list[dict] = []
        self.replies = []  # BackendReply objects, for cost
        self.unusable = 0
        # Live files (appended as the run goes): actions, turns, compactions and a status file.
        self.live = Path(live_dir) if live_dir is not None else None
        self.run_id = run
        if self.live is not None:
            self.live.mkdir(parents=True, exist_ok=True)
            for kind in ("actions", "turns", "compactions"):
                self._path(kind).write_text("")

    def context_tokens(self) -> int:
        turns = self.conv.turns if self.conv is not None else []
        return (len(self.system) + sum(len(t["text"]) for t in turns)) // 4

    def _send(self, user: str, schema: dict):
        if self.conv is None:
            self.conv = self.backend.conversation(self.system)
        reply = self.conv.send(user, schema)
        self.replies.append(reply)
        new = [{**t, "conversation": len(self.compactions)} for t in self.conv.turns[-2:]]
        self.turns += new
        self._append("turns", new)
        return reply

    def _compact(self, last: str) -> str:
        """Summarize, restart; returns the next user message (resume head + last result)."""
        tokens = self.context_tokens()
        rep = self._send(COMPACT_REQUEST.format(summary_chars=self.summary_chars), SUMMARY_SCHEMA)
        d = rep.data if isinstance(rep.data, dict) else {}
        summary = str(d.get("summary", ""))[: self.summary_chars]
        self.compactions.append({"q": self.api.s[self.name].q, "tokens": tokens, "summary": summary})
        self._append("compactions", self.compactions[-1:])
        self.conv = None
        return RESUME_AGENTIC.format(summary=summary or "(no summary)", last=last)

    def _execute(self, d: dict) -> tuple[str, bool]:
        """Run one action; returns (result, run_over)."""
        a, k = d.get("action"), self.name
        if a == "next_training_cases":
            return self.api.next_training_cases(k, int(d.get("n") or 25)), False
        if a == "regress":
            preds = [p for p in d.get("predictors") or [] if p in self.columns]
            if not preds:
                return "Error: give at least one allowed predictor.", False
            return self.api.regress(k, preds), False
        if a == "next_query":
            return self.api.next_query(k)
        if a in ("submit_opening", "submit_final"):
            try:
                y = float(d["y"])
                assert np.isfinite(y)
            except (TypeError, KeyError, ValueError, AssertionError):
                return "Error: y must be a number.", False
            fn = self.api.submit_opening if a == "submit_opening" else self.api.submit_final
            return fn(k, y), False
        if a == "send_message":
            return self.api.send_message(k, str(d.get("text", ""))), False
        if a == "read_messages":
            return self.api.read_messages(k), False
        return "Error: unknown action.", False

    def run(self) -> None:
        """The loop: until the run is over (or an action budget is exhausted)."""
        user = (
            "The data API is ready. You are in the training phase. Take your first action."
        )
        n_query_actions = 0  # actions since the last next_query
        n_train_actions = 0
        q_last = -1
        # Hard stop for a twin that never moves on (e.g. keeps acting after its final).
        max_total = self.max_train_actions + self.T * (self.max_actions + 4)
        try:
            while True:
                if len(self.replies) >= max_total:
                    log.warning("twin %s: action cap (%d calls) reached, stopping", self.name, max_total)
                    break
                if self.max_cost_usd is not None and self.cost() > self.max_cost_usd:
                    log.warning("twin %s: cost cap $%.2f reached, stopping", self.name, self.max_cost_usd)
                    break
                st = self.api.s[self.name]
                # Action limits: force what the twin did not do.
                if st.phase == "train" and n_train_actions >= self.max_train_actions:
                    res, over = self.api.next_query(self.name)
                    self._log("next_query (forced)", {}, res)
                    user = f"Training-action limit reached; next_query was called for you. {res}"
                    n_query_actions = 0
                    if over:
                        break
                    continue
                if st.phase == "query" and st.q == q_last and n_query_actions >= self.max_actions:
                    q = st.q
                    if np.isnan(self.api.final[self.name][q]):
                        op = self.api.opening[self.name][q]
                        res = self.api.submit_final(
                            self.name, 0.0 if np.isnan(op) else float(op), forced=True
                        )
                        self._log("submit_final (forced)", {}, res)
                        self._progress()
                        if q == self.T - 1:  # that was the last query: the run is over
                            break
                        user = (
                            "Action limit for this query reached; your opening was submitted as "
                            f"your final. {res}"
                        )
                        user = self._maybe_compact(user)
                    n_query_actions = 0
                    q_last = -2  # until the next next_query
                    continue
                reply = self._send(user, self.schema)
                d = reply.data if isinstance(reply.data, dict) else None
                if st.phase == "train":
                    n_train_actions += 1
                else:
                    n_query_actions += 1
                if d is None or d.get("action") not in ACTIONS:
                    self.unusable += 1
                    user = "Unusable reply: answer with exactly one action in the required format."
                    continue
                res, over = self._execute(d)
                self._log(d["action"], d, res)
                if over:
                    break
                if d["action"] == "next_query" and not res.startswith("Error"):
                    q_last, n_query_actions = self.api.s[self.name].q, 0
                if d["action"] == "submit_final" and not res.startswith("Error"):
                    self._progress()
                    if self.api.s[self.name].q == self.T - 1:
                        break  # the last query is scored: the run is over (no idle actions)
                    res = self._maybe_compact(res)
                user = res
        except Exception:
            log.exception("twin %s: loop failed", self.name)
            raise
        finally:
            log.info("twin %s: finished after %d calls, $%.3f", self.name, len(self.replies), self.cost())
            self.write_status("finished")
            self.api.finish(self.name)

    def cost(self) -> float:
        return float(np.nansum([r.cost_usd for r in self.replies]))

    def _path(self, kind: str) -> Path:
        return self.live / f"run{self.run_id}_{kind}_{self.name}.jsonl"

    def _append(self, kind: str, rows: list[dict]) -> None:
        """Append rows to this twin's live file of `kind` (one writer per file: no lock)."""
        if self.live is None or not rows:
            return
        with self._path(kind).open("a") as f:
            f.write("".join(json.dumps(r) + "\n" for r in rows))

    def write_status(self, state: str = "running") -> None:
        """Overwrite this twin's status file: query, calls, spend, context, compactions."""
        if self.live is None:
            return
        st = self.api.s[self.name]
        status = {
            "twin": self.name,
            "run": self.run_id,
            "state": state,
            "phase": st.phase,
            "query": st.q + 1,
            "T": self.T,
            "training_cases": st.n_train_seen,
            "calls": len(self.replies),
            "cost_usd": round(self.cost(), 4),
            "context_tokens_est": self.context_tokens(),
            "compactions": len(self.compactions),
            "unusable": self.unusable,
            "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        (self.live / f"run{self.run_id}_status_{self.name}.json").write_text(
            json.dumps(status, indent=1)
        )

    def _progress(self) -> None:
        """One log line and a status update per finished query: calls, spend, context size."""
        q = self.api.s[self.name].q
        self.write_status()
        log.info(
            "twin %s: query %d/%d done; %d calls, $%.3f, context ~%d tokens, %d compactions",
            self.name, q + 1, self.T, len(self.replies), self.cost(), self.context_tokens(),
            len(self.compactions),
        )

    def _maybe_compact(self, last: str) -> str:
        if self.compact_tokens is not None and self.context_tokens() > self.compact_tokens:
            return self._compact(last)
        return last

    def _log(self, action: str, d: dict, result: str) -> None:
        args = {}
        if action.startswith("next_training_cases"):
            args = {"n": d.get("n")}
        elif action == "regress":
            args = {"predictors": d.get("predictors")}
        elif action in ("submit_opening", "submit_final"):
            args = {"y": d.get("y")}
        elif action == "send_message":
            args = {"text": d.get("text")}
        self.log.append(
            {"q": self.api.s[self.name].q, "action": action, "args": args, "result": result[:2000]}
        )
        self._append("actions", self.log[-1:])
        if self.api.s[self.name].phase == "train":
            self.write_status()  # training progress too (queries update it in _progress)


def run_episode(api: DataGeneratorAPI, twins: dict[str, AgentTwin]) -> None:
    """Run both twins' loops in parallel threads until both are done."""
    threads = [threading.Thread(target=tw.run, name=k) for k, tw in twins.items()]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
