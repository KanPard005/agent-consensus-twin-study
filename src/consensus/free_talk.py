"""Free talk between persistent, self-learning twins, with context compaction.

LogBook 2026-11-13 (agent-consensus): does compacting the twins' context change how they use
their partner, and do they become entrenched over a run?

- `PersistentTwin`: a self-learning twin (`LLMTwin` with `learner="self"`: it first studies its
  training cases and writes notes) that then talks in **one continuing conversation** for the
  whole run. Its learning-phase notes open the conversation. It has a least-squares tool over the
  past steps of the run (its features, its own round-1 estimate `own_open`, its partner's
  `partner_open`), run once automatically every round and callable on demand.
- Free talk: each step has up to `max_rounds` rounds of simultaneous messages (up to `max_chars`
  characters each). In every round each twin also says whether it is done; the step ends after a
  round in which both are done. The team's prediction is the average of the two final estimates.
- Compaction (`FreeTalkExchange`, `compact_tokens`): after a step, a twin whose conversation is
  longer than `compact_tokens` (estimated as characters / 4) is asked to summarize it. The
  conversation then restarts from the same system prompt, with the summary and the last step's
  turns verbatim as the first user message. `compact_tokens=None` never compacts (the full arm).

The twin's best response with independent features (`cov: overlap`) is
final = own_open + partner_open - <x_shared, w_shared> (see LogBook note 2026-11-12).
"""

from __future__ import annotations

import numpy as np

from consensus.llm_agent import TWIN_LINE_SELF, LLMTwin, StepRecord, TwinReply, regress
from consensus.llm_backends import BackendReply
from consensus.protocols import LLMExchange, _errors

SYSTEM_FREE = """\
You are agent {name}, one of two agents (A and B) who predict a number together.

Each step there is a new case with an unknown real-valued target y. You see {d} numeric \
features of the case; your partner sees some other features of the same case. {partner_par} \
{twin_line}

Before this phase you studied {n_train} earlier cases on your own (your features and the true y) \
and wrote notes with the model or rules you use to predict y from your features. Your notes are \
in the first message of this conversation.

Each step you and your partner talk for up to {max_rounds} rounds. In every round you \
simultaneously send each other a message (as long as you need, up to {max_chars} characters) \
and each give an estimate of y, and you say whether you are done talking for this step. From \
round 2 on you see your partner's messages and estimates from the earlier rounds of the step. \
The step ends after a round in which both of you say you are done, or after round {max_rounds}. \
The team's prediction is the average of your two final estimates (those of the last round), \
and the team is scored by squared error. After the step the true y is revealed to both of you.

This is one continuing conversation for the whole run: every step, round, message and outcome \
stays in it, so you can look back at anything earlier when deciding what to do.{compact_par}

You also have a regression tool. To use it, put predictor names in `regress` (allowed: \
{columns}; {own_names} are your features, own_open is your round-1 estimate of the step and \
partner_open is your partner's round-1 estimate). It fits y on them plus an intercept by least \
squares over all past steps of this phase, and returns the coefficients, the in-sample and \
leave-one-out mean squared errors, and its prediction for the current case when all predictors \
are known. A reply with a non-empty `regress` is a tool call: its message and estimate are \
ignored and you are asked again with the result. You can make at most {max_tool_calls} tool \
calls per round. Leave `regress` empty to give your answer. Every round the tool is also run \
once for you automatically, in round 1 on your features and from round 2 on on own_open and \
partner_open, and its result is shown before you answer."""

COMPACT_PAR = (
    " When the conversation gets long, you will be asked to summarize it; it then continues "
    "from your summary and the last step, and everything earlier is removed."
)

COMPACT_REQUEST = """\
The conversation has grown long and is about to be compacted: everything before this point will \
be removed, except the last step. Write a summary of everything you will need to continue the \
task well (at most {summary_chars} characters). The conversation will continue from your summary."""

RESUME_HEAD = """\
This conversation continues an earlier one that grew too long and was compacted. Your summary \
of it:
{summary}

The last step before compaction, verbatim:
{last_step}

Now the next step."""

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string", "description": "Your summary."}},
    "required": ["summary"],
    "additionalProperties": False,
}


def free_schema(columns: list[str]) -> dict:
    """Reply schema: message, estimate, done, and the regression tool field."""
    props = {
        "message": {"type": "string", "description": "Message to your partner."},
        "estimate": {"type": "number", "description": "Your estimate of y for this round."},
        "done": {
            "type": "boolean",
            "description": "True if you are done talking for this step.",
        },
        "regress": {
            "type": "array",
            "items": {"type": "string", "enum": list(columns)},
            "description": "Predictors for a regression tool call; empty to answer.",
        },
    }
    return {
        "type": "object",
        "properties": props,
        "required": list(props),
        "additionalProperties": False,
    }


class PersistentTwin(LLMTwin):
    """Self-learning twin that talks freely in one conversation, with a regression tool.

    Parameters (beyond `LLMTwin`'s, which are passed through with `learner="self"`)
    ----------
    max_rounds : int
        Most rounds per step (`n_rounds` of `LLMTwin`).
    compact_tokens : int or None
        Compaction threshold; only used to say in the system prompt that compaction happens.
    summary_chars : int
        Most characters of a compaction summary.
    max_tool_calls : int
        On-demand regression calls per round (the automatic run is extra).
    """

    def __init__(
        self,
        name: str,
        linear,
        backend,
        n_rounds: int,
        T: int,
        n_train: int,
        compact_tokens: int | None = None,
        summary_chars: int = 4000,
        max_tool_calls: int = 2,
        **kw,
    ) -> None:
        kw.update(learner="self", memory="notes", tools=False, show_partner_estimates=True)
        super().__init__(name, linear, backend, n_rounds, T, n_train, **kw)
        self.persistent = True  # one conversation from here on (the parent built notes prompts)
        self.max_tool_calls = max_tool_calls
        self.compact_tokens = compact_tokens
        self.summary_chars = summary_chars
        self.columns = list(self.labels) + ["own_open", "partner_open"]
        self.schema = free_schema(self.columns)
        twin_line = TWIN_LINE_SELF if kw.get("tell_twin", True) else ""
        self.system = SYSTEM_FREE.format(
            name=name,
            d=linear.d,
            partner_par=self._partner_par(),
            twin_line=twin_line,
            n_train=n_train,
            max_rounds=n_rounds,
            max_chars=self.max_chars,
            compact_par=COMPACT_PAR if compact_tokens else "",
            columns=", ".join(self.columns),
            own_names=", ".join(self.labels),
            max_tool_calls=max_tool_calls,
        ).replace("  ", " ")
        self.conv = None
        self._resume = ""  # first-message head after a compaction
        self._step_turns: list[dict] = []  # this step's turns, kept verbatim through compaction
        self.done = False  # the latest round's "done"
        self.compactions: list[dict] = []
        self.all_turns: list[dict] = []  # every turn of every conversation, for saving

    # ---- conversation -----------------------------------------------------------------------
    def context_tokens(self) -> int:
        """Estimated tokens in the current conversation (system + turns), at 4 characters each."""
        turns = self.conv.turns if self.conv is not None else []
        return (len(self.system) + sum(len(t["text"]) for t in turns)) // 4

    def _send(self, user: str) -> BackendReply:
        step_user = user  # what the step itself said, without a conversation head
        if self.conv is None:
            self.conv = self.backend.conversation(self.system)
            head = self._resume or f"Your notes from the learning phase:\n{self.notes or '(empty)'}"
            user = f"{head}\n\n{user}"
            self._resume = ""
        reply = self.conv.send(user, self.schema)
        # Kept verbatim through a compaction: the step's own turns only (not an earlier summary
        # or step carried in the head, which would otherwise nest and grow).
        self._step_turns += [{"role": "user", "text": step_user}, self.conv.turns[-1]]
        self.all_turns += [
            {**t, "conversation": len(self.compactions)} for t in self.conv.turns[-2:]
        ]
        return reply

    def compact(self) -> BackendReply | None:
        """Summarize the conversation and restart it from the summary + the last step.

        The summary request is sent into the current conversation, so the twin sees everything
        it is about to lose. On an unusable reply the summary is empty (the restart still
        happens, so the context stays bounded)."""
        if self.conv is None:
            return None
        tokens = self.context_tokens()
        req = COMPACT_REQUEST.format(summary_chars=self.summary_chars)
        reply = self.conv.send(req, SUMMARY_SCHEMA)
        d = reply.data if isinstance(reply.data, dict) else {}
        summary = str(d.get("summary", ""))[: self.summary_chars]
        self.all_turns += [
            {**t, "conversation": len(self.compactions)} for t in self.conv.turns[-2:]
        ]
        last = "\n".join(f"[{t['role']}] {t['text']}" for t in self._step_turns)
        self._resume = RESUME_HEAD.format(summary=summary or "(no summary)", last_step=last)
        self.compactions.append(
            {"t": self.history[-1].t if self.history else -1, "tokens": tokens, "summary": summary}
        )
        self.conv = None
        return reply

    # ---- prompts ----------------------------------------------------------------------------
    def prompt_free(self, t: int, rnd: int, x: np.ndarray, partner, extra=()) -> str:
        """User turn for step t, round rnd (0-based): the previous outcome and the case in round
        1, the partner's latest message and estimate after that."""
        lines = []
        if rnd == 0:
            if self.history:
                h = self.history[-1]
                lines.append(
                    f"Outcome of step {h.t + 1}: true y {h.y:.3f}; team {h.team:.3f}; your final "
                    f"{h.final_est:.3f}; partner final {h.partner_final:.3f}; your round 1 "
                    f"{h.model_pred:.3f}; partner round 1 {h.partner_open:.3f}."
                )
            lines += [
                f"Step {t + 1} of {self.T}, round 1 of up to {self.n_rounds}.",
                self._features_line(x),
                "Partner's messages this step: none yet.",
            ]
        else:
            m, e = partner[-1]
            lines += [
                f"Step {t + 1}, round {rnd + 1} of up to {self.n_rounds}.",
                f'Partner\'s round {rnd} message: estimate {e:.3f}, "{m}"',
            ]
        lines += list(extra)
        lines.append(
            "Reply with your message to your partner, your estimate of y for this round, and "
            "whether you are done talking for this step."
        )
        return "\n".join(lines)

    def _table(self) -> list[dict]:
        rows = []
        for h in self.history:
            row = {lab: float(v) for lab, v in zip(self.labels, h.x)}
            row.update(own_open=h.model_pred, partner_open=h.partner_open, y=h.y)
            rows.append(row)
        return rows

    # ---- one round --------------------------------------------------------------------------
    def act(self, t: int, rnd: int, x: np.ndarray, partner: list[tuple[str, float]]) -> TwinReply:
        """One round: the automatic regression, up to `max_tool_calls` tool calls, the answer."""
        if rnd == 0:
            self._own_step = []
            self._step_turns = []
        current = {lab: float(v) for lab, v in zip(self.labels, x)}
        if rnd > 0:
            current.update(own_open=self._own_step[0][1], partner_open=partner[0][1])
        auto = list(self.labels) if rnd == 0 else ["own_open", "partner_open"]
        result = regress(self._table(), auto, current)
        calls = [{"predictors": auto, "result": result, "auto": True}]
        extra = [f"Automatic tool run: regress on {', '.join(auto)}. Result: {result}"]
        backends, n_sent = [], 0
        for k in range(self.max_tool_calls + 1):
            if k == 0:
                user = self.prompt_free(t, rnd, x, partner, extra)
            else:  # tool follow-up: only the new result
                user = "\n".join(extra[n_sent:] + ["Reply as before."])
            n_sent = len(extra)
            reply = self._send(user)
            backends.append(reply)
            d = reply.data if isinstance(reply.data, dict) else {}
            req = [c for c in d.get("regress", []) or [] if c in self.columns]
            if not req or k == self.max_tool_calls:
                break
            result = regress(self._table(), req, current)
            calls.append({"predictors": req, "result": result})
            extra.append(f"Tool call {k + 1}: regress on {', '.join(req)}. Result: {result}")
            if k + 1 == self.max_tool_calls:
                extra.append("No tool calls left this round: leave `regress` empty and answer.")
        try:
            est = float(d["estimate"])
            ok = bool(np.isfinite(est))
        except (TypeError, KeyError, ValueError):
            ok = False
        if not ok:
            fb = self._fallback(x)
            self._own_step.append(("", fb))
            self.done = True
            return TwinReply("", fb, "", False, backends, calls)
        msg = str(d.get("message", ""))[: self.max_chars]
        self.done = bool(d.get("done", True))
        self._own_step.append((msg, est))
        return TwinReply(msg, est, "", True, backends, calls)


class FreeTalkExchange(LLMExchange):
    """Free talk between `PersistentTwin`s with compaction (see the module docstring).

    Parameters
    ----------
    max_rounds : int
        Most rounds per step.
    compact_tokens : int or None
        Compact a twin's conversation after a step when it exceeds this many estimated tokens;
        None never compacts.
    summary_chars : int
        Most characters of a compaction summary.
    **kw
        Passed to `LLMExchange` and on to `PersistentTwin` (backend, labels, max_chars,
        structure_hint, partner_info, train_batch, notes_chars, max_tool_calls, ...).
    """

    twin_cls = PersistentTwin

    def __init__(
        self,
        max_rounds: int = 3,
        compact_tokens: int | None = None,
        summary_chars: int = 4000,
        **kw,
    ) -> None:
        kw.update(learner="self")
        super().__init__(
            n_exchanges=max_rounds - 1,
            compact_tokens=compact_tokens,
            summary_chars=summary_chars,
            **kw,
        )
        self.max_rounds = max_rounds
        self.compact_tokens = compact_tokens
        self.name = "llm_free_talk"

    def step(self, A, B, x_A, x_B, y):
        tA, tB = self.twins["A"], self.twins["B"]
        rounds, msgs = [], {"A": [], "B": []}
        for r in range(self.max_rounds):
            fA = self._pool.submit(tA.act, self.t, r, x_A, list(msgs["B"]))
            fB = self._pool.submit(tB.act, self.t, r, x_B, list(msgs["A"]))
            rA, rB = fA.result(), fB.result()
            msgs["A"].append((rA.message, rA.estimate))
            msgs["B"].append((rB.message, rB.estimate))
            rounds.append({"A": rA, "B": rB, "done": {"A": tA.done, "B": tB.done}})
            if tA.done and tB.done:
                break
        open_A, open_B = rounds[0]["A"].estimate, rounds[0]["B"].estimate
        fin_A, fin_B = rounds[-1]["A"].estimate, rounds[-1]["B"].estimate
        cons = (fin_A + fin_B) / 2
        tA.record(StepRecord(self.t, open_A, fin_A, fin_B, cons, y, partner_open=open_B), x_A)
        tB.record(StepRecord(self.t, open_B, fin_B, fin_A, cons, y, partner_open=open_A), x_B)

        # Compaction after the step, for each twin over the threshold (in parallel).
        ctx = {"A": tA.context_tokens(), "B": tB.context_tokens()}
        comp_replies, compacted = [], {}
        if self.compact_tokens is not None:
            futs = {
                k: self._pool.submit(tw.compact)
                for k, tw in self.twins.items()
                if ctx[k] > self.compact_tokens
            }
            for k, f in futs.items():
                rep = f.result()
                if rep is not None:
                    comp_replies.append(rep)
                    compacted[k] = self.twins[k].compactions[-1]

        replies = [rd[k] for rd in rounds for k in ("A", "B")]
        self.transcript.append(
            {
                "t": self.t,
                "y": y,
                "x": {"A": [float(v) for v in x_A], "B": [float(v) for v in x_B]},
                "model_pred": {"A": open_A, "B": open_B},
                "linear_ref": {"A": tA.model_prediction(x_A), "B": tB.model_prediction(x_B)},
                "consensus": cons,
                "n_rounds": len(rounds),
                "context_tokens": ctx,
                "compacted": compacted,
                "rounds": [
                    {
                        k: {
                            "message": rd[k].message,
                            "estimate": rd[k].estimate,
                            "done": rd["done"][k],
                            "ok": rd[k].ok,
                            "tool_calls": rd[k].tool_calls,
                        }
                        for k in ("A", "B")
                    }
                    for rd in rounds
                ],
            }
        )
        rec = _errors(cons, fin_A, fin_B, y, open_A, open_B, self.logistic)
        backs = [b for rp in replies for b in rp.backends] + comp_replies
        rec.update(
            parse_fail=float(sum(not rp.ok for rp in replies)),
            msg_chars=float(np.mean([len(rp.message) for rp in replies])),
            tokens_in=float(sum(b.tokens_in for b in backs)),
            tokens_out=float(sum(b.tokens_out for b in backs)),
            cost_usd=float(sum(b.cost_usd for b in backs)),
            tool_calls=float(sum(1 for rp in replies for c in rp.tool_calls if not c.get("auto"))),
            n_rounds=float(len(rounds)),
            context_tokens=float(np.mean(list(ctx.values()))),
            compactions=float(len(compacted)),
        )
        if self.t == 0:  # charge the learning phase to step 0
            ti, to, c = self._train_cost
            rec["tokens_in"] += ti
            rec["tokens_out"] += to
            rec["cost_usd"] += c
        if self.t == self.T - 1:
            self._pool.shutdown()
            self.conversations = {k: tw.all_turns for k, tw in self.twins.items()}
        return rec
