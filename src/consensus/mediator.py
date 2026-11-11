"""Mediated twins: the self-learning twins (S1) only predict, and a third LLM, the mediator, decides.

Each step both twins give one estimate of y from their own features and notes, without talking
(`LLMTwin` with `team_rule="mediator"`). The mediator is then shown, separately for twin A and
twin B, the twin's features (named a1..a{d} and b1..b{d}), its current notes (the model or rules
it learned and predicts with) and its estimate, and gives the team's prediction. It keeps its
own notes and a table of recent steps with the true y, so it can learn over the run how to
combine the two twins.

This moves the combination step from two peers, who in S1/S1n held on to their own rule and at
best met halfway, to an agent that owns neither opinion. The linear theory gives the target: at
rho = 0 the best combination adds the two half-signals (blend weights about 1 each), and the
mediator's fitted weights on the twins' estimates are compared with it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from consensus.llm_agent import StepRecord
from consensus.llm_backends import BackendReply
from consensus.protocols import LLMExchange, _errors

MEDIATOR_SYSTEM = """\
You are a mediator who makes the final prediction for a team of two agents, A and B.

Each step there is a new case with an unknown real-valued target y. Agent A sees {d_A} numeric \
features of the case, called a1..a{d_A}; agent B sees {d_B} different features of the same case, \
called b1..b{d_B}. They are different variables, so each agent knows things about the case that \
the other does not. A and B are two instances of the same AI model with the same instructions. \
Before this phase each studied {n_A} earlier cases on its own (only its own features and the \
true y) and wrote notes with the model or rules it uses to predict y; it keeps updating its notes \
as outcomes come in. A and B do not talk to each other.

Each step you are shown, separately for each agent, its features for the case, its current notes \
and its estimate of y. You then give the team's prediction of y, which is scored by squared \
error. After each step the true y is revealed to you.

You keep private notes that persist across steps (at most {notes_chars} characters). Rewrite them \
every step with whatever will help you later, for example how reliable each agent is and how to \
combine their information."""

MEDIATOR_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string", "description": "Your reasoning for this step (brief)."},
        "estimate": {"type": "number", "description": "The team's prediction of y."},
        "notes": {"type": "string", "description": "Your updated private notes."},
    },
    "required": ["reasoning", "estimate", "notes"],
    "additionalProperties": False,
}


@dataclass
class MediatorRecord:
    """What the mediator remembers of a finished step."""

    t: int
    est_A: float
    est_B: float
    decision: float
    y: float


@dataclass
class MediatorReply:
    """The mediator's decision for one step (the twins' average if the reply is unusable)."""

    estimate: float
    reasoning: str
    notes: str
    ok: bool
    backends: list[BackendReply] = field(repr=False, default_factory=list)


class Mediator:
    """LLM mediator over two self-learning twins (see the module docstring).

    Parameters
    ----------
    backend
        Object with `complete(system, user, schema) -> BackendReply`.
    d_A, d_B : int
        Number of features of each twin (named a1..a{d_A}, b1..b{d_B}).
    n_train : int
        Training cases each twin studied (shown to the mediator).
    T : int
        Number of steps (shown to the mediator).
    notes_chars : int
        Length limit for the mediator's notes (enforced by truncation).
    history_len : int
        Past steps shown in the prompt.
    """

    def __init__(
        self,
        backend,
        d_A: int,
        d_B: int,
        n_train: int,
        T: int,
        notes_chars: int = 1200,
        history_len: int = 10,
    ) -> None:
        self.backend, self.T = backend, T
        self.notes_chars, self.history_len = notes_chars, history_len
        self.system = MEDIATOR_SYSTEM.format(d_A=d_A, d_B=d_B, n_A=n_train, notes_chars=notes_chars)
        self.notes = ""
        self.history: list[MediatorRecord] = []

    @staticmethod
    def _agent_block(name: str, x: np.ndarray, notes: str, estimate: float) -> list[str]:
        f = name.lower()
        feats = ", ".join(f"{f}{i + 1}={v:.3f}" for i, v in enumerate(x))
        return [
            f"Agent {name}:",
            f"  features: {feats}",
            f"  notes: {notes or '(empty)'}",
            f"  estimate: {estimate:.3f}",
        ]

    def prompt(
        self,
        t: int,
        x_A: np.ndarray,
        x_B: np.ndarray,
        notes: dict[str, str],
        est: dict[str, float],
    ) -> str:
        """User prompt for step t (0-based): each agent's features, notes and estimate, then the
        mediator's own notes and recent steps."""
        lines = [f"Step {t + 1} of {self.T}."]
        lines += self._agent_block("A", x_A, notes["A"], est["A"])
        lines += self._agent_block("B", x_B, notes["B"], est["B"])
        lines.append(f"Your notes: {self.notes or '(empty)'}")
        if self.history:
            lines.append("Recent steps (oldest first):")
            lines.append("  step | A estimate | B estimate | your prediction | true y")
            lines += [
                f"  {h.t + 1} | {h.est_A:.3f} | {h.est_B:.3f} | {h.decision:.3f} | {h.y:.3f}"
                for h in self.history[-self.history_len :]
            ]
            H = self.history
            mse = lambda f: np.mean([(f(h) - h.y) ** 2 for h in H])  # noqa: E731
            lines.append(
                f"Mean squared error over all {len(H)} past steps: A {mse(lambda h: h.est_A):.3f}, "
                f"B {mse(lambda h: h.est_B):.3f}, your prediction {mse(lambda h: h.decision):.3f}."
            )
        lines.append(
            "Reply with your reasoning, the team's prediction of y, and your updated notes."
        )
        return "\n".join(lines)

    def act(
        self,
        t: int,
        x_A: np.ndarray,
        x_B: np.ndarray,
        notes: dict[str, str],
        est: dict[str, float],
    ) -> MediatorReply:
        """Decide step t. An unusable reply falls back to the average of the two estimates."""
        reply = self.backend.complete(
            self.system, self.prompt(t, x_A, x_B, notes, est), MEDIATOR_SCHEMA
        )
        d = reply.data if isinstance(reply.data, dict) else {}
        try:
            decision = float(d["estimate"])
            ok = bool(np.isfinite(decision))
        except (TypeError, KeyError, ValueError):
            ok = False
        if not ok:
            return MediatorReply((est["A"] + est["B"]) / 2, "", self.notes, False, [reply])
        if isinstance(d.get("notes"), str) and d["notes"].strip():
            self.notes = d["notes"][: self.notes_chars]
        return MediatorReply(decision, str(d.get("reasoning", "")), self.notes, True, [reply])

    def record(self, rec: MediatorRecord) -> None:
        self.history.append(rec)


class MediatedExchange(LLMExchange):
    """Self-learning twins that only predict, plus a `Mediator` (config name `llm_mediated`).

    Takes the `LLMExchange` arguments; `n_exchanges` is fixed to 0 and `team_rule` to
    "mediator", and the twins must be self-learning with distinct feature names (the mediator
    tells their features apart by name). The learning phase is as in S1.

    Per step: both twins answer in parallel from their features and notes; the mediator sees
    each twin's features, the notes it predicted with (before this step's rewrite) and its
    estimate, and gives the consensus. Metrics are as in `LLMExchange`, with consensus = the
    mediator's prediction and the twins' final = their only estimate (so the opinion shift is
    0); `mediator_fail` counts unusable mediator replies.

    Parameters
    ----------
    mediator_notes_chars, mediator_history_len : int
        Passed to `Mediator`.
    """

    def __init__(
        self,
        backend: dict,
        mediator_notes_chars: int = 1200,
        mediator_history_len: int = 10,
        **kw,
    ) -> None:
        if kw.pop("n_exchanges", 0) != 0:
            raise ValueError("llm_mediated has no exchanges: the twins only predict")
        if kw.get("learner") != "self" or kw.get("feature_names") != "distinct":
            raise ValueError("llm_mediated needs learner: self and feature_names: distinct")
        kw["team_rule"] = "mediator"
        super().__init__(backend, n_exchanges=0, **kw)
        self.name = "llm_mediated"
        self.mediator_kw = dict(notes_chars=mediator_notes_chars, history_len=mediator_history_len)

    def setup(self, A, B) -> None:
        super().setup(A, B)
        if self.n_train["A"] != self.n_train["B"]:
            raise ValueError(
                "the mediator prompt assumes both twins studied the same number of cases"
            )
        self.mediator = Mediator(
            self.backend,
            self.twins["A"].linear.d,
            self.twins["B"].linear.d,
            self.n_train["A"],
            self.T,
            **self.mediator_kw,
        )

    def step(self, A, B, x_A, x_B, y):
        tA, tB = self.twins["A"], self.twins["B"]
        notes = {"A": tA.notes, "B": tB.notes}  # the notes each twin predicts with this step
        fA = self._pool.submit(tA.act, self.t, 0, x_A, [])
        fB = self._pool.submit(tB.act, self.t, 0, x_B, [])
        rA, rB = fA.result(), fB.result()
        est = {"A": rA.estimate, "B": rB.estimate}
        med = self.mediator.act(self.t, x_A, x_B, notes, est)
        cons = med.estimate
        tA.record(
            StepRecord(self.t, est["A"], est["A"], est["B"], cons, y, partner_open=est["B"]), x_A
        )
        tB.record(
            StepRecord(self.t, est["B"], est["B"], est["A"], cons, y, partner_open=est["A"]), x_B
        )
        self.mediator.record(MediatorRecord(self.t, est["A"], est["B"], cons, y))
        self.transcript.append(
            {
                "t": self.t,
                "y": y,
                "model_pred": est,
                "linear_ref": {"A": tA.model_prediction(x_A), "B": tB.model_prediction(x_B)},
                "consensus": cons,
                "rounds": [
                    {
                        k: {"estimate": r.estimate, "notes": r.notes, "ok": r.ok}
                        for k, r in (("A", rA), ("B", rB))
                    }
                ],
                "mediator": {
                    "estimate": cons,
                    "reasoning": med.reasoning,
                    "notes": med.notes,
                    "ok": med.ok,
                    "twin_notes_shown": notes,
                },
            }
        )
        rec = _errors(cons, est["A"], est["B"], y, est["A"], est["B"])
        backends = [b for r in (rA, rB, med) for b in (r.backends)]
        rec.update(
            parse_fail=float((not rA.ok) + (not rB.ok)),
            mediator_fail=float(not med.ok),
            tokens_in=float(sum(b.tokens_in for b in backends)),
            tokens_out=float(sum(b.tokens_out for b in backends)),
            cost_usd=float(sum(b.cost_usd for b in backends)),
        )
        if self.t == 0:  # charge the learning phase to step 0
            ti, to, c = self._train_cost
            rec["tokens_in"] += ti
            rec["tokens_out"] += to
            rec["cost_usd"] += c
        if self.t == self.T - 1:
            self._pool.shutdown()
        return rec
