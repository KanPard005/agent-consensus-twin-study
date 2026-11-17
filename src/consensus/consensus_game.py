"""Consensus game (LogBook note 2026-11-11, "Entrenchment"): the twins must agree, not average.

Each step the twins talk for at most K rounds and stop as soon as their estimates are within
`tol` of each other. The team's prediction is then the agreed value (the average of the two
agreeing estimates); if they have not agreed after round K the team instead pays a fixed
`penalty` (much larger than a typical squared error), so no averaging rescues a disagreement.

- `ConsensusExchange`: linear twins (heads learned online, as in `protocols.Exchange`). Linear
  heads only regress y on (x, partner's messages), so the penalty does not change what they do;
  the protocol records when they would have agreed and the penalized team loss.
- `LLMConsensus`: LLM twins (`LLMTwin` with `team_rule="consensus"`), told about the rule and
  the penalty, so they can trade accuracy for agreement.

Recorded per step, besides `protocols.METRICS`:
    agreed       1 if the twins agreed within K rounds, else 0
    rounds_used  rounds played (1..K for LLMs; 0..E for linear twins, where round 0 is the
                 opening announcement)
    game_loss    (agreed value - y)^2 if agreed, else `penalty`
    agreed_sq_err (agreed value - y)^2, nan if no agreement
    concede_A    share of the opening gap closed by A: (c - m^A_0) / (m^B_0 - m^A_0) for the
                 agreed value c (0 = B came all the way to A, 1 = A went all the way to B;
                 outside [0, 1] = the consensus left the interval). nan if no agreement or the
                 opening gap is below `tol`.

The population benchmark is the Geanakoplos-Polemarchakis dialogue
(`theory.agreement_path`, `theory.agreement_probability`).
"""

from __future__ import annotations

import numpy as np

from consensus.llm_agent import StepRecord
from consensus.protocols import Exchange, LLMExchange, _drift, _errors


def _concession(c: float, open_A: float, open_B: float, tol: float) -> float:
    gap = open_B - open_A
    return float((c - open_A) / gap) if abs(gap) >= tol else float("nan")


class ConsensusExchange(Exchange):
    """Linear twins in the consensus game: E exchanges after the opening (K = E + 1 rounds).

    All rounds are always computed so that every head keeps learning (as in `Exchange`); the
    team's prediction is the average at the first round whose gap is below `tol`.
    """

    def __init__(self, penalty: float = 10.0, **kw) -> None:
        super().__init__(**kw)
        self.penalty = penalty
        self.name = f"consensus_{self.E}"

    def step(self, A, B, x_A, x_B, y):
        m_A, m_B = [A.predict(x_A, [])], [B.predict(x_B, [])]
        for e in range(1, self.E + 1):
            new_A, new_B = A.predict(x_A, m_B[:e]), B.predict(x_B, m_A[:e])
            m_A.append(new_A)
            m_B.append(new_B)
        for e in range(self.E + 1):
            A.update(x_A, m_B[:e], y)
            B.update(x_B, m_A[:e], y)
        gaps = np.abs(np.array(m_A) - np.array(m_B))
        hit = np.flatnonzero(gaps < self.tol)
        agreed = bool(hit.size)
        e = int(hit[0]) if agreed else self.E
        c = (m_A[e] + m_B[e]) / 2
        rec = _errors(c, m_A[e], m_B[e], y, m_A[0], m_B[0], self.logistic)
        rec.update(
            drift_A=_drift(A, self.E),
            drift_B=_drift(B, self.E),
            agree_round=float(e) if agreed else np.nan,
            agreed=float(agreed),
            rounds_used=float(e),
            game_loss=(c - y) ** 2 if agreed else self.penalty,
            agreed_sq_err=(c - y) ** 2 if agreed else np.nan,
            concede_A=_concession(c, m_A[0], m_B[0], self.tol) if agreed else np.nan,
        )
        return rec


class LLMConsensus(LLMExchange):
    """LLM twins in the consensus game: at most `n_exchanges + 1` rounds, stop on agreement.

    Same as `LLMExchange` (self-learning or tool-using twins, notes memory) except that a step
    ends at the first round where |estimate_A - estimate_B| <= `tol`. The twins are told the
    rule, the tolerance and the penalty (`LLMTwin` with `team_rule="consensus"`); `tol` and
    `penalty` are passed on to them. In the history each twin sees, a step without agreement
    shows no team value and counts as `penalty` in the team's mean squared error.
    """

    def __init__(self, penalty: float = 10.0, tol: float = 0.1, **kw) -> None:
        kw.setdefault("team_rule", "consensus")
        if kw["team_rule"] != "consensus":
            raise ValueError("LLMConsensus needs team_rule='consensus'")
        super().__init__(tol=tol, consensus_tol=tol, penalty=penalty, **kw)
        self.penalty = penalty
        self.name = f"llm_consensus_{self.E}"

    def step(self, A, B, x_A, x_B, y):
        tA, tB = self.twins["A"], self.twins["B"]
        rounds, msgs = [], {"A": [], "B": []}
        agreed = False
        for r in range(self.E + 1):
            fA = self._pool.submit(tA.act, self.t, r, x_A, list(msgs["B"]))
            fB = self._pool.submit(tB.act, self.t, r, x_B, list(msgs["A"]))
            rA, rB = fA.result(), fB.result()
            msgs["A"].append((rA.message, rA.estimate))
            msgs["B"].append((rB.message, rB.estimate))
            rounds.append({"A": rA, "B": rB})
            if abs(rA.estimate - rB.estimate) <= self.tol + 1e-12:
                agreed = True
                break
        open_A, open_B = rounds[0]["A"].estimate, rounds[0]["B"].estimate
        fin_A, fin_B = rounds[-1]["A"].estimate, rounds[-1]["B"].estimate
        cons = (fin_A + fin_B) / 2
        team = cons if agreed else float("nan")  # what the twins see as the team's value
        tA.record(StepRecord(self.t, open_A, fin_A, fin_B, team, y, partner_open=open_B), x_A)
        tB.record(StepRecord(self.t, open_B, fin_B, fin_A, team, y, partner_open=open_A), x_B)

        replies = [rd[k] for rd in rounds for k in ("A", "B")]
        self.transcript.append(
            {
                "t": self.t,
                "y": y,
                "model_pred": {"A": open_A, "B": open_B},
                **(
                    {"linear_ref": {"A": tA.model_prediction(x_A), "B": tB.model_prediction(x_B)}}
                    if self.self_learn
                    else {}
                ),
                "consensus": cons,
                "agreed": agreed,
                "rounds": [
                    {
                        k: {
                            "message": rd[k].message,
                            "estimate": rd[k].estimate,
                            "notes": rd[k].notes,
                            "ok": rd[k].ok,
                        }
                        for k in ("A", "B")
                    }
                    for rd in rounds
                ],
            }
        )
        rec = _errors(cons, fin_A, fin_B, y, open_A, open_B, self.logistic)
        rec.update(
            agree_round=float(len(rounds) - 1) if agreed else np.nan,
            agreed=float(agreed),
            rounds_used=float(len(rounds)),
            game_loss=(cons - y) ** 2 if agreed else self.penalty,
            agreed_sq_err=(cons - y) ** 2 if agreed else np.nan,
            concede_A=_concession(cons, open_A, open_B, self.tol) if agreed else np.nan,
            parse_fail=float(sum(not rp.ok for rp in replies)),
            msg_chars=float(np.mean([len(rp.message) for rp in replies])),
            tokens_in=float(sum(b.tokens_in for rp in replies for b in rp.backends)),
            tokens_out=float(sum(b.tokens_out for rp in replies for b in rp.backends)),
            cost_usd=float(sum(b.cost_usd for rp in replies for b in rp.backends)),
        )
        if self.t == 0 and self.self_learn:  # charge the learning phase to step 0
            ti, to, c = self._train_cost
            rec["tokens_in"] += ti
            rec["tokens_out"] += to
            rec["cost_usd"] += c
        if self.t == self.T - 1:
            self._pool.shutdown()
        return rec
