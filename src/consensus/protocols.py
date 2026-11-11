"""Communication protocols for the post-training phase (t > n).

At each step t the twins see x^A_t, x^B_t, produce a joint (consensus) prediction, and then
the joint label y_t is revealed to both (online feedback).

- P0 `NoTalk`: each twin predicts alone; consensus = average.
- P1 `FrozenBlend`: twins frozen at their trained opinions; consensus = alpha m^A + beta m^B,
  with (alpha, beta) learned online.
- P4 `LLMExchange`: LLM twins that use their trained linear learner as a tool and decide
  themselves what to say, what to estimate and how to treat the partner's messages.
- P2/P3 `Exchange`: twins announce predictions simultaneously for E exchanges; in exchange e each
  twin predicts from (own x, partner's messages 1..e) with a head learned online. P2 is E = 1,
  P3 is E > 1. Consensus = average of the final announcements.

Population limits of each protocol are in `consensus.theory`.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from consensus.llm_agent import LLMTwin, StepRecord
from consensus.llm_backends import build_backend
from consensus.models import (
    Agent,
    Blender,
    LinearAgent,
    LogisticAgent,
    LogisticBlender,
    log_loss,
    logit,
    sigmoid,
)

# Per-step quantities every protocol records; arrays of shape (T,) in `Protocol.run`.
METRICS = (
    "sq_err",  # (consensus - y)^2
    "sq_err_A",  # (final prediction of A - y)^2
    "sq_err_B",
    "disagreement",  # (final A - final B)^2
    "shift_A",  # (final prediction of A - A's solo prediction on the same x)^2: opinion shift
    "shift_B",
    "drift_A",  # ||own-feature weights of A's final head - trained weights|| (linear agents only)
    "drift_B",
    "alpha",  # blend weights (P1 only, else nan)
    "beta",
    "agree_round",  # first exchange with |m^A - m^B| < tol (Exchange only, else nan)
)
# Logistic task: predictions are probabilities, `sq_err` is the Brier score, and also
# "log_loss" (of the consensus), "log_loss_A", "log_loss_B".


class Protocol(ABC):
    """Base class: `setup` once on fresh clones of the trained twins, then `step` per time."""

    name: str = "protocol"

    @abstractmethod
    def setup(self, A: Agent, B: Agent) -> None: ...

    @abstractmethod
    def step(self, A: Agent, B: Agent, x_A: np.ndarray, x_B: np.ndarray, y: float) -> dict:
        """Predict, reveal y, update. Returns a dict with the keys in METRICS."""

    def run(
        self, A: Agent, B: Agent, x_A: np.ndarray, x_B: np.ndarray, y: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Run the protocol over a stream.

        Parameters
        ----------
        A, B : Agent
            Trained twins (modified in place; pass clones).
        x_A : ndarray, shape (T, d_A)
        x_B : ndarray, shape (T, d_B)
        y : ndarray, shape (T,)

        Returns
        -------
        dict mapping each name in METRICS (plus any protocol-specific metric) to an ndarray of
        shape (T,).
        """
        self.T = len(y)
        self.logistic = isinstance(A, LogisticAgent)
        self.setup(A, B)
        out = {k: np.full(len(y), np.nan) for k in METRICS}
        for t in range(len(y)):
            self.t = t
            rec = self.step(A, B, x_A[t], x_B[t], float(y[t]))
            for k, v in rec.items():
                out.setdefault(k, np.full(len(y), np.nan))[t] = v
        return out


def _drift(agent: Agent, head: int) -> float:
    return agent.drift(head) if isinstance(agent, LinearAgent) else np.nan


def _errors(
    cons: float,
    p_A: float,
    p_B: float,
    y: float,
    solo_A: float,
    solo_B: float,
    logistic: bool = False,
) -> dict:
    """Per-step errors; `solo_*` are the twins' solo (round-0) predictions on the same x.

    The opinion shift (p_A - solo_A)^2 measures how far communication moved A's prediction. Unlike
    the own-weight drift it does not depend on how the predictor is parameterized (a linear head
    can trade own-feature weight against a message that is itself partly a function of x^A), and
    it is defined for any `Agent`. Its mean is the population quantity
    (v_A^E - v_A^0)^T Sigma (v_A^E - v_A^0) in `consensus.theory.agreement_path`.
    """
    out = {
        "sq_err": (cons - y) ** 2,
        "sq_err_A": (p_A - y) ** 2,
        "sq_err_B": (p_B - y) ** 2,
        "disagreement": (p_A - p_B) ** 2,
        "shift_A": (p_A - solo_A) ** 2,
        "shift_B": (p_B - solo_B) ** 2,
    }
    if logistic:
        out.update(
            log_loss=float(log_loss(cons, y)),
            log_loss_A=float(log_loss(p_A, y)),
            log_loss_B=float(log_loss(p_B, y)),
        )
    return out


class NoTalk(Protocol):
    """P0: no communication; consensus is the average of the solo predictions.

    `pool="logit_sum"` (logistic task only) instead combines the two probabilities by adding
    their log-odds, sigmoid(logit p^A + logit p^B): the naive-Bayes pool, exact when the two
    feature blocks are independent and each solo model is calibrated with no prior log-odds.
    `pool="product"` (regression task) multiplies the two solo predictions, p^A p^B: the right
    rule for the product task (`data.ProductTwinEnv`) at rho = 0 with mu = 1.
    """

    name = "no_talk"

    def __init__(self, update_solo: bool = False, pool: str = "mean") -> None:
        if pool not in ("mean", "logit_sum", "product"):
            raise ValueError(f"pool must be 'mean', 'logit_sum' or 'product', got {pool!r}")
        self.update_solo = update_solo
        self.pool = pool
        if pool != "mean":
            self.name = pool

    def setup(self, A: Agent, B: Agent) -> None:
        for agent in (A, B):
            agent.begin_communication(0, update_solo=self.update_solo, frozen=not self.update_solo)

    def step(self, A, B, x_A, x_B, y):
        p_A, p_B = A.predict(x_A, []), B.predict(x_B, [])
        A.update(x_A, [], y)
        B.update(x_B, [], y)
        if self.pool == "logit_sum":
            if not self.logistic:
                raise ValueError("pool='logit_sum' needs the logistic task")
            cons = float(sigmoid(logit(p_A) + logit(p_B)))
        elif self.pool == "product":
            if self.logistic:
                raise ValueError("pool='product' needs a regression task")
            cons = p_A * p_B
        else:
            cons = (p_A + p_B) / 2
        rec = _errors(cons, p_A, p_B, y, p_A, p_B, self.logistic)
        rec.update(drift_A=_drift(A, 0), drift_B=_drift(B, 0))
        return rec


class FrozenBlend(Protocol):
    """P1: twins hold their trained weights fixed; the blend (alpha, beta) is learned online."""

    name = "blend"

    def __init__(self, prior_var: float = 1.0) -> None:
        self.prior_var = prior_var

    def setup(self, A: Agent, B: Agent) -> None:
        for agent in (A, B):
            agent.begin_communication(0, frozen=True)
        self.blender = (LogisticBlender if self.logistic else Blender)(self.prior_var)

    def step(self, A, B, x_A, x_B, y):
        m_A, m_B = A.predict(x_A, []), B.predict(x_B, [])
        cons = self.blender.predict(m_A, m_B)
        self.blender.update(m_A, m_B, y)
        # Per-agent "final predictions" are their unchanged opinions.
        rec = _errors(cons, m_A, m_B, y, m_A, m_B, self.logistic)
        rec.update(alpha=self.blender.alpha, beta=self.blender.beta, drift_A=0.0, drift_B=0.0)
        return rec


class Exchange(Protocol):
    """P2 (n_exchanges = 1) and P3 (n_exchanges > 1): iterated simultaneous announcements.

    Round 0: m^A_0 = A's solo prediction, m^B_0 likewise. Round e = 1..E:
        m^A_e = A.predict(x^A, [m^B_0, ..., m^B_{e-1}]),  m^B_e symmetric.
    All E + 1 rounds are always played so every head gets feedback; `agree_round` records the
    first round where |m^A_e - m^B_e| < tol (nan if never).

    Parameters
    ----------
    n_exchanges : int
        E >= 1.
    tol : float
        Agreement tolerance for the `agree_round` metric.
    prior_scale, msg_prior_var, update_solo
        Passed to `LinearAgent.begin_communication`.
    """

    def __init__(
        self,
        n_exchanges: int = 1,
        tol: float = 1e-2,
        prior_scale: float = 1.0,
        msg_prior_var: float = 1.0,
        update_solo: bool = False,
    ) -> None:
        if n_exchanges < 1:
            raise ValueError("n_exchanges must be >= 1")
        self.E = n_exchanges
        self.tol = tol
        self.kw = dict(
            prior_scale=prior_scale, msg_prior_var=msg_prior_var, update_solo=update_solo
        )
        self.name = f"exchange_{n_exchanges}"

    def setup(self, A: Agent, B: Agent) -> None:
        for agent in (A, B):
            agent.begin_communication(self.E, **self.kw)

    def step(self, A, B, x_A, x_B, y):
        m_A, m_B = [A.predict(x_A, [])], [B.predict(x_B, [])]
        for e in range(1, self.E + 1):
            # Simultaneous: both use only the partner's messages from rounds < e.
            new_A, new_B = A.predict(x_A, m_B[:e]), B.predict(x_B, m_A[:e])
            m_A.append(new_A)
            m_B.append(new_B)
        for e in range(self.E + 1):
            A.update(x_A, m_B[:e], y)
            B.update(x_B, m_A[:e], y)
        gaps = np.abs(np.array(m_A) - np.array(m_B))
        hit = np.flatnonzero(gaps < self.tol)
        rec = _errors((m_A[-1] + m_B[-1]) / 2, m_A[-1], m_B[-1], y, m_A[0], m_B[0], self.logistic)
        rec.update(
            drift_A=_drift(A, self.E),
            drift_B=_drift(B, self.E),
            agree_round=float(hit[0]) if hit.size else np.nan,
        )
        return rec


class LLMExchange(Protocol):
    """LLM twins (`consensus.llm_agent.LLMTwin`) talking for `n_exchanges` + 1 rounds per step.

    Round 0: each twin sees its features, its linear model's prediction and its notes, and sends
    a message and an estimate. Round e >= 1: the same plus the partner's messages from rounds < e
    (simultaneous moves, as in `Exchange`). Consensus = average of the final-round estimates.
    Both twins' calls in a round run in parallel. The per-step transcript is kept in
    `self.transcript` for the runner to save; with `memory: persistent`, each twin's full
    conversation (one per run) is kept in `self.conversations`.

    Parameters
    ----------
    backend : dict
        Backend config for `consensus.llm_backends.build_backend`, e.g.
        {kind: cli, model: claude-haiku-5-5, effort: low}.
    n_exchanges : int
        E >= 0 (E = 0: each twin answers once, without seeing the partner).
    tol : float
        Agreement tolerance for `agree_round`.
    **twin_kw
        Passed to `LLMTwin` (history_len, max_chars, notes_chars, tell_twin, update_linear,
        team_rule, show_partner_estimates, tools, max_tool_calls, auto_regress, memory,
        show_own_messages, pooling_hint, combine_hint, learner, train_batch, feature_names,
        ask_interval, train_tool, train_tool_calls).
    self_n_A, self_n_B : int, optional
        `learner: self` only: the twin learns from the first `self_n` of its linear learner's
        training samples (default all), and is told it studied that many cases. Nested subsets
        of one training set, so twins with different n are paired on the same data.
    self_offset_A, self_offset_B : int
        `learner: self` only: start of the twin's slice of training samples, i.e. it studies
        samples [offset, offset + self_n). Used to give two pairs disjoint training sets.

    With `learner: self` the twins first learn from their own training samples (the same ones
    their linear learners were trained on; `LLMTwin.train`, both twins in parallel), the batch
    notes are kept in `self.training`, and the training cost is added to step 0's cost and
    token metrics. Each twin's solo opinion (for the opinion shift) is then its round-1
    estimate, and the transcript also records the linear learners' predictions (`linear_ref`,
    never shown to the twins) for comparison.
    """

    twin_cls = LLMTwin  # subclasses swap the twin, e.g. `consensus.worded.WordedExchange`

    def __init__(
        self,
        backend: dict,
        n_exchanges: int = 1,
        tol: float = 1e-2,
        self_n_A: int | None = None,
        self_n_B: int | None = None,
        self_offset_A: int = 0,
        self_offset_B: int = 0,
        **twin_kw,
    ) -> None:
        if (self_n_A or self_n_B or self_offset_A or self_offset_B) and twin_kw.get(
            "learner", "linear"
        ) != "self":
            raise ValueError("self_n_* / self_offset_* need learner: self")
        self.self_n = {"A": self_n_A, "B": self_n_B}
        self.self_offset = {"A": self_offset_A, "B": self_offset_B}
        self.backend_cfg = backend
        self.E = n_exchanges
        self.tol = tol
        self.twin_kw = twin_kw
        self.name = f"llm_exchange_{n_exchanges}"
        self.transcript: list[dict] = []

    def setup(self, A: Agent, B: Agent) -> None:
        for agent in (A, B):
            agent.begin_communication(0, frozen=True)
        backend = build_backend(self.backend_cfg)
        self.backend = backend  # kept for subclasses that add agents (`consensus.mediator`)
        n_train = {}
        for name, lin in (("A", A), ("B", B)):
            n_all = lin.heads[0].n
            off = self.self_offset[name]
            n = self.self_n[name] or n_all - off
            if off + n > n_all:
                raise ValueError(
                    f"self_offset_{name} + self_n_{name} = {off + n} exceeds the {n_all} "
                    "training samples"
                )
            n_train[name] = n
        self.n_train = n_train  # cases each twin studies (learner: self) / its learner's n
        self.twins = {
            name: self.twin_cls(
                name, lin, backend, self.E + 1, self.T, n_train[name], **self.twin_kw
            )
            for name, lin in (("A", A), ("B", B))
        }
        self.transcript = []
        self.conversations = {}  # persistent memory: {twin name: list of turns}, filled at the end
        self._pool = ThreadPoolExecutor(max_workers=2)
        self.self_learn = self.twin_kw.get("learner", "linear") == "self"
        self.training = {}
        self._train_cost = (0.0, 0.0, 0.0)
        if self.self_learn:
            futs = {}
            for (name, tw), lin in zip(self.twins.items(), (A, B)):
                X, y_tr = lin.training_data()
                off, n = self.self_offset[name], n_train[name]
                futs[name] = self._pool.submit(tw.train, X[off : off + n], y_tr[off : off + n])
            logs = {name: f.result() for name, f in futs.items()}
            reps = [
                r
                for log in logs.values()
                for rec in log
                for r in [rec.pop("reply"), *rec.pop("tool_replies", [])]
            ]
            self.training = logs
            self._train_cost = (
                sum(r.tokens_in for r in reps),
                sum(r.tokens_out for r in reps),
                sum(r.cost_usd for r in reps),
            )

    def fork(self) -> LLMExchange:
        """An independent copy of this pair in its current state (between steps), to continue
        it along a different branch: twins forked (`LLMTwin.fork`), transcript copied, own thread
        pool. The learning-phase cost stays with the original."""
        new = copy.copy(self)
        new.twins = {k: tw.fork() for k, tw in self.twins.items()}
        new.transcript = list(self.transcript)
        new._pool = ThreadPoolExecutor(max_workers=2)
        new._train_cost = (0.0, 0.0, 0.0)
        new.conversations = {}
        return new

    def step(self, A, B, x_A, x_B, y):
        tA, tB = self.twins["A"], self.twins["B"]
        rounds = []
        msgs = {"A": [], "B": []}
        for r in range(self.E + 1):
            fA = self._pool.submit(tA.act, self.t, r, x_A, list(msgs["B"]))
            fB = self._pool.submit(tB.act, self.t, r, x_B, list(msgs["A"]))
            rA, rB = fA.result(), fB.result()
            msgs["A"].append((rA.message, rA.estimate))
            msgs["B"].append((rB.message, rB.estimate))
            rounds.append({"A": rA, "B": rB})
        if self.self_learn:  # solo opinion = round-1 estimate, before seeing the partner
            solo_A, solo_B = rounds[0]["A"].estimate, rounds[0]["B"].estimate
        else:
            solo_A, solo_B = tA.model_prediction(x_A), tB.model_prediction(x_B)
        fin_A, fin_B = rounds[-1]["A"].estimate, rounds[-1]["B"].estimate
        cons = (fin_A + fin_B) / 2
        open_A, open_B = rounds[0]["A"].estimate, rounds[0]["B"].estimate
        tA.record(StepRecord(self.t, solo_A, fin_A, fin_B, cons, y, partner_open=open_B), x_A)
        tB.record(StepRecord(self.t, solo_B, fin_B, fin_A, cons, y, partner_open=open_A), x_B)

        replies = [rd[k] for rd in rounds for k in ("A", "B")]
        gaps = np.array([abs(rd["A"].estimate - rd["B"].estimate) for rd in rounds])
        hit = np.flatnonzero(gaps < self.tol)
        self.transcript.append(
            {
                "t": self.t,
                "y": y,
                "model_pred": {"A": solo_A, "B": solo_B},
                **(
                    {"linear_ref": {"A": tA.model_prediction(x_A), "B": tB.model_prediction(x_B)}}
                    if self.self_learn
                    else {}
                ),
                "consensus": cons,
                "rounds": [
                    {
                        k: {
                            "message": rd[k].message,
                            "estimate": rd[k].estimate,
                            "notes": rd[k].notes,
                            "ok": rd[k].ok,
                            "tool_calls": rd[k].tool_calls,
                            **(
                                {"interval": list(rd[k].interval)}
                                if rd[k].interval is not None
                                else {}
                            ),
                        }
                        for k in ("A", "B")
                    }
                    for rd in rounds
                ],
            }
        )
        rec = _errors(cons, fin_A, fin_B, y, solo_A, solo_B, self.logistic)
        rec.update(
            agree_round=float(hit[0]) if hit.size else np.nan,
            parse_fail=float(sum(not rp.ok for rp in replies)),
            msg_chars=float(np.mean([len(rp.message) for rp in replies])),
            tokens_in=float(sum(b.tokens_in for rp in replies for b in rp.backends)),
            tokens_out=float(sum(b.tokens_out for rp in replies for b in rp.backends)),
            cost_usd=float(sum(b.cost_usd for rp in replies for b in rp.backends)),
            # Tool calls the twins asked for (automatic runs are not counted).
            tool_calls=float(sum(1 for rp in replies for c in rp.tool_calls if not c.get("auto"))),
        )
        if self.t == 0 and self.self_learn:  # charge the learning phase to step 0
            ti, to, c = self._train_cost
            rec["tokens_in"] += ti
            rec["tokens_out"] += to
            rec["cost_usd"] += c
        if self.t == self.T - 1:
            self._pool.shutdown()
            self.conversations = {
                k: tw.conv.turns for k, tw in self.twins.items() if tw.conv is not None
            }
        return rec


class SwitchExchange(Protocol):
    """Partner switch: two LLM pairs, and partway through, the B twins change pairs.

    Pairs 1 = (A1, B1) and 2 = (A2, B2) are `LLMExchange` pairs on the same stream; with
    `learner: self` and `pair_n`, pair k studies training samples [k n, (k + 1) n), so the two
    pairs grow up on disjoint data and their partners really differ. At step `switch_at` both
    pairs are forked (`LLMExchange.fork`):

    - streams p1, p2 continue unchanged (the control);
    - streams s1 = (A1, B2) and s2 = (A2, B1) continue with swapped partners. Each twin keeps its
      feature block, notes and history; nothing tells it that the partner changed.

    So s and p share the steps before `switch_at` and then differ only by the switch, which
    pairs the comparison run by run. The main METRICS are the mean over the two pairs' streams
    (p1, p2 before the switch, s1, s2 after it), the control's are prefixed `ctrl_`, and the
    per-stream ones `p1_`, `p2_`, `s1_`, `s2_` (nan where the stream does not exist yet); LLM
    usage totals are summed over all streams. The transcript has one record per stream and step
    (`stream` key; s1/s2 start at `switch_at`) and `training` holds the four twins' notes
    (A1, B1, A2, B2).
    """

    USAGE = ("parse_fail", "tokens_in", "tokens_out", "cost_usd", "tool_calls")

    def __init__(self, switch_at: int, pair_n: int | None = None, **llm_kw) -> None:
        self.switch_at = switch_at
        self.pair_n = pair_n
        self.llm_kw = llm_kw
        self.name = "llm_switch"
        self.transcript: list[dict] = []

    def setup(self, A: Agent, B: Agent) -> None:
        if not 0 < self.switch_at < self.T:
            raise ValueError(f"switch_at must be in (0, T={self.T}), got {self.switch_at}")
        self_learn = self.llm_kw.get("learner", "linear") == "self"
        self.pairs = {}
        for k in range(2):
            extra = {}
            if self_learn and self.pair_n:
                off = k * self.pair_n
                extra = dict(
                    self_n_A=self.pair_n,
                    self_n_B=self.pair_n,
                    self_offset_A=off,
                    self_offset_B=off,
                )
            eng = LLMExchange(**{**self.llm_kw, **extra})
            eng.T, eng.logistic = self.T, self.logistic
            self.pairs[f"p{k + 1}"] = eng
        self._pool = ThreadPoolExecutor(max_workers=4)
        futs = [self._pool.submit(e.setup, A.clone(), B.clone()) for e in self.pairs.values()]
        for f in futs:
            f.result()
        self.training = {
            f"{twin}{key[1]}": batches
            for key, eng in self.pairs.items()
            for twin, batches in eng.training.items()
        }
        self.streams = dict(self.pairs)  # stream name -> pair; s1/s2 are added at the switch
        self.transcript = []
        self.conversations = {}

    def _switch(self) -> None:
        s1, s2 = self.pairs["p1"].fork(), self.pairs["p2"].fork()
        s1.twins["B"], s2.twins["B"] = s2.twins["B"], s1.twins["B"]
        self.streams.update(s1=s1, s2=s2)

    def step(self, A, B, x_A, x_B, y):
        if self.t == self.switch_at:
            self._switch()
        futs = {}
        for name, eng in self.streams.items():
            eng.t = self.t
            futs[name] = self._pool.submit(eng.step, A, B, x_A, x_B, y)
        recs = {name: f.result() for name, f in futs.items()}
        main = ("s1", "s2") if self.t >= self.switch_at else ("p1", "p2")
        out = {}
        for k in recs["p1"]:
            for name in ("p1", "p2", "s1", "s2"):
                out[f"{name}_{k}"] = recs[name][k] if name in recs else np.nan
            if k in self.USAGE:
                out[k] = float(sum(rec[k] for rec in recs.values()))
            else:
                out[k] = float(np.mean([recs[n][k] for n in main]))
                out[f"ctrl_{k}"] = float(np.mean([recs["p1"][k], recs["p2"][k]]))
        for name, eng in self.streams.items():
            self.transcript.append({"stream": name, **eng.transcript[-1]})
        if self.t == self.T - 1:
            self._pool.shutdown()
        return out


def build_protocol(cfg: dict) -> Protocol:
    """Construct a protocol from a config entry like {name: exchange, n_exchanges: 3, ...}."""
    cfg = dict(cfg)
    name = cfg.pop("name")
    cfg.pop("label", None)
    registry = {
        "no_talk": NoTalk,
        "blend": FrozenBlend,
        "exchange": Exchange,
        "llm_exchange": LLMExchange,
        "llm_switch": SwitchExchange,
    }
    if name == "llm_mediated":  # imported here: consensus.mediator imports this module
        from consensus.mediator import MediatedExchange

        registry[name] = MediatedExchange
    if name in ("consensus", "llm_consensus"):  # consensus.consensus_game imports this module
        from consensus.consensus_game import ConsensusExchange, LLMConsensus

        registry.update(consensus=ConsensusExchange, llm_consensus=LLMConsensus)
    if name == "llm_worded":  # imported here: consensus.worded imports this module
        from consensus.worded import WordedExchange

        registry[name] = WordedExchange
    if name not in registry:
        raise ValueError(f"Unknown protocol {name!r}; choose from {sorted(registry)}")
    return registry[name](**cfg)
