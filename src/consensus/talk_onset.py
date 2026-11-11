"""When should the twins start talking? (LogBook note 2026-11-11, third idea.)

A fixed budget of n_T labelled steps on one common stream (both twins see the same case, each
its own feature block). For t < s each twin only learns alone (head 0, ridge on its own x) and
the team predicts the average of the two solo predictions. From t = s on the twins talk and
learn to use each other (`mode`):

- "exchange": P2, one exchange; head 1 regresses y on (own x, partner's solo prediction) online,
  warm-started from head 0 as in `LinearAgent.begin_communication`; team = average of the two
  round-1 predictions.
- "blend": P1, team = alpha m^A + beta m^B, with (alpha, beta) learned online from (1/2, 1/2).

`update_solo` says whether each twin's own solo model (its announced opinion) keeps learning
after talk starts. With it on, every label still trains the solo model, so talking early costs
only the non-stationarity of the partner's messages (the head regresses on a message whose
coefficients are still moving). With it off, the opinion is frozen at time s (like an LLM twin's
notes from its learning phase), so starting early means talking with a weaker opinion: a real
trade-off between learning alone and learning to combine.

After the budget, everything is frozen and the team is scored on a held-out test set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from consensus.data import LinearTwinEnv
from consensus.models import Blender, LinearAgent


@dataclass
class TalkOnsetResult:
    """Team test excess MSE (over sigma^2) at each checkpoint step, and at the end."""

    checkpoints: np.ndarray  # steps after which the team was scored, shape (C,)
    excess: np.ndarray  # shape (C,)


class TalkOnset:
    """One pair of linear twins trained with talk starting at step `s` of `n_T`.

    Parameters
    ----------
    env : LinearTwinEnv
    n_T : int
        Labelled steps in the budget.
    s : int
        Steps of learning alone before talking (0 = talk from the start, n_T = never talk).
    mode : str
        "exchange" or "blend".
    update_solo : bool
        Whether the solo models keep learning after talk starts.
    lam, prior_scale, msg_prior_var, blend_prior_var : float
        Ridge parameter of the solo models, and the priors of the communication heads.
    """

    def __init__(
        self,
        env: LinearTwinEnv,
        n_T: int,
        s: int,
        mode: str = "exchange",
        update_solo: bool = True,
        lam: float = 1.0,
        prior_scale: float = 1.0,
        msg_prior_var: float = 1.0,
        blend_prior_var: float = 1.0,
    ) -> None:
        if mode not in ("exchange", "blend"):
            raise ValueError(f"mode must be 'exchange' or 'blend', got {mode!r}")
        if not 0 <= s <= n_T:
            raise ValueError(f"need 0 <= s <= n_T, got s = {s}, n_T = {n_T}")
        self.env, self.n_T, self.s, self.mode = env, n_T, s, mode
        self.update_solo = update_solo
        self.prior = dict(prior_scale=prior_scale, msg_prior_var=msg_prior_var)
        self.A = LinearAgent(env.d_A, lam, "A")
        self.B = LinearAgent(env.d_B, lam, "B")
        self.blender = Blender(blend_prior_var) if mode == "blend" else None
        self.talking = False

    def _start_talking(self) -> None:
        E = 1 if self.mode == "exchange" else 0
        for ag in (self.A, self.B):
            ag.begin_communication(E, update_solo=self.update_solo, **self.prior)
        self.talking = True

    def step(self, xa: np.ndarray, xb: np.ndarray, y: float, t: int) -> None:
        """Learn from case t of the budget."""
        if t == self.s:
            self._start_talking()
        A, B = self.A, self.B
        if not self.talking:  # learning alone
            A.update(xa, [], y)
            B.update(xb, [], y)
            return
        mA, mB = A.predict(xa, []), B.predict(xb, [])
        if self.mode == "blend":
            self.blender.update(mA, mB, y)
            A.update(xa, [], y)  # no-ops for frozen solo heads
            B.update(xb, [], y)
            return
        for ag, x, m in ((A, xa, mB), (B, xb, mA)):
            ag.update(x, [], y)
            ag.update(x, [m], y)

    def team_predict(self, XA: np.ndarray, XB: np.ndarray) -> np.ndarray:
        """Team predictions on a test set (vectorized; heads are linear in their inputs)."""
        tA, tB = self.A.heads[0].theta, self.B.heads[0].theta
        mA, mB = XA @ tA, XB @ tB
        if not self.talking:
            return (mA + mB) / 2
        if self.mode == "blend":
            return self.blender.alpha * mA + self.blender.beta * mB
        hA, hB = self.A.heads[1].theta, self.B.heads[1].theta
        pA = XA @ hA[:-1] + hA[-1] * mB
        pB = XB @ hB[:-1] + hB[-1] * mA
        return (pA + pB) / 2

    def run(
        self, xa: np.ndarray, xb: np.ndarray, y: np.ndarray, test: tuple, checkpoints
    ) -> TalkOnsetResult:
        """Train on the n_T budget cases (xa, xb, y), scoring on `test` = (XA, XB, f) after each
        step in `checkpoints`, where f = E[y | x] (noise-free), so the score is the excess MSE."""
        XA, XB, f = test
        cps = sorted(set(int(c) for c in checkpoints) | {self.n_T})
        out = []
        for t in range(self.n_T):
            self.step(xa[t], xb[t], float(y[t]), t)
            if t + 1 in cps:
                out.append(float(np.mean((self.team_predict(XA, XB) - f) ** 2)))
        return TalkOnsetResult(np.array(cps), np.array(out))
