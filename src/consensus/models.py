"""Learners and agents for the twin study.

- `OnlineRidge`: recursive least squares with a ridge prior (Sherman-Morrison updates).
- `Agent`: the interface every twin implements (linear now, LLM-backed later).
- `LinearAgent`: one online-ridge "head" per number of partner messages it conditions on.
- `Blender`: online least squares for a consensus blend alpha * m^A + beta * m^B (protocol P1).
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np


class OnlineRidge:
    """Recursive least squares: theta_t = argmin sum_s (y_s - <theta, z_s>)^2 + prior penalty.

    Keeps P_t = (P_0^{-1} + sum_s z_s z_s^T)^{-1} and updates both with Sherman-Morrison, so each
    step is O(k^2). With the default prior (theta_0 = 0, P_0 = I / lam) this is exactly ridge
    regression on all data seen so far.

    Parameters
    ----------
    dim : int
        Feature dimension k.
    lam : float
        Ridge parameter; ignored if `theta` and `P` are given.
    theta : ndarray, shape (k,), optional
        Prior mean (warm start).
    P : ndarray, shape (k, k), optional
        Prior covariance (inverse precision).
    """

    def __init__(
        self,
        dim: int,
        lam: float = 1.0,
        theta: np.ndarray | None = None,
        P: np.ndarray | None = None,
    ) -> None:
        self.theta = np.zeros(dim) if theta is None else np.asarray(theta, float).copy()
        self.P = np.eye(dim) / lam if P is None else np.asarray(P, float).copy()
        self.n = 0  # number of updates seen

    def predict(self, z: np.ndarray) -> float:
        return float(self.theta @ z)

    def update(self, z: np.ndarray, y: float) -> None:
        """One RLS step on the pair (z, y)."""
        Pz = self.P @ z
        gain = Pz / (1.0 + z @ Pz)
        self.theta = self.theta + gain * (y - self.theta @ z)
        self.n += 1
        self.P = self.P - np.outer(gain, Pz)
        # Re-symmetrize to stop round-off from accumulating over long horizons.
        self.P = 0.5 * (self.P + self.P.T)


class Agent(ABC):
    """Interface for a twin. Protocols and the runner only use these methods.

    `messages` is the list of the partner's announcements received so far in the current
    step (empty for a solo prediction). An agent's own announcement is just its prediction.
    """

    @abstractmethod
    def predict(self, x: np.ndarray, messages: Sequence[float]) -> float:
        """Prediction of y from own features x and the partner's messages."""

    @abstractmethod
    def update(self, x: np.ndarray, messages: Sequence[float], y: float) -> None:
        """Learn from the revealed label y (for the predictor that used these messages)."""

    def begin_communication(self, max_messages: int, **kwargs) -> None:
        """Called once after the training phase, before any messages are exchanged."""

    def clone(self) -> "Agent":
        """Independent copy, so every protocol starts from the same trained twins."""
        return copy.deepcopy(self)


class LinearAgent(Agent):
    """Linear twin: head k is an online ridge regression of y on (x, m_1, ..., m_k).

    Head 0 is the solo learner trained in the training phase. At `begin_communication`, head
    k >= 1 is warm-started at (theta_0, 0, ..., 0) with prior covariance
    blockdiag(prior_scale * P_0, msg_prior_var * I_k), i.e. the training phase acts as a prior on
    the own-feature weights. `prior_scale > 1` makes the twin less attached to its training
    opinion. Drift of the x-block of a head away from theta_0 measures "who changes their weights".

    Parameters
    ----------
    d : int
        Own feature dimension.
    lam : float
        Ridge parameter for the training phase.
    name : str
        "A" or "B", for logging.
    """

    def __init__(self, d: int, lam: float = 1.0, name: str = "", intercept: bool = False) -> None:
        self.d = d
        self.name = name
        # With `intercept`, a constant 1 is prepended to x (needed when E[y] != 0, e.g. the
        # product task); the "own block" of every head is then (intercept, x), of size p = d + 1.
        self.intercept = intercept
        self.p = d + int(intercept)
        self.heads: dict[int, OnlineRidge] = {0: OnlineRidge(self.p, lam)}
        self.frozen: set[int] = set()
        self.theta_trained: np.ndarray | None = None
        # Training samples (x, y), kept so other learners (e.g. a self-learning LLM twin) can be
        # shown exactly the data this twin was trained on.
        self._train: list[tuple[np.ndarray, float]] = []

    def _features(self, x: np.ndarray, messages: Sequence[float]) -> np.ndarray:
        const = [1.0] if self.intercept else []
        return np.concatenate([const, x, np.asarray(messages, float)])

    def predict(self, x: np.ndarray, messages: Sequence[float]) -> float:
        return self.heads[len(messages)].predict(self._features(x, messages))

    def update(self, x: np.ndarray, messages: Sequence[float], y: float) -> None:
        if self.theta_trained is None and not messages:
            self._train.append((np.asarray(x, float), float(y)))
        k = len(messages)
        if k not in self.frozen:
            self.heads[k].update(self._features(x, messages), y)

    def begin_communication(
        self,
        max_messages: int,
        prior_scale: float = 1.0,
        msg_prior_var: float = 1.0,
        update_solo: bool = False,
        frozen: bool = False,
    ) -> None:
        """Create heads 1..max_messages from the trained head 0 and set which heads may learn.

        Parameters
        ----------
        max_messages : int
            Largest number of partner messages any prediction will condition on.
        prior_scale : float
            Multiplier on the trained covariance P_0 used as prior for the x-block.
        msg_prior_var : float
            Prior variance of the message coefficients (prior mean 0).
        update_solo : bool
            Whether head 0 (the twin's announced "opinion") keeps learning after training.
        frozen : bool
            Freeze every head (protocols P0 and P1).
        """
        h0 = self.heads[0]
        self.theta_trained = h0.theta.copy()
        for k in range(1, max_messages + 1):
            theta = np.concatenate([h0.theta, np.zeros(k)])
            P = np.zeros((self.p + k, self.p + k))
            P[: self.p, : self.p] = prior_scale * h0.P
            P[self.p :, self.p :] = msg_prior_var * np.eye(k)
            self.heads[k] = OnlineRidge(self.p + k, theta=theta, P=P)
        self.frozen = set(self.heads) if frozen else ({0} if not update_solo else set())

    def training_data(self) -> tuple[np.ndarray, np.ndarray]:
        """The training samples (X, y) seen before `begin_communication`."""
        return np.array([z for z, _ in self._train]), np.array([v for _, v in self._train])

    def own_weights(self, k: int) -> np.ndarray:
        """Coefficients on own features x in head k (with the intercept first if any), shape (p,)."""
        return self.heads[k].theta[: self.p].copy()

    def message_weights(self, k: int) -> np.ndarray:
        """Coefficients on the partner's messages in head k, shape (k,)."""
        return self.heads[k].theta[self.p :].copy()

    def drift(self, k: int) -> float:
        """||own weights of head k - trained solo weights||_2."""
        return float(np.linalg.norm(self.own_weights(k) - self.theta_trained))


class Blender:
    """Online least squares for the consensus yhat = alpha * m^A + beta * m^B (no intercept).

    Starts at plain averaging (alpha = beta = 1/2) with prior covariance `prior_var * I`.
    Both twins see the same (m^A, m^B, y), so they would learn the same blend; one object suffices.
    """

    def __init__(self, prior_var: float = 1.0) -> None:
        self.rls = OnlineRidge(2, theta=np.array([0.5, 0.5]), P=prior_var * np.eye(2))

    @property
    def alpha(self) -> float:
        return float(self.rls.theta[0])

    @property
    def beta(self) -> float:
        return float(self.rls.theta[1])

    def predict(self, m_A: float, m_B: float) -> float:
        return self.rls.predict(np.array([m_A, m_B]))

    def update(self, m_A: float, m_B: float, y: float) -> None:
        self.rls.update(np.array([m_A, m_B]), y)


# ---------------------------------------------------------------------------------------------
# Logistic task: binary y, predictions and messages are probabilities P(y = 1).

P_CLIP = 1e-3  # probabilities are clipped to [P_CLIP, 1 - P_CLIP] before taking logits / logs


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, float)))


def logit(p):
    p = np.clip(np.asarray(p, float), P_CLIP, 1 - P_CLIP)
    return np.log(p / (1 - p))


def log_loss(p, y):
    """-[y log p + (1 - y) log(1 - p)] with p clipped to [P_CLIP, 1 - P_CLIP]."""
    p = np.clip(np.asarray(p, float), P_CLIP, 1 - P_CLIP)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit_logistic(Z: np.ndarray, y: np.ndarray, theta0=None, P0=None, iters: int = 50):
    """MAP logistic regression with Gaussian prior N(theta0, P0) by Newton's method.

    Returns (theta, P) where P is the inverse Hessian at the optimum (Laplace covariance).
    """
    k = Z.shape[1]
    theta0 = np.zeros(k) if theta0 is None else np.asarray(theta0, float)
    prec0 = np.linalg.inv(np.eye(k) if P0 is None else P0)
    theta = theta0.copy()
    for _ in range(iters):
        p = sigmoid(Z @ theta)
        g = Z.T @ (p - y) + prec0 @ (theta - theta0)
        H = (Z * (p * (1 - p))[:, None]).T @ Z + prec0
        step = np.linalg.solve(H, g)
        theta -= step
        if np.max(np.abs(step)) < 1e-10:
            break
    p = sigmoid(Z @ theta)
    H = (Z * (p * (1 - p))[:, None]).T @ Z + prec0
    return theta, np.linalg.inv(H)


class OnlineLogistic:
    """Online Bayesian logistic regression with a Gaussian (Laplace) posterior N(theta, P).

    Each update is one Newton step on the new point with the current posterior as prior
    (the online Laplace / extended-Kalman approximation): P <- (P^{-1} + p(1-p) z z^T)^{-1},
    theta <- theta + P z (y - p). `predict` returns P(y = 1 | z) at the posterior mean.
    """

    def __init__(self, dim: int, lam: float = 1.0, theta=None, P=None) -> None:
        self.theta = np.zeros(dim) if theta is None else np.asarray(theta, float).copy()
        self.P = np.eye(dim) / lam if P is None else np.asarray(P, float).copy()
        self.n = 0

    def predict(self, z: np.ndarray) -> float:
        return float(sigmoid(self.theta @ z))

    def update(self, z: np.ndarray, y: float) -> None:
        p = float(sigmoid(self.theta @ z))
        w = max(p * (1 - p), 1e-6)
        Pz = self.P @ z
        self.P = self.P - np.outer(Pz, Pz) * (w / (1.0 + w * (z @ Pz)))
        self.P = 0.5 * (self.P + self.P.T)
        self.theta = self.theta + self.P @ z * (y - p)
        self.n += 1


class LogisticAgent(LinearAgent):
    """Logistic twin: head k is an online logistic regression of y on (x, logit m_1..m_k).

    Predictions and announcements are probabilities; partner messages enter as log-odds. In
    the training phase head 0 stores its data and is refitted exactly (MAP, ridge prior `lam`)
    at `begin_communication`, so the trained opinion does not depend on the online
    approximation. Heads k >= 1 are warm-started as in `LinearAgent` with the Laplace covariance
    of head 0 as the prior on the own-feature block.
    """

    def __init__(self, d: int, lam: float = 1.0, name: str = "") -> None:
        super().__init__(d, lam, name)
        self.lam = lam
        self.heads = {0: OnlineLogistic(d, lam)}

    def _features(self, x, messages):
        return np.concatenate([x, logit(np.asarray(messages, float))]) if len(messages) else x

    def begin_communication(
        self, max_messages, prior_scale=1.0, msg_prior_var=1.0, update_solo=False, frozen=False
    ):
        if self.theta_trained is None and self._train:
            Z = np.array([z for z, _ in self._train])
            y = np.array([v for _, v in self._train])
            theta, P = fit_logistic(Z, y, P0=np.eye(self.d) / self.lam)
            h0 = self.heads[0]
            h0.theta, h0.P = theta, P
        h0 = self.heads[0]
        self.theta_trained = h0.theta.copy()
        for k in range(1, max_messages + 1):
            theta = np.concatenate([h0.theta, np.zeros(k)])
            P = np.zeros((self.d + k, self.d + k))
            P[: self.d, : self.d] = prior_scale * h0.P
            P[self.d :, self.d :] = msg_prior_var * np.eye(k)
            h = OnlineLogistic(self.d + k, theta=theta, P=P)
            h.n = h0.n
            self.heads[k] = h
        self.frozen = set(self.heads) if frozen else ({0} if not update_solo else set())


class LogisticBlender:
    """Online logistic blend: P(y = 1) = sigmoid(alpha logit m^A + beta logit m^B).

    Starts at averaging the log-odds (alpha = beta = 1/2), prior covariance `prior_var * I`.
    """

    def __init__(self, prior_var: float = 1.0) -> None:
        self.rls = OnlineLogistic(2, theta=np.array([0.5, 0.5]), P=prior_var * np.eye(2))

    alpha = Blender.alpha
    beta = Blender.beta

    def predict(self, m_A: float, m_B: float) -> float:
        return self.rls.predict(logit([m_A, m_B]))

    def update(self, m_A: float, m_B: float, y: float) -> None:
        self.rls.update(logit([m_A, m_B]), y)
