"""Data generator for the linear twin model.

Implements the model in LogBook note `agent-consensus/notes/2026-11-05-initial-idea`:

    y = <x^A, w^A> + <x^B, w^B> + eps,   x = (x^A, x^B) ~ N(0, Sigma),   eps ~ N(0, sigma^2).

Agent A only ever sees the first `d_A` coordinates of x, agent B the last `d_B`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def block_covariance(d_A: int, d_B: int, rho: float) -> np.ndarray:
    """Covariance with identity blocks and cross-correlation `rho` between paired coordinates.

    Sigma_AA = I, Sigma_BB = I and (Sigma_AB)_{jj} = rho for j < min(d_A, d_B), zero elsewhere.
    This is positive definite for |rho| < 1 (each pair is an independent 2x2 block).

    Parameters
    ----------
    d_A, d_B : int
        Feature dimensions seen by agents A and B.
    rho : float
        Correlation between x^A_j and x^B_j, in (-1, 1).

    Returns
    -------
    Sigma : ndarray, shape (d_A + d_B, d_A + d_B)
    """
    if not -1.0 < rho < 1.0:
        raise ValueError(f"rho must be in (-1, 1), got {rho}")
    d = d_A + d_B
    Sigma = np.eye(d)
    for j in range(min(d_A, d_B)):
        Sigma[j, d_A + j] = Sigma[d_A + j, j] = rho
    return Sigma


@dataclass
class LinearTwinEnv:
    """Joint linear-Gaussian environment shared by the two agents.

    Parameters
    ----------
    w_A : ndarray, shape (d_A,)
        True weights on A's features.
    w_B : ndarray, shape (d_B,)
        True weights on B's features.
    Sigma : ndarray, shape (d_A + d_B, d_A + d_B)
        Covariance of the full feature vector x = (x^A, x^B).
    sigma : float
        Standard deviation of the label noise eps.
    """

    w_A: np.ndarray
    w_B: np.ndarray
    Sigma: np.ndarray
    sigma: float

    def __post_init__(self) -> None:
        self.d_A = len(self.w_A)
        self.d_B = len(self.w_B)
        if self.Sigma.shape != (self.d_A + self.d_B,) * 2:
            raise ValueError("Sigma shape does not match d_A + d_B")
        # Cholesky factor for sampling x = L z with z ~ N(0, I).
        self._L = np.linalg.cholesky(self.Sigma)

    @property
    def w(self) -> np.ndarray:
        """Full weight vector (w^A, w^B), shape (d_A + d_B,)."""
        return np.concatenate([self.w_A, self.w_B])

    def sample(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Draw n iid samples.

        Returns
        -------
        x_A : ndarray, shape (n, d_A)
        x_B : ndarray, shape (n, d_B)
        y : ndarray, shape (n,)
        """
        x = rng.standard_normal((n, self.d_A + self.d_B)) @ self._L.T
        y = x @ self.w + self.sigma * rng.standard_normal(n)
        return x[:, : self.d_A], x[:, self.d_A :], y

    @classmethod
    def from_config(cls, cfg: dict, rng: np.random.Generator) -> "LinearTwinEnv":
        """Build from the `env` config block; random weight directions are scaled to `norm_A`, `norm_B`.

        Explicit weights can be given as lists under `w_A` / `w_B` instead.
        """
        d_A, d_B = cfg["d_A"], cfg["d_B"]
        w_A = (
            np.asarray(cfg["w_A"], float)
            if "w_A" in cfg
            else _random_direction(d_A, rng) * cfg["norm_A"]
        )
        w_B = (
            np.asarray(cfg["w_B"], float)
            if "w_B" in cfg
            else _random_direction(d_B, rng) * cfg["norm_B"]
        )
        Sigma = block_covariance(d_A, d_B, cfg["rho"])
        return cls(w_A=w_A, w_B=w_B, Sigma=Sigma, sigma=cfg["sigma"])


def _random_direction(d: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform unit vector in R^d."""
    v = rng.standard_normal(d)
    return v / np.linalg.norm(v)


@dataclass
class LogisticTwinEnv(LinearTwinEnv):
    """Binary-label variant: y ~ Bernoulli(sigmoid(scale * (<x^A, w^A> + <x^B, w^B>))).

    Features and weights are as in `LinearTwinEnv`; `sigma` is unused (kept for the interface).
    `scale` sets the signal strength: the true log-odds have standard deviation
    scale * sqrt(w^T Sigma w).
    """

    scale: float = 1.0

    def sample(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = rng.standard_normal((n, self.d_A + self.d_B)) @ self._L.T
        p = 1.0 / (1.0 + np.exp(-self.scale * (x @ self.w)))
        y = (rng.random(n) < p).astype(float)
        return x[:, : self.d_A], x[:, self.d_A :], y

    def true_prob(self, x_A: np.ndarray, x_B: np.ndarray) -> np.ndarray:
        """P(y = 1 | x^A, x^B) under the true model."""
        z = x_A @ self.w_A + x_B @ self.w_B
        return 1.0 / (1.0 + np.exp(-self.scale * z))

    @classmethod
    def from_config(cls, cfg: dict, rng: np.random.Generator) -> "LogisticTwinEnv":
        lin = LinearTwinEnv.from_config({**cfg, "sigma": cfg.get("sigma", 0.0)}, rng)
        return cls(lin.w_A, lin.w_B, lin.Sigma, lin.sigma, scale=cfg.get("scale", 1.0))


@dataclass
class ProductTwinEnv(LinearTwinEnv):
    """Multiplicative variant: the two blocks' signals multiply instead of adding,

        y = g_A * g_B + eps,   g_A = mu + <x^A, w^A>,   g_B = mu + <x^B, w^B>,   eps ~ N(0, sigma^2).

    At rho = 0 each twin's best solo prediction is E[y | x^A] = mu * g_A (since E[g_B] = mu), and
    the full-information prediction is g_A * g_B = (mu g_A)(mu g_B) / mu^2. With the default
    mu = 1 the best team rule is therefore exactly the product of the two solo predictions, while
    averaging or adding them leaves the interaction <x^A, w^A><x^B, w^B> unexplained (variance
    |w^A|^2 |w^B|^2 at rho = 0). Features and weights are as in `LinearTwinEnv`.
    """

    mu: float = 1.0

    def true_mean(self, x_A: np.ndarray, x_B: np.ndarray) -> np.ndarray:
        """E[y | x^A, x^B] = g_A g_B, shape (n,)."""
        return (self.mu + x_A @ self.w_A) * (self.mu + x_B @ self.w_B)

    def sample(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = rng.standard_normal((n, self.d_A + self.d_B)) @ self._L.T
        x_A, x_B = x[:, : self.d_A], x[:, self.d_A :]
        y = self.true_mean(x_A, x_B) + self.sigma * rng.standard_normal(n)
        return x_A, x_B, y

    @classmethod
    def from_config(cls, cfg: dict, rng: np.random.Generator) -> "ProductTwinEnv":
        lin = LinearTwinEnv.from_config(cfg, rng)
        return cls(lin.w_A, lin.w_B, lin.Sigma, lin.sigma, mu=cfg.get("mu", 1.0))


def build_env(cfg: dict, rng: np.random.Generator) -> LinearTwinEnv:
    """Environment from the `env` config block; `task: logistic` gives binary labels and
    `task: product` multiplies the two blocks' signals (`ProductTwinEnv`)."""
    task = cfg.get("task", "regression")
    if task == "regression":
        return LinearTwinEnv.from_config(cfg, rng)
    if task == "logistic":
        return LogisticTwinEnv.from_config(cfg, rng)
    if task == "product":
        return ProductTwinEnv.from_config(cfg, rng)
    raise ValueError(f"unknown task {task!r}; choose 'regression', 'logistic' or 'product'")
