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


def random_orthogonal(d: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-distributed d x d orthogonal matrix (QR of a Gaussian matrix, with the sign fix)."""
    Q, R = np.linalg.qr(rng.standard_normal((d, d)))
    return Q * np.sign(np.diag(R))


def controlled_env_parts(
    d: int,
    lambda_xz: float,
    rng: np.random.Generator,
    rotate: bool = True,
    align: float = 1.0,
    normalize: bool = False,
    alpha: float = 1.0,
    weak: str = "B",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Covariance and weights of the controlled design (LogBook note 2026-11-11, "More controlled
    experiment on differences").

    Sigma = [[I, lambda U V^T], [lambda V U^T, I]], so the cross-covariance has SVD U (lambda I) V^T,
    and the weights are w_A = U s_A, w_B = V s_B with s_A = 1_d / sqrt(d), s_B = align * s_A.
    The population solo opinions then disagree by

        E[(yhat_A - yhat_B)^2] = 2 (1 - lambda^2)(1 + align * lambda),

    which peaks at lambda = 1/3 for align = 1. The signal variance is 2 (1 + align * lambda);
    `normalize` divides the weights by its square root so that it is 1 for every lambda, and the
    disagreement becomes 1 - lambda^2.

    Parameters
    ----------
    d : int
        Dimension of each block (d_A = d_B = d).
    lambda_xz : float
        The common singular value of the cross-covariance, in (-1, 1).
    rng : numpy Generator
        Draws U and V (only when `rotate`).
    rotate : bool
        Random orthogonal U, V; otherwise U = V = I (the paired covariance with rho = lambda).
    align : float
        c = s_A' s_B, +1 (aligned) or -1 (opposed).
    normalize : bool
        Scale the weights so that Var(<x, w>) = 1.
    alpha : float
        Asymmetric information (LogBook 2026-11-14): the `weak` twin's weights are multiplied by
        alpha, so s_strong = 1_d / sqrt(d) and s_weak = alpha * align * 1_d / sqrt(d). With
        align = 1 the population solo excess errors are alpha^2 (1 - lambda^2) for the strong twin
        and (1 - lambda^2) for the weak one (see `consensus.theory.asymmetric_limits`); alpha = 0
        with lambda > 0 nests the weak twin's information inside the strong one's.
    weak : "A" or "B"
        Which twin gets the scaled weights (alternated across pairs to balance label effects).

    Returns
    -------
    Sigma : ndarray, shape (2d, 2d)
    w_A, w_B : ndarray, shape (d,)
    """
    if not -1.0 < lambda_xz < 1.0:
        raise ValueError(f"lambda_xz must be in (-1, 1), got {lambda_xz}")
    U = random_orthogonal(d, rng) if rotate else np.eye(d)
    V = random_orthogonal(d, rng) if rotate else np.eye(d)
    C = lambda_xz * U @ V.T
    Sigma = np.block([[np.eye(d), C], [C.T, np.eye(d)]])
    if weak not in ("A", "B"):
        raise ValueError(f"weak must be 'A' or 'B', got {weak!r}")
    s = np.ones(d) / np.sqrt(d)
    # Coefficients of the two weight vectors along s (in the U and V bases).
    c_A = alpha if weak == "A" else 1.0
    c_B = align * (alpha if weak == "B" else 1.0)
    # Signal variance ||s_A||^2 + ||s_B||^2 + 2 lambda s_A's_B; equals 2 (1 + align lambda) at alpha = 1.
    var = c_A**2 + c_B**2 + 2.0 * lambda_xz * c_A * c_B
    scale = 1.0 / np.sqrt(var) if normalize else 1.0
    return Sigma, scale * c_A * U @ s, scale * c_B * V @ s


def overlap_env_parts(
    d_A: int, d_B: int, k: int, w: list[float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Covariance and weights when the twins share k of D = d_A + d_B - k independent features.

    The features z_1..z_D are iid N(0, 1) and y = <z, w> + eps. A sees z_1..z_{d_A}, B sees
    z_{d_A-k+1}..z_D, so the last k of A's features are the first k of B's (the same variables, and
    in the prompts the same names). In the twin model's (x^A, x^B) coordinates the shared weights
    are put on A's block and B's copies get weight 0; the pair (x^A_{d_A-k+j}, x^B_j) has
    correlation 1, so Sigma is singular (sampled with a square root, not Cholesky).

    Parameters
    ----------
    d_A, d_B : int
        Number of features each twin sees.
    k : int
        Number of shared features, 0 <= k <= min(d_A, d_B).
    w : list of float, length D
        True weights on z_1..z_D.

    Returns
    -------
    Sigma : ndarray, shape (d_A + d_B, d_A + d_B)
    w_A : ndarray, shape (d_A,)
    w_B : ndarray, shape (d_B,)
    """
    if not 0 <= k <= min(d_A, d_B):
        raise ValueError(f"overlap must be in [0, min(d_A, d_B)], got {k}")
    D = d_A + d_B - k
    w = np.asarray(w, float)
    if w.shape != (D,):
        raise ValueError(f"w must have d_A + d_B - overlap = {D} entries, got {w.shape}")
    # Selection matrix S (d_A + d_B, D): x = S z.
    S = np.zeros((d_A + d_B, D))
    S[np.arange(d_A), np.arange(d_A)] = 1.0
    S[d_A + np.arange(d_B), d_A - k + np.arange(d_B)] = 1.0
    w_B = np.concatenate([np.zeros(k), w[d_A:]])
    return S @ S.T, w[:d_A].copy(), w_B


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
        # Factor for sampling x = L z with z ~ N(0, I): Cholesky, or a symmetric square root when
        # Sigma is singular (`cov: overlap`, where a shared feature appears in both blocks).
        try:
            self._L = np.linalg.cholesky(self.Sigma)
        except np.linalg.LinAlgError:
            vals, vecs = np.linalg.eigh(self.Sigma)
            self._L = vecs * np.sqrt(np.clip(vals, 0.0, None))

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

        `cov: controlled` uses `controlled_env_parts` instead (keys `lambda_xz`, `rotate`, `align`,
        `normalize`, `alpha`, `weak`; needs d_A = d_B; `rho`, `norm_*` and explicit weights are ignored).

        `cov: overlap` uses `overlap_env_parts` (keys `overlap`, `w`): independent features, some
        of them seen by both twins.
        """
        d_A, d_B = cfg["d_A"], cfg["d_B"]
        if cfg.get("cov", "paired") == "overlap":
            Sigma, w_A, w_B = overlap_env_parts(d_A, d_B, cfg["overlap"], cfg["w"])
            return cls(w_A=w_A, w_B=w_B, Sigma=Sigma, sigma=cfg["sigma"])
        if cfg.get("cov", "paired") == "controlled":
            if d_A != d_B:
                raise ValueError("cov: controlled needs d_A = d_B")
            Sigma, w_A, w_B = controlled_env_parts(
                d_A,
                cfg["lambda_xz"],
                rng,
                rotate=cfg.get("rotate", True),
                align=cfg.get("align", 1.0),
                normalize=cfg.get("normalize", False),
                alpha=cfg.get("alpha", 1.0),
                weak=cfg.get("weak", "B"),
            )
            return cls(w_A=w_A, w_B=w_B, Sigma=Sigma, sigma=cfg["sigma"])
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


def tilted_direction(v: np.ndarray, c: float, rng: np.random.Generator) -> np.ndarray:
    """A vector with the norm of v and cosine c with it: c v + sqrt(1 - c^2) |v| u, where u is a
    random unit vector orthogonal to v."""
    if not -1.0 <= c <= 1.0:
        raise ValueError(f"cosine must be in [-1, 1], got {c}")
    u = rng.standard_normal(len(v))
    nv = float(np.linalg.norm(v))
    if nv > 0:
        u -= (u @ v) / nv**2 * v  # remove the component along v
    u /= np.linalg.norm(u)
    return c * v + np.sqrt(1.0 - c**2) * nv * u


@dataclass
class RelatedTargetsEnv(LinearTwinEnv):
    """Each twin predicts its own target (LogBook note 2026-11-16, "Alignment sweep: twins
    predicting related targets"):

        y_A = <x^A, w^A> + <x^B, w^B> + eps,      y_B = <x^A, v^A> + <x^B, v^B> + eps,

    with cos(w^A, v^A) = cos(w^B, v^B) = c, |v^A| = |w^A|, |v^B| = |w^B| and the noise eps
    shared, so c = 1 gives y_A = y_B (the base task). At Sigma = I twin A's best use of B's
    opening <x^B, v^B> is the coefficient c, which removes a fraction c^2 of what B's raw
    features would remove.

    `sample` returns (x^A, x^B, y_A), so code that knows only one target sees A's;
    `sample_pair` also returns y_B from the same draws.
    """

    v_A: np.ndarray = None  # B's target weights on A's features
    v_B: np.ndarray = None  # B's target weights on B's features
    target_align: float = 1.0

    def sample_pair(self, n: int, rng: np.random.Generator):
        """(x^A, x^B, y_A, y_B); the same random draws as `sample`."""
        x = rng.standard_normal((n, self.d_A + self.d_B)) @ self._L.T
        eps = self.sigma * rng.standard_normal(n)
        x_A, x_B = x[:, : self.d_A], x[:, self.d_A :]
        return x_A, x_B, x @ self.w + eps, x_A @ self.v_A + x_B @ self.v_B + eps

    def sample(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_A, x_B, y_A, _ = self.sample_pair(n, rng)
        return x_A, x_B, y_A

    @classmethod
    def from_config(cls, cfg: dict, rng: np.random.Generator) -> "RelatedTargetsEnv":
        """The base env from the same block (e.g. `cov: controlled`), then B's target weights
        tilted by cosine `target_align` (drawn from rng after the base env)."""
        lin = LinearTwinEnv.from_config(cfg, rng)
        c = float(cfg["target_align"])
        v_A, v_B = tilted_direction(lin.w_A, c, rng), tilted_direction(lin.w_B, c, rng)
        return cls(lin.w_A, lin.w_B, lin.Sigma, lin.sigma, v_A=v_A, v_B=v_B, target_align=c)


def build_env(cfg: dict, rng: np.random.Generator) -> LinearTwinEnv:
    """Environment from the `env` config block; `task: logistic` gives binary labels,
    `task: product` multiplies the two blocks' signals (`ProductTwinEnv`), and `task: related`
    gives each twin its own target (`RelatedTargetsEnv`)."""
    task = cfg.get("task", "regression")
    if task == "regression":
        return LinearTwinEnv.from_config(cfg, rng)
    if task == "related":
        return RelatedTargetsEnv.from_config(cfg, rng)
    if task == "logistic":
        return LogisticTwinEnv.from_config(cfg, rng)
    if task == "product":
        return ProductTwinEnv.from_config(cfg, rng)
    raise ValueError(
        f"unknown task {task!r}; choose 'regression', 'logistic', 'product' or 'related'")
