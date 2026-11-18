"""Boolean task: correlated binary features from a Gaussian copula, a noisy CNF label.

LogBook note 2026-11-18, "Brainstorming: boolean experiments". Features X_1..X_n are thresholded
correlated Gaussians,

    Z ~ N(0, Sigma),   X_i = 1[Z_i > t_{c(i)}],

where c(i) is the OR-clause feature i belongs to, Sigma_ii = 1, Sigma_ij = r_within when i and j
are in the same clause and r_cross otherwise. The label is

    pi(X) = AND_k OR_{i in clause k} X_i,     P(Y = pi(X)) = rho.

The threshold of clause k is calibrated so that P(D_k = 1) = p_clause, where D_k is the clause's
OR: D_k = 1 iff max_{i in k} Z_i > t_k, so t_k is the (1 - p_clause) quantile of that maximum.
Holding p_clause fixed while r_within / r_cross vary keeps the base rate of Y fixed (note
variation 4). Calibration and the Bayes references use one large Monte Carlo sample from a fixed
seed, so they are the same for every run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BooleanCopulaEnv:
    """Gaussian-copula features with a noisy AND-of-ORs label (see the module docstring).

    Parameters
    ----------
    clauses : list of lists of int
        Feature indices (0-based) of each OR clause; together they cover 0..n-1 exactly once.
    views : {"A": [...], "B": [...]}
        Feature indices each twin sees (their union must be every feature).
    r_within, r_cross : float
        Gaussian correlation between features of the same clause / of different clauses.
    p_clause : float
        Target P(D_k = 1) for every clause k.
    rho : float
        P(Y = pi(X)), in (0.5, 1].
    n_mc : int
        Monte Carlo sample size for the thresholds and the Bayes references.
    mc_seed : int
        Seed of that sample (fixed, so every run shares the same thresholds).
    """

    clauses: list[list[int]]
    views: dict[str, list[int]]
    r_within: float = 0.5
    r_cross: float = 0.0
    p_clause: float = float(np.sqrt(0.5))
    rho: float = 0.9
    n_mc: int = 1_000_000
    mc_seed: int = 12345
    Sigma: np.ndarray = field(init=False, repr=False)
    t: np.ndarray = field(init=False)  # threshold per feature, shape (n,)

    def __post_init__(self) -> None:
        idx = sorted(i for c in self.clauses for i in c)
        self.n = len(idx)
        if idx != list(range(self.n)):
            raise ValueError("clauses must cover features 0..n-1 exactly once")
        if sorted(set(self.views["A"]) | set(self.views["B"])) != idx:
            raise ValueError("the twins' views must together cover every feature")
        if not 0.5 < self.rho <= 1.0:
            raise ValueError("rho must be in (0.5, 1]")
        self.clause_of = np.empty(self.n, int)
        for k, c in enumerate(self.clauses):
            self.clause_of[c] = k
        same = self.clause_of[:, None] == self.clause_of[None, :]
        self.Sigma = np.where(same, self.r_within, self.r_cross)
        np.fill_diagonal(self.Sigma, 1.0)
        if np.linalg.eigvalsh(self.Sigma).min() <= 1e-9:
            raise ValueError("Sigma is not positive definite for these r_within, r_cross")
        self._L = np.linalg.cholesky(self.Sigma)
        Z = self._gauss(self.n_mc, np.random.default_rng(self.mc_seed))
        self.t = np.empty(self.n)
        for c in self.clauses:  # D_k = 1 iff max over the clause exceeds t_k
            self.t[c] = np.quantile(Z[:, c].max(axis=1), 1.0 - self.p_clause)
        X = (Z > self.t).astype(np.int8)
        self._bayes_tables(X)

    @classmethod
    def from_config(cls, cfg: dict) -> "BooleanCopulaEnv":
        return cls(
            clauses=cfg["clauses"],
            views=cfg["views"],
            r_within=float(cfg.get("r_within", 0.5)),
            r_cross=float(cfg.get("r_cross", 0.0)),
            p_clause=float(cfg.get("p_clause", np.sqrt(0.5))),
            rho=float(cfg.get("rho", 0.9)),
            n_mc=int(cfg.get("n_mc", 1_000_000)),
            mc_seed=int(cfg.get("mc_seed", 12345)),
        )

    # ---- sampling --------------------------------------------------------------------------
    def _gauss(self, m: int, rng: np.random.Generator) -> np.ndarray:
        """Z ~ N(0, Sigma), shape (m, n)."""
        return rng.standard_normal((m, self.n)) @ self._L.T

    def pi(self, X: np.ndarray) -> np.ndarray:
        """The noiseless label AND_k OR_{i in k} X_i, shape (m,), for X of shape (m, n)."""
        return np.all([X[:, c].max(axis=1) for c in self.clauses], axis=0).astype(np.int8)

    def clause_values(self, X: np.ndarray) -> np.ndarray:
        """D_k for every clause, shape (m, n_clauses)."""
        return np.stack([X[:, c].max(axis=1) for c in self.clauses], axis=1).astype(np.int8)

    def sample(self, m: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Draw m cases.

        Returns
        -------
        X : ndarray of int8, shape (m, n)
        pi : ndarray of int8, shape (m,)   the noiseless label
        y : ndarray of int8, shape (m,)    pi flipped with probability 1 - rho
        """
        X = (self._gauss(m, rng) > self.t).astype(np.int8)
        p = self.pi(X)
        flip = rng.random(m) > self.rho
        return X, p, np.where(flip, 1 - p, p).astype(np.int8)

    def view(self, X: np.ndarray, k: str) -> np.ndarray:
        """The columns twin k sees, shape (m, |S_k|)."""
        return X[:, self.views[k]]

    # ---- Bayes references --------------------------------------------------------------------
    def _bayes_tables(self, X: np.ndarray) -> None:
        """P(pi = 1 | x_S) for each twin's view S, from the Monte Carlo sample X.

        Stored as {pattern tuple: probability}; a pattern with no Monte Carlo mass gets the
        marginal. The joint reference needs no table: P(Y = 1 | x) = rho if pi(x) else 1 - rho.
        """
        p = self.pi(X)
        self.p_pi = float(p.mean())
        self.p_d = self.clause_values(X).mean(axis=0)
        self.pi_given_view = {}
        for k, S in self.views.items():
            codes = X[:, S] @ (1 << np.arange(len(S)))  # pattern -> integer code
            tot = np.bincount(codes, minlength=1 << len(S))
            pos = np.bincount(codes, weights=p, minlength=1 << len(S))
            # code c has bit j = value of the j-th feature in S
            self.pi_given_view[k] = {
                tuple((c >> j) & 1 for j in range(len(S))): (pos[c] / tot[c] if tot[c]
                                                             else self.p_pi)
                for c in range(1 << len(S))
            }

    def p_solo(self, x_view: np.ndarray, k: str) -> np.ndarray:
        """Bayes P(Y = 1 | x_{S_k}), shape (m,), for x_view of shape (m, |S_k|)."""
        q = np.array([self.pi_given_view[k][tuple(int(v) for v in row)] for row in x_view])
        return self.rho * q + (1 - self.rho) * (1 - q)

    def p_joint(self, X: np.ndarray) -> np.ndarray:
        """Bayes P(Y = 1 | x) for full feature vectors, shape (m,)."""
        return np.where(self.pi(X) == 1, self.rho, 1 - self.rho)

    def references(self, m: int = 400_000, seed: int = 7) -> dict:
        """Accuracy and log loss of the Bayes-optimal solo and joint predictors (Monte Carlo)."""
        X, _, y = self.sample(m, np.random.default_rng(seed))
        out = {"p_pi": self.p_pi, "p_clause": [float(v) for v in self.p_d],
               "phi": self.phi(X)}
        preds = {k: self.p_solo(self.view(X, k), k) for k in self.views}
        preds["joint"] = self.p_joint(X)
        for name, pr in preds.items():
            out[f"acc_{name}"] = float(np.mean((pr > 0.5) == (y == 1)))
            out[f"logloss_{name}"] = float(np.mean(log_loss(pr, y)))
        return out

    def phi(self, X: np.ndarray) -> dict:
        """Mean phi coefficient (Pearson correlation of the binary features) within a clause and
        across clauses: the binary-scale counterpart of r_within / r_cross."""
        C = np.corrcoef(X.T.astype(float))
        same = self.clause_of[:, None] == self.clause_of[None, :]
        off = ~np.eye(self.n, dtype=bool)
        return {"within": float(C[same & off].mean()) if (same & off).any() else None,
                "cross": float(C[~same].mean()) if (~same).any() else None}


def log_loss(p: np.ndarray, y: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """-log of the probability given to the true label, with p clipped to [eps, 1 - eps]
    (the same clipping the twins are told about)."""
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    y = np.asarray(y)
    return -np.where(y == 1, np.log(p), np.log(1 - p))
