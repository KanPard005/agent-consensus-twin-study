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

Clause noise (`noise_clause = k`, Kanad 2026-11-18): instead of flipping y, flip only clause k's
value, y = D_k' AND (AND_{j != k} D_j) with D_k' = D_k xor Bernoulli(eps). Then y = pi exactly
whenever another clause is 0, and all label noise sits in cases that clause k decides. eps is
calibrated so that P(Y = pi) = rho overall: a flip of D_k changes y only when every other clause
is 1, so eps = (1 - rho) / P(all D_j = 1, j != k). The same uniform draw decides the flip as in
the uniform model, so the features of every case are unchanged.

XOR label (`label = "xor"`, Kanad 2026-11-19): pi(X) = D_1 xor D_2 xor ... (parity of the clause
ORs) instead of their AND. With two clauses, p_clause = 0.5 and r_cross = 0, each twin's own
clause says nothing about y (P(y = 1 | own features) = 0.5 exactly), so pooling probabilities
gains nothing; only exchanging the clause values does.
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
    noise_clause : int or None
        None: flip y with probability 1 - rho (uniform label noise). k: flip only clause k's value,
        with eps calibrated so that P(Y = pi) = rho (see the module docstring).
    p_pi : float or None
        If set, p_clause is ignored and the common P(D_k = 1) is solved for so that
        P(pi = 1) = p_pi. With r_cross = 0 the clauses are independent and p_clause =
        p_pi^(1/K) does this; with r_cross > 0 they are positively dependent and that choice
        gives P(pi = 1) > p_pi, so the base rate would drift with the correlation.
    label : {"and", "xor"}
        How the clause ORs combine into pi: their AND (the CNF) or their parity.
    """

    clauses: list[list[int]]
    views: dict[str, list[int]]
    r_within: float = 0.5
    r_cross: float = 0.0
    p_clause: float = float(np.sqrt(0.5))
    rho: float = 0.9
    n_mc: int = 1_000_000
    mc_seed: int = 12345
    noise_clause: int | None = None
    p_pi: float | None = None
    label: str = "and"
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
        if self.label not in ("and", "xor"):
            raise ValueError(f"label must be 'and' or 'xor', got {self.label!r}")
        if self.label == "xor" and (self.noise_clause is not None or self.p_pi is not None):
            # Both calibrations assume the AND: a flip of one clause changes y only when the
            # others are 1, and P(pi = 1) increases with p_clause. Neither holds for parity.
            raise ValueError("label 'xor' supports neither noise_clause nor p_pi")
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
        if self.p_pi is not None:
            self.p_clause = self._solve_p_clause(Z, float(self.p_pi))
        self.t = self._thresholds(Z, self.p_clause)
        X = (Z > self.t).astype(np.int8)
        self.eps = None
        if self.noise_clause is not None:
            k = int(self.noise_clause)
            others = np.delete(self.clause_values(X), k, axis=1).all(axis=1)
            self.eps = (1.0 - self.rho) / float(others.mean())
            if not 0.0 <= self.eps < 0.5:
                raise ValueError(f"clause noise needs eps < 0.5, got {self.eps:.3f}")
        self._bayes_tables(X)

    def _thresholds(self, Z: np.ndarray, p_clause: float) -> np.ndarray:
        """Per-feature thresholds with P(D_k = 1) = p_clause for every clause: D_k = 1 iff the
        clause's largest Z exceeds t_k, so t_k is the (1 - p_clause) quantile of that max."""
        t = np.empty(self.n)
        for c in self.clauses:
            t[c] = np.quantile(Z[:, c].max(axis=1), 1.0 - p_clause)
        return t

    def _solve_p_clause(self, Z: np.ndarray, target: float, iters: int = 40) -> float:
        """Bisection on the common P(D_k = 1) so that P(pi = 1) = target on the Monte Carlo
        sample; P(pi = 1) increases with p_clause, so the root is unique."""
        lo, hi = target, 1.0  # P(pi = 1) <= P(D_k = 1), so p_clause >= target
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            p = self.pi((Z > self._thresholds(Z, mid)).astype(np.int8)).mean()
            lo, hi = (mid, hi) if p < target else (lo, mid)
        return 0.5 * (lo + hi)

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
            noise_clause=cfg.get("noise_clause"),
            p_pi=cfg.get("p_pi"),
            label=str(cfg.get("label", "and")),
        )

    # ---- sampling --------------------------------------------------------------------------
    def _gauss(self, m: int, rng: np.random.Generator) -> np.ndarray:
        """Z ~ N(0, Sigma), shape (m, n)."""
        return rng.standard_normal((m, self.n)) @ self._L.T

    def pi(self, X: np.ndarray) -> np.ndarray:
        """The noiseless label, shape (m,), for X of shape (m, n): AND_k OR_{i in k} X_i, or the
        parity of the clause ORs for label = "xor"."""
        D = self.clause_values(X)
        if self.label == "xor":
            return (D.sum(axis=1) % 2).astype(np.int8)
        return D.all(axis=1).astype(np.int8)

    def clause_values(self, X: np.ndarray) -> np.ndarray:
        """D_k for every clause, shape (m, n_clauses)."""
        return np.stack([X[:, c].max(axis=1) for c in self.clauses], axis=1).astype(np.int8)

    def sample(self, m: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Draw m cases.

        Returns
        -------
        X : ndarray of int8, shape (m, n)
        pi : ndarray of int8, shape (m,)   the noiseless label
        y : ndarray of int8, shape (m,)    the noisy label (P(y = pi) = rho)
        """
        X = (self._gauss(m, rng) > self.t).astype(np.int8)
        p = self.pi(X)
        u = rng.random(m)
        if self.noise_clause is None:
            return X, p, np.where(u > self.rho, 1 - p, p).astype(np.int8)
        D = self.clause_values(X)
        D[:, self.noise_clause] ^= (u < self.eps).astype(np.int8)
        return X, p, D.all(axis=1).astype(np.int8)

    def view(self, X: np.ndarray, k: str) -> np.ndarray:
        """The columns twin k sees, shape (m, |S_k|)."""
        return X[:, self.views[k]]

    # ---- Bayes references --------------------------------------------------------------------
    def _bayes_tables(self, X: np.ndarray) -> None:
        """P(Y = 1 | x_S) for each twin's view S: the Monte Carlo mean of the exact P(Y = 1 | x)
        (`p_joint`) over the cases with pattern x_S.

        Stored as {pattern tuple: probability}; a pattern with no Monte Carlo mass gets the
        marginal. With uniform noise this equals rho q + (1 - rho)(1 - q), q = P(pi = 1 | x_S).
        """
        self.p_pi = float(self.pi(X).mean())
        p = self.p_joint(X)
        self.p_d = self.clause_values(X).mean(axis=0)
        self.py_given_view = {}
        for k, S in self.views.items():
            codes = X[:, S] @ (1 << np.arange(len(S)))  # pattern -> integer code
            tot = np.bincount(codes, minlength=1 << len(S))
            pos = np.bincount(codes, weights=p, minlength=1 << len(S))
            # code c has bit j = value of the j-th feature in S
            self.py_given_view[k] = {
                tuple((c >> j) & 1 for j in range(len(S))): (pos[c] / tot[c] if tot[c]
                                                             else float(p.mean()))
                for c in range(1 << len(S))
            }

    def p_solo(self, x_view: np.ndarray, k: str) -> np.ndarray:
        """Bayes P(Y = 1 | x_{S_k}), shape (m,), for x_view of shape (m, |S_k|)."""
        return np.array([self.py_given_view[k][tuple(int(v) for v in row)] for row in x_view])

    def p_joint(self, X: np.ndarray) -> np.ndarray:
        """Bayes P(Y = 1 | x) for full feature vectors, shape (m,)."""
        if self.noise_clause is None:
            return np.where(self.pi(X) == 1, self.rho, 1 - self.rho)
        D = self.clause_values(X)
        k = self.noise_clause
        others = np.delete(D, k, axis=1).all(axis=1)
        return np.where(others, np.where(D[:, k] == 1, 1 - self.eps, self.eps), 0.0)

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
