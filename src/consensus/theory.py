"""Population (n -> infinity) quantities for the linear twin model.

Every linear predictor is represented by a vector v in R^{d_A + d_B} acting on the full
feature vector x = (x^A, x^B): yhat = <v, x>. A predictor that only uses features z = M x
(rows of M = what an agent observes: its own coordinates and linear messages) is optimal at
the L2 projection of y onto span(M x):

    v = M^T (M Sigma M^T)^+ M Sigma w,     MSE(v) = (v - w)^T Sigma (v - w) + sigma^2.

Follows LogBook note `agent-consensus/notes/2026-11-05-initial-idea`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def selector(d_A: int, d_B: int, agent: str) -> np.ndarray:
    """Rows picking agent A's (or B's) coordinates out of x. Shape (d_i, d_A + d_B)."""
    E = np.eye(d_A + d_B)
    return E[:d_A] if agent == "A" else E[d_A:]


def predictor_mse(v: np.ndarray, Sigma: np.ndarray, w: np.ndarray, sigma: float) -> float:
    """Population MSE of the linear predictor yhat = <v, x>: (v-w)^T Sigma (v-w) + sigma^2."""
    r = v - w
    return float(r @ Sigma @ r + sigma**2)


def project(M: np.ndarray, Sigma: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Best linear predictor of y from z = M x, written in x-space.

    Parameters
    ----------
    M : ndarray, shape (k, d)
        Feature map; rows may be linearly dependent (pseudo-inverse handles that, e.g. when a
        message is already in the span of the agent's own features).
    Sigma : ndarray, shape (d, d)
    w : ndarray, shape (d,)

    Returns
    -------
    v : ndarray, shape (d,)
        v = M^T (M Sigma M^T)^+ M Sigma w.
    """
    G = M @ Sigma @ M.T
    c = M @ Sigma @ w
    return M.T @ (np.linalg.pinv(G, hermitian=True) @ c)


def solo_predictors(Sigma: np.ndarray, w: np.ndarray, d_A: int) -> tuple[np.ndarray, np.ndarray]:
    """Population solo predictors (a, b) in x-space, i.e. what each twin learns in training.

    A's block of `a` equals w^A + Sigma_AA^{-1} Sigma_AB w^B (omitted-variable bias), and its
    excess MSE is w^B^T S_{B|A} w^B with S_{B|A} = Sigma_BB - Sigma_BA Sigma_AA^{-1} Sigma_AB.
    """
    d_B = len(w) - d_A
    a = project(selector(d_A, d_B, "A"), Sigma, w)
    b = project(selector(d_A, d_B, "B"), Sigma, w)
    return a, b


def schur_excess(Sigma: np.ndarray, w: np.ndarray, d_A: int, agent: str = "A") -> float:
    """Closed-form excess MSE of an agent's solo predictor via the Schur complement.

    For agent A: w_B^T (Sigma_BB - Sigma_BA Sigma_AA^{-1} Sigma_AB) w_B (and symmetrically for B).
    """
    i = slice(0, d_A) if agent == "A" else slice(d_A, None)
    j = slice(d_A, None) if agent == "A" else slice(0, d_A)
    S = Sigma[j, j] - Sigma[j, i] @ np.linalg.solve(Sigma[i, i], Sigma[i, j])
    return float(w[j] @ S @ w[j])


def blend_weights(Sigma: np.ndarray, w: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Optimal static blend (alpha*, beta*) of two fixed predictors: 2x2 normal equations.

    Solves [a; b] Sigma [a; b]^T (alpha, beta)^T = [a; b] Sigma w.
    """
    M = np.vstack([a, b])
    return np.linalg.pinv(M @ Sigma @ M.T, hermitian=True) @ (M @ Sigma @ w)


@dataclass
class AgreementPath:
    """Population quantities along the exchange protocol, indexed by round e = 0..E.

    Round 0 is the solo prediction (first message); round e >= 1 conditions on the partner's
    first e messages. Arrays have shape (E + 1,).
    """

    mse_A: np.ndarray
    mse_B: np.ndarray
    mse_consensus: np.ndarray  # MSE of the average (m^A_e + m^B_e) / 2
    disagreement: np.ndarray  # E[(m^A_e - m^B_e)^2]
    shift_A: (
        np.ndarray
    )  # opinion shift E[(m^A_e - m^A_0)^2] = (v_A^e - v_A^0)^T Sigma (v_A^e - v_A^0)
    shift_B: np.ndarray
    v_A: list[np.ndarray]  # x-space predictors per round
    v_B: list[np.ndarray]


def agreement_path(
    Sigma: np.ndarray, w: np.ndarray, d_A: int, sigma: float, n_exchanges: int
) -> AgreementPath:
    """Population version of the exchange protocol (P2 = 1 exchange, P3 = several).

    Messages are simultaneous: in round e each agent announces the projection of y onto its own
    features and the partner's messages from rounds 0..e-1. With Gaussian x this is exactly the
    Geanakoplos-Polemarchakis protocol (announced conditional expectations under a common prior),
    since all messages are linear in x.
    """
    d_B = len(w) - d_A
    E_A, E_B = selector(d_A, d_B, "A"), selector(d_A, d_B, "B")
    a, b = solo_predictors(Sigma, w, d_A)
    v_A, v_B = [a], [b]
    for _ in range(n_exchanges):
        a_new = project(np.vstack([E_A, *v_B]), Sigma, w)
        b_new = project(np.vstack([E_B, *v_A]), Sigma, w)
        v_A.append(a_new)
        v_B.append(b_new)
    mse_A = np.array([predictor_mse(v, Sigma, w, sigma) for v in v_A])
    mse_B = np.array([predictor_mse(v, Sigma, w, sigma) for v in v_B])
    cons = np.array([predictor_mse((p + q) / 2, Sigma, w, sigma) for p, q in zip(v_A, v_B)])

    def sq_dist(p, q):
        return float((p - q) @ Sigma @ (p - q))

    dis = np.array([sq_dist(p, q) for p, q in zip(v_A, v_B)])
    shift_A = np.array([sq_dist(p, a) for p in v_A])
    shift_B = np.array([sq_dist(q, b) for q in v_B])
    return AgreementPath(mse_A, mse_B, cons, dis, shift_A, shift_B, v_A, v_B)


def protocol_benchmarks(
    Sigma: np.ndarray, w: np.ndarray, d_A: int, sigma: float, n_exchanges: int = 4
) -> dict[str, float]:
    """Population MSE of every protocol, for horizontal reference lines in plots.

    Keys: full_info, solo_A, solo_B, no_talk (P0), blend (P1), exchange_<e> for e = 1..n_exchanges,
    and the opinion shifts shift_A_exchange_<e>, shift_B_exchange_<e> (zero for P0 and P1).
    """
    a, b = solo_predictors(Sigma, w, d_A)
    alpha, beta = blend_weights(Sigma, w, a, b)
    path = agreement_path(Sigma, w, d_A, sigma, n_exchanges)
    out = {
        "full_info": sigma**2,
        "solo_A": predictor_mse(a, Sigma, w, sigma),
        "solo_B": predictor_mse(b, Sigma, w, sigma),
        "no_talk": predictor_mse((a + b) / 2, Sigma, w, sigma),
        "blend": predictor_mse(alpha * a + beta * b, Sigma, w, sigma),
        "blend_alpha": float(alpha),
        "blend_beta": float(beta),
    }
    for e in range(1, n_exchanges + 1):
        out[f"exchange_{e}"] = float(path.mse_consensus[e])
        out[f"shift_A_exchange_{e}"] = float(path.shift_A[e])
        out[f"shift_B_exchange_{e}"] = float(path.shift_B[e])
    return out
