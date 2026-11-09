"""Population (n -> infinity) limits for the logistic twin model, by Monte Carlo.

There is no closed form as in `consensus.theory`: each protocol's limiting predictor is the
best logistic fit (in log loss) of y on what the agent observes, computed on a large sample
with the true probabilities p*(x) as soft targets. Scores are expectations over y given x:

    Brier(p) = E[(p - p*)^2 + p*(1 - p*)],   log loss(p) = E[-p* log p - (1 - p*) log(1 - p)].

Keys mirror `theory.protocol_benchmarks`: full_info (Bayes), solo_A, solo_B, no_talk (average of
probabilities), logit_sum (sigmoid of the summed log-odds), blend (best logistic blend of the two
log-odds), exchange_<e>, each as Brier score, plus `logloss_<key>` and `blend_alpha/beta`.
"""

from __future__ import annotations

import numpy as np

from consensus.data import LogisticTwinEnv
from consensus.models import P_CLIP, fit_logistic, logit, sigmoid


def _scores(p: np.ndarray, ps: np.ndarray) -> tuple[float, float]:
    q = np.clip(p, P_CLIP, 1 - P_CLIP)
    brier = np.mean((p - ps) ** 2 + ps * (1 - ps))
    ll = np.mean(-ps * np.log(q) - (1 - ps) * np.log(1 - q))
    return float(brier), float(ll)


def _fit_prob(Z: np.ndarray, ps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Best logistic fit with soft targets (tiny ridge); returns (probabilities, coefficients)."""
    theta, _ = fit_logistic(Z, ps, P0=np.eye(Z.shape[1]) * 1e6)
    return sigmoid(Z @ theta), theta


def logistic_benchmarks(
    env: LogisticTwinEnv, n_exchanges: int = 4, n_mc: int = 200_000, seed: int = 12345
) -> dict[str, float]:
    """Monte Carlo population Brier score and log loss of every protocol (see module doc)."""
    rng = np.random.default_rng(seed)
    x_A, x_B, _ = env.sample(n_mc, rng)
    ps = env.true_prob(x_A, x_B)
    pA, _ = _fit_prob(x_A, ps)
    pB, _ = _fit_prob(x_B, ps)
    preds = {
        "full_info": ps,
        "solo_A": pA,
        "solo_B": pB,
        "no_talk": (pA + pB) / 2,
        "logit_sum": sigmoid(logit(pA) + logit(pB)),
    }
    preds["blend"], ab = _fit_prob(np.column_stack([logit(pA), logit(pB)]), ps)
    mA, mB = [pA], [pB]
    for e in range(1, n_exchanges + 1):
        newA, _ = _fit_prob(np.column_stack([x_A] + [logit(m) for m in mB[:e]]), ps)
        newB, _ = _fit_prob(np.column_stack([x_B] + [logit(m) for m in mA[:e]]), ps)
        mA.append(newA)
        mB.append(newB)
        preds[f"exchange_{e}"] = (newA + newB) / 2
    out: dict[str, float] = {"blend_alpha": float(ab[0]), "blend_beta": float(ab[1])}
    for k, p in preds.items():
        out[k], out[f"logloss_{k}"] = _scores(p, ps)
    return out
