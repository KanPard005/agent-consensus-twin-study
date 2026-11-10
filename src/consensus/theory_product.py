"""Population (n -> infinity) limits for the product twin model (`data.ProductTwinEnv`), by Monte Carlo.

Each twin's solo learner is a linear regression of y on (1, own x); its population limit is the
least-squares fit on a large sample (at rho = 0 this is mu * g_A, the conditional mean). Keys
mirror `theory.protocol_benchmarks` where they exist; all values are mean squared errors:

- full_info: sigma^2 (the Bayes predictor g_A g_B)
- solo_A, solo_B: the solo fits
- no_talk: average of the two solo fits (protocol P0)
- product: product of the two solo fits (`NoTalk(pool="product")`); exact at rho = 0, mu = 1
- best_additive: best affine combination c + a p_A + b p_B of the two solo fits (any rule that
  only adds or averages opinions does at best this)
"""

from __future__ import annotations

import numpy as np

from consensus.data import ProductTwinEnv


def _ls_fit(Z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least-squares fitted values of y on the columns of Z, shape (n,)."""
    coef, *_ = np.linalg.lstsq(Z, y, rcond=None)
    return Z @ coef


def product_benchmarks(env: ProductTwinEnv, n_mc: int = 400_000, seed: int = 12345) -> dict:
    """Monte Carlo population MSE of each combination rule (see module doc)."""
    rng = np.random.default_rng(seed)
    x_A, x_B, _ = env.sample(n_mc, rng)
    m = env.true_mean(x_A, x_B)  # score against E[y | x] and add sigma^2: less MC noise
    one = np.ones((n_mc, 1))
    p_A = _ls_fit(np.hstack([one, x_A]), m)
    p_B = _ls_fit(np.hstack([one, x_B]), m)
    preds = {
        "solo_A": p_A,
        "solo_B": p_B,
        "no_talk": (p_A + p_B) / 2,
        "product": p_A * p_B,
        "best_additive": _ls_fit(np.column_stack([one[:, 0], p_A, p_B]), m),
    }
    s2 = env.sigma**2
    out = {"full_info": float(s2)}
    out.update({k: float(np.mean((p - m) ** 2) + s2) for k, p in preds.items()})
    return out
