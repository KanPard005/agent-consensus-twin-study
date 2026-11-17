"""Rigged first partner queries (LogBook note 2026-11-17): rows 0..m-1 are selected cases,
later rows are untouched."""

import numpy as np
import pytest

from consensus.data import build_env, rig_first_queries

BASE = dict(d_A=3, d_B=3, cov="controlled", lambda_xz=0.0, rotate=True, align=1.0,
            normalize=False, alpha=1.0, weak="B", sigma=0.5, task="related", target_align=0.5)


@pytest.mark.parametrize("partner,noise", [("low", "low"), ("low", "high"), ("high", "low")])
def test_rig_first_queries(partner, noise):
    env = build_env(BASE, np.random.default_rng(0))
    q = env.sample_pair(30, np.random.default_rng(1))
    out = rig_first_queries(env, q, {"m": 10, "partner": partner, "noise": noise},
                            np.random.default_rng(2))
    for a, b in zip(q, out):
        np.testing.assert_array_equal(a[10:], b[10:])
    x_A, x_B, y_A, y_B = (o[:10] for o in out)
    s_A = np.abs(x_B @ env.w_B) / np.linalg.norm(env.w_B)
    s_B = np.abs(x_A @ env.v_A) / np.linalg.norm(env.v_A)
    e = np.abs(y_A - x_A @ env.w_A - x_B @ env.w_B) / env.sigma
    assert (np.maximum(s_A, s_B) < 0.3).all() if partner == "low" else (np.minimum(s_A, s_B) > 1.0).all()
    assert (e < 0.3).all() if noise == "low" else (e > 1.6).all()
    # B's target uses the same noise
    np.testing.assert_allclose(y_B - x_A @ env.v_A - x_B @ env.v_B, y_A - x_A @ env.w_A - x_B @ env.w_B)
