"""Tests for the population closed forms in consensus.theory."""

import numpy as np
import pytest

from consensus.data import LinearTwinEnv, block_covariance
from consensus.theory import (
    agreement_path,
    blend_weights,
    predictor_mse,
    protocol_benchmarks,
    schur_excess,
    solo_predictors,
)

SIGMA_NOISE = 0.5


def make_env(d_A: int, d_B: int, rho: float, seed: int = 0) -> LinearTwinEnv:
    cfg = dict(d_A=d_A, d_B=d_B, rho=rho, norm_A=1.0, norm_B=1.5, sigma=SIGMA_NOISE)
    return LinearTwinEnv.from_config(cfg, np.random.default_rng(seed))


@pytest.mark.parametrize("d_A,d_B,rho", [(1, 1, 0.5), (3, 3, 0.5), (4, 2, -0.7)])
def test_solo_predictor_is_omitted_variable_formula(d_A, d_B, rho):
    env = make_env(d_A, d_B, rho)
    S = env.Sigma
    a, _ = solo_predictors(S, env.w, d_A)
    expected = env.w_A + np.linalg.solve(S[:d_A, :d_A], S[:d_A, d_A:] @ env.w_B)
    np.testing.assert_allclose(a[:d_A], expected, atol=1e-10)
    np.testing.assert_allclose(a[d_A:], 0.0, atol=1e-12)  # A cannot use B's coordinates


@pytest.mark.parametrize("agent", ["A", "B"])
def test_solo_excess_equals_schur_complement(agent):
    env = make_env(3, 2, 0.6)
    a, b = solo_predictors(env.Sigma, env.w, 3)
    v = a if agent == "A" else b
    excess = predictor_mse(v, env.Sigma, env.w, SIGMA_NOISE) - SIGMA_NOISE**2
    assert excess == pytest.approx(schur_excess(env.Sigma, env.w, 3, agent))


def test_blend_is_optimal_in_scalar_case():
    # d_A = d_B = 1: span(yhat^A, yhat^B) = span(x^A, x^B), so the blend reaches full information.
    env = make_env(1, 1, 0.5)
    assert protocol_benchmarks(env.Sigma, env.w, 1, SIGMA_NOISE)["blend"] == pytest.approx(
        SIGMA_NOISE**2
    )


def test_blend_is_optimal_with_independent_features():
    # rho = 0: no omitted-variable bias, so yhat^A + yhat^B = <w, x> and alpha = beta = 1.
    env = make_env(3, 3, 0.0)
    a, b = solo_predictors(env.Sigma, env.w, 3)
    np.testing.assert_allclose(blend_weights(env.Sigma, env.w, a, b), [1.0, 1.0], atol=1e-10)


def test_blend_has_gap_with_correlated_vector_features():
    env = make_env(3, 3, 0.5)
    bench = protocol_benchmarks(env.Sigma, env.w, 3, SIGMA_NOISE)
    assert bench["blend"] > SIGMA_NOISE**2 + 1e-3
    assert bench["blend"] <= bench["no_talk"] + 1e-12


def test_agreement_path_monotone_and_agrees():
    env = make_env(4, 4, 0.6, seed=3)
    path = agreement_path(env.Sigma, env.w, 4, SIGMA_NOISE, n_exchanges=8)
    # Conditioning on more messages can only help each agent.
    assert np.all(np.diff(path.mse_A) <= 1e-10)
    assert np.all(np.diff(path.mse_B) <= 1e-10)
    # Geanakoplos-Polemarchakis: announcements eventually agree.
    assert path.disagreement[-1] == pytest.approx(0.0, abs=1e-8)
    assert path.mse_consensus[-1] >= SIGMA_NOISE**2 - 1e-10


def test_block_covariance_is_positive_definite():
    S = block_covariance(3, 5, 0.9)
    assert np.all(np.linalg.eigvalsh(S) > 0)
    with pytest.raises(ValueError):
        block_covariance(2, 2, 1.0)


def test_population_opinion_shift():
    env = make_env(3, 3, 0.5)
    path = agreement_path(env.Sigma, env.w, 3, SIGMA_NOISE, n_exchanges=3)
    assert path.shift_A[0] == 0.0 and path.shift_B[0] == 0.0
    # Shift is the Sigma-norm distance between the round-e and solo predictors.
    r = path.v_A[2] - path.v_A[0]
    assert path.shift_A[2] == pytest.approx(float(r @ env.Sigma @ r))
    # At full information both twins sit at w, so the shift is ||w - solo||_Sigma^2 = solo excess.
    a, b = solo_predictors(env.Sigma, env.w, 3)
    assert path.shift_A[-1] == pytest.approx(
        predictor_mse(a, env.Sigma, env.w, SIGMA_NOISE) - SIGMA_NOISE**2, abs=1e-8
    )
