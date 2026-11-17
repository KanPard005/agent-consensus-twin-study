"""Tests for the population closed forms in consensus.theory."""

import numpy as np
import pytest

from consensus.data import LinearTwinEnv, block_covariance
from consensus.theory import (
    agreement_path,
    asymmetric_limits,
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


@pytest.mark.parametrize("align", [1.0, -1.0])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("lam", [0.0, 1 / 3, 0.6, -0.4])
def test_controlled_disagreement_formula(lam, normalize, align):
    """Population solo opinions disagree by 2(1 - lam^2)(1 + c lam), divided by the signal
    variance 2(1 + c lam) when normalized (note 2026-11-11, more controlled experiment)."""
    cfg = dict(d_A=4, d_B=4, cov="controlled", lambda_xz=lam, align=align,
               normalize=normalize, sigma=SIGMA_NOISE)
    env = LinearTwinEnv.from_config(cfg, np.random.default_rng(3))
    S = env.Sigma
    a, b = solo_predictors(S, env.w, 4)
    disagreement = (a - b) @ S @ (a - b)
    raw = 2 * (1 - lam**2) * (1 + align * lam)
    expected = raw / (2 * (1 + align * lam)) if normalize else raw
    assert disagreement == pytest.approx(expected, abs=1e-10)
    assert env.w @ S @ env.w == pytest.approx(1.0 if normalize else 2 * (1 + align * lam))


def test_controlled_without_rotation_is_paired():
    cfg = dict(d_A=3, d_B=3, cov="controlled", lambda_xz=0.5, rotate=False, sigma=SIGMA_NOISE)
    env = LinearTwinEnv.from_config(cfg, np.random.default_rng(0))
    np.testing.assert_allclose(env.Sigma, block_covariance(3, 3, 0.5))
    np.testing.assert_allclose(env.w_A, np.ones(3) / np.sqrt(3))


@pytest.mark.parametrize("weak", ["A", "B"])
@pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 1.0])
@pytest.mark.parametrize("lam", [0.0, 0.5, 0.8])
def test_asymmetric_limits_match_numerics(lam, alpha, weak):
    """Closed forms of `asymmetric_limits` against the generic population formulas."""
    cfg = dict(d_A=3, d_B=3, cov="controlled", lambda_xz=lam, alpha=alpha, weak=weak,
               sigma=SIGMA_NOISE)
    env = LinearTwinEnv.from_config(cfg, np.random.default_rng(1))
    S, w = env.Sigma, env.w
    lim = asymmetric_limits(lam, alpha, SIGMA_NOISE)
    strong = "B" if weak == "A" else "A"
    assert w @ S @ w == pytest.approx(lim["signal"])
    # schur_excess(agent) is the excess of that agent's solo predictor.
    assert schur_excess(S, w, 3, strong) == pytest.approx(lim["excess_strong"], abs=1e-10)
    assert schur_excess(S, w, 3, weak) == pytest.approx(lim["excess_weak"], abs=1e-10)
    a, b = solo_predictors(S, w, 3)
    assert (a - b) @ S @ (a - b) == pytest.approx(lim["disagreement"], abs=1e-10)
    if alpha + lam > 0:
        ps, pw = (a, b) if strong == "A" else (b, a)
        coef = blend_weights(S, w, ps, pw)
        mix = coef[0] * ps + coef[1] * pw  # the blend recovers the full predictor
        assert predictor_mse(mix, S, w, SIGMA_NOISE) == pytest.approx(SIGMA_NOISE**2)
        if alpha > 0:
            np.testing.assert_allclose(coef, [lim["beta_strong"], lim["beta_weak"]], atol=1e-8)
        # Best convex weight on the partner, from the population moments.
        for own, partner, key in ((ps, pw, "convex_strong"), (pw, ps, "convex_weak")):
            dlt = partner - own
            w_conv = ((w - own) @ S @ dlt) / (dlt @ S @ dlt)
            assert w_conv == pytest.approx(lim[key], abs=1e-10)
