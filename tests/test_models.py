"""Tests for the online learners, agents and protocols."""

import numpy as np
import pytest

from consensus.data import LinearTwinEnv
from consensus.models import LinearAgent, OnlineRidge
from consensus.protocols import METRICS, Exchange, FrozenBlend, NoTalk
from consensus.theory import solo_predictors


def make_env(seed: int = 0) -> LinearTwinEnv:
    cfg = dict(d_A=3, d_B=2, rho=0.6, norm_A=1.0, norm_B=1.0, sigma=0.3)
    return LinearTwinEnv.from_config(cfg, np.random.default_rng(seed))


def train(env, n, rng, lam=1.0):
    A, B = LinearAgent(env.d_A, lam, "A"), LinearAgent(env.d_B, lam, "B")
    xA, _, yA = env.sample(n, rng)
    _, xB, yB = env.sample(n, rng)
    for t in range(n):
        A.update(xA[t], [], yA[t])
        B.update(xB[t], [], yB[t])
    return A, B


def test_online_ridge_matches_batch_ridge():
    rng = np.random.default_rng(1)
    Z, y, lam = rng.standard_normal((50, 4)), rng.standard_normal(50), 0.7
    rls = OnlineRidge(4, lam)
    for z, yy in zip(Z, y):
        rls.update(z, yy)
    batch = np.linalg.solve(Z.T @ Z + lam * np.eye(4), Z.T @ y)
    np.testing.assert_allclose(rls.theta, batch, atol=1e-10)
    np.testing.assert_allclose(rls.P, np.linalg.inv(Z.T @ Z + lam * np.eye(4)), atol=1e-10)


def test_trained_twin_converges_to_population_solo():
    env = make_env()
    A, _ = train(env, 20_000, np.random.default_rng(2))
    a, _ = solo_predictors(env.Sigma, env.w, env.d_A)
    np.testing.assert_allclose(A.heads[0].theta, a[: env.d_A], atol=0.03)


def test_exchange_heads_start_at_solo_opinion():
    env = make_env()
    A, _ = train(env, 200, np.random.default_rng(3))
    A.begin_communication(2)
    x = np.ones(env.d_A)
    assert A.predict(x, [1.3]) == pytest.approx(A.predict(x, []))
    assert A.predict(x, [1.3, -0.4]) == pytest.approx(A.predict(x, []))
    assert A.drift(2) == 0.0


def test_frozen_protocols_do_not_change_weights():
    env = make_env()
    A, B = train(env, 100, np.random.default_rng(4))
    xA, xB, y = env.sample(200, np.random.default_rng(5))
    for proto in (NoTalk(), FrozenBlend()):
        a, b = A.clone(), B.clone()
        out = proto.run(a, b, xA, xB, y)
        np.testing.assert_array_equal(a.heads[0].theta, A.heads[0].theta)
        assert set(out) == set(METRICS)


def test_runs_are_reproducible():
    def once():
        env = make_env(7)
        rng = np.random.default_rng(8)
        A, B = train(env, 100, rng)
        xA, xB, y = env.sample(300, rng)
        return Exchange(n_exchanges=3).run(A, B, xA, xB, y)["sq_err"]

    np.testing.assert_array_equal(once(), once())


def test_opinion_shift_is_zero_when_frozen_and_matches_formula():
    env = make_env()
    A, B = train(env, 100, np.random.default_rng(9))
    xA, xB, y = env.sample(200, np.random.default_rng(10))
    for proto in (NoTalk(), FrozenBlend()):
        out = proto.run(A.clone(), B.clone(), xA, xB, y)
        assert np.all(out["shift_A"] == 0) and np.all(out["shift_B"] == 0)
    # Train the exchange heads a little, then check one step against a by-hand computation.
    proto = Exchange(n_exchanges=2)
    a, b = A.clone(), B.clone()
    proto.run(a, b, xA[:-1], xB[:-1], y[:-1])
    a2, b2 = a.clone(), b.clone()
    rec = proto.step(a, b, xA[-1], xB[-1], float(y[-1]))
    mA0, mB0 = a2.predict(xA[-1], []), b2.predict(xB[-1], [])
    mB1 = b2.predict(xB[-1], [mA0])
    final_A = a2.predict(xA[-1], [mB0, mB1])
    assert rec["shift_A"] == pytest.approx((final_A - mA0) ** 2)
    assert rec["shift_A"] > 0
