"""Tests for the product task: y = (mu + <x^A, w^A>)(mu + <x^B, w^B>) + eps."""

import numpy as np
import pytest

from consensus import protocols
from consensus.data import ProductTwinEnv, build_env
from consensus.llm_agent import LLMTwin
from consensus.llm_backends import StubBackend
from consensus.models import LinearAgent
from consensus.protocols import LLMExchange, NoTalk
from consensus.theory_product import product_benchmarks

CFG = dict(task="product", d_A=3, d_B=3, rho=0.0, norm_A=1.0, norm_B=1.0, sigma=0.5)


def test_env_mean_and_benchmarks():
    env = build_env(CFG, np.random.default_rng(0))
    assert isinstance(env, ProductTwinEnv) and env.mu == 1.0
    x_A, x_B, y = env.sample(200_000, np.random.default_rng(1))
    np.testing.assert_allclose(np.mean(y), 1.0, atol=0.02)  # E[g_A] E[g_B] = mu^2 at rho = 0
    resid = y - env.true_mean(x_A, x_B)
    np.testing.assert_allclose(np.var(resid), 0.25, rtol=0.02)
    b = product_benchmarks(env, n_mc=200_000)
    # At rho = 0, mu = 1, |w| = 1: solo = (mu^2 + 1) * 1 + s2, product exact, the best additive
    # rule misses the interaction (variance 1), averaging also halves both signals.
    np.testing.assert_allclose(b["solo_A"] - b["full_info"], 2.0, rtol=0.03)
    np.testing.assert_allclose(b["product"] - b["full_info"], 0.0, atol=0.01)
    np.testing.assert_allclose(b["best_additive"] - b["full_info"], 1.0, rtol=0.03)
    np.testing.assert_allclose(b["no_talk"] - b["full_info"], 1.5, rtol=0.03)


def test_intercept_agent_learns_solo_mean_and_product_pool():
    env = build_env(CFG, np.random.default_rng(0))
    A = LinearAgent(3, 1.0, "A", intercept=True)
    B = LinearAgent(3, 1.0, "B", intercept=True)
    assert A.heads[0].theta.shape == (4,)
    x_A, x_B, y = env.sample(3000, np.random.default_rng(2))
    out = NoTalk(update_solo=True, pool="product").run(A, B, x_A, x_B, y)
    # The online solo fits converge to mu * g = 1 + <x, w>, so the product pool ends near Bayes.
    np.testing.assert_allclose(A.heads[0].theta, np.r_[1.0, env.w_A], atol=0.1)
    assert np.mean(out["sq_err"][-1000:]) - 0.25 < 0.1
    with pytest.raises(ValueError):
        NoTalk(pool="max")


def test_online_self_learning_twins(monkeypatch):
    """learner='self' with no training cases: no learning phase, the online system prompt, and
    (history_features) the own features in every row of the recent-steps table."""
    seen = []

    def fn(system, user, schema):
        assert "learning phase" not in system
        seen.append((system, user))
        return {"message": "m", "estimate": 1.0, "notes": "n"}

    monkeypatch.setattr(protocols, "build_backend", lambda cfg: StubBackend(fn))
    env = build_env(CFG, np.random.default_rng(0))
    A = LinearAgent(3, 1.0, "A", intercept=True)
    B = LinearAgent(3, 1.0, "B", intercept=True)
    x_A, x_B, y = env.sample(3, np.random.default_rng(3))
    proto = LLMExchange(
        backend={}, n_exchanges=1, learner="self", feature_names="distinct",
        show_partner_estimates=True, history_features=True, history_len=50,
    )
    proto.run(A.clone(), B.clone(), x_A, x_B, y)
    sysA = proto.twins["A"].system
    assert "You have not seen any earlier cases" in sysA and "studied" not in sysA.split("features.")[0]
    assert "it also starts with no earlier cases" in sysA
    assert proto.training == {"A": [], "B": []}
    last = [u for s_, u in seen if "Step 3 " in u and "agent A" in s_][0]
    assert "step | a1 | a2 | a3 | your round 1 | partner round 1 |" in last
    assert f"  1 | {x_A[0, 0]:.3f} | {x_A[0, 1]:.3f} | {x_A[0, 2]:.3f} | 1.000" in last


def test_combine_hint():
    """combine_hint='interact' adds HINT_INTERACT to the communication prompt (regression only)."""
    backend = StubBackend(lambda s, u, sc: {"message": "m", "estimate": 1.0, "notes": "n"})
    A = LinearAgent(3, 1.0, "A", intercept=True)
    tw = LLMTwin("A", A, backend, 2, 5, 0, learner="self", combine_hint="interact")
    assert "need not be to add or average them" in tw.system
    assert "multipl" not in tw.system.lower()
    plain = LLMTwin("A", A, backend, 2, 5, 0, learner="self")
    assert "interact" not in plain.system
    with pytest.raises(ValueError):
        LLMTwin("A", A, backend, 2, 5, 0, learner="self", combine_hint="multiply")
