"""Tests for the logistic twin task (binary y, probabilities exchanged)."""

import re

import numpy as np

from consensus import protocols
from consensus.data import LogisticTwinEnv, build_env
from consensus.llm_agent import LLMTwin, logistic_regress
from consensus.llm_backends import StubBackend
from consensus.models import LogisticAgent, OnlineLogistic, fit_logistic, log_loss, sigmoid
from consensus.protocols import Exchange, FrozenBlend, LLMExchange, NoTalk
from consensus.theory_logistic import logistic_benchmarks

CFG = dict(task="logistic", d_A=3, d_B=3, rho=0.0, norm_A=1.0, norm_B=1.0, scale=2.0)


def make_twins(seed=0, n=300, **kw):
    env = build_env({**CFG, **kw}, np.random.default_rng(seed))
    rng = np.random.default_rng(seed + 1)
    A, B = LogisticAgent(env.d_A, 1.0, "A"), LogisticAgent(env.d_B, 1.0, "B")
    xA, _, yA = env.sample(n, rng)
    _, xB, yB = env.sample(n, rng)
    for t in range(n):
        A.update(xA[t], [], yA[t])
        B.update(xB[t], [], yB[t])
    return env, A, B


def test_env_labels_are_binary_with_true_probabilities():
    env = build_env(CFG, np.random.default_rng(0))
    assert isinstance(env, LogisticTwinEnv)
    xA, xB, y = env.sample(20000, np.random.default_rng(1))
    assert set(np.unique(y)) <= {0.0, 1.0}
    p = env.true_prob(xA, xB)
    assert abs(y.mean() - p.mean()) < 0.01


def test_online_logistic_close_to_batch_fit():
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((3000, 3))
    y = (rng.random(3000) < sigmoid(Z @ np.array([1.0, -2.0, 0.5]))).astype(float)
    theta, _ = fit_logistic(Z, y)
    h = OnlineLogistic(3)
    for z, v in zip(Z, y):
        h.update(z, v)
    np.testing.assert_allclose(h.theta, theta, atol=0.1)
    np.testing.assert_allclose(theta, [1.0, -2.0, 0.5], atol=0.15)


def test_agent_refits_head0_exactly_at_begin_communication():
    env, A, _ = make_twins()
    A.begin_communication(1)
    Z = np.array([z for z, _ in A._train])
    y = np.array([v for _, v in A._train])
    theta, _ = fit_logistic(Z, y, P0=np.eye(3))
    np.testing.assert_allclose(A.heads[0].theta, theta, atol=1e-8)
    # Head 1 starts at the solo opinion: a partner message of 1/2 (log-odds 0) changes nothing.
    x = np.array([0.3, -1.0, 0.2])
    assert abs(A.predict(x, [0.5]) - A.predict(x, [])) < 1e-12


def test_benchmarks_order_at_rho0():
    env = build_env(CFG, np.random.default_rng(0))
    b = logistic_benchmarks(env, n_exchanges=2, n_mc=50_000)
    ll = {k[8:]: v for k, v in b.items() if k.startswith("logloss_")}
    assert ll["full_info"] < ll["blend"] + 1e-3 <= ll["logit_sum"] + 1e-3 < ll["no_talk"]
    assert ll["no_talk"] < ll["solo_A"]
    # Independent blocks: the best blend puts weight > 1 on each log-odds (the solo models are
    # attenuated by the partner's unseen half), and exchange reaches full information.
    assert b["blend_alpha"] > 1 and b["blend_beta"] > 1
    assert ll["exchange_1"] - ll["full_info"] < 2e-3


def test_linear_protocols_record_log_loss():
    env, A, B = make_twins()
    xA, xB, y = env.sample(200, np.random.default_rng(7))
    for proto in (NoTalk(), NoTalk(pool="logit_sum"), FrozenBlend(), Exchange(2)):
        out = proto.run(A.clone(), B.clone(), xA, xB, y)
        assert np.all(np.isfinite(out["log_loss"])) and np.all(out["sq_err"] <= 1)
    # On average, summing log-odds beats averaging probabilities at rho = 0.
    ls = NoTalk(pool="logit_sum").run(A.clone(), B.clone(), xA, xB, y)["log_loss"].mean()
    avg = NoTalk().run(A.clone(), B.clone(), xA, xB, y)["log_loss"].mean()
    assert ls < avg


def test_logistic_regress_report():
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(40):
        p = float(rng.uniform(0.05, 0.95))
        rows.append({"own_model": p, "partner_open": 0.5, "y": float(rng.random() < p)})
    out = logistic_regress(rows, ["own_model"], {"own_model": 0.8})
    assert "logit(own_model)" in out and "leave-one-out log loss" in out
    p = float(re.search(r"current case: (\d+\.\d+)", out).group(1))
    assert 0 < p < 1
    assert "Not enough" in logistic_regress(rows[:2], ["own_model"], {})


def test_llm_twins_logistic_prompt_and_scoring(monkeypatch):
    def fn(system, user, schema):
        assert "probability that y = 1" in user and "linear model" not in user
        assert schema["properties"]["estimate"]["maximum"] == 1
        p = float(re.search(r"probability that y = 1: ([\d.]+)", user).group(1))
        return {"message": "m", "estimate": p, "notes": "", "regress": []}

    backend = StubBackend(fn)
    monkeypatch.setattr(protocols, "build_backend", lambda cfg: backend)
    env, A, B = make_twins()
    xA, xB, y = env.sample(12, np.random.default_rng(3))
    proto = LLMExchange(
        backend={}, n_exchanges=1, tools=True, auto_regress=True, show_partner_estimates=True
    )
    out = proto.run(A.clone(), B.clone(), xA, xB, y)
    # The twins' tools are the refitted solo models (refit happens at begin_communication).
    A2, B2 = A.clone(), B.clone()
    A2.begin_communication(0, frozen=True)
    B2.begin_communication(0, frozen=True)
    expected = (
        np.array([A2.predict(a, []) for a in xA]) + np.array([B2.predict(b, []) for b in xB])
    ) / 2
    np.testing.assert_allclose(out["log_loss"], log_loss(expected, y), atol=1e-2)
    twin = proto.twins["A"]
    assert "log loss" in twin.system and "yes/no outcome" in twin.system
    assert (
        "Logistic regression of y"
        in proto.transcript[-1]["rounds"][1]["A"]["tool_calls"][0]["result"]
        or "Not enough" in proto.transcript[-1]["rounds"][1]["A"]["tool_calls"][0]["result"]
    )
    assert isinstance(twin, LLMTwin) and twin.logistic


def test_pooling_hint_in_system_prompt():
    _, A, _ = make_twins()
    plain = LLMTwin("A", A, None, 2, 10, 100)
    ind = LLMTwin("A", A, None, 2, 10, 100, pooling_hint="independent")
    lo = LLMTwin("A", A, None, 2, 10, 100, pooling_hint="logodds")
    assert "independent" not in plain.system
    assert "statistically independent" in ind.system and "log-odds add" not in ind.system
    assert "log-odds add" in lo.system and lo.system.startswith(plain.system.split("\n\n")[0])
