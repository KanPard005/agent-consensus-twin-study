"""Worded twins (idea 3a): the S1 task told as house sales."""

import re

import numpy as np
import pytest

from consensus import protocols
from consensus.data import LinearTwinEnv
from consensus.llm_backends import StubBackend
from consensus.models import LinearAgent
from consensus.protocols import build_protocol
from consensus.worded import HOUSE, LOCATION, WordedTwin


def make_twins(seed=0, n=60):
    env = LinearTwinEnv.from_config(
        dict(d_A=3, d_B=3, rho=0.0, norm_A=1.0, norm_B=1.0, sigma=0.5),
        np.random.default_rng(seed),
    )
    rng = np.random.default_rng(seed + 1)
    A, B = LinearAgent(3, 1.0, "A"), LinearAgent(3, 1.0, "B")
    xA, xB, y = env.sample(n, rng)
    for t in range(n):
        A.update(xA[t], [], y[t])
        B.update(xB[t], [], y[t])
    return env, A, B


def test_worded_prompts(monkeypatch):
    """Training cases and step prompts are sentences about the house (A) or location (B), no
    abstract feature vectors or x1..x3 appear, and the system prompt tells the story."""
    seen = {"train": [], "rounds": []}

    def fn(system, user, schema):
        if "learning phase" in system:
            seen["train"].append((system, user))
            return {"notes": "rule"}
        seen["rounds"].append((system, user))
        return {"message": "hi", "estimate": 0.5, "notes": "n"}

    monkeypatch.setattr(protocols, "build_backend", lambda cfg: StubBackend(fn))
    env, A, B = make_twins()
    xA, xB, y = env.sample(2, np.random.default_rng(3))
    proto = build_protocol(
        dict(
            name="llm_worded",
            backend={},
            n_exchanges=1,
            learner="self",
            train_batch=25,
            show_partner_estimates=True,
        )
    )
    proto.run(A.clone(), B.clone(), xA, xB, y)
    sysA = proto.twins["A"].system
    assert "house in the same city is sold" in sysA
    assert "3 facts about the house itself (floor area, condition, garden)" in sysA
    assert "about its location, that you do not see" in sysA
    assert "numeric features" not in sysA
    trainA = [u for s, u in seen["train"] if "agent A" in s]
    trainB = [u for s, u in seen["train"] if "agent B" in s]
    assert len(trainA) == 3 and "Sale 1: floor area" in trainA[0] and "Sale 26:" in trainA[1]
    assert "Sale 1: school rating" in trainB[0]
    rounds = [u for _, u in seen["rounds"]]
    assert any("This house (the house itself): floor area" in u for u in rounds)
    assert any("This house (its location): school rating" in u for u in rounds)
    assert not any(re.search(r"\bx[1-3]\b", u) for u in rounds + trainA + trainB)


def test_signs_follow_own_fit():
    """Display signs make each attribute raise y in the twin's own training data, and showing a
    case loses no information beyond rounding."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 3))
    y = X @ np.array([0.8, -0.5, 0.3]) + 0.1 * rng.standard_normal(200)
    twin = WordedTwin(
        "A",
        LinearAgent(3, 1.0, "A"),
        StubBackend(lambda *a: {"notes": "r"}),
        2,
        60,
        200,
        learner="self",
    )
    twin.train(X, y)
    np.testing.assert_array_equal(twin.signs, [1.0, -1.0, 1.0])
    x = np.array([1.0, 1.0, -2.0])
    text = twin.describe(x)
    assert text == "floor area 160 m², condition 4.9/10, garden 140 m²"
    # Inverting the display recovers x up to the rounding (here exact).
    assert HOUSE[1].mean - HOUSE[1].scale * x[1] == pytest.approx(4.9)


def test_worded_needs_self_learning():
    with pytest.raises(ValueError):
        WordedTwin("A", LinearAgent(3, 1.0, "A"), None, 2, 60, 200)
    with pytest.raises(ValueError):
        WordedTwin("B", LinearAgent(2, 1.0, "B"), None, 2, 60, 200, learner="self")
    assert len(LOCATION) == 3
