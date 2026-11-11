"""Tests for the consensus game (early stop on agreement, penalty) and its population theory."""

import re

import numpy as np

from consensus import protocols
from consensus.consensus_game import ConsensusExchange, LLMConsensus
from consensus.data import LinearTwinEnv
from consensus.llm_backends import StubBackend
from consensus.models import LinearAgent
from consensus.protocols import build_protocol
from consensus.theory import agreement_path, agreement_probability, rounds_to_agree


def make_twins(seed=0, n=100, rho=0.5):
    env = LinearTwinEnv.from_config(
        dict(d_A=3, d_B=3, rho=rho, norm_A=1.0, norm_B=1.0, sigma=0.5),
        np.random.default_rng(seed),
    )
    rng = np.random.default_rng(seed + 1)
    A, B = LinearAgent(3, 1.0, "A"), LinearAgent(3, 1.0, "B")
    xA, _, yA = env.sample(n, rng)
    _, xB, yB = env.sample(n, rng)
    for t in range(n):
        A.update(xA[t], [], yA[t])
        B.update(xB[t], [], yB[t])
    return env, A, B


def test_population_dialogue_agrees_at_full_information():
    # rho = 0: one exchange of solo predictions already gives both twins the full signal.
    env, _, _ = make_twins(rho=0.0)
    path = agreement_path(env.Sigma, env.w, 3, env.sigma, 3)
    assert rounds_to_agree(path) == 1
    assert np.allclose(agreement_probability(path, 0.1)[1:], 1.0)
    assert np.isclose(path.mse_consensus[1], env.sigma**2)
    # rho = 0.5: the opening gap is real, so agreement within 0.1 is rare at round 0.
    env, _, _ = make_twins(rho=0.5)
    path = agreement_path(env.Sigma, env.w, 3, env.sigma, 3)
    p = agreement_probability(path, 0.1)
    assert p[0] < 0.2 and np.all(np.diff(p) >= -1e-12)


def test_linear_consensus_records_game_loss():
    env, A, B = make_twins()
    rng = np.random.default_rng(5)
    xA, xB, y = env.sample(200, rng)
    proto = build_protocol({"name": "consensus", "n_exchanges": 3, "tol": 0.1, "penalty": 7.0})
    assert isinstance(proto, ConsensusExchange)
    out = proto.run(A.clone(), B.clone(), xA, xB, y)
    agreed = out["agreed"].astype(bool)
    assert agreed.any() and (~agreed).any()
    assert np.all(out["game_loss"][~agreed] == 7.0)
    assert np.allclose(out["game_loss"][agreed], out["agreed_sq_err"][agreed])
    assert np.all(out["rounds_used"][~agreed] == 3)


def _stub_fn(agree: bool):
    """Round 1: say 1.0 (A) or 2.0 (B). Later rounds: move to 1.5 if `agree`, else stay."""

    def fn(system, user, schema):
        me = re.search(r"You are agent (\w)", system).group(1)
        own = 1.0 if me == "A" else 2.0
        later = "round 1 " not in user.split("\n")[0]
        est = 1.5 if (agree and later) else own
        return {"message": f"I say {est}", "estimate": est, "notes": "n"}

    return fn


def _run_llm(monkeypatch, agree: bool, T: int = 3):
    backend = StubBackend(_stub_fn(agree))
    monkeypatch.setattr(protocols, "build_backend", lambda cfg: backend)
    env, A, B = make_twins()
    xA, xB, y = env.sample(T, np.random.default_rng(3))
    proto = LLMConsensus(
        backend={"kind": "stub"},
        n_exchanges=3,
        tol=0.1,
        penalty=10.0,
        learner="self",
        train_batch=50,
        show_partner_estimates=True,
        feature_names="distinct",
    )
    out = proto.run(A.clone(), B.clone(), xA, xB, y)
    return proto, out, backend, y


def test_llm_consensus_stops_on_agreement(monkeypatch):
    proto, out, backend, y = _run_llm(monkeypatch, agree=True)
    assert np.all(out["agreed"] == 1) and np.all(out["rounds_used"] == 2)
    assert np.allclose(out["game_loss"], (1.5 - y) ** 2)
    assert np.allclose(out["concede_A"], 0.5)  # both met in the middle
    # 2 training batches per twin + 3 steps x 2 rounds x 2 twins.
    assert len(backend.calls) == 2 * 2 + 3 * 2 * 2
    system = backend.calls[-1][0]
    assert "must reach a consensus" in system and "penalty of 10" in system
    assert "within 0.1" in system
    assert "round 1 of up to 4" in backend.calls[4][1]
    assert len(proto.transcript[0]["rounds"]) == 2


def test_llm_consensus_penalty_without_agreement(monkeypatch):
    proto, out, backend, _ = _run_llm(monkeypatch, agree=False)
    assert np.all(out["agreed"] == 0) and np.all(out["rounds_used"] == 4)
    assert np.all(out["game_loss"] == 10.0) and np.all(np.isnan(out["agreed_sq_err"]))
    last_user = backend.calls[-1][1]
    assert "no agreement" in last_user and "penalties included; no agreement in 2" in last_user
