"""Tests for the LLM twin protocol with a stub backend (no model calls)."""

import re

import numpy as np
import pytest

from consensus import protocols
from consensus.data import LinearTwinEnv
from consensus.llm_agent import LLMTwin, StepRecord, regress
from consensus.llm_backends import StubBackend
from consensus.models import LinearAgent
from consensus.protocols import METRICS, LLMExchange


def make_twins(seed=0, n=100):
    env = LinearTwinEnv.from_config(
        dict(d_A=3, d_B=2, rho=0.6, norm_A=1.0, norm_B=1.0, sigma=0.3),
        np.random.default_rng(seed),
    )
    rng = np.random.default_rng(seed + 1)
    A, B = LinearAgent(env.d_A, 1.0, "A"), LinearAgent(env.d_B, 1.0, "B")
    xA, _, yA = env.sample(n, rng)
    _, xB, yB = env.sample(n, rng)
    for t in range(n):
        A.update(xA[t], [], yA[t])
        B.update(xB[t], [], yB[t])
    return env, A, B


def model_pred_from_prompt(user: str) -> float:
    """The linear prediction shown in the prompt (0 if absent, e.g. a persistent round-2 turn)."""
    m = re.search(r"linear model's prediction: (-?[\d.]+)", user)
    return float(m.group(1)) if m else 0.0


@pytest.fixture
def stub(monkeypatch):
    """Patch build_backend so LLMExchange uses a stub that echoes its linear prediction."""

    def fn(system, user, schema):
        m = model_pred_from_prompt(user)
        return {"message": f"my model says {m:.3f}", "estimate": m, "notes": "n"}

    backend = StubBackend(fn)
    monkeypatch.setattr(protocols, "build_backend", lambda cfg: backend)
    return backend


def test_echo_twins_match_no_talk(stub):
    env, A, B = make_twins()
    xA, xB, y = env.sample(20, np.random.default_rng(5))
    out = LLMExchange(backend={}, n_exchanges=1).run(A.clone(), B.clone(), xA, xB, y)
    assert set(METRICS) <= set(out)
    expected = np.array([(A.predict(a, []) + B.predict(b, [])) / 2 for a, b in zip(xA, xB)])
    # The stub rounds the prediction to 3 decimals, as the prompt does.
    np.testing.assert_allclose(out["sq_err"], (expected - y) ** 2, atol=1e-2)
    assert np.all(out["parse_fail"] == 0)
    assert len(stub.calls) == 20 * 2 * 2  # steps x rounds x twins


def test_second_round_sees_partner_message_and_history(stub):
    env, A, B = make_twins()
    xA, xB, y = env.sample(3, np.random.default_rng(6))
    proto = LLMExchange(backend={}, n_exchanges=1)
    proto.run(A.clone(), B.clone(), xA, xB, y)
    users = [u for _, u in stub.calls]
    assert "none yet" in users[0]
    assert any('round 1: "my model says' in u for u in users[2:4])
    assert "Recent steps" in users[-1] and "Mean squared error" in users[-1]
    systems = {s for s, _ in stub.calls}
    assert any("agent A" in s for s in systems) and any("agent B" in s for s in systems)
    assert len(proto.transcript) == 3 and len(proto.transcript[0]["rounds"]) == 2


def test_unusable_reply_falls_back_to_linear_prediction():
    _, A, _ = make_twins()
    twin = LLMTwin("A", A, StubBackend(lambda s, u, sc: None), n_rounds=2, T=5, n_train=100)
    x = np.array([0.1, -0.2, 0.3])
    reply = twin.act(0, 0, x, [])
    assert not reply.ok and reply.estimate == pytest.approx(A.predict(x, []))


# System prompt of the first LLM run (results/llm_haiku/2026-11-05_15-04-08); the defaults must
# reproduce it so later runs stay comparable.
FIRST_RUN_SYSTEM_PROMPT = """\
You are agent {name}, one of two agents (A and B) who predict a number together.

Each step there is a new case with an unknown real-valued target y. You see {d} numeric \
features of the case; your partner sees different features of the same case. {twin_line}

You have a linear model that was fitted to {n_train} earlier cases using only your features. \
Its prediction for the current case is given to you every step. It is a tool: you decide how \
much to rely on it.

Each step has {n_rounds} rounds. In every round you and your partner simultaneously send each \
other a message (at most {max_chars} characters) and each give an estimate of y. From round 2 \
on you see what your partner sent in the earlier rounds of this step. The team's prediction is \
the average of your two final-round estimates, and the team is scored by squared error. After \
the final round the true y is revealed to both of you.

You keep private notes that persist across steps (at most {notes_chars} characters). Rewrite \
them every round with whatever will help you later, for example how reliable your model and \
your partner are, how to combine your information, or conventions you and your partner use.\
"""


def test_default_prompt_matches_first_run():
    _, A, _ = make_twins()
    twin = LLMTwin("A", A, StubBackend(lambda *a: None), n_rounds=2, T=60, n_train=100)
    expected = FIRST_RUN_SYSTEM_PROMPT.format(
        name="A",
        d=3,
        twin_line=(
            "Your partner is another instance of the same AI model as you, with the same "
            "instructions and its own linear model fitted to its own features."
        ),
        n_train=100,
        n_rounds=2,
        max_chars=300,
        notes_chars=800,
    )
    assert twin.system == expected
    assert "regress" not in twin.schema["properties"]


def test_regress_recovers_coefficients_and_predicts():
    rng = np.random.default_rng(0)
    rows = [{"a": a, "b": b, "y": 1.0 + 2.0 * a - 0.5 * b} for a, b in rng.standard_normal((30, 2))]
    out = regress(rows, ["a", "b"], {"a": 1.0, "b": 2.0})
    assert "intercept +1.000" in out and "a +2.000" in out and "b -0.500" in out
    assert "Prediction for the current case: 2.000" in out
    assert "not known yet" in regress(rows, ["a", "b"], {"a": 1.0})
    assert regress(rows[:3], ["a", "b"], {}).startswith("Not enough past steps")


def test_tool_call_then_answer_and_joint_prompt():
    _, A, _ = make_twins()
    replies = iter(
        [
            {"message": "", "estimate": 0.0, "notes": "", "regress": ["own_model", "partner_open"]},
            {"message": "hi", "estimate": 1.5, "notes": "n", "regress": []},
        ]
    )
    backend = StubBackend(lambda s, u, sc: next(replies))
    twin = LLMTwin(
        "A",
        A,
        backend,
        n_rounds=2,
        T=60,
        n_train=100,
        team_rule="joint",
        show_partner_estimates=True,
        tools=True,
    )
    for t in range(6):  # some history for the regression
        twin.record(
            StepRecord(t, 0.1 * t, 0.0, 0.0, 0.0, 0.3 * t, partner_open=0.2 * t), np.ones(3)
        )
    reply = twin.act(6, 1, np.ones(3), [("partner says 0.5", 0.5)])
    assert reply.ok and reply.estimate == 1.5 and len(reply.tool_calls) == 1
    assert "Prediction for the current case" in reply.tool_calls[0]["result"]
    assert "Tool call 1" in backend.calls[1][1]
    assert "joint" in twin.system and "regress" in twin.schema["properties"]
    assert "estimate 0.500" in backend.calls[0][1]


def test_cli_backend_error_handling(monkeypatch):
    from consensus.llm_backends import ClaudeCLIBackend

    outs = iter(
        [
            {"is_error": True, "terminal_reason": "api_error", "api_error_status": 529},
            {"is_error": True, "subtype": "error_max_structured_output_retries"},
        ]
    )
    b = ClaudeCLIBackend("m", backoff=0.0)
    monkeypatch.setattr(b, "_run", lambda cmd, user: next(outs))
    reply = b.complete("s", "u", {})  # retried once, then an unusable (not fatal) reply
    assert reply.data is None and reply.meta["subtype"] == "error_max_structured_output_retries"
    monkeypatch.setattr(b, "_run", lambda cmd, user: {"is_error": True, "result": "Not logged in"})
    with pytest.raises(RuntimeError, match="not logged in"):
        b.complete("s", "u", {})


def test_auto_regress_shown_every_round():
    _, A, _ = make_twins()
    backend = StubBackend(
        lambda s, u, sc: {"message": "", "estimate": 1.0, "notes": "", "regress": []}
    )
    twin = LLMTwin(
        "A",
        A,
        backend,
        n_rounds=2,
        T=60,
        n_train=100,
        show_partner_estimates=True,
        tools=True,
        auto_regress=True,
    )
    for t in range(6):
        twin.record(
            StepRecord(t, 0.1 * t, 0.0, 0.0, 0.0, 0.3 * t, partner_open=0.2 * t), np.ones(3)
        )
    r0 = twin.act(6, 0, np.ones(3), [])
    r1 = twin.act(6, 1, np.ones(3), [("hi", 0.5)])
    assert "Automatic tool run: regress on own_model. Result" in backend.calls[0][1]
    assert "regress on own_model, partner_open" in backend.calls[1][1]
    assert r0.tool_calls[0]["auto"] and r1.tool_calls[0]["predictors"] == [
        "own_model",
        "partner_open",
    ]
    assert "run once for you automatically" in twin.system
    with pytest.raises(ValueError):
        LLMTwin("A", A, backend, n_rounds=2, T=60, n_train=100, auto_regress=True)


def test_persistent_twins_keep_one_conversation_per_run(stub):
    env, A, B = make_twins()
    xA, xB, y = env.sample(3, np.random.default_rng(7))
    proto = LLMExchange(backend={}, n_exchanges=1, memory="persistent")
    out = proto.run(A.clone(), B.clone(), xA, xB, y)
    assert np.all(out["parse_fail"] == 0)
    assert len(stub.conversations) == 2  # one per twin
    conv_A = next(c for c in stub.conversations if "agent A" in c.system)
    users = [t["text"] for t in conv_A.turns if t["role"] == "user"]
    assert len(users) == 3 * 2  # steps x rounds, all in one conversation
    assert "Step 1 of 3, round 1" in users[0] and "Outcome of step" not in users[0]
    assert users[1].startswith("Step 1, round 2") and "Partner's round 1 message" in users[1]
    assert "Your features" not in users[1]  # later rounds only add the new message
    assert users[2].startswith("Outcome of step 1: true y")
    assert "continuing conversation" in conv_A.system and "private notes" not in conv_A.system
    assert set(proto.conversations) == {"A", "B"} and len(proto.conversations["A"]) == 12
    # A new run builds new twins, hence new conversations.
    proto.run(A.clone(), B.clone(), xA, xB, y)
    assert len(stub.conversations) == 4


def test_persistent_schema_and_tool_followup():
    _, A, _ = make_twins()
    replies = iter(
        [
            {"message": "", "estimate": 0.0, "regress": ["own_model"]},
            {"message": "hi", "estimate": 1.5, "regress": []},
        ]
    )
    backend = StubBackend(lambda s, u, sc: next(replies))
    twin = LLMTwin("A", A, backend, n_rounds=2, T=60, n_train=100, tools=True, memory="persistent")
    assert "notes" not in twin.schema["properties"]
    for t in range(6):
        twin.record(StepRecord(t, 0.1 * t, 0.0, 0.0, 0.0, 0.3 * t), np.ones(3))
    reply = twin.act(6, 0, np.ones(3), [])
    assert reply.ok and reply.estimate == 1.5 and reply.notes == ""
    followup = twin.conv.turns[2]["text"]
    assert (
        followup.startswith("Tool call 1: regress on own_model") and "Your features" not in followup
    )


def test_cli_conversation_uses_session_id_then_resume(monkeypatch):
    from consensus.llm_backends import ClaudeCLIBackend

    b = ClaudeCLIBackend("m", backoff=0.0)
    conv = b.conversation("sys")
    cmds = []

    def fake_run(cmd, user):
        cmds.append(cmd)
        return {"is_error": False, "session_id": conv.session_id, "structured_output": {"n": 1}}

    monkeypatch.setattr(b, "_run", fake_run)
    conv.send("first", {})
    conv.send("second", {})
    assert cmds[0][cmds[0].index("--session-id") + 1] == conv.session_id
    assert cmds[1][cmds[1].index("--resume") + 1] == conv.session_id
    assert "--no-session-persistence" not in cmds[0] + cmds[1]
    assert [t["role"] for t in conv.turns] == ["user", "assistant", "user", "assistant"]


def test_api_conversation_appends_history_and_caches():
    from types import SimpleNamespace

    from consensus.llm_backends import AnthropicBackend

    seen = []

    def create(**kw):
        seen.append({**kw, "messages": list(kw["messages"])})
        block = SimpleNamespace(type="text", text='{"message": "", "estimate": 1.0}')
        usage = SimpleNamespace(
            input_tokens=1,
            output_tokens=1,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        )
        return SimpleNamespace(content=[block], usage=usage, stop_reason="end_turn")

    b = object.__new__(AnthropicBackend)  # skip __init__: no API key needed
    b.client = SimpleNamespace(messages=SimpleNamespace(create=create))
    b.model, b.effort, b.max_tokens = "m", "low", 100
    conv = b.conversation("sys")
    assert conv.send("u1", {}).data == {"message": "", "estimate": 1.0}
    conv.send("u2", {})
    assert [m["role"] for m in seen[1]["messages"]] == ["user", "assistant", "user"]
    assert seen[1]["messages"][1]["content"][0].text.startswith('{"message"')  # passed back as is
    assert seen[1]["cache_control"] == {"type": "ephemeral"} and seen[1]["system"] == "sys"


def test_conversation_shows_both_sides(stub):
    env, A, B = make_twins()
    xA, xB, y = env.sample(2, np.random.default_rng(7))
    proto = LLMExchange(backend={}, n_exchanges=2, show_own_messages=True)
    proto.run(A.clone(), B.clone(), xA, xB, y)
    users = [u for s, u in stub.calls if "agent A" in s]
    round3 = [u for u in users if "round 3 of 3" in u]
    assert len(round3) == 2  # one per step
    for u in round3:
        assert "Conversation this step so far:" in u
        assert 'round 1, you: "my model says' in u and 'round 2, partner: "my model says' in u
        assert u.index("round 1, you") < u.index("round 1, partner") < u.index("round 2, you")
    # Own messages from the previous step do not leak into the next step's round 2.
    round2 = [u for u in users if "round 2 of 3" in u]
    assert all(u.count(", you:") == 1 for u in round2)
    # Default (P8-style) prompts are unchanged.
    stub.calls.clear()
    LLMExchange(backend={}, n_exchanges=2).run(A.clone(), B.clone(), xA, xB, y)
    assert not any("Conversation this step" in u or ", you:" in u for _, u in stub.calls)


def test_self_learning_twins(monkeypatch):
    """learner='self': training batches build the notes, no linear prediction is shown, the solo
    opinion is the round-1 estimate, and unusable replies fall back to the last own estimate."""
    seen = {"train": 0, "rounds": []}

    def fn(system, user, schema):
        if "learning phase" in system:
            seen["train"] += 1
            assert "Cases (x1" in user and "| y" in user
            return {"notes": f"after batch {seen['train']}"}
        assert "linear model" not in user and "linear model" not in system
        seen["rounds"].append(user)
        if "round 2 of 2" in user and "Step 2 " in user:
            return {"bad": 1}  # unusable -> keep round-1 estimate
        return {"message": "hi", "estimate": 1.0 if "round 1 of 2" in user else 2.0, "notes": "n"}

    backend = StubBackend(fn)
    monkeypatch.setattr(protocols, "build_backend", lambda cfg: backend)
    env, A, B = make_twins(n=60)
    xA, xB, y = env.sample(3, np.random.default_rng(5))
    proto = LLMExchange(
        backend={}, n_exchanges=1, learner="self", train_batch=25, show_partner_estimates=True
    )
    out = proto.run(A.clone(), B.clone(), xA, xB, y)
    assert seen["train"] == 2 * 3  # 60 cases in batches of 25 -> 3 batches, per twin
    assert len(proto.training["A"]) == 3
    assert all(r["ok"] and r["notes"].startswith("after batch") for r in proto.training["A"])
    assert "Before this phase you studied 60 earlier cases" in proto.twins["A"].system
    first = [u for u in seen["rounds"] if "Step 1 " in u and "round 1" in u][0]
    assert "after batch" in first  # notes from training carry over
    np.testing.assert_allclose(out["shift_A"], [1.0, 0.0, 1.0])  # (final - round 1)^2
    assert proto.transcript[1]["rounds"][1]["A"]["estimate"] == 1.0  # fallback
    assert "linear_ref" in proto.transcript[0]
