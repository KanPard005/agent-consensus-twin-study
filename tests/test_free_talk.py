"""Tests for free talk with persistent self-learning twins and compaction (stub backend)."""

import re

import numpy as np
import pytest

from consensus import protocols
from consensus.data import build_env
from consensus.free_talk import FreeTalkExchange, PersistentTwin
from consensus.llm_backends import StubBackend
from consensus.models import LinearAgent

ENV = dict(cov="overlap", d_A=3, d_B=3, overlap=1, w=[0.7, -0.5, 0.8, 0.6, -0.6], sigma=0.5)
LABELS = {"A": ["x1", "x2", "x3"], "B": ["x3", "x4", "x5"]}


def trained_twins(n=50, seed=0):
    env = build_env(ENV, np.random.default_rng(seed))
    rng = np.random.default_rng(seed + 1)
    A, B = LinearAgent(3, 1.0, "A"), LinearAgent(3, 1.0, "B")
    xA, _, yA = env.sample(n, rng)
    _, xB, yB = env.sample(n, rng)
    for t in range(n):
        A.update(xA[t], [], yA[t])
        B.update(xB[t], [], yB[t])
    return env, A, B


def feature_sum(user: str) -> float:
    """Sum of the feature values shown in the turn (0 if none)."""
    m = re.search(r"Your features: (.*)", user)
    if not m:
        return 0.0
    return sum(float(v) for v in re.findall(r"=(-?[\d.]+)", m.group(1)))


@pytest.fixture
def stub(monkeypatch):
    """Training -> notes; summary request -> summary; a round -> the feature sum, done in
    round 2; one tool call in round 1 of step 3."""

    def fn(system, user, schema):
        props = schema["properties"]
        if "summary" in props:
            return {"summary": "SUMMARY: add the partner's non-shared part."}
        if "notes" in props and "estimate" not in props:
            return {"notes": "y = x1 + x2 + x3"}
        rnd = int(re.search(r"round (\d+) of up to", user).group(1)) if "of up to" in user else 1
        call = "Step 3 of" in user and "Tool call" not in user
        return {
            "message": "m" * 10,
            "estimate": feature_sum(user),
            "done": rnd >= 2,
            "regress": ["own_open", "x1"] if call else [],
        }

    backend = StubBackend(fn)
    monkeypatch.setattr(protocols, "build_backend", lambda cfg: backend)
    return backend


def run(compact_tokens, T=8):
    env, A, B = trained_twins()
    proto = FreeTalkExchange(
        max_rounds=3,
        compact_tokens=compact_tokens,
        backend={"kind": "stub"},
        labels=LABELS,
        max_chars=2000,
        structure_hint=True,
        partner_info=True,
    )
    rng = np.random.default_rng(5)
    xA, xB, y = env.sample(T, rng)
    out = proto.run(A.clone(), B.clone(), xA, xB, y)
    return proto, out


def test_free_talk_stops_when_both_done(stub):
    proto, out = run(None)
    assert np.all(out["n_rounds"] == 2)  # both say done in round 2
    assert np.all(out["compactions"] == 0)
    # Never compacted: one conversation per twin, opened with the learning-phase notes.
    tw = proto.twins["A"]
    assert len(stub.conversations) == 2
    assert tw.conv.turns[0]["text"].startswith("Your notes from the learning phase:\ny = x1")
    assert "compacted" not in tw.system and "regression tool" in tw.system


def test_compaction_restarts_from_summary(stub):
    proto, out = run(500)  # tiny threshold: compact after every step
    tw = proto.twins["A"]
    assert out["compactions"].sum() > 0 and len(tw.compactions) >= 1
    # The first turn of the second conversation (the last step's compaction leaves conv empty).
    first = next(t["text"] for t in tw.all_turns if t["conversation"] == 1 and t["role"] == "user")
    assert first.startswith("This conversation continues")
    assert "SUMMARY: add the partner's non-shared part." in first
    assert "The last step before compaction, verbatim:" in first
    assert "When the conversation gets long" in tw.system
    # Bounded: every conversation stays small (one restarted step + one new step).
    sizes = [sum(len(t["text"]) for t in tw.all_turns if t["conversation"] == c) // 4
             for c in range(len(tw.compactions))]
    assert max(sizes) < 2000


def test_tool_columns_and_auto_run(stub):
    proto, _ = run(None, T=7)
    tw = proto.twins["B"]
    assert tw.columns == ["x3", "x4", "x5", "own_open", "partner_open"]
    step3 = proto.transcript[2]["rounds"][0]["B"]["tool_calls"]
    assert step3[0]["auto"] and step3[0]["predictors"] == ["x3", "x4", "x5"]
    assert step3[1]["predictors"] == ["own_open", "x1"] or step3[1]["predictors"] == ["own_open"]
    # Round 2's automatic run is on own_open and partner_open, with a prediction.
    r2 = proto.transcript[6]["rounds"][1]["A"]["tool_calls"][0]
    assert r2["predictors"] == ["own_open", "partner_open"]
    assert "Prediction for the current case" in r2["result"]


def test_persistent_twin_prompt_has_no_notes_field():
    _, A, _ = trained_twins()
    tw = PersistentTwin("A", A, StubBackend(lambda *a: {}), 3, 10, 50, feature_labels=LABELS["A"],
                        partner_labels=LABELS["B"])
    assert "notes" not in tw.schema["properties"] and "done" in tw.schema["properties"]
