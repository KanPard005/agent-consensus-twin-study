"""Hidden-profile task: clue bank, case generator, twins and protocols."""

import numpy as np
import pytest

from consensus import hp_protocol
from consensus.hp_clues import CLUE_BANK
from consensus.hp_data import HiddenProfileGenerator
from consensus.hp_protocol import HPDiscussion, SoloFull
from consensus.llm_backends import StubBackend


def test_keywords_identify_one_clue():
    """No clue's keyword occurs in another clue's text, so a mention maps to one clue."""
    assert {v for v, _, _ in CLUE_BANK} == {-2, -1, 1, 2}
    for i, (_, _, kws) in enumerate(CLUE_BANK):
        for j, (_, text, _) in enumerate(CLUE_BANK):
            if i != j:
                assert not any(k in text.lower() for k in kws), (kws, text)


@pytest.mark.parametrize("p_shared", [0.25, 0.5, 0.75])
def test_generator_constraints(p_shared):
    gen = HiddenProfileGenerator(p_shared=p_shared)
    cases = gen.sample_set(np.random.default_rng(1), n_hidden=8, n_control=4)
    assert sorted(c.kind for c in cases).count("hidden") == 8
    for c in cases:
        w = np.sort(c.worth)[::-1]
        assert w[0] - w[1] >= 2
        assert len({cl.bank_id for cl in c.clues}) == 18  # distinct clues
        own = [int(np.argmax(c.view(t))) for t in "AB"]
        if c.kind == "hidden":
            assert c.best not in own
        else:
            assert own == [c.best, c.best]
        # the views add up: shared clues counted in both, private ones in one
        shared = np.zeros(3)
        for cl in c.clues:
            if cl.seen_by == "both":
                shared[cl.site] += cl.value
        np.testing.assert_allclose(c.view("A") + c.view("B") - shared, c.worth)


def test_discussion_prompts_and_metrics(monkeypatch):
    """Each twin sees only its clues; partner messages and probabilities arrive from round 2;
    the team picks the argmax of the averaged final probabilities; deference is the change in
    the probability on the partner's round-1 favourite."""
    gen = HiddenProfileGenerator()
    cases = gen.sample_set(np.random.default_rng(2), n_hidden=2, n_control=0)
    seen = []

    def fn(system, user, schema):
        seen.append((system, user))
        name = "A" if "agent A" in system else "B"
        rnd = int(user.split("round ")[1].split(" ")[0])
        if rnd == 1:  # A favours site 1, B site 2
            p = [0.7, 0.2, 0.1] if name == "A" else [0.2, 0.7, 0.1]
        else:  # A moves toward B's favourite, B holds
            p = [0.3, 0.6, 0.1] if name == "A" else [0.2, 0.7, 0.1]
        return {
            "message": f"{name} says hi",
            "site_1": p[0],
            "site_2": p[1],
            "site_3": p[2],
            "notes": "n",
        }

    monkeypatch.setattr(hp_protocol, "build_backend", lambda cfg: StubBackend(fn))
    proto = HPDiscussion(backend={}, n_exchanges=2)
    m = proto.run(cases)
    case = cases[0]
    userA = [u for s, u in seen if "agent A" in s and "Case 1 of 2, round 1" in u][0]
    for cl in case.clues:
        assert (cl.text in userA) == cl.visible_to("A")
    userA2 = [u for s, u in seen if "agent A" in s and "Case 1 of 2, round 2" in u][0]
    assert 'round 1: probabilities Site 1 0.20, Site 2 0.70, Site 3 0.10; "B says hi"' in userA2
    assert "Past cases" in [u for s, u in seen if "Case 2 of 2, round 1" in u][0]
    np.testing.assert_allclose(m["defer_A"], [0.4, 0.4])  # 0.6 - 0.2 on site 2
    np.testing.assert_allclose(m["defer_B"], [0.0, 0.0])
    assert all(r["team_pick"] == 1 for r in proto.transcript)  # (0.25, 0.65, 0.1)
    np.testing.assert_array_equal(m["correct"], [c.best == 1 for c in cases])
    assert "exactly one of them is the best choice" in proto.twins["A"].system
    assert "facts that you do not" in proto.twins["A"].system


def test_solo_full_sees_everything(monkeypatch):
    gen = HiddenProfileGenerator()
    cases = gen.sample_set(np.random.default_rng(3), n_hidden=1, n_control=1)
    seen = []

    def fn(system, user, schema):
        seen.append(user)
        return {"site_1": 1.0, "site_2": 0.0, "site_3": 0.0}

    monkeypatch.setattr(hp_protocol, "build_backend", lambda cfg: StubBackend(fn))
    m = SoloFull(backend={}).run(cases)
    for c, u in zip(
        cases, sorted(seen, key=lambda u: [cl.text in u for cl in cases[0].clues], reverse=True)
    ):
        assert all(cl.text in u for cl in c.clues)
    np.testing.assert_array_equal(m["correct"], [c.best == 0 for c in cases])


def test_unusable_reply_keeps_previous(monkeypatch):
    gen = HiddenProfileGenerator()
    cases = gen.sample_set(np.random.default_rng(4), n_hidden=1, n_control=0)

    def fn(system, user, schema):
        if "round 1 of" in user:
            return {"message": "m", "site_1": 0.0, "site_2": 0.0, "site_3": 2.0, "notes": "n"}
        return {"bad": 1}

    monkeypatch.setattr(hp_protocol, "build_backend", lambda cfg: StubBackend(fn))
    proto = HPDiscussion(backend={}, n_exchanges=1)
    m = proto.run(cases)
    assert proto.transcript[0]["rounds"][1]["A"]["probs"] == [0.0, 0.0, 1.0]
    assert m["parse_fail"][0] == 2


def test_persistent_private_thinking(monkeypatch):
    """memory=persistent: one conversation per twin; round 1 of a case gives the facts (and from
    case 2 the previous outcome), later rounds only the partner's latest message; no notes in
    the schema. private_thinking: a `thinking` field the partner never sees."""
    gen = HiddenProfileGenerator()
    cases = gen.sample_set(np.random.default_rng(5), n_hidden=2, n_control=0)

    def fn(system, user, schema):
        assert "thinking" in schema["properties"] and "notes" not in schema["properties"]
        assert list(schema["properties"])[0] == "thinking"
        name = "A" if "agent A" in system else "B"
        return {
            "thinking": f"SECRET-{name}",
            "message": f"{name} says hi",
            "site_1": 0.5,
            "site_2": 0.3,
            "site_3": 0.2,
        }

    stub = StubBackend(fn)
    monkeypatch.setattr(hp_protocol, "build_backend", lambda cfg: stub)
    proto = HPDiscussion(backend={}, n_exchanges=2, memory="persistent", private_thinking=True)
    proto.run(cases)
    assert len(stub.conversations) == 2
    assert set(proto.conversations) == {"A", "B"}
    convA = [t["text"] for t in proto.conversations["A"] if t["role"] == "user"]
    assert len(convA) == 2 * 3  # 2 cases x 3 rounds
    assert "Your facts:" in convA[0] and "Outcome of case" not in convA[0]
    assert "Your facts:" not in convA[1]  # later rounds: only the partner's message
    assert "Partner's round 1 message: probabilities Site 1 0.50" in convA[1]
    assert convA[3].startswith("Outcome of case 1: the best site was Site")
    assert all("SECRET-B" not in u for u in convA)  # B's thinking never reaches A
    assert proto.transcript[0]["rounds"][0]["A"]["thinking"] == "SECRET-A"
    sysA = proto.twins["A"].system
    assert "one continuing conversation for all 2 cases" in sysA
    assert "private notes" not in sysA and "Your partner never sees it" in sysA


def test_bad_memory_rejected():
    with pytest.raises(ValueError):
        hp_protocol.HPTwin("A", None, 3, 3, 10, memory="forever")
