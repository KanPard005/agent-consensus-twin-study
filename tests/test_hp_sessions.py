"""Tests for the café task with session twins (`consensus.hp_sessions`), with scripted twins in
threads instead of live models."""

import sys
import threading
from pathlib import Path

import numpy as np

from consensus.hp_data import HiddenProfileGenerator
from consensus.hp_sessions import SYSTEM_HP, HPSessionAPI, HPTwinSession, normalise_probs

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_hp_sessions import case_metrics  # noqa: E402


def cases(n_hidden=2, n_control=1):
    return HiddenProfileGenerator().sample_set(np.random.default_rng(0), n_hidden, n_control)


def probs(site, n=3):
    return {f"site_{s + 1}": (0.8 if s == site else 0.1) for s in range(n)}


def test_normalise_probs():
    p = normalise_probs({"site_1": 2, "site_2": 1, "site_3": 1}, 3)
    assert np.allclose(p, [0.5, 0.25, 0.25])
    assert isinstance(normalise_probs({"site_1": 1}, 3), str)
    assert isinstance(normalise_probs({"site_1": -1, "site_2": 1, "site_3": 1}, 3), str)
    assert isinstance(normalise_probs({"site_1": 0, "site_2": 0, "site_3": 0}, 3), str)


def test_full_run_with_scripted_twins():
    cs = cases()
    api = HPSessionAPI(cs)
    recs, out = [], {"A": [], "B": []}
    api.on_case = recs.append

    def twin(k):
        for t in range(len(cs)):
            facts = api.next_case(k)
            assert f"Case {t + 1} of 3" in facts
            # each twin sees only its own clues
            mine = {c.text for c in cs[t].clues if c.visible_to(k)}
            assert all(text in facts for text in mine)
            assert api.send_message(k, "hi").startswith("Error")  # opening first
            api.submit_opening(k, probs(0))
            if k == "A":
                api.send_message(k, f"case {t} from A")
            out[k].append(api.read_messages(k))
            out[k].append(api.submit_final(k, probs(cs[t].best if k == "A" else 1)))
        out[k].append(api.next_case(k))

    th = [threading.Thread(target=twin, args=(k,)) for k in "AB"]
    for x in th:
        x.start()
    for x in th:
        x.join(timeout=10)
    assert not any(x.is_alive() for x in th)
    assert len(recs) == 3 and api.resolved.all()
    assert all(len(r["messages"]) == 1 for r in recs)
    assert "run is over" in out["A"][-1] and "run is over" in out["B"][-1]
    # B always reads A's message for the case
    assert all('Partner: "case' in s for s in out["B"][0::2][:3])
    # team = average of finals
    r0 = recs[0]
    assert np.allclose(r0["team"], (np.array(r0["final"]["A"]) + r0["final"]["B"]) / 2)
    m = case_metrics(r0)
    assert m["messages"] == 1 and m["agree"] in (0.0, 1.0)


def test_both_reading_releases():
    cs = cases(1, 0)
    api = HPSessionAPI(cs)
    res = {}

    def twin(k):
        api.next_case(k)
        api.submit_opening(k, probs(0))
        res[k] = api.read_messages(k)
        api.submit_final(k, probs(0))

    th = [threading.Thread(target=twin, args=(k,)) for k in "AB"]
    for x in th:
        x.start()
    for x in th:
        x.join(timeout=10)
    assert not any(x.is_alive() for x in th)
    assert "No new messages" in res["A"] and "No new messages" in res["B"]


def test_partner_gone_resolves_case():
    cs = cases(1, 0)
    api = HPSessionAPI(cs)
    res = {}

    def a():
        api.next_case("A")
        api.submit_opening("A", probs(0))
        res["A"] = api.submit_final("A", probs(0))

    th = threading.Thread(target=a)
    b_started = threading.Thread(target=lambda: api.next_case("B"))
    th.start()
    b_started.start()
    b_started.join(timeout=5)
    api.finish("B")
    th.join(timeout=5)
    assert not th.is_alive()
    assert "partner stopped" in res["A"] and api.resolved[0]


def test_cap_blocks_talk_but_not_final():
    cs = cases(1, 0)
    api = HPSessionAPI(cs, cap=2)
    api.q["A"] = 0  # pretend the case started
    api.opening["A"][0] = [1 / 3] * 3
    calls = [api.call("A", "send_message", {}, lambda: api.send_message("A", "x"))
             for _ in range(3)]
    assert calls[:2] == ["Message sent."] * 2 and "limit" in calls[2]


def test_system_prompt_neutral_and_complete():
    api = HPSessionAPI(cases())
    tw = HPTwinSession("A", api)
    s = tw.system
    for word in ("next_case", "submit_opening", "send_message", "read_messages", "submit_final",
                 "facts that you do not", "same AI model", "3 cases"):
        assert word in s, word
    # nothing tells the twins how to use the partner or what to share
    for word in ("should share", "unshared", "hidden profile", "pool"):
        assert word not in s.lower()
    assert "{" not in s and SYSTEM_HP


def test_allowed_tools_match_server_key():
    # TwinSession._main registers the server as "data"; allowed names must use that prefix.
    from consensus.hp_sessions import build_hp_tools

    _, allowed = build_hp_tools(HPSessionAPI(cases()), "A")
    assert sorted(allowed) == sorted(
        f"mcp__data__{n}"
        for n in ("next_case", "submit_opening", "send_message", "read_messages", "submit_final")
    )
