"""Tests for the agentic twins and the data-generator API (scripted stub policy, no model)."""

import re
import threading

import numpy as np

from consensus.agentic import AgentTwin, DataGeneratorAPI, run_episode
from consensus.data import build_env
from consensus.llm_backends import StubBackend

ENV = dict(cov="overlap", d_A=3, d_B=3, overlap=1, w=[0.7, -0.5, 0.8, 0.6, -0.6], sigma=0.5)
LABELS = {"A": ["x1", "x2", "x3"], "B": ["x3", "x4", "x5"]}


def make_api(T=6, budget=50, seed=0):
    env = build_env(ENV, np.random.default_rng(seed))
    rng = np.random.default_rng(seed + 1)
    xA, _, yA = env.sample(budget, rng)
    _, xB, yB = env.sample(budget, rng)
    qA, qB, y = env.sample(T, rng)
    api = DataGeneratorAPI(
        {"A": (xA, yA), "B": (xB, yB)}, {"A": qA, "B": qB}, y, LABELS, budget=budget
    )
    return api


def policy(talk: bool):
    """A scripted agent: pull all training cases, regress once, then per query: opening = sum
    of features, (send a message), read, final = own opening + partner opening."""
    state = {}
    lock = threading.Lock()

    def fn(system, user, schema):
        if "summary" in schema["properties"]:
            return {"summary": "S: add the partner's opening."}
        name = re.search(r"You are agent (\w)", system).group(1)
        with lock:
            st = state.setdefault(name, {"regressed": False})
        base = {"n": 25, "predictors": [], "y": 0.0, "text": ""}
        if "Unusable" in user:
            raise AssertionError("unusable reply")
        if "Query" in user and "Your features" in user:
            xs = [float(v) for v in re.findall(r"=(-?\d+\.\d+)", user.split("Your features:")[1])]
            st["open"] = sum(xs)
            return {**base, "action": "submit_opening", "y": st["open"]}
        if "recorded" in user:
            return {**base, "action": "send_message" if talk else "read_messages", "text": "hi"}
        if "Message sent." in user:
            return {**base, "action": "read_messages"}
        if "Partner's opening" in user:
            m = re.search(r"Partner's opening for query \d+: (-?[\d.]+)\.", user)
            po = float(m.group(1)) if m else 0.0
            return {**base, "action": "submit_final", "y": st["open"] + po}
        if "training phase" in user or "Training cases" in user:
            if "of 50" in user and re.search(r"Training cases \d+-50 of 50", user):
                return {**base, "action": "regress", "predictors": LABELS[name]}
            return {**base, "action": "next_training_cases", "n": 25}
        if "Regression of y" in user or "result" in user or "continues" in user:
            return {**base, "action": "next_query"}
        return {**base, "action": "next_query"}

    return fn


def twins_for(api, backend, T, compact_tokens=None):
    return {
        k: AgentTwin(k, api, backend, LABELS[k], LABELS["B" if k == "A" else "A"], T, 50,
                     compact_tokens=compact_tokens)
        for k in "AB"
    }


def test_full_episode_with_talk():
    T = 6
    api = make_api(T)
    twins = twins_for(api, StubBackend(policy(talk=True)), T)
    run_episode(api, twins)
    for k in "AB":
        assert api.s[k].n_train_seen == 50
        assert not np.isnan(api.final[k]).any()
        assert not api.forced[k].any()
    # final = own opening + partner opening; both twins messaged every query.
    assert np.allclose(api.final["A"], api.opening["A"] + api.opening["B"])
    assert len(api.messages) == 2 * T
    assert twins["A"].unusable == 0


def test_both_only_listen_does_not_deadlock():
    T = 4
    api = make_api(T)
    twins = twins_for(api, StubBackend(policy(talk=False)), T)
    th = threading.Thread(target=run_episode, args=(api, twins))
    th.start()
    th.join(timeout=20)
    assert not th.is_alive(), "deadlock"
    assert not np.isnan(api.final["B"]).any() and len(api.messages) == 0


def test_compaction_and_regression_columns():
    T = 6
    api = make_api(T)
    twins = twins_for(api, StubBackend(policy(talk=True)), T, compact_tokens=800)
    run_episode(api, twins)
    tw = twins["A"]
    assert tw.compactions and tw.compactions[0]["summary"].startswith("S:")
    first_after = next(t["text"] for t in tw.turns if t["conversation"] == 1 and t["role"] == "user")
    assert first_after.startswith("This conversation continues")
    # A regression on the openings runs over past queries, with a prediction for the current one.
    res = api.regress("A", ["own_opening", "partner_opening"])
    assert "over 6 cases with known y" in res


def test_opening_required_before_talk():
    api = make_api(3)
    api.s["A"].phase = api.s["B"].phase = "query"
    api.s["A"].q = api.s["B"].q = 0
    assert api.send_message("A", "x").startswith("Error")
    assert api.read_messages("A").startswith("Error")


def test_live_files_written_during_run(tmp_path):
    T = 5
    api = make_api(T)
    seen = []
    api.on_query = seen.append
    backend = StubBackend(policy(talk=True))
    twins = {
        k: AgentTwin(k, api, backend, LABELS[k], LABELS["B" if k == "A" else "A"], T, 50,
                     compact_tokens=None, live_dir=tmp_path, run=7)
        for k in "AB"
    }
    run_episode(api, twins)
    assert [r["t"] for r in seen] == list(range(T))  # one record per query, in order
    acts = (tmp_path / "run7_actions_A.jsonl").read_text().splitlines()
    assert len(acts) == len(twins["A"].log)
    import json
    st = json.loads((tmp_path / "run7_status_B.json").read_text())
    assert st["state"] == "finished" and st["query"] == T
