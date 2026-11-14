"""Tests for the consensus data API and the session tools (scripted policies, no model)."""

import asyncio
import threading

import numpy as np

from consensus.data import build_env
from consensus.sessions import ConsensusAPI, build_tools, intro_text

ENV = dict(cov="overlap", d_A=3, d_B=3, overlap=1, w=[0.7, -0.5, 0.8, 0.6, -0.6], sigma=0.5)
LABELS = {"A": ["x1", "x2", "x3"], "B": ["x3", "x4", "x5"]}


def make_api(T=5, budget=50, seed=0, **kw):
    env = build_env(ENV, np.random.default_rng(seed))
    rng = np.random.default_rng(seed + 1)
    xA, _, yA = env.sample(budget, rng)
    _, xB, yB = env.sample(budget, rng)
    qA, qB, y = env.sample(T, rng)
    return ConsensusAPI({"A": (xA, yA), "B": (xB, yB)}, {"A": qA, "B": qB}, y, LABELS,
                        budget=budget, **kw)


def scripted(api, k, intro, agree=True, talk=True):
    """Train on everything, then per query: opening = sum of own features, (message), read,
    submit; with agree=False the twins first disagree once, then both submit A's opening."""
    while not api.get_training_data(k, 25).startswith("You have used"):
        pass
    assert "Regression of y" in api.regress(k, LABELS[k])
    assert "partner" in api.start_queries(k, intro)
    while True:
        res = api.next_query(k)
        if "run is over" in res:
            break
        q = api.q[k]
        api.submit_opening(k, float(api.qx[k][q].sum()))
        if talk:
            api.send_message(k, f"{k} says hi")
        api.read_messages(k)
        target = float(api.opening["A"][q])
        if not agree:
            first = api.submit_prediction(k, target + (0.5 if k == "B" else 0.0))
            assert first.startswith("No consensus")
        res = api.submit_prediction(k, target)
        assert res.startswith("Consensus reached")
        if q == api.T - 1:
            assert "run is over" in res
            break
    api.finish(k)


def run_scripted(api, **kw):
    ths = [threading.Thread(target=scripted, args=(api, k, intro_text(api, k, True, True)),
                            kwargs=kw) for k in "AB"]
    for th in ths:
        th.start()
    for th in ths:
        th.join(timeout=20)
    assert not any(th.is_alive() for th in ths), "deadlock"


def test_consensus_episode():
    api = make_api()
    recs = []
    api.on_query = recs.append
    run_scripted(api)
    assert api.resolved.all() and not api.failed.any()
    assert np.allclose(api.agreed, api.opening["A"])
    assert [r["t"] for r in recs] == list(range(api.T))
    assert len(api.messages) == 2 * api.T


def test_disagreement_then_agreement_and_silent_twins():
    api = make_api(T=3)
    run_scripted(api, agree=False, talk=False)  # nobody talks: read_messages must not deadlock
    assert api.resolved.all() and all(len(a) == 2 for a in api.attempts["A"])
    assert len(api.messages) == 0


def test_failure_after_max_attempts():
    api = make_api(T=1, max_attempts=2)
    for k in "AB":
        api.phase[k] = "query"
    out = {}

    def twin(k):
        api.next_query(k)
        api.submit_opening(k, 0.0)
        out[k] = [api.submit_prediction(k, 1.0 if k == "A" else 2.0) for _ in range(2)]

    ths = [threading.Thread(target=twin, args=(k,)) for k in "AB"]
    [th.start() for th in ths]
    [th.join(timeout=10) for th in ths]
    assert api.failed[0] and "failed (penalty 10)" in out["A"][1]


def test_tools_handlers_and_intro():
    api = make_api()
    intro = intro_text(api, "A", True, True)
    assert "you both see x3" in intro and "statistically independent" in intro
    assert "{n_train}" in intro
    server, allowed, tools = build_tools(api, "A", intro)
    assert allowed[0] == "mcp__data__get_training_data" and len(allowed) == 8
    by = {t.name: t for t in tools}
    res = asyncio.run(by["get_training_data"].handler({"n": 10}))
    assert res["content"][0]["text"].startswith("Training cases 1-10 of 50")
    res = asyncio.run(by["start_queries"].handler({}))
    assert "(10 training cases studied)" in res["content"][0]["text"]
    # Before next_query the query tools refuse politely.
    res = asyncio.run(by["submit_opening"].handler({"y": 1.0}))
    assert res["content"][0]["text"].startswith("Error")


def test_weight_of_advice_in_records():
    from consensus.sessions import weight_of_advice

    assert weight_of_advice(0.0, 2.0, 1.0) == 0.5
    assert weight_of_advice(0.0, 2.0, 4.0) == 1.5  # clipped
    assert np.isnan(weight_of_advice(0.0, 0.1, 1.0))  # openings too close
    api = make_api(T=2)
    recs = []
    api.on_query = recs.append
    run_scripted(api)  # both agree on A's opening: A keeps its opening, B adopts A's
    for r in recs:
        if not np.isnan(r["woa"]["A"]):
            assert r["woa"]["A"] == 0.0 and r["woa"]["B"] == 1.0
            assert r["woa_first"] == r["woa"]


def test_per_twin_budget_and_study_all():
    api = make_api(T=2, budget=50)  # data for 50 cases each
    api2 = ConsensusAPI(
        {"A": (api.Xtr["A"], api.ytr["A"]), "B": (api.Xtr["B"], api.ytr["B"])},
        api.qx, api.y, LABELS, budget={"A": 50, "B": 25}, study_all=True,
    )
    assert api2.budget == {"A": 50, "B": 25} and len(api2.Xtr["B"]) == 25
    api2.get_training_data("A", 25)
    assert api2.start_queries("A", "intro {n_train}").startswith("Error: study all 50")
    api2.get_training_data("A", 25)
    assert api2.start_queries("A", "intro {n_train}") == "intro 50"
    assert api2.get_training_data("B", 25).startswith("Training cases 1-25 of 25")
    assert api2.get_training_data("B", 25).startswith("You have used all 25")
