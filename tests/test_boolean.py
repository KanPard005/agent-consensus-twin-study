"""Tests for the boolean task: copula calibration, Bayes references, the data API and its
partner-phase waiting logic (scripted twins, no model)."""

import threading

import numpy as np
import pytest

from consensus.boolean_data import BooleanCopulaEnv, log_loss
from consensus.boolean_sessions import (
    BooleanAPI,
    ConditionError,
    check_condition,
    eval_condition,
    prompts,
)

CLAUSES = [[0, 1, 2], [3, 4, 5]]
VIEWS = {"A": [0, 1, 2], "B": [3, 4, 5]}
LABELS = {"A": ["x1", "x2", "x3"], "B": ["x4", "x5", "x6"]}


def make_env(**kw):
    return BooleanCopulaEnv(clauses=CLAUSES, views=VIEWS, n_mc=200_000, **kw)


@pytest.mark.parametrize("r_within,r_cross", [(0.5, 0.0), (0.8, 0.3), (0.0, 0.0)])
def test_clause_probability_is_calibrated(r_within, r_cross):
    env = make_env(r_within=r_within, r_cross=r_cross, p_clause=0.6)
    X, _, _ = env.sample(200_000, np.random.default_rng(1))
    assert np.allclose(env.clause_values(X).mean(axis=0), 0.6, atol=0.006)


def test_references_match_hand_calculation():
    # Independent clauses: solo accuracy = P(D=0) rho + P(D=1) max(q, 1-q),
    # q = rho p + (1 - rho)(1 - p) with p = P(D_other = 1).
    p, rho = np.sqrt(0.5), 0.9
    env = make_env(p_clause=p, rho=rho)
    ref = env.references(m=200_000)
    q = rho * p + (1 - rho) * (1 - p)
    assert ref["acc_A"] == pytest.approx((1 - p) * rho + p * max(q, 1 - q), abs=0.005)
    assert ref["acc_joint"] == pytest.approx(rho, abs=0.005)
    assert ref["p_pi"] == pytest.approx(0.5, abs=0.005)


def test_label_noise_rate():
    env = make_env(rho=0.8)
    _, pi, y = env.sample(100_000, np.random.default_rng(2))
    assert np.mean(pi == y) == pytest.approx(0.8, abs=0.005)


def test_not_positive_definite_raises():
    with pytest.raises(ValueError):
        make_env(r_within=0.0, r_cross=0.9)


def test_log_loss_clips():
    assert log_loss(np.array([0.0]), np.array([1]))[0] == pytest.approx(-np.log(1e-3))


def test_conditions():
    row = {"x1": 1, "x2": 0, "b": 1}
    for expr, want in [("x1 == 1 and not (x2 == 1)", True), ("x2 or b", True),
                       ("x1 != b", False), ("", True), ("0 < x1 <= 1", True)]:
        assert eval_condition(check_condition(expr), row) is want
    for bad in ["__import__('os')", "x1.real", "'a' == x1"]:
        with pytest.raises(ConditionError):
            check_condition(bad)
    with pytest.raises(KeyError):
        eval_condition(check_condition("x9 == 1"), row)


def make_api(K=2, T=3, n=6, seed=0):
    env = make_env()
    rng = np.random.default_rng(seed)
    Xtr, _, ytr = env.sample(n, rng)
    Xs, _, ys = env.sample(K, rng)
    Xq, _, yq = env.sample(T, rng)
    api = BooleanAPI({k: (env.view(Xtr, k), ytr) for k in "AB"},
                     {"x": {k: env.view(Xs, k) for k in "AB"}, "y": ys},
                     {k: env.view(Xq, k) for k in "AB"}, yq, LABELS, page_size=4, cap=30)
    for k in "AB":
        pr = prompts(api, k)
        api.solo_intro[k], api.partner_intro[k] = pr["solo"], pr["partner"]
    return api


def scripted(api, k, out):
    """Study everything, answer the solo queries, then per partner query: open, send own
    features, read, submit the AND of both ORs."""
    assert api.start_queries(k).startswith("Error: study all")
    while not api.get_training_data(k, 4).startswith("You have seen all"):
        pass
    assert "rows with y where" in api.count(k, "training", f"{LABELS[k][0]} == 1")
    api.start_queries(k)
    for _ in range(api.K):
        assert "on your own" in api.next_query(k)
        assert "added" in api.add_case(k, "training")
        res = api.submit_prediction(k, 0.5, 1)
    assert "partner, agent" in res
    assert "Dataset joint created with 2 rows" in api.new_dataset(k, "joint", False, [1, 2])
    o = api.other(k)
    while True:
        r = api.next_query(k)
        if "run is over" in r:
            break
        q = api.q[k]
        own = int(api.qx[k][q].max())
        assert "recorded" in api.submit_opening(k, 0.5, own)
        api.send_message(k, f"D={own}")
        txt = api.read_messages(k)
        while "D=" not in txt:
            txt = api.read_messages(k)
        other = int(txt.split("D=")[1][0])
        assert "added" in api.add_case(k, "joint", values={"d_" + o: other})
        res = api.submit_prediction(k, 0.9 if own and other else 0.1, own * other)
        assert "Both finals are in" in res
        if "run is over" in res:
            break
    out[k] = True


def test_scripted_pair_runs_and_reveals():
    api = make_api()
    out = {}
    th = [threading.Thread(target=scripted, args=(api, k, out)) for k in "AB"]
    for t in th:
        t.start()
    for t in th:
        t.join(timeout=20)
    assert out == {"A": True, "B": True}
    assert api.resolved.all()
    # The scripted twins combine the clause values, so their finals equal pi on every query.
    rec = [api.record(q) for q in range(api.T)]
    for k in "AB":
        d = api.datasets[k]
        assert len(d["joint"]["rows"]) == 2 + api.T and not d["joint"]["pending"]
        assert len(d["training"]["rows"]) == 6 + api.K
        assert "d_" + api.other(k) in api.show_dataset(k, "joint")
    for q, r in enumerate(rec):
        xa, xb = r["x"]["A"], r["x"]["B"]
        assert r["final"]["A"][1] == float(max(xa) * max(xb))


def test_past_query_rules():
    api = make_api()
    k = "A"
    while not api.get_training_data(k, 4).startswith("You have seen all"):
        pass
    api.start_queries(k)
    assert "not a past query" in api.add_case(k, "training", query=1)
    api.next_query(k)
    assert "not a past query" in api.add_case(k, "training", query=1)
    api.submit_prediction(k, 0.3, 0)
    assert "added to training" in api.add_case(k, "training", query=1)
    assert "cannot be given" in api.add_case(k, "training", values={"x1": 1})
    assert "between 0 and 1" in api.submit_prediction(k, 1.5, 1)
