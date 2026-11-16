"""Related targets (LogBook 2026-11-16): each twin predicts its own target, with cosine c
between the two targets' weights on each feature block."""

import threading

import numpy as np
import pytest

from consensus.data import RelatedTargetsEnv, build_env, tilted_direction
from consensus.sessions import ConsensusAPI, TwinSession

BASE = dict(d_A=3, d_B=3, cov="controlled", lambda_xz=0.0, rotate=True, align=1.0,
            normalize=False, alpha=1.0, weak="B", sigma=0.5)
LABELS = {"A": ["a1", "a2", "a3"], "B": ["b1", "b2", "b3"]}


@pytest.mark.parametrize("c", [1.0, 0.5, 0.0, -0.5])
def test_tilted_weights_have_cosine_c_and_same_norm(c):
    env = build_env({**BASE, "task": "related", "target_align": c}, np.random.default_rng(0))
    assert isinstance(env, RelatedTargetsEnv)
    for w, v in ((env.w_A, env.v_A), (env.w_B, env.v_B)):
        assert np.isclose(np.linalg.norm(v), np.linalg.norm(w))
        assert np.isclose(w @ v / (w @ w), c)


def test_c_one_reproduces_the_base_task_and_noise_is_shared():
    base = build_env(BASE, np.random.default_rng(0))
    rel = build_env({**BASE, "task": "related", "target_align": 1.0}, np.random.default_rng(0))
    xA, xB, y = base.sample(50, np.random.default_rng(1))
    rA, rB, yA, yB = rel.sample_pair(50, np.random.default_rng(1))
    assert np.allclose(xA, rA) and np.allclose(xB, rB) and np.allclose(y, yA)
    assert np.allclose(yA, yB)
    # At c = 0 the targets differ only by the signal: y_A - y_B has no noise term.
    rel0 = build_env({**BASE, "task": "related", "target_align": 0.0}, np.random.default_rng(0))
    xA, xB, yA, yB = rel0.sample_pair(50, np.random.default_rng(1))
    diff = xA @ (rel0.w_A - rel0.v_A) + xB @ (rel0.w_B - rel0.v_B)
    assert np.allclose(yA - yB, diff)


def test_population_value_of_the_partner_opening():
    """With Sigma = I, regressing y_A on (own opening, partner opening) gives coefficients
    (1, c) and MSE 1 - c^2 + sigma^2."""
    c = -0.5
    env = build_env({**BASE, "task": "related", "target_align": c}, np.random.default_rng(0))
    xA, xB, yA, _ = env.sample_pair(200_000, np.random.default_rng(2))
    X = np.column_stack([xA @ env.w_A, xB @ env.v_B])
    beta, *_ = np.linalg.lstsq(X, yA, rcond=None)
    assert np.allclose(beta, [1.0, c], atol=0.02)
    assert np.isclose(np.mean((yA - X @ beta) ** 2), 1 - c**2 + 0.25, atol=0.02)


def test_tilted_keeps_norm_of_zero_vector():
    v = tilted_direction(np.zeros(3), 0.0, np.random.default_rng(0))
    assert np.isclose(np.linalg.norm(v), 0.0)


def make_api(c=0.0, T=3, budget=50):
    env = build_env({**BASE, "task": "related", "target_align": c}, np.random.default_rng(0))
    rng = np.random.default_rng(1)
    tA, _, tyA, _ = env.sample_pair(budget, rng)
    _, tB, _, tyB = env.sample_pair(budget, rng)
    qA, qB, yA, yB = env.sample_pair(T, rng)
    api = ConsensusAPI({"A": (tA, tyA), "B": (tB, tyB)}, {"A": qA, "B": qB},
                       {"A": yA, "B": yB}, LABELS, budget=budget, team_rule="individual",
                       models_in_training=True, history_tools=True, cap=40)
    return api, (tyA, tyB, yA, yB)


def test_api_uses_each_twins_own_target():
    api, (tyA, tyB, yA, yB) = make_api()
    assert np.allclose(api.ytr["A"], tyA) and np.allclose(api.ytr["B"], tyB)
    out = {}

    def twin(k, intro="intro {n_train}"):
        while not api.get_training_data(k, 25).startswith("You have used"):
            pass
        api.start_queries(k, intro)
        api.new_model(k, "m", LABELS[k])
        while "run is over" not in api.next_query(k):
            q = api.q[k]
            api.submit_opening(k, 1.0 if k == "A" else 3.0)
            api.add_case(k, "m", {})
            out.setdefault(k, []).append(api.submit_prediction(k, 2.0))
            if q == api.T - 1:
                break
        api.finish(k)

    ths = [threading.Thread(target=twin, args=(k,)) for k in "AB"]
    [th.start() for th in ths]
    [th.join(timeout=20) for th in ths]
    assert api.resolved.all()
    # Each twin is told its own y and its own error only.
    assert f"true y was {yA[0]:.3f}" in out["A"][0] and f"true y was {yB[0]:.3f}" in out["B"][0]
    assert "your partner's" in out["A"][0] and out["A"][0].count("squared error") == 1
    # Model rows and regression rows carry the twin's own y.
    assert [r["y"] for r in api.models["B"]["m"]["rows"]] == pytest.approx(list(yB))
    assert api._known_rows("A")[-1]["y"] == pytest.approx(yA[-1])
    rec = api.record(0)
    assert rec["y_own"] == {"A": pytest.approx(yA[0]), "B": pytest.approx(yB[0])}


def test_related_needs_individual_rule():
    env = build_env({**BASE, "task": "related", "target_align": 0.5}, np.random.default_rng(0))
    xA, xB, yA, yB = env.sample_pair(4, np.random.default_rng(1))
    with pytest.raises(ValueError):
        ConsensusAPI({"A": (xA, yA), "B": (xB, yB)}, {"A": xA, "B": xB}, {"A": yA, "B": yB},
                     LABELS, budget=4, team_rule="average")


def _session(api, partner_target, query_api=True):
    return TwinSession("A", api, structure_hint=False, partner_info=True, same_y=True,
                       model_tools=True, prompt_style="clear_history",
                       partner_target=partner_target, query_api=query_api)


def test_prompt_arms_and_query_api():
    api, _ = make_api()
    untold, told = _session(api, "untold").intro, _session(api, "told").intro
    for t in (untold, told):
        assert "the same quantity y" not in t
        assert "both finals are revealed to both of you, and the true y to you" in t
        assert "Query API:\n- next_query():" in t and "- submit_prediction(y):" in t
        # The query API comes after the per-query steps and before the freedoms.
        assert t.index("For each one:") < t.index("Query API:") < t.index("At any time")
        assert "own_error" not in t
    assert "its own quantity" not in untold and "never see its value" not in untold
    assert "Your partner also predicts its own quantity for each case, which is different " \
           "from your y but related to it; you never see its value." in told
    assert "final estimate of its own quantity" in told
    with pytest.raises(ValueError):
        _session(api, "same")
    assert "Query API" not in _session(api, "told", query_api=False).intro
