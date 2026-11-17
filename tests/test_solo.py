"""Solo queries first (LogBook 2026-11-16): each twin answers K queries alone after training,
then the partner phase starts with the usual start_queries text."""

import threading

import numpy as np
import pytest

from consensus.data import build_env
from consensus.sessions import ConsensusAPI, TwinSession

BASE = dict(d_A=3, d_B=3, cov="controlled", lambda_xz=0.0, rotate=True, align=1.0,
            normalize=False, alpha=1.0, weak="B", sigma=0.5, task="related", target_align=0.5)
LABELS = {"A": ["a1", "a2", "a3"], "B": ["b1", "b2", "b3"]}


def make_api(K=2, T=2, budget=30):
    env = build_env(BASE, np.random.default_rng(0))
    rng = np.random.default_rng(1)
    tA, _, tyA, _ = env.sample_pair(budget, rng)
    _, tB, _, tyB = env.sample_pair(budget, rng)
    qA, qB, yA, yB = env.sample_pair(T, rng)
    sA, sB, syA, syB = env.sample_pair(K, np.random.default_rng(2))
    solo = {"x": {"A": sA, "B": sB}, "y": {"A": syA, "B": syB}} if K else None
    api = ConsensusAPI({"A": (tA, tyA), "B": (tB, tyB)}, {"A": qA, "B": qB},
                       {"A": yA, "B": yB}, LABELS, budget=budget, team_rule="individual",
                       models_in_training=True, history_tools=True, cap=40, page_size=budget,
                       solo=solo)
    return api, (syA, syB, yA, yB)


def _session(api, k="A"):
    return TwinSession(k, api, structure_hint=False, partner_info=True, same_y=True,
                       model_tools=True, prompt_style="clear_history", partner_target="told",
                       query_api=True)


def test_solo_phase_then_partner_phase():
    api, (syA, syB, yA, yB) = make_api()
    intro = {k: _session(api, k).intro for k in "AB"}
    out = {k: [] for k in "AB"}

    def twin(k):
        api.get_training_data(k, 30)
        out[k].append(api.start_queries(k, intro[k]))
        api.new_model(k, "m", LABELS[k])
        # Solo: no opening, no messages; a prediction reveals the twin's own y.
        out[k].append(api.next_query(k))
        out[k].append(api.submit_opening(k, 0.0))
        out[k].append(api.send_message(k, "hi"))
        out[k].append(api.next_query(k))  # refused: predict first
        api.add_case(k, "m", {})
        out[k].append(api.fit_model(k, "m", {}))
        out[k].append(api.submit_prediction(k, 0.5))
        api.next_query(k)
        out[k].append(api.submit_prediction(k, 0.5))  # the last solo case: partner text follows
        # Partner phase, as without solo.
        while "run is over" not in api.next_query(k):
            q = api.q[k]
            api.submit_opening(k, 1.0)
            api.submit_prediction(k, 1.0)
            if q == api.T - 1:
                break
        api.finish(k)

    ths = [threading.Thread(target=twin, args=(k,)) for k in "AB"]
    [th.start() for th in ths]
    [th.join(timeout=20) for th in ths]
    assert api.resolved.all()
    a = out["A"]
    assert "Next you will answer 2 queries on your own" in a[0] and "partner" not in a[0]
    assert a[1].startswith("Query 1 of 2. Your features: a1=")
    assert a[2].startswith("Error: while you answer queries on your own")
    assert a[3].startswith("Error: while you answer queries on your own")
    assert a[4].startswith("Error: submit your prediction for query 1")
    assert f"true y was {syA[0]:.3f}" in a[6] and "Call next_query" in a[6]
    assert f"true y was {syB[0]:.3f}" in out["B"][-1] or f"{syB[1]:.3f}" in out["B"][-1]
    last = a[7]
    assert "That was the last of your 2 queries on your own." in last
    assert "answering 2 queries on your own. From now on you have a partner, agent B" in last
    assert "also studied its own cases and answered 2 queries on its own" in last
    assert "There are 2 queries." in last and "{n_train}" not in last
    # Solo cases are known rows (own features and y) and the add_case row got its y.
    rows = api._known_rows("A")
    assert sum("own_opening" not in r for r in rows) == 30 + 2
    assert [r["y"] for r in api.models["A"]["m"]["rows"] if "_solo" in r] == [pytest.approx(syA[0])]
    assert np.allclose(api.solo_pred["A"], 0.5)
    assert api.record(0)["y_own"]["A"] == pytest.approx(yA[0])


def test_no_solo_is_unchanged():
    api, _ = make_api(K=0)
    assert api.K == 0
    intro = _session(api).intro
    assert "on your own" not in intro and "You have finished studying ({n_train}" in intro
    api.get_training_data("A", 30)
    assert api.start_queries("A", intro).startswith("You have finished studying (30")
    assert api.phase["A"] == "query"
