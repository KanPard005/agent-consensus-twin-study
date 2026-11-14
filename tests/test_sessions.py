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


def test_average_team_rule():
    api = make_api(T=3, team_rule="average")
    intro = intro_text(api, "A", False, True)
    assert "average of your two final estimates" in intro and "consensus" not in intro
    recs = []
    api.on_query = recs.append

    def twin(k):
        while not api.get_training_data(k, 25).startswith("You have used"):
            pass
        api.start_queries(k, intro)
        while "run is over" not in api.next_query(k):
            q = api.q[k]
            api.submit_opening(k, 1.0 if k == "A" else 3.0)
            api.read_messages(k)
            res = api.submit_prediction(k, 1.5 if k == "A" else 3.0)  # A moves a quarter, B holds
            assert res.startswith("Both finals are in")
            assert api.submit_prediction(k, 0.0).startswith("Error")
            if q == api.T - 1:
                break
        api.finish(k)

    ths = [threading.Thread(target=twin, args=(k,)) for k in "AB"]
    [th.start() for th in ths]
    [th.join(timeout=20) for th in ths]
    assert api.resolved.all() and np.allclose(api.agreed, 2.25)
    assert recs[0]["woa"] == {"A": 0.25, "B": 0.0}


def test_individual_scoring_rule():
    api = make_api(T=2, team_rule="individual")
    intro = intro_text(api, "A", False, True)
    assert "squared error of your own final" in intro and "average" not in intro
    out = {}

    def twin(k):
        while not api.get_training_data(k, 25).startswith("You have used"):
            pass
        api.start_queries(k, intro)
        while "run is over" not in api.next_query(k):
            q = api.q[k]
            api.submit_opening(k, 1.0 if k == "A" else 3.0)
            out.setdefault(k, []).append(api.submit_prediction(k, 2.0))
            if q == api.T - 1:
                break
        api.finish(k)

    ths = [threading.Thread(target=twin, args=(k,)) for k in "AB"]
    [th.start() for th in ths]
    [th.join(timeout=20) for th in ths]
    assert api.resolved.all() and "your squared error" in out["A"][0]
    assert api.record(0)["woa"] == {"A": 0.5, "B": 0.5}


def test_same_y_and_model_instances():
    api = make_api(T=2, team_rule="individual")
    intro = intro_text(api, "A", False, True, same_y=True, model_tools=True)
    assert "predict the same quantity y" in intro and "new_model" in intro
    assert "partner's features" not in intro.split("new_model")[1]  # no hint on what to use
    server, allowed, tools = build_tools(api, "A", intro, model_tools=True)
    assert "mcp__data__new_model" in allowed and len(allowed) == 11
    for k in "AB":
        api.phase[k] = "query"
    seen = {}

    def twin(k):
        api.next_query(k)
        api.submit_opening(k, 1.0 if k == "A" else 2.0)
        if k == "A":
            seen["new"] = api.new_model("A", "both", ["x1", "partner_opening", "b1"])
            seen["missing"] = api.add_case("A", "both", {})
            api.read_messages("A")
            seen["add"] = api.add_case("A", "both", {"b1": 0.3})
        else:
            api.send_message("B", "b1 is 0.3")
        api.submit_prediction(k, 1.5)
        api.next_query(k)  # query 2: A's model now has one case with y

    ths = [threading.Thread(target=twin, args=(k,)) for k in "AB"]
    [th.start() for th in ths]
    [th.join(timeout=20) for th in ths]
    assert "no data yet" in seen["new"] and "Filled in automatically: x1, partner_opening" in seen["new"]
    assert seen["missing"].startswith("Error") and "b1" in seen["missing"]
    m = api.models["A"]["both"]
    assert len(m["rows"]) == 1 and m["rows"][0]["b1"] == 0.3 and m["rows"][0]["y"] == api.y[0]
    assert m["rows"][0]["partner_opening"] == 2.0
    assert api.record(0)["models"]["A"]["both"]["n_rows"] == 1


def test_models_in_training_with_expressions():
    from consensus.sessions import TwinSession, eval_expr

    assert eval_expr("x1*x2 + x3^2", {"x1": 2, "x2": 3, "x3": 1}) == 7.0
    assert eval_expr("b1", {"x1": 1.0}) is None and eval_expr("__import__('os')", {}) is None
    api = make_api(T=2, team_rule="individual", models_in_training=True)
    # Training: a model with expression columns, filled with the training cases seen so far.
    api.get_training_data("A", 25)
    res = api.new_model("A", "quad", ["x1", "x1*x2", "x3^2"])
    assert "25 training cases" in res and "Filled in automatically: x1, x1*x2, x3^2" in res
    api.get_training_data("A", 25)  # later pulls are added too
    m = api.models["A"]["quad"]
    assert len(m["rows"]) == 50
    assert np.isclose(m["rows"][0]["x1*x2"], api.Xtr["A"][0, 0] * api.Xtr["A"][0, 1])
    assert "Regression of y on x1, x1*x2, x3^2 over 50" in api.fit_model("A", "quad", {})
    bad = api.new_model("A", "p", ["x1", "b1"])
    assert bad.startswith("Error: include_training needs") and "b1" in bad
    # Prompts: the training prompt names the twin's own features as examples.
    tw = TwinSession("B", api, model_tools=True, same_y=True)
    assert "x3*x4" in tw.system and "include_training" in tw.system
    assert "stay available" in tw.intro


def test_clear_prompt_style():
    """`prompt_style: clear` (LogBook 2026-11-14): no "training phase" anywhere the twin reads,
    both APIs listed, and the start_queries text names the three freedoms."""
    from consensus.sessions import TwinSession, _tool_docs

    api = make_api(budget=50, team_rule="individual", models_in_training=True)
    t = TwinSession("A", api, structure_hint=False, partner_info=True, same_y=True,
                    model_tools=True, prompt_style="clear")
    intro = t.intro.format(n_train=50)
    texts = [t.system, intro, *_tool_docs("clear").values()]
    assert not any("training phase" in x for x in texts)
    assert "Data API:" in t.system and "Regression API:" in t.system
    for s in ("train new models", "such as your features or your predictions",
              "make use of your partner's features in any way you want",
              "you both predict the same quantity y"):
        assert s in intro
    # The standard style is unchanged.
    assert "The training phase is over" in intro_text(api, "A", False, True, True, True)


def test_clear_models_prompt_style():
    """`clear_models` adds only the line about building models on any features."""
    from consensus.sessions import TwinSession

    api = make_api(budget=50, team_rule="individual", models_in_training=True)
    kw = dict(structure_hint=False, partner_info=True, same_y=True, model_tools=True)
    a = TwinSession("A", api, prompt_style="clear", **kw)
    b = TwinSession("A", api, prompt_style="clear_models", **kw)
    assert a.system == b.system
    extra = b.intro.replace(a.intro.split("You can make at most")[0], "")
    assert "build a new model using any features you see" in b.intro
    assert "build a new model" not in a.intro and extra


def test_history_tools_past_queries_rows_and_log():
    """`history_tools` (LogBook 2026-11-14): add_case with a past query number (own features and
    openings filled in, partner columns given, y at once, re-adding replaces), add_row with
    values and y given, and the per-twin log."""
    api = make_api(T=5, budget=50, team_rule="individual", models_in_training=True,
                   history_tools=True)
    api.phase["A"], api.q["A"] = "query", 3
    api.resolved[:3] = True
    api.opening["A"][:4] = [0.1, 0.2, 0.3, 0.4]
    api.opening["B"][:4] = [1.0, 2.0, 3.0, 4.0]
    assert "created" in api.new_model("A", "joint", ["x1", "x4", "partner_opening"], False)
    assert "Error" in api.add_case("A", "joint", {}, query=2)  # x4 not given
    assert "Error" in api.add_case("A", "joint", {"x4": 1.0}, query=4)  # not resolved yet
    assert "with its y" in api.add_case("A", "joint", {"x4": 0.5}, query=2)
    assert "with its y" in api.add_case("A", "joint", {"x4": 0.7}, query=2)  # replaces
    rows = api.models["A"]["joint"]["rows"]
    assert len(rows) == 1
    r = rows[0]
    assert r["x4"] == 0.7 and r["partner_opening"] == 2.0
    assert r["x1"] == float(api.qx["A"][1][0]) and r["y"] == float(api.y[1])
    # add_row: every column and y given by the twin; expressions computed from given values
    api.new_model("A", "sq", ["x1", "x1^2"], False)
    assert "Error" in api.add_row("A", "sq", {}, 1.0)
    assert "Row added" in api.add_row("A", "sq", {"x1": 2.0}, 1.5)
    assert api.models["A"]["sq"]["rows"][0] == {"x1": 2.0, "x1^2": 4.0, "y": 1.5}
    # the log
    assert "empty" in api.read_log("A")
    api.write_log("A", "q2: b = (0.7, -1, 2), y = 1.3")
    assert "q2: b = (0.7, -1, 2)" in api.read_log("A") and "empty" in api.read_log("B")
    # without the flag a `query` argument is ignored (the standard current-case path)
    api2 = make_api(T=5, budget=50, team_rule="individual", models_in_training=True)
    api2.phase["A"], api2.q["A"] = "query", 3
    api2.new_model("A", "m", ["x1"], False)
    assert "will be added" in api2.add_case("A", "m", {}, query=2)


def test_clear_history_prompt_style():
    from consensus.sessions import TwinSession

    kw = dict(structure_hint=False, partner_info=True, same_y=True, model_tools=True)
    api = make_api(budget=50, team_rule="individual", models_in_training=True,
                   history_tools=True)
    t = TwinSession("A", api, prompt_style="clear_history", **kw)
    assert "Log API:" in t.system and "add_row(name, values, y)" in t.system
    for s in ("send the features and labels of past queries to any regression model",
              "log the features (your own and your partner's)", "You can do this at any step",
              "construct your own from past queries", "build a new model using any features"):
        assert s in t.intro
    try:
        TwinSession("A", make_api(budget=50, team_rule="individual", models_in_training=True),
                    prompt_style="clear_history", **kw)
        raise AssertionError("needs history_tools")
    except ValueError:
        pass
