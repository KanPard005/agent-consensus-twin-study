"""LLM twin: a language model that is handed a trained linear learner as its prediction tool.

Each step the twin sees its own features x and its linear model's prediction. It then decides
what to say to its partner, what estimate to give, and what to write in private notes that
persist across steps. How it uses the linear prediction, the partner's messages and the feedback
is up to the model; nothing here combines numbers for it.

Both twins get the same system prompt (apart from their name) and the same kind of linear
learner, trained on their own feature block, as in the linear twin study.

Options (all off by default, which reproduces the first LLM experiment's prompts exactly):
- `team_rule="joint"`: the twins are asked to propose one joint team estimate each round
  (scored as the average of the two final proposals, so it only differs if they disagree).
- `show_partner_estimates`: the partner's numeric estimates are shown next to its messages.
- `tools`: a least-squares regression tool over the twin's own history (see `regress`).
- Logistic task (the linear learner is a `LogisticAgent`): y is 0/1, the model gives a
  probability, estimates are probabilities that y = 1, the team is scored by log loss and the
  regression tool is a logistic regression (probability columns enter as log-odds).
- `learner="self"` (regression task): no linear model. The twin itself learns from its
  training cases: before the communication phase it is shown its own training samples in
  batches and writes, in its notes, the model or rules it will predict with (`LLMTwin.train`).
  In the communication phase it predicts from its features and those notes alone; its solo
  opinion is its round-1 estimate.
- `memory="persistent"`: one continuing conversation per twin for the whole run instead of a
  fresh call per round. Each round only adds a short user turn (the previous step's outcome,
  the new case, the partner's latest message); there are no notes and no history table, since
  everything earlier is in context.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from consensus.llm_backends import BackendReply
from consensus.models import LinearAgent, LogisticAgent, fit_logistic, log_loss, logit, sigmoid

COLUMN_HELP = "x1..x{d} are your features, own_model is your linear model's prediction"
PARTNER_OPEN_HELP = ", partner_open is the estimate your partner gave in round 1 of the step"

SYSTEM_PROMPT = """\
You are agent {name}, one of two agents (A and B) who predict a number together.

Each step there is a new case with an unknown real-valued target y. You see {d} numeric \
features of the case; your partner sees different features of the same case. {twin_line}

You have a linear model that was fitted to {n_train} earlier cases using only your features. \
Its prediction for the current case is given to you every step. It is a tool: you decide how \
much to rely on it.

{rounds_par}

{memory_par}{tool_par}"""

NOTES_PAR = """\
You keep private notes that persist across steps (at most {notes_chars} characters). Rewrite \
them every round with whatever will help you later, for example how reliable your model and \
your partner are, how to combine your information, or conventions you and your partner use.\
"""

PERSISTENT_PAR = """\
This is one continuing conversation for the whole run: every step, round, message and \
outcome stays in it, so you can look back at anything earlier when deciding what to do.\
"""

ROUNDS_AVERAGE = """\
Each step has {n_rounds} rounds. In every round you and your partner simultaneously send each \
other a message (at most {max_chars} characters) and each give an estimate of y. From round 2 \
on you see what your partner sent in the earlier rounds of this step{shown}. The team's \
prediction is the average of your two final-round estimates, and the team is scored by squared \
error. After the final round the true y is revealed to both of you."""

ROUNDS_JOINT = """\
Each step has {n_rounds} rounds. In every round you and your partner simultaneously send each \
other a message (at most {max_chars} characters) and each propose a joint team estimate of y. \
From round 2 on you see what your partner sent in the earlier rounds of this step{shown}. The \
two of you must submit one joint estimate: the team's prediction is the joint estimate you both \
propose in the final round (if your final proposals differ, their average is used), and the \
team is scored by squared error. After the final round the true y is revealed to both of you."""

TOOL_PAR = """

You also have a regression tool. To use it, put predictor names in `regress` (allowed: \
{columns}; {column_help}). It fits y on them plus an intercept by least squares over all \
past steps, and returns the coefficients, the in-sample and leave-one-out mean squared \
errors, and its prediction for the current case when all predictors are known. A reply with a \
non-empty `regress` is a tool call: its message and estimate are ignored and you are asked \
again with the result. You can make at most {max_tool_calls} tool calls per round. Leave \
`regress` empty to give your answer."""

AUTO_PAR = """ \
Every round the tool is also run once for you automatically, on own_model{auto_partner}, and \
its result is shown before you answer."""

# Logistic task (binary y, probabilities). Same structure as the regression prompts above.
COLUMN_HELP_LOGISTIC = (
    "x1..x{d} are your features, own_model is your logistic model's probability that y = 1"
)
PARTNER_OPEN_HELP_LOGISTIC = (
    ", partner_open is the probability your partner gave in round 1 of the step"
)

SYSTEM_PROMPT_LOGISTIC = """\
You are agent {name}, one of two agents (A and B) who forecast a yes/no outcome together.

Each step there is a new case with an unknown outcome y, which is either 1 or 0. You see {d} \
numeric features of the case; your partner sees different features of the same case. {twin_line}

You have a logistic regression model that was fitted to {n_train} earlier cases using only your \
features. Its probability that y = 1 for the current case is given to you every step. It is a \
tool: you decide how much to rely on it.

{rounds_par}

{memory_par}{tool_par}"""

ROUNDS_AVERAGE_LOGISTIC = """\
Each step has {n_rounds} rounds. In every round you and your partner simultaneously send each \
other a message (at most {max_chars} characters) and each give your probability that y = 1. \
From round 2 on you see what your partner sent in the earlier rounds of this step{shown}. The \
team's forecast is the average of your two final-round probabilities, and the team is scored \
by log loss: minus the natural log of the probability the team gave to the outcome that \
happened (lower is better). After the final round the true y is revealed to both of you."""

ROUNDS_JOINT_LOGISTIC = """\
Each step has {n_rounds} rounds. In every round you and your partner simultaneously send each \
other a message (at most {max_chars} characters) and each propose a joint team probability that \
y = 1. From round 2 on you see what your partner sent in the earlier rounds of this \
step{shown}. The two of you must submit one joint probability: the team's forecast is the joint \
probability you both propose in the final round (if your final proposals differ, their average \
is used), and the team is scored by log loss: minus the natural log of the probability the team \
gave to the outcome that happened (lower is better). After the final round the true y is \
revealed to both of you."""

TOOL_PAR_LOGISTIC = """

You also have a regression tool. To use it, put predictor names in `regress` (allowed: \
{columns}; {column_help}). It fits a logistic regression of y on them plus an intercept over \
all past steps (probability columns enter as log-odds, logit(p) = ln(p / (1 - p))), and returns \
the coefficients, the in-sample and leave-one-out log loss, and its probability that y = 1 for \
the current case when all predictors are known. A reply with a non-empty `regress` is a tool \
call: its message and estimate are ignored and you are asked again with the result. You can \
make at most {max_tool_calls} tool calls per round. Leave `regress` empty to give your answer."""

# Optional hints on how to pool the two probabilities (logistic task, `pooling_hint`). They are
# true only when the two feature blocks are independent (rho = 0).
HINT_INDEPENDENT = """\
Your features and your partner's features are statistically independent of each other, so your \
partner's information is entirely separate evidence about y, not a second opinion on the same \
evidence. Your model's probability reflects only your half of the evidence."""

HINT_LOGODDS = (
    HINT_INDEPENDENT
    + """ For independent evidence, log-odds add: if your model says p_A and your partner's says \
p_B, the combined log-odds are about logit(p_A) + logit(p_B), where logit(p) = ln(p / (1 - p)). \
Because each model was fitted without the other half of the evidence, the best weights on the \
two log-odds can be even larger than 1. Averaging the two probabilities throws information away \
and makes the forecast too unsure."""
)

POOLING_HINTS = {"none": "", "independent": HINT_INDEPENDENT, "logodds": HINT_LOGODDS}

# Self-learning twins (`learner="self"`): the LLM learns from its own training cases itself.
SYSTEM_PROMPT_SELF = """\
You are agent {name}, one of two agents (A and B) who predict a number together.

Each step there is a new case with an unknown real-valued target y. You see {d} numeric \
features of the case; your partner sees different features of the same case. {twin_line}

Before this phase you studied {n_train} earlier cases on your own (your features and the true \
y) and wrote in your notes the model or rules you use to predict y from your features. You \
make your predictions yourself; there is no other model or tool.

{rounds_par}

{memory_par}"""

TWIN_LINE_SELF = (
    "Your partner is another instance of the same AI model as you, with the same instructions; "
    "it studied its own earlier cases, using only its own features."
)

TRAIN_SYSTEM = """\
You are agent {name}. You are in a learning phase on your own. You will be shown {n_train} \
earlier cases in {n_batches} batches. Each case has {d} numeric features x1..x{d} and a \
real-valued target y. Later you will have to predict y for new cases from these features, \
together with a partner who sees other features of the same cases. You will not see these \
cases again: only your notes are kept. After each batch, rewrite your notes (at most \
{notes_chars} characters) with the model or rules you will use to predict y from x1..x{d}, how \
reliable they are, and anything else worth remembering. Make them concrete enough that you can \
compute a prediction from your notes alone."""

TRAIN_SCHEMA = {
    "type": "object",
    "properties": {"notes": {"type": "string", "description": "Your updated notes."}},
    "required": ["notes"],
    "additionalProperties": False,
}

TWIN_LINE = (
    "Your partner is another instance of the same AI model as you, with the same instructions "
    "and its own linear model fitted to its own features."
)


def reply_schema(
    joint: bool = False,
    columns: list[str] | None = None,
    notes: bool = True,
    logistic: bool = False,
) -> dict:
    """JSON schema of a twin's reply; adds the `regress` tool field if `columns` is given."""
    if logistic:
        est = (
            "Your proposed joint probability that y = 1" if joint else "Your probability that y = 1"
        )
    else:
        est = "Your proposed joint estimate of y" if joint else "Your estimate of y"
    props = {
        "message": {"type": "string", "description": "Message to your partner (short)."},
        "estimate": {"type": "number", "description": f"{est} for this round."},
    }
    if logistic:
        props["estimate"].update(minimum=0, maximum=1)
    if notes:
        props["notes"] = {"type": "string", "description": "Your updated private notes."}
    if columns:
        props["regress"] = {
            "type": "array",
            "items": {"type": "string", "enum": list(columns)},
            "description": "Predictors for a regression tool call; empty to answer.",
        }
    return {
        "type": "object",
        "properties": props,
        "required": list(props),
        "additionalProperties": False,
    }


REPLY_SCHEMA = reply_schema()


def regress(table: list[dict], predictors: list[str], current: dict) -> str:
    """Least squares of y on `predictors` + intercept over `table` rows; a text report.

    Parameters
    ----------
    table : list of dict
        One row per past step with every column name and "y".
    predictors : list of str
        Column names.
    current : dict
        Column values known for the current case (a prediction is given if all are known).
    """
    predictors = list(dict.fromkeys(predictors))  # dedupe, keep order
    n, k = len(table), len(predictors) + 1
    if n < k + 2:
        return f"Not enough past steps: {n} (need at least {k + 2} for {k} coefficients)."
    X = np.column_stack([np.ones(n)] + [[row[c] for row in table] for c in predictors])
    y = np.array([row["y"] for row in table])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    H = X @ np.linalg.pinv(X)
    loo = resid / np.clip(1 - np.diag(H), 1e-6, None)
    parts = [f"intercept {coef[0]:+.3f}"] + [f"{c} {b:+.3f}" for c, b in zip(predictors, coef[1:])]
    out = (
        f"Regression of y on {', '.join(predictors)} over {n} past steps: "
        f"{'; '.join(parts)}. In-sample MSE {np.mean(resid**2):.3f}, "
        f"leave-one-out MSE {np.mean(loo**2):.3f}."
    )
    if all(c in current for c in predictors):
        pred = coef[0] + sum(b * current[c] for c, b in zip(predictors, coef[1:]))
        out += f" Prediction for the current case: {pred:.3f}."
    else:
        missing = [c for c in predictors if c not in current]
        out += f" No prediction for the current case ({', '.join(missing)} not known yet)."
    return out


PROB_COLUMNS = ("own_model", "partner_open")  # enter the logistic tool as log-odds


def logistic_regress(table: list[dict], predictors: list[str], current: dict) -> str:
    """Logistic regression of y (0/1) on `predictors` + intercept over `table`; a text report.

    Probability columns (`PROB_COLUMNS`) enter as log-odds. A weak Gaussian prior (variance 100)
    keeps the fit finite when the classes are separable. The leave-one-out log loss refits
    without each row.
    """
    predictors = list(dict.fromkeys(predictors))
    n, k = len(table), len(predictors) + 1
    if n < k + 2:
        return f"Not enough past steps: {n} (need at least {k + 2} for {k} coefficients)."

    def col(c, rows):
        v = np.array([row[c] for row in rows], float)
        return logit(v) if c in PROB_COLUMNS else v

    X = np.column_stack([np.ones(n)] + [col(c, table) for c in predictors])
    y = np.array([row["y"] for row in table], float)
    P0 = 100.0 * np.eye(k)
    coef, _ = fit_logistic(X, y, P0=P0)
    ll_in = float(np.mean(log_loss(sigmoid(X @ coef), y)))
    loo = []
    for i in range(n):
        m = np.arange(n) != i
        c_i, _ = fit_logistic(X[m], y[m], theta0=coef, P0=P0)
        loo.append(float(log_loss(sigmoid(X[i] @ c_i), y[i])))
    names = [f"logit({c})" if c in PROB_COLUMNS else c for c in predictors]
    parts = [f"intercept {coef[0]:+.3f}"] + [f"{c} {b:+.3f}" for c, b in zip(names, coef[1:])]
    out = (
        f"Logistic regression of y on {', '.join(names)} over {n} past steps: "
        f"{'; '.join(parts)}. In-sample log loss {ll_in:.3f}, "
        f"leave-one-out log loss {np.mean(loo):.3f}."
    )
    if all(c in current for c in predictors):
        z = coef[0] + sum(
            b * (float(logit(current[c])) if c in PROB_COLUMNS else current[c])
            for c, b in zip(predictors, coef[1:])
        )
        out += f" Probability that y = 1 for the current case: {float(sigmoid(z)):.3f}."
    else:
        missing = [c for c in predictors if c not in current]
        out += f" No prediction for the current case ({', '.join(missing)} not known yet)."
    return out


@dataclass
class StepRecord:
    """What one twin remembers about a finished step."""

    t: int
    model_pred: float
    final_est: float
    partner_final: float
    team: float
    y: float
    x: np.ndarray | None = None
    partner_open: float = float("nan")


@dataclass
class TwinReply:
    """Parsed reply of one twin in one round (falls back to the linear prediction on failure)."""

    message: str
    estimate: float
    notes: str
    ok: bool
    backends: list[BackendReply] = field(repr=False, default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)


class LLMTwin:
    """One LLM twin with a linear learner as its tool.

    Parameters
    ----------
    name : str
        "A" or "B".
    linear : LinearAgent
        Trained linear learner on this twin's features (its solo head is the tool).
    backend
        Object with `complete(system, user, schema) -> BackendReply`.
    n_rounds : int
        Rounds per step (E + 1 in the notation of `protocols.Exchange`).
    T : int
        Number of steps in the communication phase (shown to the model).
    n_train : int
        Training-set size of the linear model (shown to the model).
    history_len : int
        How many past steps are shown in the prompt.
    max_chars, notes_chars : int
        Length limits for messages and notes (enforced by truncation).
    tell_twin : bool
        Whether the system prompt says the partner is another instance of the same model.
    update_linear : bool
        Whether the linear tool keeps learning from revealed labels (solo regression on own x).
    team_rule : str
        "average" (each gives its own estimate) or "joint" (each proposes the team estimate).
    show_partner_estimates : bool
        Show the partner's numeric estimates next to its messages (and in the history table).
    tools : bool
        Give the twin the `regress` tool over its history (implies the partner_open column is
        available only if `show_partner_estimates`).
    max_tool_calls : int
        Tool calls allowed per round.
    auto_regress : bool
        Run the tool once per round without being asked: y on own_model (round 1) and on
        own_model + partner_open (later rounds, if `show_partner_estimates`); requires `tools`.
    memory : str
        "notes": a fresh stateless call per round, memory = notes + last `history_len` steps.
        "persistent": one conversation per twin (`backend.conversation`), created on the first
        call, so a new twin (one per run) starts a new conversation.
    show_own_messages : bool
        With "notes" memory, show the step's conversation so far with both sides (your earlier
        messages and estimates as well as the partner's), not just the partner's messages. Meant
        for several exchanges per step; "persistent" memory has the whole conversation anyway.
    learner : str
        "linear": the trained linear learner's prediction is shown as a tool. "self": no linear
        model; the twin learns from the linear learner's training samples itself (`train`) and
        predicts from its notes (regression task, notes memory, no tools).
    train_batch : int
        Training cases per call in `train` (learner="self").
    pooling_hint : str
        Logistic task only: "none", "independent" (the two feature blocks are independent
        evidence) or "logodds" (that, plus: log-odds add, weights can exceed 1, averaging
        probabilities is too unsure). True only at rho = 0.
    """

    def __init__(
        self,
        name: str,
        linear: LinearAgent,
        backend,
        n_rounds: int,
        T: int,
        n_train: int,
        history_len: int = 10,
        max_chars: int = 300,
        notes_chars: int = 800,
        tell_twin: bool = True,
        update_linear: bool = False,
        team_rule: str = "average",
        show_partner_estimates: bool = False,
        tools: bool = False,
        max_tool_calls: int = 2,
        auto_regress: bool = False,
        memory: str = "notes",
        show_own_messages: bool = False,
        pooling_hint: str = "none",
        learner: str = "linear",
        train_batch: int = 25,
    ) -> None:
        if learner not in ("linear", "self"):
            raise ValueError(f"learner must be 'linear' or 'self', got {learner!r}")
        if learner == "self" and (
            tools or memory != "notes" or isinstance(linear, LogisticAgent) or update_linear
        ):
            raise ValueError(
                "learner='self' needs the regression task, memory='notes', no tools and "
                "update_linear=False"
            )
        self.self_learn = learner == "self"
        self.train_batch = train_batch
        if pooling_hint not in POOLING_HINTS:
            raise ValueError(f"pooling_hint must be one of {sorted(POOLING_HINTS)}")
        if pooling_hint != "none" and not isinstance(linear, LogisticAgent):
            raise ValueError("pooling_hint is only defined for the logistic task")
        if memory not in ("notes", "persistent"):
            raise ValueError(f"memory must be 'notes' or 'persistent', got {memory!r}")
        if auto_regress and not tools:
            raise ValueError("auto_regress requires tools=True")
        if team_rule not in ("average", "joint"):
            raise ValueError(f"team_rule must be 'average' or 'joint', got {team_rule!r}")
        self.name, self.linear, self.backend = name, linear, backend
        self.n_rounds, self.T = n_rounds, T
        self.history_len, self.max_chars, self.notes_chars = history_len, max_chars, notes_chars
        self.update_linear = update_linear
        self.joint = team_rule == "joint"
        self.show_est = show_partner_estimates
        self.max_tool_calls = max_tool_calls if tools else 0
        self.auto_regress = auto_regress
        self.persistent = memory == "persistent"
        self.show_own = show_own_messages
        self.logistic = isinstance(linear, LogisticAgent)
        self._regress = logistic_regress if self.logistic else regress
        self._own_step: list[tuple[str, float]] = []  # own (message, estimate) this step
        self.conv = None  # persistent conversation, created on the first call
        self.columns = (
            (
                [f"x{i + 1}" for i in range(linear.d)]
                + ["own_model"]
                + (["partner_open"] if show_partner_estimates else [])
            )
            if tools
            else []
        )
        self.schema = reply_schema(
            self.joint, self.columns or None, notes=not self.persistent, logistic=self.logistic
        )
        self.notes = ""
        self.history: list[StepRecord] = []
        fmt = dict(n_rounds=n_rounds, max_chars=max_chars)
        shown = " (its messages and estimates)" if show_partner_estimates else ""
        if show_own_messages and not self.persistent:
            shown += ", together with what you sent"
        L = self.logistic
        if L:
            rounds_tpl = ROUNDS_JOINT_LOGISTIC if self.joint else ROUNDS_AVERAGE_LOGISTIC
        else:
            rounds_tpl = ROUNDS_JOINT if self.joint else ROUNDS_AVERAGE
        rounds_par = rounds_tpl.format(shown=shown, **fmt)
        if pooling_hint != "none":
            rounds_par += "\n\n" + POOLING_HINTS[pooling_hint]
        tool_par = (
            (TOOL_PAR_LOGISTIC if L else TOOL_PAR).format(
                columns=", ".join(self.columns),
                column_help=(COLUMN_HELP_LOGISTIC if L else COLUMN_HELP).format(d=linear.d)
                + (
                    (PARTNER_OPEN_HELP_LOGISTIC if L else PARTNER_OPEN_HELP)
                    if show_partner_estimates
                    else ""
                ),
                max_tool_calls=max_tool_calls,
            )
            + (
                AUTO_PAR.format(
                    auto_partner=(
                        " (and in rounds 2 and later on own_model and partner_open)"
                        if show_partner_estimates
                        else ""
                    )
                )
                if auto_regress
                else ""
            )
            if tools
            else ""
        )
        if self.self_learn:
            sys_tpl, twin_line = SYSTEM_PROMPT_SELF, TWIN_LINE_SELF
        elif L:
            sys_tpl = SYSTEM_PROMPT_LOGISTIC
            twin_line = TWIN_LINE.replace("linear model", "logistic regression model")
        else:
            sys_tpl, twin_line = SYSTEM_PROMPT, TWIN_LINE
        self.system = sys_tpl.format(
            name=name,
            d=linear.d,
            twin_line=twin_line if tell_twin else "",
            n_train=n_train,
            rounds_par=rounds_par,
            memory_par=PERSISTENT_PAR
            if self.persistent
            else NOTES_PAR.format(notes_chars=notes_chars),
            tool_par=tool_par,
        )

    def model_prediction(self, x: np.ndarray) -> float:
        """The linear tool's prediction (solo head)."""
        return self.linear.predict(x, [])

    def _fallback(self, x: np.ndarray) -> float:
        """Estimate used for an unusable reply: the linear prediction, or for a self-learning
        twin its own last estimate this step (0, the prior mean of y, in round 1)."""
        if not self.self_learn:
            return self.model_prediction(x)
        return self._own_step[-1][1] if self._own_step else 0.0

    def train(self, X: np.ndarray, y: np.ndarray) -> list[dict]:
        """Learning phase (learner="self"): show the training cases in batches of `train_batch`;
        after each batch the twin rewrites its notes. Returns one record per batch (notes, ok,
        backend replies); an unusable reply keeps the previous notes."""
        n, d = X.shape
        n_batches = -(-n // self.train_batch)
        system = TRAIN_SYSTEM.format(
            name=self.name, n_train=n, n_batches=n_batches, d=d, notes_chars=self.notes_chars
        )
        log = []
        for b in range(n_batches):
            sl = slice(b * self.train_batch, (b + 1) * self.train_batch)
            head = ", ".join(f"x{i + 1}" for i in range(d)) + " | y"
            rows = [
                ", ".join(f"{v:.3f}" for v in xr) + f" | {yr:.3f}" for xr, yr in zip(X[sl], y[sl])
            ]
            user = "\n".join(
                [
                    f"Batch {b + 1} of {n_batches}.",
                    f"Your notes so far: {self.notes or '(empty)'}",
                    f"Cases ({head}):",
                    *rows,
                    "Reply with your updated notes.",
                ]
            )
            reply = self.backend.complete(system, user, TRAIN_SCHEMA)
            d_ = reply.data if isinstance(reply.data, dict) else {}
            ok = isinstance(d_.get("notes"), str) and bool(d_["notes"].strip())
            if ok:
                self.notes = d_["notes"][: self.notes_chars]
            log.append({"batch": b + 1, "notes": self.notes, "ok": ok, "reply": reply})
        return log

    def prompt(
        self, t: int, rnd: int, x: np.ndarray, partner: list[tuple[str, float]], extra=()
    ) -> str:
        """User prompt for step t, round rnd (both 0-based); `partner` = [(message, estimate)]."""
        lines = [
            f"Step {t + 1} of {self.T}, round {rnd + 1} of {self.n_rounds}.",
            f"Your features: {np.array2string(x, precision=3, separator=', ')}",
        ]
        if not self.self_learn:
            lines.append(self._model_line(x))

        def fmt(m: str, e: float) -> str:
            return f'estimate {e:.3f}, "{m}"' if self.show_est else f'"{m}"'

        if self.show_own and partner:
            lines.append("Conversation this step so far:")
            for i, (m, e) in enumerate(partner):
                if i < len(self._own_step):
                    lines.append(f"  round {i + 1}, you: {fmt(*self._own_step[i])}")
                lines.append(f"  round {i + 1}, partner: {fmt(m, e)}")
        elif partner:
            lines.append("Partner's messages this step:")
            lines += [f"  round {i + 1}: {fmt(m, e)}" for i, (m, e) in enumerate(partner)]
        else:
            lines.append("Partner's messages this step: none yet.")
        lines.append(f"Your notes: {self.notes or '(empty)'}")
        if self.history:
            recent = self.history[-self.history_len :]
            lines.append("Recent steps (oldest first):")
            po = " | partner round 1" if self.show_est else ""
            solo = "your round 1" if self.self_learn else "your model"
            lines.append(f"  step | {solo}{po} | your final | partner final | team | true y")
            lines += [
                f"  {h.t + 1} | {h.model_pred:.3f}"
                + (f" | {h.partner_open:.3f}" if self.show_est else "")
                + f" | {h.final_est:.3f} | {h.partner_final:.3f} | {h.team:.3f} | {h.y:.3f}"
                for h in recent
            ]
            H = self.history
            if self.logistic:
                ll = lambda f: np.mean([log_loss(f(h), h.y) for h in H])  # noqa: E731
                mse = ll
                score = "Mean log loss"
            else:
                mse = lambda f: np.mean([(f(h) - h.y) ** 2 for h in H])  # noqa: E731
                score = "Mean squared error"
            lines.append(
                f"{score} over all {len(H)} past steps: {solo} "
                f"{mse(lambda h: h.model_pred):.3f}, your final {mse(lambda h: h.final_est):.3f}, "
                f"team {mse(lambda h: h.team):.3f}."
            )
        lines += list(extra)
        lines.append(self._reply_line())
        return "\n".join(lines)

    def _model_line(self, x: np.ndarray) -> str:
        if self.logistic:
            return f"Your logistic model's probability that y = 1: {self.model_prediction(x):.3f}"
        return f"Your linear model's prediction: {self.model_prediction(x):.3f}"

    def _reply_line(self) -> str:
        if self.logistic:
            what = (
                "your proposal for the joint probability that y = 1"
                if self.joint
                else "your probability that y = 1"
            )
        else:
            what = (
                "your proposal for the joint estimate of y" if self.joint else "your estimate of y"
            )
        if self.persistent:
            return f"Reply with your message to your partner and {what} for this round."
        return (
            f"Reply with your message to your partner, {what} for this round, and "
            "your updated notes."
        )

    def prompt_persistent(
        self, t: int, rnd: int, x: np.ndarray, partner: list[tuple[str, float]], extra=()
    ) -> str:
        """New user turn of the persistent conversation for step t, round rnd (0-based).

        Round 1 of a step reports the previous step's outcome and the new case; later rounds
        only add the partner's message from the round before (the rest is already in context).
        """
        lines = []
        if rnd == 0:
            if self.history:
                h = self.history[-1]
                po = f", partner round 1 {h.partner_open:.3f}" if self.show_est else ""
                lines.append(
                    f"Outcome of step {h.t + 1}: true y {h.y:.3f}; team {h.team:.3f}; your final "
                    f"{h.final_est:.3f}; partner final {h.partner_final:.3f}; your model "
                    f"{h.model_pred:.3f}{po}."
                )
            lines += [
                f"Step {t + 1} of {self.T}, round 1 of {self.n_rounds}.",
                f"Your features: {np.array2string(x, precision=3, separator=', ')}",
                self._model_line(x),
                "Partner's messages this step: none yet.",
            ]
        else:
            m, e = partner[-1]
            sent = f'estimate {e:.3f}, "{m}"' if self.show_est else f'"{m}"'
            lines += [
                f"Step {t + 1}, round {rnd + 1} of {self.n_rounds}.",
                f"Partner's round {rnd} message: {sent}",
            ]
        lines += list(extra)
        lines.append(self._reply_line())
        return "\n".join(lines)

    def _send(self, user: str) -> BackendReply:
        if not self.persistent:
            return self.backend.complete(self.system, user, self.schema)
        if self.conv is None:
            self.conv = self.backend.conversation(self.system)
        return self.conv.send(user, self.schema)

    def _table(self) -> list[dict]:
        """History as rows of tool columns + y."""
        rows = []
        for h in self.history:
            row = {f"x{i + 1}": float(v) for i, v in enumerate(h.x)}
            row.update(own_model=h.model_pred, partner_open=h.partner_open, y=h.y)
            rows.append(row)
        return rows

    def act(self, t: int, rnd: int, x: np.ndarray, partner: list[tuple[str, float]]) -> TwinReply:
        """One round, with up to `max_tool_calls` regression calls before the answer.

        On an unusable reply, falls back to the linear prediction.
        """
        if rnd == 0:
            self._own_step = []
        current = {f"x{i + 1}": float(v) for i, v in enumerate(x)}
        if not self.self_learn:
            current["own_model"] = self.model_prediction(x)
        if partner and self.show_est:
            current["partner_open"] = partner[0][1]
        extra, backends, calls = [], [], []
        if self.auto_regress:
            req = ["own_model"] + (["partner_open"] if "partner_open" in current else [])
            result = self._regress(self._table(), req, current)
            calls.append({"predictors": req, "result": result, "auto": True})
            extra.append(f"Automatic tool run: regress on {', '.join(req)}. Result: {result}")
        n_sent = 0  # extra lines already sent (persistent tool follow-ups send only new ones)
        for k in range(self.max_tool_calls + 1):
            if not self.persistent:
                user = self.prompt(t, rnd, x, partner, extra)
            elif k == 0:
                user = self.prompt_persistent(t, rnd, x, partner, extra)
            else:  # tool follow-up: only the new results, the rest is in the conversation
                user = "\n".join(extra[n_sent:] + [self._reply_line()])
            n_sent = len(extra)
            reply = self._send(user)
            backends.append(reply)
            d = reply.data if isinstance(reply.data, dict) else {}
            req = [c for c in d.get("regress", []) or [] if c in self.columns]
            if not req or k == self.max_tool_calls:
                break
            result = self._regress(self._table(), req, current)
            calls.append({"predictors": req, "result": result})
            extra.append(f"Tool call {k + 1}: regress on {', '.join(req)}. Result: {result}")
            if k + 1 == self.max_tool_calls:
                extra.append("No tool calls left this round: leave `regress` empty and answer.")
        try:
            est = float(d["estimate"])
            ok = bool(np.isfinite(est))
            if ok and self.logistic:
                est = min(max(est, 0.0), 1.0)
        except (TypeError, KeyError, ValueError):
            ok = False
        if not ok:
            fb = self._fallback(x)
            self._own_step.append(("", fb))
            return TwinReply("", fb, self.notes, False, backends, calls)
        if not self.persistent:
            self.notes = str(d.get("notes", ""))[: self.notes_chars]
        msg = str(d.get("message", ""))[: self.max_chars]
        self._own_step.append((msg, est))
        return TwinReply(msg, est, self.notes, True, backends, calls)

    def record(self, rec: StepRecord, x: np.ndarray) -> None:
        """Store the finished step; optionally let the linear tool learn from (x, y)."""
        rec.x = np.asarray(x, float)
        self.history.append(rec)
        if self.update_linear:
            self.linear.heads[0].update(x, rec.y)
