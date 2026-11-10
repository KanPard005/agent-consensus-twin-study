"""Worded twins (idea 3a): the self-learning twin task (S1) told as a house-price story.

The numbers are the same as in S1: y = <x^A, w^A> + <x^B, w^B> + eps from `LinearTwinEnv`, and
the population limits and the linear baselines (P0-P3) apply unchanged. Only the presentation
changes. Each case is a house sale written as a sentence:

- twin A sees the house itself (floor area, condition, garden size);
- twin B sees its location (school rating, walkability, tree cover);
- y is the sale price relative to what is typical for the area, in units of $50,000.

Each standardized feature x_j is shown as `mean_j + scale_j * s_j * x_j` in natural units, where
s_j = +-1 is the sign of the twin's own least-squares coefficient on its training cases. So in the
data, every attribute moves the price in the direction common sense expects ("bigger, better
kept, nicer area -> pricier"): the story's priors agree with the data rather than fight it. Flipping
a feature's sign is a bijection, so no information is lost or added, except for the rounding to
the display precision (at most 1/30 of a standard deviation).

The question is whether the twins' behaviour from S1 (they learn cruder rules than OLS, hold on to
them, and at best meet the partner partway instead of adding the two half-signals) survives when
the data reads as a familiar real-world problem instead of an abstract regression.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from consensus.llm_agent import TRAIN_SCHEMA, LLMTwin
from consensus.protocols import LLMExchange


@dataclass(frozen=True)
class Attribute:
    """One feature in natural units: shown value = mean + scale * x, rounded to `decimals`."""

    label: str  # phrase before the value, e.g. "floor area"
    mean: float
    scale: float
    decimals: int
    unit: str  # appended after the value, e.g. " m²" or "/10"

    def show(self, x: float) -> str:
        return f"{self.label} {self.mean + self.scale * x:.{self.decimals}f}{self.unit}"


# Means and scales keep +-3.5 standard deviations inside the attribute's natural range (positive
# areas, ratings within 0-10, percentages within 0-100), so no value needs clipping.
HOUSE = (
    Attribute("floor area", 130.0, 30.0, 0, " m²"),
    Attribute("condition", 6.0, 1.1, 1, "/10"),
    Attribute("garden", 300.0, 80.0, 0, " m²"),
)
LOCATION = (
    Attribute("school rating", 6.0, 1.1, 1, "/10"),
    Attribute("walkability", 55.0, 12.0, 0, "/100"),
    Attribute("tree cover", 35.0, 9.0, 0, "%"),
)
STORY = {
    "A": ("the house itself", "its location", HOUSE),
    "B": ("its location", "the house itself", LOCATION),
}

# Replaces the abstract task paragraph of the self-learning system prompt.
TASK_ABSTRACT = (
    "Each step there is a new case with an unknown real-valued target y. You see {d} numeric "
    "features of the case; your partner sees different features of the same case."
)
TASK_WORDED = (
    "Each step a house in the same city is sold. You predict y, its sale price relative to what "
    "is typical for the city, in units of $50,000 (y = 1.2 means $60,000 above typical; y = -0.5 "
    "means $25,000 below). You see {d} facts about {own_side} ({names}); your partner sees "
    "different facts, about {other_side}, that you do not see."
)

TRAIN_SYSTEM_WORDED = """\
You are agent {name}. You are in a learning phase on your own. You will be shown {n_train} earlier \
house sales in {n_batches} batches. For each sale you see {d} facts about {own_side} ({names}) and \
y, its sale price relative to what is typical for the city, in units of $50,000. Later you will \
have to predict y for new sales from these facts, together with a partner who sees other facts \
about the same houses ({other_side}). You will not see these sales again: only your notes are \
kept. After each batch, rewrite your notes (at most {notes_chars} characters) with the model or \
rules you will use to predict y from your facts, how reliable they are, and anything else worth \
remembering. Make them concrete enough that you can compute a prediction from your notes alone."""


class WordedTwin(LLMTwin):
    """Self-learning LLM twin that sees its cases as house sales described in words.

    Same arguments as `LLMTwin`; requires learner="self" (the story replaces the abstract feature
    vectors, which only the self-learning prompts show) and d = 3 (one attribute per feature).
    """

    def __init__(self, name: str, linear, backend, *args, **kwargs) -> None:
        if kwargs.get("learner") != "self":
            raise ValueError("WordedTwin needs learner='self'")
        if kwargs.get("feature_names", "shared") != "shared":
            raise ValueError("WordedTwin names its features itself; leave feature_names unset")
        if kwargs.get("history_features"):
            raise ValueError("WordedTwin shows past cases without their facts (history_features)")
        super().__init__(name, linear, backend, *args, **kwargs)
        self.own_side, self.other_side, self.attrs = STORY[name]
        if linear.d != len(self.attrs):
            raise ValueError(f"WordedTwin needs d = {len(self.attrs)}, got {linear.d}")
        self.signs = np.ones(linear.d)  # set from the training cases in `train`
        abstract = TASK_ABSTRACT.format(d=linear.d)
        if abstract not in self.system:
            raise RuntimeError("self-learning system prompt changed; update TASK_ABSTRACT")
        self.system = self.system.replace(abstract, TASK_WORDED.format(**self._story()))
        # The rest of the prompt says "features"; in the story they are "facts".
        self.system = self.system.replace("your features", "your facts").replace(
            "its own features", "its own facts"
        )

    def _story(self) -> dict:
        names = ", ".join(a.label for a in self.attrs)
        return dict(
            d=len(self.attrs), own_side=self.own_side, other_side=self.other_side, names=names
        )

    def describe(self, x: np.ndarray) -> str:
        """The facts of one case in words, e.g. 'floor area 142 m², condition 7.3/10, ...'."""
        return ", ".join(a.show(s * v) for a, s, v in zip(self.attrs, self.signs, x))

    def _features_line(self, x: np.ndarray) -> str:
        return f"This house ({self.own_side}): {self.describe(x)}."

    def train(self, X: np.ndarray, y: np.ndarray) -> list[dict]:
        """Learning phase as in `LLMTwin.train`, with each case written as a sentence.

        The display signs are fixed first, from the twin's own least-squares fit on these cases,
        so that every attribute raises the price in its own data (see the module docstring).
        """
        if len(X) == 0:
            return []
        coef, *_ = np.linalg.lstsq(np.c_[np.ones(len(X)), X], y, rcond=None)
        self.signs = np.where(coef[1:] < 0, -1.0, 1.0)
        n = len(X)
        n_batches = -(-n // self.train_batch)
        system = TRAIN_SYSTEM_WORDED.format(
            name=self.name,
            n_train=n,
            n_batches=n_batches,
            notes_chars=self.notes_chars,
            **self._story(),
        )
        log = []
        for b in range(n_batches):
            sl = slice(b * self.train_batch, (b + 1) * self.train_batch)
            start = b * self.train_batch
            rows = [
                f"Sale {start + i + 1}: {self.describe(xr)}; y = {yr:.2f}."
                for i, (xr, yr) in enumerate(zip(X[sl], y[sl]))
            ]
            user = "\n".join(
                [
                    f"Batch {b + 1} of {n_batches}.",
                    f"Your notes so far: {self.notes or '(empty)'}",
                    "Sales:",
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


class WordedExchange(LLMExchange):
    """`LLMExchange` with `WordedTwin`s (config name `llm_worded`)."""

    twin_cls = WordedTwin
