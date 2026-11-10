"""LLM twin for the hidden-profile task (`consensus.hp_data`): reads verbal clues about café
sites, talks with its partner, and gives a probability for each site being the best choice.

Memory is as in the numeric task's "notes" mode: each round is a fresh call with the case, the
partner's messages so far, the twin's private notes (rewritten every round, kept across cases)
and a short table of past cases with the revealed best site.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from consensus.hp_data import Case
from consensus.llm_backends import BackendReply

SYSTEM_PROMPT = """\
You are agent {name}, one of two agents (A and B) who advise a small company on where to open its \
next café. In each case there are {n_sites} candidate sites (Site 1 to Site {n_sites}), and \
exactly one of them is the best choice. You and your partner have each been given facts about \
the sites. {split_line}{twin_line}

Each case has {n_rounds} rounds. In every round you and your partner simultaneously send each \
other a message (at most {max_chars} characters) and each give your probability that each site \
is the best choice. From round 2 on you see what your partner sent in the earlier rounds of the \
case. The team's choice is the site with the highest average of your two final-round \
probabilities. After the final round you are told which site was in fact the best.

You keep private notes that persist across cases (at most {notes_chars} characters). Rewrite them \
every round with whatever will help you later, for example what kinds of facts matter, how \
reliable your partner is, or conventions you and your partner use."""

SPLIT_LINE = (
    "Some facts you both have; others only you have, and your partner has facts that you do not. "
)
TWIN_LINE = (
    "Your partner is another instance of the same AI model as you, with the same instructions."
)

SOLO_SYSTEM = """\
You advise a small company on where to open its next café. There are {n_sites} candidate sites \
(Site 1 to Site {n_sites}), and exactly one of them is the best choice. You are given all the \
facts known about the sites. Give your probability that each site is the best choice."""


def probs_schema(n_sites: int, message: bool = True, notes: bool = True) -> dict:
    """JSON schema: a probability per site (+ message and notes for the twins)."""
    props: dict = {}
    if message:
        props["message"] = {"type": "string", "description": "Your message to your partner."}
    for s in range(n_sites):
        props[f"site_{s + 1}"] = {
            "type": "number",
            "description": f"Your probability that Site {s + 1} is the best choice.",
        }
    if notes:
        props["notes"] = {"type": "string", "description": "Your updated private notes."}
    return {
        "type": "object",
        "properties": props,
        "required": list(props),
        "additionalProperties": False,
    }


def parse_probs(data: dict | None, n_sites: int) -> np.ndarray | None:
    """Probabilities from a reply, normalised to sum to 1; None if unusable."""
    if not isinstance(data, dict):
        return None
    try:
        p = np.array([float(data[f"site_{s + 1}"]) for s in range(n_sites)])
    except (KeyError, TypeError, ValueError):
        return None
    p = np.clip(np.nan_to_num(p, nan=0.0), 0.0, None)
    return p / p.sum() if p.sum() > 0 else None


def facts_block(case: Case, twin: str | None) -> list[str]:
    """The facts a twin sees (twin=None: all facts), grouped by site, in the case's order."""
    lines = []
    for s in range(case.n_sites):
        clues = [c for c in case.clues if c.site == s and (twin is None or c.visible_to(twin))]
        lines.append(f"Site {s + 1}:")
        lines += [f"  - {c.text}" for c in clues] or ["  (you have no facts about this site)"]
    return lines


def fmt_probs(p: np.ndarray) -> str:
    return ", ".join(f"Site {s + 1} {v:.2f}" for s, v in enumerate(p))


@dataclass
class HPReply:
    """One round's answer: the message, the probabilities and the backend reply."""

    message: str
    probs: np.ndarray
    ok: bool
    replies: list[BackendReply] = field(default_factory=list)


@dataclass
class CaseRecord:
    """What a twin remembers of a finished case."""

    t: int
    own_pick: int  # round-1 favourite (0-based)
    partner_pick: int
    team_pick: int
    best: int


class HPTwin:
    """One LLM twin in the hidden-profile task.

    Parameters
    ----------
    name : str
        "A" or "B".
    backend
        Object with `complete(system, user, schema) -> BackendReply`.
    n_sites, n_rounds, T : int
        Sites per case, rounds per case, number of cases (shown to the model).
    max_chars, notes_chars : int
        Length limits for messages and notes (enforced by truncation).
    history_len : int
        Past cases shown in the prompt.
    tell_split : bool
        Say that each twin has facts the other lacks (true by construction).
    tell_twin : bool
        Say that the partner is another instance of the same model.
    """

    def __init__(
        self,
        name: str,
        backend,
        n_sites: int,
        n_rounds: int,
        T: int,
        max_chars: int = 600,
        notes_chars: int = 800,
        history_len: int = 10,
        tell_split: bool = True,
        tell_twin: bool = True,
    ) -> None:
        self.name, self.backend = name, backend
        self.n_sites, self.n_rounds, self.T = n_sites, n_rounds, T
        self.max_chars, self.notes_chars, self.history_len = max_chars, notes_chars, history_len
        self.system = SYSTEM_PROMPT.format(
            name=name,
            n_sites=n_sites,
            n_rounds=n_rounds,
            max_chars=max_chars,
            notes_chars=notes_chars,
            split_line=SPLIT_LINE if tell_split else "",
            twin_line=TWIN_LINE if tell_twin else "",
        ).replace(" \n", "\n")
        self.schema = probs_schema(n_sites)
        self.notes = ""
        self.history: list[CaseRecord] = []

    def prompt(self, t: int, rnd: int, case: Case, partner: list[tuple[str, np.ndarray]]) -> str:
        """User prompt for case t, round rnd (0-based); `partner` = [(message, probabilities)]."""
        lines = [f"Case {t + 1} of {self.T}, round {rnd + 1} of {self.n_rounds}.", "Your facts:"]
        lines += facts_block(case, self.name)
        if partner:
            lines.append("Partner's messages this case:")
            lines += [
                f'  round {i + 1}: probabilities {fmt_probs(p)}; "{m}"'
                for i, (m, p) in enumerate(partner)
            ]
        else:
            lines.append("Partner's messages this case: none yet.")
        lines.append(f"Your notes: {self.notes or '(empty)'}")
        if self.history:
            lines.append("Past cases (oldest first; sites are numbered afresh in each case):")
            lines.append(
                "  case | your round-1 pick | partner round-1 pick | team pick | best site"
            )
            lines += [
                f"  {h.t + 1} | {h.own_pick + 1} | {h.partner_pick + 1} | {h.team_pick + 1} | "
                f"{h.best + 1}"
                for h in self.history[-self.history_len :]
            ]
            hits = sum(h.team_pick == h.best for h in self.history)
            lines.append(f"Team picked the best site in {hits} of {len(self.history)} past cases.")
        lines.append(
            "Reply with your message to your partner, your probability for each site, and your "
            "updated notes."
        )
        return "\n".join(lines)

    def act(
        self,
        t: int,
        rnd: int,
        case: Case,
        partner: list[tuple[str, np.ndarray]],
        previous: np.ndarray | None,
    ) -> HPReply:
        """One round. An unusable reply keeps the twin's previous probabilities (uniform in
        round 1) and sends an empty message."""
        reply = self.backend.complete(self.system, self.prompt(t, rnd, case, partner), self.schema)
        p = parse_probs(reply.data, self.n_sites)
        if p is None:
            fallback = previous if previous is not None else np.full(self.n_sites, 1 / self.n_sites)
            return HPReply("", fallback, False, [reply])
        msg = str(reply.data.get("message", ""))[: self.max_chars]
        notes = reply.data.get("notes")
        if isinstance(notes, str) and notes.strip():
            self.notes = notes[: self.notes_chars]
        return HPReply(msg, p, True, [reply])

    def record(self, rec: CaseRecord) -> None:
        self.history.append(rec)
