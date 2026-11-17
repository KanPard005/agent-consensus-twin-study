"""LLM twin for the hidden-profile task (`consensus.hp_data`): reads verbal clues about café
sites, talks with its partner, and gives a probability for each site being the best choice.

Two memory modes, as in the numeric task:

- `memory="notes"`: each round is a fresh call with the case, the partner's messages so far, the
  twin's private notes (rewritten every round, kept across cases) and a short table of past
  cases with the revealed best site.
- `memory="persistent"`: one continuing conversation per twin for the whole run. Round 1 of a
  case reports the previous case's outcome and gives the new facts; later rounds only add the
  partner's latest message. No notes: everything the twin said or thought stays in context.

`private_thinking=True` adds a `thinking` field, generated before the message, that the
partner never sees. With persistent memory the twin keeps all its earlier private reasoning in
context: hidden thinking that persists, plus a public channel (message and probabilities).

`facts_first=True` turns round 1 into a facts-only round: the twins send each other their facts
but give no probabilities and are asked not to say which site they favour, so neither commits
to (or hears) a first read before the partner's facts arrive.
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

Each case has {n_rounds} rounds. {rounds_line} From round 2 on you see what your partner sent in the earlier rounds of the \
case. The team's choice is the site with the highest average of your two final-round \
probabilities. After the final round you are told which site was in fact the best.

{memory_line}{thinking_line}"""

NOTES_LINE = (
    "You keep private notes that persist across cases (at most {notes_chars} characters). "
    "Rewrite them every round with whatever will help you later, for example what kinds of "
    "facts matter, how reliable your partner is, or conventions you and your partner use."
)
PERSISTENT_LINE = (
    "This is one continuing conversation for all {T} cases: you remember everything you have "
    "seen, said and thought in earlier cases and rounds. Each new case starts with the outcome "
    "of the previous one."
)
THINKING_LINE = (
    " Each reply has a private thinking field for your own reasoning. Your partner never sees "
    "it; only your message{probs_part} is sent to your partner."
)

SPLIT_LINE = (
    "Some facts you both have; others only you have, and your partner has facts that you do not. "
)
ROUNDS_OPINION = (
    "In every round you and your partner simultaneously send each other a message (at most "
    "{max_chars} characters) and each give your probability that each site is the best choice."
)
ROUNDS_FACTS_FIRST = (
    "In every round you and your partner simultaneously send each other a message (at most "
    "{max_chars} characters). Round 1 is for facts only: share the facts you have, without "
    "giving probabilities or saying which site you favour. From round 2 on you also each give "
    "your probability that each site is the best choice."
)
FACTS_ROUND_ASK = (
    "This is the facts-only round: in your message, share your facts (especially those your "
    "partner may not have). Do not give probabilities or say which site you favour. Reply with "
    "your message and your updated notes."
)
TWIN_LINE = (
    "Your partner is another instance of the same AI model as you, with the same instructions."
)

SOLO_SYSTEM = """\
You advise a small company on where to open its next café. There are {n_sites} candidate sites \
(Site 1 to Site {n_sites}), and exactly one of them is the best choice. You are given all the \
facts known about the sites. Give your probability that each site is the best choice."""


def probs_schema(
    n_sites: int, message: bool = True, notes: bool = True, thinking: bool = False
) -> dict:
    """JSON schema: a probability per site (+ message and notes for the twins, + a private
    `thinking` field first, so the reasoning is written before the message and probabilities)."""
    props: dict = {}
    if thinking:
        props["thinking"] = {
            "type": "string",
            "description": "Your private reasoning. Your partner never sees it.",
        }
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


def message_schema(notes: bool = True, thinking: bool = False) -> dict:
    """JSON schema of the facts-only round: a message (and notes / thinking), no probabilities."""
    return probs_schema(0, notes=notes, thinking=thinking)


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
    probs: np.ndarray | None  # None in a facts-only round
    ok: bool
    replies: list[BackendReply] = field(default_factory=list)
    thinking: str = ""  # private reasoning (private_thinking=True), never sent to the partner


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
    facts_first : bool
        Round 1 is facts only (no probabilities, no stated favourite).
    memory : str
        "notes" (fresh call per round with notes and a table of past cases) or "persistent"
        (one conversation per twin for the whole run; needs `backend.conversation`).
    private_thinking : bool
        Add a private `thinking` field to every reply (never shown to the partner).
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
        facts_first: bool = False,
        memory: str = "notes",
        private_thinking: bool = False,
    ) -> None:
        if memory not in ("notes", "persistent"):
            raise ValueError(f"memory must be 'notes' or 'persistent', got {memory!r}")
        self.name, self.backend = name, backend
        self.persistent, self.private_thinking = memory == "persistent", private_thinking
        self.conv = None  # persistent conversation, created on the first call
        self.n_sites, self.n_rounds, self.T = n_sites, n_rounds, T
        self.max_chars, self.notes_chars, self.history_len = max_chars, notes_chars, history_len
        self.facts_first = facts_first
        rounds_line = ROUNDS_FACTS_FIRST if facts_first else ROUNDS_OPINION
        self.system = SYSTEM_PROMPT.format(
            name=name,
            n_sites=n_sites,
            n_rounds=n_rounds,
            rounds_line=rounds_line.format(max_chars=max_chars),
            max_chars=max_chars,
            split_line=SPLIT_LINE if tell_split else "",
            twin_line=TWIN_LINE if tell_twin else "",
            memory_line=(
                PERSISTENT_LINE.format(T=T)
                if self.persistent
                else NOTES_LINE.format(notes_chars=notes_chars)
            ),
            thinking_line=(
                THINKING_LINE.format(
                    probs_part=" and your probabilities"
                    if not facts_first
                    else " (and, from round 2, your probabilities)"
                )
                if private_thinking
                else ""
            ),
        ).replace(" \n", "\n")
        notes = not self.persistent
        self.schema = probs_schema(n_sites, notes=notes, thinking=private_thinking)
        self.facts_schema = message_schema(notes=notes, thinking=private_thinking)
        self.notes = ""
        self.history: list[CaseRecord] = []

    def is_facts_round(self, rnd: int) -> bool:
        return self.facts_first and rnd == 0

    def prompt(
        self, t: int, rnd: int, case: Case, partner: list[tuple[str, np.ndarray | None]]
    ) -> str:
        """User prompt for case t, round rnd (0-based); `partner` = [(message, probabilities)],
        probabilities None for a facts-only round."""
        lines = [f"Case {t + 1} of {self.T}, round {rnd + 1} of {self.n_rounds}.", "Your facts:"]
        lines += facts_block(case, self.name)
        if partner:
            lines.append("Partner's messages this case:")
            lines += [
                f'  round {i + 1}: probabilities {fmt_probs(p)}; "{m}"'
                if p is not None
                else f'  round {i + 1}: "{m}"'
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
        lines.append(self._reply_line(rnd))
        return "\n".join(lines)

    def _reply_line(self, rnd: int) -> str:
        if self.is_facts_round(rnd):
            ask = FACTS_ROUND_ASK
            if self.persistent:
                ask = ask.replace(" and your updated notes", "")
        elif self.persistent:
            ask = "Reply with your message to your partner and your probability for each site."
        else:
            ask = (
                "Reply with your message to your partner, your probability for each site, and "
                "your updated notes."
            )
        if self.private_thinking:
            ask += " Reason privately in thinking first."
        return ask

    def prompt_persistent(
        self, t: int, rnd: int, case: Case, partner: list[tuple[str, np.ndarray | None]]
    ) -> str:
        """New user turn of the persistent conversation for case t, round rnd (0-based).

        Round 1 reports the previous case's outcome and gives the new facts; later rounds only
        add the partner's message from the round before (the rest is already in context).
        """
        lines = []
        if rnd == 0:
            if self.history:
                h = self.history[-1]
                hits = sum(r.team_pick == r.best for r in self.history)
                lines.append(
                    f"Outcome of case {h.t + 1}: the best site was Site {h.best + 1}. Team pick "
                    f"Site {h.team_pick + 1}; your round-1 pick Site {h.own_pick + 1}; partner "
                    f"round-1 pick Site {h.partner_pick + 1}. Team picked the best site in {hits} "
                    f"of {len(self.history)} cases so far."
                )
            lines += [
                f"Case {t + 1} of {self.T}, round 1 of {self.n_rounds} (sites are numbered "
                "afresh in each case).",
                "Your facts:",
                *facts_block(case, self.name),
                "Partner's messages this case: none yet.",
            ]
        else:
            m, p = partner[-1]
            sent = f'probabilities {fmt_probs(p)}; "{m}"' if p is not None else f'"{m}"'
            lines += [
                f"Case {t + 1}, round {rnd + 1} of {self.n_rounds}.",
                f"Partner's round {rnd} message: {sent}",
            ]
        lines.append(self._reply_line(rnd))
        return "\n".join(lines)

    def _send(self, user: str, schema: dict) -> BackendReply:
        if not self.persistent:
            return self.backend.complete(self.system, user, schema)
        if self.conv is None:
            self.conv = self.backend.conversation(self.system)
        return self.conv.send(user, schema)

    def act(
        self,
        t: int,
        rnd: int,
        case: Case,
        partner: list[tuple[str, np.ndarray | None]],
        previous: np.ndarray | None,
    ) -> HPReply:
        """One round. An unusable reply keeps the twin's previous probabilities (uniform in
        its first probability round) and sends an empty message. A facts-only round returns
        probs=None."""
        if self.persistent:
            user = self.prompt_persistent(t, rnd, case, partner)
        else:
            user = self.prompt(t, rnd, case, partner)
        if self.is_facts_round(rnd):
            reply = self._send(user, self.facts_schema)
            data = reply.data if isinstance(reply.data, dict) else {}
            msg = data.get("message")
            if not isinstance(msg, str):
                return HPReply("", None, False, [reply])
            self._keep_notes(data.get("notes"))
            return HPReply(msg[: self.max_chars], None, True, [reply], _thinking(data))
        reply = self._send(user, self.schema)
        p = parse_probs(reply.data, self.n_sites)
        if p is None:
            fallback = previous if previous is not None else np.full(self.n_sites, 1 / self.n_sites)
            return HPReply("", fallback, False, [reply])
        msg = str(reply.data.get("message", ""))[: self.max_chars]
        self._keep_notes(reply.data.get("notes"))
        return HPReply(msg, p, True, [reply], _thinking(reply.data))

    def _keep_notes(self, notes) -> None:
        if isinstance(notes, str) and notes.strip():
            self.notes = notes[: self.notes_chars]

    def record(self, rec: CaseRecord) -> None:
        self.history.append(rec)


def _thinking(data: dict) -> str:
    t = data.get("thinking", "")
    return t if isinstance(t, str) else ""
