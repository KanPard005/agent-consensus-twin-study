"""The café hidden-profile task with twins as live agent sessions (LogBook 2026-11-18).

Same cases as `consensus.hp_data` (sites, verbal clues split between the twins), but each twin
is one Claude Agent SDK session for the whole run, acting only through the tools of an
in-process MCP server, as in `consensus.sessions`. Per case:

- `next_case()` waits for both twins and returns the twin's own facts about each site;
- `submit_opening(site_1..)` records the twin's probabilities before any talk (the analogue of
  the round-1 read in the fixed-round protocol, used for deference);
- `send_message(text)` / `read_messages()`: free talk, as much or as little as the twins choose
  (`read_messages` waits until the partner sends something new, submits a final, or is also
  waiting);
- `submit_final(site_1..)` (once per case) waits for the partner's final; then both twins see
  both finals, the team's pick (argmax of the average of the two finals, ties to the lowest
  site) and which site was in fact the best.

Unlike the fixed-round protocol (`consensus.hp_protocol.HPDiscussion`) there is no message
length per round, no forced rounds and no notes: one session keeps everything in context.
Prompts are neutral about how to use the partner (LogBook rule: observe, don't steer).
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

import numpy as np

from consensus.hp_agent import SPLIT_LINE, TWIN_LINE, facts_block, fmt_probs
from consensus.hp_data import Case
from consensus.sessions import TwinSession

SYSTEM_HP = """\
You are agent {name}, one of two agents (A and B) who advise a small company on where to open its \
next café. There are {T} cases. In each case there are {n_sites} candidate sites (Site 1 to Site \
{n_sites}), and exactly one of them is the best choice. You and your partner have each been given \
facts about the sites. {split_line}{twin_line}

You work only through the tools of a case API. For each case:
- next_case gives you your facts about the sites (it waits until your partner has also asked, so \
you both work on the same case);
- submit_opening records your own probability that each site is the best choice, before you talk;
- then you and your partner can communicate freely: send_message sends a message (at most \
{max_chars} characters), and read_messages shows your partner's opening and new messages (it waits \
until your partner sends something new, submits a final, or is also waiting);
- when you are ready, submit_final records your final probability for each site (once per case). \
It waits until your partner has also submitted its final.
The team's choice is the site with the highest average of your two final probabilities. After \
both finals are in, you are both told which site was in fact the best. You can make at most \
{cap} tool calls per case.

Keep working with the tools until a tool result says that the run is over; do not stop before \
that."""

TOOL_DOCS = {
    "next_case": "Start the next case: returns your facts about each site. Waits until your "
    "partner has also asked.",
    "submit_opening": "Record your own probability that each site is the best choice, before "
    "talking with your partner (once per case).",
    "send_message": "Send a message to your partner.",
    "read_messages": "Show your partner's opening and any new messages. Waits until your "
    "partner sends something new, submits a final, or is also waiting.",
    "submit_final": "Record your final probability that each site is the best choice (once per "
    "case). Waits until your partner has also submitted its final, then shows the outcome.",
}


def normalise_probs(args: dict, n_sites: int) -> np.ndarray | str:
    """Probabilities from tool arguments site_1..site_n, normalised; an error string if unusable."""
    try:
        p = np.array([float(args[f"site_{s + 1}"]) for s in range(n_sites)])
    except (KeyError, TypeError, ValueError):
        names = ", ".join(f"site_{s + 1}" for s in range(n_sites))
        return f"Error: give a probability for every site ({names})."
    if not np.all(np.isfinite(p)) or np.any(p < 0) or p.sum() <= 0:
        return "Error: probabilities must be non-negative numbers that are not all zero."
    return p / p.sum()


class HPSessionAPI:
    """Shared, thread-safe case API for two twins (see the module docstring).

    Parameters
    ----------
    cases : list[Case]
        The run's cases, in order.
    max_chars : int
        Longest message; longer ones are cut.
    cap : int
        Tool calls per twin per case (submit_final and next_case stay open past it).
    """

    def __init__(self, cases: list[Case], max_chars: int = 2000, cap: int = 25) -> None:
        self.cases, self.T, self.n_sites = cases, len(cases), cases[0].n_sites
        self.max_chars, self.cap = max_chars, cap
        self.cond = threading.Condition()
        # Attributes TwinSession.write_status reads: phase, q, T, n_seen, calls.
        self.phase = {k: "query" for k in "AB"}  # query | done
        self.q = {k: -1 for k in "AB"}
        self.n_seen = {k: 0 for k in "AB"}
        self.calls: dict[str, dict[int, int]] = {k: {} for k in "AB"}
        self.want = {k: -1 for k in "AB"}
        nan = np.full((self.T, self.n_sites), np.nan)
        self.opening = {k: nan.copy() for k in "AB"}
        self.final = {k: nan.copy() for k in "AB"}
        self.team = nan.copy()
        self.resolved = np.zeros(self.T, bool)
        self.inbox: dict[str, list[tuple[int, str]]] = {k: [] for k in "AB"}
        self.n_read = {k: 0 for k in "AB"}
        self.waiting: dict[str, str | None] = {k: None for k in "AB"}  # "read" | "final" | None
        self.messages: list[dict] = []
        self.on_case = None  # fn(record), called once per resolved case
        self.on_call = None  # fn(twin, record), called after every tool call

    @staticmethod
    def other(k: str) -> str:
        return "B" if k == "A" else "A"

    def done(self, k: str) -> bool:
        return self.phase[k] == "done"

    def finish(self, k: str) -> None:
        with self.cond:
            self.phase[k] = "done"
            self.cond.notify_all()

    def call(self, k: str, tool: str, args: dict, fn) -> str:
        """Run one tool call for twin k with the per-case cap, and report it."""
        q = self.q[k]
        with self.cond:
            n = self.calls[k].get(q, 0) + 1
            self.calls[k][q] = n
        if q >= 0 and n > self.cap and tool not in ("submit_final", "next_case"):
            res = (f"Tool-call limit for case {q + 1} reached: call submit_final now (or "
                   "next_case if the case is resolved).")
        else:
            res = fn()
        if self.on_call is not None:
            self.on_call(k, {"q": q, "tool": tool, "args": args, "result": res,
                             "time": time.time()})
        return res

    def _in_case(self, k: str) -> str | None:
        if self.phase[k] == "done":
            return "The run is over. Stop."
        if self.q[k] < 0:
            return "Error: no current case (call next_case)."
        if self.resolved[self.q[k]]:
            return "Error: this case is resolved; call next_case."
        return None

    def next_case(self, k: str) -> str:
        with self.cond:
            if self.phase[k] == "done":
                return "The run is over. Stop."
            if self.q[k] >= 0 and not self.resolved[self.q[k]]:
                return f"Error: case {self.q[k] + 1} is not resolved yet (submit your final first)."
            nq = self.q[k] + 1
            if nq >= self.T:
                self.phase[k] = "done"
                self.cond.notify_all()
                return "There are no more cases: the run is over. Stop."
            self.want[k] = nq
            self.cond.notify_all()
            o = self.other(k)
            self.cond.wait_for(lambda: self.want[o] >= nq or self.q[o] >= nq or self.done(o))
            self.q[k], self.n_read[k] = nq, 0
            self.n_seen[k] = nq + 1
            lines = [f"Case {nq + 1} of {self.T} (sites are numbered afresh in each case).",
                     "Your facts:", *facts_block(self.cases[nq], k)]
            return "\n".join(lines)

    def submit_opening(self, k: str, args: dict) -> str:
        with self.cond:
            if err := self._in_case(k):
                return err
            q = self.q[k]
            if not np.isnan(self.opening[k][q][0]):
                return "Error: you already submitted your opening for this case."
            p = normalise_probs(args, self.n_sites)
            if isinstance(p, str):
                return p
            self.opening[k][q] = p
            self.cond.notify_all()
            return f"Opening recorded for case {q + 1}: {fmt_probs(p)}."

    def send_message(self, k: str, text: str) -> str:
        with self.cond:
            if err := self._in_case(k):
                return err
            q = self.q[k]
            if np.isnan(self.opening[k][q][0]):
                return "Error: submit your opening first."
            text = str(text)[: self.max_chars]
            self.inbox[self.other(k)].append((q, text))
            self.messages.append({"q": q, "from": k, "text": text, "order": len(self.messages),
                                  "time": time.time()})
            self.cond.notify_all()
            return "Message sent."

    def _unread(self, k: str, q: int) -> list[str]:
        return [t for qq, t in self.inbox[k] if qq == q][self.n_read[k] :]

    def _final_in(self, k: str, q: int) -> bool:
        return not np.isnan(self.final[k][q][0])

    def read_messages(self, k: str) -> str:
        with self.cond:
            if err := self._in_case(k):
                return err
            q, o = self.q[k], self.other(k)
            if np.isnan(self.opening[k][q][0]):
                return "Error: submit your opening first."
            self.waiting[k] = "read"
            self.cond.notify_all()
            self.cond.wait_for(
                lambda: self._unread(k, q)
                or self._final_in(o, q)
                or (self.waiting[o] == "read" and not self._unread(o, q))
                or self.done(o)
                or self.resolved[q]
            )
            self.waiting[k] = None
            new = self._unread(k, q)
            self.n_read[k] += len(new)
            po = self.opening[o][q]
            lines = [f"Partner's opening for case {q + 1}: "
                     + ("not submitted yet." if np.isnan(po[0]) else f"{fmt_probs(po)}.")]
            lines += [f'Partner: "{t}"' for t in new] or ["No new messages."]
            if self._final_in(o, q):
                lines.append("Your partner has submitted its final and is waiting for yours.")
            elif not new and self.waiting[o] == "read":
                lines.append("Your partner is also waiting for a message.")
            return "\n".join(lines)

    def submit_final(self, k: str, args: dict) -> str:
        with self.cond:
            if err := self._in_case(k):
                return err
            q, o = self.q[k], self.other(k)
            if np.isnan(self.opening[k][q][0]):
                return "Error: submit your opening first."
            if self._final_in(k, q):
                return "Error: you already submitted your final for this case."
            p = normalise_probs(args, self.n_sites)
            if isinstance(p, str):
                return p
            self.final[k][q] = p
            self.waiting[k] = "final"
            self.cond.notify_all()
            self.cond.wait_for(lambda: self._final_in(o, q) or self.done(o) or self.resolved[q])
            self.waiting[k] = None
            if not self.resolved[q]:
                self._resolve(q)
            case = self.cases[q]
            pf = self.final[o][q]
            last = q == self.T - 1
            if last:
                self.phase[k] = "done"
                self.cond.notify_all()
            pick = int(np.argmax(self.team[q]))
            return (
                f"Both finals are in. Yours: {fmt_probs(p)}. Your partner's: "
                + ("none (partner stopped)" if np.isnan(pf[0]) else fmt_probs(pf))
                + f". Team choice: Site {pick + 1}. The best site was Site {case.best + 1}"
                + (" (the team chose it)." if pick == case.best else " (the team missed it).")
                + (" That was the last case: the run is over. Stop." if last
                   else " Call next_case for the next case.")
            )

    def _resolve(self, q: int) -> None:
        """Under the lock, once both finals are in (or the partner is gone): team = average of
        the finals that exist."""
        finals = [self.final[k][q] for k in "AB" if self._final_in(k, q)]
        self.team[q] = np.mean(finals, axis=0)
        self.resolved[q] = True
        self.cond.notify_all()
        if self.on_case is not None:
            self.on_case(self.record(q))

    def record(self, q: int) -> dict:
        """Everything about case q, for the transcript."""
        def lst(a: np.ndarray):
            return None if np.isnan(a[0]) else a.tolist()

        team = self.team[q]
        return {
            "t": q,
            "case": self.cases[q].to_dict(),
            "opening": {k: lst(self.opening[k][q]) for k in "AB"},
            "final": {k: lst(self.final[k][q]) for k in "AB"},
            "team": lst(team),
            "team_pick": None if np.isnan(team[0]) else int(np.argmax(team)),
            "messages": [m for m in self.messages if m["q"] == q],
            "tool_calls": {k: self.calls[k].get(q, 0) for k in "AB"},
        }


def build_hp_tools(api: HPSessionAPI, k: str):
    """In-process MCP server with the case tools for twin k, and the allowed tool names (the
    server is registered as "data" by TwinSession._main, hence the mcp__data__ prefix)."""
    from claude_agent_sdk import create_sdk_mcp_server, tool

    obj = {"type": "object", "additionalProperties": False}
    probs = {
        **obj,
        "properties": {
            f"site_{s + 1}": {"type": "number",
                              "description": f"Probability that Site {s + 1} is the best choice."}
            for s in range(api.n_sites)
        },
        "required": [f"site_{s + 1}" for s in range(api.n_sites)],
    }

    async def run(name: str, args: dict, fn) -> dict:
        res = await asyncio.to_thread(api.call, k, name, args, fn)
        return {"content": [{"type": "text", "text": res}]}

    @tool("next_case", TOOL_DOCS["next_case"], {**obj, "properties": {}})
    async def next_case(args):
        return await run("next_case", args, lambda: api.next_case(k))

    @tool("submit_opening", TOOL_DOCS["submit_opening"], probs)
    async def submit_opening(args):
        return await run("submit_opening", args, lambda: api.submit_opening(k, args))

    @tool("send_message", TOOL_DOCS["send_message"],
          {**obj, "properties": {"text": {"type": "string"}}, "required": ["text"]})
    async def send_message(args):
        return await run("send_message", args, lambda: api.send_message(k, args.get("text", "")))

    @tool("read_messages", TOOL_DOCS["read_messages"], {**obj, "properties": {}})
    async def read_messages(args):
        return await run("read_messages", args, lambda: api.read_messages(k))

    @tool("submit_final", TOOL_DOCS["submit_final"], probs)
    async def submit_final(args):
        return await run("submit_final", args, lambda: api.submit_final(k, args))

    tools = [next_case, submit_opening, send_message, read_messages, submit_final]
    server = create_sdk_mcp_server("data", tools=tools)
    allowed = [f"mcp__data__{t.name}" for t in tools]
    return server, allowed


class HPTwinSession(TwinSession):
    """One café twin as one live agent session; reuses `TwinSession`'s SDK loop and live files."""

    def __init__(
        self,
        name: str,
        api: HPSessionAPI,
        model: str = "claude-haiku-5-5",
        effort: str = "low",
        tell_split: bool = True,
        tell_twin: bool = True,
        max_budget_usd: float = 3.0,
        max_continues: int = 10,
        live_dir: str | Path | None = None,
        run: int = 0,
    ) -> None:
        self.name, self.api, self.model, self.effort = name, api, model, effort
        self.max_budget_usd, self.max_continues = max_budget_usd, max_continues
        self.system = SYSTEM_HP.format(
            name=name, T=api.T, n_sites=api.n_sites, max_chars=api.max_chars, cap=api.cap,
            split_line=SPLIT_LINE if tell_split else "",
            twin_line=TWIN_LINE if tell_twin else "",
        ).replace(" \n", "\n")
        self.live = Path(live_dir) if live_dir is not None else None
        self.run_id = run
        self.cost_usd = 0.0
        self.tokens = {"input": 0, "cache_read": 0, "cache_write": 0, "output": 0}
        self.results: list[dict] = []
        self.compactions: list[dict] = []
        self.error: str | None = None
        self.n_continues = 0
        if self.live is not None:
            self.live.mkdir(parents=True, exist_ok=True)
            for kind in ("calls", "messages", "compactions"):
                self._path(kind).write_text("")

    def _tools(self) -> tuple:
        return build_hp_tools(self.api, self.name)

    def _extra_env(self) -> dict:
        return {}

    def _first_prompt(self) -> str:
        return "Begin."

    def _continue_prompt(self) -> str:
        return ("The run is not over yet. Continue working through the tools (next_case, "
                "submit_opening, talking with your partner, submit_final) until a tool result "
                "says the run is over.")
