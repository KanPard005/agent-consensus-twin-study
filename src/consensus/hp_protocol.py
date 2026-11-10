"""Protocols for the hidden-profile task: two LLM twins discussing each case (`HPDiscussion`),
and one LLM that sees every clue (`SoloFull`, the full-information reference).

Both take the same list of cases and return per-case metrics; the transcripts are kept on the
object for the runner to save. Metrics per case (arrays of length T):

- correct: the (team) pick is the pooled best site.
- logscore: log of the (team) probability on the best site, floored at 0.01.
- r1_correct_A / r1_correct_B: the twin's round-1 favourite is the best site (before talking).
- agree: the twins' final favourites are the same site.
- defer_A / defer_B: change from round 1 to the final round in the twin's probability on the
  partner's round-1 favourite, when their round-1 favourites differ (NaN otherwise). This is the
  verbal-task analogue of the weight of advice.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from consensus.hp_agent import (
    SOLO_SYSTEM,
    CaseRecord,
    HPTwin,
    facts_block,
    parse_probs,
    probs_schema,
)
from consensus.hp_data import Case
from consensus.llm_backends import build_backend

LOG_FLOOR = 0.01


def _cost(replies) -> dict:
    return {
        "tokens_in": sum(r.tokens_in for r in replies),
        "tokens_out": sum(r.tokens_out for r in replies),
        "cost_usd": float(np.nansum([r.cost_usd for r in replies])),
    }


class HPDiscussion:
    """Two `HPTwin`s discuss each case for `n_exchanges` + 1 rounds; the team picks the argmax
    of the average of their final-round probabilities (ties: lowest site number).

    Parameters
    ----------
    backend : dict
        Backend config (`consensus.llm_backends.build_backend`).
    n_exchanges : int
        Rounds after the first.
    **twin_kw
        Passed to `HPTwin` (max_chars, notes_chars, history_len, tell_split, tell_twin).
    """

    name = "hp_talk"

    def __init__(self, backend: dict, n_exchanges: int = 2, **twin_kw) -> None:
        self.backend_cfg, self.E, self.twin_kw = backend, n_exchanges, twin_kw
        self.transcript: list[dict] = []

    def run(self, cases: list[Case]) -> dict[str, np.ndarray]:
        backend = build_backend(self.backend_cfg)
        n_sites, T, R = cases[0].n_sites, len(cases), self.E + 1
        twins = {n: HPTwin(n, backend, n_sites, R, T, **self.twin_kw) for n in "AB"}
        self.twins, self.transcript = twins, []
        m = {
            k: np.full(T, np.nan)
            for k in (
                "correct",
                "logscore",
                "r1_correct_A",
                "r1_correct_B",
                "agree",
                "defer_A",
                "defer_B",
                "parse_fail",
            )
        }
        replies = []
        with ThreadPoolExecutor(max_workers=2) as pool:
            for t, case in enumerate(cases):
                sent: dict[str, list[tuple[str, np.ndarray]]] = {"A": [], "B": []}
                rounds, prev = [], {"A": None, "B": None}
                fails = 0
                for rnd in range(R):
                    futs = {
                        n: pool.submit(
                            twins[n].act, t, rnd, case, sent["B" if n == "A" else "A"], prev[n]
                        )
                        for n in "AB"
                    }
                    out = {n: f.result() for n, f in futs.items()}
                    for n, r in out.items():
                        sent[n].append((r.message, r.probs))
                        prev[n] = r.probs
                        replies += r.replies
                        fails += not r.ok
                    rounds.append(
                        {
                            n: {
                                "message": r.message,
                                "probs": r.probs.tolist(),
                                "ok": r.ok,
                                "notes": twins[n].notes,
                            }
                            for n, r in out.items()
                        }
                    )
                team = (prev["A"] + prev["B"]) / 2
                pick, best = int(np.argmax(team)), case.best
                first = {n: np.array(rounds[0][n]["probs"]) for n in "AB"}
                fav1 = {n: int(np.argmax(first[n])) for n in "AB"}
                m["correct"][t] = pick == best
                m["logscore"][t] = np.log(max(team[best], LOG_FLOOR))
                m["agree"][t] = np.argmax(prev["A"]) == np.argmax(prev["B"])
                m["parse_fail"][t] = fails
                for n, o in (("A", "B"), ("B", "A")):
                    m[f"r1_correct_{n}"][t] = fav1[n] == best
                    if fav1[n] != fav1[o]:
                        m[f"defer_{n}"][t] = prev[n][fav1[o]] - first[n][fav1[o]]
                    twins[n].record(CaseRecord(t, fav1[n], fav1[o], pick, best))
                self.transcript.append(
                    {
                        "t": t,
                        "case": case.to_dict(),
                        "rounds": rounds,
                        "team": team.tolist(),
                        "team_pick": pick,
                    }
                )
        self.cost = _cost(replies)
        return m


class SoloFull:
    """One LLM sees every clue of a case (no partner, no memory) and gives its probabilities.

    It measures how far the model's reading of the clues agrees with their hidden values: its
    accuracy is the ceiling for what the twins can reach by pooling everything.
    """

    name = "solo_full"

    def __init__(self, backend: dict) -> None:
        self.backend_cfg = backend
        self.transcript: list[dict] = []

    def run(self, cases: list[Case]) -> dict[str, np.ndarray]:
        backend = build_backend(self.backend_cfg)
        n_sites = cases[0].n_sites
        system = SOLO_SYSTEM.format(n_sites=n_sites)
        schema = probs_schema(n_sites, message=False, notes=False)
        T = len(cases)
        m = {k: np.full(T, np.nan) for k in ("correct", "logscore", "parse_fail")}
        self.transcript, replies = [], []

        def ask(case: Case):
            user = "\n".join(
                ["Facts:", *facts_block(case, None), "Reply with your probability for each site."]
            )
            return backend.complete(system, user, schema)

        with ThreadPoolExecutor(max_workers=4) as pool:
            outs = list(pool.map(ask, cases))
        for t, (case, reply) in enumerate(zip(cases, outs)):
            replies.append(reply)
            p = parse_probs(reply.data, n_sites)
            m["parse_fail"][t] = p is None
            if p is None:
                p = np.full(n_sites, 1 / n_sites)
            m["correct"][t] = int(np.argmax(p)) == case.best
            m["logscore"][t] = np.log(max(p[case.best], LOG_FLOOR))
            self.transcript.append({"t": t, "case": case.to_dict(), "probs": p.tolist()})
        self.cost = _cost(replies)
        return m


def build_hp_protocol(cfg: dict):
    """Protocol from a config entry like {name: hp_talk, n_exchanges: 2, ...}."""
    cfg = dict(cfg)
    name = cfg.pop("name")
    cfg.pop("label", None)
    registry = {"hp_talk": HPDiscussion, "solo_full": SoloFull}
    if name not in registry:
        raise ValueError(f"Unknown protocol {name!r}; choose from {sorted(registry)}")
    return registry[name](**cfg)
