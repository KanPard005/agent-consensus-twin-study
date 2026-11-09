# agent-consensus

Research code for the LogBook project `agent-consensus`
(~/Documents/Research/LogBook/projects/agent-consensus/index.md): a "twin study" of how two
identical learners that saw different features reach a consensus on a joint prediction.

## Model

$y = \langle x^A, w^A\rangle + \langle x^B, w^B\rangle + \epsilon$, $x=(x^A,x^B)\sim N(0,\Sigma)$,
with cross-correlation $\rho$ between paired coordinates of $x^A$ and $x^B$.

1. **Training** ($t \le n$): each twin fits online ridge on its own iid stream $(x^{(i)}_t, y_t)$.
2. **Communication** ($t > n$): each step the twins see $x^A_t, x^B_t$, produce one joint
   prediction, then $y_t$ is revealed to both.

| Label | Protocol (`configs`) | What it does |
|---|---|---|
| P0 | `no_talk` | solo predictions, averaged (floor) |
| P1 | `blend` | twins frozen; $\alpha m^A + \beta m^B$ learned online |
| P2 | `exchange`, `n_exchanges: 1` | each twin regresses $y$ on (own $x$, partner's prediction) |
| P3 | `exchange`, `n_exchanges: E>1` | E rounds of simultaneous announcements (learned Geanakoplos–Polemarchakis) |
| P4 | `llm_exchange` | LLM twins (Claude Haiku 5.5) with the trained linear learner as a tool; they choose messages, estimates and how to combine |

### LLM twins (P4)

Each twin is an LLM (`src/consensus/llm_agent.py`) that, every step, is shown its own features,
its frozen linear learner's prediction, the partner's messages so far this step, its own
private notes (rewritten every round, kept across steps) and a table of the last few steps
(own model, own final, partner final, team, true $y$). It replies in JSON with a message
(≤ 300 chars), an estimate and updated notes; the team prediction is the average of the
final-round estimates, as in P2/P3. Both twins get the same system prompt, which says the
partner is another instance of the same model (`tell_twin`). Transcripts are saved to
`results/<run>/transcripts/<label>_run<r>.jsonl`.

Options (all off in `llm_haiku.yaml`; one variant config each):
`team_rule: joint` asks each twin to propose the team's joint estimate instead of its own
(`llm_haiku_joint.yaml`, P5); `tools: true` gives a regression tool: the reply's `regress`
field lists predictors (`x1..xd`, `own_model`, `partner_open`), the runner fits least squares of
$y$ on them over the twin's history and asks again with coefficients, in-sample / leave-one-out
MSE and the prediction for the current case (`llm_haiku_tool.yaml`, P6; with
`show_partner_estimates: true` so partner estimates are visible); both together in
`llm_haiku_tool_joint.yaml` (P7). All configs share seed and streams with `llm_haiku.yaml`.

Memory (`memory`): by default (`notes`) every round is a fresh, stateless call whose prompt
carries the twin's notes and the last 10 steps. With `memory: persistent` each twin is one
continuing conversation for the whole run (a new one per run/seed): each round appends a short
user turn (previous outcome, new case, partner's latest message), with no notes or history
table. API backend: append-only message history with automatic prompt caching; CLI backend: one
saved Claude Code session per twin (`--session-id`, then `--resume`; session files land under
`~/.claude/projects/`). Full conversations are saved as
`transcripts/<label>_run<r>_conversation_<A|B>.jsonl`. Configs: `llm_haiku_persistent.yaml`
(P10, = P4 + persistent) and `llm_haiku_persistent_auto.yaml` (P11, = P8 + persistent).

Backends (`src/consensus/llm_backends.py`, `backend.kind` in the config):
`cli` runs `claude -p` (uses your Claude Code login, no key; isolated from settings, tools and
MCP); `api` uses the Anthropic SDK (`pip install -e ".[llm]"`, needs `ANTHROPIC_API_KEY`). Each
step costs 2 × (n_exchanges + 1) calls, so `configs/llm_haiku.yaml` uses T = 60 and 3 runs;
LLM replies are not seeded, so runs are not exactly reproducible.

Population limits of P0–P3 (no closed form for P4) are in `src/consensus/theory.py` (dashed lines in the plots).
Two facts worth knowing: the blend (P1) is exactly optimal if $d_A=d_B=1$ or $\rho=0$, so
the interesting regime is $d\ge 2$ with $\rho\ne0$; and in the population, the exchange
protocol reaches full information after a few rounds.

"Who moves" is measured by the **opinion shift** `shift_A` $=(\hat y^A_{\text{final}}-\hat y^A_{\text{solo}})^2$
(prediction space, works for any `Agent`; population limit in `benchmarks.json`). The older
`drift_A` (norm of the change in own-feature weights) is kept as a diagnostic only: it depends on
the parameterization, since messages are partly functions of the agent's own features.

## Layout

```
src/consensus/
  data.py       LinearTwinEnv (environment, sampling)
  models.py     OnlineRidge, Agent interface, LinearAgent, Blender
  protocols.py  NoTalk (P0), FrozenBlend (P1), Exchange (P2/P3), LLMExchange (P4)
  llm_agent.py  LLMTwin: prompt, notes, history, reply parsing
  llm_backends.py  claude CLI / Anthropic SDK / stub backends
  theory.py     closed forms: solo / Schur complement, optimal blend, agreement path
  plotting.py   figure functions (take saved results, return (fig, ax))
  utils.py      config, seeding, run folders, I/O
scripts/run.py  runner;  scripts/plot.py  regenerate figures from a run folder
configs/        default.yaml (symmetric), asymmetric_data.yaml (n_A << n_B), llm_haiku.yaml (P4)
tests/          closed forms, RLS vs batch ridge, reproducibility
```

The `Agent` interface (`predict(x, messages)`, `update(x, messages, y)`,
`begin_communication(...)`) is all that protocols use, so an LLM-backed twin can be added
as another `Agent` subclass.

## Usage

```bash
pip install -e ".[dev]"
pytest
python scripts/run.py configs/default.yaml --plot
python scripts/run.py configs/default.yaml --set env.rho=0.8 train.n_A=50 --plot
python scripts/plot.py results/default/<timestamp>
```

Each run writes `results/<experiment>/<timestamp>/` with the resolved config + git commit,
`metrics.npz` (per protocol and metric, shape `(n_runs, T)`), `benchmarks.json`
(population limits), `summary.json` and `figures/`.
