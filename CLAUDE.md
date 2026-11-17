# agent-consensus

Research code for the LogBook project `agent-consensus`
(~/Documents/Research/LogBook/projects/agent-consensus/index.md).

@~/Documents/Research/LogBook/claude/code-style.md

## AI authorship

When committing changes that you wrote entirely and I did not edit, add the trailer
`AI-Written: true` to the commit message. Never add it to commits that contain my edits.

## Logging

After a meaningful result or at the end of a session, run `/logbook agent-consensus`.

## Running and handoff

- Python: the venv in the main checkout, from any worktree:
  `PYTHONPATH=src ../agent-consensus/.venv/bin/python ...`; tests `python -m pytest -q`.
- Live twins are Claude Agent SDK sessions on the logged-in Claude account (no API key). Pilot
  with one pair and `T=10` before a grid; grids go through `scripts/launch_related.py`
  (resumable, `--set` for overrides). `results/` is not in git.
- Keep prompts and tools neutral; any nudge is a config flag that defaults to off and is
  described as a manipulation in the log entry.
- Current state, experiment index, data formats and next steps for the related-targets / long-run
  work: LogBook note `projects/agent-consensus/notes/2026-11-17-handoff-related-targets-and-long-runs.md`.
