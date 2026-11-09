"""Backends that turn (system prompt, user prompt, JSON schema) into a parsed JSON reply.

- `AnthropicBackend`: the Anthropic Python SDK with structured outputs (needs ANTHROPIC_API_KEY
  or an `ant auth login` profile).
- `ClaudeCLIBackend`: the Claude Code CLI in print mode (`claude -p --json-schema ...`), which
  runs on the user's Claude Code login; no API key needed.
- `StubBackend`: a Python function, for tests.

Two ways to call a backend:
- `complete(system, user, schema)`: one stateless call; the agent puts everything it should
  remember into the prompt.
- `conversation(system)`: a persistent, append-only conversation; `send(user, schema)` adds a
  user turn and returns the reply, with all earlier turns in context. API: the full message
  history is resent each call (with automatic prompt caching) and assistant content, thinking
  blocks included, is passed back unchanged. CLI: one saved Claude Code session per
  conversation (`--session-id`, then `--resume`).
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# Fields of a failed `claude -p --output-format json` result worth reporting.
ERROR_KEYS = ("subtype", "terminal_reason", "api_error_status", "result", "stop_reason")


@dataclass
class BackendReply:
    """One model reply. `data` is None if the reply could not be parsed against the schema."""

    data: dict | None
    raw: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = float("nan")
    meta: dict = field(default_factory=dict)


class AnthropicBackend:
    """Messages API with `output_config.format` (JSON schema) and an effort level.

    Parameters
    ----------
    model : str
        e.g. "claude-haiku-5-5".
    effort : str
        "low" | "medium" | "high" | ... (thinking depth; Haiku 5.5 defaults to "medium").
    max_tokens : int
        Output cap, including thinking.
    """

    def __init__(self, model: str, effort: str = "low", max_tokens: int = 4000) -> None:
        import anthropic  # optional dependency: pip install -e ".[llm]"

        self.client = anthropic.Anthropic()
        self.model, self.effort, self.max_tokens = model, effort, max_tokens

    def complete(self, system: str, user: str, schema: dict) -> BackendReply:
        return self._parse(self._create(system, [{"role": "user", "content": user}], schema))

    def conversation(self, system: str) -> AnthropicConversation:
        return AnthropicConversation(self, system)

    def _create(self, system: str, messages: list, schema: dict, **kw):
        return self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,
            output_config={
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": schema},
            },
            **kw,
        )

    @staticmethod
    def _parse(resp) -> BackendReply:
        usage = resp.usage
        reply = BackendReply(
            data=None,
            tokens_in=usage.input_tokens
            + (usage.cache_read_input_tokens or 0)
            + (usage.cache_creation_input_tokens or 0),
            tokens_out=usage.output_tokens,
            meta={"stop_reason": resp.stop_reason},
        )
        # A refusal or a truncated reply carries no valid JSON.
        if resp.stop_reason in ("refusal", "max_tokens"):
            return reply
        reply.raw = next((b.text for b in resp.content if b.type == "text"), "")
        try:
            reply.data = json.loads(reply.raw)
        except json.JSONDecodeError:
            pass
        return reply


class AnthropicConversation:
    """Append-only multi-turn conversation over the Messages API (see module docstring)."""

    def __init__(self, backend: AnthropicBackend, system: str) -> None:
        self.backend, self.system = backend, system
        self.messages: list[dict] = []
        self.turns: list[dict] = []  # text-only log for saving

    def send(self, user: str, schema: dict) -> BackendReply:
        self.messages.append({"role": "user", "content": user})
        resp = self.backend._create(
            self.system, self.messages, schema, cache_control={"type": "ephemeral"}
        )
        # Keep the history valid and append-only: pass the content back unchanged (thinking
        # blocks included); an empty reply (e.g. a refusal) still needs an assistant turn.
        content = resp.content or [{"type": "text", "text": "(no reply)"}]
        self.messages.append({"role": "assistant", "content": content})
        reply = AnthropicBackend._parse(resp)
        self.turns += [{"role": "user", "text": user}, {"role": "assistant", "text": reply.raw}]
        return reply


class ClaudeCLIBackend:
    """Claude Code CLI in print mode, isolated from the user's settings, tools and MCP servers.

    Runs in an empty temporary directory with `--setting-sources ""`, no tools, no session
    persistence and our own system prompt, so the model only sees the prompt we build.
    API errors are retried (`retries` times, backoff `backoff * 3**k` seconds); a reply without
    valid structured output is returned with `data=None`; not being logged in raises.
    """

    def __init__(
        self,
        model: str,
        effort: str = "low",
        timeout: float = 300.0,
        executable: str = "claude",
        retries: int = 3,
        backoff: float = 5.0,
    ) -> None:
        self.model, self.effort, self.timeout, self.executable = model, effort, timeout, executable
        self.retries, self.backoff = retries, backoff
        self._cwd = tempfile.mkdtemp(prefix="consensus-llm-")

    def complete(self, system: str, user: str, schema: dict) -> BackendReply:
        cmd = self._cmd(system, schema, ["--no-session-persistence"])
        return self._call(lambda: cmd, user)

    def conversation(self, system: str) -> CLIConversation:
        return CLIConversation(self, system)

    def _cmd(self, system: str, schema: dict, session_flags: list[str]) -> list[str]:
        return [
            self.executable,
            "-p",
            "--model",
            self.model,
            "--effort",
            self.effort,
            "--system-prompt",
            system,
            "--tools",
            "",
            "--strict-mcp-config",
            "--setting-sources",
            "",
            *session_flags,
            "--json-schema",
            json.dumps(schema),
            "--output-format",
            "json",
        ]

    def _call(self, make_cmd: Callable[[], list[str]], user: str, on_output=None) -> BackendReply:
        """Run `make_cmd()` with retries on API errors; `on_output(out)` sees every attempt."""
        for attempt in range(self.retries + 1):
            out = self._run(make_cmd(), user)
            if on_output is not None:
                on_output(out)
            if not out.get("is_error"):
                break
            info = {k: out.get(k) for k in ERROR_KEYS}
            if "logged in" in str(out.get("result", "")).lower():
                raise RuntimeError(f"claude CLI is not logged in: {info}")
            if out.get("terminal_reason") != "api_error" and out.get("api_error_status") is None:
                # The model ran but gave no valid structured reply: an unusable reply, not a crash.
                log.warning("claude CLI reply unusable: %s", info)
                return self._reply(out, data=None, meta=info)
            if attempt == self.retries:
                raise RuntimeError(f"claude CLI API error after {attempt + 1} attempts: {info}")
            log.warning("claude CLI API error (attempt %d), retrying: %s", attempt + 1, info)
            time.sleep(self.backoff * 3**attempt)
        return self._reply(out, data=out.get("structured_output"))

    def _run(self, cmd: list[str], user: str) -> dict:
        proc = subprocess.run(
            cmd, input=user, capture_output=True, text=True, cwd=self._cwd, timeout=self.timeout
        )
        try:
            return json.loads(proc.stdout)
        except json.JSONDecodeError as err:
            raise RuntimeError(
                f"claude CLI gave no JSON: {proc.stdout[:300]} {proc.stderr[:300]}"
            ) from err

    @staticmethod
    def _reply(out: dict, data: dict | None, meta: dict | None = None) -> BackendReply:
        usage = out.get("usage", {})
        return BackendReply(
            data=data,
            raw=str(out.get("result", "")),
            tokens_in=usage.get("input_tokens", 0)
            + usage.get("cache_read_input_tokens", 0)
            + usage.get("cache_creation_input_tokens", 0),
            tokens_out=usage.get("output_tokens", 0),
            cost_usd=float(out.get("total_cost_usd", float("nan"))),
            meta=meta or {"stop_reason": out.get("stop_reason")},
        )


class CLIConversation:
    """One saved Claude Code session: `--session-id <uuid>` on the first call, `--resume` after.

    Sessions are stored by the CLI under ~/.claude/projects/ (keyed by the backend's temporary
    working directory), so a run leaves one session file per twin there.
    """

    def __init__(self, backend: ClaudeCLIBackend, system: str) -> None:
        self.backend, self.system = backend, system
        self.session_id = str(uuid.uuid4())
        self.started = False
        self.turns: list[dict] = []

    def send(self, user: str, schema: dict) -> BackendReply:
        def make_cmd():
            flag = "--resume" if self.started else "--session-id"
            return self.backend._cmd(self.system, schema, [flag, self.session_id])

        def on_output(out):
            # Once the CLI reports our session id, the session exists: resume from now on.
            if out.get("session_id") == self.session_id:
                self.started = True

        reply = self.backend._call(make_cmd, user, on_output)
        self.turns += [{"role": "user", "text": user}, {"role": "assistant", "text": reply.raw}]
        return reply


class StubBackend:
    """Calls `fn(system, user, schema) -> dict | None`; for tests and dry runs."""

    def __init__(self, fn: Callable[[str, str, dict], dict | None]) -> None:
        self.fn = fn
        self.calls: list[tuple[str, str]] = []
        self.conversations: list[StubConversation] = []

    def complete(self, system: str, user: str, schema: dict) -> BackendReply:
        self.calls.append((system, user))
        data = self.fn(system, user, schema)
        return BackendReply(data=data, raw=json.dumps(data), cost_usd=0.0)

    def conversation(self, system: str) -> StubConversation:
        conv = StubConversation(self, system)
        self.conversations.append(conv)
        return conv


class StubConversation:
    """Records turns; replies come from the backend's `fn(system, user, schema)`."""

    def __init__(self, backend: StubBackend, system: str) -> None:
        self.backend, self.system = backend, system
        self.turns: list[dict] = []

    def send(self, user: str, schema: dict) -> BackendReply:
        reply = self.backend.complete(self.system, user, schema)
        self.turns += [{"role": "user", "text": user}, {"role": "assistant", "text": reply.raw}]
        return reply


def build_backend(cfg: dict):
    """Backend from a config block like {kind: cli, model: claude-haiku-5-5, effort: low}."""
    cfg = dict(cfg)
    kind = cfg.pop("kind")
    registry = {"api": AnthropicBackend, "cli": ClaudeCLIBackend}
    if kind not in registry:
        raise ValueError(f"Unknown backend {kind!r}; choose from {sorted(registry)}")
    return registry[kind](**cfg)
