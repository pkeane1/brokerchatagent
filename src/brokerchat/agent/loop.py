"""The agent loop: one chat message in, tool calls out, events on the board (PRD §6).

Hand-written over the Messages API (not the SDK Tool Runner) because owning the loop is the point of the project.
Each chat message starts a fresh, append-only conversation: context in, tool calls and results appended until Claude
stops calling tools.
"""

from typing import Any

import anthropic

from brokerchat.agent.prompt import build_context, build_system_prompt
from brokerchat.agent.tools import TOOL_SCHEMAS, ToolExecutor
from brokerchat.glossary import Glossary
from brokerchat.models import ChatMessage
from brokerchat.pipeline import AgentResult, Usage
from brokerchat.store import Store

MODEL = "claude-opus-5-5"
MAX_TOKENS = 16000
DEFAULT_EFFORT = "medium"
DEFAULT_MAX_ITERATIONS = 8
DEFAULT_CONTEXT_MESSAGES = 20
# Server-side refusal fallback: on a policy decline the API re-runs the request on a suitable fallback model.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class ClaudeAgent:
    def __init__(
        self,
        client: Any,
        glossary: Glossary,
        *,
        model: str = MODEL,
        effort: str = DEFAULT_EFFORT,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        context_messages: int = DEFAULT_CONTEXT_MESSAGES,
    ) -> None:
        self.client = client
        self.model = model
        self.effort = effort
        self.max_iterations = max_iterations
        self.context_messages = context_messages
        # The system prompt never changes during a run, so it is cached across messages.
        self.system = [{"type": "text", "text": build_system_prompt(glossary), "cache_control": {"type": "ephemeral"}}]

    def _call(self, messages: list[dict[str, Any]]) -> Any:
        return self.client.beta.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=self.system,
            tools=TOOL_SCHEMAS,
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort},
            betas=[FALLBACK_BETA],
            fallbacks="default",
            messages=messages,
        )

    def process(self, message: ChatMessage, store: Store) -> AgentResult:
        executor = ToolExecutor(store, message)
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": build_context(store, message, self.context_messages)}
        ]
        usage = Usage()

        def result(status: str, reason: str | None = None) -> AgentResult:
            return AgentResult(message_id=message.id, status=status, reason=reason, tool_calls=executor.calls, usage=usage)

        for _ in range(self.max_iterations):
            try:
                response = self._call(messages)
            except anthropic.APIStatusError as e:  # the SDK has already retried 429s and 5xx
                return result("unprocessed", f"api_error {e.status_code}")
            except anthropic.APIConnectionError:
                return result("unprocessed", "connection_error")

            usage.add(Usage.from_api(response.usage))
            if response.stop_reason in ("refusal", "max_tokens", "pause_turn"):
                return result("unprocessed", response.stop_reason)

            tool_uses = [block for block in response.content if block.type == "tool_use"]
            if not tool_uses:
                return result("ok")

            # Append the whole assistant turn (thinking blocks included), then all tool results in one user turn.
            messages.append({"role": "assistant", "content": response.content})
            messages.append(
                {"role": "user", "content": [executor.execute(b.name, b.input, b.id) for b in tool_uses]}
            )

        executor.raise_cap_flag()
        return result("cap_hit", f"no end_turn after {self.max_iterations} model calls")
