"""Feeds chat messages through an agent, then applies the deterministic rules."""

import json
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Protocol

from pydantic import BaseModel, Field

from brokerchat.models import ChatMessage
from brokerchat.rules import DEFAULT_MAX_AGE, detect_conflicts, sweep_stale
from brokerchat.store import Store, write_events

# claude-opus-5-5 list prices in USD per million tokens (2026-09). Cache writes are 1.25x input (5-minute TTL).
PRICE_PER_MTOK = {"input": 4.00, "output": 20.00, "cache_read": 0.20, "cache_write": 5.00}


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    @classmethod
    def from_api(cls, usage: Any) -> "Usage":
        return cls(
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
        )

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_input_tokens += other.cache_read_input_tokens
        self.cache_creation_input_tokens += other.cache_creation_input_tokens

    def cost_usd(self) -> float:
        return (
            self.input_tokens * PRICE_PER_MTOK["input"]
            + self.output_tokens * PRICE_PER_MTOK["output"]
            + self.cache_read_input_tokens * PRICE_PER_MTOK["cache_read"]
            + self.cache_creation_input_tokens * PRICE_PER_MTOK["cache_write"]
        ) / 1_000_000


class AgentResult(BaseModel):
    message_id: str
    status: Literal["ok", "unprocessed", "cap_hit"]
    reason: str | None = None
    tool_calls: int = 0
    usage: Usage = Field(default_factory=Usage)


class Processor(Protocol):
    def process(self, message: ChatMessage, store: Store) -> AgentResult: ...


class NullAgent:
    """Does nothing. Used to test the pipeline and eval harness without calling the API."""

    def process(self, message: ChatMessage, store: Store) -> AgentResult:
        return AgentResult(message_id=message.id, status="ok")


class Pipeline:
    def __init__(self, store: Store, agent: Processor, max_age: timedelta = DEFAULT_MAX_AGE) -> None:
        self.store = store
        self.agent = agent
        self.max_age = max_age
        self.results: list[AgentResult] = []

    def step(self, message: ChatMessage) -> AgentResult:
        self.store.add_message(message)
        result = self.agent.process(message, self.store)
        sweep_stale(self.store, message.ts, self.max_age)
        detect_conflicts(self.store, message.ts)
        self.results.append(result)
        return result

    def run(
        self,
        messages: Iterable[ChatMessage],
        on_step: Callable[[ChatMessage, AgentResult], None] | None = None,
    ) -> list[AgentResult]:
        for message in messages:
            result = self.step(message)
            if on_step is not None:
                on_step(message, result)
        return self.results

    def total_usage(self) -> Usage:
        total = Usage()
        for result in self.results:
            total.add(result.usage)
        return total


def write_run_outputs(store: Store, results: list[AgentResult], out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "board.json").write_text(json.dumps(store.board.snapshot(), indent=2), encoding="utf-8")
    write_events(store.events, out_dir / "events.jsonl")
    (out_dir / "results.jsonl").write_text("".join(r.model_dump_json() + "\n" for r in results), encoding="utf-8")
