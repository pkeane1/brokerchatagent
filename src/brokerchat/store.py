"""Append-only event log and the board projection built from it (PRD §5)."""

from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from brokerchat.glossary import Glossary
from brokerchat.models import Axe, ChatMessage, Event, EventType, Flag, Interest, Quote, QuoteStatus


class Board:
    """Current state of the market, derived purely from events."""

    def __init__(self) -> None:
        self.quotes: dict[str, Quote] = {}
        self.axes: list[Axe] = []
        self.interests: list[Interest] = []
        self.flags: dict[str, Flag] = {}

    @classmethod
    def from_events(cls, events: Iterable[Event]) -> "Board":
        board = cls()
        for event in events:
            board.apply(event)
        return board

    def apply(self, event: Event) -> None:
        payload = event.payload
        match event.type:
            case EventType.QUOTE_UPSERTED:
                quote = Quote.model_validate(payload)
                self.quotes[quote.quote_id] = quote
            case EventType.QUOTE_STATUS_CHANGED:
                quote = self.quotes[payload["quote_id"]]
                status = QuoteStatus(payload["status"])
                ids = list(quote.source_message_ids)
                message_id = payload.get("message_id")
                if message_id and message_id not in ids:
                    ids.append(message_id)
                self.quotes[quote.quote_id] = quote.model_copy(
                    update={
                        "status": status,
                        # Going stale is not a market update, so the quote keeps its age.
                        "updated_ts": quote.updated_ts if status == QuoteStatus.STALE else event.ts,
                        "source_message_ids": ids,
                    }
                )
            case EventType.AXE_RECORDED:
                self.axes.append(Axe.model_validate(payload))
            case EventType.INTEREST_RECORDED:
                self.interests.append(Interest.model_validate(payload))
            case EventType.FLAG_RAISED:
                flag = Flag.model_validate(payload)
                self.flags[flag.flag_id] = flag
            case EventType.FLAG_RESOLVED:
                flag = self.flags[payload["flag_id"]]
                self.flags[flag.flag_id] = flag.model_copy(
                    update={"resolved": True, "resolution": payload["resolution"]}
                )

    def find_quote(self, name: str, tenor: str, source: str) -> Quote | None:
        for quote in self.quotes.values():
            if quote.instrument.name == name and quote.instrument.tenor == tenor and quote.source == source:
                return quote
        return None

    def snapshot(self) -> dict[str, Any]:
        return {
            "quotes": [q.model_dump(mode="json") for q in self.quotes.values()],
            "axes": [a.model_dump(mode="json") for a in self.axes],
            "interests": [i.model_dump(mode="json") for i in self.interests],
            "flags": [f.model_dump(mode="json") for f in self.flags.values()],
        }


class Store:
    """Owns the event log, the board projection and the chat messages seen so far."""

    def __init__(self, glossary: Glossary) -> None:
        self.glossary = glossary
        self.events: list[Event] = []
        self.board = Board()
        self.messages: dict[str, ChatMessage] = {}
        self._quote_seq = 0
        self._flag_seq = 0

    def add_message(self, message: ChatMessage) -> None:
        if message.id in self.messages:
            raise ValueError(f"duplicate message id {message.id}")
        self.messages[message.id] = message

    def emit(self, type: EventType, payload: dict[str, Any], ts: datetime, caused_by: str | None = None) -> Event:
        event = Event(
            event_id=f"e{len(self.events) + 1:05d}",
            ts=ts,
            type=type,
            payload=payload,
            caused_by_message_id=caused_by,
        )
        self.board.apply(event)  # apply first: an event the board rejects never enters the log
        self.events.append(event)
        return event

    def next_quote_id(self) -> str:
        self._quote_seq += 1
        return f"q{self._quote_seq:04d}"

    def next_flag_id(self) -> str:
        self._flag_seq += 1
        return f"f{self._flag_seq:04d}"


def write_events(events: Iterable[Event], path: Path) -> None:
    Path(path).write_text("".join(e.model_dump_json() + "\n" for e in events), encoding="utf-8")


def read_events(path: Path) -> list[Event]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [Event.model_validate_json(line) for line in lines if line.strip()]
