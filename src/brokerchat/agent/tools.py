"""Tool definitions for the agent and the executor that validates calls and turns them into events."""

import json
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from brokerchat.models import (
    Axe, ChatMessage, Confidence, Direction, EventType, Flag, FlagKind, Instrument, Interest, Quote, QuoteStatus, Unit,
)
from brokerchat.store import Store


def _nullable(type_: str, description: str) -> dict[str, Any]:
    return {"type": [type_, "null"], "description": description}


def _tool(name: str, description: str, properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        },
    }


_INSTRUMENT = {"type": "string", "description": "Canonical instrument name from lookup_instrument, e.g. ITRAXX_EUR_MAIN"}
_TENOR = {"type": "string", "description": "Tenor such as 5Y or 10Y"}
_DIRECTION = {"type": "string", "enum": [d.value for d in Direction]}
_MESSAGE_IDS = {"type": "array", "items": {"type": "string"}, "description": "Ids of the chat messages this is based on"}

TOOL_SCHEMAS: list[dict[str, Any]] = [
    _tool(
        "lookup_instrument",
        "Resolve an instrument alias as written in chat (e.g. 'Main', 'XO', 'vw') to its canonical name, kind and "
        "quoting unit. Call this before recording anything about an instrument you have not resolved yet.",
        {"text": {"type": "string", "description": "The alias exactly as written in chat"}},
    ),
    _tool(
        "get_quotes",
        "List quotes on the board, optionally filtered by instrument (canonical name or alias) and/or source. "
        "Use this to find the market that a message like 'off that' or 'unch' refers to.",
        {
            "instrument": _nullable("string", "Filter by instrument, or null for all"),
            "source": _nullable("string", "Filter by source (chat sender), or null for all"),
        },
    ),
    _tool(
        "upsert_quote",
        "Create or update the quote for (instrument, tenor, source). There is at most one quote per instrument, "
        "tenor and source: a new market from the same source replaces the old one and makes it live. Pass quote_id "
        "to update a specific quote, or null to match on instrument, tenor and source. bid and offer are levels in "
        "the instrument's unit (bp, or price for price-quoted indices); null means that side is not quoted. Sizes "
        "are in millions; null if not stated.",
        {
            "quote_id": _nullable("string", "Existing quote id, or null"),
            "instrument": _INSTRUMENT,
            "tenor": _TENOR,
            "bid": _nullable("number", "Bid level, or null"),
            "offer": _nullable("number", "Offer level, or null"),
            "bid_size": _nullable("number", "Bid size in millions, or null"),
            "offer_size": _nullable("number", "Offer size in millions, or null"),
            "source": {"type": "string", "description": "The chat sender who made the market"},
            "source_message_ids": _MESSAGE_IDS,
            "confidence": {
                "type": "string",
                "enum": ["high", "low"],
                "description": "low if you had to guess any field; a flag is raised for a human to check",
            },
        },
    ),
    _tool(
        "set_quote_status",
        "Change a quote's status: referred (pulled), traded, or live (confirmed still there).",
        {
            "quote_id": {"type": "string"},
            "status": {"type": "string", "enum": ["live", "referred", "traded"]},
            "message_id": {"type": "string", "description": "Id of the chat message that changed the status"},
        },
    ),
    _tool(
        "record_axe",
        "Record an axe: a participant's stated desire to buy or sell protection.",
        {
            "instrument": _INSTRUMENT,
            "tenor": _TENOR,
            "direction": _DIRECTION,
            "size": _nullable("number", "Size in millions, or null"),
            "source": {"type": "string", "description": "The chat sender who is axed"},
            "source_message_ids": _MESSAGE_IDS,
        },
    ),
    _tool(
        "record_interest",
        "Record a client's interest in buying or selling protection.",
        {
            "client": {"type": "string", "description": "Client name exactly as written"},
            "instrument": _INSTRUMENT,
            "tenor": _TENOR,
            "direction": _DIRECTION,
            "note": _nullable("string", "Any extra detail, or null"),
            "source_message_ids": _MESSAGE_IDS,
        },
    ),
    _tool(
        "raise_flag",
        "Ask a human instead of guessing. Use unknown_instrument for an alias lookup_instrument does not know, and "
        "ambiguous for a market or reference you cannot resolve with confidence.",
        {
            "kind": {"type": "string", "enum": ["ambiguous", "unknown_instrument"]},
            "question": {"type": "string", "description": "A specific question a broker can answer"},
            "message_ids": _MESSAGE_IDS,
        },
    ),
]


class ToolError(Exception):
    pass


class _Input(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LookupInput(_Input):
    text: str


class GetQuotesInput(_Input):
    instrument: str | None = None
    source: str | None = None


class UpsertQuoteInput(_Input):
    quote_id: str | None = None
    instrument: str
    tenor: str
    bid: float | None = None
    offer: float | None = None
    bid_size: float | None = None
    offer_size: float | None = None
    source: str
    source_message_ids: list[str]
    confidence: Confidence = Confidence.HIGH


class SetStatusInput(_Input):
    quote_id: str
    status: Literal["live", "referred", "traded"]
    message_id: str


class RecordAxeInput(_Input):
    instrument: str
    tenor: str
    direction: Direction
    size: float | None = None
    source: str
    source_message_ids: list[str]


class RecordInterestInput(_Input):
    client: str
    instrument: str
    tenor: str
    direction: Direction
    note: str | None = None
    source_message_ids: list[str]


class RaiseFlagInput(_Input):
    kind: Literal["ambiguous", "unknown_instrument"]
    question: str
    message_ids: list[str]


def _dec(x: float | None) -> Decimal | None:
    # str() first so 104.625 becomes Decimal("104.625"), not the binary float's expansion.
    return None if x is None else Decimal(str(x))


class ToolExecutor:
    """Executes one message's tool calls against the store. Every successful write is an event."""

    def __init__(self, store: Store, current: ChatMessage) -> None:
        self.store = store
        self.glossary = store.glossary
        self.current = current
        self.calls = 0

    def execute(self, name: str, tool_input: dict[str, Any], tool_use_id: str) -> dict[str, Any]:
        self.calls += 1
        handler = getattr(self, f"_tool_{name}", None)
        try:
            if handler is None:
                raise ToolError(f"unknown tool {name!r}")
            result = handler(tool_input)
        except (ToolError, ValidationError) as e:
            return {"type": "tool_result", "tool_use_id": tool_use_id, "content": f"Error: {e}", "is_error": True}
        return {"type": "tool_result", "tool_use_id": tool_use_id, "content": json.dumps(result, default=str)}

    def raise_cap_flag(self) -> str:
        return self._flag(
            FlagKind.AMBIGUOUS,
            "The agent hit its tool-call limit on this message; please review it.",
            [self.current.id],
            origin="cap",
        )

    # ---- helpers -------------------------------------------------------------

    def _instrument(self, name: str, tenor: str) -> tuple[Instrument, Unit]:
        info = self.glossary.get(name)
        if info is None:
            raise ToolError(
                f"unknown instrument {name!r}; call lookup_instrument with the alias from chat, "
                "or raise_flag(kind='unknown_instrument')"
            )
        normalized = self.glossary.normalize_tenor(tenor)
        if normalized is None:
            raise ToolError(f"invalid tenor {tenor!r}; expected one of {', '.join(self.glossary.tenors)}")
        return Instrument(kind=info.kind, name=info.canonical, tenor=normalized), info.unit

    def _check_ids(self, ids: list[str]) -> None:
        if not ids:
            raise ToolError("cite at least one chat message id")
        unknown = [i for i in ids if i not in self.store.messages]
        if unknown:
            raise ToolError(f"unknown message ids {unknown}; cite only ids shown in the chat context")

    def _flag(self, kind: FlagKind, question: str, message_ids: list[str], origin: str = "agent") -> str:
        flag = Flag(
            flag_id=self.store.next_flag_id(), kind=kind, question=question, message_ids=message_ids, origin=origin
        )
        self.store.emit(EventType.FLAG_RAISED, flag.model_dump(mode="json"), ts=self.current.ts, caused_by=self.current.id)
        return flag.flag_id

    # ---- tools ---------------------------------------------------------------

    def _tool_lookup_instrument(self, raw: dict[str, Any]) -> dict[str, Any]:
        inp = LookupInput.model_validate(raw)
        info = self.glossary.lookup(inp.text)
        if info is None:
            return {
                "found": False,
                "text": inp.text,
                "hint": "Not a known alias. If it is meant to be an instrument, raise_flag(kind='unknown_instrument').",
            }
        return {"found": True, "canonical": info.canonical, "kind": info.kind.value, "unit": info.unit.value}

    def _tool_get_quotes(self, raw: dict[str, Any]) -> list[dict[str, Any]]:
        inp = GetQuotesInput.model_validate(raw)
        name = None
        if inp.instrument is not None:
            info = self.glossary.get(inp.instrument) or self.glossary.lookup(inp.instrument)
            if info is None:
                raise ToolError(f"unknown instrument {inp.instrument!r}")
            name = info.canonical
        return [
            q.model_dump(mode="json")
            for q in self.store.board.quotes.values()
            if (name is None or q.instrument.name == name) and (inp.source is None or q.source == inp.source)
        ]

    def _tool_upsert_quote(self, raw: dict[str, Any]) -> dict[str, Any]:
        inp = UpsertQuoteInput.model_validate(raw)
        instrument, unit = self._instrument(inp.instrument, inp.tenor)
        self._check_ids(inp.source_message_ids)
        if inp.bid is None and inp.offer is None:
            raise ToolError("a quote needs at least a bid or an offer")

        if inp.quote_id is not None:
            existing = self.store.board.quotes.get(inp.quote_id)
            if existing is None:
                raise ToolError(f"no quote with id {inp.quote_id}")
            if (existing.instrument, existing.source) != (instrument, inp.source):
                raise ToolError(
                    f"quote {inp.quote_id} is {existing.instrument.name} {existing.instrument.tenor} from "
                    f"{existing.source}; pass quote_id null for a different instrument, tenor or source"
                )
        else:
            existing = self.store.board.find_quote(instrument.name, instrument.tenor, inp.source)

        prior_ids = existing.source_message_ids if existing else []
        quote = Quote(
            quote_id=existing.quote_id if existing else self.store.next_quote_id(),
            instrument=instrument,
            bid=_dec(inp.bid),
            offer=_dec(inp.offer),
            unit=unit,
            bid_size=_dec(inp.bid_size),
            offer_size=_dec(inp.offer_size),
            source=inp.source,
            status=QuoteStatus.LIVE,
            updated_ts=self.current.ts,
            source_message_ids=list(dict.fromkeys([*prior_ids, *inp.source_message_ids])),
            confidence=inp.confidence,
        )
        self.store.emit(EventType.QUOTE_UPSERTED, quote.model_dump(mode="json"), ts=self.current.ts, caused_by=self.current.id)
        if inp.confidence == Confidence.LOW:
            self._flag(
                FlagKind.AMBIGUOUS,
                f"Low-confidence read of {instrument.name} {instrument.tenor} from {inp.source}; please check.",
                inp.source_message_ids,
                origin="low_confidence",
            )
        return {"quote_id": quote.quote_id, "created": existing is None}

    def _tool_set_quote_status(self, raw: dict[str, Any]) -> dict[str, Any]:
        inp = SetStatusInput.model_validate(raw)
        if inp.quote_id not in self.store.board.quotes:
            raise ToolError(f"no quote with id {inp.quote_id}")
        self._check_ids([inp.message_id])
        self.store.emit(
            EventType.QUOTE_STATUS_CHANGED,
            {"quote_id": inp.quote_id, "status": inp.status, "message_id": inp.message_id},
            ts=self.current.ts,
            caused_by=self.current.id,
        )
        return {"quote_id": inp.quote_id, "status": inp.status}

    def _tool_record_axe(self, raw: dict[str, Any]) -> dict[str, Any]:
        inp = RecordAxeInput.model_validate(raw)
        instrument, _ = self._instrument(inp.instrument, inp.tenor)
        self._check_ids(inp.source_message_ids)
        axe = Axe(instrument=instrument, direction=inp.direction, size=_dec(inp.size), source=inp.source,
                  source_message_ids=inp.source_message_ids)
        self.store.emit(EventType.AXE_RECORDED, axe.model_dump(mode="json"), ts=self.current.ts, caused_by=self.current.id)
        return {"recorded": True}

    def _tool_record_interest(self, raw: dict[str, Any]) -> dict[str, Any]:
        inp = RecordInterestInput.model_validate(raw)
        instrument, _ = self._instrument(inp.instrument, inp.tenor)
        self._check_ids(inp.source_message_ids)
        interest = Interest(client=inp.client, instrument=instrument, direction=inp.direction, note=inp.note,
                            source_message_ids=inp.source_message_ids)
        self.store.emit(EventType.INTEREST_RECORDED, interest.model_dump(mode="json"), ts=self.current.ts,
                        caused_by=self.current.id)
        return {"recorded": True}

    def _tool_raise_flag(self, raw: dict[str, Any]) -> dict[str, Any]:
        inp = RaiseFlagInput.model_validate(raw)
        self._check_ids(inp.message_ids)
        return {"flag_id": self._flag(FlagKind(inp.kind), inp.question, inp.message_ids)}
