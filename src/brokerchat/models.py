"""Domain types for the broker chat quote board (PRD §5)."""

from decimal import Decimal
from enum import StrEnum
from typing import Any, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field


class InstrumentKind(StrEnum):
    INDEX = "index"
    SINGLE_NAME = "single_name"


class Unit(StrEnum):
    BP = "bp"
    PRICE = "price"


class QuoteStatus(StrEnum):
    LIVE = "live"
    REFERRED = "referred"
    TRADED = "traded"
    STALE = "stale"


class Confidence(StrEnum):
    HIGH = "high"
    LOW = "low"


class Direction(StrEnum):
    BUY_PROTECTION = "buy_protection"
    SELL_PROTECTION = "sell_protection"


class FlagKind(StrEnum):
    AMBIGUOUS = "ambiguous"
    CONFLICT = "conflict"
    UNKNOWN_INSTRUMENT = "unknown_instrument"


class EventType(StrEnum):
    QUOTE_UPSERTED = "quote_upserted"
    QUOTE_STATUS_CHANGED = "quote_status_changed"
    AXE_RECORDED = "axe_recorded"
    INTEREST_RECORDED = "interest_recorded"
    FLAG_RAISED = "flag_raised"
    FLAG_RESOLVED = "flag_resolved"


class ChatMessage(BaseModel):
    id: str
    ts: AwareDatetime
    room: str
    sender: str
    text: str


class Instrument(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: InstrumentKind
    name: str
    tenor: str
    series: int | None = None


class Quote(BaseModel):
    quote_id: str
    instrument: Instrument
    bid: Decimal | None = None
    offer: Decimal | None = None
    unit: Unit
    bid_size: Decimal | None = None
    offer_size: Decimal | None = None
    source: str
    status: QuoteStatus = QuoteStatus.LIVE
    updated_ts: AwareDatetime
    source_message_ids: list[str] = Field(min_length=1)
    confidence: Confidence = Confidence.HIGH


class Axe(BaseModel):
    instrument: Instrument
    direction: Direction
    size: Decimal | None = None
    source: str
    source_message_ids: list[str] = Field(min_length=1)


class Interest(BaseModel):
    client: str
    instrument: Instrument
    direction: Direction
    note: str | None = None
    source_message_ids: list[str] = Field(min_length=1)


class Flag(BaseModel):
    flag_id: str
    kind: FlagKind
    question: str
    message_ids: list[str]
    # agent: an explicit raise_flag; low_confidence: the agent guessed and recorded anyway;
    # cap: the agent hit its call limit; rule: raised by deterministic code.
    origin: Literal["agent", "low_confidence", "cap", "rule"] = "agent"
    resolved: bool = False
    resolution: str | None = None


class Event(BaseModel):
    event_id: str
    ts: AwareDatetime
    type: EventType
    payload: dict[str, Any]
    caused_by_message_id: str | None = None


def fmt_decimal(x: Decimal) -> str:
    """Render a level or size the way a broker would type it: no trailing zeros, no exponent."""
    return format(x.normalize(), "f")
