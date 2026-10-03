from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from brokerchat.models import ChatMessage, EventType, Instrument, InstrumentKind, Quote, QuoteStatus, Unit

REPO_ROOT = Path(__file__).resolve().parents[1]
T0 = datetime(2026, 10, 1, 8, 0, tzinfo=UTC)


def at(minutes: float) -> datetime:
    return T0 + timedelta(minutes=minutes)


def msg(id: str, text: str, minutes: float = 0, room: str = "ldn-cds-1", sender: str = "jpm_cds") -> ChatMessage:
    return ChatMessage(id=id, ts=at(minutes), room=room, sender=sender, text=text)


def make_quote(
    quote_id: str = "q0001",
    name: str = "ITRAXX_EUR_MAIN",
    tenor: str = "5Y",
    bid: str | None = "58",
    offer: str | None = "58.5",
    source: str = "jpm_cds",
    minutes: float = 0,
    status: QuoteStatus = QuoteStatus.LIVE,
    unit: Unit = Unit.BP,
    ids: tuple[str, ...] = ("m001",),
    kind: InstrumentKind = InstrumentKind.INDEX,
) -> Quote:
    return Quote(
        quote_id=quote_id,
        instrument=Instrument(kind=kind, name=name, tenor=tenor),
        bid=None if bid is None else Decimal(bid),
        offer=None if offer is None else Decimal(offer),
        unit=unit,
        source=source,
        status=status,
        updated_ts=at(minutes),
        source_message_ids=list(ids),
    )


def emit_quote(store, quote: Quote):
    return store.emit(
        EventType.QUOTE_UPSERTED,
        quote.model_dump(mode="json"),
        ts=quote.updated_ts,
        caused_by=quote.source_message_ids[-1],
    )
