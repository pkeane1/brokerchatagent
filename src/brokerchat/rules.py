"""Deterministic board rules that the LLM never decides: staleness and crossed markets."""

from datetime import datetime, timedelta

from brokerchat.models import Event, EventType, Flag, FlagKind, Quote, QuoteStatus, fmt_decimal
from brokerchat.store import Store

DEFAULT_MAX_AGE = timedelta(minutes=30)


def sweep_stale(store: Store, now: datetime, max_age: timedelta = DEFAULT_MAX_AGE) -> list[Event]:
    events = []
    for quote in list(store.board.quotes.values()):
        if quote.status == QuoteStatus.LIVE and now - quote.updated_ts > max_age:
            events.append(
                store.emit(
                    EventType.QUOTE_STATUS_CHANGED,
                    {"quote_id": quote.quote_id, "status": QuoteStatus.STALE.value, "message_id": None},
                    ts=now,
                )
            )
    return events


def _crossed(a: Quote, b: Quote) -> bool:
    return a.bid is not None and b.offer is not None and a.bid >= b.offer


def _market(q: Quote) -> str:
    bid = "-" if q.bid is None else fmt_decimal(q.bid)
    offer = "-" if q.offer is None else fmt_decimal(q.offer)
    return f"{bid}/{offer}"


def detect_conflicts(store: Store, now: datetime) -> list[Event]:
    live = [q for q in store.board.quotes.values() if q.status == QuoteStatus.LIVE]
    events = []
    for i, a in enumerate(live):
        for b in live[i + 1 :]:
            if a.instrument != b.instrument or a.source == b.source:
                continue
            if not (_crossed(a, b) or _crossed(b, a)):
                continue
            low, high = sorted([a, b], key=lambda q: q.quote_id)
            flag_id = f"conflict:{low.quote_id}:{high.quote_id}"
            if flag_id in store.board.flags:
                continue
            flag = Flag(
                flag_id=flag_id,
                kind=FlagKind.CONFLICT,
                question=(
                    f"Crossed market in {a.instrument.name} {a.instrument.tenor}: "
                    f"{low.source} {_market(low)} vs {high.source} {_market(high)}. Which is right?"
                ),
                message_ids=sorted({low.source_message_ids[-1], high.source_message_ids[-1]}),
                origin="rule",
            )
            events.append(store.emit(EventType.FLAG_RAISED, flag.model_dump(mode="json"), ts=now))
    return events
