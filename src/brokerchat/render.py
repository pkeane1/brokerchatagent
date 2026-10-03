"""Terminal rendering of the board with rich."""

from decimal import Decimal

from rich.console import Group
from rich.table import Table

from brokerchat.models import QuoteStatus, fmt_decimal
from brokerchat.store import Board

STATUS_STYLE = {
    QuoteStatus.LIVE: "green",
    QuoteStatus.REFERRED: "yellow",
    QuoteStatus.TRADED: "cyan",
    QuoteStatus.STALE: "dim",
}


def _num(x: Decimal | None) -> str:
    return "-" if x is None else fmt_decimal(x)


def render_board(board: Board) -> Group:
    quotes = Table(title="Quotes", expand=True)
    for column in ("ID", "Instrument", "Tenor", "Bid", "Offer", "Size", "Source", "Status", "Updated"):
        quotes.add_column(column)
    for q in sorted(board.quotes.values(), key=lambda q: (q.instrument.name, q.instrument.tenor, q.source)):
        size = "" if q.bid_size is None and q.offer_size is None else f"{_num(q.bid_size)}x{_num(q.offer_size)}"
        quotes.add_row(
            q.quote_id,
            q.instrument.name,
            q.instrument.tenor,
            _num(q.bid),
            _num(q.offer),
            size,
            q.source,
            q.status.value,
            q.updated_ts.strftime("%H:%M:%S"),
            style=STATUS_STYLE[q.status],
        )

    axes = Table(title="Axes", expand=True)
    for column in ("Instrument", "Tenor", "Direction", "Size", "Source"):
        axes.add_column(column)
    for a in board.axes:
        axes.add_row(a.instrument.name, a.instrument.tenor, a.direction.value, _num(a.size), a.source)

    interests = Table(title="Client interest", expand=True)
    for column in ("Client", "Instrument", "Tenor", "Direction", "Note"):
        interests.add_column(column)
    for i in board.interests:
        interests.add_row(i.client, i.instrument.name, i.instrument.tenor, i.direction.value, i.note or "")

    flags = Table(title="Open flags", expand=True)
    for column in ("ID", "Kind", "Question", "Messages"):
        flags.add_column(column)
    for f in board.flags.values():
        if not f.resolved:
            flags.add_row(f.flag_id, f.kind.value, f.question, ", ".join(f.message_ids))

    return Group(quotes, axes, interests, flags)
