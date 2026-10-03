from datetime import datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from brokerchat.models import (
    ChatMessage,
    Confidence,
    Instrument,
    InstrumentKind,
    Quote,
    QuoteStatus,
    Unit,
    fmt_decimal,
)
from tests.helpers import make_quote


def test_price_quoted_levels_survive_json_roundtrip():
    q = make_quote(name="CDX_NA_HY", bid="104.5", offer="104.625", unit=Unit.PRICE)
    again = Quote.model_validate_json(q.model_dump_json())
    assert again.offer == Decimal("104.625")
    assert again == q


def test_quote_defaults_to_live_high_confidence():
    q = make_quote()
    assert q.status == QuoteStatus.LIVE
    assert q.confidence == Confidence.HIGH


def test_quote_requires_a_source_message():
    with pytest.raises(ValidationError):
        make_quote(ids=())


def test_chat_message_rejects_naive_timestamp():
    with pytest.raises(ValidationError):
        ChatMessage(id="m1", ts=datetime(2026, 10, 1, 8, 0), room="r", sender="s", text="hi")


def test_instruments_compare_and_hash_by_value():
    a = Instrument(kind=InstrumentKind.INDEX, name="ITRAXX_EUR_MAIN", tenor="5Y")
    b = Instrument(kind=InstrumentKind.INDEX, name="ITRAXX_EUR_MAIN", tenor="5Y")
    assert a == b
    assert len({a, b}) == 1


@pytest.mark.parametrize(
    "value,text",
    [(Decimal("58.50"), "58.5"), (Decimal("60"), "60"), (Decimal("104.625"), "104.625"), (Decimal("0"), "0")],
)
def test_fmt_decimal_drops_trailing_zeros_without_exponent(value, text):
    assert fmt_decimal(value) == text
