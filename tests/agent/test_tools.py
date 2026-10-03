import json
from decimal import Decimal

import pytest

from brokerchat.agent.tools import TOOL_SCHEMAS, ToolExecutor
from brokerchat.models import Direction, FlagKind, QuoteStatus, Unit
from brokerchat.store import Store
from tests.helpers import msg


@pytest.fixture
def store(glossary):
    s = Store(glossary)
    for m in [msg("m001", "Main 5y 58/58.5 10x10", 0), msg("m002", "refer", 1), msg("m003", "Main 5y 59/59.5", 2)]:
        s.add_message(m)
    return s


def run(store, name, tool_input, current="m001"):
    return ToolExecutor(store, store.messages[current]).execute(name, tool_input, "tu_1")


def ok(result):
    assert result["tool_use_id"] == "tu_1"
    assert "is_error" not in result, result["content"]
    return json.loads(result["content"])


def upsert(**overrides):
    fields = dict(quote_id=None, instrument="ITRAXX_EUR_MAIN", tenor="5Y", bid=58, offer=58.5, bid_size=10,
                  offer_size=10, source="jpm_cds", source_message_ids=["m001"], confidence="high")
    fields.update(overrides)
    return fields


def test_every_schema_is_strict_with_all_properties_required():
    for tool in TOOL_SCHEMAS:
        schema = tool["input_schema"]
        assert tool["strict"] is True
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"]), tool["name"]


def test_every_schema_has_a_handler():
    for tool in TOOL_SCHEMAS:
        assert hasattr(ToolExecutor, f"_tool_{tool['name']}"), tool["name"]


def test_lookup_instrument(store):
    assert ok(run(store, "lookup_instrument", {"text": "itrx main"})) == {
        "found": True, "canonical": "ITRAXX_EUR_MAIN", "kind": "index", "unit": "bp"
    }
    assert ok(run(store, "lookup_instrument", {"text": "ZXQ"}))["found"] is False


def test_upsert_creates_quote(store):
    assert ok(run(store, "upsert_quote", upsert())) == {"quote_id": "q0001", "created": True}
    q = store.board.quotes["q0001"]
    assert (q.bid, q.offer, q.bid_size, q.unit) == (Decimal("58"), Decimal("58.5"), Decimal("10"), Unit.BP)
    assert q.updated_ts == store.messages["m001"].ts
    assert store.events[-1].caused_by_message_id == "m001"


def test_requote_after_refer_updates_same_quote_and_goes_live(store):
    ok(run(store, "upsert_quote", upsert()))
    ok(run(store, "set_quote_status", {"quote_id": "q0001", "status": "referred", "message_id": "m002"}, "m002"))
    result = ok(run(store, "upsert_quote", upsert(bid=59, offer=59.5, source_message_ids=["m003"]), "m003"))
    assert result == {"quote_id": "q0001", "created": False}
    assert len(store.board.quotes) == 1
    q = store.board.quotes["q0001"]
    assert q.status == QuoteStatus.LIVE
    assert q.source_message_ids == ["m001", "m002", "m003"]


def test_price_level_from_float_is_exact(store):
    ok(run(store, "upsert_quote", upsert(instrument="CDX_NA_HY", bid=104.5, offer=104.625, bid_size=None,
                                         offer_size=None, source="citi_ix")))
    q = store.board.quotes["q0001"]
    assert q.offer == Decimal("104.625")
    assert q.unit == Unit.PRICE


def test_tenor_is_normalised(store):
    ok(run(store, "upsert_quote", upsert(tenor="5y")))
    assert store.board.quotes["q0001"].instrument.tenor == "5Y"


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({"instrument": "Main"}, "lookup_instrument"),
        ({"tenor": "6Y"}, "invalid tenor"),
        ({"source_message_ids": []}, "at least one"),
        ({"source_message_ids": ["m999"]}, "unknown message ids"),
        ({"bid": None, "offer": None}, "at least a bid or an offer"),
        ({"quote_id": "q0042"}, "no quote with id"),
        ({"surprise": 1}, "Extra inputs"),
    ],
)
def test_upsert_rejects_bad_input_without_touching_board(store, overrides, error):
    result = run(store, "upsert_quote", upsert(**overrides))
    assert result["is_error"] is True
    assert error in result["content"]
    assert store.events == []


def test_upsert_with_quote_id_for_other_instrument_rejected(store):
    ok(run(store, "upsert_quote", upsert()))
    result = run(store, "upsert_quote", upsert(quote_id="q0001", instrument="CDX_NA_IG"))
    assert result["is_error"] is True


def test_low_confidence_raises_flag(store):
    ok(run(store, "upsert_quote", upsert(confidence="low")))
    flag = next(iter(store.board.flags.values()))
    assert flag.kind == FlagKind.AMBIGUOUS
    assert flag.message_ids == ["m001"]


def test_set_status_traded(store):
    ok(run(store, "upsert_quote", upsert()))
    assert ok(run(store, "set_quote_status", {"quote_id": "q0001", "status": "traded", "message_id": "m002"}, "m002")) == {
        "quote_id": "q0001", "status": "traded"
    }
    assert store.board.quotes["q0001"].status == QuoteStatus.TRADED


def test_set_status_rejects_unknown_quote_and_stale(store):
    assert run(store, "set_quote_status", {"quote_id": "q0009", "status": "traded", "message_id": "m001"})["is_error"]
    ok(run(store, "upsert_quote", upsert()))
    assert run(store, "set_quote_status", {"quote_id": "q0001", "status": "stale", "message_id": "m001"})["is_error"]


def test_record_axe_and_interest(store):
    ok(run(store, "record_axe", {"instrument": "ITRAXX_EUR_XOVER", "tenor": "5Y", "direction": "buy_protection",
                                 "size": 25, "source": "bnp_cds", "source_message_ids": ["m001"]}))
    ok(run(store, "record_interest", {"client": "Alder Capital", "instrument": "VOLKSWAGEN_AG", "tenor": "5Y",
                                      "direction": "sell_protection", "note": None, "source_message_ids": ["m001"]}))
    assert store.board.axes[0].size == Decimal("25")
    assert store.board.interests[0].direction == Direction.SELL_PROTECTION


def test_get_quotes_filters(store):
    ok(run(store, "upsert_quote", upsert()))
    ok(run(store, "upsert_quote", upsert(instrument="CDX_NA_IG", source="bnp_cds")))
    assert len(ok(run(store, "get_quotes", {"instrument": None, "source": None}))) == 2
    rows = ok(run(store, "get_quotes", {"instrument": "main", "source": None}))
    assert [r["quote_id"] for r in rows] == ["q0001"]
    assert [r["quote_id"] for r in ok(run(store, "get_quotes", {"instrument": None, "source": "bnp_cds"}))] == ["q0002"]


def test_raise_flag(store):
    assert ok(run(store, "raise_flag", {"kind": "unknown_instrument", "question": "What is ZXQ?",
                                        "message_ids": ["m001"]})) == {"flag_id": "f0001"}
    assert run(store, "raise_flag", {"kind": "conflict", "question": "?", "message_ids": ["m001"]})["is_error"]


def test_unknown_tool(store):
    assert run(store, "delete_everything", {})["is_error"] is True


def test_cap_flag(store):
    executor = ToolExecutor(store, store.messages["m003"])
    flag_id = executor.raise_cap_flag()
    assert store.board.flags[flag_id].message_ids == ["m003"]


def test_flag_origins(store):
    ok(run(store, "raise_flag", {"kind": "ambiguous", "question": "Which?", "message_ids": ["m001"]}))
    ok(run(store, "upsert_quote", upsert(confidence="low")))
    ToolExecutor(store, store.messages["m003"]).raise_cap_flag()
    assert [f.origin for f in store.board.flags.values()] == ["agent", "low_confidence", "cap"]
