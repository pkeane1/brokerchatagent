import pytest

from brokerchat.models import EventType, Flag, FlagKind, QuoteStatus
from brokerchat.store import Board, Store, read_events, write_events
from tests.helpers import at, emit_quote, make_quote, msg


@pytest.fixture
def store(glossary):
    return Store(glossary)


def test_emit_appends_event_and_updates_board(store):
    event = emit_quote(store, make_quote())
    assert event.event_id == "e00001"
    assert store.events == [event]
    assert store.board.quotes["q0001"].bid == make_quote().bid


def test_status_change_updates_time_and_cites_message(store):
    emit_quote(store, make_quote())
    store.emit(EventType.QUOTE_STATUS_CHANGED, {"quote_id": "q0001", "status": "referred", "message_id": "m002"}, ts=at(5))
    q = store.board.quotes["q0001"]
    assert q.status == QuoteStatus.REFERRED
    assert q.updated_ts == at(5)
    assert q.source_message_ids == ["m001", "m002"]


def test_stale_status_keeps_original_update_time(store):
    emit_quote(store, make_quote())
    store.emit(EventType.QUOTE_STATUS_CHANGED, {"quote_id": "q0001", "status": "stale", "message_id": None}, ts=at(40))
    q = store.board.quotes["q0001"]
    assert q.status == QuoteStatus.STALE
    assert q.updated_ts == at(0)


def test_status_change_for_unknown_quote_is_not_logged(store):
    with pytest.raises(KeyError):
        store.emit(EventType.QUOTE_STATUS_CHANGED, {"quote_id": "q9999", "status": "traded", "message_id": None}, ts=at(1))
    assert store.events == []


def test_flags_raise_and_resolve(store):
    flag = Flag(flag_id="f0001", kind=FlagKind.AMBIGUOUS, question="Which tenor?", message_ids=["m001"])
    store.emit(EventType.FLAG_RAISED, flag.model_dump(mode="json"), ts=at(0))
    store.emit(EventType.FLAG_RESOLVED, {"flag_id": "f0001", "resolution": "5Y"}, ts=at(1))
    assert store.board.flags["f0001"].resolved is True
    assert store.board.flags["f0001"].resolution == "5Y"


def test_find_quote_matches_instrument_tenor_and_source(store):
    emit_quote(store, make_quote())
    assert store.board.find_quote("ITRAXX_EUR_MAIN", "5Y", "jpm_cds").quote_id == "q0001"
    assert store.board.find_quote("ITRAXX_EUR_MAIN", "5Y", "gs_flow") is None
    assert store.board.find_quote("ITRAXX_EUR_MAIN", "10Y", "jpm_cds") is None


def test_board_rebuilt_from_events_matches_live_board(store, tmp_path):
    emit_quote(store, make_quote())
    emit_quote(store, make_quote(quote_id="q0002", name="CDX_NA_IG", source="bnp_cds"))
    store.emit(EventType.QUOTE_STATUS_CHANGED, {"quote_id": "q0002", "status": "traded", "message_id": "m001"}, ts=at(3))
    path = tmp_path / "events.jsonl"
    write_events(store.events, path)
    rebuilt = Board.from_events(read_events(path))
    assert rebuilt.snapshot() == store.board.snapshot()


def test_ids_are_sequential(store):
    assert [store.next_quote_id(), store.next_quote_id()] == ["q0001", "q0002"]
    assert store.next_flag_id() == "f0001"


def test_duplicate_message_id_rejected(store):
    store.add_message(msg("m001", "hi"))
    with pytest.raises(ValueError, match="duplicate message id m001"):
        store.add_message(msg("m001", "again"))
