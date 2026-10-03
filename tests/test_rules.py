import pytest

from brokerchat.models import FlagKind, QuoteStatus
from brokerchat.rules import detect_conflicts, sweep_stale
from brokerchat.store import Store
from tests.helpers import at, emit_quote, make_quote


@pytest.fixture
def store(glossary):
    return Store(glossary)


def test_live_quote_older_than_30_minutes_goes_stale(store):
    emit_quote(store, make_quote(minutes=0))
    events = sweep_stale(store, now=at(31))
    assert len(events) == 1
    assert store.board.quotes["q0001"].status == QuoteStatus.STALE


def test_quote_exactly_30_minutes_old_stays_live(store):
    emit_quote(store, make_quote(minutes=0))
    assert sweep_stale(store, now=at(30)) == []
    assert store.board.quotes["q0001"].status == QuoteStatus.LIVE


def test_non_live_quotes_are_left_alone(store):
    emit_quote(store, make_quote(status=QuoteStatus.REFERRED))
    assert sweep_stale(store, now=at(120)) == []


def test_crossed_markets_from_two_sources_raise_one_conflict(store):
    emit_quote(store, make_quote(quote_id="q0001", bid="58", offer="58.5", source="jpm_cds", ids=("m001",)))
    emit_quote(store, make_quote(quote_id="q0002", bid="57", offer="57.5", source="gs_flow", ids=("m002",)))
    events = detect_conflicts(store, now=at(1))
    assert len(events) == 1
    flag = store.board.flags["conflict:q0001:q0002"]
    assert flag.kind == FlagKind.CONFLICT
    assert flag.message_ids == ["m001", "m002"]
    assert detect_conflicts(store, now=at(2)) == []  # not raised twice


def test_touching_markets_count_as_crossed(store):
    emit_quote(store, make_quote(quote_id="q0001", bid="58", offer="58.5", source="jpm_cds"))
    emit_quote(store, make_quote(quote_id="q0002", bid="58.5", offer="59", source="gs_flow"))
    assert len(detect_conflicts(store, now=at(1))) == 1


def test_no_conflict_for_same_source_or_different_tenor(store):
    emit_quote(store, make_quote(quote_id="q0001", bid="58", offer="58.5", source="jpm_cds"))
    emit_quote(store, make_quote(quote_id="q0002", bid="57", offer="57.5", source="jpm_cds", tenor="10Y"))
    emit_quote(store, make_quote(quote_id="q0003", bid="56.5", offer="58", source="gs_flow", tenor="10Y"))
    assert detect_conflicts(store, now=at(1)) == []


def test_one_sided_quotes_only_cross_on_the_sides_they_have(store):
    emit_quote(store, make_quote(quote_id="q0001", bid="58", offer=None, source="jpm_cds"))
    emit_quote(store, make_quote(quote_id="q0002", bid="59", offer=None, source="gs_flow"))
    assert detect_conflicts(store, now=at(1)) == []


def test_conflict_flags_are_marked_as_rule_flags(store):
    emit_quote(store, make_quote(quote_id="q0001", bid="58", offer="58.5", source="jpm_cds"))
    emit_quote(store, make_quote(quote_id="q0002", bid="57", offer="57.5", source="gs_flow"))
    detect_conflicts(store, now=at(1))
    assert store.board.flags["conflict:q0001:q0002"].origin == "rule"
