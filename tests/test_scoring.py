from decimal import Decimal

import pytest

from brokerchat.expected import ExpectedAxe, ExpectedBoard, ExpectedInterest, ExpectedQuote
from brokerchat.models import Axe, Direction, EventType, Flag, FlagKind, Instrument, InstrumentKind, Interest, QuoteStatus, Unit
from brokerchat.scoring import EvalReport, meets_targets, score_case
from brokerchat.store import Store
from tests.helpers import at, emit_quote, make_quote

MAIN = Instrument(kind=InstrumentKind.INDEX, name="ITRAXX_EUR_MAIN", tenor="5Y")
IDS = {"m001", "m002", "m003"}


def expected(**overrides) -> ExpectedBoard:
    fields = dict(
        name="case",
        end_ts=at(10),
        quotes=[
            ExpectedQuote(instrument="ITRAXX_EUR_MAIN", tenor="5Y", source="jpm_cds", bid=Decimal("58"),
                          offer=Decimal("58.5"), unit=Unit.BP, status=QuoteStatus.LIVE, updated_ts=at(0))
        ],
        axes=[ExpectedAxe(instrument="ITRAXX_EUR_MAIN", tenor="5Y", direction=Direction.BUY_PROTECTION,
                          size=Decimal("25"), source="citi_ix")],
        interests=[ExpectedInterest(client="Alder Capital", instrument="ITRAXX_EUR_MAIN", tenor="5Y",
                                    direction=Direction.SELL_PROTECTION)],
        ambiguous_message_ids=["m003"],
    )
    fields.update(overrides)
    return ExpectedBoard(**fields)


@pytest.fixture
def perfect(glossary) -> Store:
    store = Store(glossary)
    emit_quote(store, make_quote())
    store.emit(EventType.AXE_RECORDED, Axe(instrument=MAIN, direction=Direction.BUY_PROTECTION, size=Decimal("25"),
                                           source="citi_ix", source_message_ids=["m002"]).model_dump(mode="json"), ts=at(1))
    store.emit(EventType.INTEREST_RECORDED, Interest(client="alder capital", instrument=MAIN,
                                                     direction=Direction.SELL_PROTECTION,
                                                     source_message_ids=["m002"]).model_dump(mode="json"), ts=at(1))
    store.emit(EventType.FLAG_RAISED, Flag(flag_id="f0001", kind=FlagKind.UNKNOWN_INSTRUMENT, question="?",
                                           message_ids=["m003"]).model_dump(mode="json"), ts=at(2))
    return store


def test_perfect_board_scores_full_marks(perfect):
    s = score_case(expected(), perfect.board, IDS)
    assert (s.quote_fields_correct, s.quote_fields_total) == (5, 5)
    assert (s.status_correct, s.status_total) == (1, 1)
    assert (s.axes_matched, s.interests_matched) == (1, 1)
    assert (s.flag_tp, s.flag_fn, s.flag_fp) == (1, 0, 0)
    assert s.unsupported_entries == 0
    assert s.spurious_quotes == [] and s.missing_quotes == []


def test_missing_quote_counts_every_field_wrong(glossary):
    s = score_case(expected(), Store(glossary).board, IDS)
    assert (s.quote_fields_correct, s.quote_fields_total) == (0, 5)
    assert s.missing_quotes == ["ITRAXX_EUR_MAIN 5Y jpm_cds"]


def test_wrong_level_and_status_reported(glossary):
    store = Store(glossary)
    emit_quote(store, make_quote(bid="57", status=QuoteStatus.REFERRED))
    s = score_case(expected(), store.board, IDS)
    assert s.quote_fields_correct == 4
    assert s.status_correct == 0
    assert any("bid expected 58 got 57" in e for e in s.field_errors)


def test_spurious_and_unsupported_entries(glossary):
    store = Store(glossary)
    emit_quote(store, make_quote())
    emit_quote(store, make_quote(quote_id="q0002", name="CDX_NA_IG", ids=("m999",)))
    s = score_case(expected(), store.board, IDS)
    assert s.spurious_quotes == ["q0002 CDX_NA_IG 5Y jpm_cds"]
    assert s.unsupported_entries == 1


def test_flag_on_unambiguous_message_is_false_positive(glossary):
    store = Store(glossary)
    store.emit(EventType.FLAG_RAISED, Flag(flag_id="f0001", kind=FlagKind.AMBIGUOUS, question="?",
                                           message_ids=["m001"]).model_dump(mode="json"), ts=at(0))
    s = score_case(expected(), store.board, IDS)
    assert (s.flag_tp, s.flag_fn, s.flag_fp) == (0, 1, 1)
    assert s.missed_flags == ["m003"]


def test_conflict_flags_are_not_scored(glossary):
    store = Store(glossary)
    store.emit(EventType.FLAG_RAISED, Flag(flag_id="conflict:q1:q2", kind=FlagKind.CONFLICT, question="?",
                                           message_ids=["m001"]).model_dump(mode="json"), ts=at(0))
    assert score_case(expected(), store.board, IDS).flag_fp == 0


def test_summary_and_targets(perfect, glossary):
    good = score_case(expected(), perfect.board, IDS)
    empty = score_case(expected(), Store(glossary).board, IDS)
    summary = EvalReport(cases=[good, empty]).summary()
    assert summary["s1_quote_field_accuracy"] == pytest.approx(0.5)
    assert summary["s4_flag_recall"] == pytest.approx(0.5)
    met = meets_targets(summary)
    assert met["s1_quote_field_accuracy"] is False
    assert met["s3_unsupported_entries"] is True


def test_empty_report_counts_as_perfect():
    assert EvalReport(cases=[]).summary()["s1_quote_field_accuracy"] == 1.0


def test_invented_entry_citing_a_real_message_is_unsupported(glossary):
    store = Store(glossary)
    emit_quote(store, make_quote())
    emit_quote(store, make_quote(quote_id="q0002", name="CDX_NA_IG", ids=("m001",)))
    store.emit(EventType.AXE_RECORDED, Axe(instrument=MAIN, direction=Direction.SELL_PROTECTION, size=None,
                                           source="gs_flow", source_message_ids=["m001"]).model_dump(mode="json"), ts=at(1))
    assert score_case(expected(), store.board, IDS).unsupported_entries == 2


def test_low_confidence_and_cap_flags_do_not_count_as_flagging(glossary):
    store = Store(glossary)
    for i, origin in enumerate(["low_confidence", "cap"]):
        store.emit(EventType.FLAG_RAISED, Flag(flag_id=f"f{i}", kind=FlagKind.AMBIGUOUS, question="?",
                                               message_ids=["m003"], origin=origin).model_dump(mode="json"), ts=at(0))
    s = score_case(expected(), store.board, IDS)
    assert (s.flag_tp, s.flag_fn) == (0, 1)


def test_flag_plus_a_guess_on_the_same_message_is_not_credited(perfect):
    emit_quote(perfect, make_quote(quote_id="q0009", name="TESCO_PLC", source="gs_flow", ids=("m003",)))
    s = score_case(expected(), perfect.board, IDS)
    assert (s.flag_tp, s.flag_fn) == (0, 1)
