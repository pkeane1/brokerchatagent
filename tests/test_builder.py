from decimal import Decimal

import pytest

from brokerchat.expected import read_case, write_case
from brokerchat.generator.builder import REFER_PHRASES, DayBuilder, ScenarioError, run_scenario
from brokerchat.models import Direction, QuoteStatus, Unit

R = "ldn-cds-1"


@pytest.fixture
def builder(glossary):
    return DayBuilder(glossary, name="t", seed=7)


def key(instrument="ITRAXX_EUR_MAIN", tenor="5Y", source="jpm_cds"):
    return (instrument, tenor, source)


def test_quote_writes_alias_and_levels(builder):
    builder.quote(R, "jpm_cds", "ITRAXX_EUR_MAIN", "5Y", 58, 58.5, size=10)
    text = builder.messages[-1].text
    assert "58/58.5" in text and "10x10" in text
    q = builder.quotes[key()]
    assert (q.bid, q.offer, q.bid_size, q.unit) == (Decimal("58"), Decimal("58.5"), Decimal("10"), Unit.BP)
    assert builder.last_action[(R, "jpm_cds")] == "quote"


def test_refer_marks_last_market_referred(builder):
    builder.quote(R, "jpm_cds", "ITRAXX_EUR_MAIN", "5Y", 58, 58.5)
    builder.refer(R, "jpm_cds")
    assert builder.messages[-1].text in REFER_PHRASES
    assert builder.quotes[key()].status == QuoteStatus.REFERRED
    assert builder.last_action[(R, "jpm_cds")] == "refer"


def test_refer_without_previous_market_fails(builder):
    with pytest.raises(ScenarioError, match="no previous market"):
        builder.refer(R, "jpm_cds")


def test_trade_mentions_a_quoted_level(builder):
    builder.quote(R, "gs_flow", "ITRAXX_EUR_XOVER", "5Y", 302, 304)
    builder.trade(R, "gs_flow")
    assert builder.messages[-1].text.split()[-1] in ("302", "304")
    assert builder.quotes[key("ITRAXX_EUR_XOVER", source="gs_flow")].status == QuoteStatus.TRADED


def test_shift_wider_moves_both_sides(builder):
    builder.quote(R, "jpm_cds", "ITRAXX_EUR_MAIN", "5Y", 58, 58.5)
    builder.shift(R, "jpm_cds", 1)
    q = builder.quotes[key()]
    assert (q.bid, q.offer) == (Decimal("59"), Decimal("59.5"))
    text = builder.messages[-1].text
    assert "wider" in text or text.startswith("+")


def test_shift_rejects_price_quoted_market(builder):
    builder.quote(R, "citi_ix", "CDX_NA_HY", "5Y", 104.5, 104.625)
    with pytest.raises(ScenarioError, match="bp"):
        builder.shift(R, "citi_ix", 1)


def test_one_sided_offer(builder):
    builder.one_sided(R, "bnp_cds", "CDX_NA_IG", "5Y", "offer", 52.75)
    q = builder.quotes[key("CDX_NA_IG", source="bnp_cds")]
    assert q.bid is None and q.offer == Decimal("52.75")


def test_same_again_quotes_new_tenor_of_same_instrument(builder):
    builder.quote(R, "jpm_cds", "VOLKSWAGEN_AG", "5Y", 120, 125)
    builder.same_again(R, "jpm_cds", "10Y", 165, 172)
    q = builder.quotes[key("VOLKSWAGEN_AG", "10Y")]
    assert (q.bid, q.offer) == (Decimal("165"), Decimal("172"))
    assert builder.last_quote[(R, "jpm_cds")] == key("VOLKSWAGEN_AG", "10Y")


def test_axe_and_interest_are_expected(builder):
    builder.axe(R, "bnp_cds", "ITRAXX_EUR_XOVER", "5Y", "buy_protection", 25)
    builder.interest(R, "bnp_cds", "Alder Capital", "VOLKSWAGEN_AG", "5Y", "sell_protection")
    case = builder.build()
    assert case.expected.axes[0].direction == Direction.BUY_PROTECTION
    assert case.expected.interests[0].client == "Alder Capital"
    assert "protection" in builder.messages[0].text


def test_ambiguous_messages_are_recorded(builder, glossary):
    builder.ambiguous_unknown(R, "gs_flow", 88, 90)
    builder.ambiguous_no_context("bnp_cds", 61, 62, 10)
    assert builder.ambiguous_ids == ["m001", "m002"]
    assert glossary.lookup(builder.messages[0].text.split()[0]) is None
    assert builder.messages[1].room == "desk-misc"


def test_build_marks_old_live_quotes_stale(glossary):
    b = DayBuilder(glossary, name="slow", seed=1, gap_range=(2000, 2000))
    b.quote(R, "jpm_cds", "ITRAXX_EUR_MAIN", "5Y", 58, 58.5)
    b.chatter(R, "gs_flow")
    case = b.build()
    assert case.expected.quotes[0].status == QuoteStatus.STALE


def test_same_seed_same_text(glossary):
    texts = []
    for _ in range(2):
        b = DayBuilder(glossary, name="t", seed=3)
        b.quote(R, "jpm_cds", "ITRAXX_EUR_MAIN", "5Y", 58, 58.5)
        texts.append(b.messages[0].text)
    assert texts[0] == texts[1]


def test_run_scenario_rejects_unknown_op(glossary):
    with pytest.raises(ScenarioError, match="step 1: unknown op 'teleport'"):
        run_scenario(glossary, {"name": "x", "steps": [{"op": "teleport"}]})


def test_run_scenario_reports_bad_arguments(glossary):
    with pytest.raises(ScenarioError, match=r"step 1 \(quote\)"):
        run_scenario(glossary, {"name": "x", "steps": [{"op": "quote", "room": R}]})


def test_case_roundtrip(builder, tmp_path):
    builder.quote(R, "jpm_cds", "ITRAXX_EUR_MAIN", "5Y", 58, 58.5)
    case = builder.build()
    path = write_case(case, tmp_path)
    assert path.name == "t.jsonl"
    assert (tmp_path / "t.expected.json").exists()
    assert read_case(path) == case


def test_two_markets_then_refer_is_ambiguous(builder):
    builder.two_markets(R, "jpm_cds", "ITRAXX_EUR_MAIN", "5Y", 58, 58.5, "ITRAXX_EUR_XOVER", "5Y", 302, 304)
    builder.ambiguous_refer(R, "jpm_cds")
    assert "58/58.5" in builder.messages[0].text and "302/304" in builder.messages[0].text
    assert builder.ambiguous_ids == ["m002"]
    assert builder.quotes[key()].status == QuoteStatus.LIVE
    assert builder.quotes[key("ITRAXX_EUR_XOVER")].status == QuoteStatus.LIVE


def test_ambiguous_refer_needs_a_multi_market_message(builder):
    builder.quote(R, "jpm_cds", "ITRAXX_EUR_MAIN", "5Y", 58, 58.5)
    with pytest.raises(ScenarioError, match="several markets"):
        builder.ambiguous_refer(R, "jpm_cds")
