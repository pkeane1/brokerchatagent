from decimal import Decimal

import pytest

from brokerchat.generator.builder import load_scenario
from brokerchat.generator.random_day import random_day
from brokerchat.models import QuoteStatus
from tests.helpers import REPO_ROOT


def test_same_seed_same_day(glossary):
    a, b = random_day(glossary, seed=1), random_day(glossary, seed=1)
    assert [m.text for m in a.messages] == [m.text for m in b.messages]


def test_different_seeds_differ(glossary):
    assert [m.text for m in random_day(glossary, 1).messages] != [m.text for m in random_day(glossary, 2).messages]


def test_one_message_per_step(glossary):
    case = random_day(glossary, seed=3, n_steps=30)
    assert len(case.messages) == 30
    assert case.expected.name == "day03"


@pytest.mark.parametrize("seed", range(1, 21))
def test_generated_days_are_well_formed(glossary, seed):
    case = random_day(glossary, seed=seed)
    for q in case.expected.quotes:
        if q.bid is not None and q.offer is not None:
            assert q.bid < q.offer
    ids = {m.id for m in case.messages}
    assert set(case.expected.ambiguous_message_ids) <= ids


def test_basic_flow_scenario(glossary):
    case = load_scenario(glossary, REPO_ROOT / "scenarios" / "basic_flow.yaml")
    quotes = {(q.instrument, q.source): q for q in case.expected.quotes}
    assert len(quotes) == 4
    main = quotes[("ITRAXX_EUR_MAIN", "jpm_cds")]
    assert (main.bid, main.offer) == (Decimal("59"), Decimal("59.5"))
    assert quotes[("ITRAXX_EUR_XOVER", "gs_flow")].status == QuoteStatus.TRADED
    assert quotes[("CDX_NA_IG", "bnp_cds")].status == QuoteStatus.REFERRED
    assert len(case.expected.axes) == 1
    assert len(case.expected.interests) == 1


def test_context_and_ambiguity_scenario(glossary):
    case = load_scenario(glossary, REPO_ROOT / "scenarios" / "context_and_ambiguity.yaml")
    db = [q for q in case.expected.quotes if q.instrument == "DEUTSCHE_BANK_AG"]
    assert len(db) == 1
    assert (db[0].bid, db[0].offer, db[0].status) == (Decimal("96"), Decimal("101"), QuoteStatus.LIVE)
    assert len(case.expected.ambiguous_message_ids) == 2


def test_generator_uses_phrasings_the_prompt_does_not_list(glossary):
    from brokerchat.agent.prompt import build_system_prompt

    prompt = build_system_prompt(glossary).lower()
    texts = [m.text.lower() for seed in range(1, 60) for m in random_day(glossary, seed).messages]
    for phrase in ["lifted ", "hit ", "cxl", "still good", "+1", " too "]:
        assert any(phrase in t for t in texts), f"generator never writes {phrase!r}"
        assert phrase not in prompt, f"prompt spells out {phrase!r}"


def test_random_days_include_ambiguous_refers(glossary):
    from brokerchat.generator.builder import REFER_PHRASES

    cases = [random_day(glossary, seed) for seed in range(1, 30)]
    ambiguous_refers = [
        m for c in cases for m in c.messages if m.id in c.expected.ambiguous_message_ids and m.text in REFER_PHRASES
    ]
    assert ambiguous_refers
