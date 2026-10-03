from brokerchat.agent.prompt import build_context, build_system_prompt
from brokerchat.store import Store
from tests.helpers import emit_quote, make_quote, msg


def test_system_prompt_lists_every_instrument_and_tenor(glossary):
    prompt = build_system_prompt(glossary)
    for canonical in glossary.instruments:
        assert canonical in prompt
    assert "Valid tenors: 1Y, 2Y, 3Y, 4Y, 5Y, 7Y, 10Y" in prompt


def test_system_prompt_is_deterministic(glossary):
    assert build_system_prompt(glossary) == build_system_prompt(glossary)


def test_context_shows_recent_same_room_messages_only(glossary):
    store = Store(glossary)
    for i in range(1, 26):
        store.add_message(msg(f"a{i:02d}", f"line {i}", minutes=i, room="ldn-cds-1"))
    store.add_message(msg("b01", "other room", minutes=26, room="ldn-cds-2"))
    new = msg("a26", "off that", minutes=27, room="ldn-cds-1", sender="gs_flow")
    store.add_message(new)

    context = build_context(store, new, recent=20)
    assert "[a05 " not in context
    assert "[a06 " in context and "[a25 " in context
    assert "b01" not in context
    assert '<new_message id="a26" room="ldn-cds-1" sender="gs_flow"' in context
    assert context.count("[a") == 20


def test_context_includes_board_rows(glossary):
    store = Store(glossary)
    emit_quote(store, make_quote())
    new = msg("m002", "refer", 1)
    store.add_message(new)
    context = build_context(store, new)
    assert '"quote_id": "q0001"' in context
    assert '"offer": "58.5"' in context


def test_context_with_empty_board(glossary):
    store = Store(glossary)
    new = msg("m001", "morning all")
    store.add_message(new)
    assert "<board>\n(empty)\n</board>" in build_context(store, new)


def test_system_prompt_covers_refers_after_multi_market_messages(glossary):
    assert "several markets" in build_system_prompt(glossary)
