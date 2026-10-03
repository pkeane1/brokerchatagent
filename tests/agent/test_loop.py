import pytest

from brokerchat.agent.loop import FALLBACK_BETA, MODEL, ClaudeAgent
from brokerchat.agent.tools import TOOL_SCHEMAS
from brokerchat.models import FlagKind
from brokerchat.store import Store
from tests.agent.fakes import FakeClient, FakeConnectionError, response, text, tool_use
from tests.helpers import msg

UPSERT = {"quote_id": None, "instrument": "ITRAXX_EUR_MAIN", "tenor": "5Y", "bid": 58, "offer": 58.5, "bid_size": 10,
          "offer_size": 10, "source": "jpm_cds", "source_message_ids": ["m001"], "confidence": "high"}


@pytest.fixture
def store(glossary):
    s = Store(glossary)
    s.add_message(msg("m001", "Main 5y 58/58.5 10x10"))
    return s


def process(glossary, store, *items, **agent_kwargs):
    client = FakeClient(*items)
    result = ClaudeAgent(client, glossary, **agent_kwargs).process(store.messages["m001"], store)
    return result, client


def test_message_without_tool_calls_is_ok(glossary, store):
    result, client = process(glossary, store, response("end_turn", text("Nothing to record.")))
    assert result.status == "ok"
    assert result.tool_calls == 0
    call = client.calls[0]
    assert call["model"] == MODEL
    assert call["tools"] == TOOL_SCHEMAS
    assert call["thinking"] == {"type": "adaptive"}
    assert call["output_config"] == {"effort": "medium"}
    assert call["betas"] == [FALLBACK_BETA]
    assert call["fallbacks"] == "default"
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "tool_choice" not in call
    assert '<new_message id="m001"' in call["messages"][0]["content"]


def test_tool_call_then_end_turn_records_quote(glossary, store):
    first = response("tool_use", tool_use("tu_1", "upsert_quote", UPSERT))
    result, client = process(glossary, store, first, response("end_turn", text("Done.")))
    assert result.status == "ok"
    assert result.tool_calls == 1
    assert "q0001" in store.board.quotes
    second_messages = client.calls[1]["messages"]
    assert second_messages[1] == {"role": "assistant", "content": first.content}
    tool_result = second_messages[2]["content"][0]
    assert tool_result["type"] == "tool_result" and tool_result["tool_use_id"] == "tu_1"


def test_parallel_tool_calls_return_in_one_user_message(glossary, store):
    first = response(
        "tool_use",
        tool_use("tu_1", "lookup_instrument", {"text": "Main"}),
        tool_use("tu_2", "lookup_instrument", {"text": "ZXQ"}),
    )
    _, client = process(glossary, store, first, response("end_turn"))
    results = client.calls[1]["messages"][2]["content"]
    assert [r["tool_use_id"] for r in results] == ["tu_1", "tu_2"]


def test_tool_error_is_sent_back_not_raised(glossary, store):
    bad = dict(UPSERT, source_message_ids=["m999"])
    _, client = process(glossary, store, response("tool_use", tool_use("tu_1", "upsert_quote", bad)), response("end_turn"))
    assert client.calls[1]["messages"][2]["content"][0]["is_error"] is True
    assert store.board.quotes == {}


def test_iteration_cap_raises_flag(glossary, store):
    loops = [response("tool_use", tool_use(f"tu_{i}", "lookup_instrument", {"text": "Main"})) for i in range(3)]
    result, client = process(glossary, store, *loops, max_iterations=3)
    assert result.status == "cap_hit"
    assert len(client.calls) == 3
    flag = next(iter(store.board.flags.values()))
    assert flag.kind == FlagKind.AMBIGUOUS and flag.message_ids == ["m001"]


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
def test_refusal_and_truncation_leave_message_unprocessed(glossary, store, stop_reason):
    result, _ = process(glossary, store, response(stop_reason))
    assert (result.status, result.reason) == ("unprocessed", stop_reason)


def test_connection_error_leaves_message_unprocessed(glossary, store):
    result, _ = process(glossary, store, FakeConnectionError())
    assert (result.status, result.reason) == ("unprocessed", "connection_error")


def test_usage_accumulates_across_calls(glossary, store):
    result, _ = process(
        glossary, store, response("tool_use", tool_use("tu_1", "lookup_instrument", {"text": "Main"})), response("end_turn")
    )
    assert result.usage.input_tokens == 200
    assert result.usage.output_tokens == 40


def test_effort_is_passed_through(glossary, store):
    _, client = process(glossary, store, response("end_turn"), effort="high")
    assert client.calls[0]["output_config"] == {"effort": "high"}
