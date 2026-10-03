import json

import pytest

from brokerchat.models import QuoteStatus
from brokerchat.pipeline import AgentResult, NullAgent, Pipeline, Usage, write_run_outputs
from brokerchat.store import Store
from tests.helpers import emit_quote, make_quote, msg


class QuoteOnFirstMessage:
    def process(self, message, store):
        if message.id == "m001":
            emit_quote(store, make_quote(minutes=0, ids=("m001",)))
        return AgentResult(message_id=message.id, status="ok", usage=Usage(input_tokens=10, output_tokens=2))


def test_pipeline_runs_agent_then_staleness(glossary):
    store = Store(glossary)
    pipeline = Pipeline(store, QuoteOnFirstMessage())
    results = pipeline.run([msg("m001", "Main 5y 58/58.5", 0), msg("m002", "ty", 45)])
    assert [r.message_id for r in results] == ["m001", "m002"]
    assert store.board.quotes["q0001"].status == QuoteStatus.STALE
    assert pipeline.total_usage().input_tokens == 20


def test_pipeline_rejects_duplicate_message(glossary):
    pipeline = Pipeline(Store(glossary), NullAgent())
    pipeline.step(msg("m001", "hi"))
    with pytest.raises(ValueError):
        pipeline.step(msg("m001", "hi again"))


def test_usage_cost_uses_opus_pricing():
    assert Usage(input_tokens=1_000_000).cost_usd() == pytest.approx(4.0)
    assert Usage(output_tokens=1_000_000).cost_usd() == pytest.approx(20.0)
    assert Usage(cache_read_input_tokens=1_000_000).cost_usd() == pytest.approx(0.2)


def test_usage_from_api_treats_missing_cache_fields_as_zero():
    class ApiUsage:
        input_tokens = 5
        output_tokens = 3
        cache_read_input_tokens = None
        cache_creation_input_tokens = None

    assert Usage.from_api(ApiUsage()) == Usage(input_tokens=5, output_tokens=3)


def test_write_run_outputs(glossary, tmp_path):
    store = Store(glossary)
    pipeline = Pipeline(store, QuoteOnFirstMessage())
    pipeline.run([msg("m001", "Main 5y 58/58.5", 0)])
    write_run_outputs(store, pipeline.results, tmp_path / "out")
    board = json.loads((tmp_path / "out" / "board.json").read_text(encoding="utf-8"))
    assert board["quotes"][0]["quote_id"] == "q0001"
    assert (tmp_path / "out" / "events.jsonl").read_text(encoding="utf-8").count("\n") == 1
    assert (tmp_path / "out" / "results.jsonl").exists()
