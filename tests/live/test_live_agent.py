import os

import anthropic
import pytest

from brokerchat.agent.loop import ClaudeAgent
from brokerchat.cli import load_env
from brokerchat.generator.builder import load_scenario
from brokerchat.pipeline import Pipeline
from brokerchat.scoring import score_case
from brokerchat.store import Store
from tests.helpers import REPO_ROOT

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("BROKERCHAT_LIVE") != "1", reason="set BROKERCHAT_LIVE=1 to call the Claude API"),
]


def test_basic_flow_scenario_end_to_end(glossary):
    load_env(REPO_ROOT / ".env")  # inside the test: a .env must never switch paid tests on
    case = load_scenario(glossary, REPO_ROOT / "scenarios" / "basic_flow.yaml")
    store = Store(glossary)
    pipeline = Pipeline(store, ClaudeAgent(anthropic.Anthropic(), glossary))
    results = pipeline.run(case.messages)
    print(f"cost approx ${pipeline.total_usage().cost_usd():.3f}")
    assert all(r.status == "ok" for r in results), [r for r in results if r.status != "ok"]
    score = score_case(case.expected, store.board, set(store.messages))
    print(score.model_dump_json(indent=2))
    assert score.unsupported_entries == 0
    assert not score.missing_quotes, score.missing_quotes
