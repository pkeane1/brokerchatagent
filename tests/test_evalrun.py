import json
from datetime import UTC, datetime

import pytest

from brokerchat.cli import main
from brokerchat.evalrun import failure_lines, run_eval, write_eval_run
from brokerchat.expected import write_case
from brokerchat.generator.random_day import random_day
from brokerchat.pipeline import NullAgent


@pytest.fixture
def golden(glossary, tmp_path):
    directory = tmp_path / "golden"
    write_case(random_day(glossary, seed=1, n_steps=15), directory)
    write_case(random_day(glossary, seed=2, n_steps=15), directory)
    return directory


def test_null_agent_eval_runs_end_to_end(glossary, golden, tmp_path):
    report, runs = run_eval(golden, NullAgent, glossary)
    assert [r.name for r in runs] == ["day01", "day02"]
    assert report.messages == 30
    assert report.summary()["s1_quote_field_accuracy"] < 1.0
    assert failure_lines(report)
    run_dir = write_eval_run(report, runs, tmp_path / "runs", now=datetime(2026, 10, 2, 12, 0, tzinfo=UTC))
    assert run_dir.name == "20261002T120000Z"
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["targets_met"]["s3_unsupported_entries"] is True
    assert (run_dir / "day01" / "score.json").exists()
    assert (run_dir / "day01" / "board.json").exists()


def test_case_filter(glossary, golden):
    report, runs = run_eval(golden, NullAgent, glossary, case_names=["day02"])
    assert [r.name for r in runs] == ["day02"]


def test_unknown_case_name_fails(glossary, golden):
    with pytest.raises(ValueError, match="day99"):
        run_eval(golden, NullAgent, glossary, case_names=["day99"])


def test_eval_cli(golden, tmp_path):
    code = main(["eval", "--golden", str(golden), "--agent", "null", "--runs-dir", str(tmp_path / "runs"),
                 "--show-failures"])
    assert code == 0
    assert len(list((tmp_path / "runs").iterdir())) == 1


def test_each_case_is_saved_before_the_next_one_runs(glossary, golden, tmp_path):
    from brokerchat.evalrun import new_run_dir, write_case_run

    run_dir = new_run_dir(tmp_path / "runs", now=datetime(2026, 10, 2, 12, 0, tzinfo=UTC))
    built = []

    def factory():
        built.append(1)
        if len(built) == 2:
            raise RuntimeError("crash in the second case")
        return NullAgent()

    with pytest.raises(RuntimeError):
        run_eval(golden, factory, glossary, on_case=lambda run: write_case_run(run, run_dir))
    assert (run_dir / "day01" / "score.json").exists()
