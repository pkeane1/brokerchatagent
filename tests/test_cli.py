import json

from brokerchat.cli import main
from tests.helpers import REPO_ROOT


def test_replay_with_null_agent_writes_outputs(tmp_path):
    out = tmp_path / "run"
    code = main(
        ["replay", str(REPO_ROOT / "examples" / "handwritten.jsonl"), "--agent", "null", "--no-live", "--out", str(out)]
    )
    assert code == 0
    assert json.loads((out / "board.json").read_text(encoding="utf-8"))["quotes"] == []
    assert len((out / "results.jsonl").read_text(encoding="utf-8").splitlines()) == 20


def test_generate_writes_seeded_days_and_scenarios(tmp_path):
    out = tmp_path / "golden"
    code = main(
        ["generate", "--out", str(out), "--seeds", "1", "2", "--steps", "10",
         "--scenario", str(REPO_ROOT / "scenarios" / "basic_flow.yaml")]
    )
    assert code == 0
    assert sorted(p.name for p in out.glob("*.jsonl")) == ["basic_flow.jsonl", "day01.jsonl", "day02.jsonl"]
    assert (out / "day01.expected.json").exists()


def test_generate_with_nothing_to_do_fails(tmp_path):
    assert main(["generate", "--out", str(tmp_path)]) == 2


def test_load_env_reads_dotenv_without_overriding_real_env(tmp_path, monkeypatch):
    from brokerchat.cli import load_env

    (tmp_path / ".env").write_text("BROKERCHAT_TEST_A=from-file\nBROKERCHAT_TEST_B=from-file\n", encoding="utf-8")
    monkeypatch.delenv("BROKERCHAT_TEST_A", raising=False)
    monkeypatch.setenv("BROKERCHAT_TEST_B", "from-shell")
    load_env(tmp_path / ".env")
    import os

    assert os.environ["BROKERCHAT_TEST_A"] == "from-file"
    assert os.environ["BROKERCHAT_TEST_B"] == "from-shell"
    monkeypatch.delenv("BROKERCHAT_TEST_A")


def test_load_env_ignores_comments_blank_lines_and_quotes(tmp_path, monkeypatch):
    import os

    from brokerchat.cli import load_env

    (tmp_path / ".env").write_text('# comment\n\nBROKERCHAT_TEST_C="quoted value"\n', encoding="utf-8")
    monkeypatch.delenv("BROKERCHAT_TEST_C", raising=False)
    load_env(tmp_path / ".env")
    assert os.environ["BROKERCHAT_TEST_C"] == "quoted value"
    monkeypatch.delenv("BROKERCHAT_TEST_C")


def test_load_env_missing_file_is_fine(tmp_path):
    from brokerchat.cli import load_env

    load_env(tmp_path / "nope.env")


def test_claude_agent_without_credentials_stops_with_a_clear_message(tmp_path, monkeypatch, capsys):
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)  # no .env here
    monkeypatch.setattr("brokerchat.cli.ANT_PROFILE_DIR", tmp_path / "no-profile")
    code = main(["replay", str(REPO_ROOT / "examples" / "handwritten.jsonl"), "--no-live"])
    assert code == 2
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().out
