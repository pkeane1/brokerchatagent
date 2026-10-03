import pytest

from brokerchat.chatio import ChatFileError, read_chat, write_chat
from tests.helpers import REPO_ROOT, msg


def test_reads_handwritten_examples():
    messages = read_chat(REPO_ROOT / "examples" / "handwritten.jsonl")
    assert len(messages) == 20
    assert messages[1].text == "ITRX Main 5y 58/58.5 10x10"


def test_roundtrip(tmp_path):
    messages = [msg("m001", "Main 5y 58/58.5", 0), msg("m002", "refer", 1)]
    path = tmp_path / "chat.jsonl"
    write_chat(messages, path)
    assert read_chat(path) == messages


def test_blank_lines_are_skipped(tmp_path):
    path = tmp_path / "chat.jsonl"
    path.write_text("\n" + msg("m001", "hi").model_dump_json() + "\n\n", encoding="utf-8")
    assert len(read_chat(path)) == 1


def test_malformed_line_reports_file_and_line(tmp_path):
    path = tmp_path / "chat.jsonl"
    good = msg("m001", "hi").model_dump_json()
    path.write_text(f"{good}\n\n{{not json\n", encoding="utf-8")
    with pytest.raises(ChatFileError, match=r"chat\.jsonl:3:"):
        read_chat(path)


def test_naive_timestamp_reports_line(tmp_path):
    path = tmp_path / "chat.jsonl"
    path.write_text('{"id":"m1","ts":"2026-10-01T08:00:00","room":"r","sender":"s","text":"hi"}\n', encoding="utf-8")
    with pytest.raises(ChatFileError, match=r":1:"):
        read_chat(path)


def test_backwards_timestamp_rejected(tmp_path):
    path = tmp_path / "chat.jsonl"
    write_chat([msg("m001", "a", 5), msg("m002", "b", 1)], path)
    with pytest.raises(ChatFileError, match=r":2: timestamp goes backwards"):
        read_chat(path)


def test_duplicate_id_rejected(tmp_path):
    path = tmp_path / "chat.jsonl"
    write_chat([msg("m001", "a", 0), msg("m001", "b", 1)], path)
    with pytest.raises(ChatFileError, match=r":2: duplicate message id m001"):
        read_chat(path)
