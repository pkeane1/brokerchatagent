"""Read and write chat logs as JSONL, one ChatMessage per line."""

from pathlib import Path
from typing import Iterable

from pydantic import ValidationError

from brokerchat.models import ChatMessage


class ChatFileError(ValueError):
    pass


def read_chat(path: Path) -> list[ChatMessage]:
    path = Path(path)
    messages: list[ChatMessage] = []
    seen: set[str] = set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            message = ChatMessage.model_validate_json(line)
        except ValidationError as e:
            raise ChatFileError(f"{path}:{lineno}: {e}") from e
        if message.id in seen:
            raise ChatFileError(f"{path}:{lineno}: duplicate message id {message.id}")
        if messages and message.ts < messages[-1].ts:
            raise ChatFileError(f"{path}:{lineno}: timestamp goes backwards ({message.ts} < {messages[-1].ts})")
        seen.add(message.id)
        messages.append(message)
    return messages


def write_chat(messages: Iterable[ChatMessage], path: Path) -> None:
    Path(path).write_text("".join(m.model_dump_json() + "\n" for m in messages), encoding="utf-8")
