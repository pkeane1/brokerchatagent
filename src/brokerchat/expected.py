"""Ground truth for a generated chat: the board a correct agent ends the day with."""

from decimal import Decimal
from pathlib import Path

from pydantic import AwareDatetime, BaseModel

from brokerchat.chatio import read_chat, write_chat
from brokerchat.models import ChatMessage, Direction, QuoteStatus, Unit


class ExpectedQuote(BaseModel):
    instrument: str
    tenor: str
    source: str
    bid: Decimal | None
    offer: Decimal | None
    unit: Unit
    bid_size: Decimal | None = None
    offer_size: Decimal | None = None
    status: QuoteStatus
    updated_ts: AwareDatetime


class ExpectedAxe(BaseModel):
    instrument: str
    tenor: str
    direction: Direction
    size: Decimal | None
    source: str


class ExpectedInterest(BaseModel):
    client: str
    instrument: str
    tenor: str
    direction: Direction


class ExpectedBoard(BaseModel):
    name: str
    end_ts: AwareDatetime
    quotes: list[ExpectedQuote]
    axes: list[ExpectedAxe]
    interests: list[ExpectedInterest]
    ambiguous_message_ids: list[str]


class GeneratedCase(BaseModel):
    messages: list[ChatMessage]
    expected: ExpectedBoard


def write_case(case: GeneratedCase, directory: Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    chat_path = directory / f"{case.expected.name}.jsonl"
    write_chat(case.messages, chat_path)
    chat_path.with_suffix(".expected.json").write_text(case.expected.model_dump_json(indent=2), encoding="utf-8")
    return chat_path


def read_case(chat_path: Path) -> GeneratedCase:
    chat_path = Path(chat_path)
    expected_json = chat_path.with_suffix(".expected.json").read_text(encoding="utf-8")
    return GeneratedCase(messages=read_chat(chat_path), expected=ExpectedBoard.model_validate_json(expected_json))
