# Broker Chat Quote Parser Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local prototype that replays synthetic CDS broker chat through a Claude tool-use agent and maintains a live, auditable quote board, with a synthetic data generator and an eval harness that scores the agent against ground truth.

**Architecture:** Event-sourced core: tools emit append-only events, and a `Board` projection is rebuilt from them. A `Pipeline` feeds each chat message to a `Processor` (the Claude agent, or a null agent for testing), then runs deterministic rules (staleness sweep, crossed-market detection). A scenario-driven generator produces chat plus the exact expected board, and a scorer compares the agent's board against it.

**Tech Stack:** Python 3.12+, uv, anthropic (official SDK), pydantic v2, PyYAML, rich, pytest.

**Spec:** `docs/superpowers/specs/2026-10-02-broker-chat-parser-prd.md`

**Deliberate refinements of the spec (flag to the user if any is unwelcome):**
- `get_quotes` filters by `instrument` and `source`, not `room`. Quotes have no room in the PRD §5 data model.
- Crossed-market `conflict` flags come from deterministic code (`rules.detect_conflicts`), not the LLM, in line with the PRD's LLM-vs-deterministic split. The agent's `raise_flag` accepts only `ambiguous` and `unknown_instrument`.
- Scenario files name instruments by canonical name (`ITRAXX_EUR_MAIN`). The generator writes a random alias into the chat text.

## Global Constraints

- Python `>=3.12`, managed with `uv`. Run everything as `uv run ...`.
- Model `claude-opus-5-5`, adaptive thinking, `tool_choice` left as default `auto`, `strict: true` on every tool, effort default `medium`.
- **Never run `git commit` or `git push`** (repo `CLAUDE.md`). Each task ends with a checkpoint where the user reviews and commits.
- Synthetic chat only. Never add real broker chat to the repo.
- Every quote, axe and interest has at least one `source_message_ids` entry, and every id cited must be a chat message the pipeline has seen (PRD S3).
- Staleness: a live quote whose `updated_ts` is more than 30 minutes before the current message time becomes `stale`. This is deterministic.
- Agent limits: at most 8 model calls per message, and a context window of the last 20 messages in the same room.
- Levels and sizes are `Decimal`. All timestamps are timezone-aware.
- Success targets (PRD §3): S1 ≥ 0.95, S2 ≥ 0.95, S3 = 0, S4 ≥ 0.90.

## Review Focus

1. **Odd casing and whitespace in aliases** (`"  MAIN "`, `"itrx   main"`) must still resolve. Test in Task 2.
2. **Price-quoted levels** like `104.625` must survive JSON, events and float tool inputs exactly. Tests in Task 1 and Task 10.
3. **Malformed chat file lines** (bad JSON, naive timestamp, out-of-order time) must fail with the file and line number, not a stack trace from deep inside. Test in Task 5.
4. **Hallucinated message ids** in tool calls must be rejected with an `is_error` result, never written to the board. Test in Task 10.
5. **A repeat market from the same source** (including a re-quote after `refer`) must update one quote and make it live again, not add a duplicate. Test in Task 10.

## File Structure

```
pyproject.toml                       uv project, `brokerchat` console script
domain/glossary.yaml                 instrument aliases, units, tenors (PROVISIONAL)
examples/handwritten.jsonl           20 hand-written chat messages
scenarios/basic_flow.yaml            hand-authored scenario
scenarios/context_and_ambiguity.yaml hand-authored scenario
evals/golden/                        generated golden set (*.jsonl + *.expected.json)
evals/RESULTS.md                     eval run log (Task 13)
src/brokerchat/
  models.py        domain types + enums + fmt_decimal
  glossary.py      Glossary: alias lookup, tenor normalisation
  store.py         Board projection, Store (event log + board + messages), event JSONL io
  rules.py         sweep_stale, detect_conflicts
  chatio.py        read_chat / write_chat with line-numbered errors
  render.py        rich rendering of the board
  pipeline.py      Usage, AgentResult, Processor, NullAgent, Pipeline, write_run_outputs
  expected.py      ExpectedBoard & friends, GeneratedCase, read_case / write_case
  generator/builder.py     DayBuilder, run_scenario, load_scenario
  generator/random_day.py  random_day(seed)
  scoring.py       score_case, EvalReport, TARGETS, meets_targets
  evalrun.py       run_eval, write_eval_run, report_table, failure_lines
  agent/tools.py   TOOL_SCHEMAS, ToolExecutor
  agent/prompt.py  build_system_prompt, build_context
  agent/loop.py    ClaudeAgent (manual tool-use loop)
  cli.py           brokerchat replay | generate | eval
tests/  (mirrors src; helpers.py + conftest.py shared; tests/live for real-API tests)
```

---

### Task 1: Project scaffold and domain models

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `src/brokerchat/__init__.py`, `src/brokerchat/models.py`
- Create: `tests/__init__.py`, `tests/helpers.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Produces: enums `InstrumentKind`, `Unit`, `QuoteStatus`, `Confidence`, `Direction`, `FlagKind`, `EventType`; models `ChatMessage`, `Instrument` (frozen, hashable), `Quote`, `Axe`, `Interest`, `Flag`, `Event`; `fmt_decimal(x: Decimal) -> str`.
- Produces (tests): `tests.helpers.REPO_ROOT`, `T0`, `at(minutes) -> datetime`, `msg(id, text, minutes=0, room="ldn-cds-1", sender="jpm_cds") -> ChatMessage`, `make_quote(...) -> Quote`.

- [ ] **Step 1: Install uv if missing and create the project files**

Check with `uv --version`. If it isn't found, install it on Windows with `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"` and open a new shell.

`pyproject.toml`:

```toml
[project]
name = "brokerchat"
version = "0.1.0"
description = "Claude agent that parses CDS broker chat into a live quote board"
requires-python = ">=3.12"
dependencies = []

[project.scripts]
brokerchat = "brokerchat.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/brokerchat"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
markers = ["live: calls the real Claude API (set BROKERCHAT_LIVE=1)"]
```

`.gitignore`:

```
.venv/
__pycache__/
*.egg-info/
.pytest_cache/
evals/runs/
runs/
```

`src/brokerchat/__init__.py`:

```python
"""Claude agent that parses CDS broker chat into a live quote board."""
```

`tests/__init__.py`: empty file.

Then run:

```bash
uv add anthropic pydantic pyyaml rich
uv add --dev pytest
```

Expected: `uv` creates `.venv` (downloading Python 3.12+ if needed) and `uv.lock`.

- [ ] **Step 2: Write the test helpers and the failing model tests**

`tests/helpers.py`:

```python
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from brokerchat.models import ChatMessage, Instrument, InstrumentKind, Quote, QuoteStatus, Unit

REPO_ROOT = Path(__file__).resolve().parents[1]
T0 = datetime(2026, 10, 1, 8, 0, tzinfo=UTC)


def at(minutes: float) -> datetime:
    return T0 + timedelta(minutes=minutes)


def msg(id: str, text: str, minutes: float = 0, room: str = "ldn-cds-1", sender: str = "jpm_cds") -> ChatMessage:
    return ChatMessage(id=id, ts=at(minutes), room=room, sender=sender, text=text)


def make_quote(
    quote_id: str = "q0001",
    name: str = "ITRAXX_EUR_MAIN",
    tenor: str = "5Y",
    bid: str | None = "58",
    offer: str | None = "58.5",
    source: str = "jpm_cds",
    minutes: float = 0,
    status: QuoteStatus = QuoteStatus.LIVE,
    unit: Unit = Unit.BP,
    ids: tuple[str, ...] = ("m001",),
    kind: InstrumentKind = InstrumentKind.INDEX,
) -> Quote:
    return Quote(
        quote_id=quote_id,
        instrument=Instrument(kind=kind, name=name, tenor=tenor),
        bid=None if bid is None else Decimal(bid),
        offer=None if offer is None else Decimal(offer),
        unit=unit,
        source=source,
        status=status,
        updated_ts=at(minutes),
        source_message_ids=list(ids),
    )
```

`tests/test_models.py`:

```python
from datetime import datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from brokerchat.models import (
    ChatMessage,
    Confidence,
    Instrument,
    InstrumentKind,
    Quote,
    QuoteStatus,
    Unit,
    fmt_decimal,
)
from tests.helpers import make_quote


def test_price_quoted_levels_survive_json_roundtrip():
    q = make_quote(name="CDX_NA_HY", bid="104.5", offer="104.625", unit=Unit.PRICE)
    again = Quote.model_validate_json(q.model_dump_json())
    assert again.offer == Decimal("104.625")
    assert again == q


def test_quote_defaults_to_live_high_confidence():
    q = make_quote()
    assert q.status == QuoteStatus.LIVE
    assert q.confidence == Confidence.HIGH


def test_quote_requires_a_source_message():
    with pytest.raises(ValidationError):
        make_quote(ids=())


def test_chat_message_rejects_naive_timestamp():
    with pytest.raises(ValidationError):
        ChatMessage(id="m1", ts=datetime(2026, 10, 1, 8, 0), room="r", sender="s", text="hi")


def test_instruments_compare_and_hash_by_value():
    a = Instrument(kind=InstrumentKind.INDEX, name="ITRAXX_EUR_MAIN", tenor="5Y")
    b = Instrument(kind=InstrumentKind.INDEX, name="ITRAXX_EUR_MAIN", tenor="5Y")
    assert a == b
    assert len({a, b}) == 1


@pytest.mark.parametrize(
    "value,text",
    [(Decimal("58.50"), "58.5"), (Decimal("60"), "60"), (Decimal("104.625"), "104.625"), (Decimal("0"), "0")],
)
def test_fmt_decimal_drops_trailing_zeros_without_exponent(value, text):
    assert fmt_decimal(value) == text
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'brokerchat.models'`.

- [ ] **Step 4: Implement the models**

`src/brokerchat/models.py`:

```python
"""Domain types for the broker chat quote board (PRD §5)."""

from decimal import Decimal
from enum import StrEnum
from typing import Any

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field


class InstrumentKind(StrEnum):
    INDEX = "index"
    SINGLE_NAME = "single_name"


class Unit(StrEnum):
    BP = "bp"
    PRICE = "price"


class QuoteStatus(StrEnum):
    LIVE = "live"
    REFERRED = "referred"
    TRADED = "traded"
    STALE = "stale"


class Confidence(StrEnum):
    HIGH = "high"
    LOW = "low"


class Direction(StrEnum):
    BUY_PROTECTION = "buy_protection"
    SELL_PROTECTION = "sell_protection"


class FlagKind(StrEnum):
    AMBIGUOUS = "ambiguous"
    CONFLICT = "conflict"
    UNKNOWN_INSTRUMENT = "unknown_instrument"


class EventType(StrEnum):
    QUOTE_UPSERTED = "quote_upserted"
    QUOTE_STATUS_CHANGED = "quote_status_changed"
    AXE_RECORDED = "axe_recorded"
    INTEREST_RECORDED = "interest_recorded"
    FLAG_RAISED = "flag_raised"
    FLAG_RESOLVED = "flag_resolved"


class ChatMessage(BaseModel):
    id: str
    ts: AwareDatetime
    room: str
    sender: str
    text: str


class Instrument(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: InstrumentKind
    name: str
    tenor: str
    series: int | None = None


class Quote(BaseModel):
    quote_id: str
    instrument: Instrument
    bid: Decimal | None = None
    offer: Decimal | None = None
    unit: Unit
    bid_size: Decimal | None = None
    offer_size: Decimal | None = None
    source: str
    status: QuoteStatus = QuoteStatus.LIVE
    updated_ts: AwareDatetime
    source_message_ids: list[str] = Field(min_length=1)
    confidence: Confidence = Confidence.HIGH


class Axe(BaseModel):
    instrument: Instrument
    direction: Direction
    size: Decimal | None = None
    source: str
    source_message_ids: list[str] = Field(min_length=1)


class Interest(BaseModel):
    client: str
    instrument: Instrument
    direction: Direction
    note: str | None = None
    source_message_ids: list[str] = Field(min_length=1)


class Flag(BaseModel):
    flag_id: str
    kind: FlagKind
    question: str
    message_ids: list[str]
    resolved: bool = False
    resolution: str | None = None


class Event(BaseModel):
    event_id: str
    ts: AwareDatetime
    type: EventType
    payload: dict[str, Any]
    caused_by_message_id: str | None = None


def fmt_decimal(x: Decimal) -> str:
    """Render a level or size the way a broker would type it: no trailing zeros, no exponent."""
    return format(x.normalize(), "f")
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_models.py -v`
Expected: all PASS.

- [ ] **Step 6: Checkpoint**

Run `uv run pytest -q` and expect all to pass. Stop and tell the user Task 1 is ready to review and commit. Do not commit.

---

### Task 2: Domain glossary

**Files:**
- Create: `domain/glossary.yaml`, `src/brokerchat/glossary.py`, `tests/conftest.py`
- Test: `tests/test_glossary.py`

**Interfaces:**
- Consumes: `InstrumentKind`, `Unit` from Task 1.
- Produces: `InstrumentInfo(canonical: str, kind: InstrumentKind, unit: Unit, aliases: tuple[str, ...])` (frozen dataclass); `Glossary` with `.instruments: dict[str, InstrumentInfo]`, `.tenors: tuple[str, ...]`, `Glossary.load(path=DEFAULT_GLOSSARY_PATH)`, `.lookup(text) -> InstrumentInfo | None`, `.get(canonical) -> InstrumentInfo | None`, `.normalize_tenor(text) -> str | None`. Pytest fixture `glossary` (session scope).

- [ ] **Step 1: Write the glossary data**

`domain/glossary.yaml`:

```yaml
# Instrument aliases as they appear in broker chat. Matching ignores case and repeated spaces.
# PROVISIONAL: to be reviewed by a practising CDS broker (PRD §4, §11).
tenors: [1Y, 2Y, 3Y, 4Y, 5Y, 7Y, 10Y]
instruments:
  - canonical: ITRAXX_EUR_MAIN
    kind: index
    unit: bp
    aliases: [main, itrx main, itraxx main, europe main]
  - canonical: ITRAXX_EUR_XOVER
    kind: index
    unit: bp
    aliases: [xo, xover, crossover, itrx xo]
  - canonical: CDX_NA_IG
    kind: index
    unit: bp
    aliases: [ig, cdx ig, cdx.ig]
  - canonical: CDX_NA_HY
    kind: index
    unit: price
    aliases: [hy, cdx hy, cdx.hy]
  - canonical: VOLKSWAGEN_AG
    kind: single_name
    unit: bp
    aliases: [vw, volkswagen, vow]
  - canonical: DEUTSCHE_BANK_AG
    kind: single_name
    unit: bp
    aliases: [db, deutsche, dbk]
  - canonical: TESCO_PLC
    kind: single_name
    unit: bp
    aliases: [tesco, tsco]
```

- [ ] **Step 2: Write the fixture and failing tests**

`tests/conftest.py`:

```python
import pytest

from brokerchat.glossary import Glossary


@pytest.fixture(scope="session")
def glossary() -> Glossary:
    return Glossary.load()
```

`tests/test_glossary.py`:

```python
import pytest

from brokerchat.glossary import Glossary, InstrumentInfo
from brokerchat.models import InstrumentKind, Unit


@pytest.mark.parametrize(
    "text,canonical",
    [
        ("Main", "ITRAXX_EUR_MAIN"),
        ("  MAIN ", "ITRAXX_EUR_MAIN"),
        ("itrx   main", "ITRAXX_EUR_MAIN"),
        ("XO", "ITRAXX_EUR_XOVER"),
        ("cdx.hy", "CDX_NA_HY"),
        ("vw", "VOLKSWAGEN_AG"),
        ("itraxx_eur_main", "ITRAXX_EUR_MAIN"),
    ],
)
def test_lookup_resolves_aliases_ignoring_case_and_spacing(glossary, text, canonical):
    info = glossary.lookup(text)
    assert info is not None
    assert info.canonical == canonical


def test_lookup_unknown_alias_returns_none(glossary):
    assert glossary.lookup("ZXQ") is None


def test_units_and_kinds(glossary):
    assert glossary.get("CDX_NA_HY").unit == Unit.PRICE
    assert glossary.get("ITRAXX_EUR_MAIN").unit == Unit.BP
    assert glossary.get("VOLKSWAGEN_AG").kind == InstrumentKind.SINGLE_NAME


@pytest.mark.parametrize(
    "text,tenor", [("5y", "5Y"), ("5Y", "5Y"), ("5yr", "5Y"), ("10s", "10Y"), (" 7 years ", "7Y"), ("10", "10Y")]
)
def test_normalize_tenor(glossary, text, tenor):
    assert glossary.normalize_tenor(text) == tenor


@pytest.mark.parametrize("text", ["6Y", "five", "", "5m"])
def test_normalize_tenor_rejects_unknown(glossary, text):
    assert glossary.normalize_tenor(text) is None


def test_duplicate_alias_across_instruments_is_rejected():
    with pytest.raises(ValueError, match="maps to two instruments"):
        Glossary(
            [
                InstrumentInfo("A", InstrumentKind.INDEX, Unit.BP, ("x",)),
                InstrumentInfo("B", InstrumentKind.INDEX, Unit.BP, ("X",)),
            ],
            ["5Y"],
        )
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_glossary.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'brokerchat.glossary'`.

- [ ] **Step 4: Implement the glossary**

`src/brokerchat/glossary.py`:

```python
"""Deterministic instrument alias lookup and tenor normalisation (PRD §4)."""

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from brokerchat.models import InstrumentKind, Unit

DEFAULT_GLOSSARY_PATH = Path(__file__).resolve().parents[2] / "domain" / "glossary.yaml"
_TENOR_RE = re.compile(r"^(\d{1,2})\s*(y|yr|yrs|year|years|s)?$")


def _norm(text: str) -> str:
    return " ".join(text.strip().lower().split())


@dataclass(frozen=True)
class InstrumentInfo:
    canonical: str
    kind: InstrumentKind
    unit: Unit
    aliases: tuple[str, ...]


class Glossary:
    def __init__(self, instruments: list[InstrumentInfo], tenors: list[str]):
        self.instruments: dict[str, InstrumentInfo] = {i.canonical: i for i in instruments}
        self.tenors: tuple[str, ...] = tuple(tenors)
        self._alias_index: dict[str, InstrumentInfo] = {}
        for info in instruments:
            for alias in (*info.aliases, info.canonical):
                key = _norm(alias)
                existing = self._alias_index.get(key)
                if existing is not None and existing is not info:
                    raise ValueError(f"alias {alias!r} maps to two instruments: {existing.canonical}, {info.canonical}")
                self._alias_index[key] = info

    @classmethod
    def load(cls, path: Path = DEFAULT_GLOSSARY_PATH) -> "Glossary":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        instruments = [
            InstrumentInfo(
                canonical=item["canonical"],
                kind=InstrumentKind(item["kind"]),
                unit=Unit(item["unit"]),
                aliases=tuple(item["aliases"]),
            )
            for item in data["instruments"]
        ]
        return cls(instruments, data["tenors"])

    def lookup(self, text: str) -> InstrumentInfo | None:
        return self._alias_index.get(_norm(text))

    def get(self, canonical: str) -> InstrumentInfo | None:
        return self.instruments.get(canonical)

    def normalize_tenor(self, text: str) -> str | None:
        match = _TENOR_RE.match(_norm(text))
        if match is None:
            return None
        tenor = f"{int(match.group(1))}Y"
        return tenor if tenor in self.tenors else None
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_glossary.py -v`
Expected: all PASS.

- [ ] **Step 6: Checkpoint**

Run `uv run pytest -q` and expect all to pass. Tell the user Task 2 is ready, and remind them that `domain/glossary.yaml` should go to their broker friend for review (PRD M0). Do not commit.

---

### Task 3: Event log, board projection and store

**Files:**
- Create: `src/brokerchat/store.py`
- Modify: `tests/helpers.py` (append `emit_quote`)
- Test: `tests/test_store.py`

**Interfaces:**
- Consumes: models from Task 1, `Glossary` from Task 2.
- Produces:
  - `Board` with `.quotes: dict[str, Quote]`, `.axes: list[Axe]`, `.interests: list[Interest]`, `.flags: dict[str, Flag]`, `.apply(event: Event) -> None`, `Board.from_events(events) -> Board`, `.find_quote(name: str, tenor: str, source: str) -> Quote | None`, `.snapshot() -> dict`.
  - `Store(glossary)` with `.glossary`, `.events: list[Event]`, `.board: Board`, `.messages: dict[str, ChatMessage]` (insertion-ordered), `.add_message(m)`, `.emit(type: EventType, payload: dict, ts: datetime, caused_by: str | None = None) -> Event`, `.next_quote_id() -> str` (`q0001`, ...), `.next_flag_id() -> str` (`f0001`, ...).
  - `write_events(events, path)` and `read_events(path) -> list[Event]`.
  - Payload contracts: `quote_upserted` takes a full `Quote.model_dump(mode="json")`. `quote_status_changed` takes `{"quote_id", "status", "message_id": str | None}`. `axe_recorded`, `interest_recorded` and `flag_raised` take the model dump. `flag_resolved` takes `{"flag_id", "resolution"}`.
- Produces (tests): `tests.helpers.emit_quote(store, quote) -> Event`.

- [ ] **Step 1: Append the helper and write the failing tests**

In `tests/helpers.py`, add `EventType` to the existing `from brokerchat.models import ...` line, then append:

```python
def emit_quote(store, quote: Quote):
    return store.emit(
        EventType.QUOTE_UPSERTED,
        quote.model_dump(mode="json"),
        ts=quote.updated_ts,
        caused_by=quote.source_message_ids[-1],
    )
```

`tests/test_store.py`:

```python
import pytest

from brokerchat.models import EventType, Flag, FlagKind, QuoteStatus
from brokerchat.store import Board, Store, read_events, write_events
from tests.helpers import at, emit_quote, make_quote, msg


@pytest.fixture
def store(glossary):
    return Store(glossary)


def test_emit_appends_event_and_updates_board(store):
    event = emit_quote(store, make_quote())
    assert event.event_id == "e00001"
    assert store.events == [event]
    assert store.board.quotes["q0001"].bid == make_quote().bid


def test_status_change_updates_time_and_cites_message(store):
    emit_quote(store, make_quote())
    store.emit(EventType.QUOTE_STATUS_CHANGED, {"quote_id": "q0001", "status": "referred", "message_id": "m002"}, ts=at(5))
    q = store.board.quotes["q0001"]
    assert q.status == QuoteStatus.REFERRED
    assert q.updated_ts == at(5)
    assert q.source_message_ids == ["m001", "m002"]


def test_stale_status_keeps_original_update_time(store):
    emit_quote(store, make_quote())
    store.emit(EventType.QUOTE_STATUS_CHANGED, {"quote_id": "q0001", "status": "stale", "message_id": None}, ts=at(40))
    q = store.board.quotes["q0001"]
    assert q.status == QuoteStatus.STALE
    assert q.updated_ts == at(0)


def test_status_change_for_unknown_quote_is_not_logged(store):
    with pytest.raises(KeyError):
        store.emit(EventType.QUOTE_STATUS_CHANGED, {"quote_id": "q9999", "status": "traded", "message_id": None}, ts=at(1))
    assert store.events == []


def test_flags_raise_and_resolve(store):
    flag = Flag(flag_id="f0001", kind=FlagKind.AMBIGUOUS, question="Which tenor?", message_ids=["m001"])
    store.emit(EventType.FLAG_RAISED, flag.model_dump(mode="json"), ts=at(0))
    store.emit(EventType.FLAG_RESOLVED, {"flag_id": "f0001", "resolution": "5Y"}, ts=at(1))
    assert store.board.flags["f0001"].resolved is True
    assert store.board.flags["f0001"].resolution == "5Y"


def test_find_quote_matches_instrument_tenor_and_source(store):
    emit_quote(store, make_quote())
    assert store.board.find_quote("ITRAXX_EUR_MAIN", "5Y", "jpm_cds").quote_id == "q0001"
    assert store.board.find_quote("ITRAXX_EUR_MAIN", "5Y", "gs_flow") is None
    assert store.board.find_quote("ITRAXX_EUR_MAIN", "10Y", "jpm_cds") is None


def test_board_rebuilt_from_events_matches_live_board(store, tmp_path):
    emit_quote(store, make_quote())
    emit_quote(store, make_quote(quote_id="q0002", name="CDX_NA_IG", source="bnp_cds"))
    store.emit(EventType.QUOTE_STATUS_CHANGED, {"quote_id": "q0002", "status": "traded", "message_id": "m001"}, ts=at(3))
    path = tmp_path / "events.jsonl"
    write_events(store.events, path)
    rebuilt = Board.from_events(read_events(path))
    assert rebuilt.snapshot() == store.board.snapshot()


def test_ids_are_sequential(store):
    assert [store.next_quote_id(), store.next_quote_id()] == ["q0001", "q0002"]
    assert store.next_flag_id() == "f0001"


def test_duplicate_message_id_rejected(store):
    store.add_message(msg("m001", "hi"))
    with pytest.raises(ValueError, match="duplicate message id m001"):
        store.add_message(msg("m001", "again"))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'brokerchat.store'`.

- [ ] **Step 3: Implement the store**

`src/brokerchat/store.py`:

```python
"""Append-only event log and the board projection built from it (PRD §5)."""

from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from brokerchat.glossary import Glossary
from brokerchat.models import Axe, ChatMessage, Event, EventType, Flag, Interest, Quote, QuoteStatus


class Board:
    """Current state of the market, derived purely from events."""

    def __init__(self) -> None:
        self.quotes: dict[str, Quote] = {}
        self.axes: list[Axe] = []
        self.interests: list[Interest] = []
        self.flags: dict[str, Flag] = {}

    @classmethod
    def from_events(cls, events: Iterable[Event]) -> "Board":
        board = cls()
        for event in events:
            board.apply(event)
        return board

    def apply(self, event: Event) -> None:
        payload = event.payload
        match event.type:
            case EventType.QUOTE_UPSERTED:
                quote = Quote.model_validate(payload)
                self.quotes[quote.quote_id] = quote
            case EventType.QUOTE_STATUS_CHANGED:
                quote = self.quotes[payload["quote_id"]]
                status = QuoteStatus(payload["status"])
                ids = list(quote.source_message_ids)
                message_id = payload.get("message_id")
                if message_id and message_id not in ids:
                    ids.append(message_id)
                self.quotes[quote.quote_id] = quote.model_copy(
                    update={
                        "status": status,
                        # Going stale is not a market update, so the quote keeps its age.
                        "updated_ts": quote.updated_ts if status == QuoteStatus.STALE else event.ts,
                        "source_message_ids": ids,
                    }
                )
            case EventType.AXE_RECORDED:
                self.axes.append(Axe.model_validate(payload))
            case EventType.INTEREST_RECORDED:
                self.interests.append(Interest.model_validate(payload))
            case EventType.FLAG_RAISED:
                flag = Flag.model_validate(payload)
                self.flags[flag.flag_id] = flag
            case EventType.FLAG_RESOLVED:
                flag = self.flags[payload["flag_id"]]
                self.flags[flag.flag_id] = flag.model_copy(
                    update={"resolved": True, "resolution": payload["resolution"]}
                )

    def find_quote(self, name: str, tenor: str, source: str) -> Quote | None:
        for quote in self.quotes.values():
            if quote.instrument.name == name and quote.instrument.tenor == tenor and quote.source == source:
                return quote
        return None

    def snapshot(self) -> dict[str, Any]:
        return {
            "quotes": [q.model_dump(mode="json") for q in self.quotes.values()],
            "axes": [a.model_dump(mode="json") for a in self.axes],
            "interests": [i.model_dump(mode="json") for i in self.interests],
            "flags": [f.model_dump(mode="json") for f in self.flags.values()],
        }


class Store:
    """Owns the event log, the board projection and the chat messages seen so far."""

    def __init__(self, glossary: Glossary) -> None:
        self.glossary = glossary
        self.events: list[Event] = []
        self.board = Board()
        self.messages: dict[str, ChatMessage] = {}
        self._quote_seq = 0
        self._flag_seq = 0

    def add_message(self, message: ChatMessage) -> None:
        if message.id in self.messages:
            raise ValueError(f"duplicate message id {message.id}")
        self.messages[message.id] = message

    def emit(self, type: EventType, payload: dict[str, Any], ts: datetime, caused_by: str | None = None) -> Event:
        event = Event(
            event_id=f"e{len(self.events) + 1:05d}",
            ts=ts,
            type=type,
            payload=payload,
            caused_by_message_id=caused_by,
        )
        self.board.apply(event)  # apply first: an event the board rejects never enters the log
        self.events.append(event)
        return event

    def next_quote_id(self) -> str:
        self._quote_seq += 1
        return f"q{self._quote_seq:04d}"

    def next_flag_id(self) -> str:
        self._flag_seq += 1
        return f"f{self._flag_seq:04d}"


def write_events(events: Iterable[Event], path: Path) -> None:
    Path(path).write_text("".join(e.model_dump_json() + "\n" for e in events), encoding="utf-8")


def read_events(path: Path) -> list[Event]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [Event.model_validate_json(line) for line in lines if line.strip()]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_store.py -v`
Expected: all PASS.

- [ ] **Step 5: Checkpoint**

Run `uv run pytest -q` and expect all to pass. Tell the user Task 3 is ready. Do not commit.

---

### Task 4: Deterministic rules: staleness and crossed markets

**Files:**
- Create: `src/brokerchat/rules.py`
- Test: `tests/test_rules.py`

**Interfaces:**
- Consumes: `Store`, `EventType`, `Flag`, `FlagKind`, `QuoteStatus`, `fmt_decimal`.
- Produces: `DEFAULT_MAX_AGE = timedelta(minutes=30)`; `sweep_stale(store, now, max_age=DEFAULT_MAX_AGE) -> list[Event]`; `detect_conflicts(store, now) -> list[Event]`. A conflict flag id is `conflict:<lower quote id>:<higher quote id>`.

- [ ] **Step 1: Write the failing tests**

`tests/test_rules.py`:

```python
import pytest

from brokerchat.models import FlagKind, QuoteStatus
from brokerchat.rules import detect_conflicts, sweep_stale
from brokerchat.store import Store
from tests.helpers import at, emit_quote, make_quote


@pytest.fixture
def store(glossary):
    return Store(glossary)


def test_live_quote_older_than_30_minutes_goes_stale(store):
    emit_quote(store, make_quote(minutes=0))
    events = sweep_stale(store, now=at(31))
    assert len(events) == 1
    assert store.board.quotes["q0001"].status == QuoteStatus.STALE


def test_quote_exactly_30_minutes_old_stays_live(store):
    emit_quote(store, make_quote(minutes=0))
    assert sweep_stale(store, now=at(30)) == []
    assert store.board.quotes["q0001"].status == QuoteStatus.LIVE


def test_non_live_quotes_are_left_alone(store):
    emit_quote(store, make_quote(status=QuoteStatus.REFERRED))
    assert sweep_stale(store, now=at(120)) == []


def test_crossed_markets_from_two_sources_raise_one_conflict(store):
    emit_quote(store, make_quote(quote_id="q0001", bid="58", offer="58.5", source="jpm_cds", ids=("m001",)))
    emit_quote(store, make_quote(quote_id="q0002", bid="57", offer="57.5", source="gs_flow", ids=("m002",)))
    events = detect_conflicts(store, now=at(1))
    assert len(events) == 1
    flag = store.board.flags["conflict:q0001:q0002"]
    assert flag.kind == FlagKind.CONFLICT
    assert flag.message_ids == ["m001", "m002"]
    assert detect_conflicts(store, now=at(2)) == []  # not raised twice


def test_touching_markets_count_as_crossed(store):
    emit_quote(store, make_quote(quote_id="q0001", bid="58", offer="58.5", source="jpm_cds"))
    emit_quote(store, make_quote(quote_id="q0002", bid="58.5", offer="59", source="gs_flow"))
    assert len(detect_conflicts(store, now=at(1))) == 1


def test_no_conflict_for_same_source_or_different_tenor(store):
    emit_quote(store, make_quote(quote_id="q0001", bid="58", offer="58.5", source="jpm_cds"))
    emit_quote(store, make_quote(quote_id="q0002", bid="57", offer="57.5", source="jpm_cds", tenor="10Y"))
    emit_quote(store, make_quote(quote_id="q0003", bid="56.5", offer="58", source="gs_flow", tenor="10Y"))
    assert detect_conflicts(store, now=at(1)) == []


def test_one_sided_quotes_only_cross_on_the_sides_they_have(store):
    emit_quote(store, make_quote(quote_id="q0001", bid="58", offer=None, source="jpm_cds"))
    emit_quote(store, make_quote(quote_id="q0002", bid="59", offer=None, source="gs_flow"))
    assert detect_conflicts(store, now=at(1)) == []
```

This covers both cases. `q0001` and `q0002` share a source. `q0003` (gs_flow 10Y 56.5/58) doesn't cross `q0002` (jpm 10Y 57/57.5), because 57 < 58 and 56.5 < 57.5, and it is a different tenor from `q0001`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_rules.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'brokerchat.rules'`.

- [ ] **Step 3: Implement the rules**

`src/brokerchat/rules.py`:

```python
"""Deterministic board rules that the LLM never decides: staleness and crossed markets."""

from datetime import datetime, timedelta

from brokerchat.models import Event, EventType, Flag, FlagKind, Quote, QuoteStatus, fmt_decimal
from brokerchat.store import Store

DEFAULT_MAX_AGE = timedelta(minutes=30)


def sweep_stale(store: Store, now: datetime, max_age: timedelta = DEFAULT_MAX_AGE) -> list[Event]:
    events = []
    for quote in list(store.board.quotes.values()):
        if quote.status == QuoteStatus.LIVE and now - quote.updated_ts > max_age:
            events.append(
                store.emit(
                    EventType.QUOTE_STATUS_CHANGED,
                    {"quote_id": quote.quote_id, "status": QuoteStatus.STALE.value, "message_id": None},
                    ts=now,
                )
            )
    return events


def _crossed(a: Quote, b: Quote) -> bool:
    return a.bid is not None and b.offer is not None and a.bid >= b.offer


def _market(q: Quote) -> str:
    bid = "-" if q.bid is None else fmt_decimal(q.bid)
    offer = "-" if q.offer is None else fmt_decimal(q.offer)
    return f"{bid}/{offer}"


def detect_conflicts(store: Store, now: datetime) -> list[Event]:
    live = [q for q in store.board.quotes.values() if q.status == QuoteStatus.LIVE]
    events = []
    for i, a in enumerate(live):
        for b in live[i + 1 :]:
            if a.instrument != b.instrument or a.source == b.source:
                continue
            if not (_crossed(a, b) or _crossed(b, a)):
                continue
            low, high = sorted([a, b], key=lambda q: q.quote_id)
            flag_id = f"conflict:{low.quote_id}:{high.quote_id}"
            if flag_id in store.board.flags:
                continue
            flag = Flag(
                flag_id=flag_id,
                kind=FlagKind.CONFLICT,
                question=(
                    f"Crossed market in {a.instrument.name} {a.instrument.tenor}: "
                    f"{low.source} {_market(low)} vs {high.source} {_market(high)}. Which is right?"
                ),
                message_ids=sorted({low.source_message_ids[-1], high.source_message_ids[-1]}),
            )
            events.append(store.emit(EventType.FLAG_RAISED, flag.model_dump(mode="json"), ts=now))
    return events
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_rules.py -v`
Expected: all PASS.

- [ ] **Step 5: Checkpoint**

Run `uv run pytest -q` and expect all to pass. Tell the user Task 4 is ready. Do not commit.

---

### Task 5: Chat file I/O, hand-written examples and board rendering

**Files:**
- Create: `src/brokerchat/chatio.py`, `src/brokerchat/render.py`, `examples/handwritten.jsonl`
- Test: `tests/test_chatio.py`, `tests/test_render.py`

**Interfaces:**
- Consumes: `ChatMessage`, `Board`, `fmt_decimal`.
- Produces: `ChatFileError(ValueError)`; `read_chat(path) -> list[ChatMessage]`, which skips blank lines and raises `ChatFileError` with `"<path>:<line>: ..."` for bad lines, duplicate ids or backwards timestamps; `write_chat(messages, path)`; `render_board(board) -> rich.console.Group`.

- [ ] **Step 1: Write the 20 hand-written example messages (PRD M0)**

`examples/handwritten.jsonl`:

```
{"id":"h01","ts":"2026-10-01T08:00:05Z","room":"ldn-cds-1","sender":"jpm_cds","text":"morning all"}
{"id":"h02","ts":"2026-10-01T08:01:10Z","room":"ldn-cds-1","sender":"jpm_cds","text":"ITRX Main 5y 58/58.5 10x10"}
{"id":"h03","ts":"2026-10-01T08:02:30Z","room":"ldn-cds-1","sender":"gs_flow","text":"XO 5y 302/304 5x5"}
{"id":"h04","ts":"2026-10-01T08:03:00Z","room":"ldn-cds-1","sender":"bnp_cds","text":"CDX IG 5y 52.25 bid"}
{"id":"h05","ts":"2026-10-01T08:04:45Z","room":"ldn-cds-2","sender":"citi_ix","text":"cdx hy 5y 104.5/104.625"}
{"id":"h06","ts":"2026-10-01T08:06:00Z","room":"ldn-cds-1","sender":"jpm_cds","text":"same in 10y 84/85"}
{"id":"h07","ts":"2026-10-01T08:07:20Z","room":"ldn-cds-1","sender":"gs_flow","text":"who's axed XO?"}
{"id":"h08","ts":"2026-10-01T08:08:00Z","room":"ldn-cds-1","sender":"bnp_cds","text":"axed to buy protection XO 5y 25mm"}
{"id":"h09","ts":"2026-10-01T08:09:15Z","room":"ldn-cds-2","sender":"citi_ix","text":"VW 5y 120/125"}
{"id":"h10","ts":"2026-10-01T08:10:00Z","room":"ldn-cds-2","sender":"citi_ix","text":"refer"}
{"id":"h11","ts":"2026-10-01T08:11:30Z","room":"ldn-cds-1","sender":"gs_flow","text":"1 tighter"}
{"id":"h12","ts":"2026-10-01T08:12:40Z","room":"ldn-cds-1","sender":"jpm_cds","text":"10y unch"}
{"id":"h13","ts":"2026-10-01T08:14:00Z","room":"ldn-cds-2","sender":"bnp_cds","text":"Alder Capital looking to sell protection VW 5y"}
{"id":"h14","ts":"2026-10-01T08:15:20Z","room":"ldn-cds-1","sender":"gs_flow","text":"trades 301"}
{"id":"h15","ts":"2026-10-01T08:16:00Z","room":"ldn-cds-1","sender":"citi_ix","text":"Main 7y 72/73 25x25"}
{"id":"h16","ts":"2026-10-01T08:17:30Z","room":"ldn-cds-2","sender":"jpm_cds","text":"DB 5y 95/100"}
{"id":"h17","ts":"2026-10-01T08:18:10Z","room":"ldn-cds-2","sender":"jpm_cds","text":"off that"}
{"id":"h18","ts":"2026-10-01T08:19:00Z","room":"desk-misc","sender":"bnp_cds","text":"61/62 10x10"}
{"id":"h19","ts":"2026-10-01T08:20:30Z","room":"ldn-cds-1","sender":"gs_flow","text":"ZXQ 5y 88/90"}
{"id":"h20","ts":"2026-10-01T08:22:00Z","room":"ldn-cds-1","sender":"jpm_cds","text":"ty"}
```

- [ ] **Step 2: Write the failing tests**

`tests/test_chatio.py`:

```python
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
```

`tests/test_render.py`:

```python
from rich.console import Console

from brokerchat.models import Flag, FlagKind
from brokerchat.render import render_board
from brokerchat.store import Board
from tests.helpers import make_quote


def render_text(board: Board) -> str:
    console = Console(record=True, width=200)
    console.print(render_board(board))
    return console.export_text()


def test_render_shows_quotes_and_open_flags():
    board = Board()
    q = make_quote(name="CDX_NA_HY", bid="104.5", offer="104.625")
    board.quotes[q.quote_id] = q
    board.flags["f0001"] = Flag(flag_id="f0001", kind=FlagKind.AMBIGUOUS, question="Which tenor?", message_ids=["m001"])
    board.flags["f0002"] = Flag(
        flag_id="f0002", kind=FlagKind.AMBIGUOUS, question="Already answered", message_ids=["m002"], resolved=True
    )
    text = render_text(board)
    assert "CDX_NA_HY" in text
    assert "104.625" in text
    assert "Which tenor?" in text
    assert "Already answered" not in text


def test_render_empty_board():
    assert "Quotes" in render_text(Board())
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_chatio.py tests/test_render.py -v`
Expected: FAIL with `ModuleNotFoundError` for `brokerchat.chatio` and `brokerchat.render`.

- [ ] **Step 4: Implement chat I/O**

`src/brokerchat/chatio.py`:

```python
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
```

- [ ] **Step 5: Implement rendering**

`src/brokerchat/render.py`:

```python
"""Terminal rendering of the board with rich."""

from decimal import Decimal

from rich.console import Group
from rich.table import Table

from brokerchat.models import QuoteStatus, fmt_decimal
from brokerchat.store import Board

STATUS_STYLE = {
    QuoteStatus.LIVE: "green",
    QuoteStatus.REFERRED: "yellow",
    QuoteStatus.TRADED: "cyan",
    QuoteStatus.STALE: "dim",
}


def _num(x: Decimal | None) -> str:
    return "-" if x is None else fmt_decimal(x)


def render_board(board: Board) -> Group:
    quotes = Table(title="Quotes", expand=True)
    for column in ("ID", "Instrument", "Tenor", "Bid", "Offer", "Size", "Source", "Status", "Updated"):
        quotes.add_column(column)
    for q in sorted(board.quotes.values(), key=lambda q: (q.instrument.name, q.instrument.tenor, q.source)):
        size = "" if q.bid_size is None and q.offer_size is None else f"{_num(q.bid_size)}x{_num(q.offer_size)}"
        quotes.add_row(
            q.quote_id,
            q.instrument.name,
            q.instrument.tenor,
            _num(q.bid),
            _num(q.offer),
            size,
            q.source,
            q.status.value,
            q.updated_ts.strftime("%H:%M:%S"),
            style=STATUS_STYLE[q.status],
        )

    axes = Table(title="Axes", expand=True)
    for column in ("Instrument", "Tenor", "Direction", "Size", "Source"):
        axes.add_column(column)
    for a in board.axes:
        axes.add_row(a.instrument.name, a.instrument.tenor, a.direction.value, _num(a.size), a.source)

    interests = Table(title="Client interest", expand=True)
    for column in ("Client", "Instrument", "Tenor", "Direction", "Note"):
        interests.add_column(column)
    for i in board.interests:
        interests.add_row(i.client, i.instrument.name, i.instrument.tenor, i.direction.value, i.note or "")

    flags = Table(title="Open flags", expand=True)
    for column in ("ID", "Kind", "Question", "Messages"):
        flags.add_column(column)
    for f in board.flags.values():
        if not f.resolved:
            flags.add_row(f.flag_id, f.kind.value, f.question, ", ".join(f.message_ids))

    return Group(quotes, axes, interests, flags)
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_chatio.py tests/test_render.py -v`
Expected: all PASS.

- [ ] **Step 7: Checkpoint**

Run `uv run pytest -q` and expect all to pass. Tell the user Task 5 is ready. Do not commit.

---

### Task 6: Pipeline, null agent and `brokerchat replay`

**Files:**
- Create: `src/brokerchat/pipeline.py`, `src/brokerchat/cli.py`
- Test: `tests/test_pipeline.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: `Store`, `sweep_stale`, `detect_conflicts`, `DEFAULT_MAX_AGE`, `read_chat`, `render_board`, `write_events`, `Glossary`.
- Produces:
  - `Usage` (pydantic, int fields `input_tokens`, `output_tokens`, `cache_read_input_tokens`, `cache_creation_input_tokens`; `.add(other)`, `Usage.from_api(api_usage)`, `.cost_usd() -> float`).
  - `AgentResult(message_id: str, status: Literal["ok", "unprocessed", "cap_hit"], reason: str | None = None, tool_calls: int = 0, usage: Usage)`.
  - `Processor` protocol with `process(message: ChatMessage, store: Store) -> AgentResult`, and `NullAgent`.
  - `Pipeline(store, agent, max_age=DEFAULT_MAX_AGE)` with `.step(message) -> AgentResult`, `.run(messages, on_step=None) -> list[AgentResult]`, `.results`, `.total_usage() -> Usage`.
  - `write_run_outputs(store, results, out_dir)` writes `board.json`, `events.jsonl` and `results.jsonl`.
  - `cli.main(argv=None) -> int`, `cli.build_parser()`, `cli.make_agent(name: str, glossary: Glossary, effort: str = "medium") -> Processor`, `cli.AGENT_CHOICES: list[str]`, `cli.EFFORT_CHOICES`.

- [ ] **Step 1: Write the failing tests**

`tests/test_pipeline.py`:

```python
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
```

`tests/test_cli.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_pipeline.py tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError` for `brokerchat.pipeline` and `brokerchat.cli`.

- [ ] **Step 3: Implement the pipeline**

`src/brokerchat/pipeline.py`:

```python
"""Feeds chat messages through an agent, then applies the deterministic rules."""

import json
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Protocol

from pydantic import BaseModel, Field

from brokerchat.models import ChatMessage
from brokerchat.rules import DEFAULT_MAX_AGE, detect_conflicts, sweep_stale
from brokerchat.store import Store, write_events

# claude-opus-5-5 list prices in USD per million tokens (2026-09). Cache writes are 1.25x input (5-minute TTL).
PRICE_PER_MTOK = {"input": 4.00, "output": 20.00, "cache_read": 0.20, "cache_write": 5.00}


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    @classmethod
    def from_api(cls, usage: Any) -> "Usage":
        return cls(
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
        )

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_input_tokens += other.cache_read_input_tokens
        self.cache_creation_input_tokens += other.cache_creation_input_tokens

    def cost_usd(self) -> float:
        return (
            self.input_tokens * PRICE_PER_MTOK["input"]
            + self.output_tokens * PRICE_PER_MTOK["output"]
            + self.cache_read_input_tokens * PRICE_PER_MTOK["cache_read"]
            + self.cache_creation_input_tokens * PRICE_PER_MTOK["cache_write"]
        ) / 1_000_000


class AgentResult(BaseModel):
    message_id: str
    status: Literal["ok", "unprocessed", "cap_hit"]
    reason: str | None = None
    tool_calls: int = 0
    usage: Usage = Field(default_factory=Usage)


class Processor(Protocol):
    def process(self, message: ChatMessage, store: Store) -> AgentResult: ...


class NullAgent:
    """Does nothing. Used to test the pipeline and eval harness without calling the API."""

    def process(self, message: ChatMessage, store: Store) -> AgentResult:
        return AgentResult(message_id=message.id, status="ok")


class Pipeline:
    def __init__(self, store: Store, agent: Processor, max_age: timedelta = DEFAULT_MAX_AGE) -> None:
        self.store = store
        self.agent = agent
        self.max_age = max_age
        self.results: list[AgentResult] = []

    def step(self, message: ChatMessage) -> AgentResult:
        self.store.add_message(message)
        result = self.agent.process(message, self.store)
        sweep_stale(self.store, message.ts, self.max_age)
        detect_conflicts(self.store, message.ts)
        self.results.append(result)
        return result

    def run(
        self,
        messages: Iterable[ChatMessage],
        on_step: Callable[[ChatMessage, AgentResult], None] | None = None,
    ) -> list[AgentResult]:
        for message in messages:
            result = self.step(message)
            if on_step is not None:
                on_step(message, result)
        return self.results

    def total_usage(self) -> Usage:
        total = Usage()
        for result in self.results:
            total.add(result.usage)
        return total


def write_run_outputs(store: Store, results: list[AgentResult], out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "board.json").write_text(json.dumps(store.board.snapshot(), indent=2), encoding="utf-8")
    write_events(store.events, out_dir / "events.jsonl")
    (out_dir / "results.jsonl").write_text("".join(r.model_dump_json() + "\n" for r in results), encoding="utf-8")
```

- [ ] **Step 4: Implement the CLI with `replay`**

`src/brokerchat/cli.py`:

```python
"""Command-line entry point: brokerchat replay | generate | eval."""

import argparse
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.live import Live

from brokerchat.chatio import read_chat
from brokerchat.glossary import Glossary
from brokerchat.pipeline import NullAgent, Pipeline, Processor, write_run_outputs
from brokerchat.render import render_board
from brokerchat.store import Store

AGENT_CHOICES = ["null"]
EFFORT_CHOICES = ["low", "medium", "high", "xhigh", "max"]

console = Console()


def make_agent(name: str, glossary: Glossary, effort: str = "medium") -> Processor:
    if name == "null":
        return NullAgent()
    raise ValueError(f"unknown agent {name!r}")


def cmd_replay(args: argparse.Namespace) -> int:
    glossary = Glossary.load()
    messages = read_chat(args.chat_file)
    store = Store(glossary)
    pipeline = Pipeline(store, make_agent(args.agent, glossary, args.effort))

    if args.no_live:
        pipeline.run(messages)
        console.print(render_board(store.board))
    else:
        with Live(render_board(store.board), console=console, refresh_per_second=8) as live:
            previous = None
            for message in messages:
                if args.speed > 0 and previous is not None:
                    time.sleep((message.ts - previous).total_seconds() / args.speed)
                pipeline.step(message)
                live.update(render_board(store.board))
                previous = message.ts

    not_ok = [r for r in pipeline.results if r.status != "ok"]
    console.print(
        f"{len(messages)} messages, {len(not_ok)} not fully processed, "
        f"approx ${pipeline.total_usage().cost_usd():.2f} API cost"
    )
    for result in not_ok:
        console.print(f"  [yellow]{result.message_id}: {result.status} ({result.reason})[/yellow]")
    if args.out is not None:
        write_run_outputs(store, pipeline.results, args.out)
        console.print(f"Outputs written to {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="brokerchat", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    replay = sub.add_parser("replay", help="Replay a chat file through the agent and show the live board")
    replay.add_argument("chat_file", type=Path)
    replay.add_argument("--speed", type=float, default=0.0, help="Replay speed multiplier; 0 means no delays")
    replay.add_argument("--agent", choices=AGENT_CHOICES, default=AGENT_CHOICES[0])
    replay.add_argument("--effort", choices=EFFORT_CHOICES, default="medium")
    replay.add_argument("--out", type=Path, default=None, help="Directory for board.json, events.jsonl, results.jsonl")
    replay.add_argument("--no-live", action="store_true", help="Print the final board instead of a live view")
    replay.set_defaults(func=cmd_replay)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_pipeline.py tests/test_cli.py -v`
Expected: all PASS.

- [ ] **Step 6: Smoke-test the live view by hand**

Run: `uv run brokerchat replay examples/handwritten.jsonl --agent null --speed 600`
Expected: an empty board updates for about 2 seconds, then the summary line `20 messages, 0 not fully processed, approx $0.00 API cost`.

- [ ] **Step 7: Checkpoint**

Run `uv run pytest -q` and expect all to pass. Tell the user Task 6 is ready. This completes milestone M1: the board renders from an event log with no LLM. Do not commit.

---

### Task 7: Expected-board models and the scenario builder

**Files:**
- Create: `src/brokerchat/expected.py`, `src/brokerchat/generator/__init__.py`, `src/brokerchat/generator/builder.py`
- Test: `tests/test_builder.py`

**Interfaces:**
- Consumes: `Glossary`, models, `DEFAULT_MAX_AGE`, `read_chat`/`write_chat`, `fmt_decimal`.
- Produces:
  - `expected.py`: `ExpectedQuote(instrument, tenor, source, bid, offer, unit, bid_size, offer_size, status, updated_ts)`, `ExpectedAxe(instrument, tenor, direction, size, source)`, `ExpectedInterest(client, instrument, tenor, direction)`, `ExpectedBoard(name, end_ts, quotes, axes, interests, ambiguous_message_ids)`, `GeneratedCase(messages, expected)`, `write_case(case, directory) -> Path` (writes `<name>.jsonl` and `<name>.expected.json` and returns the chat path), `read_case(chat_path) -> GeneratedCase`. In these models `instrument` is the canonical name `str`.
  - `builder.py`: `ScenarioError(ValueError)`; `DayBuilder(glossary, name, seed=0, start=START, gap_range=(20, 400))`.
    - Public state: `.messages`, `.quotes: dict[(instrument, tenor, source), ExpectedQuote]`, `.last_quote: dict[(room, sender), key]`, `.last_action: dict[(room, sender), str]`, `.ambiguous_ids`.
    - Ops: `quote(room, sender, instrument, tenor, bid, offer, size=None)`, `one_sided(room, sender, instrument, tenor, side, level)`, `refer(room, sender)`, `trade(room, sender)`, `shift(room, sender, bp)`, `unch(room, sender)`, `same_again(room, sender, tenor, bid, offer)`, `axe(room, sender, instrument, tenor, direction, size)`, `interest(room, sender, client, instrument, tenor, direction)`, `chatter(room, sender)`, `ambiguous_unknown(room, sender, bid, offer)`, `ambiguous_no_context(sender, bid, offer, size)`, and `build() -> GeneratedCase`.
    - Also: `OPS: frozenset[str]`, `run_scenario(glossary, data: dict) -> GeneratedCase`, `load_scenario(glossary, path) -> GeneratedCase`.
  - The `last_action` value is `"quote"` after any op that leaves the sender's last market live (`quote`, `one_sided`, `shift`, `unch`, `same_again`). Otherwise it is the op name: `refer`, `trade`, `axe`, `interest`, `chatter` or `ambiguous`.

- [ ] **Step 1: Write the failing tests**

`tests/test_builder.py`:

```python
from decimal import Decimal

import pytest

from brokerchat.expected import read_case, write_case
from brokerchat.generator.builder import REFER_PHRASES, DayBuilder, ScenarioError, run_scenario
from brokerchat.models import Direction, QuoteStatus, Unit

R = "ldn-cds-1"


@pytest.fixture
def builder(glossary):
    return DayBuilder(glossary, name="t", seed=7)


def key(instrument="ITRAXX_EUR_MAIN", tenor="5Y", source="jpm_cds"):
    return (instrument, tenor, source)


def test_quote_writes_alias_and_levels(builder):
    builder.quote(R, "jpm_cds", "ITRAXX_EUR_MAIN", "5Y", 58, 58.5, size=10)
    text = builder.messages[-1].text
    assert "58/58.5" in text and "10x10" in text
    q = builder.quotes[key()]
    assert (q.bid, q.offer, q.bid_size, q.unit) == (Decimal("58"), Decimal("58.5"), Decimal("10"), Unit.BP)
    assert builder.last_action[(R, "jpm_cds")] == "quote"


def test_refer_marks_last_market_referred(builder):
    builder.quote(R, "jpm_cds", "ITRAXX_EUR_MAIN", "5Y", 58, 58.5)
    builder.refer(R, "jpm_cds")
    assert builder.messages[-1].text in REFER_PHRASES
    assert builder.quotes[key()].status == QuoteStatus.REFERRED
    assert builder.last_action[(R, "jpm_cds")] == "refer"


def test_refer_without_previous_market_fails(builder):
    with pytest.raises(ScenarioError, match="no previous market"):
        builder.refer(R, "jpm_cds")


def test_trade_mentions_a_quoted_level(builder):
    builder.quote(R, "gs_flow", "ITRAXX_EUR_XOVER", "5Y", 302, 304)
    builder.trade(R, "gs_flow")
    assert builder.messages[-1].text.split()[-1] in ("302", "304")
    assert builder.quotes[key("ITRAXX_EUR_XOVER", source="gs_flow")].status == QuoteStatus.TRADED


def test_shift_wider_moves_both_sides(builder):
    builder.quote(R, "jpm_cds", "ITRAXX_EUR_MAIN", "5Y", 58, 58.5)
    builder.shift(R, "jpm_cds", 1)
    q = builder.quotes[key()]
    assert (q.bid, q.offer) == (Decimal("59"), Decimal("59.5"))
    assert "wider" in builder.messages[-1].text


def test_shift_rejects_price_quoted_market(builder):
    builder.quote(R, "citi_ix", "CDX_NA_HY", "5Y", 104.5, 104.625)
    with pytest.raises(ScenarioError, match="bp"):
        builder.shift(R, "citi_ix", 1)


def test_one_sided_offer(builder):
    builder.one_sided(R, "bnp_cds", "CDX_NA_IG", "5Y", "offer", 52.75)
    q = builder.quotes[key("CDX_NA_IG", source="bnp_cds")]
    assert q.bid is None and q.offer == Decimal("52.75")


def test_same_again_quotes_new_tenor_of_same_instrument(builder):
    builder.quote(R, "jpm_cds", "VOLKSWAGEN_AG", "5Y", 120, 125)
    builder.same_again(R, "jpm_cds", "10Y", 165, 172)
    q = builder.quotes[key("VOLKSWAGEN_AG", "10Y")]
    assert (q.bid, q.offer) == (Decimal("165"), Decimal("172"))
    assert builder.last_quote[(R, "jpm_cds")] == key("VOLKSWAGEN_AG", "10Y")


def test_axe_and_interest_are_expected(builder):
    builder.axe(R, "bnp_cds", "ITRAXX_EUR_XOVER", "5Y", "buy_protection", 25)
    builder.interest(R, "bnp_cds", "Alder Capital", "VOLKSWAGEN_AG", "5Y", "sell_protection")
    case = builder.build()
    assert case.expected.axes[0].direction == Direction.BUY_PROTECTION
    assert case.expected.interests[0].client == "Alder Capital"
    assert "protection" in builder.messages[0].text


def test_ambiguous_messages_are_recorded(builder, glossary):
    builder.ambiguous_unknown(R, "gs_flow", 88, 90)
    builder.ambiguous_no_context("bnp_cds", 61, 62, 10)
    assert builder.ambiguous_ids == ["m001", "m002"]
    assert glossary.lookup(builder.messages[0].text.split()[0]) is None
    assert builder.messages[1].room == "desk-misc"


def test_build_marks_old_live_quotes_stale(glossary):
    b = DayBuilder(glossary, name="slow", seed=1, gap_range=(2000, 2000))
    b.quote(R, "jpm_cds", "ITRAXX_EUR_MAIN", "5Y", 58, 58.5)
    b.chatter(R, "gs_flow")
    case = b.build()
    assert case.expected.quotes[0].status == QuoteStatus.STALE


def test_same_seed_same_text(glossary):
    texts = []
    for _ in range(2):
        b = DayBuilder(glossary, name="t", seed=3)
        b.quote(R, "jpm_cds", "ITRAXX_EUR_MAIN", "5Y", 58, 58.5)
        texts.append(b.messages[0].text)
    assert texts[0] == texts[1]


def test_run_scenario_rejects_unknown_op(glossary):
    with pytest.raises(ScenarioError, match="step 1: unknown op 'teleport'"):
        run_scenario(glossary, {"name": "x", "steps": [{"op": "teleport"}]})


def test_run_scenario_reports_bad_arguments(glossary):
    with pytest.raises(ScenarioError, match=r"step 1 \(quote\)"):
        run_scenario(glossary, {"name": "x", "steps": [{"op": "quote", "room": R}]})


def test_case_roundtrip(builder, tmp_path):
    builder.quote(R, "jpm_cds", "ITRAXX_EUR_MAIN", "5Y", 58, 58.5)
    case = builder.build()
    path = write_case(case, tmp_path)
    assert path.name == "t.jsonl"
    assert (tmp_path / "t.expected.json").exists()
    assert read_case(path) == case
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_builder.py -v`
Expected: FAIL with `ModuleNotFoundError` for `brokerchat.expected`.

- [ ] **Step 3: Implement the expected-board models**

`src/brokerchat/expected.py`:

```python
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
```

- [ ] **Step 4: Implement the builder**

`src/brokerchat/generator/__init__.py`: empty file.

`src/brokerchat/generator/builder.py`:

```python
"""Builds a synthetic day of broker chat together with the board it should produce (PRD §7).

The scenario is the source of truth: every op writes one chat message and updates the
expected board, so each generated chat comes with exact ground truth.
"""

import random
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

from brokerchat.expected import ExpectedAxe, ExpectedBoard, ExpectedInterest, ExpectedQuote, GeneratedCase
from brokerchat.glossary import Glossary
from brokerchat.models import ChatMessage, Direction, QuoteStatus, Unit, fmt_decimal
from brokerchat.rules import DEFAULT_MAX_AGE

START = datetime(2026, 10, 1, 8, 0, tzinfo=UTC)
AMBIGUOUS_ROOM = "desk-misc"
UNKNOWN_ALIASES = ("ZXQ", "KRTL", "MFNX")
CHATTER = ("morning all", "gm", "ty", "anyone seeing flows today?", "lunch?")
REFER_PHRASES = ("refer", "off that", "pulled", "refer that")
UNCH_PHRASES = ("unch", "still there", "unchanged")
OPS = frozenset(
    {
        "quote", "one_sided", "refer", "trade", "shift", "unch", "same_again",
        "axe", "interest", "chatter", "ambiguous_unknown", "ambiguous_no_context",
    }
)

QuoteKey = tuple[str, str, str]  # (instrument, tenor, source)


class ScenarioError(ValueError):
    pass


def D(x: Any) -> Decimal:
    return x if isinstance(x, Decimal) else Decimal(str(x))


class DayBuilder:
    def __init__(
        self,
        glossary: Glossary,
        name: str,
        seed: int = 0,
        start: datetime = START,
        gap_range: tuple[int, int] = (20, 400),
    ) -> None:
        self.glossary = glossary
        self.name = name
        self.rng = random.Random(seed)
        self.clock = start
        self.gap_range = gap_range
        self.messages: list[ChatMessage] = []
        self.quotes: dict[QuoteKey, ExpectedQuote] = {}
        self.last_quote: dict[tuple[str, str], QuoteKey] = {}
        self.last_action: dict[tuple[str, str], str] = {}
        self.axes: list[ExpectedAxe] = []
        self.interests: list[ExpectedInterest] = []
        self.ambiguous_ids: list[str] = []

    # ---- helpers -------------------------------------------------------------

    def _alias(self, instrument: str) -> str:
        info = self.glossary.get(instrument)
        if info is None:
            raise ScenarioError(f"unknown instrument {instrument!r}")
        alias = self.rng.choice(info.aliases)
        return self.rng.choice([alias, alias.upper(), alias.title()])

    def _tenor(self, tenor: str) -> str:
        if tenor not in self.glossary.tenors:
            raise ScenarioError(f"unknown tenor {tenor!r}")
        n = tenor.removesuffix("Y")
        return self.rng.choice([f"{n}y", f"{n}Y", f"{n}yr"])

    def _emit(self, room: str, sender: str, text: str) -> ChatMessage:
        self.clock += timedelta(seconds=self.rng.randint(*self.gap_range))
        message = ChatMessage(id=f"m{len(self.messages) + 1:03d}", ts=self.clock, room=room, sender=sender, text=text)
        self.messages.append(message)
        return message

    def _set_quote(self, room, sender, instrument, tenor, bid, offer, bid_size=None, offer_size=None) -> None:
        key = (instrument, tenor, sender)
        self.quotes[key] = ExpectedQuote(
            instrument=instrument,
            tenor=tenor,
            source=sender,
            bid=bid,
            offer=offer,
            unit=self.glossary.get(instrument).unit,
            bid_size=bid_size,
            offer_size=offer_size,
            status=QuoteStatus.LIVE,
            updated_ts=self.clock,
        )
        self.last_quote[(room, sender)] = key
        self.last_action[(room, sender)] = "quote"

    def _last_key(self, room: str, sender: str) -> QuoteKey:
        key = self.last_quote.get((room, sender))
        if key is None:
            raise ScenarioError(f"{sender} has no previous market in {room}")
        return key

    def _update(self, key: QuoteKey, **changes: Any) -> None:
        self.quotes[key] = self.quotes[key].model_copy(update={**changes, "updated_ts": self.clock})

    # ---- ops: each writes exactly one chat message ---------------------------

    def quote(self, room, sender, instrument, tenor, bid, offer, size=None) -> None:
        bid, offer = D(bid), D(offer)
        size = None if size is None else D(size)
        text = f"{self._alias(instrument)} {self._tenor(tenor)} {fmt_decimal(bid)}/{fmt_decimal(offer)}"
        if size is not None:
            text += f" {fmt_decimal(size)}x{fmt_decimal(size)}"
        self._emit(room, sender, text)
        self._set_quote(room, sender, instrument, tenor, bid, offer, size, size)

    def one_sided(self, room, sender, instrument, tenor, side, level) -> None:
        if side not in ("bid", "offer"):
            raise ScenarioError(f"side must be 'bid' or 'offer', not {side!r}")
        level = D(level)
        a, t, lv = self._alias(instrument), self._tenor(tenor), fmt_decimal(level)
        if side == "bid":
            templates = [f"{a} {t} {lv} bid", f"{lv} bid for {a} {t}"]
        else:
            templates = [f"{a} {t} offered at {lv}", f"{a} {t} {lv} offer"]
        self._emit(room, sender, self.rng.choice(templates))
        self._set_quote(
            room, sender, instrument, tenor, level if side == "bid" else None, level if side == "offer" else None
        )

    def refer(self, room, sender) -> None:
        key = self._last_key(room, sender)
        self._emit(room, sender, self.rng.choice(REFER_PHRASES))
        self._update(key, status=QuoteStatus.REFERRED)
        self.last_action[(room, sender)] = "refer"

    def trade(self, room, sender) -> None:
        key = self._last_key(room, sender)
        q = self.quotes[key]
        level = fmt_decimal(self.rng.choice([x for x in (q.bid, q.offer) if x is not None]))
        self._emit(room, sender, self.rng.choice([f"trades {level}", f"done at {level}", f"dealt {level}"]))
        self._update(key, status=QuoteStatus.TRADED)
        self.last_action[(room, sender)] = "trade"

    def shift(self, room, sender, bp) -> None:
        key = self._last_key(room, sender)
        q = self.quotes[key]
        if q.unit != Unit.BP:
            raise ScenarioError("shift only applies to markets quoted in bp")
        bp = D(bp)
        if bp == 0:
            raise ScenarioError("shift needs a non-zero bp move")
        word = "wider" if bp > 0 else "tighter"
        n = fmt_decimal(abs(bp))
        self._emit(room, sender, self.rng.choice([f"{n} {word}", f"move {n}bp {word}", f"{word} by {n}"]))
        self._update(
            key,
            status=QuoteStatus.LIVE,
            bid=None if q.bid is None else q.bid + bp,
            offer=None if q.offer is None else q.offer + bp,
        )
        self.last_action[(room, sender)] = "quote"

    def unch(self, room, sender) -> None:
        key = self._last_key(room, sender)
        self._emit(room, sender, self.rng.choice(UNCH_PHRASES))
        self._update(key, status=QuoteStatus.LIVE)
        self.last_action[(room, sender)] = "quote"

    def same_again(self, room, sender, tenor, bid, offer) -> None:
        instrument, last_tenor, _ = self._last_key(room, sender)
        if tenor == last_tenor:
            raise ScenarioError("same_again needs a different tenor")
        bid, offer = D(bid), D(offer)
        t, n = self._tenor(tenor), tenor.removesuffix("Y")
        market = f"{fmt_decimal(bid)}/{fmt_decimal(offer)}"
        self._emit(room, sender, self.rng.choice([f"same in {t} {market}", f"and {t} {market}", f"{n}s {market}"]))
        self._set_quote(room, sender, instrument, tenor, bid, offer)

    def axe(self, room, sender, instrument, tenor, direction, size) -> None:
        direction, size = Direction(direction), D(size)
        verb = "buy" if direction == Direction.BUY_PROTECTION else "sell"
        a, t, s = self._alias(instrument), self._tenor(tenor), fmt_decimal(size)
        self._emit(
            room,
            sender,
            self.rng.choice(
                [f"axed to {verb} protection {a} {t} {s}mm", f"we're axed to {verb} protection in {a} {t}, {s}mm"]
            ),
        )
        self.axes.append(ExpectedAxe(instrument=instrument, tenor=tenor, direction=direction, size=size, source=sender))
        self.last_action[(room, sender)] = "axe"

    def interest(self, room, sender, client, instrument, tenor, direction) -> None:
        direction = Direction(direction)
        verb = "buy" if direction == Direction.BUY_PROTECTION else "sell"
        a, t = self._alias(instrument), self._tenor(tenor)
        self._emit(
            room,
            sender,
            self.rng.choice(
                [f"{client} looking to {verb} protection {a} {t}", f"have interest from {client} to {verb} protection in {a} {t}"]
            ),
        )
        self.interests.append(ExpectedInterest(client=client, instrument=instrument, tenor=tenor, direction=direction))
        self.last_action[(room, sender)] = "interest"

    def chatter(self, room, sender) -> None:
        self._emit(room, sender, self.rng.choice(CHATTER))
        self.last_action[(room, sender)] = "chatter"

    def ambiguous_unknown(self, room, sender, bid, offer) -> None:
        alias = self.rng.choice(UNKNOWN_ALIASES)
        if self.glossary.lookup(alias) is not None:
            raise ScenarioError(f"{alias} is in the glossary; pick another unknown alias")
        message = self._emit(room, sender, f"{alias} {self._tenor('5Y')} {fmt_decimal(D(bid))}/{fmt_decimal(D(offer))}")
        self.ambiguous_ids.append(message.id)
        self.last_action[(room, sender)] = "ambiguous"

    def ambiguous_no_context(self, sender, bid, offer, size) -> None:
        s = fmt_decimal(D(size))
        message = self._emit(AMBIGUOUS_ROOM, sender, f"{fmt_decimal(D(bid))}/{fmt_decimal(D(offer))} {s}x{s}")
        self.ambiguous_ids.append(message.id)
        self.last_action[(AMBIGUOUS_ROOM, sender)] = "ambiguous"

    # ---- result -------------------------------------------------------------

    def build(self) -> GeneratedCase:
        if not self.messages:
            raise ScenarioError("scenario produced no messages")
        end = self.clock
        quotes = []
        for q in self.quotes.values():
            # Same rule as rules.sweep_stale, applied at the last message time.
            if q.status == QuoteStatus.LIVE and end - q.updated_ts > DEFAULT_MAX_AGE:
                q = q.model_copy(update={"status": QuoteStatus.STALE})
            quotes.append(q)
        expected = ExpectedBoard(
            name=self.name,
            end_ts=end,
            quotes=sorted(quotes, key=lambda q: (q.instrument, q.tenor, q.source)),
            axes=list(self.axes),
            interests=list(self.interests),
            ambiguous_message_ids=list(self.ambiguous_ids),
        )
        return GeneratedCase(messages=list(self.messages), expected=expected)


def run_scenario(glossary: Glossary, data: dict[str, Any]) -> GeneratedCase:
    builder = DayBuilder(glossary, name=data["name"], seed=data.get("seed", 0))
    for i, step in enumerate(data["steps"], start=1):
        step = dict(step)
        op = step.pop("op", None)
        if op not in OPS:
            raise ScenarioError(f"step {i}: unknown op {op!r}")
        try:
            getattr(builder, op)(**step)
        except TypeError as e:
            raise ScenarioError(f"step {i} ({op}): {e}") from e
        except ScenarioError as e:
            raise ScenarioError(f"step {i} ({op}): {e}") from e
    return builder.build()


def load_scenario(glossary: Glossary, path: Path) -> GeneratedCase:
    return run_scenario(glossary, yaml.safe_load(Path(path).read_text(encoding="utf-8")))
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_builder.py -v`
Expected: all PASS.

- [ ] **Step 6: Checkpoint**

Run `uv run pytest -q` and expect all to pass. Tell the user Task 7 is ready. Do not commit.

---

### Task 8: Random trading days, hand-authored scenarios and the golden set

**Files:**
- Create: `src/brokerchat/generator/random_day.py`, `scenarios/basic_flow.yaml`, `scenarios/context_and_ambiguity.yaml`
- Modify: `src/brokerchat/cli.py` (add the `generate` subcommand)
- Create (generated): `evals/golden/*.jsonl`, `evals/golden/*.expected.json`
- Test: `tests/test_random_day.py`, `tests/test_cli.py` (append)

**Interfaces:**
- Consumes: `DayBuilder`, `load_scenario`, `write_case`, `Glossary`.
- Produces: `random_day(glossary, seed: int, n_steps: int = 30, name: str | None = None) -> GeneratedCase` (default name `day{seed:02d}`), plus the CLI command `brokerchat generate --out DIR [--seeds N ...] [--steps 30] [--scenario PATH ...]`.

- [ ] **Step 1: Write the scenario files**

`scenarios/basic_flow.yaml`:

```yaml
name: basic_flow
seed: 101
steps:
  - {op: chatter, room: ldn-cds-1, sender: jpm_cds}
  - {op: quote, room: ldn-cds-1, sender: jpm_cds, instrument: ITRAXX_EUR_MAIN, tenor: 5Y, bid: 58, offer: 58.5, size: 10}
  - {op: quote, room: ldn-cds-1, sender: gs_flow, instrument: ITRAXX_EUR_XOVER, tenor: 5Y, bid: 302, offer: 304, size: 5}
  - {op: one_sided, room: ldn-cds-1, sender: bnp_cds, instrument: CDX_NA_IG, tenor: 5Y, side: bid, level: 52.25}
  - {op: shift, room: ldn-cds-1, sender: jpm_cds, bp: 1}
  - {op: trade, room: ldn-cds-1, sender: gs_flow}
  - {op: quote, room: ldn-cds-2, sender: citi_ix, instrument: CDX_NA_HY, tenor: 5Y, bid: 104.5, offer: 104.625}
  - {op: axe, room: ldn-cds-2, sender: citi_ix, instrument: ITRAXX_EUR_MAIN, tenor: 5Y, direction: buy_protection, size: 25}
  - {op: interest, room: ldn-cds-2, sender: bnp_cds, client: Alder Capital, instrument: VOLKSWAGEN_AG, tenor: 5Y, direction: sell_protection}
  - {op: refer, room: ldn-cds-1, sender: bnp_cds}
```

`scenarios/context_and_ambiguity.yaml`:

```yaml
name: context_and_ambiguity
seed: 202
steps:
  - {op: quote, room: ldn-cds-1, sender: jpm_cds, instrument: VOLKSWAGEN_AG, tenor: 5Y, bid: 120, offer: 125}
  - {op: same_again, room: ldn-cds-1, sender: jpm_cds, tenor: 10Y, bid: 165, offer: 172}
  - {op: unch, room: ldn-cds-1, sender: jpm_cds}
  - {op: ambiguous_unknown, room: ldn-cds-1, sender: gs_flow, bid: 88, offer: 90}
  - {op: ambiguous_no_context, sender: bnp_cds, bid: 61, offer: 62, size: 10}
  - {op: quote, room: ldn-cds-2, sender: gs_flow, instrument: DEUTSCHE_BANK_AG, tenor: 5Y, bid: 95, offer: 100, size: 5}
  - {op: refer, room: ldn-cds-2, sender: gs_flow}
  - {op: quote, room: ldn-cds-2, sender: gs_flow, instrument: DEUTSCHE_BANK_AG, tenor: 5Y, bid: 96, offer: 101, size: 5}
  - {op: chatter, room: ldn-cds-2, sender: citi_ix}
```

- [ ] **Step 2: Write the failing tests**

`tests/test_random_day.py`:

```python
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
```

Append to `tests/test_cli.py`:

```python
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
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_random_day.py tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'brokerchat.generator.random_day'`, and the CLI tests fail with `invalid choice: 'generate'`.

- [ ] **Step 4: Implement `random_day`**

`src/brokerchat/generator/random_day.py`:

```python
"""Procedurally generated trading days for the golden set."""

import random
from decimal import Decimal

from brokerchat.expected import GeneratedCase
from brokerchat.generator.builder import DayBuilder
from brokerchat.glossary import Glossary
from brokerchat.models import Direction, QuoteStatus, Unit

ROOMS = ["ldn-cds-1", "ldn-cds-2"]
SENDERS = ["jpm_cds", "gs_flow", "bnp_cds", "citi_ix"]
CLIENTS = ["Alder Capital", "Birch Partners", "Cedar AM"]
BASE_LEVEL = {
    "ITRAXX_EUR_MAIN": Decimal("58"),
    "ITRAXX_EUR_XOVER": Decimal("302"),
    "CDX_NA_IG": Decimal("52"),
    "CDX_NA_HY": Decimal("104.5"),
    "VOLKSWAGEN_AG": Decimal("120"),
    "DEUTSCHE_BANK_AG": Decimal("95"),
    "TESCO_PLC": Decimal("80"),
}
WIDTH = {"ITRAXX_EUR_MAIN": Decimal("0.5"), "CDX_NA_IG": Decimal("0.5"), "ITRAXX_EUR_XOVER": Decimal("2")}
SINGLE_NAME_WIDTH = Decimal("5")
PRICE_TICK = Decimal("0.125")
TENOR_FACTOR = {"3Y": Decimal("0.6"), "5Y": Decimal("1"), "7Y": Decimal("1.25"), "10Y": Decimal("1.5")}
WEIGHTS = {
    "quote": 30, "one_sided": 8, "refer": 8, "trade": 6, "shift": 8, "unch": 5, "same_again": 6,
    "axe": 6, "interest": 6, "chatter": 8, "ambiguous_unknown": 3, "ambiguous_no_context": 3,
}
NEEDS_LIVE_MARKET = {"refer", "trade", "shift", "unch", "same_again"}


def _market(rng: random.Random, instrument: str, tenor: str, unit: Unit) -> tuple[Decimal, Decimal]:
    if unit == Unit.PRICE:
        bid = BASE_LEVEL[instrument] + rng.randint(-8, 8) * PRICE_TICK
        return bid, bid + PRICE_TICK
    bid = (BASE_LEVEL[instrument] * TENOR_FACTOR[tenor]).quantize(Decimal("1")) + rng.randint(-3, 3)
    return bid, bid + WIDTH.get(instrument, SINGLE_NAME_WIDTH)


def random_day(glossary: Glossary, seed: int, n_steps: int = 30, name: str | None = None) -> GeneratedCase:
    rng = random.Random(seed)
    b = DayBuilder(glossary, name=name or f"day{seed:02d}", seed=seed)
    ops, weights = list(WEIGHTS), list(WEIGHTS.values())

    for _ in range(n_steps):
        op = rng.choices(ops, weights=weights)[0]
        room, sender = rng.choice(ROOMS), rng.choice(SENDERS)

        if op in NEEDS_LIVE_MARKET:
            # Only refer to a market the sender has just made and nothing has happened to since.
            candidates = [
                rs for rs, key in b.last_quote.items()
                if b.last_action.get(rs) == "quote" and b.quotes[key].status == QuoteStatus.LIVE
            ]
            if candidates:
                room, sender = rng.choice(sorted(candidates))
            else:
                op = "quote"
        if op in ("shift", "same_again") and b.quotes[b.last_quote[(room, sender)]].unit != Unit.BP:
            op = "unch"

        instrument = rng.choice(list(BASE_LEVEL))
        unit = glossary.get(instrument).unit
        tenor = "5Y" if unit == Unit.PRICE else rng.choice(list(TENOR_FACTOR))

        match op:
            case "quote":
                bid, offer = _market(rng, instrument, tenor, unit)
                b.quote(room, sender, instrument, tenor, bid, offer, size=rng.choice([None, 5, 10, 25]))
            case "one_sided":
                bid, offer = _market(rng, instrument, tenor, unit)
                side = rng.choice(["bid", "offer"])
                b.one_sided(room, sender, instrument, tenor, side, bid if side == "bid" else offer)
            case "refer":
                b.refer(room, sender)
            case "trade":
                b.trade(room, sender)
            case "shift":
                b.shift(room, sender, rng.choice([-2, -1, 1, 2]))
            case "unch":
                b.unch(room, sender)
            case "same_again":
                current_instrument, current_tenor, _ = b.last_quote[(room, sender)]
                new_tenor = rng.choice([t for t in TENOR_FACTOR if t != current_tenor])
                bid, offer = _market(rng, current_instrument, new_tenor, Unit.BP)
                b.same_again(room, sender, new_tenor, bid, offer)
            case "axe":
                b.axe(room, sender, instrument, tenor, rng.choice(list(Direction)), rng.choice([10, 25, 50]))
            case "interest":
                b.interest(room, sender, rng.choice(CLIENTS), instrument, tenor, rng.choice(list(Direction)))
            case "chatter":
                b.chatter(room, sender)
            case "ambiguous_unknown":
                level = rng.randint(40, 200)
                b.ambiguous_unknown(room, sender, level, level + 2)
            case "ambiguous_no_context":
                level = rng.randint(40, 200)
                b.ambiguous_no_context(sender, level, level + 1, rng.choice([5, 10]))

    return b.build()
```

- [ ] **Step 5: Add `generate` to the CLI**

In `src/brokerchat/cli.py`, add these imports:

```python
from brokerchat.expected import write_case
from brokerchat.generator.builder import load_scenario
from brokerchat.generator.random_day import random_day
```

Add this function above `build_parser`:

```python
def cmd_generate(args: argparse.Namespace) -> int:
    glossary = Glossary.load()
    cases = [random_day(glossary, seed, n_steps=args.steps) for seed in args.seeds]
    cases += [load_scenario(glossary, path) for path in args.scenario]
    if not cases:
        console.print("[red]Nothing to generate: pass --seeds and/or --scenario[/red]")
        return 2
    for case in cases:
        path = write_case(case, args.out)
        console.print(
            f"{path}  {len(case.messages)} messages, {len(case.expected.quotes)} quotes, "
            f"{len(case.expected.ambiguous_message_ids)} ambiguous"
        )
    return 0
```

In `build_parser`, before `return parser`, add:

```python
    generate = sub.add_parser("generate", help="Generate synthetic chat with its expected board")
    generate.add_argument("--out", type=Path, required=True)
    generate.add_argument("--seeds", type=int, nargs="*", default=[], help="Random days to generate, one per seed")
    generate.add_argument("--steps", type=int, default=30, help="Messages per random day")
    generate.add_argument("--scenario", type=Path, action="append", default=[], help="Scenario YAML (repeatable)")
    generate.set_defaults(func=cmd_generate)
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_random_day.py tests/test_cli.py -v`
Expected: all PASS. If a seed in `test_generated_days_are_well_formed` raises `ScenarioError`, fix the guard in `random_day` that let an invalid op through. Don't skip the seed.

- [ ] **Step 7: Generate the golden set**

Run:

```bash
uv run brokerchat generate --out evals/golden --seeds 1 2 3 4 5 6 7 8 9 10 --steps 30 --scenario scenarios/basic_flow.yaml --scenario scenarios/context_and_ambiguity.yaml
```

Expected: 12 lines printed, about 320 messages in total, and ambiguous messages spread across the cases.

- [ ] **Step 8: Checkpoint with a hand-check**

Run `uv run pytest -q` and expect all to pass. Ask the user to open two generated days (for example `evals/golden/day01.jsonl` and its `.expected.json`) and check that the expected board is what they would write by hand. This is the PRD's "hand-checked" step. Do not commit.

---

### Task 9: Scorer, eval runner and `brokerchat eval`

**Files:**
- Create: `src/brokerchat/scoring.py`, `src/brokerchat/evalrun.py`
- Modify: `src/brokerchat/cli.py` (add the `eval` subcommand)
- Test: `tests/test_scoring.py`, `tests/test_evalrun.py`

**Interfaces:**
- Consumes: `ExpectedBoard`, `Board`, `Store`, `Pipeline`, `Usage`, `AgentResult`, `Processor`, `read_case`, `write_run_outputs`, `make_agent`.
- Produces:
  - `scoring.py`:
    - `QUOTE_FIELDS = ("bid", "offer", "unit", "bid_size", "offer_size")`.
    - `CaseScore` (pydantic): `name`, `quote_fields_total`, `quote_fields_correct`, `status_total`, `status_correct`, `missing_quotes: list[str]`, `spurious_quotes: list[str]`, `field_errors: list[str]`, `unsupported_entries`, `flag_tp`, `flag_fn`, `flag_fp`, `missed_flags: list[str]`, `axes_expected`, `axes_matched`, `interests_expected`, `interests_matched`.
    - `score_case(expected, board, message_ids: set[str]) -> CaseScore`.
    - `EvalReport(cases, messages=0, unprocessed=0, usage=Usage())` with `.summary() -> dict[str, float]`.
    - `TARGETS` and `meets_targets(summary) -> dict[str, bool]`.
  - `evalrun.py`:
    - `CaseRun` dataclass (`name`, `store`, `results`, `score`).
    - `run_eval(golden_dir, agent_factory: Callable[[], Processor], glossary, case_names: list[str] | None = None) -> tuple[EvalReport, list[CaseRun]]`.
    - `write_eval_run(report, runs, runs_dir, now=None) -> Path`.
    - `report_table(report) -> rich.table.Table` and `failure_lines(report) -> list[str]`.
  - CLI: `brokerchat eval [--golden evals/golden] [--agent ...] [--effort ...] [--runs-dir evals/runs] [--case NAME ...] [--show-failures]`.

- [ ] **Step 1: Write the failing tests**

`tests/test_scoring.py`:

```python
from decimal import Decimal

import pytest

from brokerchat.expected import ExpectedAxe, ExpectedBoard, ExpectedInterest, ExpectedQuote
from brokerchat.models import Axe, Direction, EventType, Flag, FlagKind, Instrument, InstrumentKind, Interest, QuoteStatus, Unit
from brokerchat.scoring import EvalReport, meets_targets, score_case
from brokerchat.store import Store
from tests.helpers import at, emit_quote, make_quote

MAIN = Instrument(kind=InstrumentKind.INDEX, name="ITRAXX_EUR_MAIN", tenor="5Y")
IDS = {"m001", "m002", "m003"}


def expected(**overrides) -> ExpectedBoard:
    fields = dict(
        name="case",
        end_ts=at(10),
        quotes=[
            ExpectedQuote(instrument="ITRAXX_EUR_MAIN", tenor="5Y", source="jpm_cds", bid=Decimal("58"),
                          offer=Decimal("58.5"), unit=Unit.BP, status=QuoteStatus.LIVE, updated_ts=at(0))
        ],
        axes=[ExpectedAxe(instrument="ITRAXX_EUR_MAIN", tenor="5Y", direction=Direction.BUY_PROTECTION,
                          size=Decimal("25"), source="citi_ix")],
        interests=[ExpectedInterest(client="Alder Capital", instrument="ITRAXX_EUR_MAIN", tenor="5Y",
                                    direction=Direction.SELL_PROTECTION)],
        ambiguous_message_ids=["m003"],
    )
    fields.update(overrides)
    return ExpectedBoard(**fields)


@pytest.fixture
def perfect(glossary) -> Store:
    store = Store(glossary)
    emit_quote(store, make_quote())
    store.emit(EventType.AXE_RECORDED, Axe(instrument=MAIN, direction=Direction.BUY_PROTECTION, size=Decimal("25"),
                                           source="citi_ix", source_message_ids=["m002"]).model_dump(mode="json"), ts=at(1))
    store.emit(EventType.INTEREST_RECORDED, Interest(client="alder capital", instrument=MAIN,
                                                     direction=Direction.SELL_PROTECTION,
                                                     source_message_ids=["m002"]).model_dump(mode="json"), ts=at(1))
    store.emit(EventType.FLAG_RAISED, Flag(flag_id="f0001", kind=FlagKind.UNKNOWN_INSTRUMENT, question="?",
                                           message_ids=["m003"]).model_dump(mode="json"), ts=at(2))
    return store


def test_perfect_board_scores_full_marks(perfect):
    s = score_case(expected(), perfect.board, IDS)
    assert (s.quote_fields_correct, s.quote_fields_total) == (5, 5)
    assert (s.status_correct, s.status_total) == (1, 1)
    assert (s.axes_matched, s.interests_matched) == (1, 1)
    assert (s.flag_tp, s.flag_fn, s.flag_fp) == (1, 0, 0)
    assert s.unsupported_entries == 0
    assert s.spurious_quotes == [] and s.missing_quotes == []


def test_missing_quote_counts_every_field_wrong(glossary):
    s = score_case(expected(), Store(glossary).board, IDS)
    assert (s.quote_fields_correct, s.quote_fields_total) == (0, 5)
    assert s.missing_quotes == ["ITRAXX_EUR_MAIN 5Y jpm_cds"]


def test_wrong_level_and_status_reported(glossary):
    store = Store(glossary)
    emit_quote(store, make_quote(bid="57", status=QuoteStatus.REFERRED))
    s = score_case(expected(), store.board, IDS)
    assert s.quote_fields_correct == 4
    assert s.status_correct == 0
    assert any("bid expected 58 got 57" in e for e in s.field_errors)


def test_spurious_and_unsupported_entries(glossary):
    store = Store(glossary)
    emit_quote(store, make_quote())
    emit_quote(store, make_quote(quote_id="q0002", name="CDX_NA_IG", ids=("m999",)))
    s = score_case(expected(), store.board, IDS)
    assert s.spurious_quotes == ["q0002 CDX_NA_IG 5Y jpm_cds"]
    assert s.unsupported_entries == 1


def test_flag_on_unambiguous_message_is_false_positive(glossary):
    store = Store(glossary)
    store.emit(EventType.FLAG_RAISED, Flag(flag_id="f0001", kind=FlagKind.AMBIGUOUS, question="?",
                                           message_ids=["m001"]).model_dump(mode="json"), ts=at(0))
    s = score_case(expected(), store.board, IDS)
    assert (s.flag_tp, s.flag_fn, s.flag_fp) == (0, 1, 1)
    assert s.missed_flags == ["m003"]


def test_conflict_flags_are_not_scored(glossary):
    store = Store(glossary)
    store.emit(EventType.FLAG_RAISED, Flag(flag_id="conflict:q1:q2", kind=FlagKind.CONFLICT, question="?",
                                           message_ids=["m001"]).model_dump(mode="json"), ts=at(0))
    assert score_case(expected(), store.board, IDS).flag_fp == 0


def test_summary_and_targets(perfect, glossary):
    good = score_case(expected(), perfect.board, IDS)
    empty = score_case(expected(), Store(glossary).board, IDS)
    summary = EvalReport(cases=[good, empty]).summary()
    assert summary["s1_quote_field_accuracy"] == pytest.approx(0.5)
    assert summary["s4_flag_recall"] == pytest.approx(0.5)
    met = meets_targets(summary)
    assert met["s1_quote_field_accuracy"] is False
    assert met["s3_unsupported_entries"] is True


def test_empty_report_counts_as_perfect():
    assert EvalReport(cases=[]).summary()["s1_quote_field_accuracy"] == 1.0
```

`tests/test_evalrun.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_scoring.py tests/test_evalrun.py -v`
Expected: FAIL with `ModuleNotFoundError` for `brokerchat.scoring` and `brokerchat.evalrun`.

- [ ] **Step 3: Implement the scorer**

`src/brokerchat/scoring.py`:

```python
"""Compares the agent's board with the expected board (PRD §3 S1-S4, §7)."""

from pydantic import BaseModel, Field

from brokerchat.expected import ExpectedBoard
from brokerchat.models import FlagKind
from brokerchat.pipeline import Usage
from brokerchat.store import Board

QUOTE_FIELDS = ("bid", "offer", "unit", "bid_size", "offer_size")
SCORED_FLAG_KINDS = {FlagKind.AMBIGUOUS, FlagKind.UNKNOWN_INSTRUMENT}
TARGETS: dict[str, tuple[str, float]] = {
    "s1_quote_field_accuracy": (">=", 0.95),
    "s2_status_accuracy": (">=", 0.95),
    "s3_unsupported_entries": ("<=", 0.0),
    "s4_flag_recall": (">=", 0.90),
}


class CaseScore(BaseModel):
    name: str
    quote_fields_total: int = 0
    quote_fields_correct: int = 0
    status_total: int = 0
    status_correct: int = 0
    missing_quotes: list[str] = []
    spurious_quotes: list[str] = []
    field_errors: list[str] = []
    unsupported_entries: int = 0
    flag_tp: int = 0
    flag_fn: int = 0
    flag_fp: int = 0
    missed_flags: list[str] = []
    axes_expected: int = 0
    axes_matched: int = 0
    interests_expected: int = 0
    interests_matched: int = 0


def score_case(expected: ExpectedBoard, board: Board, message_ids: set[str]) -> CaseScore:
    s = CaseScore(name=expected.name)

    for eq in expected.quotes:
        label = f"{eq.instrument} {eq.tenor} {eq.source}"
        s.quote_fields_total += len(QUOTE_FIELDS)
        s.status_total += 1
        q = board.find_quote(eq.instrument, eq.tenor, eq.source)
        if q is None:
            s.missing_quotes.append(label)
            continue
        for field in QUOTE_FIELDS:
            want, got = getattr(eq, field), getattr(q, field)
            if want == got:
                s.quote_fields_correct += 1
            else:
                s.field_errors.append(f"{label}: {field} expected {want} got {got}")
        if q.status == eq.status:
            s.status_correct += 1
        else:
            s.field_errors.append(f"{label}: status expected {eq.status} got {q.status}")

    expected_keys = {(e.instrument, e.tenor, e.source) for e in expected.quotes}
    for q in board.quotes.values():
        if (q.instrument.name, q.instrument.tenor, q.source) not in expected_keys:
            s.spurious_quotes.append(f"{q.quote_id} {q.instrument.name} {q.instrument.tenor} {q.source}")

    entries = [*board.quotes.values(), *board.axes, *board.interests]
    s.unsupported_entries = sum(
        1 for e in entries if not e.source_message_ids or not set(e.source_message_ids) <= message_ids
    )

    remaining_axes = [(a.instrument.name, a.instrument.tenor, a.direction, a.size, a.source) for a in board.axes]
    s.axes_expected = len(expected.axes)
    for ea in expected.axes:
        key = (ea.instrument, ea.tenor, ea.direction, ea.size, ea.source)
        if key in remaining_axes:
            remaining_axes.remove(key)
            s.axes_matched += 1

    remaining_interests = [
        (i.client.casefold(), i.instrument.name, i.instrument.tenor, i.direction) for i in board.interests
    ]
    s.interests_expected = len(expected.interests)
    for ei in expected.interests:
        key = (ei.client.casefold(), ei.instrument, ei.tenor, ei.direction)
        if key in remaining_interests:
            remaining_interests.remove(key)
            s.interests_matched += 1

    flags = [f for f in board.flags.values() if f.kind in SCORED_FLAG_KINDS]
    flagged = {m for f in flags for m in f.message_ids}
    ambiguous = set(expected.ambiguous_message_ids)
    for message_id in expected.ambiguous_message_ids:
        if message_id in flagged:
            s.flag_tp += 1
        else:
            s.flag_fn += 1
            s.missed_flags.append(message_id)
    s.flag_fp = sum(1 for f in flags if not set(f.message_ids) & ambiguous)
    return s


def _ratio(n: int, d: int) -> float:
    return n / d if d else 1.0


class EvalReport(BaseModel):
    cases: list[CaseScore]
    messages: int = 0
    unprocessed: int = 0
    usage: Usage = Field(default_factory=Usage)

    def summary(self) -> dict[str, float]:
        c = self.cases
        tp, fn, fp = sum(x.flag_tp for x in c), sum(x.flag_fn for x in c), sum(x.flag_fp for x in c)
        return {
            "s1_quote_field_accuracy": _ratio(sum(x.quote_fields_correct for x in c), sum(x.quote_fields_total for x in c)),
            "s2_status_accuracy": _ratio(sum(x.status_correct for x in c), sum(x.status_total for x in c)),
            "s3_unsupported_entries": float(sum(x.unsupported_entries for x in c)),
            "s4_flag_recall": _ratio(tp, tp + fn),
            # Approximate: true positives count messages, false positives count flags.
            "flag_precision": _ratio(tp, tp + fp),
            "spurious_quotes": float(sum(len(x.spurious_quotes) for x in c)),
            "axes_recall": _ratio(sum(x.axes_matched for x in c), sum(x.axes_expected for x in c)),
            "interests_recall": _ratio(sum(x.interests_matched for x in c), sum(x.interests_expected for x in c)),
        }


def meets_targets(summary: dict[str, float]) -> dict[str, bool]:
    return {
        name: (summary[name] >= target if op == ">=" else summary[name] <= target)
        for name, (op, target) in TARGETS.items()
    }
```

- [ ] **Step 4: Implement the eval runner**

`src/brokerchat/evalrun.py`:

```python
"""Runs an agent over the golden set and records a scored, comparable run."""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

from rich.table import Table

from brokerchat.expected import read_case
from brokerchat.glossary import Glossary
from brokerchat.pipeline import AgentResult, Pipeline, Processor, Usage, write_run_outputs
from brokerchat.scoring import TARGETS, CaseScore, EvalReport, meets_targets, score_case
from brokerchat.store import Store


@dataclass
class CaseRun:
    name: str
    store: Store
    results: list[AgentResult]
    score: CaseScore


def run_eval(
    golden_dir: Path,
    agent_factory: Callable[[], Processor],
    glossary: Glossary,
    case_names: list[str] | None = None,
) -> tuple[EvalReport, list[CaseRun]]:
    paths = sorted(Path(golden_dir).glob("*.jsonl"))
    if case_names:
        missing = set(case_names) - {p.stem for p in paths}
        if missing:
            raise ValueError(f"cases not found in {golden_dir}: {', '.join(sorted(missing))}")
        paths = [p for p in paths if p.stem in case_names]
    if not paths:
        raise ValueError(f"no cases found in {golden_dir}")

    runs = []
    for path in paths:
        case = read_case(path)
        store = Store(glossary)
        pipeline = Pipeline(store, agent_factory())
        pipeline.run(case.messages)
        runs.append(CaseRun(case.expected.name, store, pipeline.results,
                            score_case(case.expected, store.board, set(store.messages))))

    usage = Usage()
    for run in runs:
        for result in run.results:
            usage.add(result.usage)
    report = EvalReport(
        cases=[r.score for r in runs],
        messages=sum(len(r.results) for r in runs),
        unprocessed=sum(1 for r in runs for res in r.results if res.status != "ok"),
        usage=usage,
    )
    return report, runs


def write_eval_run(report: EvalReport, runs: list[CaseRun], runs_dir: Path, now: datetime | None = None) -> Path:
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(runs_dir) / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    summary = report.summary()
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "summary": summary,
                "targets_met": meets_targets(summary),
                "messages": report.messages,
                "unprocessed": report.unprocessed,
                "usage": report.usage.model_dump(),
                "cost_usd": report.usage.cost_usd(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    for run in runs:
        write_run_outputs(run.store, run.results, run_dir / run.name)
        (run_dir / run.name / "score.json").write_text(run.score.model_dump_json(indent=2), encoding="utf-8")
    return run_dir


def report_table(report: EvalReport) -> Table:
    summary = report.summary()
    met = meets_targets(summary)
    table = Table(title=f"Eval: {len(report.cases)} cases, {report.messages} messages, "
                        f"{report.unprocessed} unprocessed, approx ${report.usage.cost_usd():.2f}")
    for column in ("Metric", "Value", "Target", "Met"):
        table.add_column(column)
    for name, value in summary.items():
        if name in TARGETS:
            op, target = TARGETS[name]
            table.add_row(name, f"{value:.3f}", f"{op} {target}", "yes" if met[name] else "[red]no[/red]")
        else:
            table.add_row(name, f"{value:.3f}", "", "")
    return table


def failure_lines(report: EvalReport) -> list[str]:
    lines = []
    for c in report.cases:
        lines += [f"{c.name}: missing quote {m}" for m in c.missing_quotes]
        lines += [f"{c.name}: spurious quote {m}" for m in c.spurious_quotes]
        lines += [f"{c.name}: {e}" for e in c.field_errors]
        lines += [f"{c.name}: no flag for ambiguous message {m}" for m in c.missed_flags]
    return lines
```

- [ ] **Step 5: Add `eval` to the CLI**

In `src/brokerchat/cli.py`, add this import:

```python
from brokerchat.evalrun import failure_lines, report_table, run_eval, write_eval_run
```

Add this function above `build_parser`:

```python
def cmd_eval(args: argparse.Namespace) -> int:
    glossary = Glossary.load()
    report, runs = run_eval(
        args.golden, lambda: make_agent(args.agent, glossary, args.effort), glossary, case_names=args.case or None
    )
    run_dir = write_eval_run(report, runs, args.runs_dir)
    console.print(report_table(report))
    if args.show_failures:
        for line in failure_lines(report):
            console.print(f"  {line}")
    console.print(f"Run written to {run_dir}")
    return 0
```

In `build_parser`, before `return parser`, add:

```python
    evaluate = sub.add_parser("eval", help="Score the agent against the golden set")
    evaluate.add_argument("--golden", type=Path, default=Path("evals/golden"))
    evaluate.add_argument("--agent", choices=AGENT_CHOICES, default=AGENT_CHOICES[0])
    evaluate.add_argument("--effort", choices=EFFORT_CHOICES, default="medium")
    evaluate.add_argument("--runs-dir", type=Path, default=Path("evals/runs"))
    evaluate.add_argument("--case", action="append", default=[], help="Only run this case (repeatable)")
    evaluate.add_argument("--show-failures", action="store_true")
    evaluate.set_defaults(func=cmd_eval)
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_scoring.py tests/test_evalrun.py -v`
Expected: all PASS.

- [ ] **Step 7: Run the scorer end to end with the dummy agent (PRD M2)**

Run: `uv run brokerchat eval --agent null`
Expected: a table where S1, S2 and S4 are low (the null agent records nothing), S3 is 0, and the line `Run written to evals/runs/<timestamp>`.

- [ ] **Step 8: Checkpoint**

Run `uv run pytest -q` and expect all to pass. Tell the user Task 9 is ready. This completes milestone M2. Do not commit.

---

### Task 10: Agent tools and the tool executor

**Files:**
- Create: `src/brokerchat/agent/__init__.py`, `src/brokerchat/agent/tools.py`
- Test: `tests/agent/__init__.py`, `tests/agent/test_tools.py`

**Interfaces:**
- Consumes: `Store`, `Glossary`, models, `EventType`.
- Produces:
  - `TOOL_SCHEMAS: list[dict]` covering seven tools: `lookup_instrument`, `get_quotes`, `upsert_quote`, `set_quote_status`, `record_axe`, `record_interest` and `raise_flag`. All use `strict: True`, every property is required, and optional values are nullable types.
  - `ToolError(Exception)`.
  - `ToolExecutor(store, current: ChatMessage)` with `.calls: int`, `.execute(name: str, tool_input: dict, tool_use_id: str) -> dict` (a `tool_result` block, with `is_error: True` on failure) and `.raise_cap_flag() -> str`.
  - `upsert_quote` returns `{"quote_id", "created": bool}`. `set_quote_status` returns `{"quote_id", "status"}`. `raise_flag` returns `{"flag_id"}`.

- [ ] **Step 1: Write the failing tests**

`tests/agent/__init__.py`: empty file.

`tests/agent/test_tools.py`:

```python
import json
from decimal import Decimal

import pytest

from brokerchat.agent.tools import TOOL_SCHEMAS, ToolExecutor
from brokerchat.models import Direction, FlagKind, QuoteStatus, Unit
from brokerchat.store import Store
from tests.helpers import msg


@pytest.fixture
def store(glossary):
    s = Store(glossary)
    for m in [msg("m001", "Main 5y 58/58.5 10x10", 0), msg("m002", "refer", 1), msg("m003", "Main 5y 59/59.5", 2)]:
        s.add_message(m)
    return s


def run(store, name, tool_input, current="m001"):
    return ToolExecutor(store, store.messages[current]).execute(name, tool_input, "tu_1")


def ok(result):
    assert result["tool_use_id"] == "tu_1"
    assert "is_error" not in result, result["content"]
    return json.loads(result["content"])


def upsert(**overrides):
    fields = dict(quote_id=None, instrument="ITRAXX_EUR_MAIN", tenor="5Y", bid=58, offer=58.5, bid_size=10,
                  offer_size=10, source="jpm_cds", source_message_ids=["m001"], confidence="high")
    fields.update(overrides)
    return fields


def test_every_schema_is_strict_with_all_properties_required():
    for tool in TOOL_SCHEMAS:
        schema = tool["input_schema"]
        assert tool["strict"] is True
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"]), tool["name"]


def test_every_schema_has_a_handler():
    for tool in TOOL_SCHEMAS:
        assert hasattr(ToolExecutor, f"_tool_{tool['name']}"), tool["name"]


def test_lookup_instrument(store):
    assert ok(run(store, "lookup_instrument", {"text": "itrx main"})) == {
        "found": True, "canonical": "ITRAXX_EUR_MAIN", "kind": "index", "unit": "bp"
    }
    assert ok(run(store, "lookup_instrument", {"text": "ZXQ"}))["found"] is False


def test_upsert_creates_quote(store):
    assert ok(run(store, "upsert_quote", upsert())) == {"quote_id": "q0001", "created": True}
    q = store.board.quotes["q0001"]
    assert (q.bid, q.offer, q.bid_size, q.unit) == (Decimal("58"), Decimal("58.5"), Decimal("10"), Unit.BP)
    assert q.updated_ts == store.messages["m001"].ts
    assert store.events[-1].caused_by_message_id == "m001"


def test_requote_after_refer_updates_same_quote_and_goes_live(store):
    ok(run(store, "upsert_quote", upsert()))
    ok(run(store, "set_quote_status", {"quote_id": "q0001", "status": "referred", "message_id": "m002"}, "m002"))
    result = ok(run(store, "upsert_quote", upsert(bid=59, offer=59.5, source_message_ids=["m003"]), "m003"))
    assert result == {"quote_id": "q0001", "created": False}
    assert len(store.board.quotes) == 1
    q = store.board.quotes["q0001"]
    assert q.status == QuoteStatus.LIVE
    assert q.source_message_ids == ["m001", "m002", "m003"]


def test_price_level_from_float_is_exact(store):
    ok(run(store, "upsert_quote", upsert(instrument="CDX_NA_HY", bid=104.5, offer=104.625, bid_size=None,
                                         offer_size=None, source="citi_ix")))
    q = store.board.quotes["q0001"]
    assert q.offer == Decimal("104.625")
    assert q.unit == Unit.PRICE


def test_tenor_is_normalised(store):
    ok(run(store, "upsert_quote", upsert(tenor="5y")))
    assert store.board.quotes["q0001"].instrument.tenor == "5Y"


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({"instrument": "Main"}, "lookup_instrument"),
        ({"tenor": "6Y"}, "invalid tenor"),
        ({"source_message_ids": []}, "at least one"),
        ({"source_message_ids": ["m999"]}, "unknown message ids"),
        ({"bid": None, "offer": None}, "at least a bid or an offer"),
        ({"quote_id": "q0042"}, "no quote with id"),
        ({"surprise": 1}, "Extra inputs"),
    ],
)
def test_upsert_rejects_bad_input_without_touching_board(store, overrides, error):
    result = run(store, "upsert_quote", upsert(**overrides))
    assert result["is_error"] is True
    assert error in result["content"]
    assert store.events == []


def test_upsert_with_quote_id_for_other_instrument_rejected(store):
    ok(run(store, "upsert_quote", upsert()))
    result = run(store, "upsert_quote", upsert(quote_id="q0001", instrument="CDX_NA_IG"))
    assert result["is_error"] is True


def test_low_confidence_raises_flag(store):
    ok(run(store, "upsert_quote", upsert(confidence="low")))
    flag = next(iter(store.board.flags.values()))
    assert flag.kind == FlagKind.AMBIGUOUS
    assert flag.message_ids == ["m001"]


def test_set_status_traded(store):
    ok(run(store, "upsert_quote", upsert()))
    assert ok(run(store, "set_quote_status", {"quote_id": "q0001", "status": "traded", "message_id": "m002"}, "m002")) == {
        "quote_id": "q0001", "status": "traded"
    }
    assert store.board.quotes["q0001"].status == QuoteStatus.TRADED


def test_set_status_rejects_unknown_quote_and_stale(store):
    assert run(store, "set_quote_status", {"quote_id": "q0009", "status": "traded", "message_id": "m001"})["is_error"]
    ok(run(store, "upsert_quote", upsert()))
    assert run(store, "set_quote_status", {"quote_id": "q0001", "status": "stale", "message_id": "m001"})["is_error"]


def test_record_axe_and_interest(store):
    ok(run(store, "record_axe", {"instrument": "ITRAXX_EUR_XOVER", "tenor": "5Y", "direction": "buy_protection",
                                 "size": 25, "source": "bnp_cds", "source_message_ids": ["m001"]}))
    ok(run(store, "record_interest", {"client": "Alder Capital", "instrument": "VOLKSWAGEN_AG", "tenor": "5Y",
                                      "direction": "sell_protection", "note": None, "source_message_ids": ["m001"]}))
    assert store.board.axes[0].size == Decimal("25")
    assert store.board.interests[0].direction == Direction.SELL_PROTECTION


def test_get_quotes_filters(store):
    ok(run(store, "upsert_quote", upsert()))
    ok(run(store, "upsert_quote", upsert(instrument="CDX_NA_IG", source="bnp_cds")))
    assert len(ok(run(store, "get_quotes", {"instrument": None, "source": None}))) == 2
    rows = ok(run(store, "get_quotes", {"instrument": "main", "source": None}))
    assert [r["quote_id"] for r in rows] == ["q0001"]
    assert [r["quote_id"] for r in ok(run(store, "get_quotes", {"instrument": None, "source": "bnp_cds"}))] == ["q0002"]


def test_raise_flag(store):
    assert ok(run(store, "raise_flag", {"kind": "unknown_instrument", "question": "What is ZXQ?",
                                        "message_ids": ["m001"]})) == {"flag_id": "f0001"}
    assert run(store, "raise_flag", {"kind": "conflict", "question": "?", "message_ids": ["m001"]})["is_error"]


def test_unknown_tool(store):
    assert run(store, "delete_everything", {})["is_error"] is True


def test_cap_flag(store):
    executor = ToolExecutor(store, store.messages["m003"])
    flag_id = executor.raise_cap_flag()
    assert store.board.flags[flag_id].message_ids == ["m003"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/agent/test_tools.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'brokerchat.agent'`.

- [ ] **Step 3: Implement the tools**

`src/brokerchat/agent/__init__.py`: empty file.

`src/brokerchat/agent/tools.py`:

```python
"""Tool definitions for the agent and the executor that validates calls and turns them into events."""

import json
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from brokerchat.models import (
    Axe, ChatMessage, Confidence, Direction, EventType, Flag, FlagKind, Instrument, Interest, Quote, QuoteStatus, Unit,
)
from brokerchat.store import Store


def _nullable(type_: str, description: str) -> dict[str, Any]:
    return {"type": [type_, "null"], "description": description}


def _tool(name: str, description: str, properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        },
    }


_INSTRUMENT = {"type": "string", "description": "Canonical instrument name from lookup_instrument, e.g. ITRAXX_EUR_MAIN"}
_TENOR = {"type": "string", "description": "Tenor such as 5Y or 10Y"}
_DIRECTION = {"type": "string", "enum": [d.value for d in Direction]}
_MESSAGE_IDS = {"type": "array", "items": {"type": "string"}, "description": "Ids of the chat messages this is based on"}

TOOL_SCHEMAS: list[dict[str, Any]] = [
    _tool(
        "lookup_instrument",
        "Resolve an instrument alias as written in chat (e.g. 'Main', 'XO', 'vw') to its canonical name, kind and "
        "quoting unit. Call this before recording anything about an instrument you have not resolved yet.",
        {"text": {"type": "string", "description": "The alias exactly as written in chat"}},
    ),
    _tool(
        "get_quotes",
        "List quotes on the board, optionally filtered by instrument (canonical name or alias) and/or source. "
        "Use this to find the market that a message like 'off that' or 'unch' refers to.",
        {
            "instrument": _nullable("string", "Filter by instrument, or null for all"),
            "source": _nullable("string", "Filter by source (chat sender), or null for all"),
        },
    ),
    _tool(
        "upsert_quote",
        "Create or update the quote for (instrument, tenor, source). There is at most one quote per instrument, "
        "tenor and source: a new market from the same source replaces the old one and makes it live. Pass quote_id "
        "to update a specific quote, or null to match on instrument, tenor and source. bid and offer are levels in "
        "the instrument's unit (bp, or price for price-quoted indices); null means that side is not quoted. Sizes "
        "are in millions; null if not stated.",
        {
            "quote_id": _nullable("string", "Existing quote id, or null"),
            "instrument": _INSTRUMENT,
            "tenor": _TENOR,
            "bid": _nullable("number", "Bid level, or null"),
            "offer": _nullable("number", "Offer level, or null"),
            "bid_size": _nullable("number", "Bid size in millions, or null"),
            "offer_size": _nullable("number", "Offer size in millions, or null"),
            "source": {"type": "string", "description": "The chat sender who made the market"},
            "source_message_ids": _MESSAGE_IDS,
            "confidence": {
                "type": "string",
                "enum": ["high", "low"],
                "description": "low if you had to guess any field; a flag is raised for a human to check",
            },
        },
    ),
    _tool(
        "set_quote_status",
        "Change a quote's status: referred (pulled), traded, or live (confirmed still there).",
        {
            "quote_id": {"type": "string"},
            "status": {"type": "string", "enum": ["live", "referred", "traded"]},
            "message_id": {"type": "string", "description": "Id of the chat message that changed the status"},
        },
    ),
    _tool(
        "record_axe",
        "Record an axe: a participant's stated desire to buy or sell protection.",
        {
            "instrument": _INSTRUMENT,
            "tenor": _TENOR,
            "direction": _DIRECTION,
            "size": _nullable("number", "Size in millions, or null"),
            "source": {"type": "string", "description": "The chat sender who is axed"},
            "source_message_ids": _MESSAGE_IDS,
        },
    ),
    _tool(
        "record_interest",
        "Record a client's interest in buying or selling protection.",
        {
            "client": {"type": "string", "description": "Client name exactly as written"},
            "instrument": _INSTRUMENT,
            "tenor": _TENOR,
            "direction": _DIRECTION,
            "note": _nullable("string", "Any extra detail, or null"),
            "source_message_ids": _MESSAGE_IDS,
        },
    ),
    _tool(
        "raise_flag",
        "Ask a human instead of guessing. Use unknown_instrument for an alias lookup_instrument does not know, and "
        "ambiguous for a market or reference you cannot resolve with confidence.",
        {
            "kind": {"type": "string", "enum": ["ambiguous", "unknown_instrument"]},
            "question": {"type": "string", "description": "A specific question a broker can answer"},
            "message_ids": _MESSAGE_IDS,
        },
    ),
]


class ToolError(Exception):
    pass


class _Input(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LookupInput(_Input):
    text: str


class GetQuotesInput(_Input):
    instrument: str | None = None
    source: str | None = None


class UpsertQuoteInput(_Input):
    quote_id: str | None = None
    instrument: str
    tenor: str
    bid: float | None = None
    offer: float | None = None
    bid_size: float | None = None
    offer_size: float | None = None
    source: str
    source_message_ids: list[str]
    confidence: Confidence = Confidence.HIGH


class SetStatusInput(_Input):
    quote_id: str
    status: Literal["live", "referred", "traded"]
    message_id: str


class RecordAxeInput(_Input):
    instrument: str
    tenor: str
    direction: Direction
    size: float | None = None
    source: str
    source_message_ids: list[str]


class RecordInterestInput(_Input):
    client: str
    instrument: str
    tenor: str
    direction: Direction
    note: str | None = None
    source_message_ids: list[str]


class RaiseFlagInput(_Input):
    kind: Literal["ambiguous", "unknown_instrument"]
    question: str
    message_ids: list[str]


def _dec(x: float | None) -> Decimal | None:
    # str() first so 104.625 becomes Decimal("104.625"), not the binary float's expansion.
    return None if x is None else Decimal(str(x))


class ToolExecutor:
    """Executes one message's tool calls against the store. Every successful write is an event."""

    def __init__(self, store: Store, current: ChatMessage) -> None:
        self.store = store
        self.glossary = store.glossary
        self.current = current
        self.calls = 0

    def execute(self, name: str, tool_input: dict[str, Any], tool_use_id: str) -> dict[str, Any]:
        self.calls += 1
        handler = getattr(self, f"_tool_{name}", None)
        try:
            if handler is None:
                raise ToolError(f"unknown tool {name!r}")
            result = handler(tool_input)
        except (ToolError, ValidationError) as e:
            return {"type": "tool_result", "tool_use_id": tool_use_id, "content": f"Error: {e}", "is_error": True}
        return {"type": "tool_result", "tool_use_id": tool_use_id, "content": json.dumps(result, default=str)}

    def raise_cap_flag(self) -> str:
        return self._flag(
            FlagKind.AMBIGUOUS, "The agent hit its tool-call limit on this message; please review it.", [self.current.id]
        )

    # ---- helpers -------------------------------------------------------------

    def _instrument(self, name: str, tenor: str) -> tuple[Instrument, Unit]:
        info = self.glossary.get(name)
        if info is None:
            raise ToolError(
                f"unknown instrument {name!r}; call lookup_instrument with the alias from chat, "
                "or raise_flag(kind='unknown_instrument')"
            )
        normalized = self.glossary.normalize_tenor(tenor)
        if normalized is None:
            raise ToolError(f"invalid tenor {tenor!r}; expected one of {', '.join(self.glossary.tenors)}")
        return Instrument(kind=info.kind, name=info.canonical, tenor=normalized), info.unit

    def _check_ids(self, ids: list[str]) -> None:
        if not ids:
            raise ToolError("cite at least one chat message id")
        unknown = [i for i in ids if i not in self.store.messages]
        if unknown:
            raise ToolError(f"unknown message ids {unknown}; cite only ids shown in the chat context")

    def _flag(self, kind: FlagKind, question: str, message_ids: list[str]) -> str:
        flag = Flag(flag_id=self.store.next_flag_id(), kind=kind, question=question, message_ids=message_ids)
        self.store.emit(EventType.FLAG_RAISED, flag.model_dump(mode="json"), ts=self.current.ts, caused_by=self.current.id)
        return flag.flag_id

    # ---- tools ---------------------------------------------------------------

    def _tool_lookup_instrument(self, raw: dict[str, Any]) -> dict[str, Any]:
        inp = LookupInput.model_validate(raw)
        info = self.glossary.lookup(inp.text)
        if info is None:
            return {
                "found": False,
                "text": inp.text,
                "hint": "Not a known alias. If it is meant to be an instrument, raise_flag(kind='unknown_instrument').",
            }
        return {"found": True, "canonical": info.canonical, "kind": info.kind.value, "unit": info.unit.value}

    def _tool_get_quotes(self, raw: dict[str, Any]) -> list[dict[str, Any]]:
        inp = GetQuotesInput.model_validate(raw)
        name = None
        if inp.instrument is not None:
            info = self.glossary.get(inp.instrument) or self.glossary.lookup(inp.instrument)
            if info is None:
                raise ToolError(f"unknown instrument {inp.instrument!r}")
            name = info.canonical
        return [
            q.model_dump(mode="json")
            for q in self.store.board.quotes.values()
            if (name is None or q.instrument.name == name) and (inp.source is None or q.source == inp.source)
        ]

    def _tool_upsert_quote(self, raw: dict[str, Any]) -> dict[str, Any]:
        inp = UpsertQuoteInput.model_validate(raw)
        instrument, unit = self._instrument(inp.instrument, inp.tenor)
        self._check_ids(inp.source_message_ids)
        if inp.bid is None and inp.offer is None:
            raise ToolError("a quote needs at least a bid or an offer")

        if inp.quote_id is not None:
            existing = self.store.board.quotes.get(inp.quote_id)
            if existing is None:
                raise ToolError(f"no quote with id {inp.quote_id}")
            if (existing.instrument, existing.source) != (instrument, inp.source):
                raise ToolError(
                    f"quote {inp.quote_id} is {existing.instrument.name} {existing.instrument.tenor} from "
                    f"{existing.source}; pass quote_id null for a different instrument, tenor or source"
                )
        else:
            existing = self.store.board.find_quote(instrument.name, instrument.tenor, inp.source)

        prior_ids = existing.source_message_ids if existing else []
        quote = Quote(
            quote_id=existing.quote_id if existing else self.store.next_quote_id(),
            instrument=instrument,
            bid=_dec(inp.bid),
            offer=_dec(inp.offer),
            unit=unit,
            bid_size=_dec(inp.bid_size),
            offer_size=_dec(inp.offer_size),
            source=inp.source,
            status=QuoteStatus.LIVE,
            updated_ts=self.current.ts,
            source_message_ids=list(dict.fromkeys([*prior_ids, *inp.source_message_ids])),
            confidence=inp.confidence,
        )
        self.store.emit(EventType.QUOTE_UPSERTED, quote.model_dump(mode="json"), ts=self.current.ts, caused_by=self.current.id)
        if inp.confidence == Confidence.LOW:
            self._flag(
                FlagKind.AMBIGUOUS,
                f"Low-confidence read of {instrument.name} {instrument.tenor} from {inp.source}; please check.",
                inp.source_message_ids,
            )
        return {"quote_id": quote.quote_id, "created": existing is None}

    def _tool_set_quote_status(self, raw: dict[str, Any]) -> dict[str, Any]:
        inp = SetStatusInput.model_validate(raw)
        if inp.quote_id not in self.store.board.quotes:
            raise ToolError(f"no quote with id {inp.quote_id}")
        self._check_ids([inp.message_id])
        self.store.emit(
            EventType.QUOTE_STATUS_CHANGED,
            {"quote_id": inp.quote_id, "status": inp.status, "message_id": inp.message_id},
            ts=self.current.ts,
            caused_by=self.current.id,
        )
        return {"quote_id": inp.quote_id, "status": inp.status}

    def _tool_record_axe(self, raw: dict[str, Any]) -> dict[str, Any]:
        inp = RecordAxeInput.model_validate(raw)
        instrument, _ = self._instrument(inp.instrument, inp.tenor)
        self._check_ids(inp.source_message_ids)
        axe = Axe(instrument=instrument, direction=inp.direction, size=_dec(inp.size), source=inp.source,
                  source_message_ids=inp.source_message_ids)
        self.store.emit(EventType.AXE_RECORDED, axe.model_dump(mode="json"), ts=self.current.ts, caused_by=self.current.id)
        return {"recorded": True}

    def _tool_record_interest(self, raw: dict[str, Any]) -> dict[str, Any]:
        inp = RecordInterestInput.model_validate(raw)
        instrument, _ = self._instrument(inp.instrument, inp.tenor)
        self._check_ids(inp.source_message_ids)
        interest = Interest(client=inp.client, instrument=instrument, direction=inp.direction, note=inp.note,
                            source_message_ids=inp.source_message_ids)
        self.store.emit(EventType.INTEREST_RECORDED, interest.model_dump(mode="json"), ts=self.current.ts,
                        caused_by=self.current.id)
        return {"recorded": True}

    def _tool_raise_flag(self, raw: dict[str, Any]) -> dict[str, Any]:
        inp = RaiseFlagInput.model_validate(raw)
        self._check_ids(inp.message_ids)
        return {"flag_id": self._flag(FlagKind(inp.kind), inp.question, inp.message_ids)}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/agent/test_tools.py -v`
Expected: all PASS.

- [ ] **Step 5: Checkpoint**

Run `uv run pytest -q` and expect all to pass. Tell the user Task 10 is ready. Do not commit.

---

### Task 11: System prompt and per-message context

**Files:**
- Create: `src/brokerchat/agent/prompt.py`
- Test: `tests/agent/test_prompt.py`

**Interfaces:**
- Consumes: `Glossary`, `Store`, `ChatMessage`, `fmt_decimal`.
- Produces: `build_system_prompt(glossary) -> str` (deterministic, so it can be prompt-cached) and `build_context(store, message, recent: int = 20) -> str`. The new message must already be in `store.messages`.

- [ ] **Step 1: Write the failing tests**

`tests/agent/test_prompt.py`:

```python
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
```

There are 25 earlier `a` messages and the window is 20, so the recent block holds `a06` to `a25`. The new message `a26` appears only in its own block.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/agent/test_prompt.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'brokerchat.agent.prompt'`.

- [ ] **Step 3: Implement the prompt module**

`src/brokerchat/agent/prompt.py`:

```python
"""System prompt (stable, cached) and the per-message context the agent sees."""

import json
from decimal import Decimal

from brokerchat.glossary import Glossary
from brokerchat.models import ChatMessage, Quote, fmt_decimal
from brokerchat.store import Store

SYSTEM_TEMPLATE = """\
You maintain a live quote board for a credit default swap (CDS) broking desk. You read broker chat one message at a \
time and record what it means using your tools. The board is the desk's picture of the market, so accuracy matters \
more than coverage: a wrong entry on the board is worse than a missing one.

How to process a message:
- Read the new message in the context of the recent messages from the same room and the current board.
- Resolve every instrument alias with lookup_instrument before recording anything about it.
- Record markets with upsert_quote, status changes with set_quote_status, axes with record_axe and client interest \
with record_interest.
- The source of a market or axe is the sender of the message that made it.
- Cite the id of every chat message you relied on. Never record anything the chat does not say.
- Many messages carry no market information (greetings, questions, small talk). For those, call no tools.

Chat conventions (provisional, from the desk glossary):
- "58/58.5" is bid/offer. "10x10" is bid size x offer size in millions.
- "58 bid", "58 bid for Main 5y", "offered at 58.5" or "58.5 offer" is a one-sided market: the other side is null.
- Levels are in basis points, except instruments whose unit is price (such as CDX HY), which are quoted in price.
- "refer", "refer that", "off that" and "pulled" mean the sender's most recent market in this room is no longer \
live: set its status to referred.
- "trades 58.5", "done at 58.5" and "dealt 58.5" mean the sender's most recent market in this room traded: set its \
status to traded and leave its levels as quoted.
- "unch", "still there" and "unchanged" mean the sender's most recent market in this room is still live as quoted: \
set its status to live.
- "1 wider", "move 2bp tighter" or "wider by 1" move every quoted side of the sender's most recent market in this \
room up (wider) or down (tighter) by that many basis points: upsert the new levels and keep the sizes.
- "same in 10y 72/73", "and 10y 72/73" and "10s 72/73" quote the same instrument as the sender's most recent market \
in this room, in a new tenor, with no size.
- "axed to buy protection" is an axe with direction buy_protection; "axed to sell protection" is sell_protection.
- "<client> looking to buy protection ..." or "interest from <client> to sell protection ..." is client interest.
- There is at most one quote per instrument, tenor and source. A new market from the same source on the same \
instrument and tenor updates that quote, even if it was referred, traded or stale.

When you are unsure, ask instead of guessing:
- An alias that lookup_instrument does not know: raise_flag(kind="unknown_instrument") and record nothing for it.
- A market with no identifiable instrument, or a reference ("off that", "same in 10y") with no clear target: \
raise_flag(kind="ambiguous") with a specific question, and record nothing for it.
- If you record something but had to guess a field, set confidence to "low".

Known instruments:
{instruments}
Valid tenors: {tenors}
"""


def build_system_prompt(glossary: Glossary) -> str:
    instruments = "\n".join(
        f"- {i.canonical} ({i.kind.value}, quoted in {i.unit.value}): aliases {', '.join(i.aliases)}"
        for i in glossary.instruments.values()
    )
    return SYSTEM_TEMPLATE.format(instruments=instruments, tenors=", ".join(glossary.tenors))


def _num(x: Decimal | None) -> str | None:
    return None if x is None else fmt_decimal(x)


def _quote_row(q: Quote) -> str:
    return json.dumps(
        {
            "quote_id": q.quote_id,
            "instrument": q.instrument.name,
            "tenor": q.instrument.tenor,
            "bid": _num(q.bid),
            "offer": _num(q.offer),
            "bid_size": _num(q.bid_size),
            "offer_size": _num(q.offer_size),
            "unit": q.unit.value,
            "source": q.source,
            "status": q.status.value,
            "updated": q.updated_ts.strftime("%H:%M:%S"),
        }
    )


def build_context(store: Store, message: ChatMessage, recent: int = 20) -> str:
    board = "\n".join(_quote_row(q) for q in store.board.quotes.values()) or "(empty)"
    room_messages = [m for m in store.messages.values() if m.room == message.room and m.id != message.id][-recent:]
    history = "\n".join(f"[{m.id} {m.ts.strftime('%H:%M:%S')}] {m.sender}: {m.text}" for m in room_messages) or "(none)"
    return (
        f"<board>\n{board}\n</board>\n"
        f'<recent_messages room="{message.room}">\n{history}\n</recent_messages>\n'
        f'<new_message id="{message.id}" room="{message.room}" sender="{message.sender}" '
        f'time="{message.ts.strftime("%H:%M:%S")}">\n{message.text}\n</new_message>\n'
        "Process the new message."
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/agent/test_prompt.py -v`
Expected: all PASS.

- [ ] **Step 5: Checkpoint**

Run `uv run pytest -q` and expect all to pass. Tell the user Task 11 is ready. Do not commit.

---

### Task 12: The Claude agent loop and CLI wiring

**Files:**
- Create: `src/brokerchat/agent/loop.py`, `tests/agent/fakes.py`
- Modify: `src/brokerchat/cli.py` (`AGENT_CHOICES`, `make_agent`)
- Test: `tests/agent/test_loop.py`

**Interfaces:**
- Consumes: `TOOL_SCHEMAS`, `ToolExecutor`, `build_system_prompt`, `build_context`, `AgentResult`, `Usage`, `Store`.
- Produces: `ClaudeAgent(client, glossary, *, model=MODEL, effort=DEFAULT_EFFORT, max_iterations=8, context_messages=20)` with `.process(message, store) -> AgentResult`. Constants `MODEL = "claude-opus-5-5"`, `MAX_TOKENS = 16000`, `DEFAULT_EFFORT = "medium"` and `FALLBACK_BETA = "server-side-fallback-2026-07-01"`. The client is anything exposing `client.beta.messages.create(**kwargs)`.

- [ ] **Step 1: Write the fake client**

`tests/agent/fakes.py`:

```python
from types import SimpleNamespace as NS

import anthropic


def usage(input_tokens=100, output_tokens=20):
    return NS(input_tokens=input_tokens, output_tokens=output_tokens, cache_read_input_tokens=0,
              cache_creation_input_tokens=None)


def response(stop_reason, *content):
    return NS(stop_reason=stop_reason, content=list(content), usage=usage())


def text(t):
    return NS(type="text", text=t)


def tool_use(id, name, input):
    return NS(type="tool_use", id=id, name=name, input=input)


class FakeConnectionError(anthropic.APIConnectionError):
    def __init__(self):  # skip the SDK constructor, which needs a real HTTP request object
        Exception.__init__(self, "connection error")


class FakeClient:
    """Stands in for anthropic.Anthropic: returns (or raises) scripted items in order."""

    def __init__(self, *items):
        self._items = list(items)
        self.calls = []
        self.beta = NS(messages=self)

    def create(self, **kwargs):
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        item = self._items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item
```

- [ ] **Step 2: Write the failing tests**

`tests/agent/test_loop.py`:

```python
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
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/agent/test_loop.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'brokerchat.agent.loop'`.

- [ ] **Step 4: Implement the loop**

`src/brokerchat/agent/loop.py`:

```python
"""The agent loop: one chat message in, tool calls out, events on the board (PRD §6).

Hand-written over the Messages API (not the SDK Tool Runner) because owning the loop is the point of the project.
Each chat message starts a fresh, append-only conversation: context in, tool calls and results appended until Claude
stops calling tools.
"""

from typing import Any

import anthropic

from brokerchat.agent.prompt import build_context, build_system_prompt
from brokerchat.agent.tools import TOOL_SCHEMAS, ToolExecutor
from brokerchat.glossary import Glossary
from brokerchat.models import ChatMessage
from brokerchat.pipeline import AgentResult, Usage
from brokerchat.store import Store

MODEL = "claude-opus-5-5"
MAX_TOKENS = 16000
DEFAULT_EFFORT = "medium"
DEFAULT_MAX_ITERATIONS = 8
DEFAULT_CONTEXT_MESSAGES = 20
# Server-side refusal fallback: on a policy decline the API re-runs the request on a suitable fallback model.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class ClaudeAgent:
    def __init__(
        self,
        client: Any,
        glossary: Glossary,
        *,
        model: str = MODEL,
        effort: str = DEFAULT_EFFORT,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        context_messages: int = DEFAULT_CONTEXT_MESSAGES,
    ) -> None:
        self.client = client
        self.model = model
        self.effort = effort
        self.max_iterations = max_iterations
        self.context_messages = context_messages
        # The system prompt never changes during a run, so it is cached across messages.
        self.system = [{"type": "text", "text": build_system_prompt(glossary), "cache_control": {"type": "ephemeral"}}]

    def _call(self, messages: list[dict[str, Any]]) -> Any:
        return self.client.beta.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=self.system,
            tools=TOOL_SCHEMAS,
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort},
            betas=[FALLBACK_BETA],
            fallbacks="default",
            messages=messages,
        )

    def process(self, message: ChatMessage, store: Store) -> AgentResult:
        executor = ToolExecutor(store, message)
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": build_context(store, message, self.context_messages)}
        ]
        usage = Usage()

        def result(status: str, reason: str | None = None) -> AgentResult:
            return AgentResult(message_id=message.id, status=status, reason=reason, tool_calls=executor.calls, usage=usage)

        for _ in range(self.max_iterations):
            try:
                response = self._call(messages)
            except anthropic.APIStatusError as e:  # the SDK has already retried 429s and 5xx
                return result("unprocessed", f"api_error {e.status_code}")
            except anthropic.APIConnectionError:
                return result("unprocessed", "connection_error")

            usage.add(Usage.from_api(response.usage))
            if response.stop_reason in ("refusal", "max_tokens", "pause_turn"):
                return result("unprocessed", response.stop_reason)

            tool_uses = [block for block in response.content if block.type == "tool_use"]
            if not tool_uses:
                return result("ok")

            # Append the whole assistant turn (thinking blocks included), then all tool results in one user turn.
            messages.append({"role": "assistant", "content": response.content})
            messages.append(
                {"role": "user", "content": [executor.execute(b.name, b.input, b.id) for b in tool_uses]}
            )

        executor.raise_cap_flag()
        return result("cap_hit", f"no end_turn after {self.max_iterations} model calls")
```

- [ ] **Step 5: Wire the agent into the CLI**

In `src/brokerchat/cli.py`, replace:

```python
AGENT_CHOICES = ["null"]
```

with:

```python
AGENT_CHOICES = ["claude", "null"]
```

Then replace the `make_agent` function with:

```python
def make_agent(name: str, glossary: Glossary, effort: str = "medium") -> Processor:
    if name == "null":
        return NullAgent()
    if name == "claude":
        import anthropic

        from brokerchat.agent.loop import ClaudeAgent

        return ClaudeAgent(anthropic.Anthropic(), glossary, effort=effort)
    raise ValueError(f"unknown agent {name!r}")
```

The anthropic import is local so that `--agent null` never needs credentials. The existing CLI tests pass `--agent null` explicitly, so the new default (`claude`) doesn't affect them.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/agent/test_loop.py tests/test_cli.py tests/test_evalrun.py -v`
Expected: all PASS.

- [ ] **Step 7: Checkpoint**

Run `uv run pytest -q` and expect all to pass. Tell the user Task 12 is ready. Do not commit.

---

### Task 13: Live smoke test, baseline eval and tuning (M3, M4)

This task spends real API money. **Get the user's explicit go-ahead before Step 2 and again before Step 4**, and tell them the expected cost each time.

**Files:**
- Create: `tests/live/__init__.py`, `tests/live/test_live_agent.py`, `evals/RESULTS.md`
- Modify (during tuning only): `src/brokerchat/agent/prompt.py`, the tool descriptions in `src/brokerchat/agent/tools.py`, and `DEFAULT_EFFORT` in `src/brokerchat/agent/loop.py`

**Interfaces:**
- Consumes: everything above. Produces no new code interfaces.

- [ ] **Step 1: Write the live test**

`tests/live/__init__.py`: empty file.

`tests/live/test_live_agent.py`:

```python
import os

import anthropic
import pytest

from brokerchat.agent.loop import ClaudeAgent
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
```

Run `uv run pytest -q` and expect all to pass, with the live test skipped.

- [ ] **Step 2: Run the live smoke test (needs user go-ahead, about 10 messages)**

Check credentials with `ant auth status`. If none are set, ask the user to run `! ant auth login` or set `ANTHROPIC_API_KEY`.

PowerShell: `$env:BROKERCHAT_LIVE="1"; uv run pytest tests/live -v -s`
Bash: `BROKERCHAT_LIVE=1 uv run pytest tests/live -v -s`

Expected: PASS, with the printed cost and score.

If the API rejects the `fallbacks` keyword (`TypeError` from the SDK) or returns a 400 naming it, move it into `extra_body={"fallbacks": "default"}` in `ClaudeAgent._call`, update `test_message_without_tool_calls_is_ok` to match, and tell the user.

- [ ] **Step 3: Estimate the full-eval cost**

Run: `uv run brokerchat eval --case basic_flow --case day01`
Multiply the printed cost by roughly 12 / 2 to estimate one full golden run. Report the estimate to the user.

- [ ] **Step 4: Run the baseline (needs user go-ahead at the estimated cost)**

Run: `uv run brokerchat eval --show-failures`

Create `evals/RESULTS.md` and record the run:

```markdown
# Eval results

Targets: S1 ≥ 0.95, S2 ≥ 0.95, S3 = 0, S4 ≥ 0.90 (PRD §3).

| Run | Change | S1 fields | S2 status | S3 unsupported | S4 flag recall | Flag precision | Cost |
|-----|--------|-----------|-----------|----------------|----------------|----------------|------|
| <timestamp> | baseline (effort medium) | | | | | | |
```

Fill the row from the printed table.

- [ ] **Step 5: Tune one change at a time (M4)**

Repeat until all four targets are met, or for at most 5 iterations, then review with the user:
1. Read the `--show-failures` lines and the per-case `events.jsonl` in the run directory to find the most common failure pattern.
2. Make **one** change aimed at it: system prompt wording, a tool description, or the effort level.
3. Run `uv run pytest -q`. Prompt and tool-schema tests must still pass. Update a test only if the change deliberately alters a tested contract.
4. Re-run the eval, preferably with `--case` on the affected cases first, and add a row to `evals/RESULTS.md`.
5. Keep the change only if the targeted metric improves without another one regressing.

- [ ] **Step 6: Demo prep (M5)**

Run: `uv run brokerchat replay evals/golden/day01.jsonl --speed 30 --out runs/demo-day01`
Expected: a live board for your friend to review (PRD S5). Remind the user to bring `domain/glossary.yaml` and the PRD §4 shorthand table to that review.

- [ ] **Step 7: Checkpoint**

Run `uv run pytest -q` and expect all to pass. Summarise the final metrics from `evals/RESULTS.md` for the user. Do not commit.
