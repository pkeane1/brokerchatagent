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
# Phrasings deliberately include variants the system prompt does not spell out ("cxl that", "lifted", "+1",
# "10y too"), so the golden set measures understanding of the conventions rather than template matching.
REFER_PHRASES = ("refer", "off that", "pulled", "refer that", "cxl that", "out")
UNCH_PHRASES = ("unch", "still there", "unchanged", "still good")
OPS = frozenset(
    {
        "quote", "one_sided", "refer", "trade", "shift", "unch", "same_again", "two_markets",
        "axe", "interest", "chatter", "ambiguous_unknown", "ambiguous_no_context", "ambiguous_refer",
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
        side = self.rng.choice([s for s, x in (("bid", q.bid), ("offer", q.offer)) if x is not None])
        level = fmt_decimal(q.bid if side == "bid" else q.offer)
        templates = [f"trades {level}", f"done at {level}", f"dealt {level}"]
        templates.append(f"hit {level}" if side == "bid" else f"lifted {level}")
        self._emit(room, sender, self.rng.choice(templates))
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
        signed = f"+{n}" if bp > 0 else f"-{n}"
        self._emit(room, sender, self.rng.choice([f"{n} {word}", f"move {n}bp {word}", f"{word} by {n}", signed]))
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
        self._emit(
            room,
            sender,
            self.rng.choice([f"same in {t} {market}", f"and {t} {market}", f"{n}s {market}", f"{t} too {market}"]),
        )
        self._set_quote(room, sender, instrument, tenor, bid, offer)

    def two_markets(self, room, sender, instrument, tenor, bid, offer, instrument2, tenor2, bid2, offer2) -> None:
        """One message quoting two markets. A bare refer right after it is ambiguous (see ambiguous_refer)."""
        if (instrument, tenor) == (instrument2, tenor2):
            raise ScenarioError("two_markets needs two different markets")
        bid, offer, bid2, offer2 = D(bid), D(offer), D(bid2), D(offer2)
        first = f"{self._alias(instrument)} {self._tenor(tenor)} {fmt_decimal(bid)}/{fmt_decimal(offer)}"
        second = f"{self._alias(instrument2)} {self._tenor(tenor2)} {fmt_decimal(bid2)}/{fmt_decimal(offer2)}"
        self._emit(room, sender, f"{first}, {second}")
        self._set_quote(room, sender, instrument, tenor, bid, offer)
        self._set_quote(room, sender, instrument2, tenor2, bid2, offer2)
        self.last_action[(room, sender)] = "multi"

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

    def ambiguous_refer(self, room, sender) -> None:
        """A bare refer when the sender's last message quoted several markets: which one is pulled?"""
        if self.last_action.get((room, sender)) != "multi":
            raise ScenarioError(f"{sender}'s last message in {room} did not quote several markets")
        message = self._emit(room, sender, self.rng.choice(REFER_PHRASES))
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
