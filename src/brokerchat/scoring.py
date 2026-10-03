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
    unsupported = 0
    for q in board.quotes.values():
        spurious = (q.instrument.name, q.instrument.tenor, q.source) not in expected_keys
        if spurious:
            s.spurious_quotes.append(f"{q.quote_id} {q.instrument.name} {q.instrument.tenor} {q.source}")
        if spurious or not set(q.source_message_ids) <= message_ids:
            unsupported += 1

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

    # S3: an entry is unsupported if the chat does not say it: it cites unknown messages, or no expected entry
    # matches it. Citing a real but unrelated message must not make an invented entry look supported.
    s.unsupported_entries = unsupported + len(remaining_axes) + len(remaining_interests)

    # S4: credit only an explicit raise_flag on a message the agent did not also record a guess from.
    flags = [f for f in board.flags.values() if f.kind in SCORED_FLAG_KINDS and f.origin == "agent"]
    flagged = {m for f in flags for m in f.message_ids}
    guessed = {m for e in (*board.quotes.values(), *board.axes, *board.interests) for m in e.source_message_ids}
    ambiguous = set(expected.ambiguous_message_ids)
    for message_id in expected.ambiguous_message_ids:
        if message_id in flagged and message_id not in guessed:
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
