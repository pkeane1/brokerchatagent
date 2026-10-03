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
