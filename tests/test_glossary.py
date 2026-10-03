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
