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
    "two_markets": 3, "ambiguous_refer": 3,
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
        if op == "ambiguous_refer":
            multi = [rs for rs, action in b.last_action.items() if action == "multi"]
            if multi:
                room, sender = rng.choice(sorted(multi))
            else:
                op = "two_markets"
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
            case "two_markets":
                second = rng.choice([i for i in BASE_LEVEL if i != instrument])
                unit2 = glossary.get(second).unit
                tenor2 = "5Y" if unit2 == Unit.PRICE else rng.choice(list(TENOR_FACTOR))
                bid, offer = _market(rng, instrument, tenor, unit)
                bid2, offer2 = _market(rng, second, tenor2, unit2)
                b.two_markets(room, sender, instrument, tenor, bid, offer, second, tenor2, bid2, offer2)
            case "ambiguous_refer":
                b.ambiguous_refer(room, sender)
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
