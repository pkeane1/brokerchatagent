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
