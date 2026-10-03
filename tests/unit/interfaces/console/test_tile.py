"""widgets/Tile: the title strip of a tile carries its value only when there is room for both,
a row of coloured segments drops a segment whole rather than cutting it, and tinting grounds
the strip and the body in their two surfaces."""
import pytest

from interfaces.console import Theme
from interfaces.console.framework.TUIScreen import Color, TUIScreen
from interfaces.console.widgets import Tile

C1, C2 = Color(200, 10, 10), Color(10, 200, 10)
X, Y = 3, 2


def cells(screen: TUIScreen, y: int, x: int, width: int):
    return screen.next_frame[y - 1][x - 1:x - 1 + width]


def row(screen: TUIScreen, y: int, x: int, width: int) -> str:
    return screen.row_text(x, y, width).rstrip()


@pytest.mark.unit
class TestWriteSegments:
    def test_segments_follow_each_other_in_their_own_colours_with_a_faint_dot_between(self):
        screen = TUIScreen(40, 4)
        Tile.write_segments(screen, X, Y, 40, [("6h ago", C1), ("2 nights", C2)])
        text = row(screen, Y, X, 40)
        assert text == "6h ago · 2 nights"
        colours = [c.style.fg for c in cells(screen, Y, X, len(text))]
        assert colours[0] == C1 and colours[-1] == C2
        assert colours[text.index("·")] == Theme.COLOR_FAINT

    def test_a_segment_that_does_not_fit_is_left_out_whole_with_everything_after_it(self):
        both = len("abc · de")
        screen = TUIScreen(40, 4)
        Tile.write_segments(screen, X, Y, both, [("abc", C1), ("de", C2), ("f", C1)])
        assert row(screen, Y, X, 40) == "abc · de"
        screen = TUIScreen(40, 4)
        Tile.write_segments(screen, X, Y, both - 1, [("abc", C1), ("de", C2), ("f", C1)])
        assert row(screen, Y, X, 40) == "abc"

    def test_the_first_segment_alone_is_cut_to_the_width(self):
        screen = TUIScreen(40, 4)
        Tile.write_segments(screen, X, Y, 8, [("a rather long text", C1)])
        text = row(screen, Y, X, 40)
        assert len(text) == 8 and text.endswith("...")


@pytest.mark.unit
class TestDrawCard:
    WIDTH = 24

    def test_the_value_sits_at_the_right_end_of_the_title_strip(self):
        screen = TUIScreen(40, 6)
        Tile.draw_card(screen, [], X, Y, self.WIDTH, 4, "Size", "5.0 KB")
        strip = screen.row_text(X, Y, self.WIDTH)
        assert strip.index("Size") < strip.index("5.0 KB")
        assert strip.index("5.0 KB") + len("5.0 KB") == self.WIDTH - 2, "two cells of margin at the right"

    def test_a_value_the_strip_has_no_room_for_is_dropped_and_the_label_stays(self):
        screen = TUIScreen(40, 6)
        Tile.draw_card(screen, [], X, Y, len("Size") + len("5.0 KB") + 2, 4, "Size", "5.0 KB")
        strip = row(screen, Y, X, self.WIDTH)
        assert "Size" in strip and "5.0 KB" not in strip

    def test_tinting_grounds_the_strip_and_the_body_in_their_two_surfaces(self):
        screen = TUIScreen(40, 6)
        cards = []
        Tile.draw_card(screen, cards, X, Y, self.WIDTH, 3, "Size")
        Tile.tint_cards(screen, cards)
        strip, body, below = (cells(screen, y, X, self.WIDTH) for y in (Y, Y + 2, Y + 3))
        assert {c.style.bg for c in strip} == {Theme.COLOR_TITLE_BG}
        assert {c.style.bg for c in body} == {Theme.COLOR_PANEL_BG}
        assert all(c.style.bg != Theme.COLOR_PANEL_BG for c in below), "the tint stops at the tile's height"
        assert cells(screen, Y, X - 1, 1)[0].style.bg != Theme.COLOR_TITLE_BG, "and at its left edge"
