"""widgets/Board: the summary row drops tiles a narrow screen has no room for, the table brings
the selected row into view and draws a meter cell as a meter, the facts tile puts the meter
before its value, and the layout helpers split the width the way every board tab did."""
import pytest

from interfaces.console import Theme
from interfaces.console.framework.TUIScreen import Color, TUIScreen
from interfaces.console.widgets import Board, Tile
from interfaces.console.widgets.Board import Fact, Meter, Table

C = Color(200, 10, 10)
FULL = Tile.METER_BLOCKS[8]


def text(screen: TUIScreen, y: int) -> str:
    return screen.row_text(1, y, screen.width).rstrip()


def rows(screen: TUIScreen):
    return [text(screen, y) for y in range(1, screen.height + 1)]


def style(screen: TUIScreen, y: int, x: int):
    return screen.next_frame[y - 1][x - 1].style


@pytest.mark.unit
class TestLayout:
    @pytest.mark.parametrize("available,count", [(100, 3), (101, 4), (57, 2)])
    def test_columns_span_the_width_with_even_gaps_and_the_last_takes_the_remainder(self, available, count):
        columns = Board.columns(available, count)
        assert columns[0][0] == Board.LEFT
        for (x, w), (next_x, _) in zip(columns, columns[1:]):
            assert next_x - (x + w) == Board.GAP
        last_x, last_w = columns[-1]
        assert last_x + last_w == Board.LEFT + available

    def test_the_side_tile_takes_its_share_but_never_less_than_its_minimum(self):
        table_w, side_w = Board.split(200)
        assert side_w == 72 and table_w + Board.GAP + side_w == 200
        assert Board.split(100)[1] == Board.SIDE_MIN_W
        assert Board.split(100, 60)[1] == 60

    def test_scrolled_keeps_the_window_inside_the_list(self):
        assert Board.scrolled(12, 20, 5) == 12
        assert Board.scrolled(30, 20, 5) == 15
        assert Board.scrolled(3, 2, 5) == 0

    def test_scrolled_moves_the_least_that_brings_the_selected_row_into_view(self):
        assert Board.scrolled(0, 20, 5, selected=2) == 0
        assert Board.scrolled(0, 20, 5, selected=7) == 3, "the selected row becomes the last one shown"
        assert Board.scrolled(10, 20, 5, selected=4) == 4, "... or the first one"

    def test_steps_colour_the_answered_the_waited_on_and_the_coming_fields_apart(self):
        facts = Board.steps(1, [("Kind", "phone"), ("Description", "typing"), ("Value", "–")])
        assert [fact.waiting for fact in facts] == [False, True, False]
        assert facts[0].color == Theme.COLOR_TEXT and facts[1].color == Theme.COLOR_BRIGHT
        assert facts[2].color == Theme.COLOR_FAINT


@pytest.mark.unit
class TestSummary:
    TILES = [("Size", "5 KB", C, [("40 words", C)]), ("Sections", "3", C, [("7 headings", C)]),
             ("Snapshot", "1 of 4", C, [("Monday", C)])]

    def test_every_tile_shows_when_the_width_allows(self):
        screen = TUIScreen(120, 8)
        Board.draw_summary(screen, [], 2, 115, self.TILES)
        strip = text(screen, 2)
        assert "Size" in strip and "Sections" in strip and "Snapshot" in strip
        assert "40 words" in text(screen, 2 + Tile.CARD_TOP)

    def test_a_narrow_screen_keeps_the_first_tiles_and_drops_the_rest(self):
        screen = TUIScreen(80, 8)
        Board.draw_summary(screen, [], 2, 2 * Board.SUMMARY_TILE_MIN_W + 5, self.TILES)
        strip = text(screen, 2)
        assert "Size" in strip and "Sections" in strip and "Snapshot" not in strip


@pytest.mark.unit
class TestTable:
    @staticmethod
    def table(count: int, meter: bool = False) -> Table:
        cells = [("in 2h", Theme.COLOR_TEXT)] if not meter else [Meter(0.5, C, 8, "in 2h", Theme.COLOR_TEXT)]
        return Table("Items", [("Name", 12), ("Next", None)],
                     [(C, [(f"item-{i:02d}", Theme.COLOR_TEXT)] + cells) for i in range(count)], "nothing yet")

    def test_the_rows_sit_under_the_column_titles_with_the_count_in_the_strip(self):
        screen = TUIScreen(60, 12)
        Board.draw_table(screen, [], 1, 1, 50, 10, self.table(3))
        lines = rows(screen)
        assert lines[0].rstrip().endswith("3")
        header = next(i for i, line in enumerate(lines) if "Name" in line and "Next" in line)
        assert "item-00" in lines[header + 1] and lines[header + 1].lstrip().startswith("●")
        assert "item-02" in lines[header + 3]

    def test_an_empty_table_says_so_instead_of_titles(self):
        screen = TUIScreen(60, 12)
        Board.draw_table(screen, [], 1, 1, 50, 10, self.table(0))
        lines = rows(screen)
        assert any("nothing yet" in line for line in lines) and not any("Name" in line for line in lines)

    def test_the_selected_row_is_tinted_and_brought_into_view(self):
        screen = TUIScreen(60, 10)
        height = 4 + Tile.CARD_CHROME        # three rows under the titles
        top = Board.draw_table(screen, [], 1, 1, 50, height, self.table(20), top=0, selected=7)
        assert top == 5
        lines = rows(screen)
        selected_y = next(i for i, line in enumerate(lines) if "item-07" in line) + 1
        assert style(screen, selected_y, 3).bg == Theme.COLOR_SELECTED_BG
        assert "item-05" in lines[selected_y - 3]

    def test_a_meter_cell_is_a_meter_with_its_text_after_it(self):
        screen = TUIScreen(60, 10)
        Board.draw_table(screen, [], 1, 1, 50, 8, self.table(1, meter=True))
        line = next(line for line in rows(screen) if "item-00" in line)
        filled = line.count(FULL)
        assert filled == 4 and line.index("in 2h") > line.index(FULL)

    def test_a_bold_cell_is_drawn_bold(self):
        screen = TUIScreen(60, 10)
        table = Table("Items", [("Name", None)], [(C, [("shown", Theme.COLOR_TEXT, True)]), (C, [("plain", Theme.COLOR_TEXT)])], "")
        Board.draw_table(screen, [], 1, 1, 50, 8, table)
        lines = rows(screen)
        bold_y = next(i for i, line in enumerate(lines) if "shown" in line) + 1
        assert style(screen, bold_y, lines[bold_y - 1].index("shown") + 1).bold
        assert not style(screen, bold_y + 1, lines[bold_y].index("plain") + 1).bold


@pytest.mark.unit
class TestFacts:
    def test_labels_are_dim_and_the_waited_on_one_takes_the_prompt_colour(self):
        screen = TUIScreen(60, 8)
        after = Board.draw_facts(screen, 1, 1, 50, 8, [Fact("Kind", "phone", Theme.COLOR_TEXT),
                                                       Fact("Value", "typing", Theme.COLOR_BRIGHT, waiting=True)])
        assert after == 1 + Tile.CARD_TOP + 2
        lines = rows(screen)
        kind_y = next(i for i, line in enumerate(lines) if "Kind" in line) + 1
        assert style(screen, kind_y, 3).fg == Theme.COLOR_DIM
        assert style(screen, kind_y + 1, 3).fg == Board.COLOR_PROMPT
        assert style(screen, kind_y + 1, 3 + Board.LABEL_W).bold

    def test_a_meter_fact_draws_the_meter_before_a_bright_bold_value(self):
        screen = TUIScreen(80, 8)
        Board.draw_facts(screen, 1, 1, 70, 8, [Fact("Score", "87", C, meter=0.87)])
        line = next(line for line in rows(screen) if "Score" in line)
        assert FULL in line and line.index("87") > line.rindex(FULL)
        y = rows(screen).index(line) + 1
        assert style(screen, y, line.index("87") + 1).bold

    def test_rows_past_the_tile_are_not_drawn(self):
        screen = TUIScreen(60, 8)
        facts = [Fact(f"Row {i}", "x", Theme.COLOR_TEXT) for i in range(6)]
        after = Board.draw_facts(screen, 1, 1, 50, 2 + Tile.CARD_CHROME, facts)
        assert after == 1 + Tile.CARD_TOP + 2
        joined = "\n".join(rows(screen))
        assert "Row 1" in joined and "Row 2" not in joined

    def test_the_footer_takes_the_last_row_only_when_it_is_free(self):
        screen = TUIScreen(60, 8)
        Board.draw_footer(screen, 1, 1, 50, 6, "Saved", Theme.COLOR_GREEN, above=4)
        assert "Saved" in text(screen, 1 + 6 - 2)
        screen = TUIScreen(60, 8)
        Board.draw_footer(screen, 1, 1, 50, 6, "Saved", Theme.COLOR_GREEN, above=6)
        assert "Saved" not in "\n".join(rows(screen))


@pytest.mark.unit
class TestTableWidth:
    def test_columns_the_tile_has_no_room_for_are_left_out_and_nothing_runs_past_its_edge(self):
        screen = TUIScreen(60, 10)
        table = Table("Items", [("Name", 12), ("Kind", 12), ("Where", 12)],
                      [(C, [("item-00", Theme.COLOR_TEXT), ("phone", Theme.COLOR_TEXT), ("drawer", Theme.COLOR_TEXT)])], "")
        x, width = 1, 24                      # inner 18 cells: the name column, a gap and four for the kind
        Board.draw_table(screen, [], x, 1, width, 8, table)
        lines = rows(screen)
        row = next(line for line in lines if "item-00" in line)
        assert "phone" not in row and row.endswith("...") and "drawer" not in row
        assert "Where" not in "\n".join(lines)
        assert max(len(line) for line in lines) <= x + width - 1
