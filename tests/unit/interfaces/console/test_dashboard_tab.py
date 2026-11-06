"""DashboardTab._draw_columns: the one chart core every dashboard chart goes through."""
import re

import pytest

from interfaces.console.framework.TUIScreen import Color, TUIScreen
from interfaces.console import Theme
from interfaces.console.tabs.DashboardTab import DashboardTab
from interfaces.console.widgets import Board, Tile

C = Color(200, 100, 50)
CHART_X, CHART_Y, CHART_W, CHART_H = 2, 2, 5, 4


def make_tab():
    tab = DashboardTab.__new__(DashboardTab)   # no monitors: the drawing helpers need none
    tab._cards = []
    tab._chart_pictures = {}
    return tab


def column(screen, x):
    return "".join(screen.next_frame[CHART_Y - 1 + row][x - 1].char for row in range(CHART_H))


def draw(history, max_value=100.0, guides=None):
    screen = TUIScreen(20, 10)
    make_tab()._draw_columns(screen, CHART_X, CHART_Y, CHART_W, CHART_H, history, max_value,
                             lambda v: C, guides)
    return screen


@pytest.mark.unit
class TestDrawColumns:
    def test_the_newest_sample_is_the_rightmost_column_and_short_history_pads_left(self):
        screen = draw([0.0, 0.0, 100.0])
        assert column(screen, CHART_X + 4) == "▄▄▄▄"
        assert all(column(screen, CHART_X + i) == "    " for i in range(4))

    def test_long_history_keeps_the_tail(self):
        screen = draw([100.0] * 10 + [0.0] * 5)
        assert all(column(screen, CHART_X + i) == "    " for i in range(5))
        screen = draw([0.0] * 10 + [100.0] * 5)
        assert all(column(screen, CHART_X + i) == "▄▄▄▄" for i in range(5))

    def test_a_partial_row_uses_a_fractional_block(self):
        screen = draw([0.0, 0.0, 0.0, 0.0, 62.5])     # 2.5 rows of 4
        assert column(screen, CHART_X + 4) == " \u2584▄▄"

    def test_a_zero_scale_draws_an_empty_chart_instead_of_raising(self):
        screen = draw([1.0, 2.0], max_value=0.0)
        assert all(column(screen, CHART_X + i) == "    " for i in range(CHART_W))

    def test_a_full_cell_is_two_shades_and_never_shows_the_ground(self):
        # Terminal.app leaves seams between cells; a bar colour behind the block hides them
        screen = draw([0.0, 0.0, 0.0, 0.0, 62.5])
        full = screen.next_frame[CHART_Y - 1 + 3][CHART_X + 3].style      # bottom row, full cell
        assert Theme.COLOR_PANEL_BG not in (full.fg, full.bg) and full.fg != full.bg
        partial = screen.next_frame[CHART_Y - 1 + 1][CHART_X + 3].style   # the half block on top
        assert partial.bg == Theme.COLOR_PANEL_BG, "a partial block keeps the ground so its height reads"

    def test_the_shading_steps_every_half_row_up_to_the_colour(self):
        tall = 2 * DashboardTab.SHADE_MIN_ROWS      # tall enough for the shading to run its whole range
        screen = TUIScreen(20, tall + 4)
        make_tab()._draw_columns(screen, CHART_X, CHART_Y, 1, tall, [100.0], 100.0, lambda v: C)
        shades = []
        for row in reversed(range(tall)):           # bottom cell first; in a cell the glyph is the lower half
            style = screen.next_frame[CHART_Y - 1 + row][CHART_X - 1].style
            shades += [style.fg, style.bg]
        assert len(set(shades)) == 2 * tall and shades[-1] == C

    def test_per_sample_colours_follow_their_columns_through_cut_and_padding(self):
        other = Color(50, 100, 200)

        def bottoms(colors=None, color_for=lambda v: C, history=(100.0,) * 7):
            screen = TUIScreen(20, 10)
            make_tab()._draw_columns(screen, CHART_X, CHART_Y, CHART_W, CHART_H, list(history), 100.0, color_for,
                                     colors=colors)
            return [screen.next_frame[CHART_Y - 1 + 3][CHART_X - 1 + i].style.fg for i in range(CHART_W)]

        in_c, in_other = bottoms()[0], bottoms(color_for=lambda v: other)[0]
        # seven samples on five columns: the first two fall off, and their colours with them
        assert bottoms(colors=[other, other, C, other, C, C, other]) == [in_c, in_other, in_c, in_c, in_other]
        padded = bottoms(colors=[other], history=(100.0,))
        assert padded[-1] == in_other and padded[0] == Theme.COLOR_PANEL_BG

    def test_guides_show_only_where_no_bar_covers_them(self):
        screen = draw([0.0, 0.0, 0.0, 0.0, 75.0], guides={1: C, 3: C})   # rows count up from the base
        assert column(screen, CHART_X) == "┄ ┄ "
        assert column(screen, CHART_X + 4) == "┄▄▄▄"


# --- Feeds panel -------------------------------------------------------------------------------

from unittest.mock import patch

from interfaces.console.widgets.FeedStrip import SYMBOL_FAILED, SYMBOL_PAUSED, SYMBOL_RUNNING

PANEL_X, PANEL_Y, PANEL_W, PANEL_H = 1, 1, 64, 4 + Tile.CARD_CHROME   # room for four feeds


def status(state="live", **extra) -> dict:
    base = {"state": state, "paused": False, "items": "items", "stored": 23436, "remaining": None,
            "total": None, "progress": None, "eta_seconds": None, "last_error": None}
    base.update(extra)
    return base


class StubFeeds:
    def __init__(self, feeds: dict):
        self.feeds = feeds

    def names(self):
        return list(self.feeds)

    def status(self, name):
        return self.feeds.get(name)


def draw_feeds(feeds: dict, frame: int = 0, width: int = PANEL_W):
    tab = make_tab()
    tab._frame = frame
    screen = TUIScreen(width + 4, 8)
    screen.clear_buffer()
    with patch("interfaces.console.tabs.DashboardTab.Feeds.get_instance", return_value=StubFeeds(feeds)):
        tab._draw_feeds_panel(screen, PANEL_X, PANEL_Y, width, PANEL_H)
    return screen


def panel_row(screen, row: int):
    """A row of the panel: 0 is the title strip, 1 the first content row under the tile's blank row."""
    return screen.next_frame[PANEL_Y - 1 + (row + Tile.CARD_TOP - 1 if row else 0)]


def row_text(screen, row: int) -> str:
    return "".join(c.char for c in panel_row(screen, row)).rstrip()


@pytest.mark.unit
class TestFeedRow:
    def test_live_is_a_full_green_bar_with_the_count_and_the_feeds_noun(self):
        row = DashboardTab._feed_row("iMessage", status(items="messages"))
        assert (row.symbol, row.progress, row.text) == (SYMBOL_RUNNING, 1.0, "23 436 messages")
        assert row.fg == Theme.COLOR_GREEN
        assert DashboardTab._feed_row("Indigo", status(stored=512, items="events")).text == "512 events", \
            "an uncountable feed shows its rows"

    def test_a_catch_up_whose_counts_meet_shows_the_count_once(self):
        assert DashboardTab._feed_row("Photos", status("catching_up", stored=10, total=10, remaining=0)).text == "10 items"

    def test_catching_up_with_a_count_shows_done_over_total_split_by_thousands(self):
        row = DashboardTab._feed_row("iCloud Photos", status("catching_up", progress=0.34, stored=1234, total=23356, remaining=22122))
        assert (row.symbol, row.name, row.progress, row.text) == (DashboardTab.SYMBOL_CATCHING_UP, "Photos", 0.34, "1 234 / 23 356 items")
        assert row.fg == Theme.COLOR_YELLOW

    def test_catching_up_with_an_eta_appends_it_in_one_rounded_unit(self):
        def text(eta):
            return DashboardTab._feed_row("Photos", status("catching_up", stored=1234, total=22313, remaining=21079, eta_seconds=eta)).text
        assert text(13 * 3600) == "1 234 / 22 313 items ETA: 13h"
        assert text(13 * 60 + 20) == "1 234 / 22 313 items ETA: 13m"
        assert text(45) == "1 234 / 22 313 items ETA: 45s"
        assert text(3 * 86400) == "1 234 / 22 313 items ETA: 3d"
        assert text(47 * 3600) == "1 234 / 22 313 items ETA: 47h", "days only from two days up"
        assert text(None) == "1 234 / 22 313 items"

    def test_catching_up_without_a_count_is_indeterminate_and_shows_its_rows(self):
        row = DashboardTab._feed_row("Indigo", status("catching_up", stored=1345))
        assert (row.progress, row.text) == (None, "1 345 items")

    def test_paused_hollows_the_dot_fades_the_colour_and_says_only_that(self):
        row = DashboardTab._feed_row("Indigo", status("catching_up", progress=0.5, stored=5, total=10, remaining=5, paused=True))
        assert (row.symbol, row.text, row.progress) == (SYMBOL_PAUSED, "paused", 0.5)
        assert row.fg != Theme.COLOR_YELLOW

    def test_stopped_and_failed_are_red_empty_bars_saying_only_that(self):
        row = DashboardTab._feed_row("Telegram", status("stopped", last_error="no session file"))
        assert (row.symbol, row.progress, row.text) == (SYMBOL_FAILED, 0.0, "stopped")
        assert row.fg == Theme.COLOR_RED
        failed = DashboardTab._feed_row("Telegram", status("failed", paused=True))
        assert (failed.symbol, failed.text) == (SYMBOL_FAILED, "failed"), "a dead feed is never shown as paused"

    def test_starting_is_a_grey_empty_bar(self):
        row = DashboardTab._feed_row("iMessage", status("starting"))
        assert (row.symbol, row.progress, row.text) == (SYMBOL_RUNNING, 0.0, "starting")
        assert row.fg == Theme.COLOR_DIM


BLOCKS, NOTCH = Tile.METER_BLOCKS, Tile.METER_MARKER
FULL = BLOCKS[8]


def cells(screen, row: int, char: str):
    """The cells of a panel row holding `char`."""
    return [c for c in panel_row(screen, row) if c.char == char]


def filled(screen, row: int, track: Color) -> int:
    """How many cells of a row's bar are filled whole."""
    return sum(1 for c in cells(screen, row, FULL) if c.style.fg != track)


@pytest.mark.unit
class TestDrawFeedsPanel:
    def test_one_row_per_feed_with_name_symbol_bar_and_count(self):
        screen = draw_feeds({
            "iMessage": status(),
            "iCloud Photos": status("catching_up", progress=0.34, stored=1234, total=23356, remaining=22122),
            "Telegram": status("stopped", last_error="no session file"),
        })
        assert row_text(screen, 0).strip() == "Feeds"
        assert row_text(screen, 1).startswith(f"  iMessage {SYMBOL_RUNNING} {FULL}")
        assert row_text(screen, 1).endswith(f"{FULL} 23 436 items")
        assert row_text(screen, 2).startswith(f"  Photos   {DashboardTab.SYMBOL_CATCHING_UP} {FULL}")
        assert "1 234 / 23 356" in row_text(screen, 2)
        assert row_text(screen, 3).startswith(f"  Telegram {SYMBOL_FAILED}")
        assert row_text(screen, 3).endswith(" stopped") and FULL not in row_text(screen, 3)
        assert row_text(screen, 4).startswith("  Total ") and row_text(screen, 4).endswith(" 48 106 items")

    def test_the_total_sums_every_feeds_rows_in_items_whatever_their_nouns(self):
        screen = draw_feeds({"a": status(stored=1000, items="messages"), "b": status("catching_up", stored=500)})
        assert row_text(screen, 3).endswith(" 1 500 items")
        assert FULL not in row_text(screen, 3), "the total has no bar"

    def test_the_bar_takes_every_cell_the_longest_text_leaves(self):
        screen = draw_feeds({"a": status(stored=5), "b": status(stored=1234567)})
        line = row_text(screen, 2)
        assert line.endswith(f"{FULL} 1 234 567 items")
        assert len(line) == PANEL_X + PANEL_W - 3, "the longest text ends one cell of air before the tile's edge"
        assert row_text(screen, 1).rindex(FULL) == line.rindex(FULL), "one text column for every row"

    def test_the_counts_line_up_on_their_last_digit_and_a_state_word_starts_with_them(self):
        screen = draw_feeds({
            "a": status(stored=64116, items="messages"),
            "b": status("catching_up", stored=4022, total=9000, remaining=4978),
            "c": status("stopped"),
        })
        lines = [row_text(screen, row) for row in (1, 2, 4)]
        ends = {line.index(noun) - 1 for line, noun in zip(lines, ("messages", "/ 9 000", "items"))}
        assert len(ends) == 1, "the digits of 64 116, 4 022 and the total end in one column"
        assert row_text(screen, 3).index("stopped") == lines[0].index("64 116")

    def test_the_count_is_drawn_in_the_row_colour_not_grey(self):
        screen = draw_feeds({"iMessage": status()})
        line = row_text(screen, 1)
        cell = panel_row(screen, 1)[line.index("23 436")]
        assert cell.style.fg == Theme.COLOR_GREEN

    def test_the_bar_fill_follows_the_progress(self):
        screen = draw_feeds({"a": status(), "b": status("catching_up", progress=0.5), "c": status("stopped")})
        full = filled(screen, 1, Theme.COLOR_GREEN_DIM)
        half = filled(screen, 2, Theme.COLOR_YELLOW_DIM)
        assert full == len(cells(screen, 1, FULL)) > 0
        assert abs(half - full / 2) <= 1
        assert filled(screen, 3, Theme.COLOR_RED_DIM) == 0

    def test_an_indeterminate_feed_sweeps_a_notch_across_the_bar(self):
        feeds = {"Indigo": status("catching_up")}
        positions = {row_text(draw_feeds(feeds, frame=f), 1).index(NOTCH) for f in (0, 15, 30, 45)}
        assert len(positions) == 4, "the notch moves with the frame"
        bounce = DashboardTab.FEED_SWEEP_FRAMES
        assert row_text(draw_feeds(feeds, frame=10), 1) == row_text(draw_feeds(feeds, frame=bounce - 10), 1)

    def test_a_narrow_panel_keeps_a_minimum_bar_and_cuts_the_text(self):
        screen = draw_feeds({"iMessage": status(stored=123456789)}, width=30)
        line = row_text(screen, 1)
        assert line.count(FULL) == DashboardTab.FEED_MIN_BAR
        assert line.endswith("...") and len(line) == PANEL_X + 30 - 3

    def test_more_feeds_than_rows_are_clipped_to_the_box_and_keep_the_total(self):
        screen = draw_feeds({f"f{i}": status(stored=1) for i in range(6)})
        assert row_text(screen, 3).startswith(f"  f2    {SYMBOL_RUNNING}")
        assert row_text(screen, 4).startswith("  Total ") and row_text(screen, 4).endswith(" 6 items"), \
            "the total counts the clipped feeds too"
        assert row_text(screen, 5) == "", "the tile's closing row stays empty"


@pytest.mark.unit
class TestSummaryHeight:
    """The summary row grows with the feeds and the mics so no row is clipped."""

    @staticmethod
    def height(feeds: int, mics: int = 0) -> int:
        tab = make_tab()
        tab._feed_statuses = lambda: [(f"f{i}", status()) for i in range(feeds)]
        tab._mic_noise_floors = lambda: [(f"mic{i}", None, None, None) for i in range(mics)]
        return tab._summary_height() - Tile.CARD_CHROME

    def test_few_feeds_leave_the_utilization_meters_their_rows(self):
        assert self.height(feeds=0) == self.height(feeds=3) == DashboardTab.SUMMARY_ROWS

    def test_one_row_per_feed_and_the_total_once_they_outnumber_the_meters(self):
        assert self.height(feeds=6) == 7

    def test_the_mics_count_too(self):
        assert self.height(feeds=2, mics=5) == 5


# --- Meters, tiles and the grid ------------------------------------------------------------------

from interfaces.console.widgets.Board import MeterRow

METER_X, METER_Y = 3, 2


def meter(length, fraction, **kwargs) -> str:
    screen = TUIScreen(40, 4)
    Tile.draw_meter(screen, METER_X, METER_Y, length, fraction, C, **kwargs)
    return meter_cells(screen, length)[0]


def meter_cells(screen, length):
    row = screen.next_frame[METER_Y - 1][METER_X - 1:METER_X - 1 + length]
    return "".join(c.char for c in row), [c.style.fg for c in row]


def fill_mask(length, fraction, **kwargs):
    """Per cell: 'F' filled whole, 'p' filled from its left edge to some eighth, 's' a fill
    starting inside the cell (the track's block over the fill's ground), '.' track."""
    screen = TUIScreen(40, 4)
    track = Color(1, 2, 3)
    Tile.draw_meter(screen, METER_X, METER_Y, length, fraction, C, track=track, **kwargs)
    chars, colours = meter_cells(screen, length)
    return "".join("." if ch == " " else "F" if ch == FULL else "s" if fg == track else "p"
                   for ch, fg in zip(chars, colours))


@pytest.mark.unit
class TestDrawMeter:
    def test_the_fill_is_the_fraction_of_the_track_and_the_rest_is_track(self):
        assert fill_mask(10, 0.5) == "FFFFF....."
        assert fill_mask(10, 1.0) == "F" * 10
        assert fill_mask(10, 0.0) == "." * 10

    def test_a_fill_ending_mid_cell_ends_on_that_many_eighths(self):
        assert fill_mask(8, 2.5 / 8) == "FFp....."
        assert meter(8, 2.5 / 8)[2] == BLOCKS[4]
        assert meter(8, 2.75 / 8)[2] == BLOCKS[6]

    def test_anything_above_zero_shows_at_least_an_eighth(self):
        assert fill_mask(10, 0.001) == "p........."
        assert meter(10, 0.001)[0] == BLOCKS[1]

    def test_start_offsets_the_fill_and_a_mid_cell_start_is_drawn_inverted(self):
        assert fill_mask(10, 0.3, start=0.5) == ".....FFF.."
        assert fill_mask(10, 0.35, start=0.45) == "....sFFF.."
        assert meter(10, 0.35, start=0.45)[4] == BLOCKS[4], "the track covers the half before the fill starts"

    def test_a_fill_never_runs_past_the_end_of_the_track(self):
        assert fill_mask(10, 0.9, start=0.8) == "........FF"

    def test_a_cell_has_the_same_shade_whatever_the_value(self):
        def colours(fraction):
            screen = TUIScreen(40, 4)
            Tile.draw_meter(screen, METER_X, METER_Y, 10, fraction, C)
            return meter_cells(screen, 10)[1]
        half, full = colours(0.5), colours(1.0)
        assert half[:5] == full[:5]
        assert full[9] == C and len(set(full)) == 10, "the shading runs the whole track, up to the colour itself"

    def test_the_marker_is_a_notch_at_its_share_of_the_track(self):
        assert meter(10, 0.2, marker=0.7).index(NOTCH) == 7
        assert meter(10, 0.2, marker=1.0).index(NOTCH) == 9, "a marker at full scale stays on the track"

    def test_a_zero_length_meter_draws_nothing(self):
        screen = TUIScreen(40, 4)
        Tile.draw_meter(screen, METER_X, METER_Y, 0, 0.5, C)
        assert all(c.char == " " for row in screen.next_frame for c in row)


@pytest.mark.unit
class TestDrawMeterRows:
    ROWS = [MeterRow("Database", "0.3/1 GB", 0.3, C), MeterRow("Open FDs", "59/1024", 0.06, C, note="RMS")]

    @staticmethod
    def draw(width):
        screen = TUIScreen(80, 6)
        Board.draw_meter_rows(screen, 1, 1, width, 2 + Tile.CARD_CHROME, TestDrawMeterRows.ROWS)
        return ["".join(c.char for c in screen.next_frame[r]).rstrip() for r in (Tile.CARD_TOP, Tile.CARD_TOP + 1)]

    def test_the_meter_takes_the_cells_the_label_and_value_leave(self):
        wide, wider = self.draw(40), self.draw(50)
        assert wider[0].index("0.3/1 GB") == wide[0].index("0.3/1 GB") + 10
        assert wide[0].index("0.3/1 GB") == wide[1].index("59/1024"), "one value column"
        assert wide[1].endswith("RMS")

    def test_too_narrow_for_a_meter_drops_it_and_keeps_label_and_value(self):
        lines = self.draw(26)
        assert not set("".join(lines)) & set(BLOCKS.strip())
        assert lines[0].split() == ["Database", "0.3/1", "GB"]

    def test_rows_the_tile_has_no_room_for_are_not_drawn(self):
        screen = TUIScreen(80, 6)
        Board.draw_meter_rows(screen, 1, 1, 40, 1 + Tile.CARD_CHROME, self.ROWS)
        text = "".join(c.char for row in screen.next_frame for c in row)
        assert "Database" in text and "Open FDs" not in text


@pytest.mark.unit
class TestFitPromptStats:
    STATS = [("rules", 900, 0.0), ("calendar", 0, 0.0), ("memory", 4000, 0.0), ("music", 0, 0.0), ("history", 120, 0.0)]

    @staticmethod
    def names(room):
        return [name for name, _, _ in DashboardTab._fit_prompt_stats(TestFitPromptStats.STATS, room)]

    def test_everything_shows_when_it_fits_empty_sections_included(self):
        assert self.names(5) == ["rules", "calendar", "memory", "music", "history"]

    def test_a_short_tile_drops_the_empty_sections_before_any_with_content(self):
        assert self.names(3) == ["rules", "memory", "history"]

    def test_a_tile_shorter_still_drops_the_smallest_and_keeps_the_order(self):
        assert self.names(2) == ["rules", "memory"]
        assert self.names(1) == ["memory"]

    def test_a_large_section_at_the_end_of_the_list_outlives_small_ones_before_it(self):
        stats = [("phones", 20, 0.0), ("weather", 1400, 0.0), ("static", 12000, 0.0)]
        assert [name for name, _, _ in DashboardTab._fit_prompt_stats(stats, 2)] == ["weather", "static"]

    def test_no_room_is_no_rows(self):
        assert self.names(0) == []


@pytest.mark.unit
class TestColumns:
    @pytest.mark.parametrize("width", [120, 181, 182, 183])
    @pytest.mark.parametrize("count", [3, 4])
    def test_tiles_span_the_width_with_even_gaps_and_the_last_takes_the_remainder(self, width, count):
        columns = DashboardTab._columns(TUIScreen(width, 10), count)
        assert len(columns) == count
        assert columns[0][0] == Board.LEFT
        for (x, w), (next_x, _) in zip(columns, columns[1:]):
            assert next_x - (x + w) == Board.GAP
        assert len({w for _, w in columns[:-1]}) == 1
        last_x, last_w = columns[-1]
        assert last_x + last_w == width - Board.RIGHT_MARGIN, "the last tile ends at the right margin"
        assert 0 <= last_w - columns[0][1] < count


@pytest.mark.unit
class TestIntegrationGrid:
    def test_a_narrow_tile_is_one_column(self):
        assert DashboardTab._integration_grid(7, DashboardTab.INTEGRATION_COLUMN_MIN - 1) == (1, 7)

    def test_columns_fill_top_to_bottom_so_rows_round_up(self):
        assert DashboardTab._integration_grid(19, 2 * DashboardTab.INTEGRATION_COLUMN_MIN) == (2, 10)
        assert DashboardTab._integration_grid(20, 2 * DashboardTab.INTEGRATION_COLUMN_MIN) == (2, 10)

    def test_a_very_wide_tile_stops_at_three_columns(self):
        assert DashboardTab._integration_grid(9, 10 * DashboardTab.INTEGRATION_COLUMN_MIN) == (3, 3)

    def test_no_integrations_is_no_rows(self):
        assert DashboardTab._integration_grid(0, 40)[1] == 0

    def test_a_short_tile_caps_the_rows_so_the_tail_of_the_list_is_clipped_not_every_column(self):
        two_columns = 2 * DashboardTab.INTEGRATION_COLUMN_MIN
        assert DashboardTab._integration_grid(20, two_columns, max_rows=4) == (2, 4)
        assert DashboardTab._integration_grid(5, two_columns, max_rows=4) == (2, 3), "a tall enough tile is not capped"
        assert DashboardTab._integration_grid(5, two_columns, max_rows=-1) == (2, 0)


@pytest.mark.unit
class TestIntegrationRows:
    """Every integration is a row of the grid: the enabled ones first, the disabled ones after them."""

    @staticmethod
    def status(name, state="ok"):
        return {"name": name, "state": state}

    def test_enabled_come_first_and_each_group_is_sorted_by_name(self):
        rows = DashboardTab._integration_rows([
            self.status("Ring", "disabled"), self.status("iMessage"), self.status("Arlo", "disabled"),
            self.status("Gmail", "error"),
        ])
        assert [s["name"] for s in rows] == ["Gmail", "iMessage", "Arlo", "Ring"]

    def test_disabled_integrations_are_not_dropped(self):
        rows = DashboardTab._integration_rows([self.status("Arlo", "disabled")])
        assert [s["name"] for s in rows] == ["Arlo"]

    def test_a_tile_too_short_for_the_list_clips_disabled_integrations_before_any_enabled_one(self, monkeypatch):
        from assistant.integration import IntegrationMonitor as monitor_module
        statuses = [self.status(f"On{i}") for i in range(5)] + [self.status(f"Off{i}", "disabled") for i in range(4)]
        monitor = type("Monitor", (), {"get_all_statuses": staticmethod(lambda: statuses)})()
        monkeypatch.setattr(monitor_module.IntegrationMonitor, "get_instance", classmethod(lambda cls: monitor))
        width = 2 * DashboardTab.INTEGRATION_COLUMN_MIN + 4      # two columns
        height = Tile.CARD_CHROME + 3                           # six cells for nine integrations
        screen = TUIScreen(width + 2, height + 2)
        tab = make_tab()
        tab._statuses = tab._integration_rows(monitor.get_all_statuses())   # what draw() does once per frame
        tab._draw_integrations(screen, 1, 1, width, height)
        text = "\n".join("".join(cell.char for cell in row) for row in screen.next_frame)
        assert all(f"On{i}" in text for i in range(5)), "every enabled integration is shown"
        assert "Off0" in text and "Off1" not in text, "the disabled ones fill the cells left and are clipped first"


@pytest.mark.unit
class TestFeedsTotalEdges:
    def test_a_feed_with_no_count_yet_adds_nothing_and_a_thousand_is_split(self):
        # A stopped feed that never stored a row reports stored=None: it counts as nothing,
        # and exactly 1000 is the first count the thousands split touches.
        screen = draw_feeds({"a": status(stored=1000), "b": status("stopped", stored=None)})
        assert row_text(screen, 2).endswith(" stopped")
        assert row_text(screen, 3).endswith(" 1 000 items")

    def test_a_paused_feeds_rows_still_count_in_the_total(self):
        screen = draw_feeds({"a": status(stored=1000), "b": status("catching_up", stored=500, total=900,
                                                                  remaining=400, progress=0.5, paused=True)})
        assert "paused" in row_text(screen, 2)
        assert row_text(screen, 3).endswith(" 1 500 items")


class TestCostSummary:
    """The Anthropic cost card's title strip (acc3490, ea11b79): the month total, then yesterday's
    cost per workspace, each workspace figure in its bar colour so the reader needs no legend.
    The total is no workspace's and must not wear a workspace colour: drawn in the first
    workspace's cyan it read as that workspace's figure. The colour is the middle of the bars'
    shading, not the accent: the accent shows only at the top of the scale, so the figures
    looked a different colour from the bars."""

    WIDTH = 70

    @staticmethod
    def tab():
        tab = make_tab()
        tab.cost_date_strs = ["2026-10-17", "2026-10-18", "2026-10-19"]
        tab.daily_costs = {"2026-10-17": {"ws-alpha": 9.0, "ws-beta": 9.0},
                           "2026-10-18": {"ws-alpha": 12.4, "ws-beta": 3.2},
                           "2026-10-19": {"ws-alpha": 50.0, "ws-beta": 60.0}}
        tab.cost_workspaces = ["ws-alpha", "ws-beta"]
        tab.total_cost_30d = 321.7
        return tab

    def strip(self, width=WIDTH):
        screen = TUIScreen(width, 3)
        self.tab()._draw_cost_summary(screen, 1, 1, width)
        row = screen.next_frame[0]
        return "".join(cell.char for cell in row), row

    def test_the_figures_are_the_month_total_and_yesterdays_cost_per_workspace(self):
        text, _ = self.strip()
        figures = [piece for piece in text.split() if piece.startswith("$")]
        assert figures == ["$322", "$12", "$3"], "rounded; yesterday's figures, not today's"

    def test_each_workspace_figure_wears_the_middle_shade_of_its_bars_and_the_total_wears_none(self):
        tab = self.tab()
        text, row = self.strip()
        total, alpha, beta = [row[m.start()].style.fg for m in re.finditer(r"\$\d+", text)]
        assert (alpha, beta) == (tab._workspace_shade(0, 0.5), tab._workspace_shade(1, 0.5))
        assert alpha not in (tab._workspace_shade(0, 0.0), tab._workspace_shade(0, 1.0))
        assert total not in (alpha, beta)

    def test_the_strip_ends_two_cells_before_the_cards_right_edge(self):
        text, _ = self.strip()
        assert len(text.rstrip()) == self.WIDTH - 2

    def test_a_card_too_narrow_for_the_title_and_the_strip_draws_nothing(self):
        text, _ = self.strip(width=len("Anthropic API Cost") + len("Month $322 · Day $12 $3") + 5)
        assert text.strip() == ""


class TestCostLegendNames:
    """The cost card's legend cuts the longest workspace names to fit every entry, rather than
    dropping the entries past the card's edge (the fifth workspace went missing on a 70-cell card)."""

    WORKSPACES = ["Default", "Alice Carlsson", "Bob", "Marcus-dev", "Marcus-prod"]

    @staticmethod
    def legend_width(names):
        return sum(len(n) for n in names) + DashboardTab.LEGEND_GAP * (len(names) - 1)

    def test_names_that_fit_are_left_whole(self):
        assert DashboardTab._legend_names(self.WORKSPACES, 200) == self.WORKSPACES

    def test_every_entry_fits_and_the_longest_names_are_cut_first(self):
        names = DashboardTab._legend_names(self.WORKSPACES, 50)
        assert self.legend_width(names) <= 50
        assert names[0] == "Default" and names[2] == "Bob", "short names keep their length"
        assert names[1].endswith("…") and names[1].startswith("Alice")

    def test_no_name_is_cut_below_the_minimum(self):
        names = DashboardTab._legend_names(self.WORKSPACES, 10)
        assert min(len(n) for n in names) == len("Bob")
        assert all(len(n) >= DashboardTab.LEGEND_MIN_NAME for n in names if n != "Bob")


@pytest.mark.unit
class TestChartPictures:
    def draw(self, tab, history, graphics=True):
        screen = TUIScreen(20, 10)
        screen.graphics = graphics
        tab._draw_columns(screen, CHART_X, CHART_Y, CHART_W, CHART_H, history, 100.0, lambda v: C,
                          picture="GPU Usage")
        return screen

    def test_on_a_graphics_terminal_the_columns_are_one_picture_over_the_plot(self):
        screen = self.draw(make_tab(), [50.0, 100.0])
        [(picture, placement)] = screen._placements.values()
        assert placement[:4] == (CHART_X, CHART_Y, CHART_W, CHART_H)
        assert all(column(screen, CHART_X + i) == " " * CHART_H for i in range(CHART_W))

    def test_a_chart_keeps_its_picture_from_frame_to_frame(self):
        tab = make_tab()
        first = next(iter(self.draw(tab, [50.0])._placements))
        assert next(iter(self.draw(tab, [50.0, 60.0])._placements)) == first

    def test_without_graphics_the_columns_stay_characters(self):
        screen = self.draw(make_tab(), [100.0], graphics=False)
        assert screen._placements == {}
        assert column(screen, CHART_X + CHART_W - 1) == "▄" * CHART_H

    def test_a_chart_with_nothing_above_zero_stays_characters(self):
        screen = self.draw(make_tab(), [0.0, 0.0])
        assert screen._placements == {}


@pytest.mark.unit
class TestCostBars:
    def test_a_day_stacks_its_workspaces_in_order_and_skips_the_idle_ones(self):
        tab = make_tab()
        costs = {"d1": {"a": 2.0, "b": 0.0, "c": 3.0}}
        [bar] = tab._cost_bars(["d1"], ["a", "b", "c"], costs, 10.0, bar_w=2, gap=1, chart_w=40)
        assert (bar.x0, bar.x1) == (0, 2)
        assert [(s.start, s.end) for s in bar.segments] == [(0.0, 0.2), (0.2, 0.5)]
        assert [s.color for s in bar.segments] == [DashboardTab.WORKSPACE_COLORS[0], DashboardTab.WORKSPACE_COLORS[2]]

    def test_days_without_cost_are_gaps_and_days_past_the_width_are_left_out(self):
        tab = make_tab()
        costs = {"d1": {"a": 1.0}, "d3": {"a": 1.0}, "d4": {"a": 1.0}}
        bars = tab._cost_bars(["d1", "d2", "d3", "d4"], ["a"], costs, 10.0, bar_w=1, gap=1, chart_w=5)
        assert [bar.x0 for bar in bars] == [0, 4]


@pytest.mark.unit
class TestLineChartFallback:
    """A line chart (`_draw_area_chart`) is a picture only where the terminal shows pictures;
    elsewhere a line as half-block pixels is a jagged staircase, so the plot is the block
    columns the chart was drawn as before, without the average's legend."""

    X, Y, WIDTH = 1, 1, 40    # screen coordinates are 1-based; the width fits the title and the legend
    PLOT_X, PLOT_Y, PLOT_W = X + 2 + DashboardTab.Y_AXIS_W, Y + Tile.CARD_TOP, WIDTH - 4 - DashboardTab.Y_AXIS_W

    def draw(self, values, graphics, **kwargs):
        screen = TUIScreen(60, 12)
        screen.graphics = graphics
        make_tab()._draw_area_chart(screen, self.X, self.Y, self.WIDTH, CHART_H, values, C, "Long Term Memory", "",
                                    window=3, window_label="3d avg", **kwargs)
        return screen

    def plot_cell(self, screen, i, row):
        return screen.next_frame[self.PLOT_Y - 1 + row][self.PLOT_X - 1 + i]

    def plot_column(self, screen, i):
        return "".join(self.plot_cell(screen, i, row).char for row in range(CHART_H))

    def title_strip(self, screen):
        return "".join(screen.next_frame[self.Y - 1][self.X - 1 + x].char for x in range(self.WIDTH))

    def test_on_a_graphics_terminal_the_line_is_a_picture_with_the_average_in_the_legend(self):
        screen = self.draw([10.0, 50.0, 100.0], graphics=True)
        [(_, placement)] = screen._placements.values()
        assert placement[:4] == (self.PLOT_X, self.PLOT_Y, self.PLOT_W, CHART_H)
        assert "3d avg" in self.title_strip(screen)

    def test_the_legend_carries_the_latest_average_in_the_window_format(self):
        screen = self.draw([10.0, 50.0, 90.0, None], graphics=True, window_format=lambda v: f"{v:.0f}KB")
        assert "3d avg (70KB)" in self.title_strip(screen), "the mean of the last three known values"

    def test_the_legend_figure_defaults_to_the_y_axis_format(self):
        screen = self.draw([10.0, 50.0, 90.0], graphics=True)
        assert "3d avg (50%)" in self.title_strip(screen)

    def test_without_graphics_the_values_are_block_columns_and_the_legend_is_left_out(self):
        screen = self.draw([10.0, None, 100.0], graphics=False)
        assert screen._placements == {}
        assert self.plot_column(screen, self.PLOT_W - 1) == "▄" * CHART_H, "the newest value, full scale, at the right"
        assert "▄" not in self.plot_column(screen, self.PLOT_W - 2), "a missing value is an empty column (bar the rules)"
        assert "3d avg" not in self.title_strip(screen)

    def test_the_fallback_columns_take_the_colour_the_chart_names_for_a_value(self):
        hot, cool = Color(255, 0, 0), Color(0, 255, 0)
        screen = self.draw([20.0, 90.0], graphics=False, color_for=lambda v: hot if v > 50 else cool)
        newest = self.plot_cell(screen, self.PLOT_W - 1, CHART_H - 1).style.fg
        older = self.plot_cell(screen, self.PLOT_W - 2, CHART_H - 1).style.fg
        assert newest.r > newest.g and older.g > older.r


# --- Utilization meters: the "Total memory" bar ------------------------------------------------

from types import SimpleNamespace


def resource_console(**values):
    fields = dict(photos_indexed=0, photos_total=0, voice_cache_size_gb=0.0, voice_cache_max_gb=1.0, db_size_gb=0.0,
                  system_total_memory_gb=0.0, system_used_memory_gb=0.0, system_available_memory_gb=0.0,
                  process_memory_gb=0.0, process_cpu_percent=0.0, system_cpu_percent=0.0, fd_count=0,
                  target_server_volume=0, current_noise_db=0.0, outpost_latencies={})
    return SimpleNamespace(**{**fields, **values})


def total_memory_row(**memory):
    tab = make_tab()
    tab.console = resource_console(**memory)
    with patch("interfaces.console.tabs.DashboardTab.STTConfidenceMonitor.get_instance") as stt, \
            patch("interfaces.console.tabs.DashboardTab.PrefillTracker.current_progress", return_value=None):
        stt.return_value.mics = {}
        rows = tab._resource_rows()
    return next(row for row in rows if row.label == "Total memory")


@pytest.mark.unit
class TestTotalMemoryRow:
    """The bar warns on an absolute floor of available memory, not on the share used: on a
    large machine most of the RAM is in use as a matter of course, with tens of GB still free."""

    def test_a_large_machine_mostly_in_use_with_gigabytes_free_is_not_a_warning(self):
        row = total_memory_row(system_total_memory_gb=256.0, system_used_memory_gb=205.0, system_available_memory_gb=50.0)
        assert row.color != Theme.COLOR_RED
        assert row.fraction == pytest.approx(205 / 256), "the bar's length is still the share used"

    def test_under_the_floor_warns_whatever_the_share_used(self):
        # macOS counts the reclaimable cache as neither used nor free: 10 of 16 GB used can leave 1.5 GB available
        row = total_memory_row(system_total_memory_gb=16.0, system_used_memory_gb=10.0, system_available_memory_gb=1.5)
        assert row.color == Theme.COLOR_RED

    def test_before_the_first_sample_the_bar_is_not_red(self):
        row = total_memory_row()
        assert row.color != Theme.COLOR_RED and row.fraction == 0


# --- The live KV-cache chart -------------------------------------------------------------------

from common.TTFTMonitor import TTFTMonitor

SMART, FAST, CURATOR = "claude-opus-5-5", "claude-haiku-5-5", "deepseek-v4"
CACHE_X, CACHE_Y, CACHE_W, CACHE_H = 1, 1, 60, 4   # wide enough for a three-part legend


def draw_cache_chart(samples):
    monitor = TTFTMonitor()
    monitor.smart_model, monitor.fast_model = SMART, FAST
    monitor.smart_model_short, monitor.fast_model_short = "opus", "haiku"
    for model, pct in samples:
        monitor.record_cache_pct(model, pct)
    screen = TUIScreen(CACHE_W + 4, 12)
    screen.clear_buffer()
    make_tab()._draw_cache_chart(screen, CACHE_X, CACHE_Y, CACHE_W, CACHE_H, monitor)
    return screen


def title_strip(screen) -> str:
    return "".join(c.char for c in screen.next_frame[CACHE_Y - 1]).rstrip()


def bottom_row(screen):
    chart_x = CACHE_X + 2 + DashboardTab.Y_AXIS_W
    chart_w = CACHE_W - 4 - DashboardTab.Y_AXIS_W
    bottom = CACHE_Y + Tile.CARD_TOP + CACHE_H - 1
    return [screen.next_frame[bottom - 1][chart_x - 1 + i] for i in range(chart_w)]


@pytest.mark.unit
class TestCacheChart:
    def test_the_legend_names_the_tiers_and_adds_other_only_when_another_model_served(self):
        tiers_only = title_strip(draw_cache_chart([(SMART, 80), (FAST, 60)]))
        assert "opus" in tiers_only and "haiku" in tiers_only and "other" not in tiers_only
        assert tiers_only.endswith("60%"), "the value is the latest request's hit"
        with_curator = title_strip(draw_cache_chart([(SMART, 80), (CURATOR, 90)]))
        assert "other" in with_curator and with_curator.endswith("90%")

    def test_a_request_of_a_model_outside_the_tiers_is_charted_in_its_own_colour(self):
        # unlike the TTFT chart beside it, which shows the two tiers alone
        def drawn(samples):
            return [cell for cell in bottom_row(draw_cache_chart(samples)) if cell.char == "▄"]

        assert len(drawn([(SMART, 50), (CURATOR, 50)])) == 2
        smart_colour = drawn([(SMART, 50)])[-1].style.fg
        curator_colour = drawn([(CURATOR, 50)])[-1].style.fg
        assert smart_colour != curator_colour
        both = drawn([(SMART, 50), (CURATOR, 50)])
        assert [cell.style.fg for cell in both] == [smart_colour, curator_colour]
