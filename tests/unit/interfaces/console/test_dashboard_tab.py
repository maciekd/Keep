"""DashboardTab._draw_columns: the one chart core every dashboard chart goes through."""
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
        assert (row.symbol, row.progress, row.text) == (SYMBOL_RUNNING, 1.0, "23436 messages")
        assert row.fg == Theme.COLOR_GREEN
        assert DashboardTab._feed_row("Indigo", status(stored=512, items="events")).text == "512 events", \
            "an uncountable feed shows its rows"

    def test_a_catch_up_whose_counts_meet_shows_the_count_once(self):
        assert DashboardTab._feed_row("Photos", status("catching_up", stored=10, total=10, remaining=0)).text == "10 items"

    def test_catching_up_with_a_count_shows_done_over_total_without_separators(self):
        row = DashboardTab._feed_row("iCloud Photos", status("catching_up", progress=0.34, stored=1234, total=23356, remaining=22122))
        assert (row.symbol, row.name, row.progress, row.text) == (DashboardTab.SYMBOL_CATCHING_UP, "Photos", 0.34, "1234 / 23356 items")
        assert row.fg == Theme.COLOR_YELLOW

    def test_catching_up_with_an_eta_appends_it_in_one_rounded_unit(self):
        def text(eta):
            return DashboardTab._feed_row("Photos", status("catching_up", stored=1234, total=22313, remaining=21079, eta_seconds=eta)).text
        assert text(13 * 3600) == "1234 / 22313 items ETA: 13h"
        assert text(13 * 60 + 20) == "1234 / 22313 items ETA: 13m"
        assert text(45) == "1234 / 22313 items ETA: 45s"
        assert text(3 * 86400) == "1234 / 22313 items ETA: 3d"
        assert text(47 * 3600) == "1234 / 22313 items ETA: 47h", "days only from two days up"
        assert text(None) == "1234 / 22313 items"

    def test_catching_up_without_a_count_is_indeterminate_and_shows_its_rows(self):
        row = DashboardTab._feed_row("Indigo", status("catching_up", stored=1345))
        assert (row.progress, row.text) == (None, "1345 items")

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
        assert row_text(screen, 1).endswith(f"{FULL} 23436 items")
        assert row_text(screen, 2).startswith(f"  Photos   {DashboardTab.SYMBOL_CATCHING_UP} {FULL}")
        assert "1234 / 23356" in row_text(screen, 2)
        assert row_text(screen, 3).startswith(f"  Telegram {SYMBOL_FAILED}")
        assert row_text(screen, 3).endswith(" stopped") and FULL not in row_text(screen, 3)

    def test_the_bar_takes_every_cell_the_longest_text_leaves(self):
        screen = draw_feeds({"a": status(stored=5), "b": status(stored=1234567)})
        line = row_text(screen, 2)
        assert line.endswith(f"{FULL} 1234567 items")
        assert len(line) == PANEL_X + PANEL_W - 3, "the longest text ends one cell of air before the tile's edge"
        assert row_text(screen, 1).rindex(FULL) == line.rindex(FULL), "one text column for every row"

    def test_the_count_is_drawn_in_the_row_colour_not_grey(self):
        screen = draw_feeds({"iMessage": status()})
        line = row_text(screen, 1)
        cell = panel_row(screen, 1)[line.index("23436")]
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

    def test_more_feeds_than_rows_are_clipped_to_the_box(self):
        screen = draw_feeds({f"f{i}": status() for i in range(6)})
        assert row_text(screen, 4).startswith(f"  f3 {SYMBOL_RUNNING}")
        assert row_text(screen, 5) == "", "the tile's closing row stays empty"


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


@pytest.mark.unit
class TestIntegrationRows:
    """Disabled integrations leave the bullet list and share wrapped "Disabled: …" lines instead."""

    @staticmethod
    def status(name, state="ok"):
        return {"name": name, "state": state}

    def test_enabled_are_sorted_and_disabled_are_summarised(self):
        enabled, lines = DashboardTab._integration_rows([
            self.status("Ring", "disabled"), self.status("iMessage"), self.status("Arlo", "disabled"),
            self.status("Gmail"),
        ])
        assert [s["name"] for s in enabled] == ["Gmail", "iMessage"]
        assert lines == ["Disabled: Arlo, Ring"]

    def test_no_disabled_means_no_summary_lines(self):
        enabled, lines = DashboardTab._integration_rows([self.status("Gmail")])
        assert [s["name"] for s in enabled] == ["Gmail"]
        assert lines == []

    def test_a_long_disabled_list_wraps_inside_the_column(self):
        names = ["Arlo", "CalDAV", "Gmail", "Home Assistant", "Indigo", "Photos", "Reminders",
                 "Ring", "Self Improvement", "Sonos & Spotify", "Telegram", "Tesla", "X", "YR Weather"]
        _, lines = DashboardTab._integration_rows([self.status(n, "disabled") for n in names])
        assert len(lines) > 1
        assert all(len(line) <= DashboardTab.INTEGRATIONS_WIDTH for line in lines)
        assert lines[0].startswith("Disabled: ")
        assert all(not line.startswith(" ") for line in lines[1:])   # continuation lines sit flush left
        assert " ".join(lines) == "Disabled: " + ", ".join(names)
