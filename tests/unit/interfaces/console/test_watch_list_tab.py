"""WatchListTab: every open item gets a state — what it needs and when its next check falls —
and the board lists them most urgent first."""
from datetime import datetime, timedelta

import pytest

from assistant import WatchList
from common.vault.Redactor import Redactor
from common.vault.Vault import Vault
from interfaces.console import Theme
from interfaces.console.framework.TUIScreen import TUIScreen
from interfaces.console.tabs import WatchListTab as tab_module
from interfaces.console.tabs.TabView import SUB_CONTENT_TOP
from interfaces.console.tabs.WatchListTab import (CHECK_DUE, DUE_SOON_DAYS, ON_TRACK, OPEN, OVERDUE, PAST_DUE, POLLED,
                                                  STALE, WAITING, WatchListTab, assess, span)

NOW = datetime(2026, 3, 10, 12, 0)


def item(name="parcel", every="", checked: timedelta = None, due_in: int = None, updated_days_ago=0):
    day = lambda offset: (NOW + timedelta(days=offset)).date().isoformat()
    return WatchList.Item(name, "order 4711", "" if due_in is None else day(due_in), day(-30), day(-updated_days_ago),
                          every, (NOW - checked).strftime("%Y-%m-%d %H:%M") if checked is not None else "")


@pytest.mark.unit
class TestAssess:
    @pytest.mark.parametrize("entry,kind", [
        (item(every="6h", checked=timedelta(hours=2)), ON_TRACK),
        (item(every="6h", checked=timedelta(hours=7)), CHECK_DUE),
        (item(every="6h", checked=timedelta(hours=10)), OVERDUE),
        (item(every="6h"), CHECK_DUE),
        (item(every="6h", checked=timedelta(hours=1), due_in=-1), PAST_DUE),
        (item(every="weekly", checked=timedelta(days=1), updated_days_ago=WatchList.STALE_AFTER_DAYS), STALE),
        (item(every="when it rains", checked=timedelta(hours=1)), POLLED),
        (item(due_in=5), WAITING),
        (item(due_in=0), WAITING),
        (item(), OPEN),
    ])
    def test_an_item_gets_the_state_its_cadence_and_dates_put_it_in(self, entry, kind):
        assert assess(entry, NOW).kind is kind

    def test_the_next_check_is_one_cadence_after_the_last(self):
        entry = assess(item(every="6h", checked=timedelta(hours=2)), NOW)
        assert entry.next_check == NOW + timedelta(hours=4)
        assert entry.fraction == pytest.approx(1 / 3)

    def test_an_item_without_a_cadence_has_no_next_check(self):
        entry = assess(item(due_in=5, checked=timedelta(hours=2)), NOW)
        assert entry.next_check is None and entry.fraction is None and entry.due_days == 5

    def test_the_most_urgent_sorts_first(self):
        entries = [assess(i, NOW) for i in (
            item("on-track-later", every="1d", checked=timedelta(hours=1)),
            item("waiting", due_in=3),
            item("on-track-sooner", every="1d", checked=timedelta(hours=20)),
            item("late", every="6h", checked=timedelta(hours=20)),
            item("past", due_in=-2),
        )]
        assert [e.item.name for e in sorted(entries, key=lambda e: e.urgency)] == \
            ["past", "late", "on-track-sooner", "on-track-later", "waiting"]


@pytest.mark.unit
class TestSpan:
    @pytest.mark.parametrize("delta,text", [
        (timedelta(seconds=20), "<1m"), (timedelta(minutes=42), "42m"), (timedelta(hours=3), "3h"),
        (timedelta(hours=3, minutes=50), "3h 50m"), (timedelta(days=2, hours=4), "2d 4h"),
        (timedelta(days=12, hours=5), "12d"), (-timedelta(minutes=5), "5m"),
    ])
    def test_a_span_reads_in_its_two_largest_units(self, delta, text):
        assert span(delta) == text


WIDTH, HEIGHT = 120, 40


@pytest.mark.unit
class TestBoard:
    @pytest.fixture(autouse=True)
    def store(self, tmp_path, monkeypatch):
        monkeypatch.setattr(WatchList, "WATCHLIST_DIR", tmp_path / "watchlist")
        monkeypatch.setattr(WatchList, "LOG_DIR", tmp_path / "log")
        monkeypatch.setattr(WatchListTab, "now", staticmethod(lambda: NOW))
        WatchList.WATCHLIST_DIR.mkdir()
        for entry in (item("sofa-delivery", every="1d", checked=timedelta(hours=1)),
                      item("festival-tickets", every="6h", checked=timedelta(hours=20))):
            (WatchList.WATCHLIST_DIR / f"{entry.name}.md").write_text(entry.text().replace("order 4711", f"note of {entry.name}"))
        self.tab = WatchListTab(Redactor(Vault(tmp_path / "vault.bin", "test")))

    def rows(self, width=WIDTH) -> list[str]:
        screen = TUIScreen(width, HEIGHT)
        self.tab.draw(screen, 0)
        return [screen.row_text(1, y, width).rstrip() for y in range(SUB_CONTENT_TOP, HEIGHT)]

    def test_the_late_item_is_listed_above_the_one_on_schedule(self):
        rows = self.rows()
        rows = rows[next(i for i, row in enumerate(rows) if "Every" in row):]
        late = next(i for i, row in enumerate(rows) if "festival-tickets" in row and OVERDUE.label in row)
        on_track = next(i for i, row in enumerate(rows) if "sofa-delivery" in row and ON_TRACK.label in row)
        assert late < on_track
        assert "14h late" in rows[late] and "in 23h" in rows[on_track]

    def test_the_most_urgent_item_is_selected_and_down_moves_to_the_next(self):
        assert "note of festival-tickets" in "\n".join(self.rows())
        assert self.tab.handle_input("down")
        text = "\n".join(self.rows())
        assert "note of sofa-delivery" in text and "note of festival-tickets" not in text

    def test_the_selection_stops_at_the_ends(self):
        self.rows()
        for _ in range(5):
            self.tab.handle_input("down")
        assert "note of sofa-delivery" in "\n".join(self.rows())
        for _ in range(5):
            self.tab.handle_input("up")
        assert "note of festival-tickets" in "\n".join(self.rows())

    def test_a_wide_screen_puts_the_selected_item_beside_the_table(self):
        rows = self.rows(WatchListTab.SIDE_BY_SIDE_MIN + 40)
        reason = assess(item("festival-tickets", every="6h", checked=timedelta(hours=20)), NOW).reason
        assert any("Every" in row and reason in row for row in rows), "the table's titles and the item's state share a row"
        assert not any("Every" in row and reason in row for row in self.rows())

    def test_a_deleted_selection_falls_back_to_the_first_item(self):
        self.rows()
        WatchList.delete("festival-tickets")
        assert "note of sofa-delivery" in "\n".join(self.rows())

    @pytest.mark.parametrize("width,height", [(60, 12), (30, 8), (250, 14)])
    def test_a_cramped_screen_still_draws(self, width, height):
        self.tab.draw(TUIScreen(width, height), 0)

    def test_the_longest_lateness_is_shown_whole(self):
        late = item("router-firmware", every="1d", checked=timedelta(days=1, hours=23, minutes=59))
        (WatchList.WATCHLIST_DIR / "router-firmware.md").write_text(late.text())
        for width in (WIDTH, WatchListTab.SIDE_BY_SIDE_MIN + 40):
            assert any("router-firmware" in row and "23h 59m late" in row for row in self.rows(width))


@pytest.mark.unit
class TestDueColumn:
    """The Due cell says how far off the date is and warns from DUE_SOON_DAYS out."""

    def cell(self, due_in: int):
        return WatchListTab._cell("due", assess(item(due_in=due_in), NOW), NOW)

    def test_a_date_within_the_warning_window_is_yellow_and_one_beyond_it_is_not(self):
        soon, text = self.cell(DUE_SOON_DAYS)[1], self.cell(DUE_SOON_DAYS)[0]
        assert soon == Theme.COLOR_YELLOW and f"in {DUE_SOON_DAYS}d" in text
        assert self.cell(DUE_SOON_DAYS + 1)[1] != Theme.COLOR_YELLOW

    def test_a_passed_date_is_red_and_says_how_long_ago(self):
        text, color = self.cell(-3)
        assert color == Theme.COLOR_RED and "3d ago" in text

    def test_today_and_tomorrow_are_named_not_counted(self):
        assert "today" in self.cell(0)[0] and "tomorrow" in self.cell(1)[0]

    def test_a_date_the_tab_cannot_read_is_shown_as_written(self):
        entry = assess(WatchList.Item("parcel", "order 4711", due="when the shop reopens"), NOW)
        assert WatchListTab._cell("due", entry, NOW)[0] == "when the shop reopens"


@pytest.mark.unit
class TestTableLayout:
    """Columns leave in DROP_ORDER until the name has its minimum; room the longest name does
    not need goes to the meter."""

    def test_the_columns_leave_in_their_order_until_the_name_has_room(self):
        entries = [assess(item("sofa-delivery", every="1d", checked=timedelta(hours=1)), NOW)]
        full = sum(width + 2 for _, _, width in WatchListTab.COLUMNS) + 2 + WatchListTab.NAME_MIN_W
        keys = lambda inner_w: [key for key, _, _ in WatchListTab._layout(entries, inner_w)[2]]
        assert keys(full) == [key for key, _, _ in WatchListTab.COLUMNS]
        assert keys(full - 1) == [key for key, _, _ in WatchListTab.COLUMNS if key != "every"]
        narrow = keys(full - 30)
        assert "next" in narrow and "due" in narrow and "every" not in narrow and "last" not in narrow

    def test_room_the_longest_name_leaves_over_lengthens_the_meter(self):
        entries = [assess(item("sofa", every="1d", checked=timedelta(hours=1)), NOW)]
        name_w, meter_w, columns = WatchListTab._layout(entries, 400)
        assert meter_w == WatchListTab.METER_W + WatchListTab.METER_GROW
        assert name_w == WatchListTab._layout(entries, 400 + WatchListTab.METER_GROW)[0] - WatchListTab.METER_GROW, \
            "the meter grows by what the name gives up, and no further"
        next_w = next(width for key, _, width in columns if key == "next")
        assert next_w - meter_w == next(width for key, _, width in WatchListTab.COLUMNS if key == "next") - WatchListTab.METER_W


@pytest.mark.unit
class TestSummaryTiles:
    """The four tiles over the table: what needs attention, the soonest check, the latest one
    and how many are watched, each read from the entries."""

    LABELS = ("Needs attention", "Next check", "Last check", "Watching")
    WIDE = 200

    @pytest.fixture(autouse=True)
    def store(self, tmp_path, monkeypatch):
        monkeypatch.setattr(WatchList, "WATCHLIST_DIR", tmp_path / "watchlist")
        monkeypatch.setattr(WatchList, "LOG_DIR", tmp_path / "log")
        monkeypatch.setattr(WatchListTab, "now", staticmethod(lambda: NOW))
        WatchList.WATCHLIST_DIR.mkdir()
        self.tab = WatchListTab(Redactor(Vault(tmp_path / "vault.bin", "test")))

    def add(self, *entries: WatchList.Item):
        for entry in entries:
            (WatchList.WATCHLIST_DIR / f"{entry.name}.md").write_text(entry.text())

    def tile(self, label: str) -> str:
        """The strip and the row under it of the tile called `label`, as they read."""
        screen = TUIScreen(self.WIDE, HEIGHT)
        self.tab.draw(screen, 0)
        strip_y = next(y for y in range(SUB_CONTENT_TOP, HEIGHT) if label in screen.row_text(1, y, self.WIDE))
        strip = screen.row_text(1, strip_y, self.WIDE)
        starts = sorted(strip.index(name) for name in self.LABELS)
        begin = strip.index(label)
        end = next((start for start in starts if start > begin), self.WIDE)
        return " | ".join(screen.row_text(begin, y, end - begin).strip() for y in (strip_y, strip_y + 2))

    def test_the_attention_tile_counts_the_items_that_need_a_look_by_kind(self):
        self.add(item("sofa-delivery", every="1d", checked=timedelta(hours=1)),
                 item("festival-tickets", every="6h", checked=timedelta(hours=20)),
                 item("router-firmware", every="1d"),
                 item("passport", due_in=-1))
        tile = self.tile("Needs attention")
        assert tile.split(" | ")[0].split()[-1] == "3"
        assert "1 past due" in tile and "1 overdue" in tile and "1 check due" in tile
        assert "on track" not in tile

    def test_with_nothing_due_the_attention_tile_is_all_clear(self):
        self.add(item("sofa-delivery", every="1d", checked=timedelta(hours=1)), item("passport", due_in=10))
        assert "all clear" in self.tile("Needs attention")

    def test_the_next_check_tile_names_the_one_that_falls_first(self):
        self.add(item("sofa-delivery", every="1d", checked=timedelta(hours=1)),
                 item("festival-tickets", every="6h", checked=timedelta(hours=20)))
        tile = self.tile("Next check")
        assert "festival-tickets" in tile and "14h late" in tile
        self.add(item("router-firmware", every="1d"))
        tile = self.tile("Next check")
        assert "router-firmware" in tile and "now" in tile, "a check never made comes before every late one"

    def test_the_next_check_tile_counts_down_to_a_check_still_ahead(self):
        self.add(item("sofa-delivery", every="1d", checked=timedelta(hours=1)), item("passport", due_in=10))
        tile = self.tile("Next check")
        assert "sofa-delivery" in tile and "in 23h" in tile

    def test_the_last_check_tile_names_the_latest_check_made(self):
        self.add(item("sofa-delivery", every="1d", checked=timedelta(hours=1)),
                 item("festival-tickets", every="6h", checked=timedelta(hours=20)),
                 item("passport", due_in=10))
        tile = self.tile("Last check")
        assert "sofa-delivery" in tile and "1h ago" in tile and "festival-tickets" not in tile

    def test_the_watching_tile_sorts_the_items_into_polled_dated_and_open(self):
        self.add(item("sofa-delivery", every="1d", checked=timedelta(hours=1)),
                 item("festival-tickets", every="when the shop opens"),
                 item("passport", due_in=10),
                 item("garden-plans"))
        tile = self.tile("Watching")
        assert "4 items" in tile
        assert "2 polled" in tile and "1 waiting for a date" in tile and "1 open" in tile

    def test_an_empty_list_has_nothing_to_sum_up(self):
        assert "nothing is polled" in self.tile("Next check")
        assert "nothing checked yet" in self.tile("Last check")
        assert "0 items" in self.tile("Watching")


@pytest.mark.unit
class TestNoteScrolling:
    LINES = 60

    @pytest.fixture(autouse=True)
    def store(self, tmp_path, monkeypatch):
        monkeypatch.setattr(WatchList, "WATCHLIST_DIR", tmp_path / "watchlist")
        monkeypatch.setattr(WatchList, "LOG_DIR", tmp_path / "log")
        monkeypatch.setattr(WatchListTab, "now", staticmethod(lambda: NOW))
        WatchList.WATCHLIST_DIR.mkdir()
        long = item("festival-tickets", every="6h", checked=timedelta(hours=20))
        note = "\n".join(f"note line {n}" for n in range(1, self.LINES + 1))
        (WatchList.WATCHLIST_DIR / "festival-tickets.md").write_text(long.text().replace("order 4711", note))
        (WatchList.WATCHLIST_DIR / "sofa-delivery.md").write_text(
            item("sofa-delivery", every="1d", checked=timedelta(hours=1)).text())
        self.tab = WatchListTab(Redactor(Vault(tmp_path / "vault.bin", "test")))

    def shown(self) -> list[int]:
        screen = TUIScreen(WIDTH, HEIGHT)
        self.tab.draw(screen, 0)
        return [int(row.split("note line ")[1].split()[0])
                for row in (screen.row_text(1, y, WIDTH) for y in range(SUB_CONTENT_TOP, HEIGHT)) if "note line " in row]

    def test_page_down_moves_the_note_a_screenful_and_the_end_keeps_the_last_line_in_view(self):
        first_page = self.shown()
        assert first_page[0] == 1 and len(first_page) < self.LINES
        assert self.tab.handle_input("pagedown")
        second_page = self.shown()
        assert second_page[0] == len(first_page) + 1
        for _ in range(10):
            self.tab.handle_input("pagedown")
        last_page = self.shown()
        assert last_page[-1] == self.LINES and len(last_page) == len(first_page)
        self.tab.handle_input("pageup")
        assert self.shown()[-1] < self.LINES

    def test_a_scrolled_note_is_left_behind_when_another_item_is_selected(self):
        self.shown()
        self.tab.handle_input("pagedown")
        assert self.shown()[0] > 1
        self.tab.handle_input("down")
        self.tab.handle_input("up")
        assert self.shown()[0] == 1
