import dataclasses
from datetime import datetime, timedelta
from typing import Dict, List, NamedTuple, Optional, Tuple

from .MemoryTab import MemoryTab
from .ScrollableTabView import draw_scrollbar
from .TabView import SUB_CONTENT_TOP, TabView
from assistant import FirstRun, WatchList
from common.logging import trace
from common.vault.Redactor import Redactor
from common.vault.Vault import Vault
from interfaces.console import Theme
from interfaces.console.framework.TUIScreen import TUIScreen, Color
from interfaces.console.framework.TextWidth import display_width
from interfaces.console.widgets import Board, Markdown, Tile
from interfaces.console.widgets.Board import Fact, Meter, SummaryTile, Table
from interfaces.console.widgets.SecretMarks import mark_masks

# A check this many cadences after the last one is no longer merely due: it was missed.
LATE_FACTOR = 1.5
# A due date this near is drawn as a warning.
DUE_SOON_DAYS = 2


class Kind(NamedTuple):
    """What an item needs, most urgent first by `rank`."""
    label: str
    color: Color
    rank: int
    symbol: str = "●"


PAST_DUE = Kind("past due", Theme.COLOR_RED, 0)
OVERDUE = Kind("overdue", Theme.COLOR_RED, 1)
CHECK_DUE = Kind("check due", Theme.COLOR_YELLOW, 2)
STALE = Kind("stale", Theme.COLOR_ORANGE, 3)
ON_TRACK = Kind("on track", Theme.COLOR_GREEN, 4)
POLLED = Kind("polled", Theme.COLOR_TEAL, 5)        # has a cadence the tab cannot read
WAITING = Kind("waiting", Theme.COLOR_BLUE, 6, "○")  # no cadence: it only waits for its due date
OPEN = Kind("open", Theme.COLOR_DIM, 7, "○")         # neither a cadence nor a due date


class Entry(NamedTuple):
    """One item as the tab shows it at a given moment."""
    item: WatchList.Item
    kind: Kind
    reason: str
    checked: Optional[datetime]
    next_check: Optional[datetime]   # None without a readable cadence, and for an item never checked
    fraction: Optional[float]        # the share of the cadence gone since the last check; None when not polled
    due_days: Optional[int]          # days until the due date, negative once it has passed

    @property
    def check_color(self) -> Color:
        if self.fraction is None:
            return Theme.COLOR_DIM
        if self.fraction >= LATE_FACTOR:
            return Theme.COLOR_RED
        return Theme.COLOR_YELLOW if self.fraction >= 1 else Theme.COLOR_GREEN

    @property
    def urgency(self) -> tuple:
        if self.kind in (PAST_DUE, WAITING):
            order = self.due_days
        else:
            order = self.next_check.timestamp() if self.next_check else 0
        return self.kind.rank, order, self.item.name


def span(delta: timedelta) -> str:
    """A length of time in the two units that matter: "42m", "3h 50m", "2d 4h", "12d"."""
    minutes = int(abs(delta).total_seconds() // 60)
    if minutes < 1:
        return "<1m"
    if minutes < 60:
        return f"{minutes}m"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h {minutes}m" if minutes else f"{hours}h"
    days, hours = divmod(hours, 24)
    return f"{days}d {hours}h" if days < 7 and hours else f"{days}d"


def days(count: int) -> str:
    return "1 day" if count == 1 else f"{count} days"


def when(moment: datetime, now: datetime) -> str:
    """A moment as a reader places it: "today 18:30", "tomorrow 07:00", "Sat Nov 8 12:20"."""
    offset = (moment.date() - now.date()).days
    day = {0: "today", 1: "tomorrow", -1: "yesterday"}.get(offset) or moment.strftime("%a %b %-d")
    return f"{day} {moment:%H:%M}"


def day_distance(offset: int) -> str:
    """A day `offset` days from today: "today", "in 3 days", "2 days ago"."""
    if offset == 0:
        return "today"
    return f"in {days(offset)}" if offset > 0 else f"{days(-offset)} ago"


def assess(item: WatchList.Item, now: datetime) -> Entry:
    """Where `item` stands at `now`: what it needs, why, and when its next check falls."""
    every = WatchList.interval(item.every)
    checked = WatchList.stamp(item.checked)
    due = WatchList.stamp(item.due)
    due_days = (due.date() - now.date()).days if due else None
    next_check = checked + every if every and checked else None
    fraction = None
    if every:
        fraction = (now - checked) / every if checked else 1.0
    stale = WatchList.staleness(item, now.date())

    if due_days is not None and due_days < 0:
        kind, reason = PAST_DUE, f"due date passed {days(-due_days)} ago"
    elif every and checked is None:
        kind, reason = CHECK_DUE, "never checked"
    elif every and fraction >= LATE_FACTOR:
        kind, reason = OVERDUE, f"check is {span(now - next_check)} late"
    elif every and fraction >= 1:
        kind, reason = CHECK_DUE, "check due now"
    elif stale:
        kind, reason = STALE, stale.removeprefix("STALE: ")
    elif every:
        kind, reason = ON_TRACK, f"next check in {span(next_check - now)}"
    elif item.every:
        kind, reason = POLLED, f"cadence \"{item.every}\" not understood"
    elif due_days is not None:
        kind, reason = WAITING, f"waits for its due date, {day_distance(due_days)}"
    else:
        kind, reason = OPEN, "no cadence, no due date"
    return Entry(item, kind, reason, checked, next_check, fraction, due_days)


class WatchListTab(TabView):
    """The open watch items as a board in the Dashboard's tiles: four summary tiles (what needs
    attention, the next check, the last one, how many are watched), a table sorted most urgent
    first — a state dot, a meter filling towards the next check, the last check, the cadence
    and the due date — and the selected item in full beside it (under it on a narrow screen),
    its note laid out by `widgets/Markdown`.
    Red is a passed due date or a missed check, yellow a check that is due, orange an item
    nobody has updated for long, green one on schedule, blue one that only waits for its date.
    Vaulted values are masked as in the other Memory tabs."""

    STACKED_DETAIL_MIN_H = Tile.CARD_CHROME + 5
    NAME_MIN_W = 16
    NAME_MAX_W = 40
    METER_W = 10
    METER_GROW = 20
    NEXT_TEXT_W = 12            # "23h 59m late", the longest a span gets
    # The table's columns after the name, and the order they go in on a narrow tile.
    COLUMNS = (("status", "Status", 10), ("next", "Next check", METER_W + 1 + NEXT_TEXT_W), ("last", "Last check", 11),
               ("every", "Every", 8), ("due", "Due", 17))
    DROP_ORDER = ("every", "last", "status", "due")

    def __init__(self, redactor: Optional[Redactor] = None):
        self._redactor = redactor  # over the Vault singleton, resolved on first draw
        self._cards: List[Tile.Card] = []
        self._items: List[WatchList.Item] = []
        self._loaded_version = None
        self._notes: Dict[Tuple[str, int], Markdown.Document] = {}   # (item, width) -> its note laid out
        self._order: List[str] = []
        self._selected: Optional[str] = None
        self._top = 0
        self._note_scroll = 0
        self._note_room = 1

    @property
    def name(self) -> str:
        return "Watch list"

    @property
    def redactor(self) -> Redactor:
        if self._redactor is None:
            self._redactor = Redactor(Vault.get_instance())
        return self._redactor

    @staticmethod
    def now() -> datetime:
        return datetime.now()

    def handle_input(self, key: str) -> bool:
        if key in ('up', 'down') and self._order:
            index = self._order.index(self._selected) if self._selected in self._order else 0
            index = max(0, min(len(self._order) - 1, index + (1 if key == 'down' else -1)))
            if self._order[index] != self._selected:
                self._selected = self._order[index]
                self._note_scroll = 0
            return True
        if key in ('pageup', 'pagedown'):
            self._note_scroll = max(0, self._note_scroll + (self._note_room if key == 'pagedown' else -self._note_room))
            return True
        return False

    def _load(self) -> List[WatchList.Item]:
        """The open items with every vaulted value masked, re-read when a file or the vault changes."""
        version = (WatchList.version(), self.redactor.vault.version)
        if version != self._loaded_version:
            mask = self.redactor.mask
            self._items = [dataclasses.replace(item, name=mask(item.name), note=mask(item.note))
                           for item in WatchList.items()]
            self._notes = {}
            self._loaded_version = version
            trace(f"Watch list tab: {len(self._items)} open item(s) loaded")
        return self._items

    def draw(self, screen: TUIScreen, frame: int):
        self._cards = []
        now = self.now()
        entries = sorted((assess(item, now) for item in self._load()), key=lambda entry: entry.urgency)
        self._order = [entry.item.name for entry in entries]

        y = SUB_CONTENT_TOP
        if FirstRun.curation_progress() is not None:
            MemoryTab._draw_first_curation_notice(screen, y)
            y += MemoryTab.NOTICE_ROWS + 1
        available = screen.width - Board.LEFT - Board.RIGHT_MARGIN
        Board.draw_summary(screen, self._cards, y, available,
                           [self._attention_tile(entries), self._next_check_tile(entries, now),
                            self._last_check_tile(entries, now), self._watching_tile(entries)])
        y += Board.SUMMARY_H + 1

        height = screen.height - 1 - y      # the last row is the key hints'
        if not entries:
            Board.draw_notice(screen, self._cards, Board.LEFT, y, available, height, "Watch list", WatchList.EMPTY)
        else:
            if self._selected not in self._order:
                self._selected, self._note_scroll = self._order[0], 0
            index = self._order.index(self._selected)
            if screen.width >= Board.SIDE_BY_SIDE_MIN:
                table_w, detail_w = Board.split(available)
                self._draw_table(screen, Board.LEFT, y, table_w, height, entries, index)
                self._draw_detail(screen, Board.LEFT + table_w + Board.GAP, y, detail_w, height, entries[index], now)
            else:
                # The table's row already carries the selected item's facts: the tile under it is for the note.
                table_h = min(len(entries) + Tile.CARD_CHROME + 1,
                              max(Tile.CARD_CHROME + 3, height - 1 - self.STACKED_DETAIL_MIN_H))
                self._draw_table(screen, Board.LEFT, y, available, table_h, entries, index)
                self._draw_detail(screen, Board.LEFT, y + table_h + 1, available, height - table_h - 1, entries[index], now,
                                  compact=True)
        Tile.tint_cards(screen, self._cards)
        Board.draw_hints(screen, "↑↓ select  PgUp/PgDn scroll the note")

    # --- Summary tiles ---

    @staticmethod
    def _counts(entries: List[Entry], kinds: Tuple[Kind, ...]) -> List[Tuple[str, Color]]:
        counts = [(kind, sum(1 for entry in entries if entry.kind is kind)) for kind in kinds]
        return [(f"{count} {kind.label}", kind.color) for kind, count in counts if count]

    @classmethod
    def _attention_tile(cls, entries: List[Entry]) -> SummaryTile:
        segments = cls._counts(entries, (PAST_DUE, OVERDUE, CHECK_DUE, STALE))
        if not segments:
            return "Needs attention", "all clear", Theme.COLOR_GREEN, [("nothing overdue, nothing due", Theme.COLOR_DIM)]
        count = sum(1 for entry in entries if entry.kind.rank <= STALE.rank)
        return "Needs attention", str(count), segments[0][1], segments

    @staticmethod
    def _next_check_tile(entries: List[Entry], now: datetime) -> SummaryTile:
        polled = [entry for entry in entries if entry.fraction is not None]
        if not polled:
            return "Next check", "–", Theme.COLOR_DIM, [("nothing is polled", Theme.COLOR_DIM)]
        soonest = min(polled, key=lambda entry: entry.next_check or datetime.min)
        name = (soonest.item.name, Theme.COLOR_LABEL)
        if soonest.next_check is None:
            return "Next check", "now", soonest.check_color, [name, ("never checked", Theme.COLOR_DIM)]
        at = (when(soonest.next_check, now), Theme.COLOR_DIM)
        if soonest.next_check <= now:
            return "Next check", f"{span(now - soonest.next_check)} late", soonest.check_color, [name, at]
        return "Next check", f"in {span(soonest.next_check - now)}", Theme.COLOR_GREEN, [name, at]

    @staticmethod
    def _last_check_tile(entries: List[Entry], now: datetime) -> SummaryTile:
        checked = [entry for entry in entries if entry.checked]
        if not checked:
            return "Last check", "–", Theme.COLOR_DIM, [("nothing checked yet", Theme.COLOR_DIM)]
        latest = max(checked, key=lambda entry: entry.checked)
        return ("Last check", f"{span(now - latest.checked)} ago", Theme.COLOR_CYAN,
                [(latest.item.name, Theme.COLOR_LABEL), (when(latest.checked, now), Theme.COLOR_DIM)])

    @staticmethod
    def _watching_tile(entries: List[Entry]) -> SummaryTile:
        polled = sum(1 for entry in entries if entry.item.every)
        dated = sum(1 for entry in entries if not entry.item.every and entry.due_days is not None)
        groups = ((polled, "polled", Theme.COLOR_GREEN), (dated, "waiting for a date", Theme.COLOR_BLUE),
                  (len(entries) - polled - dated, "open", Theme.COLOR_LABEL))
        segments = [(f"{count} {label}", color) for count, label, color in groups if count]
        return ("Watching", "1 item" if len(entries) == 1 else f"{len(entries)} items", Theme.COLOR_BRIGHT,
                segments or [(WatchList.EMPTY, Theme.COLOR_DIM)])

    # --- The table ---

    @classmethod
    def _layout(cls, entries: List[Entry], inner_w: int) -> Tuple[int, int, List[Tuple[str, str, int]]]:
        """(name width, meter width, the columns shown): columns go in `DROP_ORDER` until the
        name has room, and room the longest name does not need lengthens the meter."""
        columns = list(cls.COLUMNS)

        def name_width() -> int:
            return inner_w - 2 - sum(width + 2 for _, _, width in columns)

        for key in cls.DROP_ORDER:
            if name_width() >= cls.NAME_MIN_W:
                break
            columns = [column for column in columns if column[0] != key]
        name_w, meter_w = max(4, name_width()), cls.METER_W
        if any(key == "next" for key, _, _ in columns):
            wanted = min(cls.NAME_MAX_W, max(display_width(entry.item.name) for entry in entries))
            grow = min(cls.METER_GROW, max(0, name_w - wanted))
            name_w, meter_w = name_w - grow, meter_w + grow
            columns = [(key, title, width + grow if key == "next" else width) for key, title, width in columns]
        return name_w, meter_w, columns

    def _draw_table(self, screen: TUIScreen, x: int, y: int, width: int, height: int,
                    entries: List[Entry], selected: int):
        count = len(entries)
        name_w, meter_w, columns = self._layout(entries, width - 4)
        now = self.now()
        rows = []
        for i, entry in enumerate(entries):
            chosen = i == selected
            cells: List[Board.Cell] = [(entry.item.name, Theme.COLOR_BRIGHT if chosen else Theme.COLOR_TEXT, chosen)]
            for key, _, _ in columns:
                cells.append(self._next_check_cell(entry, now, meter_w) if key == "next" else self._cell(key, entry, now))
            rows.append((entry.kind.color, cells))
        table = Table("Watch list", [("Item", name_w)] + [(title, column_w) for _, title, column_w in columns],
                      rows, WatchList.EMPTY, "1 item" if count == 1 else f"{count} items")
        self._top = Board.draw_table(screen, self._cards, x, y, width, height, table, self._top, selected)

    @staticmethod
    def _next_check_text(entry: Entry, now: datetime) -> Tuple[str, Color]:
        if entry.fraction is None:
            return ("?", Theme.COLOR_DIM) if entry.item.every else ("–", Theme.COLOR_FAINT)
        if entry.next_check is None:
            return "now", entry.check_color
        if entry.next_check <= now:
            return f"{span(now - entry.next_check)} late", entry.check_color
        return f"in {span(entry.next_check - now)}", Theme.COLOR_TEXT

    @classmethod
    def _next_check_cell(cls, entry: Entry, now: datetime, meter_w: int) -> Board.Cell:
        """The cadence as a meter that fills from the last check to the next, and the time left."""
        text, color = cls._next_check_text(entry, now)
        if entry.fraction is None:
            return text, color
        return Meter(entry.fraction, entry.check_color, meter_w, text, color, entry.fraction >= 1)

    @staticmethod
    def _due_color(entry: Entry) -> Color:
        if entry.due_days is None:
            return Theme.COLOR_LABEL
        if entry.due_days < 0:
            return Theme.COLOR_RED
        return Theme.COLOR_YELLOW if entry.due_days <= DUE_SOON_DAYS else Theme.COLOR_LABEL

    @classmethod
    def _cell(cls, key: str, entry: Entry, now: datetime) -> Tuple[str, Color]:
        item = entry.item
        if key == "status":
            return entry.kind.label, entry.kind.color
        if key == "last":
            return (f"{span(now - entry.checked)} ago", Theme.COLOR_LABEL) if entry.checked else ("never", Theme.COLOR_DIM)
        if key == "every":
            return (item.every, Theme.COLOR_LABEL) if item.every else ("–", Theme.COLOR_FAINT)
        if not item.due:
            return "–", Theme.COLOR_FAINT
        if entry.due_days is None:
            return item.due, Theme.COLOR_LABEL
        offset = entry.due_days
        distance = {0: "today", 1: "tomorrow"}.get(offset) or (f"in {offset}d" if offset > 0 else f"{-offset}d ago")
        return f"{WatchList.stamp(item.due):%b %-d} · {distance}", cls._due_color(entry)

    # --- The selected item ---

    def _facts(self, entry: Entry, now: datetime) -> List[Fact]:
        """The rows of the selected item; the check meter sits under its "Next check" row."""
        item = entry.item
        text, color = self._next_check_text(entry, now)
        if entry.next_check:
            text += f" · {when(entry.next_check, now)}"
        elif entry.fraction is None and not item.every:
            text, color = "not polled", Theme.COLOR_DIM
        facts = [Fact("Status", entry.reason, entry.kind.color), Fact("Next check", text, color)]
        if entry.fraction is not None:
            facts.append(Fact("", "", entry.check_color, meter=entry.fraction))
        facts.append(Fact("Last check", f"{span(now - entry.checked)} ago · {item.checked}", Theme.COLOR_TEXT)
                     if entry.checked else Fact("Last check", "never", Theme.COLOR_DIM))
        facts.append(Fact("Cadence", item.every, Theme.COLOR_TEXT) if item.every else Fact("Cadence", "–", Theme.COLOR_FAINT))
        if entry.due_days is not None:
            facts.append(Fact("Due", f"{item.due} · {day_distance(entry.due_days)}", self._due_color(entry)))
        else:
            facts.append(Fact("Due", item.due, Theme.COLOR_TEXT) if item.due else Fact("Due", "–", Theme.COLOR_FAINT))
        for label, value in (("Updated", item.updated), ("Created", item.created)):
            moment = WatchList.stamp(value)
            distance = f" · {day_distance((moment.date() - now.date()).days)}" if moment else ""
            facts.append(Fact(label, f"{value}{distance}", Theme.COLOR_LABEL))
        return facts

    def _note(self, item: WatchList.Item, width: int) -> Markdown.Document:
        """The item's note laid out `width` cells wide, kept until the items are re-read."""
        key = (item.name, width)
        if key not in self._notes:
            self._notes[key] = Markdown.layout(item.note.splitlines(), width)
        return self._notes[key]

    def _draw_detail(self, screen: TUIScreen, x: int, y: int, width: int, height: int, entry: Entry, now: datetime,
                     compact: bool = False):
        """The selected item: its facts, then its note; `compact` keeps only the first fact, why it has its state."""
        if height < Tile.CARD_CHROME + 1:
            return
        Tile.draw_card(screen, self._cards, x, y, width, height, entry.item.name, entry.kind.label, entry.kind.color)
        inner_w = width - 4
        facts = self._facts(entry, now)[:1 if compact else None]
        row = Board.draw_facts(screen, x, y, width, height, facts) + 1
        bottom = y + height - 2
        rows = self._note(entry.item, inner_w).rows
        room = bottom - row + 1
        if room <= 0 or not rows:
            return
        self._note_room = room
        self._note_scroll = max(0, min(self._note_scroll, len(rows) - room))
        mark = self.redactor.active
        for i, note_row in enumerate(rows[self._note_scroll:self._note_scroll + room]):
            Markdown.draw_row(screen, x + 2, row + i, inner_w, note_row)
            if mark:
                mark_masks(screen, self.redactor, x + 2, row + i, inner_w)
        draw_scrollbar(screen, x + width - 1, row, len(rows), room, len(rows) - room - self._note_scroll)
        screen.reset_color()
