import re
import difflib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple
from .ScrollableTabView import draw_scrollbar
from .TabView import SUB_CONTENT_TOP, TabView
from assistant import FirstRun
from common.logging import trace
from common.vault.Redactor import Redactor
from common.vault.Vault import Vault
from interfaces.console import Theme
from interfaces.console.widgets import Board, Markdown, Tile
from interfaces.console.widgets.Board import SummaryTile
from interfaces.console.widgets.DiffText import (COLOR_ADDED_HIGHLIGHT_BG, COLOR_ADDED_LINE_BG, COLOR_GUTTER_ADD,
                                                 COLOR_GUTTER_DEL, COLOR_REMOVED, COLOR_REMOVED_HIGHLIGHT_BG,
                                                 COLOR_REMOVED_LINE_BG, inline_diff)
from interfaces.console.framework.TUIScreen import TUIScreen, Color
from interfaces.console.framework.TextWidth import display_width, fit
from interfaces.console.Theme import COLOR_YELLOW, COLOR_YELLOW_DIM
from interfaces.console.widgets.SecretMarks import mark_masks

FIRST_CURATION_LABEL = "Heads up"
FIRST_CURATION_NOTICE = ("Keep is building its memory for the first time. What you read here is a "
                         "work in progress and may look wrong until that finishes.")

class Stats(NamedTuple):
    """What the text on show amounts to (a diff's removed lines left out)."""
    chars: int = 0
    words: int = 0
    lines: int = 0
    entries: int = 0     # bullets and numbered items


def tint_added(color: Color) -> Color:
    """An added line: every colour pulled most of the way to the green."""
    return Theme.mix(color, Theme.COLOR_GREEN, 0.65)


def tint_removed(_: Color) -> Color:
    return COLOR_REMOVED


def amount(chars: int) -> str:
    return str(chars) if chars < 1000 else f"{chars / 1000:.1f}k"


def counted(count: int, noun: str, plural: str = "") -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {plural or noun + 's'}"


class MemoryTab(TabView):
    """Base of the Memory group's document sub-tabs (`BankTab`, `ProceduresTab`): one part of the
    long-term memory, `_live_text` being the part, read as a document in the Dashboard's tiles.
    Summary tiles on top (how much there is, how it is divided, and what the tab adds), an
    outline of the sections on the left from 110 columns — each with a meter of its share and
    the one being read highlighted — and the text itself, laid out by `widgets/Markdown`: no
    markdown marker is shown, each section has its own accent. ↑↓ and PgUp/PgDn scroll, n/p
    jump a section.
    A tab with `HAS_HISTORY` also browses the snapshots in `log_dir` with ←→ (the bank's: what
    each curation made of it): added and removed lines in the diff's colours, the outline
    counting the changes per section, the view opening on the first change.
    Every vaulted value is masked the way the Vault tab lists it (`•••• 1234`, `a•••@example.com`)
    as the text is loaded — before it is laid out or diffed — so the tab never shows one in the
    clear; the Vault tab reveals on request. Outside a diff the masks are marked in bold orange.
    While the first-run pipeline curates the first bank (`FirstRun.curation_progress`) a two-row
    orange heads-up band sits above the tiles: what they show is a work in progress, not a verdict."""

    HAS_HISTORY = False
    TITLE = "Memory"
    NOTES_BAND = "New notes"
    SECTIONS_LABEL = "Sections"

    BODY_MIN_H = Tile.CARD_CHROME + 6    # the document's height below which the summary tiles give way
    OUTLINE_MIN_SCREEN = 110             # screen columns from which the outline sits beside the text
    OUTLINE_MIN_W = 36
    OUTLINE_MAX_W = 46
    OUTLINE_METER_W = 6
    OUTLINE_SIZE_W = 5
    TEXT_MAX_W = 120                     # a line longer than this is hard to read back from its end
    SCAN_EVERY = 30                      # frames between looks at the snapshot directory
    END = 1 << 30                        # a scroll position past any document: `_settle` brings it to the last page
    CHANGE_LEAD = 3                      # rows of context above the first change a snapshot opens on

    def __init__(self, redactor: Optional[Redactor] = None):
        self._redactor = redactor  # over the Vault singleton, resolved on first draw
        self.memory_lines: List[str] = []
        self._line_types: List[str] = []  # 'normal', 'added', 'removed' - parallel to memory_lines
        self._inline_highlights: List[Optional[List[Tuple[int, int]]]] = []  # char spans to highlight
        self._bands: Dict[int, Tuple[str, str]] = {}  # line index -> the (title, detail) of the band drawn before it
        self.last_memory_hash = 0
        self.log_dir = Path("log/memory")

        # History browsing state
        self._snapshot_dates: List[str] = []  # sorted dates YYYYMMDD (newest first)
        self._snapshot_files: dict = {}  # date -> list of file paths
        self._history_index: Optional[int] = None  # None = live, 0 = newest snapshot, etc.
        self._last_scan_hash = 0
        self._scanned = False
        self._viewing_date_label: Optional[str] = None
        self._compared = False  # whether the snapshot on show had an older one to diff against

        # What is on screen
        self._cards: List[Tile.Card] = []
        self._version = 0  # bumped whenever the lines change: what a cached layout is good for
        self._layouts: Dict[int, Markdown.Document] = {}
        self._layouts_version = -1
        self._stats = Stats()
        self._doc: Optional[Markdown.Document] = None
        self._top = 0
        self._room = 1
        self._outline_top = 0
        self._seek_change = False
        self._notes = 0  # the notes after the curated part of the live text

    @property
    def _has_diff(self) -> bool:
        return len(self._line_types) > 0 and any(t != 'normal' for t in self._line_types)

    @property
    def redactor(self) -> Redactor:
        if self._redactor is None:
            self._redactor = Redactor(Vault.get_instance())
            trace(f"Memory tab: secret marking resolved, {len(self._redactor.vault)} vault secret(s)")
        return self._redactor

    @staticmethod
    def now() -> datetime:
        return datetime.now()

    def draw(self, screen: TUIScreen, frame: int):
        if self.HAS_HISTORY and (frame % self.SCAN_EVERY == 0 or not self._scanned):
            self._scan_snapshots()
        self._read_memory_lines()
        self._cards = []

        y = SUB_CONTENT_TOP
        if FirstRun.curation_progress() is not None:
            self._draw_first_curation_notice(screen, y)
            y += self.NOTICE_ROWS + 1
        available = screen.width - Board.LEFT - Board.RIGHT_MARGIN
        with_summary = screen.height - 1 - y - Board.SUMMARY_H - 1 >= self.BODY_MIN_H
        body_y = y + Board.SUMMARY_H + 1 if with_summary else y
        height = screen.height - 1 - body_y      # the last row is the key hints'

        gutter = 2 if self._has_diff else 0
        outline_w = 0
        if screen.width >= self.OUTLINE_MIN_SCREEN:
            outline_w = max(self.OUTLINE_MIN_W, min(self.OUTLINE_MAX_W, available * 26 // 100))
            doc = self._layout(self._text_width(available - outline_w - Board.GAP, gutter))
            if len(self._outline_entries(doc)) < 2:
                outline_w = 0
        document_w = available - outline_w - Board.GAP if outline_w else available
        text_w = self._text_width(document_w, gutter)
        doc = self._doc = self._layout(text_w)

        if with_summary:
            Board.draw_summary(screen, self._cards, y, available, self._summary_tiles(doc))
        if height > Tile.CARD_CHROME:
            self._room = height - Tile.CARD_CHROME
            self._settle(doc)
            if outline_w:
                self._draw_outline(screen, Board.LEFT, body_y, outline_w, height, doc)
                self._draw_document(screen, Board.LEFT + outline_w + Board.GAP, body_y, document_w, height, doc, text_w)
            else:
                self._draw_document(screen, Board.LEFT, body_y, document_w, height, doc, text_w)
        Tile.tint_cards(screen, self._cards)

        history = ""
        if self.HAS_HISTORY:
            history = "  ← older  → newer" if self._history_index is not None else "  ←→ history"
        Board.draw_hints(screen, f"↑↓ scroll  PgUp/PgDn page  n/p section{history}")

    def handle_input(self, key: str) -> bool:
        """Scrolling, section jumps and history navigation; `draw` keeps the position in range."""
        if key in ('up', 'down'):
            self._top += 1 if key == 'down' else -1
            return True
        if key in ('pageup', 'pagedown'):
            self._top += self._room if key == 'pagedown' else -self._room
            return True
        if key in ('home', 'end'):
            self._top = 0 if key == 'home' else self.END
            return True
        if key in ('n', 'p'):
            self._jump(1 if key == 'n' else -1)
            return True
        if not self.HAS_HISTORY:
            return False
        if key == 'left':
            self._navigate_history(-1)
            return True
        elif key == 'right':
            self._navigate_history(1)
            return True
        return False

    def _jump(self, direction: int):
        """To the next or the previous section's heading."""
        if self._doc is None:
            return
        stops = [heading.row for heading in self._doc.headings if heading.depth <= 0]
        if len(stops) < 2:
            stops = [heading.row for heading in self._doc.headings]
        ahead = [row for row in stops if row > self._top] if direction > 0 else [row for row in stops if row < self._top]
        if ahead:
            self._top = ahead[0] if direction > 0 else ahead[-1]

    # --- The text and its layout ---

    def _set_lines(self, lines: List[str], types: Optional[List[str]] = None,
                   highlights: Optional[list] = None, bands: Optional[Dict[int, Tuple[str, str]]] = None):
        self.memory_lines = lines
        self._line_types = types or []
        self._inline_highlights = highlights or []
        self._bands = bands or {}
        self._version += 1
        kept = [line for i, line in enumerate(lines) if self._line_type(i) != 'removed']
        self._stats = Stats(chars=sum(len(line) for line in kept), words=sum(len(line.split()) for line in kept),
                            lines=len(kept),
                            entries=sum(1 for line in kept if Markdown.BULLET.match(line) or Markdown.NUMBERED.match(line)))

    def _line_type(self, line: int) -> str:
        return self._line_types[line] if 0 <= line < len(self._line_types) else 'normal'

    def _text_width(self, tile_w: int, gutter: int) -> int:
        return min(self.TEXT_MAX_W, tile_w - 4 - gutter)

    def _layout(self, width: int) -> Markdown.Document:
        """The text laid out `width` cells wide, kept until the text changes."""
        if self._layouts_version != self._version:
            self._layouts, self._layouts_version = {}, self._version
        if width not in self._layouts:
            removed = frozenset(i for i, kind in enumerate(self._line_types) if kind == 'removed')
            lines = [line.rstrip('\n\r') for line in self.memory_lines]
            self._layouts[width] = Markdown.layout(lines, width, removed, self._bands)
        return self._layouts[width]

    def _settle(self, doc: Markdown.Document):
        """Keep the scroll position inside the document; a snapshot just opened starts at its first change."""
        if self._seek_change:
            self._seek_change = False
            changed = next((k for k, row in enumerate(doc.rows) if self._line_type(row.line) != 'normal'), 0)
            self._top = changed - self.CHANGE_LEAD
        self._top = max(0, min(self._top, len(doc.rows) - self._room))

    @staticmethod
    def _outline_entries(doc: Markdown.Document) -> List[Markdown.Heading]:
        return [heading for heading in doc.headings if heading.depth <= 1]

    # --- Summary tiles ---

    def _summary_tiles(self, doc: Markdown.Document) -> List[SummaryTile]:
        tiles = [self._size_tile(), self._sections_tile(doc)]
        if self._history_index is not None:
            return [self._snapshot_tile(), self._changes_tile()] + tiles
        return self._live_tiles() + tiles

    def _live_tiles(self) -> List[SummaryTile]:
        """The tiles a tab puts before the two every tab has, while it shows the live text."""
        return []

    def _size_tile(self) -> SummaryTile:
        stats = self._stats
        return ("Size", f"{stats.chars / 1024:.1f} KB", Theme.COLOR_CYAN,
                [(counted(stats.words, "word"), Theme.COLOR_LABEL), (counted(stats.lines, "line"), Theme.COLOR_DIM)])

    def _sections_tile(self, doc: Markdown.Document) -> SummaryTile:
        sections = sum(1 for heading in doc.headings if heading.depth == 0 and heading.level)
        if not sections:
            return self.SECTIONS_LABEL, "–", Theme.COLOR_DIM, [("no headings", Theme.COLOR_DIM)]
        headings = sum(1 for heading in doc.headings if heading.level)
        return (self.SECTIONS_LABEL, str(sections), Theme.COLOR_PURPLE,
                [(counted(headings, "heading"), Theme.COLOR_LABEL),
                 (counted(self._stats.entries, "entry", "entries"), Theme.COLOR_DIM)])

    def _last_snapshot(self) -> Optional[datetime]:
        """When the newest snapshot on record was written, from its file name."""
        newest = max((path.name for paths in self._snapshot_files.values() for path in paths), default="")
        try:
            return datetime.strptime(newest[:15], "%Y%m%d_%H%M%S")
        except ValueError:
            return None

    def _snapshot_tile(self) -> SummaryTile:
        return ("Snapshot", f"{self._history_index + 1} of {len(self._snapshot_dates)}", Theme.COLOR_PURPLE,
                [(self._viewing_date_label or "", Theme.COLOR_LABEL), ("→ towards live", Theme.COLOR_DIM)])

    def _change_counts(self) -> Tuple[int, int]:
        return self._line_types.count('added'), self._line_types.count('removed')

    def _changes_tile(self) -> SummaryTile:
        added, removed = self._change_counts()
        if not added and not removed:
            reason = "nothing changed" if self._compared else "the oldest snapshot: nothing to compare it with"
            return "Changes", "–", Theme.COLOR_DIM, [(reason, Theme.COLOR_DIM)]
        return ("Changes", f"+{added} −{removed}", Theme.COLOR_GREEN if added >= removed else Theme.COLOR_RED,
                [(f"{counted(added, 'line')} added", Theme.COLOR_GREEN), (f"{removed} removed", Theme.COLOR_RED)])

    # --- The outline ---

    def _focus_row(self, doc: Markdown.Document) -> int:
        """The row whose section is the one being read: the top one, or the last one once the
        document is scrolled to its end — a short last section never reaches the top."""
        at_end = self._top > 0 and self._top >= len(doc.rows) - self._room
        return self._top + self._room - 1 if at_end else self._top

    def _draw_outline(self, screen: TUIScreen, x: int, y: int, width: int, height: int, doc: Markdown.Document):
        entries = self._outline_entries(doc)
        sections = [entry for entry in entries if entry.depth <= 0]
        room = self._room
        if len(entries) > room and sections:
            entries = sections
        Tile.draw_card(screen, self._cards, x, y, width, height, "Outline",
                       f"{len(sections)} {self.SECTIONS_LABEL.lower()}" if sections else "")
        focus = self._focus_row(doc)
        chosen = max((k for k, entry in enumerate(entries) if entry.row <= focus), default=-1)
        self._outline_top = Board.scrolled(self._outline_top, len(entries), room, chosen)
        largest = max((entry.size for entry in sections), default=0) or 1
        has_diff = self._has_diff
        right_edge = x + width - 2

        for k, entry in enumerate(entries[self._outline_top:self._outline_top + room]):
            row_y = y + Tile.CARD_TOP + k
            current = self._outline_top + k == chosen
            part = entry.depth > 0
            right_w = 0
            screen.reset_color()
            if has_diff:
                kinds = self._line_types[entry.line:entry.until]
                counts = [(f"+{kinds.count('added')}", Theme.COLOR_GREEN, kinds.count('added')),
                          (f"−{kinds.count('removed')}", Theme.COLOR_RED, kinds.count('removed'))]
                texts = [(text, color) for text, color, count in counts if count]
                right_w = sum(len(text) + 1 for text, _ in texts) - 1 if texts else 0
                at = right_edge - right_w
                for text, color in texts:
                    screen.set_color(color)
                    screen.write_at(at, row_y, text)
                    at += len(text) + 1
            elif not part:
                right_w = self.OUTLINE_METER_W + 1 + self.OUTLINE_SIZE_W
                Tile.draw_meter(screen, right_edge - right_w, row_y, self.OUTLINE_METER_W, entry.size / largest, entry.accent)
                screen.set_color(Theme.COLOR_DIM)
                screen.write_at(right_edge - self.OUTLINE_SIZE_W, row_y, amount(entry.size).rjust(self.OUTLINE_SIZE_W))
            screen.reset_color()
            name_x = x + 6 if part else x + 4
            if not part:
                screen.set_color(entry.accent)
                screen.write_at(x + 2, row_y, "●")
            screen.set_color(Theme.COLOR_BRIGHT if current else Theme.COLOR_LABEL if part else Theme.COLOR_TEXT)
            if current:
                screen.bold()
            screen.write_at(name_x, row_y, fit(entry.title, right_edge - name_x - (right_w + 1 if right_w else 0)))
            screen.reset_color()
            if current:
                screen.tint_row(x, row_y, width, Theme.COLOR_SELECTED_BG)
        draw_scrollbar(screen, x + width - 1, y + Tile.CARD_TOP, len(entries), room,
                       len(entries) - room - self._outline_top)
        screen.reset_color()

    # --- The document ---

    def _document_title(self, doc: Markdown.Document) -> Tuple[str, str, Color]:
        if self._history_index is not None:
            return (self._viewing_date_label or "", f"snapshot {self._history_index + 1}/{len(self._snapshot_dates)}",
                    Theme.COLOR_PURPLE)
        focus = self._focus_row(doc)
        reading = [heading for heading in doc.headings if heading.depth <= 0 and heading.row <= focus]
        return (self.TITLE, reading[-1].title, reading[-1].accent) if reading else (self.TITLE, "", Theme.COLOR_BRIGHT)

    def _draw_document(self, screen: TUIScreen, x: int, y: int, width: int, height: int,
                       doc: Markdown.Document, text_w: int):
        label, value, color = self._document_title(doc)
        Tile.draw_card(screen, self._cards, x, y, width, height, label, fit(value, max(0, width - display_width(label) - 6)), color)
        has_diff = self._has_diff
        content_x = x + 2 + (2 if has_diff else 0)
        # The diff colours already say what changed; the orange on the masks waits for the live view
        mark_secrets = not has_diff and self.redactor.active
        top_y = y + Tile.CARD_TOP

        for k, row in enumerate(doc.rows[self._top:self._top + self._room]):
            row_y = top_y + k
            kind = self._line_type(row.line)
            if kind == 'normal':
                Markdown.draw_row(screen, content_x, row_y, text_w, row)
                if mark_secrets:
                    mark_masks(screen, self.redactor, content_x, row_y, text_w)
                continue
            added = kind == 'added'
            lit = self._inline_highlights[row.line] if row.line < len(self._inline_highlights) else None
            screen.reset_color()
            screen.set_color(COLOR_GUTTER_ADD if added else COLOR_GUTTER_DEL)
            screen.write_at(x + 2, row_y, '+' if added else '-')
            Markdown.draw_row(screen, content_x, row_y, text_w, row, tint_added if added else tint_removed, lit,
                              COLOR_ADDED_HIGHLIGHT_BG if added else COLOR_REMOVED_HIGHLIGHT_BG, fill=False)
            screen.tint_row(x, row_y, width - 1, COLOR_ADDED_LINE_BG if added else COLOR_REMOVED_LINE_BG)
        draw_scrollbar(screen, x + width - 1, top_y, len(doc.rows), self._room, len(doc.rows) - self._room - self._top)
        screen.reset_color()

    # --- History ---

    def _navigate_history(self, direction: int):
        """Navigate through memory history. direction: -1 = older, +1 = newer"""
        self._scan_snapshots()
        if not self._snapshot_dates:
            return

        if direction == -1:  # Go older
            if self._history_index is None:
                self._history_index = 0
            elif self._history_index < len(self._snapshot_dates) - 1:
                self._history_index += 1
        elif direction == 1:  # Go newer
            if self._history_index is not None:
                if self._history_index <= 0:
                    self._history_index = None
                else:
                    self._history_index -= 1

        self.last_memory_hash = 0  # Force reload
        self._top = 0
        self._seek_change = self._history_index is not None

    def _scan_snapshots(self):
        """Scan log/memory/ for snapshot files, grouped by date"""
        self._scanned = True
        if not self.log_dir.exists():
            self._snapshot_dates = []
            self._snapshot_files = {}
            return

        try:
            files = sorted(self.log_dir.glob("*_memory.md"), reverse=True)
        except OSError:
            return

        scan_hash = hash(tuple(str(f) for f in files))
        if scan_hash == self._last_scan_hash:
            return
        self._last_scan_hash = scan_hash

        dates_dict: dict = {}
        for f in files:
            match = re.match(r'^(\d{8})_', f.name)
            if match:
                date_str = match.group(1)
                dates_dict.setdefault(date_str, []).append(f)

        self._snapshot_dates = sorted(dates_dict.keys(), reverse=True)
        self._snapshot_files = dates_dict

    def _read_memory_lines(self):
        """Read memory lines from the appropriate source"""
        if self._history_index is None:
            self._show_live()
        else:
            self._read_snapshot_memory()

    def _show_live(self):
        """Back to the live memory, with no diff annotations."""
        self._history_index = None
        self._viewing_date_label = None
        self._read_live_memory()

    def _read_live_memory(self):
        """Read memory lines from the tab's part of the memory"""
        try:
            memory_text = self._live_text()
            memory_hash = self._content_hash(memory_text)
            if memory_hash != self.last_memory_hash:
                offset = self._notes_offset(memory_text)
                text, notes = (memory_text[:offset], memory_text[offset:]) if offset is not None else (memory_text, "")
                lines = self.redactor.mask(text).splitlines(keepends=True)
                bands = {}
                self._notes = len(re.split(r"\n\s*\n", notes.strip())) if notes.strip() else 0
                if self._notes:
                    bands[len(lines)] = (self.NOTES_BAND, f"{counted(self._notes, 'note')} since the last curation")
                    lines += self.redactor.mask(notes).splitlines(keepends=True)
                self._set_lines(lines, bands=bands)
                self.last_memory_hash = memory_hash
                trace(f"{self.name} tab: {len(lines)} line(s) loaded, {self._notes} note(s) after the curated part")
        except Exception as e:
            self._set_lines([f"Error reading memory: {str(e)}"])

    def _live_text(self) -> str:
        """The tab's part of the memory as it is now, unmasked."""
        raise NotImplementedError

    def _notes_offset(self, text: str) -> Optional[int]:
        """Where in `text` the notes taken since it was last curated start; None when the
        tab's part has no such tail. What follows is shown under a band of its own."""
        return None

    def _read_snapshot_memory(self):
        """Read memory from a historical snapshot with diff annotations"""
        self._scan_snapshots()
        if self._history_index is None or self._history_index >= len(self._snapshot_dates):
            self._show_live()
            return

        date_str = self._snapshot_dates[self._history_index]
        files = self._snapshot_files.get(date_str, [])
        if not files:
            self._viewing_date_label = date_str
            self._set_lines(["No snapshot data found"])
            return

        # Use the latest snapshot for this date
        snapshot_path = sorted(files, reverse=True)[0]

        try:
            dt = datetime.strptime(date_str, "%Y%m%d")
            self._viewing_date_label = dt.strftime("%A, %B %-d, %Y")
        except ValueError:
            self._viewing_date_label = date_str

        try:
            with open(snapshot_path, 'r', encoding='utf-8') as f:
                content = f.read()

            memory_hash = self._content_hash(content)
            if memory_hash != self.last_memory_hash:
                new_memory = self.redactor.mask(self._extract_updated_memory(content))

                # Load previous snapshot for diff
                old_memory = self._load_previous_memory()
                self._compared = bool(old_memory)

                if old_memory:
                    lines, types, highlights = self._build_annotated_view(old_memory, new_memory)
                    self._set_lines([l + '\n' for l in lines], types, highlights)
                else:
                    lines = new_memory.splitlines(keepends=True)
                    self._set_lines(lines, ['normal'] * len(lines), [None] * len(lines))

                self.last_memory_hash = memory_hash
                trace(f"{self.name} tab: snapshot {snapshot_path.name} loaded, "
                      f"+{self._line_types.count('added')} -{self._line_types.count('removed')} line(s)")
        except Exception as e:
            self._set_lines([f"Error reading snapshot: {str(e)}"])

    def _content_hash(self, text: str) -> int:
        """Changes with the text or the vault: a secret added while the tab is open is masked
        on the next frame, not on the next edit of the file."""
        return hash((text, self.redactor.vault.version))

    def _load_previous_memory(self) -> Optional[str]:
        """Load the memory from the previous (older) snapshot for diff comparison"""
        if self._history_index is None:
            return None

        prev_index = self._history_index + 1
        if prev_index >= len(self._snapshot_dates):
            return None

        prev_date = self._snapshot_dates[prev_index]
        prev_files = self._snapshot_files.get(prev_date, [])
        if not prev_files:
            return None

        prev_path = sorted(prev_files, reverse=True)[0]
        try:
            with open(prev_path, 'r', encoding='utf-8') as f:
                return self.redactor.mask(self._extract_updated_memory(f.read()))
        except Exception:
            return None

    @staticmethod
    def _build_annotated_view(old_memory: str, new_memory: str) -> Tuple[List[str], List[str], List[Optional[List[Tuple[int, int]]]]]:
        """Build a merged diff view with line type annotations and inline highlights.
        Returns (lines, types, highlights) where types are 'normal', 'added', 'removed'
        and highlights are optional lists of (start, end) char spans for within-line changes."""
        old_lines = old_memory.splitlines()
        new_lines = new_memory.splitlines()

        result_lines: List[str] = []
        result_types: List[str] = []
        result_highlights: List[Optional[List[Tuple[int, int]]]] = []

        sm = difflib.SequenceMatcher(None, old_lines, new_lines)
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op == 'equal':
                for line in new_lines[j1:j2]:
                    result_lines.append(line)
                    result_types.append('normal')
                    result_highlights.append(None)
            elif op == 'replace':
                old_block = old_lines[i1:i2]
                new_block = new_lines[j1:j2]
                # Pair up lines for inline diff when block sizes match
                paired = min(len(old_block), len(new_block))
                for k in range(paired):
                    old_spans, new_spans = inline_diff(old_block[k], new_block[k])
                    result_lines.append(old_block[k])
                    result_types.append('removed')
                    result_highlights.append(old_spans)
                    result_lines.append(new_block[k])
                    result_types.append('added')
                    result_highlights.append(new_spans)
                # Remaining unpaired lines
                for line in old_block[paired:]:
                    result_lines.append(line)
                    result_types.append('removed')
                    result_highlights.append(None)
                for line in new_block[paired:]:
                    result_lines.append(line)
                    result_types.append('added')
                    result_highlights.append(None)
            elif op == 'insert':
                for line in new_lines[j1:j2]:
                    result_lines.append(line)
                    result_types.append('added')
                    result_highlights.append(None)
            elif op == 'delete':
                for line in old_lines[i1:i2]:
                    result_lines.append(line)
                    result_types.append('removed')
                    result_highlights.append(None)

        return result_lines, result_types, result_highlights

    @staticmethod
    def _extract_updated_memory(debug_log_content: str) -> str:
        """Extract the '## Updated Memory' section from a debug log file"""
        marker = "## Updated Memory"
        idx = debug_log_content.find(marker)
        if idx != -1:
            after_marker = debug_log_content[idx + len(marker):]
            return after_marker.lstrip('\n')
        return debug_log_content

    NOTICE_ROWS = 2

    @staticmethod
    def _draw_first_curation_notice(screen: TUIScreen, y: int):
        """Two rows in the Home tab's progress-bar colours: the bold label centred on the
        first, the sentence centred on the second, each clipped to the width."""
        width = screen.width - 2
        for row_y, text, bold in ((y, FIRST_CURATION_LABEL, True), (y + 1, FIRST_CURATION_NOTICE, False)):
            screen.reset_color()
            screen.write_at(1, row_y, " " * width)
            text = fit(text, width)
            screen.set_color(COLOR_YELLOW)
            if bold:
                screen.bold()
            screen.write_at(1 + (width - len(text)) // 2, row_y, text)
            screen.reset_color()
            screen.tint_row(1, row_y, width, COLOR_YELLOW_DIM)
