"""The Photos tab: the indexed photo library, one photo at a time, as a board in the Dashboard's
tiles — summary tiles (how much of the library is indexed, where the browsing stands, what the
indexing last did), the photo's description and its details with the two scores as meters. From
150 columns the details sit beside the description, under it on a narrower screen."""
import os
import sqlite3
import subprocess
import threading
import time
from datetime import datetime
from typing import List, Optional

from .TabView import TabView, SUB_CONTENT_TOP
from interfaces.console import Theme
from interfaces.console.framework.TUIScreen import TUIScreen
from interfaces.console.framework.TextWidth import wrap
from interfaces.console.widgets import Board, Tile
from interfaces.console.widgets.Board import Fact, SummaryTile
from common.FloxyDatabase import FloxyDatabase
from assistant.integration.integrations.photos.PhotoProcessor import PhotoProcessor, PHOTOS_SQLITE_PATH

DETAILS_MIN_W = 60
FACT_LABEL_W = 14
DESCRIPTION_MIN_ROWS = 2
NONE = "\u2014"
HINTS = "\u2191\u2193 prev/next  PgUp/PgDn \u00b110  Home/End  Ctrl-S sort  Ctrl-O open  Ctrl-P process  Ctrl-B bulk"
# The colour of an indexing outcome by PhotoProcessor's status
PROCESS_COLORS = {"ok": Theme.COLOR_GREEN, "none": Theme.COLOR_LABEL, "error": Theme.COLOR_RED}

SORT_MODES = [
    ("creation_date", False, "Date (oldest)"),
    ("creation_date", True, "Date (newest)"),
    ("indexed_at", True, "Recently indexed"),
    ("pretty_score", True, "Pretty score"),
    ("family_importance_score", True, "Family score"),
]


class PhotosTab(TabView):
    """Tab for browsing the indexed photo library and triggering photo processing."""

    def __init__(self):
        self._index: int = 0
        self._count: int = 0
        self._total_apple: int = 0
        self._photo: Optional[dict] = None
        self._sort_mode_idx: int = 1
        self._last_load_time: float = 0
        self._reload_interval: float = 30.0
        self._loading: bool = False
        self._processing: bool = False
        self._process_status: str = ""
        self._process_status_type: str = ""
        self._cards: List[Tile.Card] = []

    def _current_sort(self):
        return SORT_MODES[self._sort_mode_idx]

    def _load_data(self):
        if self._loading:
            return
        self._loading = True

        def load():
            try:
                db = FloxyDatabase.get_instance()
                self._count = db.get_photo_count()
                self._total_apple = self._get_apple_photo_count()
                if self._count == 0:
                    self._photo = None
                    self._index = 0
                    return
                self._index = max(0, min(self._index, self._count - 1))
                sort_col, sort_desc, _ = self._current_sort()
                results = db.get_photos_paginated(
                    offset=self._index, limit=1,
                    sort_by=sort_col, sort_desc=sort_desc)
                self._photo = results[0] if results else None
            finally:
                self._last_load_time = time.time()
                self._loading = False

        threading.Thread(target=load, daemon=True).start()

    def _navigate(self, delta: int):
        if self._count == 0:
            return
        new_index = max(0, min(self._index + delta, self._count - 1))
        if new_index != self._index:
            self._index = new_index
            self._load_data()

    def _go_to(self, index: int):
        if self._count == 0:
            return
        clamped = max(0, min(index, self._count - 1))
        if clamped != self._index:
            self._index = clamped
            self._load_data()

    def draw(self, screen: TUIScreen, frame: int):
        if time.time() - self._last_load_time > self._reload_interval:
            self._load_data()

        self._cards = []
        available = screen.width - Board.LEFT - Board.RIGHT_MARGIN
        bottom = screen.height - 1                # the last row is the key hints'
        y = SUB_CONTENT_TOP
        Board.draw_summary(screen, self._cards, y, available,
                           [self._indexed_tile(), self._browsing_tile(), self._indexing_tile()])
        y += Board.SUMMARY_H + 1
        height = bottom - y

        photo = self._photo
        if photo is None:
            Board.draw_notice(screen, self._cards, Board.LEFT, y, available, height, "Photo",
                              "Loading..." if self._loading else "No photos to display \u2014 Ctrl-P indexes the next one")
        elif screen.width >= Board.SIDE_BY_SIDE_MIN:
            description_w, details_w = Board.split(available, DETAILS_MIN_W)
            facts = self._facts(photo)
            self._draw_description(screen, Board.LEFT, y, description_w, height, photo)
            self._draw_details(screen, Board.LEFT + description_w + Board.GAP, y, details_w,
                               min(height, len(facts) + Tile.CARD_CHROME), facts)
        else:
            facts = self._facts(photo)
            details_h = len(facts) + Tile.CARD_CHROME
            lines = wrap(photo.get("detailed_description") or "", available - 4)
            description_h = Tile.CARD_CHROME + max(DESCRIPTION_MIN_ROWS, min(len(lines), height - 1 - details_h - Tile.CARD_CHROME))
            self._draw_description(screen, Board.LEFT, y, available, description_h, photo)
            y += description_h + 1
            self._draw_details(screen, Board.LEFT, y, available, min(details_h, bottom - y), facts)
        Tile.tint_cards(screen, self._cards)
        Board.draw_hints(screen, HINTS)

    # --- summary tiles -----------------------------------------------------------------------

    def _indexed_tile(self) -> SummaryTile:
        if not self._count:
            return "Indexed", "0", Theme.COLOR_DIM, [("no photos indexed", Theme.COLOR_DIM)]
        if self._total_apple <= 0:
            return "Indexed", f"{self._count:,}", Theme.COLOR_BRIGHT, [("photos described", Theme.COLOR_LABEL)]
        share = min(100, round(100 * self._count / self._total_apple))
        return ("Indexed", f"{self._count:,}", Theme.COLOR_BRIGHT,
                [(f"{share}% of the library", Theme.COLOR_LABEL), (f"{self._total_apple:,} in iCloud", Theme.COLOR_DIM)])

    def _browsing_tile(self) -> SummaryTile:
        sort = (f"sorted by {self._current_sort()[2].lower()}", Theme.COLOR_LABEL)
        if not self._count:
            return "Browsing", NONE, Theme.COLOR_DIM, [sort]
        return "Browsing", f"{self._index + 1:,} of {self._count:,}", Theme.COLOR_CYAN, [sort, ("Ctrl-S changes it", Theme.COLOR_DIM)]

    def _indexing_tile(self) -> SummaryTile:
        if self._processing:
            return ("Indexing", "running", Theme.COLOR_YELLOW,
                    [(self._process_status or "Processing...", Theme.COLOR_YELLOW)])
        if not self._process_status:
            return "Indexing", "idle", Theme.COLOR_DIM, [("Ctrl-P the next photo, Ctrl-B a batch", Theme.COLOR_DIM)]
        color = PROCESS_COLORS.get(self._process_status_type, Theme.COLOR_LABEL)
        value = {"ok": "done", "error": "failed"}.get(self._process_status_type, "idle")
        return "Indexing", value, color, [(self._process_status, color)]

    # --- the photo ---------------------------------------------------------------------------

    def _draw_description(self, screen: TUIScreen, x: int, y: int, width: int, height: int, photo: dict):
        """The photo's short description as the tile's title, the detailed one wrapped under it."""
        if height <= Tile.CARD_CHROME:
            return
        Tile.draw_card(screen, self._cards, x, y, width, height, photo.get("short_description") or "Photo",
                       self._type_label(photo), Theme.COLOR_BLUE)
        screen.set_color(Theme.COLOR_TEXT)
        for i, line in enumerate(wrap(photo.get("detailed_description") or "", width - 4, height - Tile.CARD_CHROME)):
            screen.write_at(x + 2, y + Tile.CARD_TOP + i, line)
        screen.reset_color()

    def _facts(self, photo: dict) -> List[Fact]:
        location = []
        if photo.get("location_text"):
            location.append(photo["location_text"].replace("/", ", "))
        if photo.get("latitude") is not None and photo.get("longitude") is not None:
            location.append(f"({photo['latitude']:.4f}, {photo['longitude']:.4f})")

        def text(label: str, value: Optional[str]) -> Fact:
            return Fact(label, value, Theme.COLOR_TEXT) if value and value != NONE else Fact(label, NONE, Theme.COLOR_FAINT)

        def score(label: str, value: Optional[int], color) -> Fact:
            return Fact(label, str(value or 0), color, meter=(value or 0) / 100)

        return [text("Date", self._format_date(photo.get("creation_date"))),
                text("Location", "  ".join(location)),
                text("Albums", photo.get("albums")),
                text("Persons", photo.get("persons")),
                Fact("", "", Theme.COLOR_FAINT),
                score("Pretty score", photo.get("pretty_score"), Theme.COLOR_PINK),
                score("Family score", photo.get("family_importance_score"), Theme.COLOR_CYAN),
                Fact("", "", Theme.COLOR_FAINT),
                Fact("Indexed", self._format_date(photo.get("indexed_at")), Theme.COLOR_LABEL),
                Fact("Id", str(photo.get("asset_guid") or NONE), Theme.COLOR_DIM)]

    def _draw_details(self, screen: TUIScreen, x: int, y: int, width: int, height: int, facts: List[Fact]):
        if height <= Tile.CARD_CHROME:
            return
        Tile.draw_card(screen, self._cards, x, y, width, height, "Details")
        Board.draw_facts(screen, x, y, width, height, facts, FACT_LABEL_W)

    @staticmethod
    def _type_label(p: dict) -> str:
        label = p.get("photo_type") or "\u2014"
        duration = p.get("duration_seconds")
        return f"{label} \u00b7 {duration:.0f}s" if duration else label

    @staticmethod
    def _format_date(ts) -> str:
        if ts is None:
            return "\u2014"
        try:
            dt = datetime.fromtimestamp(float(ts))
            return dt.strftime("%B %d, %Y at %I:%M %p").replace(" 0", " ")
        except (ValueError, OSError):
            return "\u2014"

    @staticmethod
    def _get_apple_photo_count() -> int:
        try:
            if not os.path.exists(PHOTOS_SQLITE_PATH):
                return 0
            conn = sqlite3.connect(f"file:{PHOTOS_SQLITE_PATH}?mode=ro", uri=True)
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM ZASSET WHERE ZKIND = 0 AND ZTRASHEDSTATE = 0 AND ZHIDDEN = 0")
                return cursor.fetchone()[0]
            finally:
                conn.close()
        except Exception:
            return 0

    def _trigger_process(self, count: int = 1):
        if self._processing:
            return
        self._processing = True
        self._process_status = ""
        self._process_status_type = ""

        def on_progress(ok, fail, total):
            self._process_status = f"Processing... {ok + fail}/{total} done"

        def run():
            try:
                processor = PhotoProcessor()
                if count == 1:
                    status, message = processor.process_next_unprocessed(album="Favorites")
                else:
                    status, message = processor.process_bulk_unprocessed(
                        count=count, album="Favorites", on_progress=on_progress)
                self._process_status_type = status
                self._process_status = message
            except Exception as e:
                self._process_status_type = "error"
                self._process_status = f"Error: {e}"
            finally:
                self._processing = False
                self._load_data()

        threading.Thread(target=run, daemon=True).start()

    def _open_in_photos(self):
        if self._photo is None:
            return
        asset_guid = self._photo.get("asset_guid", "")
        if not asset_guid:
            return
        # AppleScript only understands the machine-local UUID — resolve it from the GUID.
        try:
            conn = sqlite3.connect(f"file:{PHOTOS_SQLITE_PATH}?mode=ro", uri=True)
            try:
                row = conn.execute(
                    "SELECT ZUUID FROM ZASSET WHERE ZCLOUDASSETGUID = ?", (asset_guid,)).fetchone()
            finally:
                conn.close()
        except sqlite3.Error:
            return
        if not row or not row[0]:
            return
        subprocess.Popen(
            ["osascript", "-e", f'tell application "Photos" to spotlight media item id "{row[0]}"'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def handle_input(self, key: str) -> bool:
        if key == '\x0f':
            self._open_in_photos()
            return True
        if key == '\x10':
            self._trigger_process(count=1)
            return True
        if key == '\x02':  # Ctrl+B - bulk process
            self._trigger_process(count=200)
            return True
        if key == 'up':
            self._navigate(-1)
            return True
        if key == 'down':
            self._navigate(1)
            return True
        if key == 'pageup':
            self._navigate(-10)
            return True
        if key == 'pagedown':
            self._navigate(10)
            return True
        if key == 'home':
            self._go_to(0)
            return True
        if key == 'end':
            self._go_to(self._count - 1)
            return True
        if key == '\x13':
            self._sort_mode_idx = (self._sort_mode_idx + 1) % len(SORT_MODES)
            self._index = 0
            self._load_data()
            return True
        return False

    def on_activate(self):
        self._load_data()
