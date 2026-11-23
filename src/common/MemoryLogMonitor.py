import re
import threading
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from common import MemorySizes
from common.logging import fine, trace


class MemoryLogMonitor:
    """Scans log/memory snapshots on a background thread to chart memory size per day in the
    DashboardTab, and counts the words of the live memory bank for the chart's label.

    A day's reading is a size in KB per part of the memory (`MemorySizes.PARTS`): the bank's
    from every snapshot, the folders' from the snapshots written since the curator records
    them — an older day has the bank alone. Today's reading is the memory as it stands."""

    LOG_DIR = Path("log/memory")
    MEMORY_FILE = Path("memory/bank.md")
    FILENAME_RE = re.compile(r"^(\d{8})_\d{6}_[0-9a-f]+_memory\.md$")
    NEW_SIZE_RE = re.compile(r"^- New size: (\d+) chars\s*$", re.MULTILINE)
    HEADER_READ_BYTES = 2048

    def __init__(self, dashboard_tab, fetch_interval: float = 300.0):
        self._dashboard_tab = dashboard_tab
        self._fetch_interval = fetch_interval
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if self._thread and self._thread.is_alive() and not self._stop_event.is_set():
            return
        # A stopped loop still mid-sample keeps the event it was started with and ends on it;
        # this start gets a fresh one. With one event shared, coming back to the tab before
        # the old loop had ended started nothing and left the monitor dead while the tab showed.
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._monitor_loop, args=(self._stop_event,), daemon=True,
                                        name="MemoryLogMonitor")
        self._thread.start()
        fine("MemoryLogMonitor started")

    def stop(self):
        """Returns at once — a tab switch calls it on the TUI's input thread, and a join here
        held every keypress back while the loop was mid-sample. The loop ends by itself at
        its next look at its event."""
        self._stop_event.set()

    def _monitor_loop(self, stop_event: threading.Event):
        while not stop_event.is_set():
            try:
                self._scan()
            except Exception as e:
                trace(f"MemoryLogMonitor error: {e}")
                self._dashboard_tab.memory_log_error = str(e)
            stop_event.wait(self._fetch_interval)

    def _scan(self):
        bank = self._read_bank()
        self._dashboard_tab.memory_word_count = self._count_words(bank)
        if not self.LOG_DIR.exists():
            self._dashboard_tab.memory_log_sizes = []
            self._dashboard_tab.memory_log_error = f"Directory not found: {self.LOG_DIR}"
            trace(f"MemoryLogMonitor: {self.LOG_DIR} does not exist")
            return

        # Per day and part: keep the largest reading (handles days with multiple snapshots)
        per_day: dict = {}
        skipped = 0
        for entry in self.LOG_DIR.iterdir():
            if not entry.is_file():
                continue
            match = self.FILENAME_RE.match(entry.name)
            if not match:
                skipped += 1
                fine(f"MemoryLogMonitor: skipping malformed name {entry.name}")
                continue
            try:
                day = datetime.strptime(match.group(1), "%Y%m%d").date()
            except ValueError:
                skipped += 1
                trace(f"MemoryLogMonitor: skipping unparseable date in {entry.name}")
                continue
            try:
                with entry.open("r", encoding="utf-8", errors="replace") as f:
                    header = f.read(self.HEADER_READ_BYTES)
            except OSError as e:
                skipped += 1
                trace(f"MemoryLogMonitor: cannot read {entry.name}: {e}")
                continue
            size_match = self.NEW_SIZE_RE.search(header)
            if not size_match:
                skipped += 1
                trace(f"MemoryLogMonitor: no 'New size' header in {entry.name}")
                continue
            reading = {MemorySizes.BANK: int(size_match.group(1)), **MemorySizes.parse(header)}
            sizes = per_day.setdefault(day, {})
            for part, chars in reading.items():
                sizes[part] = max(sizes.get(part, 0.0), chars / 1024.0)

        if bank is not None:
            per_day[date.today()] = self._live_sizes(bank)
        sizes = sorted(per_day.items(), key=lambda kv: kv[0])
        fine(f"MemoryLogMonitor: {len(sizes)} day(s) of memory sizes, {skipped} file(s) skipped")
        self._dashboard_tab.memory_log_sizes = sizes
        self._dashboard_tab.memory_log_error = ""

    def _live_sizes(self, bank: str) -> dict:
        """The memory as it stands, in KB per part: the bank and the folders beside it."""
        folder = self.MEMORY_FILE.parent
        return {MemorySizes.BANK: len(bank) / 1024.0,
                **{part: MemorySizes.folder_chars(folder / part) / 1024.0 for part in MemorySizes.FOLDERS}}

    def _read_bank(self) -> Optional[str]:
        try:
            return self.MEMORY_FILE.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            trace(f"MemoryLogMonitor: cannot read {self.MEMORY_FILE}: {e}")
            return None

    def _count_words(self, text: Optional[str]) -> int:
        """Words of the live bank; markdown markers (`#`, `-`, `---`) are not words."""
        if text is None:
            return 0
        words = sum(1 for token in text.split() if any(c.isalnum() for c in token))
        fine(f"MemoryLogMonitor: {self.MEMORY_FILE} holds {words} words")
        return words
