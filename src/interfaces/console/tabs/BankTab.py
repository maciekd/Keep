import os
from typing import List, Optional

from .MemoryTab import MemoryTab, counted
from .WatchListTab import span, when
from assistant import MemoryFile
from common.vault.Redactor import Redactor
from interfaces.console import Theme
from interfaces.console.widgets.Board import SummaryTile


class BankTab(MemoryTab):
    """The memory bank — people, preferences, contacts, rules — as a document in tiles: when it
    was last curated and what was noted since (shown under a band of its own, the way the
    prompt carries it), with ←→ browsing what each nightly curation made of it."""

    HAS_HISTORY = True
    TITLE = "Memory bank"

    def __init__(self, redactor: Optional[Redactor] = None):
        super().__init__(redactor)
        self.memory_file_path = MemoryFile.MEMORY_FILE

    def _live_text(self) -> str:
        if not os.path.exists(self.memory_file_path):
            return f"No memory bank found at {self.memory_file_path}\n"
        with open(self.memory_file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _notes_offset(self, text: str) -> Optional[int]:
        return MemoryFile.curated_length(self.memory_file_path)

    def _live_tiles(self) -> List[SummaryTile]:
        return [self._curation_tile(), self._notes_tile()]

    def _curation_tile(self) -> SummaryTile:
        last = self._last_snapshot()
        if last is None:
            return "Last curation", "never", Theme.COLOR_DIM, [("no curation on record", Theme.COLOR_DIM)]
        now = self.now()
        return ("Last curation", f"{span(now - last)} ago", Theme.COLOR_GREEN,
                [(when(last, now), Theme.COLOR_LABEL),
                 (f"{counted(len(self._snapshot_dates), 'night')} on record", Theme.COLOR_DIM)])

    def _notes_tile(self) -> SummaryTile:
        if not self._notes:
            return "New notes", "none", Theme.COLOR_DIM, [("as the curator left it", Theme.COLOR_DIM)]
        return ("New notes", counted(self._notes, "note"), Theme.COLOR_YELLOW,
                [("awaiting the next curation", Theme.COLOR_LABEL)])
