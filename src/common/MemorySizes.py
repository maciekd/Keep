"""The sizes of the long-term memory's folders — procedures, memos and the watch list — and the
lines of a curation log's header that keep them, a night at a time, for the Dashboard's chart.

The bank's own size is the header's `New size` line, which the curator has always written.
"""
import re
from pathlib import Path

from common.logging import trace

BANK = "bank"
# A folder of memory/ → its label in the header.
FOLDERS = {"procedures": "Procedures", "memos": "Memos", "watchlist": "Watch list"}
PARTS = (BANK, *FOLDERS)    # the chart's stack, the bottom layer first


def folder_chars(folder: Path) -> int:
    """The characters of the folder's files; a folder that is not there holds none."""
    total = 0
    for path in folder.glob("*.md"):
        try:
            total += len(path.read_text(encoding="utf-8", errors="replace"))
        except OSError as e:
            trace(f"MemorySizes: cannot read {path}: {e}")
    return total


def header_lines(folders: dict[str, Path]) -> list[str]:
    """The header lines for the folders as they stand now, `folders` naming where each one is."""
    return [f"- {label}: {folder_chars(folders[part])} chars" for part, label in FOLDERS.items() if part in folders]


def parse(header: str) -> dict[str, int]:
    """The folder sizes a header holds, in characters; a log from before the lines has none."""
    found = {}
    for part, label in FOLDERS.items():
        match = re.search(rf"^- {label}: (\d+) chars\s*$", header, re.MULTILINE)
        if match:
            found[part] = int(match.group(1))
    return found
