import re
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

from .MemoryTab import MemoryTab
from .WatchListTab import span, when
from assistant.Procedures import Procedures
from interfaces.console import Theme
from interfaces.console.framework.TUIScreen import TUIScreen, Color
from interfaces.console.widgets.Board import SummaryTile

# A file the LLM has not loaded for this many days is counted as unused.
IDLE_DAYS = 30


def age(delta: timedelta) -> str:
    """A time since, in at most four cells: "now", "42m", "5h", "3d", "6w", "4mo", "2y"."""
    minutes = int(delta.total_seconds() // 60)
    if minutes < 1:
        return "now"
    if minutes < 60:
        return f"{minutes}m"
    if minutes < 24 * 60:
        return f"{minutes // 60}h"
    days = minutes // (24 * 60)
    if days < 14:
        return f"{days}d"
    if days < 60:
        return f"{days // 7}w"
    return f"{days // 30}mo" if days < 365 else f"{days // 365}y"


def age_color(delta: timedelta) -> Color:
    """Fresher is brighter: today green, this week in text, this month a label, older dim."""
    if delta < timedelta(days=1):
        return Theme.COLOR_GREEN
    if delta < timedelta(days=7):
        return Theme.COLOR_TEXT
    return Theme.COLOR_LABEL if delta < timedelta(days=IDLE_DAYS) else Theme.COLOR_DIM


class ProceduresTab(MemoryTab):
    """Every procedure in full, each under its name. The outline's Used column and the Last
    used and Unused tiles say when the LLM last loaded each one (`MemoryDocs.last_used`)."""

    TITLE = "Procedures"
    SECTIONS_LABEL = "Procedures"
    DOCS = Procedures
    COLUMN = "Used"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._used: Dict[str, datetime] = {}

    def draw(self, screen: TUIScreen, frame: int):
        self._used = self.DOCS.last_used()
        super().draw(screen, frame)

    def _live_text(self) -> str:
        names = self.DOCS.names()
        if not names:
            return self.DOCS.EMPTY + "\n"
        parts = []
        for name in names:
            text = (self.DOCS.read(name) or "").strip()
            # The tab's "## name" replaces the file's own "# name"; every heading in its body
            # sits below that.
            body = text.removeprefix(f"# {name}\n").removeprefix("# ")
            body = re.sub(r"(?m)^(#+)(?= )", lambda m: "#" * max(len(m.group(1)) + 1, 3), body)
            parts += [f"## {name}", "", self._body(body).strip(), ""]
        return "\n".join(parts) + "\n"

    def _body(self, body: str) -> str:
        # The curator's "## Notes" is shown lower-case, like the procedure names around it; the
        # file keeps its spelling.
        return re.sub(rf"(?m)^#{re.escape(Procedures.NOTES_HEADING)}$", f"#{Procedures.NOTES_HEADING.lower()}", body)

    # --- When each one was last used ---

    def _column_cell(self, section: str) -> Tuple[str, Color]:
        used = self._used.get(section)
        if used is None:
            return "never", Theme.COLOR_FAINT
        delta = max(timedelta(0), self.now() - used)
        return age(delta), age_color(delta)

    def _live_tiles(self) -> List[SummaryTile]:
        names = self.DOCS.names()
        used = {name: self._used[name] for name in names if name in self._used}
        now = self.now()
        return [self._last_used_tile(used, now), self._unused_tile(names, used, now)]

    def _last_used_tile(self, used: Dict[str, datetime], now: datetime) -> SummaryTile:
        if not used:
            return "Last used", "–", Theme.COLOR_DIM, [(f"no {self.DOCS.NOUN} loaded yet", Theme.COLOR_DIM)]
        name, moment = max(used.items(), key=lambda pair: pair[1])
        return ("Last used", f"{span(now - moment)} ago", Theme.COLOR_CYAN,
                [(name, Theme.COLOR_LABEL), (when(moment, now), Theme.COLOR_DIM)])

    def _unused_tile(self, names: List[str], used: Dict[str, datetime], now: datetime) -> SummaryTile:
        if not names:
            return "Unused", "–", Theme.COLOR_DIM, [(self.DOCS.EMPTY, Theme.COLOR_DIM)]
        idle = [name for name in names if name not in used or now - used[name] >= timedelta(days=IDLE_DAYS)]
        if not idle:
            return "Unused", "none", Theme.COLOR_GREEN, [(f"all loaded in the last {IDLE_DAYS} days", Theme.COLOR_LABEL)]
        never = len(names) - len(used)
        segments = [(f"not loaded in {IDLE_DAYS} days", Theme.COLOR_LABEL)]
        if never:
            segments.append((f"{never} never", Theme.COLOR_DIM))
        return "Unused", f"{len(idle)} of {len(names)}", Theme.COLOR_ORANGE, segments
