from typing import List, Optional

from assistant.ChannelProfile import WINDOW_TEXT, WINDOW_VOICE
from common.logging import info
from .ScrollableTabView import ScrollableTabView
from .TabView import SUB_CONTENT_TOP
from interfaces.console.framework.TUIScreen import TUIScreen
from interfaces.console.widgets import Markdown

RECENT_MINUTES = 30
CLEAR_TODAY = 'c'
CLEAR_RECENT = 'r'
CONFIRM_HINT = " Clear today's dialog history? y/n "
CONFIRM_RECENT_HINT = f" Clear the last {RECENT_MINUTES} minutes of dialog history? y/n "
TEXT_X = 2   # the document's left edge, a column of air inside the tab
MESSAGES_BAND = "MESSAGES"


class PromptTab(ScrollableTabView):
    """Prompt viewer tab showing the current prompt from PromptManager — the system prompt
    and the history window of one channel, the text window until `v` flips it to the voice
    window (ChannelProfile.window) — laid out as a document by the Markdown widget, the
    history opened by a band of its own; with the developer's two clears, each confirmed with
    `y`: `c` wipes today's dialog history so a test can start over, `r` only the last
    RECENT_MINUTES of it so a test can be run again with the day's earlier dialog in place."""

    def __init__(self, prompt_manager):
        super().__init__()
        self.prompt_manager = prompt_manager
        self.prompt_lines: List[str] = []
        self.bands = {}
        self.last_cache_key = None
        self._version = 0   # bumped whenever prompt_lines change
        self._layout_key = None
        self._doc: Optional[Markdown.Document] = None
        self.window = WINDOW_TEXT
        self._confirm_clear = None   # the clear key waiting for its `y`
        self._status = ""   # the outcome of the last clear, shown in place of the key hints

    def draw(self, screen: TUIScreen, frame: int):
        """Draw the prompt viewer"""
        self._read_prompt_lines()

        start_y = SUB_CONTENT_TOP
        display_height = screen.height - start_y - 1
        self.display_height = display_height
        text_w = screen.width - TEXT_X - 3   # a column for the scrollbar and one of air before it

        rows = self._document(text_w).rows
        total_lines = len(rows)
        self.total_lines = total_lines
        self.scroll_offset = min(self.scroll_offset, max(0, total_lines - display_height))
        start_line = max(0, total_lines - display_height - self.scroll_offset)

        for k in range(max(0, display_height)):
            y = start_y + k
            screen.reset_color()
            screen.write_at(1, y, " " * (screen.width - 2))
            if start_line + k < total_lines:
                Markdown.draw_row(screen, TEXT_X, y, text_w, rows[start_line + k])

        if total_lines > display_height:
            self.draw_scrollbar(screen, screen.width - 1, start_y, total_lines, display_height)
        self.draw_bottom_bar(screen, self._hints(), total_lines, display_height)

    def _document(self, width: int) -> Markdown.Document:
        """The prompt laid out `width` cells wide, kept until the prompt or the width changes."""
        key = (self._version, width)
        if key != self._layout_key or self._doc is None:
            self._doc = Markdown.layout([line.rstrip('\n\r') for line in self.prompt_lines], width, bands=self.bands)
            self._layout_key = key
        return self._doc

    def _hints(self) -> str:
        if self._confirm_clear:
            return CONFIRM_RECENT_HINT if self._confirm_clear == CLEAR_RECENT else CONFIRM_HINT
        if self._status:
            return f" {self._status} "
        other = WINDOW_VOICE if self.window == WINDOW_TEXT else WINDOW_TEXT
        return (f" ↑↓ scroll  PgUp/PgDn page  v show {other} window  "
                f"{CLEAR_TODAY} clear  {CLEAR_RECENT} clear {RECENT_MINUTES}m ")

    def on_deactivate(self):
        self._confirm_clear = None

    def handle_input(self, key: str) -> bool:
        """Scrolling, `v` to show the other channel's window, plus `c` → `y` to wipe today's
        dialog history and `r` → `y` to wipe its last RECENT_MINUTES."""
        if self._confirm_clear:
            recent = self._confirm_clear == CLEAR_RECENT
            self._confirm_clear = None
            if key == 'y':
                self._status = self._clear(recent)
            else:
                info(f"PromptTab: clearing {self._span(recent)} cancelled")
                self._status = "Clear cancelled"
            return True
        self._status = ""
        if key in (CLEAR_TODAY, CLEAR_RECENT):
            self._confirm_clear = key
            return True
        if key == 'v':
            self.window = WINDOW_VOICE if self.window == WINDOW_TEXT else WINDOW_TEXT
            info(f"PromptTab: showing the {self.window} history window")
            return True
        if key == 'w':   # the document always wraps
            return False
        return self.handle_scroll_input(key, self.scroll_total(len(self.prompt_lines)))

    @staticmethod
    def _span(recent: bool) -> str:
        return f"the last {RECENT_MINUTES} minutes" if recent else "today"

    def _clear(self, recent: bool) -> str:
        if not self.prompt_manager:
            return "No PromptManager available"
        history = self.prompt_manager.message_history
        try:
            deleted = history.wipe_recent(RECENT_MINUTES) if recent else history.wipe_today()
        except Exception as e:
            info(f"PromptTab: clearing the history for {self._span(recent)} failed: {e}")
            return f"Clear failed: {e}"
        self.scroll_offset = 0
        info(f"PromptTab: wiped {deleted} history entries for {self._span(recent)}")
        return f"Wiped {deleted} history entries for {self._span(recent)}"

    def _read_prompt_lines(self):
        """Read the system prompt plus the conversation history that gets appended
        after it as message turns at request time."""
        try:
            if not self.prompt_manager:
                self._show(["No PromptManager available"])
                return

            prompt_text = self.prompt_manager.prompt
            message_history = self.prompt_manager.message_history

            # Rebuild only when the prompt or the history actually changes —
            # last_update_time bumps on every history mutation/refresh, so we
            # avoid re-rendering the whole history on every frame.
            cache_key = (hash(prompt_text), message_history.last_update_time, self.window)
            if cache_key == self.last_cache_key:
                return

            prompt = prompt_text.rstrip().splitlines()
            history = message_history.render_for_display(self.window).splitlines()
            self.prompt_lines = prompt + [""] + history
            self.bands = {len(prompt) + 1: (MESSAGES_BAND, f"sent as turns after the prompt · the {self.window} window")}
            self.last_cache_key = cache_key
            self._version += 1
        except Exception as e:
            self._show([f"Error reading prompt: {str(e)}"])

    def _show(self, lines: List[str]):
        if lines != self.prompt_lines:
            self.prompt_lines, self.bands, self.last_cache_key = lines, {}, None
            self._version += 1
