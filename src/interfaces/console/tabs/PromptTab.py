from typing import List

from assistant.ChannelProfile import WINDOW_TEXT, WINDOW_VOICE
from common.logging import info
from .MarkdownTabView import MarkdownTabView
from .TabView import SUB_CONTENT_TOP
from interfaces.console.framework.TUIScreen import TUIScreen

RECENT_MINUTES = 30
CLEAR_TODAY = 'c'
CLEAR_RECENT = 'r'
CONFIRM_HINT = " Clear today's dialog history? y/n "
CONFIRM_RECENT_HINT = f" Clear the last {RECENT_MINUTES} minutes of dialog history? y/n "


class PromptTab(MarkdownTabView):
    """Prompt viewer tab showing the current prompt from PromptManager — the system prompt
    and the history window of one channel, the text window until `v` flips it to the voice
    window (ChannelProfile.window) — with the developer's two clears, each confirmed with
    `y`: `c` wipes today's dialog history so a test can start over, `r` only the last
    RECENT_MINUTES of it so a test can be run again with the day's earlier dialog in place."""

    def __init__(self, prompt_manager):
        super().__init__()
        self.prompt_manager = prompt_manager
        self.prompt_lines: List[str] = []
        self.last_cache_key = None
        self.window = WINDOW_TEXT
        self._confirm_clear = None   # the clear key waiting for its `y`
        self._status = ""   # the outcome of the last clear, shown in place of the key hints

    def draw(self, screen: TUIScreen, frame: int):
        """Draw the prompt viewer"""
        self._read_prompt_lines()

        start_y = SUB_CONTENT_TOP
        display_height = screen.height - start_y - 1
        self.display_height = display_height

        if self.line_wrap:
            wrap_width = screen.width - 3
            display_lines = []
            for line in self.prompt_lines:
                display_lines.extend(self.word_wrap(line.rstrip('\n\r'), wrap_width))
        else:
            display_lines = [l.rstrip('\n\r') for l in self.prompt_lines]

        total_lines = len(display_lines)
        self.total_lines = total_lines
        start_line = max(0, total_lines - display_height - self.scroll_offset)
        visible_lines = display_lines[start_line:start_line + display_height] if display_height > 0 else []
        self.seek_code_block(display_lines, start_line)

        # Reserve rightmost column for scrollbar when content overflows
        has_scrollbar = total_lines > display_height
        content_width = screen.width - 3 if has_scrollbar else screen.width - 2

        y = start_y
        for line in visible_lines:
            self.draw_formatted_line(screen, 1, y, line, content_width)
            y += 1

        screen.reset_color()
        for clear_y in range(y, start_y + display_height):
            screen.write_at(1, clear_y, " " * content_width)

        if has_scrollbar:
            self.draw_scrollbar(screen, screen.width - 1, start_y, total_lines, display_height)
        self.draw_bottom_bar(screen, self._hints(), total_lines, display_height)

    def _hints(self) -> str:
        if self._confirm_clear:
            return CONFIRM_RECENT_HINT if self._confirm_clear == CLEAR_RECENT else CONFIRM_HINT
        if self._status:
            return f" {self._status} "
        other = WINDOW_VOICE if self.window == WINDOW_TEXT else WINDOW_TEXT
        return (f" {self.wrap_hint}  ↑↓ scroll  PgUp/PgDn page  v show {other} window  "
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
                self.prompt_lines = ["No PromptManager available"]
                return

            prompt_text = self.prompt_manager.prompt
            message_history = self.prompt_manager.message_history

            # Rebuild only when the prompt or the history actually changes —
            # last_update_time bumps on every history mutation/refresh, so we
            # avoid re-rendering the whole history on every frame.
            cache_key = (hash(prompt_text), message_history.last_update_time, self.window)
            if cache_key == self.last_cache_key:
                return

            history_md = message_history.render_for_display(self.window)
            combined = (
                prompt_text.rstrip()
                + f"\n\n# ── MESSAGES (sent as turns after the prompt — the {self.window} window) ──\n\n"
                + history_md
                + "\n"
            )
            self.prompt_lines = combined.splitlines(keepends=True)
            self.last_cache_key = cache_key
        except Exception as e:
            self.prompt_lines = [f"Error reading prompt: {str(e)}"]
