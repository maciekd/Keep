from typing import Optional, Tuple

from .TabView import TabView
from interfaces.console import Theme
from interfaces.console.framework.TUIScreen import TUIScreen
from interfaces.console.framework.TextWidth import wrap


COLOR_SCROLLBAR_THUMB = Theme.COLOR_LABEL
COLOR_SCROLLBAR_TRACK = Theme.COLOR_GRID
COLOR_HINT = Theme.COLOR_DIM   # the key hints and the scroll position on the bottom row


def scrollbar_thumb(total_lines: int, display_height: int, scroll_offset: int) -> Tuple[int, int]:
    """(top row, rows) of the thumb for a `display_height`-row window over `total_lines`
    lines, scrolled `scroll_offset` lines up from the bottom."""
    max_scroll = total_lines - display_height
    thumb_size = max(1, round(display_height * display_height / total_lines))
    scroll_fraction = scroll_offset / max_scroll if max_scroll > 0 else 0
    thumb_top = round((1.0 - scroll_fraction) * (display_height - thumb_size))
    return max(0, min(display_height - thumb_size, thumb_top)), thumb_size


def draw_scrollbar(screen: TUIScreen, x: int, start_y: int, total_lines: int, display_height: int,
                   scroll_offset: int):
    """Draw a vertical scrollbar track with thumb. `scroll_offset` counts lines from the
    bottom (0 = scrolled to the end). Only draws when content overflows the visible area."""
    if total_lines <= display_height or display_height < 2:
        return
    thumb_top, thumb_size = scrollbar_thumb(total_lines, display_height, scroll_offset)
    for i in range(display_height):
        screen.reset_color()
        y = start_y + i
        if thumb_top <= i < thumb_top + thumb_size:
            screen.set_color(COLOR_SCROLLBAR_THUMB)
            screen.write_at(x, y, '┃')
        else:
            screen.set_color(COLOR_SCROLLBAR_TRACK)
            screen.write_at(x, y, '│')


class ScrollableTabView(TabView):
    """Base class for tab views with scrolling and line-wrap support."""

    def __init__(self):
        self.scroll_offset = 0
        self.display_height = 20
        self.line_wrap = False
        # Display lines (wrapped) of the last frame; None until the tab has drawn once
        self.total_lines: Optional[int] = None

    def scroll_total(self, unwrapped: int) -> int:
        """The line count the scroll keys work against: the last frame's display lines, or
        the raw count before anything has been drawn."""
        return self.total_lines if self.total_lines is not None else unwrapped

    @property
    def wrap_hint(self) -> str:
        return "w WRAP" if self.line_wrap else "w wrap"

    def handle_scroll_input(self, key: str, total_lines: int) -> bool:
        """Handle up/down/pageup/pagedown scroll keys. Returns True if handled."""
        max_scroll = max(0, total_lines - self.display_height)
        if key == 'up':
            self.scroll_offset = min(self.scroll_offset + 1, max_scroll)
            return True
        elif key == 'down':
            self.scroll_offset = max(0, self.scroll_offset - 1)
            return True
        elif key == 'pageup':
            self.scroll_offset = min(self.scroll_offset + self.display_height, max_scroll)
            return True
        elif key == 'pagedown':
            self.scroll_offset = max(0, self.scroll_offset - self.display_height)
            return True
        elif key == 'w':
            self.line_wrap = not self.line_wrap
            self.scroll_offset = 0
            return True
        return False

    def draw_scroll_indicator(self, screen: TUIScreen):
        """The scroll position, bottom-right."""
        screen.reset_color()
        screen.set_color(COLOR_HINT)
        scroll_info = f" {self.scroll_offset} lines from bottom "
        screen.write_at(screen.width - len(scroll_info) - 1, screen.height - 1, scroll_info)

    def draw_bottom_bar(self, screen: TUIScreen, hints: str, total_lines: int, display_height: int):
        """Key hints bottom-left, and the scroll position bottom-right when the content
        overflows the window."""
        screen.reset_color()
        screen.set_color(COLOR_HINT)
        screen.write_at(1, screen.height - 1, hints)
        if total_lines > display_height:
            self.draw_scroll_indicator(screen)

    def draw_scrollbar(self, screen: TUIScreen, x: int, start_y: int,
                       total_lines: int, display_height: int):
        draw_scrollbar(screen, x, start_y, total_lines, display_height, self.scroll_offset)

    @staticmethod
    def word_wrap(line: str, width: int) -> list:
        """Wrap line at word boundaries (`width` in cells). Hard-breaks words longer than width."""
        return wrap(line, width)
