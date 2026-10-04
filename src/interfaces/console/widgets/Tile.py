"""The flat tile and the one-row meter the Dashboard is built from, for any tab that wants the same look.

A tile is a title strip over a tinted body. Its content is written without a background and
`tint_cards` grounds every tile once the frame is drawn, so a caller collects its tiles in a
list as it draws them."""
from typing import List, Optional, Tuple

from interfaces.console import Theme
from interfaces.console.framework.TUIScreen import TUIScreen, Color
from interfaces.console.framework.TextWidth import display_width, fit

CARD_TOP = 2                # rows from a tile's title strip to its content: one blank row of air
CARD_CHROME = CARD_TOP + 1  # a tile's rows that are not content: title strip, the blank row, a closing row

METER_BLOCKS = " ▏▎▍▌▋▊▉█"  # a cell filled from its left edge, in eighths
METER_MARKER = "▌"

Card = Tuple[int, int, int, int]


def draw_card(screen: TUIScreen, cards: List[Card], x: int, y: int, width: int, height: int, label: str,
              value: str = "", value_color: Optional[Color] = None):
    """A tile: a title strip carrying the label and, when it fits, a bold `value` at its
    right end, over a body whose content starts at (x + 2, y + CARD_TOP)."""
    cards.append((x, y, width, height))
    screen.set_color(Theme.COLOR_TITLE)
    screen.bold()
    screen.write_at(x + 2, y, fit(label, width - 4))
    screen.reset_color()
    if value and display_width(label) + display_width(value) + 6 <= width:
        screen.set_color(value_color or Theme.COLOR_BRIGHT)
        screen.bold()
        screen.write_at(x + width - display_width(value) - 2, y, value)
        screen.reset_color()


def write_segments(screen: TUIScreen, x: int, y: int, width: int, segments: List[Tuple[str, Color]]):
    """Coloured pieces of text on one row, a faint dot between them, cut at `width`."""
    used = 0
    for i, (text, color) in enumerate(segments):
        if i and used + 3 + display_width(text) > width:
            break
        if i:
            screen.set_color(Theme.COLOR_FAINT)
            screen.write_at(x + used, y, " · ")
            used += 3
        text = fit(text, width - used)
        screen.set_color(color)
        screen.write_at(x + used, y, text)
        used += display_width(text)
    screen.reset_color()


def tint_cards(screen: TUIScreen, cards: List[Card]):
    for x, y, width, height in cards:
        screen.tint_row(x, y, width, Theme.COLOR_TITLE_BG)
        for row in range(1, height):
            screen.tint_row(x, y + row, width, Theme.COLOR_PANEL_BG)


def draw_meter(screen: TUIScreen, x: int, y: int, length: int, fraction: float, fg: Color,
               start: float = 0.0, track: Optional[Color] = None,
               marker: Optional[float] = None, marker_color: Optional[Color] = None):
    """A one-row bar, `length` cells on the dim `track`: `fraction` of it (0..1) filled
    from `start`, to an eighth of a cell. The shading belongs to the track, not the fill:
    the colour's deep companion at its left end, the colour at its right, so a cell has
    the same shade whatever the value. Anything above zero shows at least an
    eighth. `marker` (0..1) is a second value drawn as a notch on top.
    A full cell carries its colour as background too, like the chart columns. A fill that
    starts inside a cell is drawn inverted — the track's block over the fill's ground —
    since the block glyphs only grow from the left."""
    if length <= 0:
        return
    start = max(0.0, min(1.0, start))
    lo = start * length
    hi = lo + max(0.0, min(1.0 - start, fraction)) * length
    if fraction > 0 and hi - lo < 1 / 8:
        lo = min(lo, length - 1 / 8)
        hi = lo + 1 / 8
    track = track or Theme.mix(Theme.COLOR_PANEL_BG, fg, 0.22)
    deep = Theme.deep(fg)
    marker_idx = None if marker is None else max(0, min(length - 1, int(max(0.0, min(1.0, marker)) * length)))

    span = max(1, length - 1)
    for i in range(length):
        begin = max(lo, i) - i
        end = min(hi, i + 1) - i
        color = Theme.mix(deep, fg, i / span)
        if i == marker_idx:
            screen.set_color(marker_color or Theme.COLOR_BRIGHT)
            screen.set_bg_color(track)
            char = METER_MARKER
        elif end <= begin:
            screen.set_color(track)
            screen.set_bg_color(track)
            char = " "
        elif begin > 0:
            screen.set_color(track)
            screen.set_bg_color(color)
            char = METER_BLOCKS[min(7, round(begin * 8))]
        elif end >= 1:
            screen.set_color(color)
            screen.set_bg_color(color)
            char = METER_BLOCKS[8]
        else:
            screen.set_color(color)
            screen.set_bg_color(track)
            char = METER_BLOCKS[max(1, int(end * 8))]
        screen.write_at(x + i, y, char)
    screen.reset_color()
