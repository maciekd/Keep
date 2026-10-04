"""The board a data tab is laid out as: summary tiles over a table with a tile beside it, all in the Dashboard's tiles.

Every tab that shows a store — the Vault, the Photos library, the Memory parts, the sign-in
tabs — draws the same picture: a row of summary tiles (a label, a value in the title strip
and a line of coloured segments under it), the thing itself in a table or a document, and the
selected entry in a tile at its side from `SIDE_BY_SIDE_MIN` columns, under it on a narrower
screen. What is here is that picture: the margins and the split, the summary row, the table
with its column titles, state dots, selected row and scrollbar, the facts tile of labelled
rows (a value, a flow's field or a meter), the meter rows of the Dashboard's tiles, and the
key hints on the bottom row. A tab supplies the data as the tuples below and keeps its own
`_cards` list for `Tile.tint_cards`.
"""
from typing import List, NamedTuple, Optional, Sequence, Tuple, Union

from interfaces.console import Theme
from interfaces.console.framework.TUIScreen import TUIScreen, Color
from interfaces.console.framework.TextWidth import display_width, fit
from interfaces.console.tabs.ScrollableTabView import COLOR_HINT, draw_scrollbar
from interfaces.console.widgets import Tile

LEFT = 3                            # the first tile's column
RIGHT_MARGIN = 2
GAP = 2                             # columns between tiles; one row between tile rows
SUMMARY_H = 1 + Tile.CARD_CHROME    # a summary tile: one row of segments under the strip
SUMMARY_TILE_MIN_W = 30             # a narrower screen shows the first tiles only
SIDE_BY_SIDE_MIN = 150              # screen columns from which the side tile sits beside the table
SIDE_SHARE = 36                     # percent of the width the side tile takes
SIDE_MIN_W = 50
LABEL_W = 13                        # the label column of a facts tile
METER_MAX_W = 40                    # a meter in a facts tile grows no wider than this
METER_MIN = 4                       # a meter row drops its meter below this

COLOR_PROMPT = Theme.COLOR_YELLOW   # the field a flow waits on

# A summary tile: its label, the value in its title strip, the value's colour, the row under it.
SummaryTile = Tuple[str, str, Color, List[Tuple[str, Color]]]


class Fact(NamedTuple):
    """One row of a facts tile: a dim label and a value in its colour. `waiting` is the field
    a flow waits on (label in the prompt colour, value bold); `meter` (0..1) draws a meter in
    the colour before the value, which is then bright and bold — or absent."""
    label: str
    value: str
    color: Color
    waiting: bool = False
    bold: bool = False
    meter: Optional[float] = None


class MeterRow(NamedTuple):
    """One line of a meter tile: a label, a meter and the value it shows."""
    label: str
    value: str
    fraction: float             # 0..1 of the meter filled
    color: Color
    start: float = 0.0          # 0..1 where the fill begins: the offset of a waterfall stage
    note: str = ""              # dim text after the value
    idle: bool = False          # nothing measured yet: the label dims with the rest of the row


class Meter(NamedTuple):
    """A table cell that is a meter with text after it: the fill, its colour, its width in
    cells, and the text with its colour (bold when `bold`)."""
    fraction: float
    color: Color
    width: int
    text: str = ""
    text_color: Color = Theme.COLOR_TEXT
    bold: bool = False


# A table cell: (text, colour) or (text, colour, bold), or a Meter.
Cell = Union[Tuple[str, Color], Tuple[str, Color, bool], Meter]


class Table(NamedTuple):
    """A tab's list in a tile: the title, the columns as (title, width) with the last width
    None for what is left, the rows as (dot colour, a cell per column), what to say while
    there are none, and the title strip's value — the row count unless given."""
    title: str
    columns: Sequence[Tuple[str, Optional[int]]]
    rows: Sequence[Tuple[Color, Sequence[Cell]]]
    empty: str
    value: Optional[str] = None
    value_color: Optional[Color] = None
    empty_color: Color = Theme.COLOR_DIM


def columns(available: int, count: int) -> List[Tuple[int, int]]:
    """(x, width) of `count` equal tiles across `available` columns from LEFT; the last takes
    the rounding remainder."""
    width = (available - GAP * (count - 1)) // count
    spots = [(LEFT + i * (width + GAP), width) for i in range(count)]
    spots[-1] = (spots[-1][0], available - (width + GAP) * (count - 1))
    return spots


def split(available: int, side_min_w: int = SIDE_MIN_W) -> Tuple[int, int]:
    """(table width, side tile width) of a table with a tile beside it."""
    side_w = max(side_min_w, available * SIDE_SHARE // 100)
    return available - GAP - side_w, side_w


def scrolled(top: int, count: int, room: int, selected: Optional[int] = None) -> int:
    """The first of `count` rows a `room`-row window shows: `top` kept inside the list and,
    given a `selected` row, moved the least that brings it into view."""
    if selected is not None:
        top = max(min(top, selected), selected - room + 1)
    return max(0, min(top, count - room))


def steps(waiting: int, fields: Sequence[Tuple[str, str]]) -> List[Fact]:
    """A flow's fields as facts: the answered ones in text, the one waited on bright, the
    ones to come faint."""
    colors = (Theme.COLOR_TEXT, Theme.COLOR_BRIGHT, Theme.COLOR_FAINT)
    return [Fact(label, text, colors[(i >= waiting) + (i > waiting)], i == waiting)
            for i, (label, text) in enumerate(fields)]


def draw_summary(screen: TUIScreen, cards: List[Tile.Card], y: int, available: int, tiles: Sequence[SummaryTile]):
    """The summary tiles in one row across `available` columns; a screen too narrow for them
    all shows the first ones."""
    tiles = tiles[:max(1, available // SUMMARY_TILE_MIN_W)]
    for (x, width), (label, value, color, segments) in zip(columns(available, len(tiles)), tiles):
        Tile.draw_card(screen, cards, x, y, width, SUMMARY_H, label, value, color)
        Tile.write_segments(screen, x + 2, y + Tile.CARD_TOP, width - 4, segments)


def draw_notice(screen: TUIScreen, cards: List[Tile.Card], x: int, y: int, width: int, height: int,
                title: str, notice: str, color: Color = Theme.COLOR_DIM):
    """A tile of one row saying `notice`: what a board shows in place of an empty table."""
    Tile.draw_card(screen, cards, x, y, width, min(height, Tile.CARD_CHROME + 1), title)
    screen.set_color(color)
    screen.write_at(x + 2, y + Tile.CARD_TOP, fit(notice, max(0, width - 4)))
    screen.reset_color()


def draw_table(screen: TUIScreen, cards: List[Tile.Card], x: int, y: int, width: int, height: int,
               table: Table, top: int = 0, selected: Optional[int] = None) -> int:
    """`table` in a tile: the column titles, then a row per entry — a state dot, the cells
    two columns apart — the `selected` one tinted and brought into view, a scrollbar when
    the rows outnumber the room. Returns the row the window starts at, for the tab to keep."""
    if height <= Tile.CARD_CHROME:
        return top
    count = len(table.rows)
    value = table.value if table.value is not None else (str(count) if count else "")
    Tile.draw_card(screen, cards, x, y, width, height, table.title, value, table.value_color)
    inner_w = width - 4
    header_y = y + Tile.CARD_TOP
    if not count:
        screen.set_color(table.empty_color)
        screen.write_at(x + 2, header_y, fit(table.empty, max(0, inner_w)))
        screen.reset_color()
        return top
    room = height - Tile.CARD_CHROME - 1    # a row goes to the column titles
    if room <= 0:
        return top
    top = scrolled(top, count, room, selected)
    widths = [column_w for _, column_w in table.columns]
    _write_cells(screen, x + 4, header_y, inner_w - 2, [(title, Theme.COLOR_DIM) for title, _ in table.columns], widths)
    for i, (dot, cells) in enumerate(table.rows[top:top + room]):
        row_y = header_y + 1 + i
        screen.reset_color()
        screen.set_color(dot)
        screen.write_at(x + 2, row_y, "●")
        _write_cells(screen, x + 4, row_y, inner_w - 2, cells, widths)
        if top + i == selected:
            screen.tint_row(x, row_y, width, Theme.COLOR_SELECTED_BG)
    draw_scrollbar(screen, x + width - 1, header_y + 1, count, room, count - room - top)
    screen.reset_color()
    return top


def _write_cells(screen: TUIScreen, x: int, y: int, width: int, cells: Sequence[Cell], widths: Sequence[Optional[int]]):
    """One table row: each cell in its column, two columns apart, the last column (width
    None) taking what is left of `width`."""
    remaining = width
    for cell, column_w in zip(cells, widths):
        column_w = remaining if column_w is None else min(column_w, remaining)
        if column_w <= 0:
            break
        cx = x + width - remaining
        screen.reset_color()
        if isinstance(cell, Meter):
            meter_w = min(cell.width, column_w)
            Tile.draw_meter(screen, cx, y, meter_w, min(1.0, cell.fraction), cell.color)
            screen.set_color(cell.text_color)
            if cell.bold:
                screen.bold()
            screen.write_at(cx + meter_w + 1, y, fit(cell.text, column_w - meter_w - 1))
        else:
            text, color, *bold = cell
            screen.set_color(color)
            if bold and bold[0]:
                screen.bold()
            screen.write_at(cx, y, fit(text, column_w))
        remaining -= column_w + 2
    screen.reset_color()


def draw_facts(screen: TUIScreen, x: int, y: int, width: int, height: int, facts: Sequence[Fact],
               label_w: int = LABEL_W, first_row: int = Tile.CARD_TOP) -> int:
    """`facts` as rows of a tile at (x, y), from `first_row` down to the tile's last content
    row: the label dim (prompt-coloured when waited on), the value in its colour, a meter
    before it when the fact has one. Returns the row after the last one drawn."""
    value_x = x + 2 + label_w
    value_w = max(1, width - 4 - label_w)
    last = y + height - 2
    row_y = y + first_row
    for fact in facts:
        if row_y > last:
            break
        screen.reset_color()
        screen.set_color(COLOR_PROMPT if fact.waiting else Theme.COLOR_DIM)
        screen.write_at(x + 2, row_y, fit(fact.label, label_w))
        if fact.meter is not None:
            text_w = display_width(fact.value) + 1 if fact.value else 0
            meter_w = max(1, min(METER_MAX_W, value_w - text_w))
            Tile.draw_meter(screen, value_x, row_y, meter_w, min(1.0, fact.meter), fact.color)
            if fact.value:
                screen.set_color(Theme.COLOR_BRIGHT)
                screen.bold()
                screen.write_at(value_x + meter_w + 1, row_y, fit(fact.value, value_w - meter_w - 1))
        else:
            screen.set_color(fact.color)
            if fact.waiting or fact.bold:
                screen.bold()
            screen.write_at(value_x, row_y, fit(fact.value, value_w))
        row_y += 1
    screen.reset_color()
    return row_y


def draw_footer(screen: TUIScreen, x: int, y: int, width: int, height: int, text: str, color: Color, above: int):
    """`text` on the tile's last content row, when that row is free: at or below `above`."""
    last = y + height - 2
    if text and last >= above:
        screen.set_color(color)
        screen.write_at(x + 2, last, fit(text, max(0, width - 4)))
        screen.reset_color()


def draw_meter_rows(screen: TUIScreen, x: int, y: int, width: int, height: int, rows: Sequence[MeterRow]):
    """A tile's body as aligned rows of label, meter, value and note. The meter takes every
    cell the longest label and value leave it, and goes when that is too few. Rows the
    tile's `height` has no room for are not drawn."""
    rows = rows[:max(0, height - Tile.CARD_CHROME)]
    if not rows:
        return
    label_w = max(display_width(row.label) for row in rows)
    value_w = max(display_width(row.value) for row in rows)
    note_w = max(display_width(row.note) for row in rows)
    tail_w = value_w + (note_w + 1 if note_w else 0)
    meter_len = width - 4 - label_w - 2 - tail_w
    if meter_len < METER_MIN:
        meter_len = 0
        if label_w + 1 + tail_w > width - 4:    # too narrow for the notes as well, then for the whole label
            note_w = 0
            label_w = max(4, min(label_w, width - 4 - 1 - value_w))
    meter_x = x + 2 + label_w + 1
    value_x = meter_x + (meter_len + 1 if meter_len else 0)
    value_room = max(0, x + width - 2 - value_x)

    for i, row in enumerate(rows):
        row_y = y + Tile.CARD_TOP + i
        screen.set_color(Theme.COLOR_DIM if row.idle else Theme.COLOR_LABEL)
        screen.write_at(x + 2, row_y, fit(row.label, label_w))
        Tile.draw_meter(screen, meter_x, row_y, meter_len, row.fraction, row.color, start=row.start)
        screen.set_color(row.color)
        screen.bold()
        screen.write_at(value_x, row_y, fit(row.value, value_room))
        screen.reset_color()
        if row.note and note_w:
            screen.set_color(Theme.COLOR_DIM)
            screen.write_at(value_x + value_w + 1, row_y, row.note)
    screen.reset_color()


def draw_hints(screen: TUIScreen, hints: str):
    """The key hints on the bottom row."""
    screen.reset_color()
    screen.set_color(COLOR_HINT)
    screen.write_at(LEFT, screen.height - 1, fit(hints, max(0, screen.width - LEFT - RIGHT_MARGIN)))
    screen.reset_color()
