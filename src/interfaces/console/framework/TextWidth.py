"""How many terminal cells a piece of text takes, and the helpers every view measures,
clips, pads and splits with. A cell is what the screen buffer holds: most characters take
one, East Asian wide characters and emoji pictographs take two, combining marks none.

Terminals agree on those widths (they all follow Unicode's East Asian Width table) but not
on the sequences built from them — a symbol with a VS16 presentation selector (☀️), a skin
tone modifier, a ZWJ family — so `normalize` drops the selectors and modifiers and collapses
a ZWJ sequence to its first pictograph. Every helper here normalizes first, and so does the
screen's `write`, which is what keeps a caller's measurement and the terminal's cursor in
step whatever the caller did with the text.

A control character measures one cell but a terminal acts on it instead of drawing it — a
tab jumps to the next tab stop and every cell after it on the row lands in the wrong
column — so `drawable` (what `write` puts in the cells) turns each one but a newline into
a space. The helpers keep them: a log line's colour codes are measured around, not drawn.
"""
import re
from typing import Dict, List, Optional

from wcwidth import wcwidth

# VS15/VS16 presentation selectors, the five skin tone modifiers, and a ZWJ with the
# code point it joins on.
_DISAGREED = re.compile("[\ufe0e\ufe0f\U0001F3FB-\U0001F3FF]|\u200d.", re.DOTALL)
# C0 and C1 control characters and DEL, except the newline `write` interprets itself.
_CONTROL = re.compile("[\x00-\x09\x0b-\x1f\x7f-\x9f]")
# Everything below U+0300 is one cell wide (control characters keep their cell as before,
# nothing draws them on purpose); wider or zero-width characters only start above it.
_NARROW_LIMIT = "\u0300"
_widths: Dict[str, int] = {}


def normalize(text: str) -> str:
    """`text` without the sequences terminals disagree on."""
    return _DISAGREED.sub("", text) if _DISAGREED.search(text) else text


def drawable(text: str) -> str:
    """`normalize(text)` with every control character but a newline replaced by a space:
    what the screen draws, the same width as what the helpers measured."""
    text = normalize(text)
    return _CONTROL.sub(" ", text) if _CONTROL.search(text) else text


def char_width(ch: str) -> int:
    """The cells one code point takes: 0, 1 or 2."""
    if ch < _NARROW_LIMIT:
        return 1
    w = _widths.get(ch)
    if w is None:
        w = wcwidth(ch)
        w = _widths[ch] = 1 if w < 0 else w
    return w


def display_width(text: str) -> int:
    """The cells `text` takes on one row."""
    return sum(char_width(ch) for ch in normalize(text))


def cells(text: str) -> List[str]:
    """`text` as one entry per cell: a wide glyph fills its cell and leaves "" in the next,
    a combining mark joins the cell before it. `"".join(cells(t))` is `normalize(t)`."""
    out: List[str] = []
    for ch in normalize(text):
        w = char_width(ch)
        if w == 0:
            if out:
                out[-2 if out[-1] == "" and len(out) > 1 else -1] += ch
            continue
        out.append(ch)
        if w == 2:
            out.append("")
    return out


def clip(text: str, width: int) -> str:
    """The longest prefix of `text` that fits in `width` cells, never splitting a glyph."""
    if width <= 0:
        return ""
    text = normalize(text)
    used = 0
    for i, ch in enumerate(text):
        used += char_width(ch)
        if used > width:
            return text[:i]
    return text


def ljust(text: str, width: int) -> str:
    """`text` followed by the spaces that bring it to `width` cells (never cut)."""
    return text + " " * (width - display_width(text))


def rjust(text: str, width: int) -> str:
    return " " * (width - display_width(text)) + text


def fit(text: str, width: int) -> str:
    """`text` cut to `width` cells, ending in an ellipsis when it had to be cut."""
    if display_width(text) <= width:
        return text
    return clip(text, width - 3) + "..." if width > 3 else clip(text, width)


def chunks(text: str, width: int) -> List[str]:
    """`text` split into pieces of at most `width` cells, on cell boundaries."""
    text = normalize(text)
    out: List[str] = []
    start = used = 0
    for i, ch in enumerate(text):
        w = char_width(ch)
        if used + w > width and i > start:
            out.append(text[start:i])
            start, used = i, 0
        used += w
    out.append(text[start:])
    return out


def wrap(text: str, width: int, rows: Optional[int] = None) -> List[str]:
    """`text` wrapped at word boundaries into lines of at most `width` cells, a word longer
    than that broken. With `rows`, the lines past the last one are joined into it and that
    line fitted, so the text never takes more than `rows` lines."""
    text = normalize(text)
    if display_width(text) <= width:
        lines = [text]
    else:
        lines = []
        while display_width(text) > width:
            head = clip(text, width + 1)
            break_at = head.rfind(' ')
            if break_at <= 0:
                head = clip(text, width)
                lines.append(head)
                text = text[len(head):]
            else:
                lines.append(text[:break_at])
                text = text[break_at + 1:]
        if text:
            lines.append(text)
    if rows is not None and len(lines) > rows:
        lines = lines[:rows - 1] + [fit(" ".join(lines[rows - 1:]), width)] if rows > 0 else []
    return lines


def column(text: str, index: int) -> int:
    """The cell offset at which character `index` of `text` sits."""
    return display_width(text[:index])


def char_at(text: str, col: int) -> str:
    """The glyph at cell `col` of `text` — "" for the second cell of a wide glyph or beyond
    the end."""
    row = cells(text)
    return row[col] if 0 <= col < len(row) else ""
