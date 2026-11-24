"""The console's one palette and its margins: every tab and widget takes its colours from
here, naming its own roles (a header, a selected row, a diff gutter) in terms of these, so
the whole TUI reads as one piece. A new colour belongs here, not in the tab that needs it.

Three groups: the surfaces things sit on, the text levels from bright to faint, and the
accents — each with a deep companion a bar or column shades up from and a dim tint for the
ground behind it."""
from common import Palette
from interfaces.console.framework.TUIScreen import Color

LEFT_MARGIN = 5
TOP_MARGIN = 5   # the Home chat starts one row under CONTENT_TOP, for air below the tab bar


def mix(a: Color, b: Color, t: float) -> Color:
    """The colour `t` of the way (0..1) from `a` to `b`."""
    t = max(0.0, min(1.0, t))
    return Color(int(a.r + (b.r - a.r) * t), int(a.g + (b.g - a.g) * t), int(a.b + (b.b - a.b) * t))


# Surfaces, darkest first; the screen's own ground is TUIScreen.BACKGROUND, black.
COLOR_PANEL_BG = Color(*Palette.PANEL_BG)       # a tile's body, a table's striped row
COLOR_TITLE_BG = Color(*Palette.TITLE_BG)       # a tile's title strip, a table header, an unselected tab
COLOR_SELECTED_BG = Color(*Palette.SELECTED_BG)    # the row or item under the cursor
COLOR_GRID = Color(*Palette.GRID)           # chart rules, bar-row fill, a scrollbar's track

# Text, brightest first.
COLOR_BRIGHT = Color(*Palette.BRIGHT)      # what must stand out: a selected tab, bold
COLOR_TEXT = Color(*Palette.TEXT)        # body text
COLOR_TITLE = Color(*Palette.TITLE)       # titles and column headers
COLOR_LABEL = Color(*Palette.LABEL)       # labels, secondary text
COLOR_DIM = Color(*Palette.DIM)           # hints, units, what is switched off
COLOR_FAINT = Color(*Palette.FAINT)          # rules and asides that should barely register

# Accents.
COLOR_CYAN = Color(*Palette.CYAN)
COLOR_TEAL = Color(*Palette.TEAL)
COLOR_GREEN = Color(*Palette.GREEN)
COLOR_YELLOW = Color(*Palette.YELLOW)
COLOR_ORANGE = Color(*Palette.ORANGE)
COLOR_RED = Color(*Palette.RED)
COLOR_PINK = Color(*Palette.PINK)
COLOR_PURPLE = Color(*Palette.PURPLE)
COLOR_BLUE = Color(*Palette.BLUE)

# The ground behind an accent: a meter's track, a plate under text of that colour.
COLOR_CYAN_DIM = Color(*Palette.CYAN_DIM)
COLOR_GREEN_DIM = Color(*Palette.GREEN_DIM)
COLOR_YELLOW_DIM = Color(*Palette.YELLOW_DIM)
COLOR_RED_DIM = Color(*Palette.RED_DIM)
COLOR_PINK_DIM = Color(*Palette.PINK_DIM)
COLOR_BLUE_DIM = Color(*Palette.BLUE_DIM)

# The colour a bar or meter starts from before it reaches its own: a neighbouring, deeper
# hue, so a fill reads as a gradient and not as a slab.
DEEP = {Color(*accent): Color(*companion) for accent, companion in Palette.DEEP.items()}


def deep(color: Color) -> Color:
    """`color`'s deep companion; a colour outside the accents fades up from the panel ground."""
    return DEEP.get(color) or mix(COLOR_PANEL_BG, color, 0.35)


COLOR_SECRET = COLOR_ORANGE   # a vaulted value wherever it shows: revealed in Vault, masked elsewhere
