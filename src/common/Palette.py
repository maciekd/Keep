"""The colours of everything Keep draws in a terminal, as plain RGB: the console's theme
(`interfaces/console/Theme.py`, which says what each one is for) and the installer's lines
(`common/install/Terminal.py`) are built from these. The module imports nothing. `install.sh` runs
before there is a checkout and carries its own copy of the accents it uses;
`tests/unit/test_install_sh.py` holds the two together."""

# Surfaces, darkest first.
PANEL_BG = (14, 16, 24)
TITLE_BG = (24, 27, 40)
SELECTED_BG = (36, 42, 68)
GRID = (38, 42, 60)

# Text, brightest first.
BRIGHT = (225, 230, 240)
TEXT = (204, 209, 222)
TITLE = (176, 183, 204)
LABEL = (140, 147, 166)
DIM = (92, 98, 118)
FAINT = (68, 73, 92)

# Accents.
CYAN = (64, 200, 255)
TEAL = (70, 215, 200)
GREEN = (70, 230, 150)
YELLOW = (255, 205, 80)
ORANGE = (255, 150, 70)
RED = (255, 90, 90)
PINK = (255, 105, 180)
PURPLE = (175, 130, 255)
BLUE = (110, 150, 255)

# The ground behind an accent.
CYAN_DIM = (14, 42, 58)
GREEN_DIM = (14, 50, 36)
YELLOW_DIM = (56, 44, 16)
RED_DIM = (58, 20, 22)
PINK_DIM = (58, 24, 44)
BLUE_DIM = (24, 34, 62)

# The deeper neighbouring hue a bar or meter fades up from.
DEEP = {
    CYAN: (40, 80, 225),
    TEAL: (0, 110, 150),
    GREEN: (0, 125, 140),
    YELLOW: (225, 95, 40),
    ORANGE: (190, 50, 70),
    RED: (150, 20, 70),
    PINK: (125, 60, 220),
    PURPLE: (70, 70, 210),
    BLUE: (70, 60, 180),
}

CUBE = (0, 95, 135, 175, 215, 255)   # the levels of a 256-colour terminal's 6×6×6 cube


def mix(a: tuple, b: tuple, t: float) -> tuple:
    """The colour `t` of the way (0..1) from `a` to `b`."""
    t = max(0.0, min(1.0, t))
    return tuple(int(x + (y - x) * t) for x, y in zip(a, b))


def nearest_256(rgb: tuple) -> int:
    """The colour of the 256-colour cube nearest to *rgb*, for a terminal without true colour."""
    r, g, b = (min(range(6), key=lambda i: abs(CUBE[i] - channel)) for channel in rgb)
    return 16 + 36 * r + 6 * g + b
