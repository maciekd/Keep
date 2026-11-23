"""The Dashboard's chart plots drawn as pixels, for a terminal that shows pictures: columns
and stacked bars on the panel ground, each shaded from its colour's deep companion up to the
colour, with rounded anti-aliased tops, dashed rules and a soft glow around the newest column;
and filled line charts (`area`) — a line over a fill fading to the ground, a rolling average
traced over it as a smooth glowing curve and a glowing dot on the newest point — which
`stacked` lays on one another as bands.

A plot is `cols` × `rows` cells at `Pictures.CELL_PX` pixels a cell, the size the screen sends
a picture at, so the PNG goes out unscaled. The value scale runs from the centre of the bottom
cell row to the centre of the top one — the rows a tile's y-axis labels stand on — which also
leaves half a row of air above full scale for a rounded top and its glow. A bar is given in
cells (`Bar.x0`/`x1`) and its segments as fractions of full scale; everything else is pixels.

`ChartPicture` is one chart's `Picture` kept across frames: it draws again only when what the
chart shows has changed, at most every `REFRESH_SECONDS` (unless the box changed, which
draws at once), and replaces the picture's image in place, so the terminal holds one image
per chart however long the tab is up."""
import time
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from common.logging import trace
from interfaces.console import Theme
from interfaces.console.framework.Pictures import CELL_PX, Picture
from interfaces.console.framework.TUIScreen import Color

BAR_PAD_PX = 2          # air either side of a column inside its cells
CAP_RADIUS_PX = 5       # the rounding of a bar's top corners
CAP_SAMPLES = 4         # sub-samples per pixel side the rounded top is anti-aliased with
DASH_PX = 3             # a dashed rule's dash, and the gap after it
SEPARATOR_PX = 2        # the panel ground showing between stacked segments
GLOW_RADIUS_PX = 7
GLOW_STRENGTH = 0.5
BASE_MIX = 0.8          # a colour's bottom shade: its deep companion this far out of the panel ground

SUPERSAMPLE = 4         # an area chart's lines and fill are drawn this many times larger, then reduced
LINE_PX = 2.5           # the value line's width
AVERAGE_PX = 2.0        # the rolling average's width ...
AVERAGE_ALPHA = 0.9     # ... its opacity ...
AVERAGE_GLOW_PX = 3     # ... and the blur of the soft glow around it
AVERAGE_GLOW = 0.5
CURVE_STEPS = 12        # points a smoothed segment is drawn through
FILL_TOP_ALPHA = 0.42   # the fill under the line: this opaque at the peak's height ...
FILL_BOTTOM_ALPHA = 0.04    # ... fading to this at the base
DOT_RADIUS_PX = 4       # the newest point
EDGE_PX = 6             # air left and right of the line, so the dot and its glow fit

Rule = Tuple[float, Color]    # (fraction of full scale, colour) of a dashed line across the plot


class Segment(NamedTuple):
    start: float        # fractions of full scale, bottom and top
    end: float
    color: Color


class Bar(NamedTuple):
    x0: int             # the cells it stands on, x1 exclusive
    x1: int
    segments: Sequence[Segment]     # bottom to top


def columns(values: Sequence[float], colors: Sequence[Color], max_value: float, cols: int, rows: int,
            rules: Sequence[Rule] = (), shade_min_rows: int = 0, glow: bool = True) -> Image.Image:
    """One column a cell for each value, the newest at the right and the plot's left empty
    while there are fewer values than cells. A column shades up to its colour at the height
    of the tallest column on show, never faster than the whole range over `shade_min_rows`
    rows, so a quiet chart keeps its colour."""
    values, colors = list(values)[-cols:], list(colors)[-cols:]
    offset = cols - len(values)
    scale = max_value if max_value > 0 else 1.0
    bars = [Bar(offset + i, offset + i + 1, [Segment(0.0, value / scale, color)])
            for i, (value, color) in enumerate(zip(values, colors)) if value > 0]
    peak = max((bar.segments[-1].end for bar in bars), default=0.0)
    span = max(peak, shade_min_rows / max(1, rows - 1))
    newest = bars[-1] if glow and bars and bars[-1].x1 == cols else None
    return render(bars, cols, rows, rules, shade_top=peak, shade_span=span, glow_bar=newest)


def rolling_mean(values: Sequence[Optional[float]], window: int) -> List[Optional[float]]:
    """The trailing mean of the last `window` values at each point, skipping the missing
    (None) ones; None where the window holds no value at all."""
    out: List[Optional[float]] = []
    for i in range(len(values)):
        known = [v for v in values[max(0, i - window + 1):i + 1] if v is not None]
        out.append(sum(known) / len(known) if known else None)
    return out


def area(values: Sequence[Optional[float]], color: Color, max_value: float, cols: int, rows: int,
         rules: Sequence[Rule] = (), average: Sequence[Optional[float]] = (), average_color: Optional[Color] = None,
         slots: int = 0) -> Image.Image:
    """A filled line chart: the values as a line across the plot, oldest at the left, the
    ground under it filled in `color` fading towards the base, `average` (aligned with
    `values`) traced over it as a smooth curve in `average_color`, and the newest point a dot
    with a soft glow. The values are spread over `slots` evenly spaced points (at least one per
    value), the newest at the right edge, so a short history fills the plot from the right.
    A missing value (None) breaks the line and the fill."""
    values, average = list(values), list(average)
    if len(values) == 1 and slots <= 1:     # one reading: a level across the plot rather than a lone dot
        values, average = values * 2, average * 2
    plot = _Plot(cols, rows, rules, len(values), slots, max_value)
    canvas, width, height, zero_y, x_of, y_of = plot.canvas, plot.width, plot.height, plot.zero_y, plot.x_of, plot.y_of

    runs = _runs([None if v is None else (x_of(i), y_of(v)) for i, v in enumerate(values)])
    if not runs:
        return plot.image()

    peak_y = min(y for run in runs for _, y in run)
    fill = _mask(width, height, lambda draw, s: [
        draw.polygon([(x * s, y * s) for x, y in run] + [(run[-1][0] * s, zero_y * s), (run[0][0] * s, zero_y * s)],
                     fill=255) for run in runs if len(run) > 1])
    _paint(canvas, color, fill * plot.fade(peak_y))

    # The average under the value line, so where the two meet the values win; it follows the
    # values' own runs and never bridges a gap in them.
    trend = _runs([None if v is None or values[i] is None else (x_of(i), y_of(v))
                   for i, v in enumerate(average[:len(values)])])
    trend = [_smooth(run) for run in trend if len(run) > 1]
    if trend:
        line = _mask(width, height, lambda draw, s: [_polyline(draw, run, s, AVERAGE_PX) for run in trend])
        _paint(canvas, average_color or Theme.COLOR_BRIGHT, AVERAGE_GLOW * _blur(line, AVERAGE_GLOW_PX))
        _paint(canvas, average_color or Theme.COLOR_BRIGHT, AVERAGE_ALPHA * line)
    _paint(canvas, color, _mask(width, height, lambda draw, s: [_polyline(draw, run, s, LINE_PX) for run in runs]))

    if values[-1] is not None:
        plot.dot(*runs[-1][-1], color)
    return plot.image()


def stacked(layers: Sequence[Tuple[Sequence[Optional[float]], Color]], max_value: float, cols: int, rows: int,
            rules: Sequence[Rule] = (), slots: int = 0) -> Image.Image:
    """Filled lines stacked on one another, the first layer at the base: each layer's values
    (all aligned, oldest at the left, spread over `slots` as in `area`) lie on the sum of the
    layers before it, as a band filled in the layer's colour under a line along its top, and
    the newest point of the whole stack is a dot with a soft glow in the colour of its top
    layer. A missing value (None) leaves the layer out there: its band and line break, and the
    layers above rest on what is under it. A layer's reading with none beside it is a point."""
    layers = [(list(values), color) for values, color in layers]
    count = max((len(values) for values, _ in layers), default=0)
    layers = [(values + [None] * (count - len(values)), color) for values, color in layers]
    if count == 1 and slots <= 1:
        layers, count = [(values * 2, color) for values, color in layers], 2
    plot = _Plot(cols, rows, rules, count, slots, max_value)
    canvas, width, height, x_of, y_of = plot.canvas, plot.width, plot.height, plot.x_of, plot.y_of

    # Each layer as its runs of (x, y of its top, y of what it rests on).
    base = [0.0] * count
    bands, top_color = [], None
    for values, color in layers:
        tops = [None if v is None else b + v for v, b in zip(values, base)]
        bands.append((color, _runs([None if t is None else (x_of(i), y_of(t), y_of(b))
                                    for i, (t, b) in enumerate(zip(tops, base))])))
        if count and tops[-1] is not None:
            top_color = color
        base = [b if t is None else t for t, b in zip(tops, base)]
    if not any(runs for _, runs in bands):
        return plot.image()

    fade = plot.fade(min(y for _, runs in bands for run in runs for _, y, _ in run))
    for color, runs in bands:
        fill = _mask(width, height, lambda draw, s: [
            draw.polygon([(x * s, top * s) for x, top, _ in run] + [(x * s, under * s) for x, _, under in reversed(run)],
                         fill=255) for run in runs if len(run) > 1])
        _paint(canvas, color, fill * fade)
    for color, runs in bands:

        def lines(draw: ImageDraw.ImageDraw, s: int):
            for run in runs:
                if len(run) > 1:
                    _polyline(draw, [(x, top) for x, top, _ in run], s, LINE_PX)
                else:
                    x, top, _ = run[0]
                    draw.ellipse([(x - LINE_PX) * s, (top - LINE_PX) * s, (x + LINE_PX) * s, (top + LINE_PX) * s], fill=255)

        _paint(canvas, color, _mask(width, height, lines))

    if top_color is not None:
        plot.dot(x_of(count - 1), y_of(base[-1]), top_color)
    return plot.image()


class _Plot:
    """The canvas of a line chart — the panel ground with its rules — and its scales: `count`
    points spread over `slots` evenly spaced places (at least one per point), the newest at
    the right edge, and a value's height on a scale from zero to `max_value`."""

    def __init__(self, cols: int, rows: int, rules: Sequence[Rule], count: int, slots: int, max_value: float):
        cell_w, cell_h = CELL_PX
        self.width, self.height = cols * cell_w, rows * cell_h
        inset = cell_h / 2 if rows > 1 else 0.0
        self.zero_y = self.height - inset
        self.plot_h = max(1.0, self.height - 2 * inset)
        self.canvas = np.empty((self.height, self.width, 3), np.float32)
        self.canvas[:] = _rgb(Theme.COLOR_PANEL_BG)
        _rules(self.canvas, rules, self.zero_y, self.plot_h)
        self.count = count
        self.slots = max(slots, count)
        self.scale = max_value if max_value > 0 else 1.0
        self.step = max(1.0, self.width - 2 * EDGE_PX) / (self.slots - 1) if self.slots > 1 else 0.0

    def x_of(self, i: int) -> float:
        return EDGE_PX + (self.slots - self.count + i) * self.step if self.slots > 1 else self.width - EDGE_PX

    def y_of(self, value: float) -> float:
        return self.zero_y - min(1.0, max(0.0, value / self.scale)) * self.plot_h

    def fade(self, peak_y: float) -> np.ndarray:
        """A fill's opacity per pixel row: `FILL_TOP_ALPHA` at the height of the peak, fading to the base."""
        t = np.clip((self.zero_y - np.arange(self.height) - 0.5) / max(1.0, self.zero_y - peak_y), 0, 1)[:, None]
        return FILL_BOTTOM_ALPHA + (FILL_TOP_ALPHA - FILL_BOTTOM_ALPHA) * t

    def dot(self, x: float, y: float, color: Color):
        """The newest point: a dot with a soft glow."""
        dot = _mask(self.width, self.height, lambda draw, s: draw.ellipse(
            [(x - DOT_RADIUS_PX) * s, (y - DOT_RADIUS_PX) * s, (x + DOT_RADIUS_PX) * s, (y + DOT_RADIUS_PX) * s], fill=255))
        halo = _blur(dot, GLOW_RADIUS_PX)
        _paint(self.canvas, color, np.clip(halo * 2 * GLOW_STRENGTH, 0, 1) * (1 - dot))
        _paint(self.canvas, color, dot)

    def image(self) -> Image.Image:
        return Image.fromarray(np.clip(self.canvas + 0.5, 0, 255).astype(np.uint8))


def _runs(points: Sequence[Optional[Tuple[float, float]]]) -> List[List[Tuple[float, float]]]:
    """The unbroken stretches of `points` between the missing ones."""
    runs: List[List[Tuple[float, float]]] = [[]]
    for point in points:
        if point is None:
            if runs[-1]:
                runs.append([])
        else:
            runs[-1].append(point)
    return [run for run in runs if run]


def _smooth(run: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """`run` resampled along a monotone cubic (Fritsch–Carlson) through its points: a smooth
    curve that never swings above or below the points either side of it."""
    if len(run) < 3:
        return list(run)
    xs, ys = np.array([p[0] for p in run]), np.array([p[1] for p in run])
    h = np.diff(xs)
    d = np.diff(ys) / h
    m = np.empty(len(run))
    m[0], m[-1] = d[0], d[-1]
    same = d[:-1] * d[1:] > 0
    m[1:-1] = np.where(same, 2 / (1 / np.where(same, d[:-1], 1) + 1 / np.where(same, d[1:], 1)), 0)
    t = np.linspace(0, 1, CURVE_STEPS, endpoint=False)[:, None]
    h00, h10, h01, h11 = 2 * t**3 - 3 * t**2 + 1, t**3 - 2 * t**2 + t, -2 * t**3 + 3 * t**2, t**3 - t**2
    cx = xs[:-1] + t * h
    cy = h00 * ys[:-1] + h10 * h * m[:-1] + h01 * ys[1:] + h11 * h * m[1:]
    return list(zip(cx.T.ravel(), cy.T.ravel())) + [run[-1]]


def _polyline(draw: ImageDraw.ImageDraw, run: Sequence[Tuple[float, float]], s: int, width_px: float):
    """A line through `run` with round joints and ends."""
    w = max(1, round(width_px * s))
    r = w / 2
    pts = [(x * s, y * s) for x, y in run]
    if len(pts) > 1:
        draw.line(pts, fill=255, width=w, joint="curve")
    for x, y in {pts[0], pts[-1]}:
        draw.ellipse([x - r, y - r, x + r, y + r], fill=255)


def _blur(mask: np.ndarray, radius: float) -> np.ndarray:
    return np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(radius)), np.float32) / 255


def _mask(width: int, height: int, paint: Callable[[ImageDraw.ImageDraw, int], Any]) -> np.ndarray:
    """A coverage mask (0..1) of what `paint(draw, scale)` draws on a canvas `SUPERSAMPLE`
    times larger, reduced back to `width` × `height` — the anti-aliasing."""
    big = Image.new("L", (width * SUPERSAMPLE, height * SUPERSAMPLE), 0)
    paint(ImageDraw.Draw(big), SUPERSAMPLE)
    return np.asarray(big.reduce(SUPERSAMPLE), np.float32) / 255


def _paint(canvas: np.ndarray, color: Color, alpha: np.ndarray):
    # A plane at a time: broadcasting the alpha over the three channels is several times slower.
    alpha = alpha.astype(np.float32, copy=False)
    for channel, level in enumerate(_rgb(color)):
        plane = canvas[..., channel]
        plane += (level - plane) * alpha


def _rules(canvas: np.ndarray, rules: Sequence[Rule], zero_y: float, plot_h: float):
    height, width = canvas.shape[:2]
    for fraction, color in rules:
        y = int(min(height - 1, max(0, np.floor(zero_y - fraction * plot_h))))
        dashes = (np.arange(width) // DASH_PX) % 2 == 0
        canvas[y, dashes] = _rgb(color)


def render(bars: Sequence[Bar], cols: int, rows: int, rules: Sequence[Rule] = (),
           shade_top: float = 1.0, shade_span: float = 1.0, glow_bar: Optional[Bar] = None) -> Image.Image:
    """The plot as an RGB image of `cols` × `rows` cells. A colour shades by height: its base
    at `shade_top - shade_span` of full scale and below, the colour itself at `shade_top`."""
    cell_w, cell_h = CELL_PX
    width, height = cols * cell_w, rows * cell_h
    inset = cell_h / 2 if rows > 1 else 0.0
    zero_y = height - inset
    plot_h = max(1.0, height - 2 * inset)
    canvas = np.empty((height, width, 3), np.float32)
    canvas[:] = _rgb(Theme.COLOR_PANEL_BG)
    _rules(canvas, rules, zero_y, plot_h)

    # The value fraction at each pixel row's centre, top row first.
    row_values = (zero_y - (np.arange(height) + 0.5)) / plot_h
    shades: Dict[Color, np.ndarray] = {}

    def shade(color: Color) -> np.ndarray:
        if color not in shades:
            base = _rgb(Theme.mix(Theme.COLOR_PANEL_BG, Theme.deep(color), BASE_MIX))
            t = np.clip(1 - (shade_top - row_values) / max(shade_span, 1e-6), 0, 1)[:, None]
            shades[color] = base + (_rgb(color) - base) * t
        return shades[color]

    for bar in bars:
        _draw_bar(canvas, bar, zero_y, plot_h, shade)
    if glow_bar is not None:
        _glow(canvas, glow_bar, zero_y, plot_h)
    return Image.fromarray(np.clip(canvas + 0.5, 0, 255).astype(np.uint8))


def _draw_bar(canvas: np.ndarray, bar: Bar, zero_y: float, plot_h: float, shade: Callable[[Color], np.ndarray]):
    height, width = canvas.shape[:2]
    x0 = bar.x0 * CELL_PX[0] + BAR_PAD_PX
    x1 = min(width, bar.x1 * CELL_PX[0] - BAR_PAD_PX)
    segments = [s for s in bar.segments if s.end > s.start]
    if x1 <= x0 or not segments:
        return
    bottom = zero_y - segments[0].start * plot_h
    top = max(0.0, min(zero_y - segments[-1].end * plot_h, bottom - 1))   # anything above zero shows a pixel
    y0, y1 = int(top), min(height, int(np.ceil(bottom)))
    if y1 <= y0:
        return

    # Each pixel row takes the colour of the segment its centre is in, at that row's shade.
    centres = np.arange(y0, y1) + 0.5
    ends = np.array([zero_y - s.end * plot_h for s in segments])
    index = np.clip(np.searchsorted(-ends, -centres, side="left"), 0, len(segments) - 1)
    colors = np.empty((y1 - y0, 3), np.float32)
    for i, segment in enumerate(segments):
        rows = index == i
        colors[rows] = shade(segment.color)[y0:y1][rows]

    alpha = np.ones((y1 - y0, x1 - x0), np.float32)
    for boundary in ends[:-1]:
        lo = max(y0, int(round(boundary - SEPARATOR_PX / 2)))
        alpha[lo - y0:max(lo, int(round(boundary + SEPARATOR_PX / 2))) - y0] = 0
    radius = min(CAP_RADIUS_PX, (x1 - x0) / 2, bottom - top)
    cap = _cap(x1 - x0, radius, top - y0)
    rows = min(len(cap), len(alpha))
    alpha[:rows] *= cap[:rows]

    region = canvas[y0:y1, x0:x1]
    a = alpha[..., None]
    region[:] = region * (1 - a) + colors[:, None, :] * a


_caps: Dict[Tuple[int, float, float], np.ndarray] = {}


def _cap(width: int, radius: float, offset: float) -> np.ndarray:
    """The coverage (0..1) of the top pixel rows of a `width`-pixel bar whose top edge lies
    `offset` (0..1) into its first row and whose top corners are rounded by `radius`."""
    key = (width, round(radius, 2), round(offset * 8) / 8)
    found = _caps.get(key)
    if found is None:
        width, radius, offset = key
        rows = max(1, int(np.ceil(offset + radius)))
        step = 1 / CAP_SAMPLES
        sub = (np.arange(CAP_SAMPLES) + 0.5) * step
        sy = (np.arange(rows)[:, None] + sub[None, :]).reshape(-1)[:, None]
        sx = (np.arange(width)[:, None] + sub[None, :]).reshape(-1)[None, :]
        centre_x = np.clip(sx, radius, width - radius)
        corner = (sy < offset + radius) & ((sx - centre_x) ** 2 + (sy - offset - radius) ** 2 > radius ** 2)
        inside = (sy >= offset) & ~corner
        found = _caps[key] = inside.reshape(rows, CAP_SAMPLES, width, CAP_SAMPLES).mean(axis=(1, 3)).astype(np.float32)
    return found


def _glow(canvas: np.ndarray, bar: Bar, zero_y: float, plot_h: float):
    """A soft halo of the bar's top colour on the ground around it, never over the bar."""
    height, width = canvas.shape[:2]
    margin = 3 * GLOW_RADIUS_PX
    x0 = bar.x0 * CELL_PX[0] + BAR_PAD_PX
    x1 = min(width, bar.x1 * CELL_PX[0] - BAR_PAD_PX)
    top = int(max(0, zero_y - bar.segments[-1].end * plot_h))
    bottom = int(min(height, np.ceil(zero_y - bar.segments[0].start * plot_h)))
    if x1 <= x0 or bottom <= top:
        return
    cx0, cx1 = max(0, x0 - margin), min(width, x1 + margin)
    mask = np.zeros((height, cx1 - cx0), np.uint8)
    mask[top:bottom, x0 - cx0:x1 - cx0] = 255
    blurred = np.asarray(Image.fromarray(mask).filter(ImageFilter.GaussianBlur(GLOW_RADIUS_PX)), np.float32) / 255
    glow = (blurred * (mask == 0) * GLOW_STRENGTH)[..., None]
    region = canvas[:, cx0:cx1]
    region[:] = region + (_rgb(bar.segments[-1].color) - region) * glow


def _rgb(color: Color) -> np.ndarray:
    return np.array((color.r, color.g, color.b), np.float32)


class ChartPicture:
    """One chart's picture across frames, drawn again only when its `signature` — any value
    that compares equal while the chart shows the same — changes."""

    REFRESH_SECONDS = 0.5

    def __init__(self, name: str):
        self.name = name
        self.picture: Optional[Picture] = None
        self._signature: Any = None
        self._box: Optional[Tuple[int, int]] = None
        self._drawn_at = 0.0

    def update(self, cols: int, rows: int, signature: Any, draw: Callable[[], Image.Image]) -> Picture:
        """The picture for a `cols` × `rows` box showing what `signature` stands for, from
        `draw()` when the one held is stale and due for a refresh."""
        now = time.monotonic()
        box = (cols, rows)
        resized = box != self._box
        due = signature != self._signature and now - self._drawn_at >= self.REFRESH_SECONDS
        if self.picture is None or resized or due:
            image = draw()
            if self.picture is None:
                self.picture = Picture(image)
                trace(f"ChartPicture {self.name}: picture {self.picture.id} for a {cols}x{rows} box")
            else:
                self.picture.replace(image)
                if resized:
                    trace(f"ChartPicture {self.name}: redrawn for a {cols}x{rows} box")
            self._signature, self._box, self._drawn_at = signature, box, now
        return self.picture

