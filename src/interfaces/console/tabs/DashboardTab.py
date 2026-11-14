import collections
import math
import time
from typing import TYPE_CHECKING, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

from .TabView import TabView, CONTENT_TOP
from interfaces.console import Theme
from interfaces.console.framework.TUIScreen import TUIScreen, Color
from interfaces.console.framework.TextWidth import clip, display_width, fit
from interfaces.console.widgets import Board, ChartImage, Tile
from interfaces.console.widgets.Board import MeterRow
from interfaces.console.widgets.ChartImage import Bar, ChartPicture, Rule, Segment
from interfaces.console.widgets.FeedStrip import SHORT_NAMES, SYMBOL_FAILED, SYMBOL_PAUSED, SYMBOL_RUNNING, format_eta
from common.GpuMonitor import GpuMonitor
from common.AnthropicCostMonitor import AnthropicCostMonitor
from common.CacheHitLogMonitor import CacheHitLogMonitor
from common.MemoryLogMonitor import MemoryLogMonitor
from common.TempMonitor import TempMonitor
from common.TTFTMonitor import TTFTMonitor
from common.STTConfidenceMonitor import STTConfidenceMonitor
from assistant.integration.Integration import IntegrationState
from assistant.integration.FeedRunner import Feeds, FeedState
from assistant.integration.IntegrationMonitor import IntegrationMonitor
from assistant.llm.PrefillTracker import PrefillTracker

if TYPE_CHECKING:
    from interfaces.console.TUIFloxyConsole import TUIFloxyConsole


class FeedRow(NamedTuple):
    """One line of the Feeds panel, worked out from a FeedRunner status dict."""
    symbol: str
    name: str
    fg: Color
    bg: Color
    progress: Optional[float]   # 0..1 bar fill; None = catching up without a count (sweeping notch)
    count: str                  # the feed's rows, "23 436"; empty when the row shows only its state
    detail: str                 # what follows the count, "/ 23 356 photos ETA: 13h" or "photos"; else the state

    @property
    def text(self) -> str:
        return f"{self.count} {self.detail}" if self.count else self.detail


def format_count(n: int) -> str:
    """Thousands split by a space: "23 436"."""
    return f"{n:,}".replace(",", " ")


class DashboardTab(TabView):
    """Tab charting GPU, CPU, temperature, memory, TTFT, prompt-cache hits and API cost over
    time, with a row of status tiles underneath: integration health, last-request timing,
    resource meters and prompt sections. Everything sits in flat tiles — a title strip over a
    tinted body — on a three-column grid (four for the status row). GPU usage, GPU
    memory, CPU usage, long-term memory, CPU temperature and the per-day cache hit are filled
    line charts, most with a smooth rolling average (`_draw_area_chart`); the per-request
    charts and the cost are columns.

    On a terminal that shows pictures a chart's plot is a picture (`ChartImage`, one
    `ChartPicture` per chart, redrawn in place as its data moves); the title strip, the axis
    labels and the legend stay text. Elsewhere (Terminal.app), and while a chart has nothing
    to show, the plot is drawn in block characters — a line chart as the columns it was
    before, since a line as half-block pixels is a jagged staircase.

    The samplers behind the charts run only while the tab shows (`on_activate` / `on_deactivate`);
    the log scan behind the per-day cache chart is the exception — a startup-and-daily job
    that starts with the console and runs whether the tab shows or not."""

    SUMMARY_ROWS = 4    # the Utilization tile's meters: the summary row is never shorter than them

    HISTORY_SIZE = 120
    BLOCK_CHARS = [' ', '\u2581', '\u2582', '\u2583', '\u2584', '\u2585', '\u2586', '\u2587', '\u2588']

    def __init__(self, console: 'TUIFloxyConsole'):
        self.console = console
        self._frame = 0
        self._statuses: List[dict] = []
        self._cards: List[Tuple[int, int, int, int]] = []
        self._chart_pictures: Dict[str, ChartPicture] = {}

        self.device_util = 0
        self.tiler_util = 0
        self.renderer_util = 0
        self.alloc_memory = 0
        self.in_use_memory = 0
        self.in_use_memory_driver = 0
        self.pb_size = 0
        self.tiled_scene_bytes = 0
        self.recovery_count = 0

        self.model = ""
        self.gpu_core_count = 0
        self.cpu_brand = ""
        self.p_cores = 0
        self.e_cores = 0
        self.total_memory_gb = 0

        self.device_history: collections.deque = collections.deque(maxlen=self.HISTORY_SIZE)
        self.tiler_history: collections.deque = collections.deque(maxlen=self.HISTORY_SIZE)
        self.renderer_history: collections.deque = collections.deque(maxlen=self.HISTORY_SIZE)
        self.memory_history: collections.deque = collections.deque(maxlen=self.HISTORY_SIZE)
        self.cpu_history: collections.deque = collections.deque(maxlen=self.HISTORY_SIZE)
        self._last_cpu_sample: float = 0.0

        # Anthropic cost data (written by AnthropicCostMonitor)
        self.daily_costs: dict = {}
        self.cost_workspaces: list = []
        self.total_cost_30d: float = 0.0
        self.total_cost_1d: float = 0.0
        self.total_cost_yesterday: float = 0.0
        self.cost_dates: list = []
        self.cost_date_strs: list = []
        self.cost_error: str = ""
        self.cost_last_updated: float = 0.0

        self.memory_log_sizes: list = []
        self.memory_word_count: int = 0
        self.memory_log_error: str = ""

        self.cache_daily: list = []             # (day, median hit % or None) per calendar day, written by CacheHitLogMonitor
        self.cache_daily_error: str = ""

        self.cpu_temp: float = 0.0
        self.gpu_temp: float = 0.0
        self.temp_error: str = ""
        self.temp_history: collections.deque = collections.deque(maxlen=self.HISTORY_SIZE)

        self._monitor = GpuMonitor(gpu_tab=self, update_interval=0.33)
        self._cost_monitor = AnthropicCostMonitor(dashboard_tab=self, fetch_interval=300.0)
        self._memory_log_monitor = MemoryLogMonitor(dashboard_tab=self, fetch_interval=300.0)
        self._temp_monitor = TempMonitor(dashboard_tab=self, update_interval=30.0)
        self._cache_log_monitor = CacheHitLogMonitor(dashboard_tab=self)
        self._cache_log_monitor.start()

    def draw(self, screen: TUIScreen, frame: int):
        self._frame = frame
        self._cards = []
        # Once per frame: a status row asks each integration for its live login hint.
        self._statuses = self._integration_rows(IntegrationMonitor.get_instance().get_all_statuses())
        self._sample_cpu()
        y = self._draw_header(screen, Board.LEFT, CONTENT_TOP) + 1

        thirds = self._columns(screen, 3)
        summary_h = self._summary_height()
        self._draw_summary_panel(screen, thirds, y, summary_h)
        y += summary_h + 1

        # The status tiles below the charts grow with the integration list and the outposts,
        # so the three chart rows (tile = chart_height + CARD_CHROME, plus a gap row) take whatever
        # height is left after reserving them.
        panel_h = self._status_panel_height(screen)
        chart_height = max(5, (screen.height - y - 1 - panel_h) // 3 - Tile.CARD_CHROME - 1)
        for draw_row in (self._draw_charts_row, self._draw_ttft_charts_row, self._draw_bottom_charts_row):
            draw_row(screen, thirds, y, chart_height)
            y += chart_height + Tile.CARD_CHROME + 1
        # On a screen too short for it the tiles end at the last row and clip their rows:
        # a write past the screen lands on its last row, on top of whatever is there.
        self._draw_status_panel(screen, y, min(panel_h, screen.height - y + 1))
        Tile.tint_cards(screen, self._cards)

    @staticmethod
    def _columns(screen: TUIScreen, count: int) -> List[Tuple[int, int]]:
        """(x, width) of `count` equal tiles across the screen."""
        return Board.columns(screen.width - Board.LEFT - Board.RIGHT_MARGIN, count)

    def handle_input(self, key: str) -> bool:
        return False

    def _chart_picture(self, name: str) -> ChartPicture:
        found = self._chart_pictures.get(name)
        if found is None:
            found = self._chart_pictures[name] = ChartPicture(name)
        return found

    def on_activate(self):
        self._monitor.start()
        self._cost_monitor.start()
        self._memory_log_monitor.start()
        self._temp_monitor.start()

    def on_deactivate(self):
        self._monitor.stop()
        self._cost_monitor.stop()
        self._memory_log_monitor.stop()
        self._temp_monitor.stop()

    def _sample_cpu(self):
        now = time.monotonic()
        if now - self._last_cpu_sample >= 1.0:
            self._last_cpu_sample = now
            self.cpu_history.append(self.console.system_cpu_percent)

    def _draw_bottom_charts_row(self, screen: TUIScreen, columns: List[Tuple[int, int]], y: int, chart_h: int):
        (ltm_x, ltm_w), (temp_x, temp_w), (cost_x, cost_w) = columns
        self._draw_long_term_memory_chart(screen, ltm_x, y, ltm_w, chart_h)
        self._draw_temp_chart(screen, temp_x, y, temp_w, chart_h)
        self._draw_cost_chart(screen, cost_x, y, cost_w, chart_h + Tile.CARD_CHROME)

    def _draw_header(self, screen: TUIScreen, x: int, y: int) -> int:
        if not self.model:
            screen.set_color(Theme.COLOR_DIM)
            screen.write_at(x, y, "Loading GPU data...")
            screen.reset_color()
            return y + 1

        screen.set_color(Theme.COLOR_CYAN)
        screen.write_at(x, y, "\u25cf")
        screen.set_color(Theme.COLOR_BRIGHT)
        screen.bold()
        screen.write_at(x + 2, y, self.model)
        screen.reset_color()

        mem = f"{self.total_memory_gb}GB Unified Memory" if self.total_memory_gb else "Unified Memory Architecture"
        screen.set_color(Theme.COLOR_DIM)
        screen.write_at(x + 2 + display_width(self.model) + 3, y,
                        f"{self.gpu_core_count}-core GPU \u00b7 {self.p_cores + self.e_cores}-core CPU \u00b7 {mem}")
        screen.reset_color()
        return y + 1

    def _summary_height(self) -> int:
        """Height of the summary tiles: one row per running feed (plus the total) and per local mic,
        never fewer than the utilization meters, plus the tile's own rows — the chart rows below shrink to fit it."""
        feeds = len(self._feed_statuses())
        rows = max(self.SUMMARY_ROWS, feeds + 1 if feeds else 0, len(self._mic_noise_floors()))
        return rows + Tile.CARD_CHROME

    def _draw_summary_panel(self, screen: TUIScreen, columns: List[Tuple[int, int]], y: int, panel_h: int):
        """Three tiles on the chart grid: utilization, mic noise floor, feeds."""
        (util_x, util_w), (noise_x, noise_w), (feeds_x, feeds_w) = columns
        self._draw_utilization_panel(screen, util_x, y, util_w, panel_h)
        self._draw_noise_floor_panel(screen, noise_x, y, noise_w, panel_h)
        self._draw_feeds_panel(screen, feeds_x, y, feeds_w, panel_h)

    def _draw_utilization_panel(self, screen: TUIScreen, x: int, y: int, panel_w: int, panel_h: int):
        Tile.draw_card(screen, self._cards, x, y, panel_w, panel_h, "Utilization")
        cpu = int(self.console.system_cpu_percent)
        Board.draw_meter_rows(screen, x, y, panel_w, panel_h, [
            MeterRow("Device", f"{self.device_util:>3}%", self.device_util / 100.0, Theme.COLOR_CYAN),
            MeterRow("Tiler", f"{self.tiler_util:>3}%", self.tiler_util / 100.0, Theme.COLOR_GREEN),
            MeterRow("Renderer", f"{self.renderer_util:>3}%", self.renderer_util / 100.0, Theme.COLOR_YELLOW),
            MeterRow("CPU", f"{cpu:>3}%", cpu / 100.0, Theme.COLOR_PINK),
        ])

    # Fixed full-scale for the noise panel bar (raw frame RMS). Both values render on
    # this one scale: the fill is the live 1s low-percentile (the raise rule's input),
    # the bright notch marks the current noise floor.
    NOISE_BAR_MAX_RMS = 0.05

    def _draw_noise_floor_panel(self, screen: TUIScreen, x: int, y: int, panel_w: int, panel_h: int):
        """Per-mic view of the adaptive energy gate (see RmsVAD): the live level as
        the bar fill, the floor as a marker notch, both on a fixed 0..NOISE_BAR_MAX_RMS
        scale. Fill red when the level sits at/above the floor (the raise rule is
        lifting it), green otherwise. Values are raw RMS matching the log lines."""
        Tile.draw_card(screen, self._cards, x, y, panel_w, panel_h, "Mic noise floor")

        floors = self._mic_noise_floors()
        if not floors:
            screen.set_color(Theme.COLOR_DIM)
            screen.write_at(x + 2, y + Tile.CARD_TOP, "no local outposts")
            screen.reset_color()
            return

        label_w = max(1, min(max(len(label) for label, *_ in floors), panel_w - 4 - 23))
        bar_x = x + 2 + label_w + 1
        # value area: " 0.0004 (floor 0.0008)" ≈ 22 cols
        bar_len = panel_w - (bar_x - x) - 24

        for i, (label, floor, _threshold, live_min) in enumerate(floors[:panel_h - Tile.CARD_CHROME]):
            row_y = y + Tile.CARD_TOP + i
            screen.set_color(Theme.COLOR_LABEL)
            screen.write_at(x + 2, row_y, fit(label, label_w))
            if floor is None or live_min is None:
                screen.set_color(Theme.COLOR_DIM)
                screen.write_at(bar_x, row_y, "calibrating…")
                continue
            rising = live_min >= floor
            fg, bg = (Theme.COLOR_RED, Theme.COLOR_RED_DIM) if rising \
                else (Theme.COLOR_GREEN, Theme.COLOR_GREEN_DIM)
            Tile.draw_meter(screen, bar_x, row_y, bar_len, live_min / self.NOISE_BAR_MAX_RMS, fg,
                             track=bg, marker=floor / self.NOISE_BAR_MAX_RMS)
            screen.set_color(fg)
            screen.bold()
            value = f"{live_min:.4f}"
            screen.write_at(bar_x + bar_len + 2, row_y, value)
            screen.reset_color()
            screen.set_color(Theme.COLOR_DIM)
            screen.write_at(bar_x + bar_len + 2 + len(value) + 1, row_y,
                            f"(floor {floor:.4f})")
            screen.reset_color()

    @staticmethod
    def _mic_noise_floors():
        # Imported lazily: the outpost stack is heavyweight and absent in some
        # deployments; the console must render regardless.
        try:
            from interfaces.voice.microphones.local_outposts.LocalOutpostManager import LocalOutpostManager
        except ImportError:
            return []
        manager = LocalOutpostManager.instance
        return manager.noise_floors() if manager is not None else []

    # --- Feeds panel ---

    FEED_MIN_BAR = 6
    FEED_SWEEP_FRAMES = 90      # one bounce of the indeterminate notch: 3 s at 30 fps
    SYMBOL_CATCHING_UP = "◐"    # the FeedStrip draws a bar for a catch-up; the panel has one of its own
    TOTAL_LABEL = "Total"

    def _draw_feeds_panel(self, screen: TUIScreen, x: int, y: int, panel_w: int, panel_h: int):
        """One row per running feed: the name, a state symbol (the FeedStrip's dots and cross, a
        half dot for a catch-up), a bar in the state's
        colour (green live, yellow catching up, red stopped or failed, grey starting; paused fades
        it) and, in that same colour, the feed's rows in the history with its item noun —
        "1 234 / 23 356 photos ETA: 13h" through a catch-up (the total is what it will hold once
        caught up), "23 436 messages" once live — or the state word when paused, stopped or
        failed. A feed catching up without a count gets a notch sweeping the empty bar instead.
        Under the feeds a Total row sums the rows of all of them in "items". The counts are
        right-aligned in one column so their digits line up; a state word starts where they do."""
        Tile.draw_card(screen, self._cards, x, y, panel_w, panel_h, "Feeds")

        statuses = self._feed_statuses()
        rows = [self._feed_row(name, status) for name, status in statuses]
        if not rows:
            screen.set_color(Theme.COLOR_DIM)
            screen.write_at(x + 2, y + Tile.CARD_TOP, "no feeds running")
            screen.reset_color()
            return

        rows = rows[:max(0, panel_h - Tile.CARD_CHROME - 1)]
        total_count = format_count(self._feeds_total(statuses))
        count_w = max([len(row.count) for row in rows] + [len(total_count)])
        texts = [self._aligned_text(row.count, row.detail, count_w) for row in rows]
        total_text = self._aligned_text(total_count, "items", count_w)
        label_w = max(len(row.name) for row in rows) if rows else 0
        label_w = max(label_w, len(self.TOTAL_LABEL))
        symbol_x = x + 2 + label_w + 1
        bar_x = symbol_x + 2
        # The bar takes every cell the longest text on screen leaves it.
        text_w = max(display_width(text) for text in texts + [total_text])
        bar_len = max(self.FEED_MIN_BAR, panel_w - (bar_x - x) - text_w - 3)
        text_x = bar_x + bar_len + 1
        text_w = x + panel_w - 2 - text_x     # one cell of air before the right border

        for i, (row, text) in enumerate(zip(rows, texts)):
            row_y = y + Tile.CARD_TOP + i
            screen.set_color(Theme.COLOR_LABEL)
            screen.write_at(x + 2, row_y, f"{row.name:<{label_w}}")
            screen.set_color(row.fg)
            screen.write_at(symbol_x, row_y, row.symbol)
            if row.progress is None:
                Tile.draw_meter(screen, bar_x, row_y, bar_len, 0.0, row.fg, track=row.bg,
                                 marker=self._sweep_position(), marker_color=row.fg)
            else:
                Tile.draw_meter(screen, bar_x, row_y, bar_len, row.progress, row.fg, track=row.bg)
            if text_w > 0:
                screen.set_color(row.fg)
                screen.bold()
                screen.write_at(text_x, row_y, fit(text, text_w))
                screen.reset_color()

        total_y = y + Tile.CARD_TOP + len(rows)
        screen.set_color(Theme.COLOR_LABEL)
        screen.write_at(x + 2, total_y, f"{self.TOTAL_LABEL:<{label_w}}")
        if text_w > 0:
            screen.bold()
            screen.write_at(text_x, total_y, fit(total_text, text_w))
        screen.reset_color()

    @staticmethod
    def _aligned_text(count: str, detail: str, count_w: int) -> str:
        return f"{count:>{count_w}} {detail}" if count else detail

    @staticmethod
    def _feeds_total(statuses: List[Tuple[str, dict]]) -> int:
        return sum(status.get("stored") or 0 for _, status in statuses)

    def _sweep_position(self) -> float:
        """0..1..0 triangle wave over FEED_SWEEP_FRAMES frames."""
        t = (self._frame % self.FEED_SWEEP_FRAMES) / self.FEED_SWEEP_FRAMES
        return 2 * t if t < 0.5 else 2 - 2 * t

    @staticmethod
    def _feed_statuses() -> List[Tuple[str, dict]]:
        feeds = Feeds.get_instance()
        statuses = []
        for name in feeds.names():
            status = feeds.status(name)
            if status is not None:
                statuses.append((name, status))
        return statuses

    @classmethod
    def _feed_row(cls, name: str, status: dict) -> FeedRow:
        state = status["state"]
        items = status.get("items", "items")
        progress: Optional[float] = 0.0
        count = ""
        if state == FeedState.LIVE.value:
            symbol, fg, bg = SYMBOL_RUNNING, Theme.COLOR_GREEN, Theme.COLOR_GREEN_DIM
            progress = 1.0
            count, detail = format_count(status["stored"]), items
        elif state == FeedState.CATCHING_UP.value:
            symbol, fg, bg = cls.SYMBOL_CATCHING_UP, Theme.COLOR_YELLOW, Theme.COLOR_YELLOW_DIM
            progress = status.get("progress")
            stored, total = status["stored"], status.get("total")
            count = format_count(stored)
            detail = items if total in (None, stored) else f"/ {format_count(total)} {items}"
            if status.get("eta_seconds") is not None:
                detail += f" ETA: {format_eta(status['eta_seconds'])}"
        elif state in (FeedState.STOPPED.value, FeedState.FAILED.value):
            symbol, fg, bg = SYMBOL_FAILED, Theme.COLOR_RED, Theme.COLOR_RED_DIM
            detail = state
        else:
            symbol, fg, bg = SYMBOL_RUNNING, Theme.COLOR_DIM, Theme.COLOR_GRID
            detail = state
        if status.get("paused") and symbol != SYMBOL_FAILED:
            symbol = SYMBOL_PAUSED
            fg = Theme.mix(Theme.COLOR_PANEL_BG, fg, 0.5)
            count, detail = "", "paused"
        return FeedRow(symbol, SHORT_NAMES.get(name, name), fg, bg, progress, count, detail)

    USAGE_AVERAGE_SAMPLES = 10  # ten seconds of the one-second readings

    def _draw_charts_row(self, screen: TUIScreen, columns: List[Tuple[int, int]], y: int, chart_h: int):
        """GPU usage, GPU memory and CPU usage as filled lines over the whole history, the
        newest reading at the right edge; the two usage charts, which jump from one reading
        to the next, carry a ten-second rolling average."""
        (gpu_x, gpu_w), (mem_x, mem_w), (cpu_x, cpu_w) = columns
        usage = dict(slots=self.HISTORY_SIZE, window=self.USAGE_AVERAGE_SAMPLES,
                     window_label=f"{self.USAGE_AVERAGE_SAMPLES}s avg")
        self._draw_area_chart(screen, gpu_x, y, gpu_w, chart_h, self.device_history, Theme.COLOR_GREEN,
                              "GPU Usage", f"{self.device_util}%", **usage)
        self._draw_area_chart(screen, mem_x, y, mem_w, chart_h, self.memory_history, Theme.COLOR_BLUE,
                              "GPU Memory", self._format_bytes(self.in_use_memory), slots=self.HISTORY_SIZE)
        self._draw_area_chart(screen, cpu_x, y, cpu_w, chart_h, self.cpu_history, Theme.COLOR_YELLOW,
                              "CPU Usage", f"{self.console.system_cpu_percent:.0f}%", **usage)

    LTM_MIN_MAX_KB = 10.0

    def _draw_long_term_memory_chart(self, screen: TUIScreen, x: int, y: int, width: int, chart_h: int):
        sizes = [kb for _, kb in self.memory_log_sizes]
        latest_str = f"{sizes[-1]:.0f} KB · {self.memory_word_count:,} words" if sizes else ""
        peak = max(sizes) if sizes else 0.0
        max_kb = self._nice_ceil(peak) if peak > self.LTM_MIN_MAX_KB else self.LTM_MIN_MAX_KB

        self._draw_area_chart(screen, x, y, width, chart_h, sizes, Theme.COLOR_PINK,
                              "Long Term Memory", latest_str, max_value=max_kb, y_axis_format=self._format_kb,
                              window=self.DAILY_AVERAGE_DAYS, window_label=f"{self.DAILY_AVERAGE_DAYS}d avg")

    @staticmethod
    def _format_kb(v: float) -> str:
        return f"{v:.0f}KB"

    # Apple Silicon junction temps: throttling starts ~100-105°C, so sustained
    # readings near 95°C mean the warm room is eating all thermal headroom.
    TEMP_WARN_C = 75.0
    MEMORY_WARN_AVAILABLE_GB = 2.0   # an absolute floor: a share of RAM means little on a large machine
    TEMP_BAD_C = 95.0
    TEMP_SCALE_MAX_C = 110.0

    def _temp_zone(self, temp_c: float):
        if temp_c >= self.TEMP_BAD_C:
            return Theme.COLOR_RED, Theme.COLOR_RED_DIM, "BAD"
        if temp_c >= self.TEMP_WARN_C:
            return Theme.COLOR_YELLOW, Theme.COLOR_YELLOW_DIM, "WARM"
        return Theme.COLOR_GREEN, Theme.COLOR_GREEN_DIM, "OK"

    TEMP_AVERAGE_SAMPLES = 10   # five minutes of the 30-second readings

    def _draw_temp_chart(self, screen: TUIScreen, x: int, y: int, width: int, chart_h: int):
        """The filled line in the colour of the current temperature zone, with dashed guides
        at the zone thresholds labelled in their colours and a five-minute rolling average."""
        fg, _, status = self._temp_zone(self.cpu_temp)
        value = "" if self.cpu_temp <= 0 else (f"{self.cpu_temp:.0f}°C" if status == "OK" else f"{self.cpu_temp:.0f}°C {status}")
        if self.temp_error and not self.temp_history:
            Tile.draw_card(screen, self._cards, x, y, width, chart_h + Tile.CARD_CHROME, "CPU Temp", value, fg)
            screen.set_color(Theme.COLOR_DIM)
            screen.write_at(x + 2, y + Tile.CARD_TOP + 1, self.temp_error[:width - 4])
            screen.reset_color()
            return

        max_value = self.TEMP_SCALE_MAX_C
        warn_guide = Theme.mix(Theme.COLOR_PANEL_BG, Theme.COLOR_YELLOW, 0.45)
        bad_guide = Theme.mix(Theme.COLOR_PANEL_BG, Theme.COLOR_RED, 0.45)
        self._draw_area_chart(screen, x, y, width, chart_h, list(self.temp_history), fg, "CPU Temp", value,
                              max_value=max_value,
                              window=self.TEMP_AVERAGE_SAMPLES, window_label="5m avg",
                              window_format=lambda v: f"{v:.0f}°C",
                              ticks=[(0.0, "", Theme.COLOR_GRID),
                                     (self.TEMP_WARN_C, f"{self.TEMP_WARN_C:.0f}°C", warn_guide),
                                     (self.TEMP_BAD_C, f"{self.TEMP_BAD_C:.0f}°C", bad_guide)],
                              color_for=lambda v: self._temp_zone(v)[0])

    DAILY_AVERAGE_DAYS = 7

    def _draw_area_chart(self, screen: TUIScreen, x: int, y: int, width: int, chart_h: int,
                         values: Sequence[Optional[float]], fg: Color, label: str, value_str: str,
                         max_value: float = 100.0, y_axis_format=None, slots: int = 0,
                         window: int = 0, window_label: str = "", window_format=None,
                         ticks: Optional[Sequence[Tuple[float, str, Color]]] = None,
                         color_for: Optional[Callable[[float], Color]] = None):
        """A chart tile with a filled line chart (`ChartImage.area`): the title strip with the
        latest value and, with a `window`, a legend for the rolling average of that many points
        traced over the line, its latest figure in parentheses (in `window_format`, the y-axis
        format by default); the y-axis labelled at 0, half and full (in `y_axis_format`,
        percent by default) unless `ticks` — (value, label, colour) — name the rules instead.
        A missing value (None) breaks the line. With nothing to show the plot is the empty
        ground with a note.
        On a screen without pictures the line would be coarse half-block pixels, so the plot
        is the values as block-character columns instead (`_draw_columns`, a missing value an
        empty column, each in `fg` or the colour `color_for(value)` gives it) with no average
        and no legend."""
        Tile.draw_card(screen, self._cards, x, y, width, chart_h + Tile.CARD_CHROME, label, value_str, fg)
        y_axis_format = y_axis_format or self._format_pct
        window_format = window_format or y_axis_format
        chart_y = y + Tile.CARD_TOP
        chart_x = x + 2 + self.Y_AXIS_W
        chart_w = width - 4 - self.Y_AXIS_W
        if chart_w <= 0 or chart_h <= 0:
            return

        if ticks is None:
            ticks = [(max_value * i / self.Y_TICKS, y_axis_format(max_value * i / self.Y_TICKS), Theme.COLOR_GRID)
                     for i in range(self.Y_TICKS + 1)]
            label_color = Theme.COLOR_DIM
        else:
            label_color = None
        rules = [(tick / max_value if max_value > 0 else 0.0, color) for tick, _, color in ticks]
        for tick, text, color in ticks:
            if text:
                row = min(chart_h - 1, int(tick / max_value * (chart_h - 1))) if max_value > 0 else 0
                screen.set_color(label_color or color)
                screen.write_at(x + 2, chart_y + chart_h - 1 - row, text.rjust(self.Y_AXIS_W - 1))
        screen.reset_color()

        values = list(values)
        guides = {min(chart_h - 1, int(r * (chart_h - 1))): c for r, c in rules}
        if not any(v is not None for v in values):
            self._draw_columns(screen, chart_x, chart_y, chart_w, chart_h, [], max_value, lambda _: fg, guides=guides)
            note = "no samples yet"
            screen.set_color(Theme.COLOR_DIM)
            screen.set_bg_color(Theme.COLOR_PANEL_BG)
            screen.write_at(chart_x + max(0, (chart_w - len(note)) // 2), chart_y + chart_h // 2, note[:chart_w])
            screen.reset_color()
            return

        if not screen.graphics:
            self._draw_columns(screen, chart_x, chart_y, chart_w, chart_h, [v or 0.0 for v in values], max_value,
                               color_for or (lambda _: fg), guides=guides)
            return

        average = ChartImage.rolling_mean(values, window) if window > 1 and len(values) > 1 else []
        average_color = Theme.mix(fg, Theme.COLOR_BRIGHT, 0.6)
        image = self._chart_picture(label).update(
            chart_w, chart_h, (values, fg, max_value, rules, slots, window),
            lambda: ChartImage.area(values, fg, max_value, chart_w, chart_h, rules, average, average_color, slots))
        screen.image(chart_x, chart_y, image, chart_w, chart_h)

        if average and window_label:
            latest = next((a for a in reversed(average) if a is not None), None)
            legend = f"─ {window_label}" + (f" ({window_format(latest)})" if latest is not None else "")
            legend_x = x + 2 + display_width(label) + 2
            legend_w = x + width - 2 - legend_x - (display_width(value_str) + 2 if value_str else 0)
            if legend_w > 0:
                Tile.write_segments(screen, legend_x, y, legend_w, [(legend, average_color)])

    Y_AXIS_W = 5

    SHADE_MIN_ROWS = 6

    def _draw_columns(self, screen: TUIScreen, chart_x: int, chart_y: int, chart_w: int, chart_h: int,
                      history, max_value: float, color_for: Callable[[float], Color],
                      guides: Optional[Dict[int, Color]] = None, colors: Optional[Sequence[Color]] = None,
                      rules: Sequence[Rule] = (), picture: str = ""):
        """One column per value, the newest at the right, each bar shading up from its colour's
        deep companion to the colour itself — reached at the height of the tallest column on
        show, so a quiet chart keeps its colour. The shading never runs faster than the whole
        range over `SHADE_MIN_ROWS`: on a chart only a row or two high it would otherwise jump
        from deep to full between neighbouring rows and read as a cap of another colour.
        The shading steps every half row: a full cell is a lower half block in one shade over
        the next shade as its background. A column's top cell, filled to an eighth, has one.
        `color_for(value)` gives a column its colour, unless `colors` names one per sample
        (aligned with `history`); `guides` maps a chart row to the colour of a dashed line
        drawn across the empty part of the chart.
        A full cell is never drawn over the ground, so the seams Terminal.app leaves between
        cells show the bar's colour.
        With `picture` (the chart's name) on a screen that shows pictures, the columns are
        the chart's picture instead, with `rules` — (fraction of `max_value`, colour) — as
        its dashed lines; a chart with no column above zero stays characters."""
        data = list(history)
        own_colors = list(colors) if colors is not None else [None] * len(data)
        if len(data) > chart_w:
            data, own_colors = data[-chart_w:], own_colors[-chart_w:]
        if picture and screen.graphics and any(value > 0 for value in data):
            shown = [own or color_for(value) for value, own in zip(data, own_colors)]
            image = self._chart_picture(picture).update(
                chart_w, chart_h, (data, shown, max_value, list(rules)),
                lambda: ChartImage.columns(data, shown, max_value, chart_w, chart_h, rules, self.SHADE_MIN_ROWS))
            screen.image(chart_x, chart_y, image, chart_w, chart_h)
            return
        padding = chart_w - len(data)
        data = [0.0] * padding + data
        own_colors = [None] * padding + own_colors
        guides = guides or {}
        base: Dict[Color, Color] = {}
        peak = max(data) if data else 0.0
        halves = chart_h * 2
        peak_half = max(1, min(halves, math.ceil(peak / max_value * halves)) - 1) if max_value > 0 and peak > 0 else 1
        shade_halves = max(peak_half, self.SHADE_MIN_ROWS * 2 - 1)

        for col_idx, value in enumerate(data):
            cx = chart_x + col_idx
            fg = own_colors[col_idx] or color_for(value)
            if fg not in base:
                base[fg] = Theme.mix(Theme.COLOR_PANEL_BG, Theme.deep(fg), 0.8)
            filled_rows = (value / max_value) * chart_h if max_value > 0 else 0
            full_rows = int(filled_rows)
            partial = filled_rows - full_rows

            def shade(half: int) -> Color:
                return Theme.mix(base[fg], fg, max(0.0, min(1.0, 1 - (peak_half - half) / shade_halves)))

            for row in range(chart_h):
                cell_y = chart_y + chart_h - 1 - row
                if row < full_rows:
                    screen.set_color(shade(row * 2))
                    screen.set_bg_color(shade(row * 2 + 1))
                    screen.write_at(cx, cell_y, '▄')
                elif row == full_rows and partial > 0:
                    screen.set_color(shade(row * 2 + (1 if partial > 0.5 else 0)))
                    screen.set_bg_color(Theme.COLOR_PANEL_BG)
                    screen.write_at(cx, cell_y, self.BLOCK_CHARS[max(1, int(partial * 8))])
                elif row in guides:
                    screen.set_color(guides[row])
                    screen.set_bg_color(Theme.COLOR_PANEL_BG)
                    screen.write_at(cx, cell_y, '┄')
                else:
                    screen.set_bg_color(Theme.COLOR_PANEL_BG)
                    screen.set_color(Theme.COLOR_PANEL_BG)
                    screen.write_at(cx, cell_y, ' ')

        screen.reset_color()

    Y_TICKS = 2                 # the scale is labelled, and ruled, at 0, half and full

    def _tick_rows(self, chart_h: int, ticks: int) -> List[int]:
        """The chart rows, counted up from the base, the scale's labels and rules sit on."""
        return [int((i / ticks) * (chart_h - 1)) for i in range(ticks + 1)]

    def _draw_bar_chart(self, screen: TUIScreen, x: int, y: int, width: int, chart_h: int,
                        history: collections.deque, fg: Color,
                        label: str, value_str: str,
                        max_value: float = 100.0, y_axis_format=None, colors: Optional[Sequence[Color]] = None):
        """A chart tile: the title strip with the latest value, a labelled y-axis (percent
        unless `y_axis_format` says otherwise) and the columns over dashed rules at the labels,
        in `fg` or each in its own colour from `colors`."""
        Tile.draw_card(screen, self._cards, x, y, width, chart_h + Tile.CARD_CHROME, label, value_str, fg)

        chart_y = y + Tile.CARD_TOP
        chart_x = x + 2 + self.Y_AXIS_W
        chart_w = width - 4 - self.Y_AXIS_W
        if chart_w <= 0 or chart_h <= 0:
            return

        y_axis_format = y_axis_format or self._format_pct
        rows = self._tick_rows(chart_h, self.Y_TICKS)
        screen.set_color(Theme.COLOR_DIM)
        for i, row in enumerate(rows):
            screen.write_at(x + 2, chart_y + chart_h - 1 - row,
                            y_axis_format(max_value * i / self.Y_TICKS).rjust(self.Y_AXIS_W - 1))
        screen.reset_color()

        self._draw_columns(screen, chart_x, chart_y, chart_w, chart_h, history, max_value, lambda _: fg,
                           guides={row: Theme.COLOR_GRID for row in rows}, colors=colors,
                           rules=[(i / self.Y_TICKS, Theme.COLOR_GRID) for i in range(self.Y_TICKS + 1)],
                           picture=label)
        if not history:
            note = "no samples yet"
            screen.set_color(Theme.COLOR_DIM)
            screen.set_bg_color(Theme.COLOR_PANEL_BG)
            screen.write_at(chart_x + max(0, (chart_w - len(note)) // 2), chart_y + chart_h // 2, note[:chart_w])
            screen.reset_color()

    @staticmethod
    def _format_seconds(v: float) -> str:
        return f"{v:.1f}s" if v < 10 else f"{v:.0f}s"

    @staticmethod
    def _format_pct(v: float) -> str:
        return f"{v:.0f}%"

    @staticmethod
    def _format_bytes(b: int) -> str:
        if b <= 0:
            return "0 B"
        if b >= 1024 ** 3:
            return f"{b / (1024 ** 3):.1f} GB"
        if b >= 1024 ** 2:
            return f"{b / (1024 ** 2):.1f} MB"
        if b >= 1024:
            return f"{b / 1024:.1f} KB"
        return f"{b} B"

    # --- TTFT Charts ---

    DEFAULT_TTFT_MAX_SECONDS = 5.0

    TTFT_SMART_COLOR = Theme.COLOR_CYAN
    TTFT_FAST_COLOR = Theme.COLOR_PURPLE
    CACHE_OTHER_COLOR = Theme.COLOR_GREEN

    def _draw_ttft_charts_row(self, screen: TUIScreen, columns: List[Tuple[int, int]], y: int, chart_height: int):
        (ttft_x, ttft_w), (daily_x, daily_w), (cache_x, cache_w) = columns
        monitor = TTFTMonitor.get_instance()
        self._draw_ttft_chart(screen, ttft_x, y, ttft_w, chart_height, monitor)
        self._draw_cache_daily_chart(screen, daily_x, y, daily_w, chart_height)
        self._draw_cache_chart(screen, cache_x, y, cache_w, chart_height, monitor)

    def _draw_ttft_chart(self, screen: TUIScreen, x: int, y: int, width: int, chart_h: int, monitor: TTFTMonitor):
        """Both models in one chart: a column per request in the order they came, the smart
        model's cyan and the fast model's purple, with the two names as a legend in the title
        strip and the latest sample as the value in its model's colour."""
        tiers = {monitor.smart_model: self.TTFT_SMART_COLOR, monitor.fast_model: self.TTFT_FAST_COLOR}
        samples = [(model, ttft) for model, ttft in monitor.samples if model in tiers]
        values = [ttft for _, ttft in samples]
        colors = [tiers[model] for model, _ in samples]
        value = f"{values[-1]:.2f}s" if values else ""
        label = "TTFT"
        self._draw_bar_chart(screen, x, y, width, chart_h, values, colors[-1] if colors else self.TTFT_SMART_COLOR,
                             label, value, max_value=self._ttft_max(values), y_axis_format=self._format_seconds,
                             colors=colors)
        self._write_chart_legend(screen, x, y, width, label, value, self._tier_legend(monitor))

    def _draw_cache_chart(self, screen: TUIScreen, x: int, y: int, width: int, chart_h: int, monitor: TTFTMonitor):
        """A column per request whose backend reported its prompt-cache hit — Claude, ds4 or
        Ollama — in the order they came, coloured by model like the TTFT chart; a model that is
        neither tier (the curator, vision, a background job) is green, with an "other" legend."""
        tiers = {monitor.smart_model: self.TTFT_SMART_COLOR, monitor.fast_model: self.TTFT_FAST_COLOR}
        samples = list(monitor.cache_samples)
        values = [pct for _, pct in samples]
        colors = [tiers.get(model, self.CACHE_OTHER_COLOR) for model, _ in samples]
        value = f"{values[-1]}%" if values else ""
        label = "KV-Cache Hit %"
        self._draw_bar_chart(screen, x, y, width, chart_h, values, colors[-1] if colors else self.CACHE_OTHER_COLOR,
                             label, value, max_value=100.0, y_axis_format=self._format_pct, colors=colors)
        legend = self._tier_legend(monitor)
        if any(model not in tiers for model, _ in samples):
            legend.append(("other", self.CACHE_OTHER_COLOR))
        self._write_chart_legend(screen, x, y, width, label, value, legend)

    def _tier_legend(self, monitor: TTFTMonitor) -> List[Tuple[str, object]]:
        return [(short, color) for short, color in ((monitor.smart_model_short, self.TTFT_SMART_COLOR),
                                                    (monitor.fast_model_short, self.TTFT_FAST_COLOR)) if short]

    @staticmethod
    def _write_chart_legend(screen: TUIScreen, x: int, y: int, width: int, label: str, value: str, legend: list):
        """The legend in a chart's title strip, between the label and the value."""
        legend_x = x + 2 + display_width(label) + 2
        legend_w = x + width - 2 - legend_x - (display_width(value) + 2 if value else 0)
        if legend and legend_w > 0:
            Tile.write_segments(screen, legend_x, y, legend_w, legend)

    def _draw_cache_daily_chart(self, screen: TUIScreen, x: int, y: int, width: int, chart_h: int):
        """A filled line, a point per calendar day: the median prompt-cache hit of the day's
        requests, from the log scan, with a seven-day rolling average. A day without a request
        breaks the line; the value is the latest day
        that had one, so a quiet morning still shows yesterday's figure."""
        series = self.cache_daily
        with_requests = [(day, median) for day, median in series if median is not None]
        value = f"{with_requests[-1][0]:%-d %b} {with_requests[-1][1]:.0f}%" if with_requests else ""
        self._draw_area_chart(screen, x, y, width, chart_h, [median for _, median in series],
                              Theme.COLOR_TEAL, "KV-Cache Hit % per day", value, max_value=100.0,
                              window=self.DAILY_AVERAGE_DAYS, window_label=f"{self.DAILY_AVERAGE_DAYS}d avg")
        if not series and self.cache_daily_error:
            screen.set_color(Theme.COLOR_DIM)
            screen.set_bg_color(Theme.COLOR_PANEL_BG)
            screen.write_at(x + 2, y + Tile.CARD_TOP, fit(self.cache_daily_error, width - 4))
            screen.reset_color()

    def _ttft_max(self, data: Sequence[float]) -> float:
        max_val = max(data) if data else 0
        return self._nice_ceil(max_val) if max_val > self.DEFAULT_TTFT_MAX_SECONDS else self.DEFAULT_TTFT_MAX_SECONDS

    # --- Status tiles (integrations, request timing, resource meters, prompt sections) ---

    STATUS_STATE_COLORS = {
        IntegrationState.OK.value: Theme.mix(Theme.COLOR_PANEL_BG, Theme.COLOR_GREEN, 0.85),
        IntegrationState.WARNING.value: Theme.COLOR_YELLOW,
        IntegrationState.NOT_CONFIGURED.value: Theme.COLOR_DIM,
        IntegrationState.ERROR.value: Theme.COLOR_RED,
        IntegrationState.NEEDS_2FA.value: Theme.COLOR_PURPLE,
        IntegrationState.RATE_LIMITED.value: Theme.COLOR_ORANGE,
        IntegrationState.UNKNOWN.value: Theme.COLOR_DIM,
        IntegrationState.DISABLED.value: Theme.COLOR_DIM,
    }
    TIMING_ROWS = 6           # the six stages; their total is in the title strip

    INTEGRATION_COLUMN_MIN = 18
    PROMPT_BUDGET_KB = 100.0

    @classmethod
    def _integration_rows(cls, statuses: List[dict]) -> List[dict]:
        """Every integration as a row of the grid: the enabled ones sorted by name, then the
        disabled ones sorted by name, so the dim rows cluster at the end."""
        disabled = IntegrationState.DISABLED.value
        return sorted(statuses, key=lambda s: (s["state"] == disabled, s["name"]))

    @classmethod
    def _integration_grid(cls, count: int, inner_w: int, max_rows: Optional[int] = None) -> Tuple[int, int]:
        """(columns, rows) the integrations are laid out in: as many columns as fit the tile,
        filled top to bottom. With `max_rows` (the content rows a clipped tile has) the rows
        are capped, so the grid holds the first `columns * rows` items and the rest are clipped."""
        columns = max(1, min(3, inner_w // cls.INTEGRATION_COLUMN_MIN))
        rows = math.ceil(count / columns)
        if max_rows is not None:
            rows = max(0, min(rows, max_rows))
        return columns, rows

    def _status_panel_height(self, screen: TUIScreen) -> int:
        """Height of the status tiles: the tallest body plus the tile's own rows, so the chart rows above can shrink to fit it."""
        inner_w = self._columns(screen, 4)[0][1] - 4
        integrations = self._integration_grid(len(self._statuses), inner_w)[1]
        pm = self.console.prompt_manager
        prompt_sections = len(pm.get_section_stats()) if pm else 0
        return max(integrations, self.TIMING_ROWS, len(self._resource_rows()), prompt_sections) + Tile.CARD_CHROME

    def _draw_status_panel(self, screen: TUIScreen, y: int, height: int):
        (ix, iw), (tx, tw), (rx, rw), (px, pw) = self._columns(screen, 4)
        self._draw_integrations(screen, ix, y, iw, height)
        self._draw_timing(screen, tx, y, tw, height)
        Tile.draw_card(screen, self._cards, rx, y, rw, height, "Resources")
        Board.draw_meter_rows(screen, rx, y, rw, height, self._resource_rows())
        self._draw_prompt_sections(screen, px, y, pw, height)

    def _draw_integrations(self, screen: TUIScreen, x: int, y: int, width: int, height: int):
        inner_w = width - 4
        statuses = self._statuses
        enabled = [s for s in statuses if s["state"] != IntegrationState.DISABLED.value]
        ok = sum(1 for status in enabled if status["state"] == IntegrationState.OK.value)
        Tile.draw_card(screen, self._cards, x, y, width, height, "Integrations", f"{ok}/{len(enabled)} ok" if enabled else "",
                        Theme.COLOR_GREEN if ok == len(enabled) else Theme.COLOR_YELLOW)

        # A tile too short for the whole list shows the first columns * rows of it: the enabled
        # ones come first in the list, so the disabled ones are clipped before any enabled one.
        columns, rows = self._integration_grid(len(statuses), inner_w, max_rows=height - Tile.CARD_CHROME)
        column_w = inner_w // columns
        for i, status in enumerate(statuses[:columns * rows]):
            cx = x + 2 + (i // rows) * column_w
            cy = y + Tile.CARD_TOP + i % rows
            healthy = status["state"] == IntegrationState.OK.value
            state_color = self.STATUS_STATE_COLORS.get(status["state"], Theme.COLOR_DIM)
            screen.set_color(state_color)
            screen.write_at(cx, cy, "●")
            screen.set_color(Theme.COLOR_LABEL if healthy else state_color)
            screen.write_at(cx + 2, cy, fit(status["name"], column_w - 3))
        screen.reset_color()

    def _draw_timing(self, screen: TUIScreen, x: int, y: int, width: int, height: int):
        """The last voice request as a waterfall: each stage's meter starts where the one
        above it ended and spans its share of the total, which the title strip carries.
        Notes name the source behind a stage's time: which gate started the winning
        decode (RMS energy vs silero VAD), and whether the LLM reply / voice audio were
        served from cache instead of generated."""
        console = self.console
        stages = [
            ("Silence detect", console.time_to_silence_detect, console.stt_trigger.upper() if console.stt_trigger else ""),
            ("Speech to text", console.time_to_first_word, ""),
            ("AI first token", console.time_to_first_token, "cached" if console.llm_from_cache else ""),
            ("AI thought for", console.time_to_user_reply, ""),
            ("Voice synthesis", console.time_to_first_sound_chunk, "cached" if console.voice_from_cache else ""),
            ("Speaker latency", console.time_to_speaker, ""),
        ]
        total = sum(value for _, value, _ in stages if value > -1)
        complete = console.time_to_first_token > -1 and console.time_to_first_sound_chunk > -1

        rows, elapsed = [], 0.0
        for name, value, note in stages:
            if value > -1:
                share = value / total if total > 0 else 0.0
                rows.append(MeterRow(name, f"{value}s", share, Theme.COLOR_YELLOW, start=elapsed, note=note))
                elapsed += share
            else:
                rows.append(MeterRow(name, "–", 0.0, Theme.COLOR_DIM, idle=True))

        Tile.draw_card(screen, self._cards, x, y, width, height, "Last request",
                        f"{round(total, 2)}s" if complete else "", Theme.COLOR_YELLOW)
        Board.draw_meter_rows(screen, x, y, width, height, rows)

    def _resource_rows(self) -> List[MeterRow]:
        console = self.console

        def ratio(part: float, whole: float) -> float:
            return part / whole if whole > 0 else 0

        def row(label: str, value: str, fraction: float, warn: bool = False) -> MeterRow:
            return MeterRow(label, value, fraction, Theme.COLOR_RED if warn else Theme.COLOR_CYAN)

        photos_frac = ratio(console.photos_indexed, console.photos_total)
        voice_frac = ratio(console.voice_cache_size_gb, console.voice_cache_max_gb)
        db_max_gb = 1.0
        db_frac = ratio(console.db_size_gb, db_max_gb)
        total_mem_frac = ratio(console.system_used_memory_gb, console.system_total_memory_gb)
        proc_mem_frac = ratio(console.process_memory_gb, console.system_total_memory_gb)
        proc_cpu_frac = console.process_cpu_percent / 100.0
        sys_cpu_frac = console.system_cpu_percent / 100.0
        fd_frac = console.fd_count / 1024.0
        stt = STTConfidenceMonitor.get_instance()
        mic_accuracy = stt.rolling_avg_12h() if stt.mics else 0.0

        prefill_pct = PrefillTracker.current_progress()
        kv_ready = prefill_pct is None or prefill_pct == 100.0
        kv_value, kv_frac = ("Hot", 1.0) if prefill_pct is None else (f"{prefill_pct:.0f}%", prefill_pct / 100.0)

        rows = [
            row("Processed photos", f"{console.photos_indexed}/{console.photos_total}", photos_frac, photos_frac < 0.9),
            row("Voice cache", f"{console.voice_cache_size_gb:.1f}/{console.voice_cache_max_gb:.1f} GB", voice_frac, voice_frac >= 0.8),
            row("Database", f"{console.db_size_gb:.1f}/1 GB", db_frac, db_frac >= 0.8),
            row("Total memory", f"{console.system_used_memory_gb:.1f}/{console.system_total_memory_gb:.1f} GB", total_mem_frac,
                console.system_total_memory_gb > 0 and console.system_available_memory_gb < self.MEMORY_WARN_AVAILABLE_GB),
            row("Keep memory", f"{console.process_memory_gb:.1f} GB", proc_mem_frac, proc_mem_frac >= 0.8),
            row("Keep CPU", f"{console.process_cpu_percent:.0f}%", proc_cpu_frac, proc_cpu_frac >= 0.8),
            row("Total CPU", f"{console.system_cpu_percent:.0f}%", sys_cpu_frac, sys_cpu_frac >= 0.8),
            row("Open FDs", f"{console.fd_count}/1024", fd_frac, console.fd_count >= 819),
            row("Mic accuracy", f"{mic_accuracy * 100:.0f}%", mic_accuracy, bool(stt.mics) and mic_accuracy < 0.3),
            row("Volume", f"{console.target_server_volume}%", console.target_server_volume / 100.0),
            row("Background noise", f"{console.current_noise_db:.0f} dB", console.current_noise_db / 80.0),
            row("KV-Cache", kv_value, kv_frac, not kv_ready),
        ]
        for addr, (latency_ms, _) in list(console.outpost_latencies.items()):
            ip_only = addr.split(',')[0] if ',' in addr else addr
            ip_only = ip_only.replace("'", "").replace("(", "").replace(")", "")
            rows.append(row(f"Outpost {ip_only}", f"{latency_ms:.1f} ms", latency_ms / 200.0))
        return rows

    @staticmethod
    def _fit_prompt_stats(stats: list, room: int) -> list:
        """The sections a tile with `room` rows shows, in their order: all of them when they
        fit, else the smallest go first — the ones reading 0.0 KB before any other."""
        if len(stats) <= room:
            return stats
        kept = sorted(range(len(stats)), key=lambda i: -stats[i][1])[:max(0, room)]
        return [stats[i] for i in sorted(kept)]

    def _draw_prompt_sections(self, screen: TUIScreen, x: int, y: int, width: int, height: int):
        pm = self.console.prompt_manager
        stats = pm.get_section_stats() if pm else []
        if not stats:
            Tile.draw_card(screen, self._cards, x, y, width, height, "Prompt")
            return

        stats = self._fit_prompt_stats(stats, height - Tile.CARD_CHROME)
        max_size = max((size for _, size, _ in stats), default=0) or 1
        quiet = Theme.mix(Theme.COLOR_PANEL_BG, Theme.COLOR_PURPLE, 0.6)
        now = time.time()

        rows = []
        for section_name, size_bytes, last_update_time in stats:
            time_ago = now - last_update_time if last_update_time > 0 else 0
            time_str = f"{int(time_ago)}s" if time_ago < 60 else f"{int(time_ago / 60)}m"
            # Sections that just changed light up for 30s so a refresh is visible.
            recent = 0 < int(time_ago) < 30
            rows.append(MeterRow(section_name.replace('_', ' ').capitalize(), f"{size_bytes / 1024.0:>5.1f} KB",
                                 size_bytes / max_size, Theme.COLOR_PURPLE if recent else quiet,
                                 note="" if int(time_ago) == 0 else time_str))

        total = ""
        over = False
        if pm.prompt:
            history_bytes = next((size for name, size, _ in stats if name == 'history'), 0)
            total_kb = (len(pm.prompt.encode('utf-8')) + history_bytes) / 1024.0
            over = total_kb > self.PROMPT_BUDGET_KB
            total = f"{total_kb:.1f} KB"

        Tile.draw_card(screen, self._cards, x, y, width, height, "Prompt", total, Theme.COLOR_RED if over else Theme.COLOR_PURPLE)
        Board.draw_meter_rows(screen, x, y, width, height, rows)

    # --- Anthropic Cost Chart ---

    WORKSPACE_COLORS = [Theme.COLOR_CYAN, Theme.COLOR_PURPLE, Theme.COLOR_YELLOW, Theme.COLOR_GREEN, Theme.COLOR_PINK,
                        Theme.COLOR_ORANGE]

    def _draw_cost_chart(self, screen: TUIScreen, x: int, y: int, width: int, box_h: int):
        if box_h < 6 or width < 30:
            return

        workspaces = self.cost_workspaces
        daily_costs = self.daily_costs
        dates = self.cost_dates
        date_strs = self.cost_date_strs

        y_axis_w = self.Y_AXIS_W
        # Legend height: 1 line for legend + 1 for x-axis labels
        legend_h = 2
        box_w = width
        chart_h = box_h - Tile.CARD_CHROME - legend_h

        Tile.draw_card(screen, self._cards, x, y, box_w, box_h, "Anthropic API Cost")
        if self.cost_last_updated > 0:
            self._draw_cost_summary(screen, x, y, box_w)

        # Show loading/error state
        if self.cost_error and not daily_costs:
            screen.set_color(Theme.COLOR_DIM)
            screen.write_at(x + 2, y + Tile.CARD_TOP, fit(self.cost_error, box_w - 4))
            screen.reset_color()
            return
        if not self.cost_last_updated or not date_strs:
            screen.set_color(Theme.COLOR_DIM)
            screen.write_at(x + 2, y + Tile.CARD_TOP, "Loading cost data...")
            screen.reset_color()
            return

        chart_x = x + 2 + y_axis_w
        chart_y = y + Tile.CARD_TOP
        chart_w = box_w - 4 - y_axis_w

        gap = 1
        bar_w = 2 if chart_w >= 90 else 1
        num_days = min((chart_w + gap) // (bar_w + gap), len(date_strs))
        if num_days <= 0:
            return

        display_date_strs = date_strs[-num_days:]
        display_dates = dates[-num_days:]

        max_cost = 0.0
        for ds in display_date_strs:
            day_data = daily_costs.get(ds, {})
            max_cost = max(max_cost, sum(day_data.get(w, 0) for w in workspaces))
        if max_cost <= 0:
            max_cost = 1.0
        # Round up to a nice number
        nice_max = self._nice_ceil(max_cost)

        bars = self._cost_bars(display_date_strs, workspaces, daily_costs, nice_max, bar_w, gap,
                               chart_w=x + box_w - 1 - chart_x)
        pictured = screen.graphics and bool(bars)

        num_labels = 4
        tick_rows = self._tick_rows(chart_h, num_labels)
        for i, row in enumerate(tick_rows):
            label_val = nice_max * i / num_labels
            label_y = chart_y + chart_h - 1 - row
            label_str = f"{label_val:.0f}" if label_val >= 1 or label_val == 0 else f"{label_val:.1f}"
            screen.set_color(Theme.COLOR_DIM)
            screen.write_at(x + 2, label_y, label_str.rjust(y_axis_w - 1))
            if not pictured:     # the picture has the rules in it, and covers these cells
                screen.set_color(Theme.COLOR_GRID)
                screen.write_at(chart_x, label_y, '┄' * chart_w)
        screen.reset_color()

        # Draw bars
        if pictured:
            rules = [(i / num_labels, Theme.COLOR_GRID) for i in range(num_labels + 1)]
            image = self._chart_picture("Anthropic API Cost").update(
                chart_w, chart_h, (bars, rules), lambda: ChartImage.render(bars, chart_w, chart_h, rules))
            screen.image(chart_x, chart_y, image, chart_w, chart_h)
        else:
            for day_idx in range(num_days):
                col_x = chart_x + day_idx * (bar_w + gap)
                if col_x + bar_w > x + box_w - 1:
                    break

                day_data = daily_costs.get(display_date_strs[day_idx], {})

                # Build segments for this day (bottom to top)
                segments = []
                for mi, w in enumerate(workspaces):
                    cost = day_data.get(w, 0)
                    if cost > 0:
                        segments.append((mi, cost))

                if not segments:
                    continue

                # Compute fractional row heights per segment
                row_positions = []
                cursor = 0.0
                for mi, cost in segments:
                    rows = (cost / nice_max) * chart_h
                    row_positions.append((mi, cursor, cursor + rows))
                    cursor += rows

                cost_max_row = max(1, chart_h - 1)
                for row in range(chart_h):
                    cell_y = chart_y + chart_h - 1 - row
                    grad_t = row / cost_max_row
                    row_bottom = float(row)
                    row_top = float(row + 1)

                    overlapping = []
                    for mi, seg_start, seg_end in row_positions:
                        if seg_end <= row_bottom or seg_start >= row_top:
                            continue
                        overlapping.append((mi, max(seg_start, row_bottom), min(seg_end, row_top)))

                    if not overlapping:
                        continue    # the rule drawn under the bars stays
                    elif len(overlapping) == 1:
                        mi, cs, ce = overlapping[0]
                        color = self._workspace_shade(mi, grad_t)
                        fill = ce - cs
                        if fill >= 0.99:
                            char = '\u2588'
                            fg_color = color
                            bg_color = color
                        else:
                            block_idx = max(1, min(7, round(fill * 8)))
                            char = self.BLOCK_CHARS[block_idx]
                            fg_color = color
                            bg_color = Theme.COLOR_PANEL_BG
                    else:
                        overlapping.sort(key=lambda s: s[1])
                        mi_low = overlapping[0][0]
                        mi_high = overlapping[-1][0]
                        color_low = self._workspace_shade(mi_low, grad_t)
                        color_high = self._workspace_shade(mi_high, grad_t)
                        fill_low = overlapping[0][2] - row_bottom
                        fill_total = overlapping[-1][2] - row_bottom
                        if fill_total >= 0.99:
                            # Row fully covered: render as dual-color block (lower seg color in
                            # the bottom portion via fg, upper seg color in the rest via bg).
                            block_idx = max(1, min(7, round(fill_low * 8)))
                            char = self.BLOCK_CHARS[block_idx]
                            fg_color = color_low
                            bg_color = color_high
                        else:
                            # Segments don't reach row_top — render only the covered portion
                            # against the panel background. Without this, bg_color = color_high
                            # would paint the empty upper portion of the cell and visually
                            # inflate the bar to a full row.
                            block_idx = max(1, min(7, round(fill_total * 8)))
                            char = self.BLOCK_CHARS[block_idx]
                            fill_high = fill_total - fill_low
                            fg_color = color_low if fill_low >= fill_high else color_high
                            bg_color = Theme.COLOR_PANEL_BG

                    screen.set_color(fg_color)
                    screen.set_bg_color(bg_color)
                    for bw in range(bar_w):
                        screen.write_at(col_x + bw, cell_y, char)

                screen.reset_color()

        label_y = chart_y + chart_h
        label_step = max(1, num_days // 5, math.ceil(8 / (bar_w + gap)))
        label_indices = list(range(0, num_days, label_step))
        if label_indices[-1] != num_days - 1:
            label_indices.append(num_days - 1)
        for idx in label_indices:
            lx = chart_x + idx * (bar_w + gap)
            if lx + 5 > x + box_w - 1:
                break
            label = display_dates[idx].strftime("%b %-d")
            screen.set_color(Theme.COLOR_DIM)
            screen.write_at(lx, label_y, label)
        screen.reset_color()

        # Legend row — each name in its bars' colour, centered under the chart area, or
        # across the whole card when wider
        legend_y = chart_y + chart_h + 1
        names = self._legend_names(workspaces, box_w - 4)
        total_legend_w = sum(display_width(n) for n in names) + self.LEGEND_GAP * (len(names) - 1)
        if total_legend_w <= chart_w:
            lx = chart_x + (chart_w - total_legend_w) // 2
        else:
            lx = x + 2 + max(0, (box_w - 4 - total_legend_w) // 2)
        for mi, name in enumerate(names):
            entry_w = display_width(name)
            if lx + entry_w > x + box_w - 2:
                break
            screen.set_color(self._workspace_color(mi))
            screen.write_at(lx, legend_y, name)
            lx += entry_w + self.LEGEND_GAP
        screen.reset_color()

    def _cost_bars(self, date_strs: Sequence[str], workspaces: Sequence[str], daily_costs: dict, nice_max: float,
                   bar_w: int, gap: int, chart_w: int) -> List[Bar]:
        """The cost chart's days as stacked bars for its picture: a segment per workspace that
        spent anything, bottom to top in workspace order, the days that fit in `chart_w` cells."""
        bars = []
        for day_idx, date_str in enumerate(date_strs):
            x0 = day_idx * (bar_w + gap)
            if x0 + bar_w > chart_w:
                break
            day_data = daily_costs.get(date_str, {})
            segments, cursor = [], 0.0
            for mi, w in enumerate(workspaces):
                cost = day_data.get(w, 0)
                if cost > 0:
                    segments.append(Segment(cursor, cursor + cost / nice_max,
                                            self.WORKSPACE_COLORS[mi % len(self.WORKSPACE_COLORS)]))
                    cursor += cost / nice_max
            if segments:
                bars.append(Bar(x0, x0 + bar_w, segments))
        return bars

    LEGEND_GAP = 3
    LEGEND_MIN_NAME = 4

    @classmethod
    def _legend_names(cls, workspaces: Sequence[str], width: int) -> List[str]:
        """The workspace names for a legend `width` cells wide: the longest one cut by a cell
        at a time, ending in an ellipsis, until every entry fits (or none is longer than
        `LEGEND_MIN_NAME`)."""
        widths = [display_width(w) for w in workspaces]
        over = sum(widths) + cls.LEGEND_GAP * (len(widths) - 1) - width
        while over > 0 and widths and max(widths) > cls.LEGEND_MIN_NAME:
            widths[widths.index(max(widths))] -= 1
            over -= 1
        return [w if display_width(w) <= cut else clip(w, cut - 1) + "\u2026"
                for w, cut in zip(workspaces, widths)]

    def _draw_cost_summary(self, screen: TUIScreen, x: int, y: int, width: int):
        """The title strip's right end: the month's total, then yesterday's cost per workspace,
        each workspace figure in the colour its bars have so the reader can tell which is
        which. The totals are in the label grey: a coloured total reads as one workspace's.
        A workspace whose figure rounds to $0 is left out; a day with none left is one grey $0."""
        yesterday = self.daily_costs.get(self.cost_date_strs[-2], {}) if len(self.cost_date_strs) >= 2 else {}
        pieces = [(f"Month ${self.total_cost_30d:.0f}", Theme.COLOR_LABEL), (" \u00b7 ", Theme.COLOR_FAINT),
                  ("Day", Theme.COLOR_LABEL)]
        figures = [(f"${yesterday.get(w, 0):.0f}", self._workspace_color(mi))
                   for mi, w in enumerate(self.cost_workspaces)]
        spent = [figure for figure in figures if figure[0] != "$0"]
        for figure in spent or [("$0", Theme.COLOR_LABEL)]:
            pieces += [(" ", Theme.COLOR_FAINT), figure]
        total_w = sum(display_width(text) for text, _ in pieces)
        if display_width("Anthropic API Cost") + total_w + 6 > width:
            return
        px = x + width - total_w - 2
        for text, color in pieces:
            screen.set_color(color)
            screen.bold()
            screen.write_at(px, y, text)
            px += display_width(text)
        screen.reset_color()

    def _workspace_color(self, workspace_index: int) -> Color:
        """A workspace's colour as text names it (the legend, the title strip's figures): the
        middle of its bars' shading, since the accent itself shows only at the top of the scale
        and reads as a different colour from the bars."""
        return self._workspace_shade(workspace_index, 0.5)

    def _workspace_shade(self, workspace_index: int, t: float) -> Color:
        """A workspace's colour at height `t` (0..1) of the cost chart: the same deep-to-bright
        shading the other charts' columns have."""
        fg = self.WORKSPACE_COLORS[workspace_index % len(self.WORKSPACE_COLORS)]
        return Theme.mix(Theme.mix(Theme.COLOR_PANEL_BG, Theme.deep(fg), 0.8), fg, t)

    @staticmethod
    def _nice_ceil(value: float) -> float:
        if value <= 0:
            return 1.0
        magnitude = 10 ** math.floor(math.log10(value))
        normalized = value / magnitude
        if normalized <= 1:
            nice = 1
        elif normalized <= 2:
            nice = 2
        elif normalized <= 5:
            nice = 5
        else:
            nice = 10
        return nice * magnitude
