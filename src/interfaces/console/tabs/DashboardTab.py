import collections
import math
import textwrap
import time
from typing import TYPE_CHECKING, Callable, Dict, List, NamedTuple, Optional, Tuple

from .TabView import TabView, CONTENT_TOP
from interfaces.console import Theme
from interfaces.console.framework.TUIScreen import TUIScreen, Color
from interfaces.console.framework.TextWidth import display_width, fit
from interfaces.console.widgets import Board, Tile
from interfaces.console.widgets.Board import MeterRow
from interfaces.console.widgets.FeedStrip import SHORT_NAMES, SYMBOL_FAILED, SYMBOL_PAUSED, SYMBOL_RUNNING, format_eta
from common.GpuMonitor import GpuMonitor
from common.AnthropicCostMonitor import AnthropicCostMonitor
from common.MemoryLogMonitor import MemoryLogMonitor
from common.TempMonitor import TempMonitor
from common.TTFTMonitor import TTFTMonitor
from common.STTConfidenceMonitor import STTConfidenceMonitor
from assistant.integration.Integration import IntegrationState
from assistant.integration.FeedRunner import Feeds, FeedState
from assistant.integration.IntegrationMonitor import IntegrationMonitor
from assistant.llm.OllamaLLM import OllamaLLM

if TYPE_CHECKING:
    from interfaces.console.TUIFloxyConsole import TUIFloxyConsole


class FeedRow(NamedTuple):
    """One line of the Feeds panel, worked out from a FeedRunner status dict."""
    symbol: str
    name: str
    fg: Color
    bg: Color
    progress: Optional[float]   # 0..1 bar fill; None = catching up without a count (sweeping notch)
    text: str                   # "1234 / 23356 photos ETA: 13h" through a catch-up, "23436 photos" live, else the state


class DashboardTab(TabView):
    """Tab charting GPU, CPU, temperature, memory, TTFT and API cost over time, with a row of
    status tiles underneath: integration health, last-request timing, resource meters and
    prompt sections. Everything sits in flat tiles — a title strip over a tinted body — on a
    three-column grid (four for the status row)."""

    SUMMARY_ROWS = 4
    SUMMARY_H = SUMMARY_ROWS + Tile.CARD_CHROME

    HISTORY_SIZE = 120
    BLOCK_CHARS = [' ', '\u2581', '\u2582', '\u2583', '\u2584', '\u2585', '\u2586', '\u2587', '\u2588']

    def __init__(self, console: 'TUIFloxyConsole'):
        self.console = console
        self._frame = 0
        self._cards: List[Tuple[int, int, int, int]] = []

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
        self.cost_models: list = []
        self.total_cost_30d: float = 0.0
        self.total_cost_7d: float = 0.0
        self.total_cost_1d: float = 0.0
        self.total_cost_yesterday: float = 0.0
        self.cost_dates: list = []
        self.cost_date_strs: list = []
        self.cost_error: str = ""
        self.cost_last_updated: float = 0.0

        self.memory_log_sizes: list = []
        self.memory_word_count: int = 0
        self.memory_log_error: str = ""

        self.cpu_temp: float = 0.0
        self.gpu_temp: float = 0.0
        self.temp_error: str = ""
        self.temp_history: collections.deque = collections.deque(maxlen=self.HISTORY_SIZE)

        self._monitor = GpuMonitor(gpu_tab=self, update_interval=0.33)
        self._cost_monitor = AnthropicCostMonitor(dashboard_tab=self, fetch_interval=300.0)
        self._memory_log_monitor = MemoryLogMonitor(dashboard_tab=self, fetch_interval=300.0)
        self._temp_monitor = TempMonitor(dashboard_tab=self, update_interval=30.0)

    def draw(self, screen: TUIScreen, frame: int):
        self._frame = frame
        self._cards = []
        self._sample_cpu()
        y = self._draw_header(screen, Board.LEFT, CONTENT_TOP) + 1

        thirds = self._columns(screen, 3)
        self._draw_summary_panel(screen, thirds, y)
        y += self.SUMMARY_H + 1

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

    def _draw_summary_panel(self, screen: TUIScreen, columns: List[Tuple[int, int]], y: int):
        """Three tiles on the chart grid: utilization, mic noise floor, feeds."""
        (util_x, util_w), (noise_x, noise_w), (feeds_x, feeds_w) = columns
        self._draw_utilization_panel(screen, util_x, y, util_w, self.SUMMARY_H)
        self._draw_noise_floor_panel(screen, noise_x, y, noise_w, self.SUMMARY_H)
        self._draw_feeds_panel(screen, feeds_x, y, feeds_w, self.SUMMARY_H)

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

    def _draw_feeds_panel(self, screen: TUIScreen, x: int, y: int, panel_w: int, panel_h: int):
        """One row per running feed: the name, a state symbol (the FeedStrip's dots and cross, a
        half dot for a catch-up), a bar in the state's
        colour (green live, yellow catching up, red stopped or failed, grey starting; paused fades
        it) and, in that same colour, the feed's rows in the history with its item noun —
        "1234 / 23356 photos ETA: 13h" through a catch-up (the total is what it will hold once
        caught up), "23436 messages" once live — or the state word when paused, stopped or
        failed. A feed catching up without a count gets a notch sweeping the empty bar instead."""
        Tile.draw_card(screen, self._cards, x, y, panel_w, panel_h, "Feeds")

        rows = [self._feed_row(name, status) for name, status in self._feed_statuses()]
        if not rows:
            screen.set_color(Theme.COLOR_DIM)
            screen.write_at(x + 2, y + Tile.CARD_TOP, "no feeds running")
            screen.reset_color()
            return

        rows = rows[:panel_h - Tile.CARD_CHROME]
        label_w = max(len(row.name) for row in rows)
        symbol_x = x + 2 + label_w + 1
        bar_x = symbol_x + 2
        # The bar takes every cell the longest text on screen leaves it.
        text_w = max(display_width(row.text) for row in rows)
        bar_len = max(self.FEED_MIN_BAR, panel_w - (bar_x - x) - text_w - 3)
        text_x = bar_x + bar_len + 1
        text_w = x + panel_w - 2 - text_x     # one cell of air before the right border

        for i, row in enumerate(rows):
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
                screen.write_at(text_x, row_y, fit(row.text, text_w))
                screen.reset_color()

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
        if state == FeedState.LIVE.value:
            symbol, fg, bg = SYMBOL_RUNNING, Theme.COLOR_GREEN, Theme.COLOR_GREEN_DIM
            progress = 1.0
            text = f"{status['stored']} {items}"
        elif state == FeedState.CATCHING_UP.value:
            symbol, fg, bg = cls.SYMBOL_CATCHING_UP, Theme.COLOR_YELLOW, Theme.COLOR_YELLOW_DIM
            progress = status.get("progress")
            stored, total = status["stored"], status.get("total")
            text = f"{stored} {items}" if total in (None, stored) else f"{stored} / {total} {items}"
            if status.get("eta_seconds") is not None:
                text += f" ETA: {format_eta(status['eta_seconds'])}"
        elif state in (FeedState.STOPPED.value, FeedState.FAILED.value):
            symbol, fg, bg = SYMBOL_FAILED, Theme.COLOR_RED, Theme.COLOR_RED_DIM
            text = state
        else:
            symbol, fg, bg = SYMBOL_RUNNING, Theme.COLOR_DIM, Theme.COLOR_GRID
            text = state
        if status.get("paused") and symbol != SYMBOL_FAILED:
            symbol = SYMBOL_PAUSED
            fg = Theme.mix(Theme.COLOR_PANEL_BG, fg, 0.5)
            text = "paused"
        return FeedRow(symbol, SHORT_NAMES.get(name, name), fg, bg, progress, text)

    def _draw_charts_row(self, screen: TUIScreen, columns: List[Tuple[int, int]], y: int, chart_h: int):
        (gpu_x, gpu_w), (mem_x, mem_w), (cpu_x, cpu_w) = columns
        self._draw_bar_chart(screen, gpu_x, y, gpu_w, chart_h, self.device_history, Theme.COLOR_GREEN,
                             "GPU Usage", f"{self.device_util}%")
        self._draw_bar_chart(screen, mem_x, y, mem_w, chart_h, self.memory_history, Theme.COLOR_BLUE,
                             "GPU Memory", self._format_bytes(self.in_use_memory))
        self._draw_bar_chart(screen, cpu_x, y, cpu_w, chart_h, self.cpu_history, Theme.COLOR_YELLOW,
                             "CPU Usage", f"{self.console.system_cpu_percent:.0f}%")

    LTM_MIN_MAX_KB = 10.0

    def _draw_long_term_memory_chart(self, screen: TUIScreen, x: int, y: int, width: int, chart_h: int):
        sizes = [kb for _, kb in self.memory_log_sizes]
        latest_str = f"{sizes[-1]:.0f} KB · {self.memory_word_count:,} words" if sizes else ""
        peak = max(sizes) if sizes else 0.0
        max_kb = self._nice_ceil(peak) if peak > self.LTM_MIN_MAX_KB else self.LTM_MIN_MAX_KB

        self._draw_bar_chart(screen, x, y, width, chart_h,
                             sizes, Theme.COLOR_PINK,
                             "Long Term Memory", latest_str,
                             max_value=max_kb, y_axis_format=self._format_kb)

    @staticmethod
    def _format_kb(v: float) -> str:
        return f"{v:.0f}KB"

    # Apple Silicon junction temps: throttling starts ~100-105°C, so sustained
    # readings near 95°C mean the warm room is eating all thermal headroom.
    TEMP_WARN_C = 75.0
    TEMP_BAD_C = 95.0
    TEMP_SCALE_MAX_C = 110.0

    def _temp_zone(self, temp_c: float):
        if temp_c >= self.TEMP_BAD_C:
            return Theme.COLOR_RED, Theme.COLOR_RED_DIM, "BAD"
        if temp_c >= self.TEMP_WARN_C:
            return Theme.COLOR_YELLOW, Theme.COLOR_YELLOW_DIM, "WARM"
        return Theme.COLOR_GREEN, Theme.COLOR_GREEN_DIM, "OK"

    def _draw_temp_chart(self, screen: TUIScreen, x: int, y: int, width: int, chart_h: int):
        """The bar chart with a colour per temperature zone and dashed guides at the zone thresholds."""
        fg, _, status = self._temp_zone(self.cpu_temp)
        value = "" if self.cpu_temp <= 0 else (f"{self.cpu_temp:.0f}°C" if status == "OK" else f"{self.cpu_temp:.0f}°C {status}")
        Tile.draw_card(screen, self._cards, x, y, width, chart_h + Tile.CARD_CHROME, "CPU Temp", value, fg)

        chart_y = y + Tile.CARD_TOP
        chart_x = x + 2 + self.Y_AXIS_W
        chart_w = width - 4 - self.Y_AXIS_W
        if chart_w <= 0 or chart_h <= 0:
            return

        if self.temp_error and not self.temp_history:
            screen.set_color(Theme.COLOR_DIM)
            screen.write_at(x + 2, chart_y + 1, self.temp_error[:width - 4])
            screen.reset_color()
            return

        max_value = self.TEMP_SCALE_MAX_C
        warn_row = min(chart_h - 1, int((self.TEMP_WARN_C / max_value) * chart_h))
        bad_row = min(chart_h - 1, int((self.TEMP_BAD_C / max_value) * chart_h))
        warn_guide = Theme.mix(Theme.COLOR_PANEL_BG, Theme.COLOR_YELLOW, 0.45)
        bad_guide = Theme.mix(Theme.COLOR_PANEL_BG, Theme.COLOR_RED, 0.45)

        # Y-axis: label the warning/bad thresholds in their zone colors
        for label_val, label_color, label_row in (
                (self.TEMP_WARN_C, warn_guide, warn_row),
                (self.TEMP_BAD_C, bad_guide, bad_row)):
            screen.set_color(label_color)
            screen.write_at(x + 2, chart_y + chart_h - 1 - label_row,
                            f"{label_val:.0f}°C".rjust(self.Y_AXIS_W - 1))

        # The bad guide wins the row when both thresholds land on it
        self._draw_columns(screen, chart_x, chart_y, chart_w, chart_h, self.temp_history, max_value,
                           lambda v: self._temp_zone(v)[0],
                           guides={0: Theme.COLOR_GRID, warn_row: warn_guide, bad_row: bad_guide})

    Y_AXIS_W = 5

    SHADE_MIN_ROWS = 6

    def _draw_columns(self, screen: TUIScreen, chart_x: int, chart_y: int, chart_w: int, chart_h: int,
                      history, max_value: float, color_for: Callable[[float], Color],
                      guides: Optional[Dict[int, Color]] = None):
        """One column per value, the newest at the right, each bar shading up from its colour's
        deep companion to the colour itself — reached at the height of the tallest column on
        show, so a quiet chart keeps its colour. The shading never runs faster than the whole
        range over `SHADE_MIN_ROWS`: on a chart only a row or two high it would otherwise jump
        from deep to full between neighbouring rows and read as a cap of another colour.
        The shading steps every half row: a full cell is a lower half block in one shade over
        the next shade as its background. A column's top cell, filled to an eighth, has one.
        `color_for(value)` gives a column its colour; `guides` maps a chart row to the colour
        of a dashed line drawn across the empty part of the chart.
        A full cell is never drawn over the ground, so the seams Terminal.app leaves between
        cells show the bar's colour."""
        data = list(history)
        if len(data) > chart_w:
            data = data[-chart_w:]
        data = [0.0] * (chart_w - len(data)) + data
        guides = guides or {}
        base: Dict[Color, Color] = {}
        peak = max(data) if data else 0.0
        halves = chart_h * 2
        peak_half = max(1, min(halves, math.ceil(peak / max_value * halves)) - 1) if max_value > 0 and peak > 0 else 1
        shade_halves = max(peak_half, self.SHADE_MIN_ROWS * 2 - 1)

        for col_idx, value in enumerate(data):
            cx = chart_x + col_idx
            fg = color_for(value)
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
                        max_value: float = 100.0, y_axis_format=None):
        """A chart tile: the title strip with the latest value, a labelled y-axis (percent
        unless `y_axis_format` says otherwise) and the columns over dashed rules at the labels."""
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
                           guides={row: Theme.COLOR_GRID for row in rows})
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

    def _draw_ttft_charts_row(self, screen: TUIScreen, columns: List[Tuple[int, int]], y: int, chart_height: int):
        (x, chart_w), (fast_x, fast_w), (cache_x, cache_w) = columns
        monitor = TTFTMonitor.get_instance()

        smart_data = monitor.smart_ttft
        smart_label = f"Smart AI TTFT ({monitor.smart_model_short})" if monitor.smart_model_short else "Smart AI TTFT"
        smart_value = f"{smart_data[-1]:.2f}s" if smart_data else ""
        smart_max = self._ttft_max(smart_data)
        self._draw_bar_chart(screen, x, y, chart_w, chart_height,
                             smart_data, Theme.COLOR_CYAN,
                             smart_label, smart_value,
                             max_value=smart_max, y_axis_format=self._format_seconds)

        fast_data = monitor.fast_ttft
        fast_label = f"Fast AI TTFT ({monitor.fast_model_short})" if monitor.fast_model_short else "Fast AI TTFT"
        fast_value = f"{fast_data[-1]:.2f}s" if fast_data else ""
        fast_max = self._ttft_max(fast_data)
        self._draw_bar_chart(screen, fast_x, y, fast_w, chart_height,
                             fast_data, Theme.COLOR_PURPLE,
                             fast_label, fast_value,
                             max_value=fast_max, y_axis_format=self._format_seconds)

        cache_data = monitor.cache_pct
        cache_value = f"{cache_data[-1]}%" if cache_data else ""
        self._draw_bar_chart(screen, cache_x, y, cache_w, chart_height,
                             cache_data, Theme.COLOR_GREEN,
                             "KV-Cache Hit %", cache_value,
                             max_value=100.0, y_axis_format=self._format_pct)

    def _ttft_max(self, data: collections.deque) -> float:
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

    INTEGRATIONS_WIDTH = 38   # what the "Disabled: …" lines wrap to unless the tile says otherwise
    INTEGRATION_COLUMN_MIN = 18
    PROMPT_BUDGET_KB = 100.0

    @classmethod
    def _integration_rows(cls, statuses: List[dict], width: Optional[int] = None) -> Tuple[List[dict], List[str]]:
        """The enabled integrations, sorted by name, and the wrapped "Disabled: a, b, c" lines
        that follow them — a disabled integration only takes a share of one line, not a row."""
        enabled = sorted((s for s in statuses if s["state"] != IntegrationState.DISABLED.value),
                         key=lambda s: s["name"])
        disabled = sorted(s["name"] for s in statuses if s["state"] == IntegrationState.DISABLED.value)
        lines = textwrap.wrap("Disabled: " + ", ".join(disabled), max(10, width or cls.INTEGRATIONS_WIDTH)) if disabled else []
        return enabled, lines

    @classmethod
    def _integration_grid(cls, count: int, inner_w: int) -> Tuple[int, int]:
        """(columns, rows) the enabled integrations are laid out in: as many columns as fit
        the tile, filled top to bottom."""
        columns = max(1, min(3, inner_w // cls.INTEGRATION_COLUMN_MIN))
        return columns, math.ceil(count / columns)

    def _status_panel_height(self, screen: TUIScreen) -> int:
        """Height of the status tiles: the tallest body plus the tile's own rows, so the chart rows above can shrink to fit it."""
        inner_w = self._columns(screen, 4)[0][1] - 4
        enabled, disabled_lines = self._integration_rows(IntegrationMonitor.get_instance().get_all_statuses(), inner_w)
        integrations = self._integration_grid(len(enabled), inner_w)[1] + (1 + len(disabled_lines) if disabled_lines else 0)
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
        enabled, disabled_lines = self._integration_rows(IntegrationMonitor.get_instance().get_all_statuses(), inner_w)
        ok = sum(1 for status in enabled if status["state"] == IntegrationState.OK.value)
        Tile.draw_card(screen, self._cards, x, y, width, height, "Integrations", f"{ok}/{len(enabled)} ok" if enabled else "",
                        Theme.COLOR_GREEN if ok == len(enabled) else Theme.COLOR_YELLOW)

        columns, rows = self._integration_grid(len(enabled), inner_w)
        column_w = inner_w // columns
        bottom = y + height - 2     # the last content row
        for i, status in enumerate(enabled):
            cx = x + 2 + (i // rows) * column_w
            cy = y + Tile.CARD_TOP + i % rows
            if cy > bottom:
                continue
            healthy = status["state"] == IntegrationState.OK.value
            state_color = self.STATUS_STATE_COLORS.get(status["state"], Theme.COLOR_DIM)
            screen.set_color(state_color)
            screen.write_at(cx, cy, "●")
            screen.set_color(Theme.COLOR_LABEL if healthy else state_color)
            screen.write_at(cx + 2, cy, fit(status["name"], column_w - 3))
        screen.set_color(Theme.COLOR_DIM)
        for i, line in enumerate(disabled_lines):
            if y + Tile.CARD_TOP + rows + 1 + i <= bottom:
                screen.write_at(x + 2, y + Tile.CARD_TOP + rows + 1 + i, line)
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

        prefill_pct = OllamaLLM.prefill_progress()
        kv_ready = prefill_pct is None or prefill_pct == 100.0
        kv_value, kv_frac = ("Hot", 1.0) if prefill_pct is None else (f"{prefill_pct:.0f}%", prefill_pct / 100.0)

        rows = [
            row("Processed photos", f"{console.photos_indexed}/{console.photos_total}", photos_frac, photos_frac < 0.9),
            row("Voice cache", f"{console.voice_cache_size_gb:.1f}/{console.voice_cache_max_gb:.1f} GB", voice_frac, voice_frac >= 0.8),
            row("Database", f"{console.db_size_gb:.1f}/1 GB", db_frac, db_frac >= 0.8),
            row("Total memory", f"{console.system_used_memory_gb:.1f}/{console.system_total_memory_gb:.1f} GB", total_mem_frac, total_mem_frac >= 0.8),
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

    MODEL_COLORS_LIST = [Theme.COLOR_CYAN, Theme.COLOR_PURPLE, Theme.COLOR_YELLOW, Theme.COLOR_GREEN, Theme.COLOR_PINK,
                         Theme.COLOR_ORANGE]

    def _draw_cost_chart(self, screen: TUIScreen, x: int, y: int, width: int, box_h: int):
        if box_h < 6 or width < 30:
            return

        models = self.cost_models
        daily_costs = self.daily_costs
        dates = self.cost_dates
        date_strs = self.cost_date_strs

        cost_summary = f"Month ${self.total_cost_30d:.0f} \u00b7 Week ${self.total_cost_7d:.0f} \u00b7 Day ${self.total_cost_yesterday:.0f}"

        y_axis_w = self.Y_AXIS_W
        # Legend height: 1 line for legend + 1 for x-axis labels
        legend_h = 2
        box_w = width
        chart_h = box_h - Tile.CARD_CHROME - legend_h

        Tile.draw_card(screen, self._cards, x, y, box_w, box_h, "Anthropic API Cost",
                        cost_summary if self.cost_last_updated > 0 else "", Theme.COLOR_CYAN)

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
            max_cost = max(max_cost, sum(day_data.get(m, 0) for m in models))
        if max_cost <= 0:
            max_cost = 1.0
        # Round up to a nice number
        nice_max = self._nice_ceil(max_cost)

        num_labels = 4
        tick_rows = self._tick_rows(chart_h, num_labels)
        for i, row in enumerate(tick_rows):
            label_val = nice_max * i / num_labels
            label_y = chart_y + chart_h - 1 - row
            label_str = f"{label_val:.0f}" if label_val >= 1 or label_val == 0 else f"{label_val:.1f}"
            screen.set_color(Theme.COLOR_DIM)
            screen.write_at(x + 2, label_y, label_str.rjust(y_axis_w - 1))
            screen.set_color(Theme.COLOR_GRID)
            screen.write_at(chart_x, label_y, '┄' * chart_w)
        screen.reset_color()

        # Draw bars
        for day_idx in range(num_days):
            col_x = chart_x + day_idx * (bar_w + gap)
            if col_x + bar_w > x + box_w - 1:
                break

            day_data = daily_costs.get(display_date_strs[day_idx], {})

            # Build segments for this day (bottom to top)
            segments = []
            for mi, m in enumerate(models):
                cost = day_data.get(m, 0)
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
                    color = self._model_shade(mi, grad_t)
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
                    color_low = self._model_shade(mi_low, grad_t)
                    color_high = self._model_shade(mi_high, grad_t)
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

        # Legend row — centered under the chart area
        legend_y = chart_y + chart_h + 1
        entries = []
        for mi, m in enumerate(models):
            short_name = self._shorten_model_name(m)
            entries.append((mi, short_name, len(f"\u2501\u2501 {short_name}   ")))
        total_legend_w = sum(e[2] for e in entries) - 3 if entries else 0
        lx = chart_x + max(0, (chart_w - total_legend_w) // 2)
        for mi, short_name, entry_w in entries:
            if lx + entry_w > x + box_w - 2:
                break
            screen.set_color(self.MODEL_COLORS_LIST[mi % len(self.MODEL_COLORS_LIST)])
            screen.write_at(lx, legend_y, "\u2501\u2501")
            screen.set_color(Theme.COLOR_LABEL)
            screen.write_at(lx + 2, legend_y, f" {short_name}")
            lx += entry_w
        screen.reset_color()

    def _model_shade(self, model_index: int, t: float) -> Color:
        """A model's colour at height `t` (0..1) of the cost chart: the same deep-to-bright
        shading the other charts' columns have."""
        fg = self.MODEL_COLORS_LIST[model_index % len(self.MODEL_COLORS_LIST)]
        return Theme.mix(Theme.mix(Theme.COLOR_PANEL_BG, Theme.deep(fg), 0.8), fg, t)

    @staticmethod
    def _shorten_model_name(model: str) -> str:
        # "claude-sonnet-4-20250514" -> "sonnet-4"
        # "claude-3-5-haiku-20241022" -> "haiku-3.5"
        name = model.lower()
        name = name.replace("claude-", "")
        # Remove date suffix like "-20250514"
        parts = name.split("-")
        cleaned = []
        for p in parts:
            if len(p) == 8 and p.isdigit():
                continue
            cleaned.append(p)
        return "-".join(cleaned)

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
