import time
from collections import deque
from typing import Callable, Deque, Optional, Tuple

from interfaces.console import Theme
from interfaces.console.framework.TUIScreen import TUIScreen, Color


class CrtCursor:
    """Animated CRT-style cursor with a fading phosphor trail.

    Call update(x, y) once per frame with the current cursor position; whenever
    the position changes the previous cell is recorded as a trail point with a
    timestamp. draw() then renders the trail (fading from the cursor color toward
    black over TRAIL_DURATION) and the current cursor head on top.

    The trail re-paints the character currently on screen at each trail cell
    (looked up via the optional `char_at` callback) so typed characters remain
    visible on the colored background instead of being wiped to a blank space.

    The cursor color defaults to BRIGHT (green) but matches the caller's `fg`
    when supplied, so the trail blends with the surrounding text color.

    All methods must be called from a single thread (the TUI animation loop).
    """

    BRIGHT = Theme.COLOR_GREEN   # the same green the Home chat speaks in
    # The trail starts at this fraction of the head colour and fades to black from there.
    TRAIL_BRIGHTNESS = 0.4
    TRAIL_DURATION = 0.6
    MAX_TRAIL = 500
    # The head blinks like a phosphor: lit at once, held for BLINK_ON seconds after it last moved
    # or relit, then decaying over BLINK_DECAY and dark for the rest of BLINK_PERIOD.
    BLINK_ON = 0.5
    BLINK_DECAY = 0.4
    BLINK_PERIOD = 1.0

    def __init__(self):
        self._trail: Deque[Tuple[int, int, float]] = deque()
        self._last_pos: Optional[Tuple[int, int]] = None
        self._last_pos_time: Optional[float] = None

    def update(self, x: int, y: int, row_start_x: Optional[int] = None):
        """Move the cursor head to (x, y), recording trail points for the path.

        `row_start_x`: if set, a row change is treated as the cursor sweeping in
        from that left margin on the new row. Trail cells are seeded from
        row_start_x to x-1 on the new row so the line "fades in" from the start
        instead of teleporting to wherever the chunk landed.
        """
        if self._last_pos == (x, y):
            return
        now = time.time()
        if self._last_pos is not None:
            ox, oy = self._last_pos
            if oy == y and abs(x - ox) > 1:
                # Big same-row jump (LLM emitting a chunk of tokens between frames,
                # wrap-collapsed spaces revealing with the next word, Home/End…) —
                # interpolate so we get a continuous trail.
                self._interpolate_row(ox, oy, x, now)
            elif oy != y and row_start_x is not None and x > row_start_x:
                # Row change: record the trail point at the OLD row's last position
                # (so it joins the comet), then seed the NEW row from row_start_x
                # so the line appears to fade in from its left edge.
                self._trail.append((ox, oy, now))
                self._interpolate_row(row_start_x, y, x, now)
            else:
                self._trail.append((ox, oy, now))
            while len(self._trail) > self.MAX_TRAIL:
                self._trail.popleft()
        self._last_pos = (x, y)
        self._last_pos_time = now

    def _interpolate_row(self, ox: int, oy: int, x: int, now: float):
        """Seed trail cells at ox, ox+step, …, x-step on row oy.

        Cells closest to ox carry the most age and cells closest to x are
        freshest, with timestamps spread linearly across the actual elapsed
        time since the last cursor move so the band's age matches the natural
        trail cadence. The cell at x itself is left to the cursor head.
        """
        step = 1 if x > ox else -1
        count = abs(x - ox)
        if count < 1:
            return
        if count == 1:
            self._trail.append((ox, oy, now))
            return
        denom = count - 1
        age_band = now - (self._last_pos_time or now)
        for i in range(count):
            cell_x = ox + step * i
            # i=0 is closest to the OLD position → oldest in the band.
            # i=count-1 is closest to the NEW position → freshest.
            age_offset = ((denom - i) / denom) * age_band
            self._trail.append((cell_x, oy, now - age_offset))

    def hide(self):
        """Clear the cursor head (trail keeps fading). Use when there's no logical cursor right now."""
        self._last_pos = None

    def reset(self):
        """Drop the trail and the head at once: the cells it remembers are screen positions,
        so once the text under them has moved (the transcript scrolled) the fade would paint
        over unrelated lines. The next update starts a fresh sweep with no trail."""
        self._trail.clear()
        self._last_pos = None
        self._last_pos_time = None

    def draw(self, screen: TUIScreen, frame: int, char_at_cursor: str = " ",
             blink: bool = True,
             char_at: Optional[Callable[[int, int], str]] = None,
             fg: Optional[Color] = None):
        now = time.time()
        # Cursor color tracks the caller's text color when supplied; otherwise the
        # classic green phosphor.
        bright = fg if fg is not None else self.BRIGHT

        # Drop expired trail points.
        while self._trail and now - self._trail[0][2] > self.TRAIL_DURATION:
            self._trail.popleft()

        # Paint trail oldest → newest so newer (brighter) cells overwrite older overlaps.
        # bg fades from `bright` → black over TRAIL_DURATION; the char stays at `fg` so
        # at t=0 it blends into a solid block and bleeds back into view as the bg darkens.
        for tx, ty, ts in self._trail:
            t = (now - ts) / self.TRAIL_DURATION  # 0 = fresh, 1 = expired
            screen.set_bg_color(self._fade_color(t, bright))
            if fg is not None:
                screen.set_color(fg)
            ch = char_at(tx, ty) if char_at is not None else " "
            screen.write_at(tx, ty, ch)

        # The head: a solid block (bg=bright, fg=bright) while lit; as its phosphor decays the
        # character under it shows through in `fg`, like a trail cell.
        if self._last_pos is not None:
            cx, cy = self._last_pos
            decay = self._blink_decay(now) if blink else 0.0
            if decay < 1.0:
                screen.set_bg_color(self._fade_color(decay, bright, strength=1.0))
                if fg is not None:
                    screen.set_color(bright if decay == 0.0 else fg)
                screen.write_at(cx, cy, char_at_cursor)

        screen.reset_color()

    def _blink_decay(self, now: float) -> float:
        """0 while the head is lit, rising to 1 as its phosphor decays; 1 while it is dark."""
        phase = (now - (self._last_pos_time or now)) % self.BLINK_PERIOD
        return max(0.0, min(1.0, (phase - self.BLINK_ON) / self.BLINK_DECAY))

    @classmethod
    def _fade_color(cls, t: float, base: Optional[Color] = None, strength: Optional[float] = None) -> Color:
        # Phosphor decay: stay near `base` for most of the lifetime, fall off fast at the end.
        # Weight = (1 - t)^2 keeps the trail bright early and fades hard at the tail.
        if base is None:
            base = cls.BRIGHT
        t = max(0.0, min(1.0, t))
        w = (cls.TRAIL_BRIGHTNESS if strength is None else strength) * (1.0 - t) ** 2
        return Color(
            int(base.r * w),
            int(base.g * w),
            int(base.b * w),
        )
