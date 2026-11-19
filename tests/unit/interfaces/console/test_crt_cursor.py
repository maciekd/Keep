import time
from types import SimpleNamespace
from unittest.mock import MagicMock

from interfaces.console.widgets.CrtCursor import CrtCursor
from interfaces.console.framework.TUIScreen import Color


def _make_screen():
    """A MagicMock screen that records set_bg_color/write_at calls in order."""
    screen = MagicMock()
    screen.calls = []

    def set_bg_color(color):
        screen.calls.append(("bg", color.r, color.g, color.b))

    def write_at(x, y, ch):
        screen.calls.append(("w", x, y, ch))

    screen.set_bg_color.side_effect = set_bg_color
    screen.write_at.side_effect = write_at
    return screen


class TestUpdate:
    def test_first_update_sets_position_no_trail(self):
        c = CrtCursor()
        c.update(5, 10)
        assert c._last_pos == (5, 10)
        assert len(c._trail) == 0

    def test_position_change_records_trail_point(self):
        c = CrtCursor()
        c.update(5, 10)
        c.update(6, 10)
        assert c._last_pos == (6, 10)
        assert len(c._trail) == 1
        assert c._trail[0][0] == 5 and c._trail[0][1] == 10

    def test_same_position_does_not_record_trail(self):
        c = CrtCursor()
        c.update(5, 10)
        c.update(5, 10)
        c.update(5, 10)
        assert len(c._trail) == 0

    def test_trail_capped_at_max(self):
        c = CrtCursor()
        for i in range(CrtCursor.MAX_TRAIL + 20):
            c.update(i, 0)
        assert len(c._trail) == CrtCursor.MAX_TRAIL


class TestInterpolation:
    def test_same_row_jump_fills_intermediate_cells(self):
        c = CrtCursor()
        c.update(5, 10)
        c.update(35, 10)  # jump of 30 cells
        # 30 trail points covering x=5..34 inclusive (new pos 35 is the head).
        xs = [p[0] for p in c._trail if p[1] == 10]
        assert xs == list(range(5, 35))

    def test_same_row_backward_jump_interpolates(self):
        c = CrtCursor()
        c.update(20, 5)
        c.update(10, 5)  # jump backwards by 10
        xs = [p[0] for p in c._trail if p[1] == 5]
        # Cells from old (20) toward new (10), exclusive of new head.
        assert xs == [20, 19, 18, 17, 16, 15, 14, 13, 12, 11]

    def test_interpolated_timestamps_are_oldest_at_old_end(self, monkeypatch):
        # Comet tail: cells closest to OLD position carry more age (fade first).
        clock = iter([100.0, 100.01])   # a measurable age band for the interpolation
        monkeypatch.setattr("interfaces.console.widgets.CrtCursor.time", SimpleNamespace(time=lambda: next(clock)))
        c = CrtCursor()
        c.update(0, 0)
        c.update(10, 0)
        # Trail in record order: cells x=0,1,2,...,9 with timestamps:
        # x=0 (oldest, biggest age_offset) up to x=9 (freshest).
        timestamps = [p[2] for p in c._trail]
        # Strictly increasing (older → newer).
        assert all(timestamps[i] < timestamps[i + 1] for i in range(len(timestamps) - 1))

    def test_row_change_records_only_old_position(self):
        c = CrtCursor()
        c.update(20, 5)
        c.update(2, 6)  # different row → no interpolation
        assert len(c._trail) == 1
        assert c._trail[0][:2] == (20, 5)

class TestHide:
    def test_hide_clears_head_keeps_trail(self):
        c = CrtCursor()
        c.update(1, 1)
        c.update(2, 1)
        assert len(c._trail) == 1
        c.hide()
        assert c._last_pos is None
        assert len(c._trail) == 1

    def test_update_after_hide_does_not_emit_trail_point(self):
        # Otherwise reappearing cursor would draw a phantom trail point at its first new spot.
        c = CrtCursor()
        c.update(1, 1)
        c.hide()
        c.update(9, 9)
        assert c._last_pos == (9, 9)
        assert len(c._trail) == 0


class TestReset:
    def test_reset_drops_trail_and_head(self):
        c = CrtCursor()
        c.update(1, 1)
        c.update(5, 1)
        assert len(c._trail) == 4
        c.reset()
        assert c._last_pos is None
        assert len(c._trail) == 0

    def test_update_after_reset_starts_a_fresh_sweep(self):
        c = CrtCursor()
        c.update(1, 1)
        c.update(5, 1)
        c.reset()
        c.update(3, 4, row_start_x=1)
        assert c._last_pos == (3, 4)
        assert len(c._trail) == 0


class TestFadeColor:
    def test_t_one_is_black(self):
        col = CrtCursor._fade_color(1.0)
        assert (col.r, col.g, col.b) == (0, 0, 0)

    def test_t_clamped_below_zero(self):
        assert CrtCursor._fade_color(-0.5) == CrtCursor._fade_color(0.0)

    def test_t_clamped_above_one(self):
        col = CrtCursor._fade_color(2.0)
        assert (col.r, col.g, col.b) == (0, 0, 0)

    def test_decay_is_monotonic(self):
        prev = 256
        for i in range(11):
            t = i / 10.0
            g = CrtCursor._fade_color(t).g
            assert g <= prev
            prev = g


class TestDraw:
    def test_expired_trail_dropped(self):
        c = CrtCursor()
        c.update(0, 0)
        c.update(1, 0)
        # Force trail timestamp into the past.
        x, y, _ = c._trail[0]
        c._trail[0] = (x, y, time.time() - CrtCursor.TRAIL_DURATION - 0.1)
        screen = _make_screen()
        c.draw(screen, frame=0)
        # No bg draw for trail (it expired); only the head should have been painted.
        # Head paints exactly one bg + write pair.
        bg_calls = [call for call in screen.calls if call[0] == "bg"]
        assert len(bg_calls) == 1

    def test_draw_with_no_position_and_no_trail_only_resets_color(self):
        c = CrtCursor()
        screen = _make_screen()
        c.draw(screen, frame=0)
        assert screen.calls == []
        screen.reset_color.assert_called_once()

    def test_blink_off_phase_skips_head(self):
        c = CrtCursor()
        c.update(3, 4)
        screen = _make_screen()
        # Frame inside the blink-off window.
        c.draw(screen, frame=CrtCursor.BLINK_ON)
        bg_calls = [call for call in screen.calls if call[0] == "bg"]
        assert len(bg_calls) == 0  # no trail, head suppressed by blink

    def test_blink_disabled_always_draws_head(self):
        c = CrtCursor()
        c.update(3, 4)
        screen = _make_screen()
        c.draw(screen, frame=CrtCursor.BLINK_ON, blink=False)
        bg_calls = [call for call in screen.calls if call[0] == "bg"]
        assert len(bg_calls) == 1

    def test_draw_renders_trail_then_head(self):
        c = CrtCursor()
        c.update(0, 0)
        c.update(1, 0)
        c.update(2, 0)
        screen = _make_screen()
        c.draw(screen, frame=0, blink=False)
        # 2 trail points + 1 head = 3 bg calls, 3 write_at calls.
        bg_calls = [call for call in screen.calls if call[0] == "bg"]
        write_calls = [call for call in screen.calls if call[0] == "w"]
        assert len(bg_calls) == 3
        assert len(write_calls) == 3
        # Last write should be the head at (2, 0).
        assert write_calls[-1][1:3] == (2, 0)

    def test_trail_uses_char_at_lookup(self):
        # The trail cells should re-paint the character on screen — not blank it out.
        text = "Hi!"
        c = CrtCursor()
        c.update(0, 0)
        c.update(1, 0)
        c.update(2, 0)  # last_pos = (2, 0); trail = [(0,0), (1,0)]
        screen = _make_screen()
        c.draw(screen, frame=0, blink=False,
               char_at=lambda x, y: text[x] if 0 <= x < len(text) else " ",
               char_at_cursor=text[2])
        write_calls = [call for call in screen.calls if call[0] == "w"]
        # Order: trail oldest→newest, then head.
        assert write_calls[0][3] == "H"  # trail at (0, 0)
        assert write_calls[1][3] == "i"  # trail at (1, 0)
        assert write_calls[2][3] == "!"  # head at (2, 0)

    def test_fg_set_per_cell_when_provided(self):
        # When fg is provided the cursor color tracks fg, so set_color is called once per
        # trail cell (with fg) plus once for the head (head fg = bright = fg).
        c = CrtCursor()
        c.update(0, 0)
        c.update(1, 0)
        c.update(2, 0)  # head at (2,0), trail at (0,0) and (1,0)
        screen = _make_screen()
        fg = Color(220, 220, 220)
        c.draw(screen, frame=0, blink=False, fg=fg)
        # 2 trail cells + 1 head = 3 set_color calls.
        assert screen.set_color.call_count == 3
        # All set_color calls should match fg (cursor color tracks the text color).
        for call in screen.set_color.call_args_list:
            arg = call.args[0]
            assert (arg.r, arg.g, arg.b) == (fg.r, fg.g, fg.b)

    def test_fg_not_set_when_omitted(self):
        c = CrtCursor()
        c.update(0, 0)
        c.update(1, 0)
        screen = _make_screen()
        c.draw(screen, frame=0, blink=False)
        screen.set_color.assert_not_called()
