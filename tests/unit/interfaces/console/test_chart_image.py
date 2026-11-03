"""ChartImage: the Dashboard's plots as pixels, and ChartPicture, one chart's picture kept across frames."""
import numpy as np
import pytest

from interfaces.console import Theme
from interfaces.console.framework.Pictures import CELL_PX
from interfaces.console.framework.TUIScreen import Color
from interfaces.console.widgets import ChartImage
from interfaces.console.widgets.ChartImage import Bar, ChartPicture, Segment

A = Color(200, 100, 50)
B = Color(40, 120, 230)
RULE = Color(90, 200, 90)
GROUND = (Theme.COLOR_PANEL_BG.r, Theme.COLOR_PANEL_BG.g, Theme.COLOR_PANEL_BG.b)

ROWS = 4
CELL_W, CELL_H = CELL_PX
HEIGHT = ROWS * CELL_H
ZERO_Y = HEIGHT - CELL_H / 2            # full scale runs from the bottom row's centre to the top row's
PLOT_H = HEIGHT - CELL_H


def y_at(fraction: float) -> int:
    """The pixel row a value at `fraction` of full scale lands on."""
    return int(ZERO_Y - fraction * PLOT_H)


def centre_x(cell: int) -> int:
    return cell * CELL_W + CELL_W // 2


def px(image, x, y):
    return tuple(int(c) for c in np.asarray(image)[y, x])


def distance(rgb, color: Color) -> float:
    return float(np.linalg.norm(np.subtract(rgb, (color.r, color.g, color.b))))


@pytest.mark.unit
class TestColumns:
    def test_a_column_rises_to_its_value_and_no_further(self):
        image = ChartImage.columns([50.0], [A], 100.0, 2, ROWS, glow=False)
        assert px(image, centre_x(1), y_at(0.25)) != GROUND
        assert px(image, centre_x(1), y_at(0.75)) == GROUND

    def test_the_newest_value_is_the_rightmost_column_and_the_left_stays_empty(self):
        image = ChartImage.columns([100.0], [A], 100.0, 3, ROWS, glow=False)
        assert px(image, centre_x(2), y_at(0.5)) != GROUND
        assert all(px(image, centre_x(cell), y_at(0.5)) == GROUND for cell in (0, 1))

    def test_more_values_than_cells_keep_the_newest(self):
        image = ChartImage.columns([100.0, 0.0, 0.0], [A] * 3, 100.0, 2, ROWS, glow=False)
        assert {px(image, x, y_at(0.5)) for x in range(2 * CELL_W)} == {GROUND}

    def test_columns_have_air_between_them(self):
        image = ChartImage.columns([100.0, 100.0], [A, A], 100.0, 2, ROWS, glow=False)
        assert px(image, CELL_W, y_at(0.5)) == GROUND

    def test_a_column_shades_up_to_its_own_colour_at_the_top(self):
        image = ChartImage.columns([100.0], [A], 100.0, 1, ROWS, glow=False)
        top, bottom = px(image, centre_x(0), y_at(0.97)), px(image, centre_x(0), y_at(0.03))
        assert distance(top, A) < 8
        assert distance(bottom, A) > distance(top, A) + 40

    def test_each_value_takes_its_own_colour(self):
        image = ChartImage.columns([100.0, 100.0], [A, B], 100.0, 2, ROWS, glow=False)
        left, right = px(image, centre_x(0), y_at(0.95)), px(image, centre_x(1), y_at(0.95))
        assert distance(left, A) < distance(left, B)
        assert distance(right, B) < distance(right, A)

    def test_a_rule_is_dashed_across_the_empty_plot(self):
        image = ChartImage.columns([], [], 100.0, 2, ROWS, rules=[(1.0, RULE)], glow=False)
        row = [px(image, x, y_at(1.0)) for x in range(2 * CELL_W)]
        assert (RULE.r, RULE.g, RULE.b) in row and GROUND in row

    def test_a_column_covers_the_rule_behind_it(self):
        image = ChartImage.columns([100.0], [A], 100.0, 1, ROWS, rules=[(0.5, RULE)], glow=False)
        assert px(image, centre_x(0), y_at(0.5)) != (RULE.r, RULE.g, RULE.b)

    def test_the_newest_column_glows_on_the_ground_beside_it(self):
        beside = (CELL_W + ChartImage.BAR_PAD_PX - 2, y_at(0.5))
        lit = ChartImage.columns([100.0], [A], 100.0, 2, ROWS, glow=True)
        dark = ChartImage.columns([100.0], [A], 100.0, 2, ROWS, glow=False)
        assert px(dark, *beside) == GROUND
        assert distance(px(lit, *beside), A) < distance(GROUND, A)

    def test_the_rounded_top_leaves_its_corners_to_the_ground(self):
        image = ChartImage.columns([100.0], [A], 100.0, 1, ROWS, glow=False)
        top = y_at(1.0)
        assert px(image, ChartImage.BAR_PAD_PX, top) == GROUND
        assert px(image, centre_x(0), top + 1) != GROUND


@pytest.mark.unit
class TestStackedBars:
    def test_segments_stack_bottom_to_top_with_the_ground_between_them(self):
        bar = Bar(0, 1, [Segment(0.0, 0.5, A), Segment(0.5, 1.0, B)])
        image = ChartImage.render([bar], 1, ROWS)
        low, high = px(image, centre_x(0), y_at(0.25)), px(image, centre_x(0), y_at(0.75))
        assert distance(low, A) < distance(low, B)
        assert distance(high, B) < distance(high, A)
        assert px(image, centre_x(0), y_at(0.5)) == GROUND

    def test_a_bar_spans_the_cells_it_is_given(self):
        image = ChartImage.render([Bar(1, 3, [Segment(0.0, 1.0, A)])], 4, ROWS)
        y = y_at(0.5)
        assert px(image, centre_x(0), y) == GROUND and px(image, centre_x(3), y) == GROUND
        assert px(image, centre_x(1), y) != GROUND and px(image, 2 * CELL_W, y) != GROUND


@pytest.mark.unit
class TestChartPicture:
    @pytest.fixture
    def clock(self, monkeypatch):
        now = [1000.0]
        monkeypatch.setattr(ChartImage.time, "monotonic", lambda: now[0])
        return now

    def test_the_same_signature_never_draws_again(self, clock):
        chart, draws = ChartPicture("test"), []
        draw = lambda: draws.append(1) or ChartImage.render([], 2, 2)
        first = chart.update(2, 2, ("a",), draw)
        clock[0] += 10
        assert chart.update(2, 2, ("a",), draw) is first
        assert len(draws) == 1 and first.revision == 0

    def test_a_change_draws_again_once_the_refresh_is_due_under_the_same_id(self, clock):
        chart, draws = ChartPicture("test"), []
        draw = lambda: draws.append(1) or ChartImage.render([], 2, 2)
        picture = chart.update(2, 2, ("a",), draw)
        chart.update(2, 2, ("b",), draw)
        assert len(draws) == 1
        clock[0] += ChartPicture.REFRESH_SECONDS
        again = chart.update(2, 2, ("b",), draw)
        assert len(draws) == 2 and again.id == picture.id and again.revision == 1

    def test_a_new_box_draws_at_once(self, clock):
        chart, draws = ChartPicture("test"), []
        draw = lambda: draws.append(1) or ChartImage.render([], 2, 2)
        chart.update(2, 2, ("a",), draw)
        chart.update(3, 2, ("a",), draw)
        assert len(draws) == 2


@pytest.mark.unit
class TestRollingMean:
    def test_each_point_averages_the_window_ending_at_it(self):
        assert ChartImage.rolling_mean([2.0, 4.0, 6.0, 8.0], 2) == [2.0, 3.0, 5.0, 7.0]

    def test_missing_values_are_skipped_and_an_empty_window_is_missing(self):
        assert ChartImage.rolling_mean([None, 4.0, None, None, 8.0], 2) == [None, 4.0, 4.0, None, 8.0]


@pytest.mark.unit
class TestArea:
    COLS = 8

    def column_lit(self, image, x) -> bool:
        return any(px(image, x, y) != GROUND for y in range(HEIGHT))

    def test_the_ground_under_the_line_is_filled_and_above_it_stays_empty(self):
        image = ChartImage.area([50.0, 50.0], A, 100.0, self.COLS, ROWS)
        middle = self.COLS * CELL_W // 2
        assert px(image, middle, y_at(0.25)) != GROUND
        assert px(image, middle, y_at(0.85)) == GROUND

    def test_a_short_history_fills_the_plot_from_the_right(self):
        image = ChartImage.area([50.0, 50.0], A, 100.0, self.COLS, ROWS, slots=4)
        assert not self.column_lit(image, CELL_W)
        assert px(image, self.COLS * CELL_W - CELL_W, y_at(0.25)) != GROUND

    def test_a_missing_value_breaks_the_fill(self):
        image = ChartImage.area([50.0, 50.0, None, None, 50.0, 50.0], A, 100.0, self.COLS, ROWS)
        middle = self.COLS * CELL_W // 2
        assert px(image, middle, y_at(0.25)) == GROUND

    def test_the_rolling_average_is_drawn_above_the_fill_in_its_own_colour(self):
        values = [10.0] * 6
        average = [80.0] * 6
        image = ChartImage.area(values, A, 100.0, self.COLS, ROWS, average=average, average_color=B)
        column = [px(image, x, y_at(0.8)) for x in range(CELL_W, (self.COLS - 1) * CELL_W)]
        assert min(distance(rgb, B) for rgb in column) < 60


@pytest.mark.unit
class TestSmooth:
    """The rolling average is resampled along a monotone cubic before it is drawn: it bends
    through its points and never swings past the two either side of it."""

    @staticmethod
    def segments(run, curve):
        """The resampled points of each segment of `run`, in order."""
        steps = ChartImage.CURVE_STEPS
        return [curve[k * steps:(k + 1) * steps] for k in range(len(run) - 1)]

    def test_the_curve_starts_each_segment_on_its_point_and_ends_on_the_last(self):
        run = [(0.0, 10.0), (10.0, 30.0), (20.0, 15.0), (30.0, 40.0)]
        curve = ChartImage._smooth(run)
        assert len(curve) == (len(run) - 1) * ChartImage.CURVE_STEPS + 1
        for point, segment in zip(run, self.segments(run, curve)):
            assert segment[0] == pytest.approx(point)
        assert curve[-1] == run[-1]

    def test_the_curve_never_leaves_the_band_between_two_neighbouring_points(self):
        """A plain cubic spline through a near-flat pair beside a steep rise dips below the
        pair; the monotone one stays between them, so the trend line never shows a trough or
        a bump the data has not."""
        run = [(0.0, 0.0), (10.0, 20.0), (20.0, 22.0), (30.0, 90.0), (40.0, 10.0), (50.0, 12.0)]
        curve = ChartImage._smooth(run)
        for (_, y0), (_, y1), segment in zip(run, run[1:], self.segments(run, curve)):
            low, high = min(y0, y1), max(y0, y1)
            assert all(low - 1e-9 <= y <= high + 1e-9 for _, y in segment), (y0, y1, segment)

    def test_a_peak_is_the_top_of_the_curve_not_overshot(self):
        run = [(0.0, 0.0), (10.0, 10.0), (20.0, 0.0)]
        ys = [y for _, y in ChartImage._smooth(run)]
        assert max(ys) == pytest.approx(10.0) and min(ys) == pytest.approx(0.0)

    def test_a_flat_stretch_stays_flat(self):
        run = [(0.0, 5.0), (10.0, 5.0), (20.0, 5.0), (30.0, 40.0)]
        curve = ChartImage._smooth(run)
        flat = self.segments(run, curve)[0] + self.segments(run, curve)[1]
        assert all(y == pytest.approx(5.0) for _, y in flat)

    def test_x_only_moves_forward_so_the_line_never_doubles_back(self):
        run = [(0.0, 3.0), (7.0, 9.0), (9.0, 1.0), (30.0, 4.0)]
        xs = [x for x, _ in ChartImage._smooth(run)]
        assert all(a < b for a, b in zip(xs, xs[1:]))

    def test_fewer_than_three_points_are_drawn_as_they_are(self):
        assert ChartImage._smooth([(0.0, 1.0), (5.0, 2.0)]) == [(0.0, 1.0), (5.0, 2.0)]
