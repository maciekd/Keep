"""PhotosTab's board: the summary tiles, the photo's description and its details, the scores
drawn as meters. The library is never read: the tab's state is set by hand and the reload
pinned off."""
import pytest

from interfaces.console import Theme
from interfaces.console.framework.TUIScreen import TUIScreen
from interfaces.console.tabs.PhotosTab import PhotosTab
from interfaces.console.widgets.Board import SIDE_BY_SIDE_MIN

PHOTO = {
    "short_description": "Charlotte and Frances on the jetty",
    "detailed_description": "Two people sit on a wooden jetty with their feet over the water. " * 3,
    "creation_date": 1700000000.0, "location_text": "Sweden/Dalarna/Leksand", "latitude": 60.7301,
    "longitude": 14.9999, "albums": "Favorites", "persons": "Charlotte, Frances", "photo_type": "photo",
    "duration_seconds": None, "pretty_score": 87, "family_importance_score": 0, "indexed_at": 1727000000.0,
    "asset_guid": "0A1B2C3D-FAKE-GUID-0001",
}


def make_tab(photo=PHOTO, count=1234, total=20000, index=11):
    tab = PhotosTab()
    tab._last_load_time = float("inf")    # no background load during the test
    tab._photo, tab._count, tab._total_apple, tab._index = photo, count, total, index
    return tab


def draw(tab, width=110, height=40):
    screen = TUIScreen(width, height)
    tab.draw(screen, 0)
    return screen


def rows(screen):
    return ["".join(c.char for c in line).rstrip() for line in screen.next_frame]


def row_with(lines, needle):
    matches = [l for l in lines if needle in l]
    assert len(matches) == 1, (needle, lines)
    return matches[0]


@pytest.mark.unit
class TestSummary:
    def test_the_tiles_say_how_much_is_indexed_and_where_the_browsing_stands(self):
        lines = rows(draw(make_tab(), width=170))
        top = row_with(lines, "Browsing")
        assert "1,234" in top.split("Browsing")[0]               # the Indexed tile's value
        assert "12 of 1,234" in top
        assert any("6%" in l and "20,000" in l for l in lines)

    def test_an_indexing_outcome_is_shown_in_its_colour(self):
        tab = make_tab()
        tab._process_status_type, tab._process_status = "error", "Error: disk full"
        screen = draw(tab, width=170)
        lines = rows(screen)
        line = row_with(lines, "disk full")
        assert screen.next_frame[lines.index(line)][line.index("disk full")].style.fg == Theme.COLOR_RED

    def test_a_running_batch_shows_its_progress(self):
        tab = make_tab()
        tab._processing, tab._process_status = True, "Processing... 3/200 done"
        assert any("3/200" in l for l in rows(draw(tab, width=170)))


@pytest.mark.unit
class TestPhoto:
    def test_the_description_and_the_details_are_drawn(self):
        lines = rows(draw(make_tab()))
        assert any("Charlotte and Frances on the jetty" in l for l in lines)
        assert sum("wooden jetty" in l for l in lines) >= 2      # the long text wraps
        assert "Sweden, Dalarna, Leksand" in row_with(lines, "Location")
        assert "Charlotte, Frances" in row_with(lines, "Persons")
        assert "0A1B2C3D-FAKE-GUID-0001" in row_with(lines, "Id ")

    def test_a_score_is_a_meter_as_long_as_the_score_with_its_number(self):
        lines = rows(draw(make_tab()))
        pretty, family = row_with(lines, "Pretty score"), row_with(lines, "Family score")
        assert pretty.rstrip().endswith("87") and family.rstrip().endswith("0")
        assert pretty.count("█") > 10 and family.count("█") == 0
        # both numbers sit in the same column, after meters of equal length
        assert len(pretty) - 2 == len(family) - 1

    def test_the_details_are_beside_the_description_on_a_wide_screen_and_under_it_on_a_narrow_one(self):
        wide = rows(draw(make_tab(), width=SIDE_BY_SIDE_MIN))
        line = row_with(wide, "Details")
        assert line.index("Charlotte and Frances") < line.index("Details")
        narrow = rows(draw(make_tab(), width=SIDE_BY_SIDE_MIN - 1))
        assert narrow.index(row_with(narrow, "Charlotte and Frances on")) < narrow.index(row_with(narrow, "Details"))

    def test_nothing_runs_past_the_right_edge(self):
        for width in (80, 110, 170):
            assert all(len(l) <= width for l in rows(draw(make_tab(), width=width)))

    def test_a_short_screen_cuts_the_details_instead_of_failing(self):
        lines = rows(draw(make_tab(), height=18))
        assert any("Browsing" in l for l in lines)

    def test_an_empty_library_says_so_and_draws_no_details(self):
        lines = rows(draw(make_tab(photo=None, count=0, total=0, index=0)))
        assert not any("Details" in l or "Pretty score" in l for l in lines)
        assert any("No photos" in l for l in lines)
        assert any("Ctrl-P" in l for l in lines[:-2])            # how to index the first one
