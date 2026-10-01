"""StartupIntro: the first frame starts the flight, the landing types the title beside the
logo and then hands over to the tabs, a key skips, and the logo is the settled object cut to
its lit cells."""
import pytest

from interfaces.console.crystal.Intro import LOGO, FRAME_SECONDS
from interfaces.console.crystal.Scene import Scene
from interfaces.console.framework.TUIScreen import TUIScreen
from interfaces.console.StartupIntro import StartupIntro, Logo, Title, ARMED, FLYING, TITLING, DONE
from interfaces.console.tabs.TabView import CONTENT_TOP


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def make():
    clock = Clock()
    scene = Scene(clock)
    scene.aa = 1
    return StartupIntro(scene), scene, clock


def test_the_first_frame_starts_the_flight_and_owns_the_screen():
    intro, scene, clock = make()
    assert intro.state == ARMED and intro.logo(TUIScreen(80, 24)) is None
    screen = TUIScreen(80, 24)
    assert intro.draw(screen, 0) is True
    assert intro.state == FLYING and intro.playing
    assert scene.flight == pytest.approx(FRAME_SECONDS), "the first frame is one frame into the flight"
    assert (scene.fb.width, scene.fb.height) == (80, 48), "the flight fills the whole screen"


def title_text(screen, logo):
    x, y = Title.position(logo)
    return screen.row_text(x, y, len(Title.TEXT)).rstrip()


def test_landing_types_the_title_beside_the_logo_then_hands_over_to_the_tabs():
    intro, scene, clock = make()
    intro.draw(TUIScreen(80, 24), 0)
    scene.flight = scene.intro.length() - 0.1   # the flight's frames are test_crystal's; this one starts at its last
    clock.now += 0.2
    screen = TUIScreen(80, 24)
    owned = intro.draw(screen, 0)
    assert owned is True and intro.state == TITLING and intro.playing and scene.shot.stage == LOGO
    logo = intro.logo(screen)
    assert logo is not None and title_text(screen, logo) == "", "nothing typed yet on the landing frame"
    x, y = Title.position(logo)
    assert x == logo.x + logo.width + Title.GAP and y == logo.y + logo.height // 2 - Title.LIFT

    clock.now += len(Title.TEXT) / Title.CPS / 2
    screen = TUIScreen(80, 24)
    assert intro.draw(screen, 1) is True
    typed = title_text(screen, logo).rstrip()
    assert 0 < len(typed) < len(Title.TEXT) and Title.TEXT.startswith(typed)
    assert screen.next_frame[y - 1][x - 1].style.fg == Title.COLOR, "the mint the Home chat speaks in"

    clock.now += len(Title.TEXT) / Title.CPS
    screen = TUIScreen(80, 24)
    assert intro.draw(screen, 2) is True, "typed, holding"
    assert title_text(screen, logo) == Title.TEXT and intro.state == TITLING
    clock.now += StartupIntro.HOLD_SECONDS
    screen = TUIScreen(80, 24)
    assert intro.draw(screen, 3) is False and intro.state == DONE and not intro.playing
    assert intro.header(TUIScreen(80, 24), 4) == logo.bottom
    home = TUIScreen(80, 24)
    intro.header(home, 5)
    assert title_text(home, logo) == Title.TEXT, "the title stays beside the logo on Home"


def test_a_key_during_the_title_shows_it_whole_and_ends_the_hold():
    intro, scene, clock = make()
    intro.draw(TUIScreen(80, 24), 0)
    intro.skip()
    clock.now += 0.01
    assert intro.draw(TUIScreen(80, 24), 0) is True and intro.state == TITLING
    intro.skip()
    screen = TUIScreen(80, 24)
    assert intro.draw(screen, 1) is False and intro.state == DONE
    assert title_text(screen, intro.logo(screen)) == Title.TEXT


def test_a_key_skips_to_the_logo():
    intro, scene, clock = make()
    intro.draw(TUIScreen(80, 24), 0)
    clock.now += 1.0
    intro.draw(TUIScreen(80, 24), 0)
    intro.skip()
    assert scene.flight == scene.intro.length()
    clock.now += 0.01
    assert intro.draw(TUIScreen(80, 24), 0) is True and intro.state == TITLING, "landed, the title is next"


def test_the_logo_is_the_lit_rectangle_of_the_final_frame_and_follows_the_screen_size():
    intro, scene, clock = make()
    intro.draw(TUIScreen(100, 40), 0)
    intro.skip()
    clock.now += 0.01
    intro.draw(TUIScreen(100, 40), 0)
    assert intro.state == TITLING
    logo = intro.logo(TUIScreen(100, 40))
    assert logo is not None and logo.width > 2 and logo.height > 1
    assert logo.x < 30 and logo.y < 20, "top-left corner, where logo_x / logo_y put it"
    assert all(Logo._lit(c) for c in logo.cells[0]) or any(Logo._lit(c) for c in logo.cells[0])
    assert intro.logo(TUIScreen(100, 40)) is logo, "cached per size"
    other = intro.logo(TUIScreen(120, 40))
    assert other is not logo and other.size == (120, 40)


def test_the_logo_blits_where_it_was_and_never_over_the_tab_bar():
    intro, scene, clock = make()
    intro.draw(TUIScreen(100, 40), 0)
    intro.skip()
    clock.now += 0.01
    intro.draw(TUIScreen(100, 40), 0)
    logo = intro.logo(TUIScreen(100, 40))
    screen = TUIScreen(100, 40)
    logo.blit(screen)
    lit = [(x, y) for y, row in enumerate(screen.next_frame, 1) for x, c in enumerate(row, 1) if Logo._lit(c)]
    assert lit and min(y for _, y in lit) >= Logo.TOP_ROW == CONTENT_TOP - 1
    assert min(x for x, _ in lit) == logo.x and max(y for _, y in lit) == logo.bottom
    small = TUIScreen(10, 5)
    logo.blit(small)   # clipped, no error


def test_the_logo_stays_whole_on_a_narrow_or_short_screen():
    """A tall thin terminal put the logo's centre at a tenth of the width, closer to the
    left edge than the object's radius, so the cube was cut off (Sep 14)."""
    wide, _, _ = make()
    wide.draw(TUIScreen(200, 50), 0)
    assert wide.scene.logo_at() == (wide.scene.tuning["logo_x"], wide.scene.tuning["logo_y"]), "where it fits nothing moves"
    ex, ey = wide.scene.logo_extent()
    assert 10 < ex < 16 and 10 < ey < 16, "about 26 cells wide and 13 rows tall on that screen"

    def landed_logo(cols, rows):
        intro, scene, clock = make()
        intro.draw(TUIScreen(cols, rows), 0)
        intro.skip()
        clock.now += 0.01
        intro.draw(TUIScreen(cols, rows), 0)
        return intro.logo(TUIScreen(cols, rows)), scene

    logo, scene = landed_logo(40, 60)
    assert logo.x >= 2 and logo.y >= 3, "pushed in from the left edge and under the tab bar"
    assert abs(logo.width - 2 * logo.height) <= 3, "a whole cube is about twice as wide in cells as it is tall"
    x, y = scene.logo_at()
    assert x > scene.tuning["logo_x"] and y == scene.tuning["logo_y"]

    logo, scene = landed_logo(200, 12)
    assert logo.y >= 3 and abs(logo.width - 2 * logo.height) <= 3


def test_a_lifted_header_moves_up_and_is_not_drawn_once_under_the_tab_bar():
    intro, scene, clock = make()
    intro.draw(TUIScreen(100, 40), 0)
    intro.skip()
    clock.now += 0.01
    intro.draw(TUIScreen(100, 40), 0)
    intro.skip()
    intro.draw(TUIScreen(100, 40), 1)
    logo = intro.logo(TUIScreen(100, 40))
    tx, ty = Title.position(logo)

    def lit_rows(lift):
        screen = TUIScreen(100, 40)
        assert intro.header(screen, 2, lift=lift) == logo.bottom
        rows = {y for y, r in enumerate(screen.next_frame, 1) for c in r if Logo._lit(c)}
        return rows, screen.row_text(tx, ty - lift, len(Title.TEXT)).rstrip() if ty - lift >= 1 else ""

    rows, title = lit_rows(0)
    assert min(rows) == logo.y and title == Title.TEXT
    rows, title = lit_rows(2)
    assert min(rows) == max(Logo.TOP_ROW, logo.y - 2) and max(rows) == logo.bottom - 2 and title == Title.TEXT
    rows, _ = lit_rows(logo.bottom - Logo.TOP_ROW + 1)
    assert rows == set(), "under the tab bar: nothing drawn, title included"
    screen = TUIScreen(100, 40)
    intro.header(screen, 3, lift=logo.bottom - Logo.TOP_ROW + 1)
    assert not any(c.char != " " for r in screen.next_frame for c in r)


def test_callbacks_run_when_the_intro_ends_and_at_once_after_it():
    intro, scene, clock = make()
    fired = []
    intro.on_done(lambda: fired.append("a"))
    intro.on_done(lambda: (_ for _ in ()).throw(RuntimeError("boom")))   # a failing one does not stop the next
    intro.on_done(lambda: fired.append("b"))
    intro.draw(TUIScreen(80, 24), 0)
    intro.skip()
    clock.now += 0.01
    intro.draw(TUIScreen(80, 24), 0)
    assert fired == [] and intro.state == TITLING
    intro.skip()
    assert intro.draw(TUIScreen(80, 24), 1) is False and intro.state == DONE
    assert fired == ["a", "b"]
    intro.on_done(lambda: fired.append("late"))
    assert fired == ["a", "b", "late"], "after the intro a callback runs right away"
