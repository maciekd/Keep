import json
from unittest.mock import Mock

import pytest

import interfaces.console.tabs.TabNavigator as navigator_module
from assistant.Telemetry import Telemetry
from interfaces.console.framework.TUIScreen import TUIScreen
from interfaces.console.tabs import TabBar
from interfaces.console.tabs.TabGroup import TabGroup
from interfaces.console.tabs.TabNavigator import TabNavigator
from interfaces.console.tabs.TabView import TabView


class FakeTab(TabView):
    def __init__(self, name):
        self._name = name
        self.events = []

    @property
    def name(self):
        return self._name

    def draw(self, screen, frame):
        self.events.append("draw")

    def handle_input(self, key):
        self.events.append(key)
        return True

    def on_activate(self):
        self.events.append("activate")

    def on_deactivate(self):
        self.events.append("deactivate")


def make_nav():
    home, dialog = FakeTab("Home"), FakeTab("Dialog")
    icloud, arlo = FakeTab("iCloud"), FakeTab("Arlo")
    group = TabGroup("Integrations", [icloud, arlo])
    return TabNavigator([home, dialog, group]), (home, dialog, group, icloud, arlo)


def test_default_name_strips_tab_suffix():
    class PhotosTab(FakeTab):
        name = TabView.name

    assert PhotosTab("x").name == "Photos"


def test_shift_left_right_walk_top_level_and_wrap():
    nav, (home, dialog, group, *_) = make_nav()
    assert nav.handle_key('shift_right')
    assert nav.active_tab is dialog
    assert home.events == ["deactivate"] and dialog.events == ["activate"]
    nav.handle_key('shift_right')
    assert nav.active_tab is group
    nav.handle_key('shift_right')
    assert nav.active_tab is home
    nav.handle_key('shift_left')
    assert nav.active_tab is group


def test_slash_and_pipe_still_switch_tabs():
    nav, (home, dialog, *_) = make_nav()
    nav.handle_key('|')
    assert nav.active_tab is dialog
    nav.handle_key('/')
    assert nav.active_tab is home


def test_shift_down_only_enters_a_group():
    nav, (home, *_) = make_nav()
    assert not nav.handle_key('shift_down')
    assert not nav.handle_key('shift_up')
    assert not nav.sub_focused
    assert home.events == []


def test_group_navigation_focus_flow():
    nav, (home, dialog, group, icloud, arlo) = make_nav()
    nav.handle_key('shift_left')  # wraps to Integrations
    assert icloud.events == ["activate"], "entering the group activates its sub-tab"
    assert not nav.sub_focused

    assert nav.handle_key('shift_down')
    assert nav.sub_focused
    assert not nav.handle_key('shift_down'), "a second Shift+Down is left to the tab (page down)"

    nav.handle_key('shift_right')
    assert nav.active_tab is group and group.content_tab() is arlo
    assert icloud.events == ["activate", "deactivate"] and arlo.events == ["activate"]
    nav.handle_key('shift_right')
    assert group.content_tab() is icloud, "sub-tabs wrap around without leaving the group"

    assert nav.handle_key('shift_up')
    assert not nav.sub_focused
    assert not nav.handle_key('shift_up'), "a second Shift+Up is left to the tab (page up)"
    nav.handle_key('shift_right')
    assert nav.active_tab is home, "after Shift+Up, Shift+Right moves along the top row again"


def test_tab_walks_every_leaf_and_wraps():
    # A phone keyboard has Tab but no Shift+arrows, so Tab alone must reach every tab
    nav, (home, dialog, group, icloud, arlo) = make_nav()
    assert nav.handle_key('tab')
    assert nav.active_tab is dialog and not nav.sub_focused
    nav.handle_key('tab')
    assert nav.active_tab is group and group.content_tab() is icloud
    assert nav.sub_focused, "landing on a sub-tab focuses the sub-tab row"
    assert icloud.events == ["activate"]
    nav.handle_key('tab')
    assert group.content_tab() is arlo
    assert icloud.events == ["activate", "deactivate"] and arlo.events == ["activate"]
    nav.handle_key('tab')
    assert nav.active_tab is home, "past the last sub-tab the walk continues on the top row"
    assert not group.focused and arlo.events == ["activate", "deactivate"]


def test_shift_tab_walks_backwards_entering_a_group_at_its_last_sub_tab():
    nav, (home, dialog, group, icloud, arlo) = make_nav()
    assert nav.handle_key('shift_tab')
    assert nav.active_tab is group and group.content_tab() is arlo and nav.sub_focused
    assert icloud.events == [], "the group's remembered first sub-tab is never activated"
    nav.handle_key('shift_tab')
    assert group.content_tab() is icloud
    nav.handle_key('shift_tab')
    assert nav.active_tab is dialog and not nav.sub_focused


def test_shift_arrows_continue_on_the_row_tab_landed_on():
    nav, (home, dialog, group, icloud, arlo) = make_nav()
    nav.handle_key('tab')
    nav.handle_key('tab')  # iCloud
    nav.handle_key('shift_right')
    assert group.content_tab() is arlo, "Shift+Right walks the sub-tabs Tab focused"
    nav.handle_key('shift_up')
    nav.handle_key('shift_right')
    assert nav.active_tab is home


def test_leaving_the_group_drops_sub_focus_and_deactivates_sub_tab():
    nav, (home, dialog, group, icloud, arlo) = make_nav()
    nav.handle_key('shift_left')
    nav.handle_key('shift_down')
    nav.handle_key('shift_right')  # Arlo
    group.focused = True
    nav.index = 2
    nav.handle_key('shift_up')
    nav.handle_key('shift_right')
    assert nav.active_tab is home
    assert not group.focused
    assert arlo.events == ["activate", "deactivate"]
    nav.handle_key('shift_left')
    assert group.content_tab() is arlo, "the group remembers which sub-tab was open"


def test_group_delegates_input_and_draw_to_its_sub_tab():
    nav, (home, dialog, group, icloud, arlo) = make_nav()
    assert group.handle_input('x')
    assert icloud.events == ['x']
    screen = TUIScreen(width=120, height=40)
    group.focused = True
    group.draw(screen, 0)
    assert icloud.events == ['x', 'draw']
    row4 = "".join(cell.char for cell in screen.next_frame[3])
    assert "iCloud" in row4 and "Arlo" in row4


@pytest.fixture
def tab_changes(monkeypatch):
    """The (tab, previous) pairs the navigator reported to the telemetry."""
    sink = Mock()
    monkeypatch.setattr(Telemetry, "get_instance", classmethod(lambda cls: sink))
    return lambda: [c.args[:2] for c in sink.tab_change.call_args_list]


def test_telemetry_gets_one_event_per_tab_shown_with_sub_tabs_under_their_group(tab_changes):
    nav, _ = make_nav()
    for key in ('shift_right', 'tab', 'tab', 'tab'):
        nav.handle_key(key)
    assert tab_changes() == [
        ("Dialog", "Home"),
        ("Integrations/iCloud", "Dialog"),
        ("Integrations/Arlo", "Integrations/iCloud"),
        ("Home", "Integrations/Arlo"),
    ]


def test_telemetry_hears_nothing_when_the_key_leaves_the_same_tab_on_screen(tab_changes):
    nav, _ = make_nav()
    nav.handle_key('shift_left')    # Home -> Integrations
    nav.handle_key('shift_down')    # focus into the sub-tab bar
    nav.handle_key('shift_up')      # and back
    nav.handle_key('x')             # not navigation
    assert tab_changes() == [("Integrations/iCloud", "Home")]


def test_telemetry_dwell_is_the_time_on_the_tab_left(tab_changes, monkeypatch):
    clock = iter([100.0, 107.5, 109.0])
    monkeypatch.setattr(navigator_module.time, "monotonic", lambda: next(clock))
    nav, _ = make_nav()
    nav.handle_key('shift_right')
    nav.handle_key('shift_right')
    dwells = [c.args[2] for c in Telemetry.get_instance().tab_change.call_args_list]
    assert dwells == pytest.approx([7.5, 1.5])


def test_first_navigation_is_remembered_across_instances(tui_state_file):
    nav, _ = make_nav()
    assert not nav.navigated
    nav.handle_key('x')
    assert not nav.navigated and not tui_state_file.exists(), "a non-navigation key does not count"
    nav.handle_key('shift_right')
    assert nav.navigated
    assert json.loads(tui_state_file.read_text()) == {"navigator": {"navigated": True}}
    assert TabNavigator([FakeTab("Home")]).navigated, "a later session starts with the flag set"


def test_unreadable_state_file_means_first_time_user(tui_state_file):
    tui_state_file.write_text("not json")
    assert not TabNavigator([FakeTab("Home")]).navigated


def test_bright_hint_is_white_and_bold():
    def hint_cell(bright):
        screen = TUIScreen(width=60, height=5)
        TabBar.draw_tab_bar(screen, 2, ["Home"], 0, True, TabBar.TOP_BAR, hint="hint ", hint_bright=bright)
        return next(c for c in screen.next_frame[1] if c.char == "h")

    bright, muted = hint_cell(True), hint_cell(False)
    assert bright.style.fg == TabBar.COLOR_ACTIVE_FG and bright.style.bold
    assert muted.style.fg == TabBar.COLOR_INACTIVE_FG and not muted.style.bold


def test_center_text_sits_midway_between_the_tabs_and_the_hint_in_the_hints_gray():
    screen = TUIScreen(width=60, height=5)
    TabBar.draw_tab_bar(screen, 2, ["Home"], 0, True, TabBar.TOP_BAR, hint="hint ", hint_bright=True, center="Keep 1")
    row = "".join(c.char for c in screen.next_frame[1])
    start, hint_start = row.index("Keep 1"), row.index("hint")
    tabs_end = len("  Home".ljust(TabBar.TAB_WIDTH + 1)) + 1
    assert abs((start - tabs_end) - (hint_start - start - len("Keep 1"))) <= 1
    cell = screen.next_frame[1][start]
    assert cell.style.fg == TabBar.COLOR_INACTIVE_FG and not cell.style.bold
    assert cell.style.bg == TabBar.TOP_BAR.fill


def test_center_text_is_dropped_when_the_gap_is_too_narrow():
    screen = TUIScreen(width=24, height=5)
    TabBar.draw_tab_bar(screen, 2, ["Home"], 0, True, TabBar.TOP_BAR, hint="hint ", center="Keep 2026.09.1")
    row = "".join(c.char for c in screen.next_frame[1])
    assert "Keep" not in row and "hint" in row and "Home" in row


def test_start_tab_selects_by_name_and_falls_back_to_the_first():
    home, dialog = FakeTab("Home"), FakeTab("Dialog")
    assert TabNavigator([home, dialog], start="Dialog").active_tab is dialog
    assert TabNavigator([home, dialog], start="Nope").active_tab is home
    assert TabNavigator([home, dialog]).active_tab is home
