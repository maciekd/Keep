import time
from typing import List

from assistant.Telemetry import Telemetry
from interfaces.console import TUIState
from interfaces.console.tabs.TabGroup import TabGroup
from interfaces.console.tabs.TabView import TabView
from common.logging import fine, info

PREVIOUS_KEYS = ('shift_left', '/')
NEXT_KEYS = ('shift_right', '|')


class TabNavigator:
    """Owns which top-level tab is selected and routes the Shift+arrow navigation keys.

    Shift+Left/Right move along the row that has focus: the top bar, or the sub-tab bar of a
    TabGroup once Shift+Down has moved focus into it. Shift+Up hands focus back to the top
    bar. Tab and Shift+Tab walk every tab in one flat cycle, sub-tabs included, for keyboards
    without Shift+arrows (a phone terminal) and for terminals that keep Shift+Up/Down to
    themselves (Terminal.app). Any other key is not navigation and is left for the active tab."""

    def __init__(self, tabs: List[TabView], start: str | None = None):
        """`start` names the tab selected at start-up; the first tab when omitted or unknown."""
        self.tabs = tabs
        names = [t.name for t in tabs]
        self.index = names.index(start) if start in names else 0
        if start is not None and start not in names:
            info(f"TUI start tab {start!r} not found among {names}; starting on {names[0]}.")
        # Whether the user has ever navigated tabs; the tab bar's hint stands out until then
        self.navigated = bool(TUIState.load("navigator").get("navigated", False))
        self._shown_since = time.monotonic()

    @property
    def active_tab(self) -> TabView:
        return self.tabs[self.index]

    @property
    def group(self) -> "TabGroup | None":
        tab = self.active_tab
        return tab if isinstance(tab, TabGroup) else None

    @property
    def sub_focused(self) -> bool:
        group = self.group
        return group is not None and group.focused

    @property
    def path(self) -> str:
        """The tab on screen: a top-level tab's name, or `Group/Sub-tab`."""
        tab = self.active_tab
        return f"{tab.name}/{tab.content_tab().name}" if isinstance(tab, TabGroup) else tab.name

    def handle_key(self, key: str) -> bool:
        """Returns True when the key changed the tab selection or focus."""
        before = self.path
        handled = self._navigate(key)
        if handled:
            self._report_change(before)
        if handled and not self.navigated:
            info("First tab navigation - remembering it so the navigation hint is muted from now on.")
            self.navigated = True
            TUIState.save("navigator", navigated=True)
        return handled

    def _report_change(self, before: str):
        """The telemetry's tab event — one per key that put another tab on screen, however
        many selections the walk took; a focus move between the two bars shows the same tab."""
        shown = self.path
        if shown == before:
            return
        now = time.monotonic()
        Telemetry.get_instance().tab_change(shown, before, now - self._shown_since)
        self._shown_since = now

    def _navigate(self, key: str) -> bool:
        group = self.group
        if key == 'shift_down' and group is not None and not group.focused:
            group.focused = True
            fine(f"TUI focus moved into the {group.name} sub-tabs.")
            return True
        if key == 'shift_up' and group is not None and group.focused:
            group.focused = False
            fine("TUI focus moved back to the top-level tabs.")
            return True
        if key in ('tab', 'shift_tab'):
            self._walk(1 if key == 'tab' else -1)
            return True
        if key in PREVIOUS_KEYS or key in NEXT_KEYS:
            delta = -1 if key in PREVIOUS_KEYS else 1
            if group is not None and group.focused:
                group.select(delta)
            else:
                self._select(delta)
            return True
        return False

    def _walk(self, delta: int):
        """One step through the flat cycle of leaves: the next sub-tab while the open group
        has one in that direction, else the next top-level tab, entered at the end nearest
        to where the walk came from."""
        group = self.group
        if group is not None and 0 <= group.index + delta < len(group.tabs):
            group.select(delta)
            group.focused = True
            return
        new = self.tabs[(self.index + delta) % len(self.tabs)]
        if isinstance(new, TabGroup):
            new.index = 0 if delta > 0 else len(new.tabs) - 1
        self._select(delta)
        if isinstance(new, TabGroup):
            new.focused = True

    def _select(self, delta: int):
        old = self.active_tab
        old.on_deactivate()
        self.index = (self.index + delta) % len(self.tabs)
        new = self.active_tab
        new.on_activate()
        fine(f"TUI tab {old.name} -> {new.name}.")

