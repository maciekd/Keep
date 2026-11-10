"""The macOS calls the typing dots are made of: an app's Accessibility tree, read and pressed, and the state of the login session.

This is the whole contact of `TypingIndicator` with the Mac, kept this small so the tests put an
in-memory tree in its place (the unit conftest refuses the real one). pyobjc is imported at the
first call, never at import: a process that shows no dots does not load it, and the module
imports on a Mac without the grant. An element is whatever the Accessibility API hands out —
opaque to the caller, compared with `same`. A call that fails answers None, False or an empty
list; only a framework that cannot be loaded raises (`ImportError`)."""
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Optional

# How long an app gets to answer one Accessibility call (the system's default is 6 s): a
# Messages that hangs must not hold a reply's thread for the length of a tree walk.
MESSAGING_TIMEOUT = 1.5
ON_CONSOLE_KEY = "kCGSSessionOnConsoleKey"
SCREEN_LOCKED_KEY = "CGSSessionScreenIsLocked"      # absent while the screen is unlocked


def _frameworks() -> SimpleNamespace:
    """pyobjc's bindings, loaded when the first call needs them."""
    import AppKit
    import ApplicationServices
    import CoreFoundation
    import Quartz
    import objc
    ApplicationServices.AXUIElementSetMessagingTimeout(ApplicationServices.AXUIElementCreateSystemWide(),
                                                       MESSAGING_TIMEOUT)
    return SimpleNamespace(ax=ApplicationServices, quartz=Quartz, appkit=AppKit, cf=CoreFoundation, objc=objc)


@dataclass(frozen=True)
class Session:
    """The login session this process runs in, as the window server sees it."""
    on_console: bool        # it is the session on the Mac's screen, not one switched away from
    locked: bool


class Accessibility:
    """Reads, sets and presses the Accessibility elements of an app, and tells what state the login session is in."""

    def __init__(self, load: Optional[Callable[[], Any]] = None):
        self._load = load       # how the frameworks are loaded; None: pyobjc's own
        self._loaded = None

    @property
    def _mac(self):
        if self._loaded is None:
            self._loaded = (self._load or _frameworks)()
        return self._loaded

    def trusted(self) -> bool:
        """Whether macOS lets this process use Accessibility — the grant of the app it holds
        responsible for it (Terminal for a Keep started by start.sh). Never prompts."""
        return bool(self._mac.ax.AXIsProcessTrusted())

    def session(self) -> Optional[Session]:
        """None for a process without a login session at the window server."""
        state = self._mac.quartz.CGSessionCopyCurrentDictionary()
        if state is None:
            return None
        return Session(bool(state.get(ON_CONSOLE_KEY, False)), bool(state.get(SCREEN_LOCKED_KEY, False)))

    def application(self, bundle_id: str):
        """The running app's root element, None when it is not running."""
        with self._mac.objc.autorelease_pool():
            running = self._mac.appkit.NSRunningApplication.runningApplicationsWithBundleIdentifier_(bundle_id)
            if not running:
                return None
            return self._mac.ax.AXUIElementCreateApplication(running[0].processIdentifier())

    def attribute(self, element, name: str):
        """The attribute's value — a str, a list of elements, an element — or None when the
        element does not have it or no longer answers."""
        failure, value = self._mac.ax.AXUIElementCopyAttributeValue(element, name, None)
        if failure or value is None:
            return None
        if isinstance(value, str):
            return str(value)
        return list(value) if isinstance(value, (list, tuple)) or hasattr(value, "objectAtIndex_") else value

    def set_attribute(self, element, name: str, value: str) -> bool:
        return not self._mac.ax.AXUIElementSetAttributeValue(element, name, value)

    def actions(self, element) -> list[str]:
        """The names of the actions an element takes; a custom one is a few lines, its title first."""
        failure, names = self._mac.ax.AXUIElementCopyActionNames(element, None)
        return [] if failure or names is None else [str(name) for name in names]

    def perform(self, element, action: str) -> bool:
        return not self._mac.ax.AXUIElementPerformAction(element, action)

    def same(self, first, second) -> bool:
        """Whether two elements are one thing on the screen: each read hands out a new object."""
        return bool(self._mac.cf.CFEqual(first, second))
