"""The default browser of the machine the server runs on, opened on a URL for the console's user.

A terminal makes a wrapped link clickable only when it honours OSC 8 hyperlinks; Terminal.app
does not, and its own URL detection stops at the row's end, so the console opens the link
itself on the key — on the server's Mac, which is where the TUI's user sits everywhere but
over `tui.py <host>`, and where an OAuth redirect to localhost has to land anyway.
"""
import os
import subprocess

from common import Platform
from common.logging import error, info

TIMEOUT = 5
DISPLAY_VARIABLES = ("WAYLAND_DISPLAY", "DISPLAY")


def has_display() -> bool:
    """Whether a Linux process has a desktop session to show a window on (a server usually has none)."""
    return any(os.environ.get(name) for name in DISPLAY_VARIABLES)


def _opener() -> list[str] | None:
    """The command that opens a URL on this system, None when there is none to try."""
    if Platform.is_mac():
        return ["open"]
    if Platform.current() == Platform.LINUX and has_display():
        return ["xdg-open"]
    return None


def open_url(url: str) -> bool:
    """Open `url` in the default browser; False when the system could not — a Linux server
    without a desktop session has no browser to open."""
    opener = _opener()
    if opener is None:
        info(f"Browser: no desktop session to open a link in on {Platform.label()}")
        return False
    try:
        subprocess.run([*opener, url], check=True, capture_output=True, timeout=TIMEOUT)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        error(f"Browser: {opener[0]} failed for {url[:80]}: {e}")
        return False
    info(f"Browser: opened {url[:120]}")
    return True
