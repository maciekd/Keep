"""The default browser of the machine the server runs on, opened on a URL for the console's user.

A terminal makes a wrapped link clickable only when it honours OSC 8 hyperlinks; Terminal.app
does not, and its own URL detection stops at the row's end, so the console opens the link
itself on the key — on the server's Mac, which is where the TUI's user sits everywhere but
over `tui.py <host>`, and where an OAuth redirect to localhost has to land anyway.
"""
import subprocess

from common import Platform
from common.logging import error, info

TIMEOUT = 5


def open_url(url: str) -> bool:
    """Open `url` in the default browser; False when the system could not."""
    if not Platform.is_mac():
        info("Browser: links are opened on macOS only")
        return False
    try:
        subprocess.run(["open", url], check=True, capture_output=True, timeout=TIMEOUT)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        error(f"Browser: open failed for {url[:80]}: {e}")
        return False
    info(f"Browser: opened {url[:120]}")
    return True
