"""The Keep.app shortcut on the Desktop, an AppleScript applet that runs `keep.sh`, built once
the first-time setup sees Full Disk Access granted (`common/FullDiskAccess`).

`keep.sh` opens a terminal (Ghostty, else iTerm2, else Terminal) and attaches the console to
the running server, or starts the server first; the applet only runs that script, so the server
is still launched by start.sh through Terminal and the Mac's grants keep applying to it.
"""
import subprocess
from pathlib import Path

from common import Platform
from common import Applet
from common.Paths import AISERVER_DIR
from common.logging import info

DESKTOP = Path.home() / "Desktop"
APP_NAME = "Keep.app"
ICON = Applet.ICON
SCRIPT_NAME = "keep.sh"


def ensure(desktop: Path = DESKTOP, repo: Path = AISERVER_DIR, icon: Path = ICON, run=subprocess.run) -> bool:
    """Build `Keep.app` on the Desktop unless it is there. Returns whether it is there now —
    never on a system that is not a Mac, where an applet means nothing."""
    if not Platform.is_mac():
        info(f"Desktop shortcut: an applet is a Mac's — not built on {Platform.current()}")
        return False
    if not desktop.is_dir():
        info(f"Desktop shortcut: no {desktop} folder — not built")
        return False
    return Applet.build(desktop / APP_NAME, repo / SCRIPT_NAME, icon, run)
