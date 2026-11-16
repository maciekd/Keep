"""Full Disk Access, the one macOS grant the first-time setup needs the user to make at the Mac
itself: the probe for it and the System Settings pane it is made in.

macOS grants Full Disk Access to the app it holds responsible for the server, which is Terminal
(start.sh launches the server through it, cli_start.sh). The probe is whether this process may
read chat.db, the Messages database the grant guards — the same test the iMessage integration's
health check makes. A grant applies only to a relaunched Terminal: macOS offers to quit it when
the toggle goes on, and the server, which has left the Terminal window's session, survives that
and is then restarted (`restart_server`) so the new process runs with the grant.

A developer's Mac has the grant already, so the grant step could never be tested there:
`pretend_missing()` makes this process report the grant missing (FirstRun calls it on a fresh
install when `is_developer` is on). The restart the step ends with starts a process that probes
for real, exactly as a beta install's does.
"""
import sqlite3
import subprocess
from pathlib import Path

from common import Platform
from common.logging import info, trace, error

CHAT_DB = Path.home() / "Library/Messages/chat.db"
# The deep link into System Settings > Privacy & Security > Full Disk Access.
SETTINGS_URL = "x-apple.systempreferences:com.apple.preference.security?Privacy_AllFiles"
SETTINGS_PATH = "System Settings > Privacy & Security > Full Disk Access"

_pretend_missing = False


def pretend_missing(on: bool = True):
    """Report the grant missing for the rest of this process, whatever chat.db says."""
    global _pretend_missing
    _pretend_missing = on
    info(f"Full Disk Access: {'reported missing for this process (developer test of the grant step)' if on else 'probed for real again'}")


def pretending_missing() -> bool:
    return _pretend_missing


def granted(chat_db: Path = CHAT_DB) -> bool:
    """Whether this process may read chat.db. A Mac without Messages set up has no file and
    answers False: the grant is then not observable this way. Only macOS has the grant at all."""
    if not Platform.is_mac():
        trace(f"Full Disk Access probe: no such grant on {Platform.current()}")
        return False
    if _pretend_missing:
        trace("Full Disk Access probe: pretending the grant is missing (developer test)")
        return False
    if not chat_db.exists():
        trace(f"Full Disk Access probe: {chat_db} does not exist — nothing to probe")
        return False
    try:
        with sqlite3.connect(f"file:{chat_db}?mode=ro", uri=True) as conn:
            conn.execute("SELECT 1 FROM message LIMIT 1")
    except sqlite3.Error as e:   # OperationalError when macOS refuses the file; any failure means no access
        trace(f"Full Disk Access probe: {chat_db.name} cannot be read ({e}) — not granted")
        return False
    return True


def observable(chat_db: Path = CHAT_DB) -> bool:
    """Whether the probe can answer at all: chat.db exists only on a Mac with Messages set up."""
    return Platform.is_mac() and chat_db.exists()


def open_settings(run=subprocess.run) -> str | None:
    """Open System Settings on the Full Disk Access pane, on the Mac's own screen. Returns None
    when the pane was opened, else why not."""
    if not Platform.is_mac():
        info(f"Full Disk Access: no System Settings on {Platform.current()} — nothing opened")
        return Platform.mac_only("Full Disk Access")
    try:
        result = run(["open", SETTINGS_URL], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as e:
        error(f"Full Disk Access: System Settings could not be opened ({e})")
        return f"System Settings could not be opened: {e}"
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        error(f"Full Disk Access: open failed ({result.returncode}): {detail}")
        return f"System Settings could not be opened: {detail or f'exit {result.returncode}'}"
    info(f"Full Disk Access: System Settings opened on {SETTINGS_PATH}")
    return None
