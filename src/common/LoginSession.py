"""The macOS login a Keep process was started in, and whether that login has ended.

A process outlives its login when WindowServer restarts or the user logs out: it still answers on
its port and reads chat.db, but reaches no app of the user's new login — a send to Messages does
not compile (-2741) and Terminal cannot be opened. Each login has an audit session id, which the
window server publishes for every logged-in user. `record()` notes the user's at launch, in this
process and in `LOGIN_SESSION_FILE` for `autostart.sh`, which restarts a process whose login is no
longer one of the user's. It is read with `ioreg`, which answers inside an ended login too; `pgrep`
and `launchctl` do not. A Mac's alone; nothing is recorded on Linux.
"""
import os
import plistlib
import subprocess

from common import Platform
from common.Paths import CACHE_DIR
from common.logging import info, trace

LOGIN_SESSION_FILE = CACHE_DIR / "login_session"
IOREG = ["/usr/sbin/ioreg", "-n", "Root", "-d1", "-a"]

_started_in: str | None = None


def current() -> list[str] | None:
    """The audit session ids of this user's logins — none when the user is logged out; None when
    they could not be read."""
    try:
        result = subprocess.run(IOREG, capture_output=True, timeout=5)
        if result.returncode != 0:
            trace(f"LoginSession: ioreg failed ({result.returncode})")
            return None
        sessions = plistlib.loads(result.stdout).get("IOConsoleUsers", [])
        return [str(s["kCGSSessionAuditIDKey"]) for s in sessions
                if s.get("kCGSSessionUserIDKey") == os.getuid() and "kCGSSessionAuditIDKey" in s]
    except Exception as e:
        trace(f"LoginSession: the logins could not be read ({type(e).__name__}: {e})")
        return None


def record():
    """Note the login this process starts in: in memory for `ended()`, on disk for autostart.sh."""
    global _started_in
    if not Platform.is_mac():
        return
    logins = current()
    _started_in = logins[0] if logins else None
    try:
        if _started_in is None:
            LOGIN_SESSION_FILE.unlink(missing_ok=True)
            info("LoginSession: no login of this user on the Mac was found — nothing recorded")
            return
        LOGIN_SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
        LOGIN_SESSION_FILE.write_text(_started_in + "\n")
        info(f"LoginSession: started in login {_started_in}")
    except OSError as e:
        info(f"LoginSession: {LOGIN_SESSION_FILE} could not be written ({e}) — autostart.sh cannot tell an ended login")


def ended() -> str:
    """Why this process's login has ended, or "" while it is one of the user's, or is not known."""
    if _started_in is None:
        return ""
    logins = current()
    if logins is None or _started_in in logins:
        return ""
    now = f"the user's login is now {', '.join(logins)}" if logins else "the user is logged out"
    return f"started in login {_started_in}, which has ended: {now}"
