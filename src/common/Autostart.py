"""Starting the server at login: `Keep Autostart.app` in the repository root, an applet
(`common/Applet`) running `autostart.sh`, registered as a login item of the user's account
(System Settings > General > Login Items & Extensions > Open at Login).

`autostart.sh` starts the server the way it always starts — start.sh, through Terminal, so the
Mac's grants apply — and opens no console; a server that is already up is left alone. The
first-time setup registers the item with the Desktop shortcut (`assistant/FirstRun`); an iMessage
satellite, which has no setup, registers it at its first start that can read Messages
(`common/imessage/Satellite.stay_up`). The user removes it in that pane. The applet lives in the
tree, not on the Desktop, so a tidied Desktop cannot break it; it is gitignored and rebuilt when
missing.

The login item is made through System Events, so the first call asks the user to let Terminal
(the app held responsible for the server) control System Events; a declined dialog is reported
as the reason. On a repository under ~/Documents the applet's first run asks for the folder once
more, under its own name — `enable` runs it once right after registering it, while the user is
at the Mac, so that dialog is not left for the next login. A login item fires at login only: a
Mac that boots to the login window, or to FileVault's unlock screen, starts the server once
someone has signed in.
"""
import subprocess
from pathlib import Path

from common import Applet
from common.Paths import AISERVER_DIR
from common.logging import info, trace, error

APP = AISERVER_DIR / "Keep Autostart.app"
SCRIPT = AISERVER_DIR / "autostart.sh"
SETTINGS_PATH = "System Settings > General > Login Items & Extensions > Open at Login"
# System Events answers at once, unless macOS is asking the user whether Terminal may control
# it: the dialog blocks the call until they answer, and this is how long they get.
OSASCRIPT_TIMEOUT = 90

_LIST = ('tell application "System Events"\n'
         "set AppleScript's text item delimiters to linefeed\n"
         "return (path of every login item) as text\n"
         "end tell")


def _osascript(lines: str, run=subprocess.run) -> tuple[str | None, str | None]:
    """Run an AppleScript; returns (stdout, None) or (None, why not)."""
    try:
        result = run(["osascript", "-e", lines], capture_output=True, text=True, timeout=OSASCRIPT_TIMEOUT)
    except subprocess.TimeoutExpired:
        return None, (f"System Events did not answer within {OSASCRIPT_TIMEOUT}s — the dialog asking whether "
                      f"Terminal may control System Events is probably still open on the Mac's screen")
    except OSError as e:
        return None, f"osascript could not run: {e}"
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        if "-1743" in detail:
            return None, ("the user declined to let Terminal control System Events (the login item cannot be made "
                          "without it; System Settings > Privacy & Security > Automation turns it on)")
        return None, detail or f"osascript exit {result.returncode}"
    return result.stdout, None


def login_items(run=subprocess.run) -> tuple[list[str] | None, str | None]:
    """The paths of the user's login items, or (None, why not)."""
    out, failure = _osascript(_LIST, run)
    if failure:
        return None, failure
    return [line for line in out.splitlines() if line.strip()], None


def registered(app: Path = APP, run=subprocess.run) -> tuple[bool | None, str | None]:
    """Whether the applet is a login item: (verdict, None) or (None, why not)."""
    items, failure = login_items(run)
    if failure:
        return None, failure
    return str(app) in items, None


def enable(app: Path = APP, script: Path = SCRIPT, run=subprocess.run) -> str | None:
    """Build the applet when missing, register it as a login item unless it is one, and run it
    once (see the module note). Returns None when the server now starts at login, else why not."""
    if not Applet.build(app, script, run=run):
        return f"{app.name} could not be built (the log says why)"
    is_registered, failure = registered(app, run)
    if failure:
        error(f"Autostart: the login items could not be read — {failure}")
        return failure
    if is_registered:
        info(f"Autostart: {app.name} is already a login item")
        return None
    make = ('tell application "System Events" to make login item at end with properties '
            f"{{path:{Applet.literal(app)}, hidden:false}}")
    _, failure = _osascript(make, run)
    if failure:
        error(f"Autostart: the login item could not be made — {failure}")
        return failure
    info(f"Autostart: {app.name} registered as a login item — the server starts at login")
    run_once(app, run)
    return None


def run_once(app: Path, run=subprocess.run):
    """Launch the applet now: with the server up its script does nothing, but the applet's own
    grants (a repository under ~/Documents) are asked for while the user is at the Mac. The
    watchdog (`common/Watchdog`) does the same after rebuilding it."""
    try:
        result = run(["open", "-a", str(app)], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as e:
        info(f"Autostart: the applet could not be launched for its first run ({e})")
        return
    if result.returncode != 0:
        info(f"Autostart: the applet's first run failed ({result.returncode}): {(result.stderr or result.stdout).strip()}")
    else:
        trace(f"Autostart: {app.name} launched once, for its grants")


def disable(app: Path = APP, run=subprocess.run) -> str | None:
    """Remove the login item (`//wipe`-style tidying, tests). Returns None when the server no longer
    starts at login, else why not."""
    is_registered, failure = registered(app, run)
    if failure:
        error(f"Autostart: the login items could not be read — {failure}")
        return failure
    if not is_registered:
        info(f"Autostart: {app.name} is not a login item — nothing to remove")
        return None
    delete = f'tell application "System Events" to delete (every login item whose path is {Applet.literal(app)})'
    _, failure = _osascript(delete, run)
    if failure:
        error(f"Autostart: the login item could not be removed — {failure}")
        return failure
    info(f"Autostart: {app.name} removed from the login items")
    return None
