"""Relaunching a server that died: a launchd agent of the user's account that opens
`Keep Autostart.app` every `INTERVAL` seconds, so the login item's script (`autostart.sh`: start
the server through start.sh unless one is up or starting) runs between logins too. The root
config key `watchdog` (off by default) turns it on; `sync` brings the Mac to the key's state at
boot and whenever the key is set. An iMessage satellite applies the key at each start as well
(`common/imessage/Satellite.stay_up`; its installer writes it on). The agent is one per macOS user,
not per install: two installs under one user would each bring it to their own key.

The agent itself touches nothing of the checkout: launchd runs only the system `open`, and the
applet — a program macOS holds responsible, with the grants it was given when the setup first
ran it — does the rest. A launchd `bash` cd-ing into a checkout under ~/Documents would instead
hang on a consent dialog nobody is there to answer (issue #127). `open -g` keeps the applet,
which `common/Applet` builds hidden from the Dock, from taking focus every round. The plist
lives in ~/Library/LaunchAgents; what `open` prints goes to ~/Library/Logs/Keep, also outside the
checkout. A user can switch the agent off under System Settings > General > Login Items &
Extensions > Allow in the Background; the next `sync` logs that launchd no longer has it.
"""
import os
import subprocess
from pathlib import Path

from common import Applet, Autostart
from common.logging import info, trace, error

KEY = "watchdog"
LABEL = "se.keep.watchdog"
INTERVAL = 300
PLIST = Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"
OUT_LOG = Path.home() / "Library" / "Logs" / "Keep" / "watchdog.log"
LAUNCHCTL_TIMEOUT = 30


def plist_text(app: Path = Autostart.APP, out_log: Path | None = None) -> str:
    out_log = out_log or OUT_LOG
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/open</string>
        <string>-g</string>
        <string>-a</string>
        <string>{app}</string>
    </array>
    <key>StartInterval</key>
    <integer>{INTERVAL}</integer>
    <key>RunAtLoad</key>
    <false/>
    <key>StandardOutPath</key>
    <string>{out_log}</string>
    <key>StandardErrorPath</key>
    <string>{out_log}</string>
</dict>
</plist>
'''


def _domain() -> str:
    return f"gui/{os.getuid()}"


def _launchctl(args: list[str], run) -> tuple[bool, str]:
    """(succeeded, what launchctl said)."""
    try:
        result = run(["launchctl"] + args, capture_output=True, text=True, timeout=LAUNCHCTL_TIMEOUT)
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, f"launchctl could not run: {e}"
    return result.returncode == 0, (result.stderr or result.stdout).strip()


def loaded(run=subprocess.run) -> bool:
    """Whether launchd has the agent in the user's domain."""
    ok, _ = _launchctl(["print", f"{_domain()}/{LABEL}"], run)
    return ok


def sync(enabled: bool, app: Path = Autostart.APP, script: Path = Autostart.SCRIPT, plist: Path | None = None,
         run=subprocess.run) -> str | None:
    """Install and load the agent, or unload and remove it, to match the config key. Returns None
    when the Mac is in the wanted state, else why not (logged as well). `plist`: `PLIST` unless
    named — read at the call, so whoever moves the module's path moves every write."""
    return enable(app, script, plist, run) if enabled else disable(plist, run)


def enable(app: Path = Autostart.APP, script: Path = Autostart.SCRIPT, plist: Path | None = None,
           run=subprocess.run) -> str | None:
    plist = plist or PLIST
    # launchd is asked before anything is built or written: a caller that may not reach it (the
    # unit suite's guard) is stopped here, with the applet and the plist as they were
    is_loaded = loaded(run)
    fresh = not Applet.current(app)
    if not Applet.build(app, script, run=run):
        return _fail(f"{app.name} could not be built (the log says why)")
    text = plist_text(app)
    same = plist.exists() and plist.read_text() == text
    if same and is_loaded:
        info(f"Watchdog: the agent is installed and loaded — the server is relaunched within {INTERVAL}s of dying")
        return None
    if plist.exists() and is_loaded:
        trace("Watchdog: the agent's plist changed — reloading it")
        _launchctl(["bootout", f"{_domain()}/{LABEL}"], run)
    plist.parent.mkdir(parents=True, exist_ok=True)
    plist.write_text(text)
    ok, said = _launchctl(["bootstrap", _domain(), str(plist)], run)
    if not ok:
        return _fail(f"launchd would not load {plist.name}: {said or 'no reason given'}")
    info(f"Watchdog: {plist.name} {'written' if not same else 'kept'} and loaded — launchd opens {app.name} "
         f"every {INTERVAL}s, so a dead server is relaunched")
    if fresh:
        Autostart.run_once(app, run)
    return None


def disable(plist: Path | None = None, run=subprocess.run) -> str | None:
    plist = plist or PLIST
    if loaded(run):
        ok, said = _launchctl(["bootout", f"{_domain()}/{LABEL}"], run)
        if not ok:
            return _fail(f"launchd would not unload the agent: {said or 'no reason given'}")
        info("Watchdog: the agent is unloaded — a dead server stays dead until the next login or a start by hand")
    if plist.exists():
        plist.unlink()
        info(f"Watchdog: {plist.name} removed")
    else:
        trace("Watchdog: off and not installed — nothing to do")
    return None


def _fail(why: str) -> str:
    error(f"Watchdog: {why}")
    return why
