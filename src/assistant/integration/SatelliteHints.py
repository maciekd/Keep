"""What a problem with the Mac that holds a Messages account means to the owner, in words to act on.

The host layer (`common/imessage`) reports facts — a reason, a detail, the satellite's own
status. The sentences are written here, for both consumers of a host, because they name what
only the assistant's side knows: its commands (`config_request`, `restart_integration`) and the
config keys of the consumer's section. Every text leads with the action: the prompt shows only
the first 80 characters of an integration's error. The Mac is the satellite's when the host has
an address and this one when it has none; `show_full_disk_access_setting` and `restart_server`
are never offered for a satellite's grants — they act on the server's Mac, and the grant is
needed on the other one. `restart_server` is offered where the server is the one behind."""
import os
import re
from pathlib import Path

from common import Version
from common.Paths import AISERVER_DIR
from common.imessage import Satellite, Wire
from common.imessage.MessagesHost import (BAD_ANSWER, HostHealth, NO_ACCESS, NO_DATABASE, NOT_CONFIGURED, REFUSED,
                                          SatelliteStatus, TYPING_NO_ACCESSIBILITY, TYPING_NO_MESSAGES,
                                          TYPING_NO_WINDOW, TYPING_NOT_ON_CONSOLE, TYPING_SCREEN_LOCKED)
from common.imessage.ServerFollower import FOLLOWED_BRANCHES
from common.logging import trace


def _start_command() -> str:
    """The line that starts the server from a fresh Terminal: `~/Keep/start.sh` for an install
    under the home folder (start.sh cds into its own folder first), the full path otherwise,
    quoted only when it needs it."""
    script = str(AISERVER_DIR / "start.sh")
    home = str(Path.home())
    if script.startswith(home + os.sep):
        script = "~" + script[len(home):]
    if re.search(r"[\s'\"$`\\]", script):
        script = '"' + script.replace("~", "$HOME", 1).replace('"', '\\"') + '"'
    return script


def full_disk_access(address: str = "") -> str:
    """The Full Disk Access hint for the Mac that holds the account — the satellite's, or
    without an address this one; reads on after "…likely" or a name.

    On this Mac the grant goes to Terminal, the app macOS holds responsible for the server
    (start.sh launches it from there, whatever terminal the user typed in), and macOS applies
    a grant to a running Terminal only once it is relaunched: it offers "quit now" or "later"
    when the toggle goes on, and after "later" every restart_server spawns from the same
    ungranted Terminal. The user sits in Terminal during the setup, so quitting it takes the
    chat down too — the way back is start.sh in a fresh Terminal, not restart_server — spelled
    with the install's real path, since a new Terminal opens in the home folder and the
    install folder is the user's choice."""
    if not address:
        return ("needs Full Disk Access for Terminal, then quit Terminal and restart_server(). Tell the user, who "
                "must be at the Mac: 1) show_full_disk_access_setting() opens System Settings > Privacy & Security > "
                "Full Disk Access — turn on Terminal; 2) macOS offers to quit Terminal: accept (the console closes, "
                "the server keeps running, Telegram stays up); 3) restart_server() from Telegram — or, without "
                f"Telegram, open Terminal again and type {_start_command()}")
    return (f"needs Full Disk Access for Terminal on the satellite's Mac ({address}), not on this one. Tell the user, "
            "who must be at that Mac: open System Settings > Privacy & Security > Full Disk Access there, turn on "
            "Terminal, accept macOS's offer to quit Terminal, then open Terminal again and run start.sh in the Keep "
            "folder (~/Keep/start.sh on a default install). show_full_disk_access_setting() and restart_server() "
            "act on the server's Mac and do not help here")


def automation(address: str) -> str:
    return (f"iMessage send blocked: Keep on the satellite's Mac ({address}) is not authorized to control Messages. "
            "At that Mac: System Settings → Privacy & Security → Automation, allow Terminal to control Messages, "
            "then retry.")


def typing_dots(address: str, reason: str, key: str) -> str:
    """Why the typing dots do not show although the chat works (`HostHealth.typing`), with what
    to do at the Mac that holds the account — the satellite's, or this one when `address` is
    empty — and the key that turns the dots, and this warning, off."""
    where = f"the satellite's Mac ({address})" if address else "this Mac"
    what = {
        TYPING_NO_ACCESSIBILITY: (f"typing dots need Accessibility for Terminal on {where}: the user, at that Mac's "
                                  "screen and in the macOS user Keep runs as there, opens System Settings > Privacy & "
                                  "Security > Accessibility and turns on Terminal — no restart, the next reply shows "
                                  "them"),
        TYPING_NOT_ON_CONSOLE: (f"no typing dots: the macOS user that holds the assistant's Messages on {where} is "
                                "not the one on the Mac's screen (another user is switched in, or nobody is logged in "
                                "at the display) — the dots show only while that user's session is the one on screen"),
        TYPING_SCREEN_LOCKED: (f"no typing dots while the screen of {where} is locked: a locked Mac shows Keep no "
                               "window of Messages. A Mac that should show dots must not lock its screen (System "
                               "Settings > Lock Screen there)"),
        TYPING_NO_MESSAGES: f"no typing dots: Messages is not open on {where} — the user opens it and leaves it open",
        TYPING_NO_WINDOW: (f"no typing dots: Messages on {where} shows no window with its conversation list — the "
                           "user opens it (Messages > Window > Messages) and leaves it open"),
    }.get(reason, f"no typing dots: the Accessibility frameworks could not be used on {where} ({reason})")
    return f'{what}. Replies are sent either way; config_set("{key}", "false") turns the dots and this warning off'


def problem(address: str, health: HostHealth, section: str, integration: str) -> str:
    """Why the consumer of `<section>.host` cannot use its satellite, the fix first."""
    host_key, token_key = f"{section}.host", f"{section}.host_token"
    retry = f'restart_integration("{integration}")'
    if health.reason == NOT_CONFIGURED:
        return (f'needs {token_key} and a usable {host_key}: config_request("{token_key}") lets the user type the '
                f"token ({Satellite.TOKEN_KEY} in the satellite's config.yaml), and {host_key} is the satellite's "
                f"address, http://<its address>:{Satellite.DEFAULT_PORT} — or local for this Mac's own Messages; then {retry} "
                f"({health.detail})")
    if health.reason == REFUSED:
        return (f'needs the right {token_key}: config_request("{token_key}") — the user types the value of '
                f"{Satellite.TOKEN_KEY} from the config.yaml on the satellite's Mac ({address}), then {retry} "
                f"({health.detail})")
    if health.reason == NO_ACCESS:
        return f"{full_disk_access(address)} ({health.detail})"
    if health.reason == NO_DATABASE:
        return (f"Messages.app is not set up on the satellite's Mac ({address}): the user signs in to iMessage "
                f"there, in the macOS user Keep runs as ({health.detail})")
    wire = health.satellite.wire if health.satellite else Wire.WIRE_VERSION
    if health.reason == BAD_ANSWER and wire > Wire.WIRE_VERSION + Wire.WIRE_TOLERANCE:
        return (f"restart_server() updates this server, which is the one behind: the iMessage satellite at {address} "
                f"runs a newer Keep than this server can talk to ({health.detail}). The satellite stays as it is; "
                f"{integration} goes on by itself once the server is on its line's latest version")
    if health.reason == BAD_ANSWER and wire < Wire.WIRE_VERSION - Wire.WIRE_TOLERANCE:
        return (f"the iMessage satellite at {address} runs an older Keep than this server can talk to and is "
                f"updating itself to this server's version ({health.detail}). Nothing to do while that takes a few "
                f"minutes; if it stays, re-run the Keep installer on the satellite's Mac")
    if health.reason == BAD_ANSWER:
        return (f"the iMessage satellite at {address} gave an answer this server cannot use ({health.detail}). "
                f"Check that {host_key} is the satellite's address and port ({Satellite.DEFAULT_PORT} unless its config "
                f"says otherwise) "
                f"and that both installs are on the same release line; re-running the installer on the satellite's "
                f"Mac updates it")
    return (f"the iMessage satellite at {address} does not answer ({health.detail}). Ask the user to check that Mac: "
            f"it is on and its user is logged in, Keep runs there (start.sh in its Keep folder), "
            f"{Satellite.LAN_ACCESS_KEY} is true when it is another Mac than the server's, and {host_key} "
            f"is its address")


def warning(address: str, status: SatelliteStatus) -> str:
    """What is out of step with a satellite that works, "" when nothing is: another wire
    version, an update of its own that failed, another version it is not about to leave."""
    own_branch, own_commit = Version.branch(), Version.commit()
    behind = status.commit != own_commit and bool(status.commit) and bool(own_commit)
    theirs = f"{status.version or 'an unknown version'}"
    if status.update_failure:
        return (f"the iMessage satellite at {address} could not update itself to this server's version and keeps "
                f"serving on {theirs}: {status.update_failure}. It tries again within the hour and whenever the "
                f"server's version changes; to fix it by hand, re-run the Keep installer on that Mac")
    if status.wire != Wire.WIRE_VERSION:
        older = "the satellite" if status.wire < Wire.WIRE_VERSION else "this server"
        return (f"the iMessage satellite at {address} speaks wire version {status.wire}, this server "
                f"{Wire.WIRE_VERSION}: it works for now, but {older} is behind and must be updated "
                f"(satellite on {theirs}, server on {Version.detail()})"
                + ("; the satellite is updating itself now" if status.updating else ""))
    if behind and not status.updating:
        if own_branch not in FOLLOWED_BRANCHES:
            trace(f"iMessage satellite {address}: on {theirs}, not following this server's branch "
                  f"({own_branch or 'detached'}) — as intended")
            return ""
        return (f"the iMessage satellite at {address} runs {theirs}, this server {Version.detail()}, and it is not "
                f"updating itself. Nothing to do while the server is the older of the two (its next update settles "
                f"that); otherwise re-run the Keep installer on that Mac")
    return ""
