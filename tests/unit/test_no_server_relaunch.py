"""The unit conftest's guard against a test relaunching the machine's server."""
import os
import subprocess
import sys

import pytest


@pytest.mark.unit
def test_the_real_restart_is_refused_under_the_suite():
    """CodeQualityMaintainer.restart spawns `bash start.sh <our pid>`: start.sh then kills this
    process and starts a server in a Terminal window. A unit test reaching it unstubbed did
    exactly that to the developer's machine (2026-10-14), so under the suite the spawn fails."""
    with pytest.raises(RuntimeError, match="relaunch"):
        subprocess.Popen(["bash", "start.sh", str(os.getpid())], stdout=subprocess.DEVNULL)


@pytest.mark.unit
def test_other_processes_still_start(tmp_path):
    script = tmp_path / "start.sh"      # a script of that name elsewhere, as test_start_sh runs one
    script.write_text("echo hi\n")
    out = subprocess.run(["bash", str(script)], capture_output=True, text=True, timeout=10)
    assert out.stdout.strip() == "hi"


@pytest.mark.unit
def test_a_relaunch_through_any_other_script_is_refused_too():
    """common/Updater relaunches through the script its caller names; what gives a relaunch
    away is this process's pid as the script's first argument."""
    with pytest.raises(RuntimeError, match="relaunch"):
        subprocess.Popen(["bash", "satellite_start.sh", str(os.getpid())], stdout=subprocess.DEVNULL)


# Each command carries an option git and pip refuse, so a guard that let one through would cost
# a usage error and nothing else.
@pytest.mark.unit
@pytest.mark.parametrize("verb", ["pull", "fetch", "checkout", "merge"])
def test_git_cannot_move_the_checkout_the_suite_runs_in(verb):
    with pytest.raises(RuntimeError, match="checkout"):
        subprocess.run(["git", verb, "--no-such-option"], capture_output=True, timeout=10)


@pytest.mark.unit
def test_git_still_moves_a_repository_elsewhere_and_reads_this_one(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, capture_output=True, timeout=10)
    switched = subprocess.run(["git", "checkout", "-q", "-b", "next"], cwd=tmp_path, capture_output=True, timeout=10)
    assert switched.returncode == 0
    assert subprocess.run(["git", "rev-parse", "--git-dir"], capture_output=True, timeout=10).returncode == 0


@pytest.mark.unit
def test_pip_cannot_install_into_the_venv_the_suite_runs_in():
    with pytest.raises(RuntimeError, match="venv"):
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-such-option"], capture_output=True, timeout=10)


@pytest.mark.unit
def test_the_real_messages_app_cannot_be_scripted():
    """A send through Messages.app reaches a real person's phone."""
    with pytest.raises(RuntimeError, match="Messages"):
        subprocess.run(["osascript", "-e", 'tell application "Messages" to return 1'], capture_output=True, timeout=10)


@pytest.mark.unit
@pytest.mark.parametrize("folder", ["Library/Messages/chat.db",
                                    "Library/Application Support/AddressBook/Sources/X/AddressBook-v22.abcddb"])
def test_the_macs_own_messages_and_contacts_cannot_be_opened(folder):
    import sqlite3
    from pathlib import Path
    with pytest.raises(RuntimeError, match="Messages or AddressBook"):
        sqlite3.connect(f"file:{Path.home() / folder}?mode=ro", uri=True)


@pytest.mark.unit
def test_the_messages_host_defaults_point_away_from_this_mac():
    from pathlib import Path
    from common.imessage.Contacts import Contacts
    from common.imessage.MessagesDatabase import MessagesDatabase
    import common.imessage.Contacts as contacts_module
    home = str(Path.home() / "Library")
    assert not MessagesDatabase().path.startswith(home)
    assert not str(contacts_module.ADDRESS_BOOK_DIR).startswith(home)
    assert len(Contacts(MessagesDatabase())) == 0


# Guarded since the iMessage satellite brings the Mac to its `watchdog` key and makes itself a
# login item at each start. Should the guard let one through, the label is nobody's and the
# AppleScript does not compile.
@pytest.mark.unit
@pytest.mark.parametrize("verb", ["bootstrap", "bootout", "print"])
def test_this_macs_launchd_agents_cannot_be_loaded_unloaded_or_asked_about(verb):
    with pytest.raises(RuntimeError, match="launchd"):
        subprocess.run(["launchctl", verb, "gui/0/se.keep.no-such-agent"], capture_output=True, timeout=10)


@pytest.mark.unit
def test_no_applet_is_built_in_the_checkout(tmp_path):
    with pytest.raises(RuntimeError, match="applet"):
        subprocess.run(["osacompile", "-o", str(tmp_path / "No Such.app"), "-e", "return ("], capture_output=True, timeout=10)


@pytest.mark.unit
@pytest.mark.parametrize("watchdog", [True, False])
def test_an_unstubbed_satellite_start_leaves_this_macs_launch_agents_alone(watchdog, tmp_path):
    """What `Satellite.stay_up` would do under the suite without its stubs, whichever way the
    developer's `watchdog` key stands: the first thing `Watchdog.sync` runs is refused, so no
    plist is written or removed and nothing is loaded."""
    from common import Watchdog
    plist = tmp_path / "LaunchAgents" / "se.keep.watchdog.plist"
    plist.parent.mkdir()
    plist.write_text("as the developer has it")
    with pytest.raises(RuntimeError):
        Watchdog.sync(watchdog, app=tmp_path / "Keep Autostart.app", script=tmp_path / "autostart.sh", plist=plist)
    assert plist.read_text() == "as the developer has it"


def _current_applet(tmp_path):
    from common import Applet
    app = tmp_path / "Keep Autostart.app"
    (app / "Contents").mkdir(parents=True)
    (app / "Contents" / "Info.plist").write_text(f"<plist><key>{Applet.GENERATION}</key><true/></plist>")
    return app


@pytest.mark.unit
def test_an_unstubbed_watchdog_writes_no_plist_where_the_applet_is_built_and_no_agent_installed(tmp_path):
    """The state of a development checkout — the applet current, `watchdog` never turned on:
    nothing is left to build and no plist to compare, so the first launchctl call used to come
    after the plist was written."""
    from common import Watchdog
    plist = tmp_path / "LaunchAgents" / "se.keep.watchdog.plist"
    with pytest.raises(RuntimeError, match="launchd"):
        Watchdog.sync(True, app=_current_applet(tmp_path), script=tmp_path / "autostart.sh", plist=plist)
    assert not plist.parent.exists()


@pytest.mark.unit
def test_the_watchdogs_own_plist_is_not_this_macs_under_the_suite(tmp_path):
    """A sync that names no plist — as the server's and the satellite's do — is aimed at a
    scratch folder, and is refused before it writes even there."""
    from pathlib import Path
    from common import Watchdog
    library = Path.home() / "Library"
    assert library not in Watchdog.PLIST.parents and library not in Watchdog.OUT_LOG.parents
    assert str(library) not in Watchdog.plist_text(tmp_path / "Keep Autostart.app")
    with pytest.raises(RuntimeError, match="launchd"):
        Watchdog.sync(True, app=_current_applet(tmp_path), script=tmp_path / "autostart.sh")
    assert not Watchdog.PLIST.exists()


@pytest.mark.unit
def test_no_watchdog_path_is_fixed_before_the_suite_can_move_it():
    """A default argument is bound when the function is defined: a path under ~/Library there
    would be written whatever the conftest set on the module afterwards."""
    import inspect
    from pathlib import Path
    from common import Watchdog
    library = Path.home() / "Library"
    bound = [f"{function.__name__}({name})"
             for function in (Watchdog.sync, Watchdog.enable, Watchdog.disable, Watchdog.plist_text)
             for name, parameter in inspect.signature(function).parameters.items()
             if isinstance(parameter.default, Path) and library in parameter.default.parents]
    assert bound == []


@pytest.mark.unit
def test_this_macs_login_items_cannot_be_scripted():
    with pytest.raises(RuntimeError, match="login items"):
        subprocess.run(["osascript", "-e", 'tell application "System Events" to make login item at end with ('],
                       capture_output=True, timeout=10)


# Guarded since the typing dots (common/imessage/TypingIndicator): every pyobjc call they are made
# of goes through common.imessage.Accessibility, whose loader the conftest refuses.
@pytest.mark.unit
@pytest.mark.parametrize("call", [
    lambda ax: ax.trusted(),
    lambda ax: ax.session(),
    lambda ax: ax.application("com.apple.MobileSMS"),
    lambda ax: ax.attribute(object(), "AXWindows"),
    lambda ax: ax.set_attribute(object(), "AXValue", "…"),
    lambda ax: ax.actions(object()),
    lambda ax: ax.perform(object(), "AXPress"),
    lambda ax: ax.same(object(), object()),
], ids=["trusted", "session", "application", "attribute", "set_attribute", "actions", "perform", "same"])
def test_this_macs_accessibility_and_login_session_cannot_be_reached(call):
    """A value set in the compose field of the real Messages shows typing dots on a real phone."""
    from common.imessage.Accessibility import Accessibility
    with pytest.raises(RuntimeError, match="Accessibility"):
        call(Accessibility())


@pytest.mark.unit
def test_typing_dots_over_the_real_seam_end_as_a_value_and_touch_nothing():
    """What an unstubbed host would do under the suite: the refusal comes back as the answer's
    error, as the host's contract has it — no dots, no exception, and no call got through."""
    from common.imessage.LocalMessagesHost import LocalMessagesHost
    from common.imessage.MessagesHost import TYPING_FAILED
    host = LocalMessagesHost()
    result = host.typing("iMessage;-;+46701740605", True)
    assert (result.showing, result.reason) == (False, TYPING_FAILED)
    assert host.typing_indicator.showing() == []
