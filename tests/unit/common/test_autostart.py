"""The login item: the applet in the tree registered with System Events, read back by path,
removed again, and the ways System Events can refuse."""
import subprocess
from types import SimpleNamespace

import pytest

from common import Autostart

# The Mac's own behaviour, whatever system the suite runs on (common/Platform).
pytestmark = pytest.mark.usefixtures("on_a_mac")


class FakeMac:
    """subprocess.run for osascript, osacompile (laying out the bundle) and open. `items` is the
    System Events login item list, changed by `make` and `delete`; `refuse` makes every System
    Events call fail with that stderr, `hang` makes it time out."""

    def __init__(self, items=(), refuse=None, hang=False):
        self.items = list(items)
        self.refuse, self.hang = refuse, hang
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append(argv)
        if argv[0] == "osacompile":
            contents = __import__("pathlib").Path(argv[argv.index("-o") + 1]) / "Contents"
            (contents / "Resources").mkdir(parents=True)
            (contents / "Info.plist").write_text("<plist/>")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if argv[0] != "osascript":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if self.hang:
            raise subprocess.TimeoutExpired(argv, kwargs.get("timeout"))
        if self.refuse:
            return SimpleNamespace(returncode=1, stdout="", stderr=self.refuse)
        script = argv[2]
        if "make login item" in script:
            self.items.append(script.split("path:")[1].split('"')[1])
            return SimpleNamespace(returncode=0, stdout="login item\n", stderr="")
        if "delete" in script:
            wanted = script.split("path is ")[1].split('"')[1]
            self.items = [i for i in self.items if i != wanted]
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=0, stdout="\n".join(self.items) + "\n", stderr="")

    def scripts(self, verb):
        return [a[2] for a in self.calls if a[0] == "osascript" and verb in a[2]]


@pytest.fixture
def tree(tmp_path):
    return tmp_path / "Keep Autostart.app", tmp_path / "autostart.sh"


class TestEnable:
    def test_builds_the_applet_registers_it_and_runs_it_once(self, tree):
        app, script = tree
        mac = FakeMac(items=["/Applications/Other.app"])
        assert Autostart.enable(app, script, run=mac) is None
        assert app.exists()
        assert str(app) in mac.items
        assert any(a[0] == "open" and str(app) in a for a in mac.calls)

    def test_an_item_already_there_is_not_made_twice(self, tree):
        app, script = tree
        app.mkdir()
        mac = FakeMac(items=[str(app)])
        assert Autostart.enable(app, script, run=mac) is None
        assert mac.scripts("make login item") == [] and mac.items == [str(app)]
        assert not any(a[0] == "open" for a in mac.calls)

    def test_a_declined_automation_grant_is_named(self, tree):
        app, script = tree
        mac = FakeMac(refuse="execution error: Not authorized to send Apple events to System Events. (-1743)")
        failure = Autostart.enable(app, script, run=mac)
        assert failure and "declined" in failure and "Automation" in failure

    def test_an_unanswered_dialog_is_reported_as_the_timeout(self, tree):
        app, script = tree
        failure = Autostart.enable(app, script, run=FakeMac(hang=True))
        assert failure and "did not answer" in failure

    def test_a_path_with_a_comma_survives_the_listing(self, tmp_path):
        app = tmp_path / "a, b" / "Keep Autostart.app"
        app.mkdir(parents=True)
        mac = FakeMac(items=["/Applications/Other.app", str(app)])
        assert Autostart.registered(app, run=mac) == (True, None)

    def test_a_failed_build_stops_before_system_events(self, tree):
        app, script = tree

        def run(argv, **kwargs):
            assert argv[0] == "osacompile"
            return SimpleNamespace(returncode=1, stdout="", stderr="syntax error")
        failure = Autostart.enable(app, script, run=run)
        assert failure and "could not be built" in failure


class TestDisable:
    def test_removes_the_item_when_it_is_there(self, tree):
        app, _ = tree
        mac = FakeMac(items=[str(app), "/Applications/Other.app"])
        assert Autostart.disable(app, run=mac) is None
        assert mac.items == ["/Applications/Other.app"]

    def test_nothing_to_remove_is_not_a_failure(self, tree):
        app, _ = tree
        mac = FakeMac(items=["/Applications/Other.app"])
        assert Autostart.disable(app, run=mac) is None
        assert mac.scripts("delete") == []

    def test_a_refusal_is_reported(self, tree):
        app, _ = tree
        failure = Autostart.disable(app, run=FakeMac(items=[str(app)], refuse="execution error: something (-1728)"))
        assert failure and "-1728" in failure
