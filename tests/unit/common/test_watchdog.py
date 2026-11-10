"""The watchdog: a launchd agent of the user's account opening the login item's applet every few
minutes, installed and loaded or unloaded and removed to match the `watchdog` config key."""
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from common import Applet, Watchdog


class FakeLaunchd:
    """subprocess.run for launchctl (a set of loaded labels), osacompile (laying out a current
    bundle), plutil, codesign and open. `refuse` makes every launchctl call fail with that text."""

    def __init__(self, loaded=(), refuse=None):
        self.loaded = set(loaded)
        self.refuse = refuse
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append(argv)
        if argv[0] == "osacompile":
            contents = Path(argv[argv.index("-o") + 1]) / "Contents"
            (contents / "Resources").mkdir(parents=True)
            (contents / "Info.plist").write_text(f"<plist><key>{Applet.GENERATION}</key><true/></plist>")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if argv[0] != "launchctl":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if self.refuse:
            return SimpleNamespace(returncode=1, stdout="", stderr=self.refuse)
        verb, target = argv[1], argv[2]
        label = target.rsplit("/", 1)[-1]
        if verb == "print":
            return SimpleNamespace(returncode=0 if label in self.loaded else 113, stdout="", stderr="")
        if verb == "bootout":
            self.loaded.discard(label)
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if verb == "bootstrap":
            self.loaded.add(Watchdog.LABEL)
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise AssertionError(f"unexpected launchctl verb {verb}")

    def launchctl(self, verb):
        return [a for a in self.calls if a[0] == "launchctl" and a[1] == verb]

    def opened(self):
        return [a for a in self.calls if a[0] == "open"]


@pytest.fixture
def tree(tmp_path):
    return tmp_path / "Keep Autostart.app", tmp_path / "autostart.sh", tmp_path / "LaunchAgents" / "keep.plist"


def current_bundle(app: Path):
    (app / "Contents").mkdir(parents=True)
    (app / "Contents" / "Info.plist").write_text(f"<plist><key>{Applet.GENERATION}</key><true/></plist>")


class TestPlist:
    def test_launchd_runs_only_the_system_open_on_the_applet(self, tree):
        """The agent must never run a shell in the checkout: a launchd bash cd-ing into a
        checkout under ~/Documents hangs on a consent dialog nobody answers (#127)."""
        app, _, _ = tree
        import plistlib
        data = plistlib.loads(Watchdog.plist_text(app).encode())
        assert data["ProgramArguments"][0] == "/usr/bin/open"
        assert data["ProgramArguments"][-1] == str(app)
        assert "-g" in data["ProgramArguments"], "the applet would take focus every round"
        assert data["Label"] == Watchdog.LABEL
        assert data["StartInterval"] == Watchdog.INTERVAL and data["RunAtLoad"] is False

    def test_the_agents_files_live_outside_the_checkout(self):
        """Read from a copy of the module loaded for this test: the suite's conftest moves the
        two paths of the imported one to a scratch folder."""
        import importlib.util
        from common.Paths import AISERVER_DIR
        spec = importlib.util.spec_from_file_location("watchdog_as_installed", Watchdog.__file__)
        installed = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(installed)
        for path in (installed.PLIST, installed.OUT_LOG):
            assert not path.is_relative_to(AISERVER_DIR)
            assert Path.home() / "Library" in path.parents

    def test_the_plist_and_its_log_are_the_modules_at_the_time_of_the_call(self, tree, monkeypatch):
        app, _, _ = tree
        monkeypatch.setattr(Watchdog, "OUT_LOG", app.parent / "Logs" / "watchdog.log")
        assert str(app.parent / "Logs" / "watchdog.log") in Watchdog.plist_text(app)


class TestEnable:
    def test_writes_the_plist_and_loads_it(self, tree):
        app, script, plist = tree
        current_bundle(app)
        mac = FakeLaunchd()
        assert Watchdog.sync(True, app, script, plist, run=mac) is None
        assert plist.read_text() == Watchdog.plist_text(app)
        assert Watchdog.LABEL in mac.loaded
        assert mac.launchctl("bootstrap")[0][-1] == str(plist)
        assert mac.opened() == [], "a current applet already has its grants — no run for them"

    def test_launchd_is_asked_before_anything_is_built_or_written(self, tree):
        """Whoever may not reach launchd is stopped with the Mac as it was — also where nothing
        was left to build and no plist to compare."""
        app, script, plist = tree
        current_bundle(app)
        seen = []

        def refusing(argv, **kwargs):
            seen.append((argv[0], plist.parent.exists()))
            raise PermissionError("launchd is not to be reached from here")
        assert Watchdog.sync(True, app, script, plist, run=refusing)
        assert seen[0] == ("launchctl", False)

    def test_a_missing_applet_is_built_and_run_once_for_its_grants(self, tree):
        app, script, plist = tree
        mac = FakeLaunchd()
        assert Watchdog.sync(True, app, script, plist, run=mac) is None
        assert Applet.current(app)
        assert any(str(app) in a for a in mac.opened())

    def test_an_installed_and_loaded_agent_is_left_alone(self, tree):
        app, script, plist = tree
        current_bundle(app)
        plist.parent.mkdir()
        plist.write_text(Watchdog.plist_text(app))
        mac = FakeLaunchd(loaded=[Watchdog.LABEL])
        assert Watchdog.sync(True, app, script, plist, run=mac) is None
        assert mac.launchctl("bootstrap") == [] and mac.launchctl("bootout") == []

    def test_an_agent_the_user_switched_off_is_loaded_again(self, tree):
        """The plist is there but launchd has dropped it (Login Items > Allow in the Background)."""
        app, script, plist = tree
        current_bundle(app)
        plist.parent.mkdir()
        plist.write_text(Watchdog.plist_text(app))
        mac = FakeLaunchd()
        assert Watchdog.sync(True, app, script, plist, run=mac) is None
        assert len(mac.launchctl("bootstrap")) == 1

    def test_a_changed_plist_is_unloaded_before_the_new_one_loads(self, tree):
        app, script, plist = tree
        current_bundle(app)
        plist.parent.mkdir()
        plist.write_text("<plist>old</plist>")
        mac = FakeLaunchd(loaded=[Watchdog.LABEL])
        assert Watchdog.sync(True, app, script, plist, run=mac) is None
        verbs = [a[1] for a in mac.calls if a[0] == "launchctl"]
        assert verbs.index("bootout") < verbs.index("bootstrap")
        assert plist.read_text() == Watchdog.plist_text(app)

    def test_a_refused_load_is_reported(self, tree):
        app, script, plist = tree
        current_bundle(app)
        mac = FakeLaunchd(refuse="Bootstrap failed: 5: Input/output error")
        failure = Watchdog.sync(True, app, script, plist, run=mac)
        assert failure and "Input/output error" in failure

    def test_a_hung_launchctl_is_reported_not_waited_for(self, tree):
        app, script, plist = tree
        current_bundle(app)

        def hang(argv, **kwargs):
            if argv[0] == "launchctl":
                raise subprocess.TimeoutExpired(argv, kwargs.get("timeout"))
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        assert Watchdog.sync(True, app, script, plist, run=hang)


class TestDisable:
    def test_unloads_and_removes_the_plist(self, tree):
        app, script, plist = tree
        plist.parent.mkdir()
        plist.write_text(Watchdog.plist_text(app))
        mac = FakeLaunchd(loaded=[Watchdog.LABEL])
        assert Watchdog.sync(False, app, script, plist, run=mac) is None
        assert not plist.exists() and Watchdog.LABEL not in mac.loaded

    def test_off_and_absent_is_nothing_to_do(self, tree):
        app, script, plist = tree
        mac = FakeLaunchd()
        assert Watchdog.sync(False, app, script, plist, run=mac) is None
        assert mac.launchctl("bootout") == [] and not plist.exists()

    def test_a_plist_launchd_no_longer_has_is_still_removed(self, tree):
        app, script, plist = tree
        plist.parent.mkdir()
        plist.write_text(Watchdog.plist_text(app))
        assert Watchdog.sync(False, app, script, plist, run=FakeLaunchd()) is None
        assert not plist.exists()


class TestConfigKey:
    def test_the_example_config_has_it_off(self):
        """Contract with config.yaml: the key the server and config_set read, off unless asked."""
        from common import ConfigSchema
        assert ConfigSchema.example_value(Watchdog.KEY) is False
