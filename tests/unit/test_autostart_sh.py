"""autostart.sh, the login item's and the watchdog's script: it starts the server through
start.sh when nothing is running and leaves a server that is up — or still booting — alone, as
well as a start.sh in flight. It never opens a console.

Run like keep.sh's tests: the script is copied into a tmp tree beside the stub `start.sh`, and
the checked-in fake bin under `tests/unit/data/keep_sh/` answers for `lsof` and `pgrep`.

An install is a Keep server unless its config.yaml says `role: imessage_satellite`; "running" is
then the satellite's own port, and either way only this user's processes of this checkout count."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

AUTOSTART_SH = Path(__file__).resolve().parents[2] / 'autostart.sh'
ROLE_SH = AUTOSTART_SH.parent / 'scripts' / 'role.sh'
SATELLITE_PORT = 8901
SATELLITE_CONFIG = f'role: imessage_satellite\nimessage_satellite:\n  port: {SATELLITE_PORT}\n'
OTHER_CHECKOUT = '/Users/marcus/KeepElsewhere'
STUBS = Path(__file__).parent / 'data' / 'keep_sh'
BASH = shutil.which('bash')


class Harness:
    def __init__(self, tmp_path):
        self.root = tmp_path / 'Keep'
        self.root.mkdir()
        (self.root / 'autostart.sh').write_text(AUTOSTART_SH.read_text())
        (self.root / 'scripts').mkdir()
        (self.root / 'scripts' / 'role.sh').write_text(ROLE_SH.read_text())
        (self.root / 'start.sh').symlink_to(STUBS / 'start.sh')
        self.calls = tmp_path / 'calls.log'
        counters = tmp_path / 'counters'
        counters.mkdir()
        self.env = {'KEEP_TEST_CALLS': str(self.calls), 'KEEP_TEST_COUNTERS': str(counters)}

    def server(self, listening: bool, process: bool):
        self.env['KEEP_TEST_LISTENS_AT'] = '1' if listening else None
        self.env['KEEP_TEST_GONE_AT'] = None if process else '1'

    def satellite(self, listening: bool, process: bool):
        """This install is a satellite; its process is there or not, listening on its port or not."""
        (self.root / 'config.yaml').write_text(SATELLITE_CONFIG)
        self.env['KEEP_TEST_PORT'] = str(SATELLITE_PORT)
        self.server(listening, process)

    def other_users_keep(self, port: int):
        """Another macOS user of this Mac runs a Keep: its process, listening on `port`."""
        self.env['KEEP_TEST_FOREIGN_PROCESS'] = '1'
        self.env['KEEP_TEST_FOREIGN_PORT'] = str(port)

    def run(self):
        env = dict(os.environ, PATH=f"{STUBS / 'bin'}:{os.environ['PATH']}",
                   **{key: value for key, value in self.env.items() if value is not None})
        return subprocess.run([BASH, str(self.root / 'autostart.sh')], stdin=subprocess.DEVNULL, env=env,
                              capture_output=True, text=True, close_fds=False, timeout=10)

    def restart_log(self) -> list[str]:
        log = self.root / 'log' / 'restart.log'
        return log.read_text().splitlines() if log.exists() else []


@pytest.fixture
def harness(tmp_path):
    return Harness(tmp_path)


@pytest.mark.unit
class TestAutostart:
    def test_with_nothing_running_the_server_is_started_through_start_sh(self, harness):
        harness.server(listening=False, process=False)
        result = harness.run()
        assert result.returncode == 0
        assert 'start.sh ran' in result.stdout
        assert len(harness.restart_log()) == 1 and 'starting' in harness.restart_log()[0]

    def test_a_listening_server_is_left_alone(self, harness):
        harness.server(listening=True, process=True)
        result = harness.run()
        assert result.returncode == 0
        assert 'start.sh ran' not in result.stdout
        assert len(harness.restart_log()) == 1 and 'already running' in harness.restart_log()[0]

    def test_a_server_still_booting_is_not_started_twice(self, harness):
        # The process is there but the port does not listen yet: a second start.sh would collide
        # with the one booting.
        harness.server(listening=False, process=True)
        result = harness.run()
        assert 'start.sh ran' not in result.stdout
        assert 'already running' in harness.restart_log()[0]

    def test_a_start_sh_in_flight_is_not_doubled(self, harness):
        # Between start.sh's kill of the old server and its launch of the new one neither the
        # port nor the process exists; the watchdog firing there must not run a second start.sh.
        harness.server(listening=False, process=False)
        harness.env['KEEP_TEST_START_SH_RUNNING'] = '1'
        result = harness.run()
        assert result.returncode == 0
        assert 'start.sh ran' not in result.stdout
        assert len(harness.restart_log()) == 1 and 'start.sh is already running' in harness.restart_log()[0]

    def test_no_console_is_opened_either_way(self, harness):
        for listening in (False, True):
            harness.server(listening=listening, process=listening)
            harness.run()
        assert not harness.calls.exists(), "autostart.sh launched an app — the Desktop shortcut's job, not the login item's"

    def test_the_restart_log_lands_in_the_scripts_own_folder_from_any_cwd(self, harness, tmp_path):
        elsewhere = tmp_path / 'elsewhere'
        elsewhere.mkdir()
        harness.server(listening=True, process=True)
        env = dict(os.environ, PATH=f"{STUBS / 'bin'}:{os.environ['PATH']}",
                   **{key: value for key, value in harness.env.items() if value is not None})
        subprocess.run([BASH, str(harness.root / 'autostart.sh')], cwd=elsewhere, stdin=subprocess.DEVNULL, env=env,
                       capture_output=True, text=True, close_fds=False, timeout=10)
        assert (harness.root / 'log' / 'restart.log').exists()
        assert not (elsewhere / 'log').exists()


@pytest.mark.unit
class TestSatellite:
    def test_a_satellite_that_is_down_is_started(self, harness):
        harness.satellite(listening=False, process=False)
        result = harness.run()
        assert 'start.sh ran' in result.stdout

    def test_a_satellite_listening_on_its_own_port_is_left_alone(self, harness):
        harness.satellite(listening=True, process=False)
        result = harness.run()
        assert 'start.sh ran' not in result.stdout

    def test_a_satellite_still_starting_is_not_started_twice(self, harness):
        harness.satellite(listening=False, process=True)
        result = harness.run()
        assert 'start.sh ran' not in result.stdout

    def test_something_on_the_servers_port_does_not_pass_for_a_running_satellite(self, harness):
        (harness.root / 'config.yaml').write_text(SATELLITE_CONFIG)
        harness.server(listening=True, process=False)      # the double listens on 8888, not on the satellite's port
        result = harness.run()
        assert 'start.sh ran' in result.stdout


@pytest.mark.unit
class TestOnlyThisInstallsProcesses:
    """At the login of a second macOS user the first user's Keep is usually up already: it must
    not pass for this install's, or the satellite would never start (and the other way round)."""

    def test_the_other_users_server_does_not_keep_a_satellite_from_starting(self, harness):
        harness.satellite(listening=False, process=False)
        harness.other_users_keep(port=8888)
        result = harness.run()
        assert 'start.sh ran' in result.stdout

    def test_the_other_users_keep_on_the_same_port_does_not_keep_this_one_from_starting(self, harness):
        harness.server(listening=False, process=False)
        harness.other_users_keep(port=8888)
        result = harness.run()
        assert 'start.sh ran' in result.stdout

    def test_the_same_users_keep_in_another_checkout_does_not_pass_for_this_one(self, harness):
        harness.server(listening=False, process=True)      # a process by that name, running elsewhere
        harness.env['KEEP_TEST_CWD'] = OTHER_CHECKOUT
        result = harness.run()
        assert 'start.sh ran' in result.stdout

    def test_a_start_sh_of_another_checkout_is_not_this_ones_start_in_flight(self, harness):
        harness.server(listening=False, process=False)
        harness.env['KEEP_TEST_START_SH_RUNNING'] = '1'
        harness.env['KEEP_TEST_CWD'] = OTHER_CHECKOUT
        result = harness.run()
        assert 'start.sh ran' in result.stdout
