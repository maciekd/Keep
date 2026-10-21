"""autostart.sh, the login item's script: it starts the server through start.sh when nothing is
running and leaves a server that is up — or still booting — alone. It never opens a console.

Run like keep.sh's tests: the script is copied into a tmp tree beside the stub `start.sh`, and
the checked-in fake bin under `tests/unit/data/keep_sh/` answers for `lsof` and `pgrep`."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

AUTOSTART_SH = Path(__file__).resolve().parents[2] / 'autostart.sh'
STUBS = Path(__file__).parent / 'data' / 'keep_sh'
BASH = shutil.which('bash')


class Harness:
    def __init__(self, tmp_path):
        self.root = tmp_path / 'Keep'
        self.root.mkdir()
        (self.root / 'autostart.sh').write_text(AUTOSTART_SH.read_text())
        (self.root / 'start.sh').symlink_to(STUBS / 'start.sh')
        self.calls = tmp_path / 'calls.log'
        counters = tmp_path / 'counters'
        counters.mkdir()
        self.env = {'KEEP_TEST_CALLS': str(self.calls), 'KEEP_TEST_COUNTERS': str(counters)}

    def server(self, listening: bool, process: bool):
        self.env['KEEP_TEST_LISTENS_AT'] = '1' if listening else None
        self.env['KEEP_TEST_GONE_AT'] = None if process else '1'

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
