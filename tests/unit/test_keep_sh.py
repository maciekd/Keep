"""keep.sh, the Desktop shortcut's script: without a terminal it opens one — Ghostty, else
iTerm2, else Terminal — running itself; with a terminal it attaches the console to a running
server or starts the server first.

The script is copied into a tmp tree beside a stub `start.sh`, and a fake bin on PATH answers
for `open` (installed apps per a canned list, every launch recorded), `lsof`, `pgrep`, `sleep`,
`seq` and `python3`. The stubs are checked in under `tests/unit/data/keep_sh/` and steered by
environment variables rather than written per test: macOS assesses an executable file the first
time it runs (~130 ms per new file, free once it is known), so a tree of fresh stubs cost every
test a quarter of a second or more.

Each system's branch runs on either system (`KEEP_OS`): on the Linux one the doubles are `ss` and
GNU `stat` beside `pgrep`, and where the real /usr/bin/stat is not BSD's, the Mac branch's call of
it is answered by process_table's shim (a function, through $BASH_ENV)."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from .process_table import BSD_STAT

KEEP_SH = Path(__file__).resolve().parents[2] / 'keep.sh'
ROLE_SH = KEEP_SH.parent / 'scripts' / 'role.sh'
SATELLITE_PORT = 8901
STUBS = Path(__file__).parent / 'data' / 'keep_sh'
BASH = shutil.which('bash')
MAC, LINUX = 'Darwin', 'Linux'


def script_env(tmp_path: Path, system: str, settings: dict) -> dict:
    """The environment a script of the tree runs in under these doubles, as `system`."""
    env = dict(os.environ, PATH=f"{STUBS / 'bin'}:{os.environ['PATH']}", KEEP_OS=system,
               **{key: value for key, value in settings.items() if value is not None})
    if BSD_STAT:
        (tmp_path / 'bsd_stat.bash').write_text(BSD_STAT)
        env['BASH_ENV'] = str(tmp_path / 'bsd_stat.bash')
    return env


class Harness:
    def __init__(self, tmp_path, system=MAC):
        self.tmp = tmp_path
        self.system = system
        self.root = tmp_path / 'Keep'
        self.root.mkdir()
        (self.root / 'keep.sh').write_text(KEEP_SH.read_text())
        (self.root / 'scripts').mkdir()
        (self.root / 'scripts' / 'role.sh').write_text(ROLE_SH.read_text())
        (self.root / 'start.sh').symlink_to(STUBS / 'start.sh')
        self.calls = tmp_path / 'calls.log'
        counters = tmp_path / 'counters'
        counters.mkdir()
        self.env = {'KEEP_TEST_CALLS': str(self.calls), 'KEEP_TEST_COUNTERS': str(counters)}
        self.installed([])
        self.server(listening=False, process=False)

    def installed(self, apps):
        """`open -Ra <app>` exits 0 for these apps; every other `open` is recorded."""
        self.env['KEEP_TEST_APPS'] = '|'.join(apps)

    def server(self, listening, process):
        self.booting_server(listens_at_call=1 if listening else None)
        self.env['KEEP_TEST_GONE_AT'] = None if process else '1'

    def satellite(self, listening):
        """This install is an iMessage satellite, listening on its port or not running."""
        (self.root / 'config.yaml').write_text(f'role: imessage_satellite\nimessage_satellite:\n  port: {SATELLITE_PORT}\n')
        self.env['KEEP_TEST_PORT'] = str(SATELLITE_PORT)
        self.server(listening=listening, process=listening)

    def booting_server(self, listens_at_call):
        """A process from the start; the port listens from the given lsof call on."""
        self.env['KEEP_TEST_GONE_AT'] = None
        self.env['KEEP_TEST_LISTENS_AT'] = None if listens_at_call is None else str(listens_at_call)

    def dying_server(self, gone_at_call):
        """Never listening; the process is gone from the given pgrep call on."""
        self.env['KEEP_TEST_LISTENS_AT'] = None
        self.env['KEEP_TEST_GONE_AT'] = str(gone_at_call)

    def polls(self, count):
        """The wait loop runs this many rounds instead of the script's 120."""
        self.env['KEEP_TEST_POLLS'] = str(count)

    def run(self, with_tty):
        env = script_env(self.tmp, self.system, self.env)
        command = [BASH, str(self.root / 'keep.sh')]
        if with_tty:
            master, slave = os.openpty()
            try:
                result = subprocess.run(command, stdin=slave, env=env, capture_output=True, text=True,
                                        close_fds=False, timeout=10)
            finally:
                os.close(master)
                os.close(slave)
        else:
            result = subprocess.run(command, stdin=subprocess.DEVNULL, env=env, capture_output=True, text=True,
                                    close_fds=False, timeout=10)
        return result

    def recorded(self):
        return self.calls.read_text().splitlines() if self.calls.exists() else []


@pytest.fixture
def harness(tmp_path):
    """The Mac's branch."""
    return Harness(tmp_path)


@pytest.fixture(params=[MAC, LINUX])
def either(request, tmp_path):
    """With a terminal the script does the same on both systems."""
    return Harness(tmp_path, request.param)


class TestOpeningATerminal:
    def test_ghostty_is_preferred_and_runs_the_script_inside(self, harness):
        harness.installed(['Ghostty', 'iTerm', 'Terminal'])
        harness.run(with_tty=False)
        assert harness.recorded() == [f"open -na Ghostty --args -e {harness.root / 'keep.sh'}"]

    def test_iterm_when_there_is_no_ghostty(self, harness):
        harness.installed(['iTerm', 'Terminal'])
        harness.run(with_tty=False)
        assert harness.recorded() == [f"open -a iTerm {harness.root / 'keep.sh'}"]

    def test_terminal_is_the_fallback(self, harness):
        harness.installed([])
        harness.run(with_tty=False)
        assert harness.recorded() == [f"open -a Terminal {harness.root / 'keep.sh'}"]

    def test_nothing_runs_in_the_window_less_half(self, harness):
        harness.server(listening=True, process=True)
        result = harness.run(with_tty=False)
        assert 'start.sh ran' not in result.stdout
        assert not any(c.startswith('python3') for c in harness.recorded())


class TestWithoutATerminalOnLinux:
    """No Desktop shortcut runs it there, and there is no window to open."""

    def test_it_says_so_and_opens_nothing(self, tmp_path):
        harness = Harness(tmp_path, LINUX)
        harness.installed(['Ghostty', 'iTerm', 'Terminal'])
        harness.server(listening=True, process=True)
        result = harness.run(with_tty=False)
        assert result.returncode != 0
        assert 'terminal' in result.stderr
        assert harness.recorded() == [] and 'start.sh ran' not in result.stdout


class TestInsideTheTerminal:
    def test_a_listening_server_gets_the_console(self, either):
        harness = either
        harness.server(listening=True, process=False)
        result = harness.run(with_tty=True)
        assert harness.recorded() == ['python3 tui.py localhost']
        assert 'start.sh ran' not in result.stdout

    def test_a_booting_server_is_waited_for_and_then_gets_the_console(self, either):
        harness = either
        harness.booting_server(listens_at_call=4)
        result = harness.run(with_tty=True)
        assert 'waiting' in result.stdout
        assert harness.recorded() == ['sleep 1'] * 3 + ['python3 tui.py localhost']

    def test_a_server_shutting_down_is_waited_out_and_a_new_one_started(self, either):
        harness = either
        harness.dying_server(gone_at_call=3)
        result = harness.run(with_tty=True)
        assert 'start.sh ran' in result.stdout
        assert 'python3 tui.py localhost' not in harness.recorded()
        assert harness.recorded().count('sleep 1') == 2

    def test_a_process_that_never_listens_gets_the_console_in_the_end(self, either):
        harness = either
        harness.server(listening=False, process=True)
        harness.polls(1)
        harness.run(with_tty=True)
        assert harness.recorded() == ['sleep 1', 'python3 tui.py localhost']

    def test_no_server_starts_one(self, either):
        harness = either
        result = harness.run(with_tty=True)
        assert 'start.sh ran' in result.stdout
        assert harness.recorded() == []


class TestSatellite:
    """An iMessage satellite has no console: keep.sh never launches one for it."""

    def test_a_running_satellite_is_reported_and_left_alone(self, harness):
        harness.satellite(listening=True)
        result = harness.run(with_tty=True)
        assert result.returncode == 0
        assert str(SATELLITE_PORT) in result.stdout
        assert 'start.sh ran' not in result.stdout
        assert harness.recorded() == []

    def test_a_satellite_that_is_down_is_started_without_a_console(self, harness):
        harness.satellite(listening=False)
        result = harness.run(with_tty=True)
        assert 'start.sh ran' in result.stdout
        assert harness.recorded() == []


class TestOnlyThisUsersServer:
    def test_the_console_is_not_attached_to_the_other_users_server(self, either):
        """The Mac's other user runs a Keep on the same port; this user's is not running."""
        harness = either
        harness.env['KEEP_TEST_FOREIGN_PROCESS'] = '1'
        harness.env['KEEP_TEST_FOREIGN_PORT'] = '8888'
        result = harness.run(with_tty=True)
        assert 'start.sh ran' in result.stdout
        assert harness.recorded() == []
