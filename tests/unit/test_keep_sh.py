"""keep.sh, the Desktop shortcut's script: without a terminal it opens one — Ghostty, else
iTerm2, else Terminal — running itself; with a terminal it attaches the console to a running
server or starts the server first.

The script is copied into a tmp tree beside a stub `start.sh`, and a fake bin on PATH answers
for `open` (installed apps per a canned list, every launch recorded), `lsof`, `pgrep`, `sleep`,
`seq` and `python3`. The stubs are checked in under `tests/unit/data/keep_sh/` and steered by
environment variables rather than written per test: macOS assesses an executable file the first
time it runs (~130 ms per new file, free once it is known), so a tree of fresh stubs cost every
test a quarter of a second or more."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

KEEP_SH = Path(__file__).resolve().parents[2] / 'keep.sh'
STUBS = Path(__file__).parent / 'data' / 'keep_sh'
BASH = shutil.which('bash')


class Harness:
    def __init__(self, tmp_path):
        self.root = tmp_path / 'Keep'
        self.root.mkdir()
        (self.root / 'keep.sh').write_text(KEEP_SH.read_text())
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
        env = dict(os.environ, PATH=f"{STUBS / 'bin'}:{os.environ['PATH']}",
                   **{key: value for key, value in self.env.items() if value is not None})
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
    return Harness(tmp_path)


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


class TestInsideTheTerminal:
    def test_a_listening_server_gets_the_console(self, harness):
        harness.server(listening=True, process=False)
        result = harness.run(with_tty=True)
        assert harness.recorded() == ['python3 tui.py localhost']
        assert 'start.sh ran' not in result.stdout

    def test_a_booting_server_is_waited_for_and_then_gets_the_console(self, harness):
        harness.booting_server(listens_at_call=4)
        result = harness.run(with_tty=True)
        assert 'waiting' in result.stdout
        assert harness.recorded() == ['sleep 1'] * 3 + ['python3 tui.py localhost']

    def test_a_server_shutting_down_is_waited_out_and_a_new_one_started(self, harness):
        harness.dying_server(gone_at_call=3)
        result = harness.run(with_tty=True)
        assert 'start.sh ran' in result.stdout
        assert 'python3 tui.py localhost' not in harness.recorded()
        assert harness.recorded().count('sleep 1') == 2

    def test_a_process_that_never_listens_gets_the_console_in_the_end(self, harness):
        harness.server(listening=False, process=True)
        harness.polls(3)
        harness.run(with_tty=True)
        assert harness.recorded() == ['sleep 1'] * 3 + ['python3 tui.py localhost']

    def test_no_server_starts_one(self, harness):
        result = harness.run(with_tty=True)
        assert 'start.sh ran' in result.stdout
        assert harness.recorded() == []
