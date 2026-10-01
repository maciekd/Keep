"""Tests for start.sh's old-server takedown logic.

Every historical auto-restart failed because pgrep missed the old process and the
replacement died on a port collision (Errno 48), so this logic must stay correct:
kill the caller-supplied pid, force-kill TERM-immune survivors, fall back to pgrep,
and refuse to launch while the server ports are still held.

The script runs with stubbed commands (bash functions sourced via $BASH_ENV):
`sleep` is a no-op (the 10s/5s wait loops become instant), `lsof`/`pgrep` replay
canned output, and `open`/`osascript`/`tail` are recorded or neutralized so no
Terminal windows appear. `kill` is a bash builtin
and cannot be stubbed, so victims are real processes — spawned double-forked so
launchd reaps them and the script's `kill -0` liveness checks never see a zombie.
"""
import os
import subprocess
import time
from pathlib import Path

import pytest

START_SH = Path(__file__).resolve().parents[2] / 'start.sh'

# Far above macOS's pid range, so "a pid that is not alive" is guaranteed.
DEAD_PID = 4999999


def spawn_orphan(cmd: str) -> int:
    """Start cmd detached from pytest (double-fork) and return its pid.

    Orphans get reparented to launchd, which reaps them on death — a plain
    subprocess.Popen child would linger as a zombie after start.sh kills it,
    and the script's `kill -0` would still report it alive.
    """
    out = subprocess.run(['bash', '-c', f'{cmd} >/dev/null 2>&1 & echo $!'],
                         capture_output=True, text=True, check=True)
    return int(out.stdout.strip())


def wait_dead(pid: int, timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


class StartShHarness:
    """Stubs are shell functions sourced through $BASH_ENV rather than executables on
    PATH: macOS scans every freshly written executable on its first exec (~150 ms
    each), which made every test cost ~1 s. Functions shadow the real commands and
    are never exec'd."""

    def __init__(self, tmp_path: Path):
        self.dir = tmp_path
        (tmp_path / 'log').mkdir()
        self.script = tmp_path / 'start.sh'
        self.script.write_text(START_SH.read_text())
        self.open_calls = tmp_path / 'open_calls'
        self.osascript_calls = tmp_path / 'osascript_calls'
        self.tty_report = tmp_path / 'tty_report'
        self.env_file = tmp_path / 'stubs.bash'
        # `open` plays the launched window: if a tty was staged, it reports it the way
        # cli_start.sh does. `osascript` records its arguments and swallows the script.
        self.env_file.write_text(
            'sleep() { :; }\n'
            f'lsof() {{ cat "{tmp_path}/lsof_out" 2>/dev/null; return 0; }}\n'
            f'pgrep() {{ cat "{tmp_path}/pgrep_out" 2>/dev/null; return 0; }}\n'
            f'open() {{ echo "$@" >> "{self.open_calls}"; '
            f'[ -f "{self.tty_report}" ] && cp "{self.tty_report}" "{tmp_path}/log/cli_start.tty"; :; }}\n'
            f'osascript() {{ echo "$@" >> "{self.osascript_calls}"; cat >/dev/null; }}\n'
            'tail() { :; }\n')

    def set_lsof_output(self, text: str):
        (self.dir / 'lsof_out').write_text(text + '\n')

    def set_pgrep_output(self, text: str):
        (self.dir / 'pgrep_out').write_text(text + '\n')

    def set_reported_tty(self, tty: str):
        self.tty_report.write_text(tty + '\n')

    @property
    def closed_ttys(self) -> list[str]:
        if not self.osascript_calls.exists():
            return []
        return [line.split()[-1] for line in self.osascript_calls.read_text().splitlines()]

    def run(self, *args) -> subprocess.CompletedProcess:
        env = dict(os.environ, BASH_ENV=str(self.env_file))
        # capture_output also makes stdout a pipe, so [ -t 1 ] is false and the
        # interactive tail is skipped, matching a detached auto-restart.
        return subprocess.run(['bash', str(self.script), *map(str, args)],
                              env=env, capture_output=True, text=True, timeout=30)

    @property
    def restart_log(self) -> str:
        path = self.dir / 'log' / 'restart.log'
        return path.read_text() if path.exists() else ''

    @property
    def launched(self) -> bool:
        return self.open_calls.exists()


@pytest.fixture
def harness(tmp_path):
    return StartShHarness(tmp_path)


class TestStartSh:
    def test_kills_caller_supplied_pid_and_launches(self, harness):
        victim = spawn_orphan('/bin/sleep 300')
        result = harness.run(victim)
        assert result.returncode == 0
        assert wait_dead(victim), f"old server pid {victim} still alive"
        assert harness.launched
        assert f'caller-supplied old pid: {victim}' in harness.restart_log
        assert 'Stopping old server pid(s)' in harness.restart_log

    def test_force_kills_term_immune_survivor(self, harness):
        # A wedged server hangs in graceful shutdown and survives SIGTERM —
        # exactly the failure mode from the Aug 5 CoreAudio wedge.
        # The victim touches a sentinel once its trap is installed; without the
        # wait, start.sh's SIGTERM could land before the trap and kill it
        # gracefully, failing the force-kill assertion spuriously.
        trap_ready = harness.dir / 'trap_ready'
        victim = spawn_orphan(
            f"""bash -c 'trap "" TERM; touch "{trap_ready}"; while :; do /bin/sleep 1; done'""")
        deadline = time.time() + 5.0
        while not trap_ready.exists() and time.time() < deadline:
            time.sleep(0.005)
        assert trap_ready.exists(), "victim never installed its TERM trap"
        result = harness.run(victim)
        assert result.returncode == 0
        assert wait_dead(victim), f"TERM-immune pid {victim} survived force-kill"
        assert 'force-killing' in harness.restart_log
        assert harness.launched

    def test_pgrep_fallback_kills_old_server_without_caller_pid(self, harness):
        victim = spawn_orphan('/bin/sleep 300')
        harness.set_pgrep_output(str(victim))
        result = harness.run()
        assert result.returncode == 0
        assert wait_dead(victim), f"pgrep-found pid {victim} still alive"
        assert harness.launched

    def test_kills_port_owner_reported_by_lsof(self, harness):
        # No caller pid and no pgrep match — the port owner reported by lsof
        # must still be found and killed. The static lsof stub keeps reporting
        # the pid after death, so the run then hits the refuse-to-launch guard;
        # the assertion that matters here is that the owner is dead.
        victim = spawn_orphan('/bin/sleep 300')
        harness.set_lsof_output(str(victim))
        result = harness.run()
        assert wait_dead(victim), f"port-owner pid {victim} still alive"
        assert result.returncode == 1
        assert not harness.launched

    def test_refuses_to_launch_while_ports_held(self, harness):
        harness.set_lsof_output(str(DEAD_PID))
        result = harness.run()
        assert result.returncode == 1
        assert not harness.launched
        assert 'aborting launch' in harness.restart_log

    def test_no_old_process_launches_directly(self, harness):
        result = harness.run()
        assert result.returncode == 0
        assert 'no old server process found' in harness.restart_log
        assert harness.launched


class TestLaunchWindow:
    """start.sh closes the Terminal window it opened and nothing else of the user's: the
    window is identified by the tty cli_start.sh reports, never guessed."""

    def test_closes_exactly_the_window_cli_start_reported(self, harness):
        harness.set_reported_tty('/dev/ttys007')
        result = harness.run()
        assert result.returncode == 0
        assert harness.closed_ttys == ['/dev/ttys007']
        assert 'closing the launch window on /dev/ttys007' in harness.restart_log

    def test_stale_report_from_an_earlier_run_closes_nothing(self, harness):
        # Left behind by a previous start; that window may be the user's own by now.
        (harness.dir / 'log' / 'cli_start.tty').write_text('/dev/ttys003\n')
        result = harness.run()
        assert result.returncode == 0
        assert harness.launched
        assert harness.closed_ttys == []
        assert 'did not report its tty' in harness.restart_log


CLI_START_SH = START_SH.with_name('cli_start.sh')


class TestCliStartSh:
    def test_server_leaves_the_window_session_before_the_tty_is_reported(self, tmp_path):
        """Closing the window sends SIGHUP to its session; the server must have started a
        session of its own (be its leader) by the time start.sh learns the tty."""
        (tmp_path / '.venv' / 'bin').mkdir(parents=True)
        (tmp_path / '.venv' / 'bin' / 'activate').write_text('')
        (tmp_path / 'log').mkdir()
        report = tmp_path / 'server_report'
        (tmp_path / 'floxy_ai_server.py').write_text(
            'import os, pathlib\n'
            f'pathlib.Path({str(report)!r}).write_text('
            '  f"{os.getsid(0)} {os.getpid()} {os.path.exists(\'log/cli_start.tty\')}")\n')
        script = tmp_path / 'cli_start.sh'
        script.write_text(CLI_START_SH.read_text())
        subprocess.run(['bash', str(script)], capture_output=True, text=True, timeout=30, check=True)
        deadline = time.time() + 5.0
        while not report.exists() and time.time() < deadline:
            time.sleep(0.02)
        assert report.exists(), "the fake server never ran"
        sid, pid, tty_seen_at_start = report.read_text().split()
        assert sid == pid, "the server is not a session leader of its own"
        assert sid != str(os.getsid(0))
        assert tty_seen_at_start == 'True', "the tty was reported after the server started"
        assert (tmp_path / 'log' / 'cli_start.tty').read_text().strip()
