"""Tests for start.sh's old-server takedown logic.

Every historical auto-restart failed because pgrep missed the old process and the
replacement died on a port collision (Errno 48), so this logic must stay correct:
kill the caller-supplied pid, force-kill TERM-immune survivors, fall back to pgrep,
and refuse to launch while the server ports are still held.

The script runs with stubbed commands (bash functions sourced via $BASH_ENV):
`sleep` is a no-op (the 10s/5s wait loops become instant), `lsof`/`pgrep` answer
from a staged process table (process_table.py: who listens where, which Keep
processes run, whose they are), and `open`/`osascript`/`tail` are recorded or
neutralized so no Terminal windows appear. `kill` is a bash builtin
and cannot be stubbed, so victims are real processes — spawned double-forked so
launchd reaps them and the script's `kill -0` liveness checks never see a zombie.

The role (the Keep server, or an iMessage satellite) is read from a config.yaml
beside the script; without one the install is a server, as on the Python side.
"""
import os
import select
import signal
import subprocess
import time
from pathlib import Path

import pytest

from .process_table import OTHER_CHECKOUT, OTHER_USER, ProcessTable, stubs

START_SH = Path(__file__).resolve().parents[2] / 'start.sh'
ROLE_SH = START_SH.parent / 'scripts' / 'role.sh'
PYTHON_STUB = Path(__file__).parent / 'data' / 'keep_sh' / 'bin' / 'python3'   # records its launch in KEEP_TEST_CALLS
SATELLITE_PORT = 8901
SATELLITE_CONFIG = (f'role: imessage_satellite\nimessage_satellite:\n  port: {SATELLITE_PORT}\n'
                    '  token: kT7vQ2mX9pL4wZ8nB3cR6yH1sD5fG0jA\n  lan_access: false\n')

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


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@pytest.fixture
def bystander():
    """A real process the script must leave alone — staged in the table as another user's or
    another checkout's — ended by the test, since the script does not."""
    pids = []

    def spawn() -> int:
        pids.append(spawn_orphan('/bin/sleep 300'))
        return pids[-1]
    yield spawn
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run_on_a_terminal(command: list[str], env: dict) -> subprocess.CompletedProcess:
    """Run with a pseudo-terminal as stdout, so the script's `[ -t 1 ]` half runs; what it
    printed there is the result's stdout."""
    master, slave = os.openpty()
    try:
        process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=slave, stderr=subprocess.PIPE)
    finally:
        os.close(slave)
    shown = b''
    deadline = time.time() + 30
    while time.time() < deadline:
        ready, _, _ = select.select([master], [], [], 0.05)
        if ready:
            try:
                chunk = os.read(master, 65536)
            except OSError:      # the last writer closed its end
                break
            if not chunk:
                break
            shown += chunk
        elif process.poll() is not None:
            break
    os.close(master)
    process.wait(timeout=30)
    return subprocess.CompletedProcess(command, process.returncode, shown.decode(errors='replace'),
                                       process.stderr.read().decode(errors='replace'))


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
        (tmp_path / 'scripts').mkdir()
        (tmp_path / 'scripts' / 'role.sh').write_text(ROLE_SH.read_text())
        self.table = ProcessTable(tmp_path)
        self.tui_calls = tmp_path / 'tui_calls'
        self.open_calls = tmp_path / 'open_calls'
        self.osascript_calls = tmp_path / 'osascript_calls'
        self.tty_report = tmp_path / 'tty_report'
        self.env_file = tmp_path / 'stubs.bash'
        # `open` plays the launched window: if a tty was staged, it reports it the way
        # cli_start.sh does. `osascript` records its arguments and swallows the script.
        self.env_file.write_text(
            'sleep() { :; }\n'
            'date() { echo "2026-10-13 03:00:00"; }\n'   # every rlog line forks one otherwise
            + stubs(tmp_path) +
            f'open() {{ echo "$@" >> "{self.open_calls}"; '
            f'[ -f "{self.tty_report}" ] && cp "{self.tty_report}" "{tmp_path}/log/cli_start.tty"; :; }}\n'
            f'osascript() {{ echo "$@" >> "{self.osascript_calls}"; cat >/dev/null; }}\n'
            'tail() { :; }\n')

    def config(self, text: str):
        """The install's config.yaml; none means a Keep server."""
        (self.dir / 'config.yaml').write_text(text)

    def comes_up_on(self, port: int):
        """The launched process listens on `port` from the moment the launch window opens."""
        with self.env_file.open('a') as env:
            env.write(f'open() {{ echo "$@" >> "{self.open_calls}"; echo "{port} 4242 {os.getuid()}" >> "{self.dir}/listeners"; }}\n')

    def with_a_console_to_launch(self):
        """`.venv/bin/python3`, which start.sh hands the terminal to, as a recording stub."""
        (self.dir / '.venv' / 'bin').mkdir(parents=True)
        (self.dir / '.venv' / 'bin' / 'python3').symlink_to(PYTHON_STUB)

    @property
    def console_launches(self) -> list[str]:
        return self.tui_calls.read_text().splitlines() if self.tui_calls.exists() else []

    def set_reported_tty(self, tty: str):
        self.tty_report.write_text(tty + '\n')

    @property
    def closed_ttys(self) -> list[str]:
        if not self.osascript_calls.exists():
            return []
        return [line.split()[-1] for line in self.osascript_calls.read_text().splitlines()]

    def run(self, *args, terminal: bool = False) -> subprocess.CompletedProcess:
        env = dict(os.environ, BASH_ENV=str(self.env_file), KEEP_TEST_CALLS=str(self.tui_calls))
        command = ['bash', str(self.script), *map(str, args)]
        if terminal:
            return run_on_a_terminal(command, env)
        # capture_output also makes stdout a pipe, so [ -t 1 ] is false and the
        # interactive tail is skipped, matching a detached auto-restart.
        return subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)

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
        harness.table.process(victim)
        result = harness.run()
        assert result.returncode == 0
        assert wait_dead(victim), f"pgrep-found pid {victim} still alive"
        assert harness.launched

    def test_kills_port_owner_reported_by_lsof(self, harness):
        # No caller pid and no pgrep match — the port owner reported by lsof
        # must still be found and killed. The staged table keeps reporting
        # the pid after death, so the run then hits the refuse-to-launch guard;
        # the assertion that matters here is that the owner is dead.
        victim = spawn_orphan('/bin/sleep 300')
        harness.table.listener(8888, victim)
        result = harness.run()
        assert wait_dead(victim), f"port-owner pid {victim} still alive"
        assert result.returncode == 1
        assert not harness.launched

    def test_refuses_to_launch_while_ports_held(self, harness):
        harness.table.listener(43523, DEAD_PID)
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


CLI_START_SH = START_SH.parent / 'scripts' / 'cli_start.sh'


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
        script = tmp_path / 'scripts' / 'cli_start.sh'
        script.parent.mkdir()
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


class TestTrialFallback:
    """start.sh <pid> <branch> guards a PR trial (PullRequestTrial): the branch is checked out
    again and the server relaunched when the trial's server never listens, and left alone when it
    does. `git` is a recording stub; the launch window's `open` marks the server as up when the
    test says the PR boots."""

    def _arm(self, harness, boots: bool):
        # Each second of the wait is a subshell for lsof; three are enough to see the
        # loop give up (the default 120 would run 16 before the pgrep check ends it).
        with harness.env_file.open('a') as env:
            env.write('TRIAL_TIMEOUT=3\n'
                      f'git() {{ echo "$@" >> "{harness.dir}/git_calls"; }}\n')
        if boots:
            harness.comes_up_on(8888)

    def _git_calls(self, harness) -> list[str]:
        path = harness.dir / 'git_calls'
        return path.read_text().splitlines() if path.exists() else []

    def test_a_pr_whose_server_never_comes_up_is_rolled_back_and_the_branch_relaunched(self, harness):
        self._arm(harness, boots=False)
        result = harness.run(DEAD_PID, 'release')
        assert result.returncode == 0, result.stderr
        assert self._git_calls(harness) == ['checkout release']
        assert len(harness.open_calls.read_text().splitlines()) == 2, "the previous branch is launched again"
        note = (harness.dir / 'log' / 'trial_rollback.txt').read_text()
        assert 'back on release' in note
        assert 'going back to release' in harness.restart_log

    def test_a_pr_whose_server_listens_is_left_running(self, harness):
        self._arm(harness, boots=True)
        result = harness.run(DEAD_PID, 'release')
        assert result.returncode == 0, result.stderr
        assert self._git_calls(harness) == []
        assert len(harness.open_calls.read_text().splitlines()) == 1
        assert not (harness.dir / 'log' / 'trial_rollback.txt').exists()
        assert 'the server is up' in harness.restart_log

    def test_a_failed_checkout_leaves_the_server_down_and_says_so(self, harness):
        self._arm(harness, boots=False)
        with harness.env_file.open('a') as env:
            env.write('git() { echo "error: pathspec did not match" >&2; return 1; }\n')
        result = harness.run(DEAD_PID, 'release')
        assert result.returncode == 1
        assert len(harness.open_calls.read_text().splitlines()) == 1
        assert 'git checkout release failed' in harness.restart_log

    def test_an_ordinary_restart_waits_for_nothing(self, harness):
        self._arm(harness, boots=False)
        result = harness.run(DEAD_PID)
        assert result.returncode == 0
        assert 'PR trial' not in harness.restart_log
        assert self._git_calls(harness) == []


class TestOnlyThisInstallsProcesses:
    """A satellite usually runs under a second macOS user of the server's Mac, and someone may
    run both from two checkouts under one user: whatever start.sh takes for "the old process" it
    kills, and whatever it takes for a held port stops the launch — so neither may ever be the
    other install's."""

    def test_a_satellite_start_leaves_the_other_users_server_alone_and_launches(self, harness, bystander):
        server = bystander()
        harness.config(SATELLITE_CONFIG)
        harness.table.process(server, user=OTHER_USER, checkout=OTHER_CHECKOUT)
        for port in (8888, 43523, SATELLITE_PORT):      # even on the satellite's own port it is not ours to kill
            harness.table.listener(port, server, user=OTHER_USER)
        result = harness.run()
        assert result.returncode == 0, harness.restart_log
        assert alive(server), "the other user's server was killed"
        assert harness.launched, "the other user's ports were taken for busy"
        assert 'force-killing' not in harness.restart_log

    def test_a_server_start_leaves_the_other_users_satellite_alone_and_launches(self, harness, bystander):
        satellite = bystander()
        harness.table.process(satellite, user=OTHER_USER, checkout=OTHER_CHECKOUT)
        harness.table.listener(8888, satellite, user=OTHER_USER)
        result = harness.run()
        assert result.returncode == 0, harness.restart_log
        assert alive(satellite)
        assert harness.launched

    def test_every_lookup_names_the_calling_user(self, harness):
        harness.config(SATELLITE_CONFIG)
        harness.run(DEAD_PID, terminal=True)
        assert harness.table.lookups, "the script looked nothing up"
        me = f'-u {os.getuid()}'
        assert [call for call in harness.table.lookups if me not in call and '-d cwd' not in call] == []

    def test_the_same_users_server_in_another_checkout_survives_a_satellite_start(self, harness, bystander):
        server = bystander()
        harness.config(SATELLITE_CONFIG)
        harness.table.process(server, checkout=OTHER_CHECKOUT)     # same name, same user
        harness.table.listener(8888, server)
        result = harness.run()
        assert result.returncode == 0, harness.restart_log
        assert alive(server), "a name match killed another checkout's process"
        assert harness.launched

    def test_the_same_users_satellite_in_another_checkout_survives_a_server_start(self, harness, bystander):
        satellite = bystander()
        harness.table.process(satellite, checkout=OTHER_CHECKOUT)
        harness.table.listener(SATELLITE_PORT, satellite)
        result = harness.run()
        assert result.returncode == 0, harness.restart_log
        assert alive(satellite)
        assert harness.launched

    def test_a_start_typed_in_another_case_still_finds_its_own_process(self, harness):
        """macOS disks ignore case and lsof reports a folder as the disk has it: ~/keep/start.sh
        must find the process running from ~/Keep."""
        typed = harness.dir.parent / harness.dir.name.upper() / 'start.sh'
        if not typed.exists() or harness.dir.name.upper() == harness.dir.name:
            pytest.skip("this disk tells upper case from lower")
        victim = spawn_orphan('/bin/sleep 300')
        harness.table.process(victim, checkout=str(harness.dir))
        harness.script = typed
        result = harness.run()
        assert result.returncode == 0
        assert wait_dead(victim), "the old process was not recognised as this checkout's"

    def test_a_process_whose_folder_cannot_be_read_is_not_taken_for_this_installs(self, harness, bystander):
        stranger = bystander()
        with harness.env_file.open('a') as env:      # pgrep finds it, lsof cannot say where it runs
            env.write(f'pgrep() {{ echo {stranger}; }}\n')
        result = harness.run()
        assert result.returncode == 0
        assert alive(stranger)


class TestSatellite:
    """`role: imessage_satellite`: the same process and the same launch, on the satellite's own
    port and with no console at the end."""

    def test_only_the_satellites_own_port_is_ever_asked_about(self, harness):
        harness.config(SATELLITE_CONFIG)
        harness.run(terminal=True)
        assert harness.table.ports_asked == {SATELLITE_PORT}

    def test_a_server_never_asks_about_a_satellites_port(self, harness):
        harness.with_a_console_to_launch()
        harness.comes_up_on(5151)
        harness.run(terminal=True)
        assert harness.table.ports_asked == {8888, 43523, 5151}

    def test_the_port_is_8898_when_the_config_names_none(self, harness):
        """A contract with common/imessage/Satellite.DEFAULT_PORT, which the process binds then."""
        from common.imessage.Satellite import DEFAULT_PORT
        harness.config('role: imessage_satellite\nimessage_satellite:\n  token: kT7vQ2mX9pL4wZ8nB3cR6yH1sD5fG0jA\n')
        harness.run()
        assert harness.table.ports_asked == {DEFAULT_PORT}

    def test_the_old_satellite_is_found_on_its_port_and_stopped(self, harness):
        victim = spawn_orphan('/bin/sleep 300')
        harness.config(SATELLITE_CONFIG)
        harness.table.listener(SATELLITE_PORT, victim)
        harness.run()
        assert wait_dead(victim), f"the satellite's port owner {victim} is still alive"

    def test_a_relaunch_after_an_update_stops_the_handed_over_pid_and_launches(self, harness):
        """Updater.relaunch("start.sh") runs `bash start.sh <pid>` with no terminal and no further argument."""
        victim = spawn_orphan('/bin/sleep 300')
        harness.config(SATELLITE_CONFIG)
        result = harness.run(victim)
        assert result.returncode == 0, harness.restart_log
        assert wait_dead(victim), f"the old satellite {victim} is still alive"
        assert harness.launched
        assert harness.console_launches == []
        assert 'PR trial' not in harness.restart_log

    def test_a_held_satellite_port_stops_the_launch(self, harness):
        harness.config(SATELLITE_CONFIG)
        harness.table.listener(SATELLITE_PORT, DEAD_PID)
        result = harness.run()
        assert result.returncode == 1
        assert not harness.launched

    def test_on_a_terminal_it_says_the_satellite_is_up_and_opens_no_console(self, harness):
        harness.config(SATELLITE_CONFIG)
        harness.with_a_console_to_launch()
        harness.comes_up_on(SATELLITE_PORT)
        result = harness.run(terminal=True)
        assert result.returncode == 0, result.stdout
        assert harness.console_launches == [], "a satellite has no console to hand the terminal to"
        said = result.stdout.splitlines()[-2:]
        assert str(SATELLITE_PORT) in said[0] and 'stop.sh' in said[1]

    def test_on_a_terminal_a_satellite_that_exits_while_starting_fails_the_script(self, harness):
        """A refused config ends the process within a second: the terminal is not left on a 60 s wait."""
        harness.config(SATELLITE_CONFIG)
        harness.with_a_console_to_launch()
        result = harness.run(terminal=True)
        assert result.returncode == 1
        assert harness.console_launches == []
        assert 'process is gone' in harness.restart_log

    def test_on_a_terminal_a_satellite_that_never_listens_fails_the_script_after_the_wait(self, harness):
        harness.config(SATELLITE_CONFIG)
        harness.table.process(DEAD_PID)       # there, in the table, and never listening
        with harness.env_file.open('a') as env:
            # the kill finds DEAD_PID gone at once; the port wait is what this test is about
            env.write('seq() { if [ "$2" = 120 ]; then command seq 1 14; else command seq "$@"; fi; }\n')
        result = harness.run(terminal=True)
        assert result.returncode == 1
        assert 'process is gone' not in harness.restart_log
        assert harness.console_launches == []

    def test_the_server_still_gets_its_console(self, harness):
        harness.with_a_console_to_launch()
        harness.comes_up_on(5151)
        harness.run(terminal=True)
        assert harness.console_launches == ['python3 tui.py localhost']


class TestARoleThePythonSideRefuses:
    """floxy_ai_server.py does not start on an unknown `role` or an unusable satellite port.
    start.sh must not guess "server" there: it would stop the owners of the server's ports and
    launch into a process that exits — so it stops nothing and starts nothing."""

    @pytest.mark.parametrize('config', [
        'role: imessage-satellite\n',
        'role: satellite\n',
        'role: imessage_satellite\nimessage_satellite:\n  port: eighty\n',
        'role: imessage_satellite\nimessage_satellite:\n  port: 70000\n',
    ], ids=['a misspelt role', 'half a role', 'a port that is a word', 'a port out of range'])
    def test_nothing_is_stopped_and_nothing_is_started(self, harness, bystander, config):
        running = bystander()
        harness.config(config)
        harness.table.process(running)
        harness.table.listener(8888, running)
        result = harness.run(running)
        assert result.returncode == 1
        assert alive(running), "the running process was stopped on a config the new one cannot start on"
        assert not harness.launched
        assert 'config.yaml' in harness.restart_log
