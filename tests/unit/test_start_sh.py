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

Each system's branch runs on either system: the harness names it in `KEEP_OS`. The
Mac's is the launch through Terminal; Linux's (scripts/launch_linux.sh, service.sh)
is the systemd user unit, with `systemctl`, `systemd-run`, `ss` and GNU `stat` as
functions of the same staged table — no test reaches the machine's real systemd,
and the unit file lands under a tmp `XDG_CONFIG_HOME`, never in the real ~/.config.
"""
import os
import select
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest

from .process_table import OTHER_CHECKOUT, OTHER_USER, ProcessTable, stubs

START_SH = Path(__file__).resolve().parents[2] / 'start.sh'
ROLE_SH = START_SH.parent / 'scripts' / 'role.sh'
SOURCED = ['role.sh', 'launch_mac.sh', 'launch_linux.sh', 'service.sh']     # what start.sh and stop.sh source
MAC, LINUX = 'Darwin', 'Linux'      # as `uname -s` names them, and KEEP_OS
PYTHON_STUB = Path(__file__).parent / 'data' / 'keep_sh' / 'bin' / 'python3'   # records its launch in KEEP_TEST_CALLS
SATELLITE_PORT = 8901
SATELLITE_CONFIG = (f'role: imessage_satellite\nimessage_satellite:\n  port: {SATELLITE_PORT}\n'
                    '  token: kT7vQ2mX9pL4wZ8nB3cR6yH1sD5fG0jA\n  lan_access: false\n')

# By its full path and with `close_fds=False` Python spawns the script; a bare name or descriptors
# to close make it fork the whole pytest worker first (~6 ms a process).
BASH = shutil.which('bash')

# Far above macOS's pid range, so "a pid that is not alive" is guaranteed.
DEAD_PID = 4999999


def spawn_orphan(cmd: str) -> int:
    """Start cmd detached from pytest (double-fork) and return its pid.

    Orphans get reparented to launchd, which reaps them on death — a plain
    subprocess.Popen child would linger as a zombie after start.sh kills it,
    and the script's `kill -0` would still report it alive.
    """
    out = subprocess.run([BASH, '-c', f'{cmd} >/dev/null 2>&1 & echo $!'],
                         capture_output=True, text=True, check=True, close_fds=False)
    return int(out.stdout.strip())


def wait_dead(pid: int, timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.002)
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
    printed there is the result's stdout. The terminal stays open on this side until the script
    has ended and everything it printed is read: a terminal whose last writer is gone drops what
    was not read yet, and a busy machine got here only after the script had come and gone."""
    master, slave = os.openpty()
    try:
        process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=slave, stderr=subprocess.PIPE,
                                   close_fds=False)
        shown = b''
        ended = False
        deadline = time.time() + 30
        while time.time() < deadline:
            ready, _, _ = select.select([master], [], [], 0 if ended else 0.005)
            if ready:
                shown += os.read(master, 65536)
            elif ended:
                break
            else:
                ended = process.poll() is not None      # one more look then: its last lines are in the terminal
    finally:
        os.close(slave)
        os.close(master)
    process.wait(timeout=30)
    return subprocess.CompletedProcess(command, process.returncode, shown.decode(errors='replace'),
                                       process.stderr.read().decode(errors='replace'))


class StartShHarness:
    """Stubs are shell functions sourced through $BASH_ENV rather than executables on
    PATH: macOS scans every freshly written executable on its first exec (~150 ms
    each), which made every test cost ~1 s. Functions shadow the real commands and
    are never exec'd."""

    def __init__(self, tmp_path: Path, system: str = MAC):
        self.dir = tmp_path
        self.system = system
        (tmp_path / 'log').mkdir(parents=True)
        self.script = tmp_path / 'start.sh'
        self.script.write_text(START_SH.read_text())
        (tmp_path / 'scripts').mkdir()
        for name in SOURCED:
            (tmp_path / 'scripts' / name).write_text((ROLE_SH.parent / name).read_text())
        self.config_home = tmp_path / 'xdg'
        self.unit = self.config_home / 'systemd' / 'user' / 'keep.service'
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
            'seq() { local i; for ((i = $1; i <= $2; i++)); do echo "$i"; done; }\n'   # a process per wait loop otherwise
            + stubs(tmp_path) +
            f'open() {{ echo "$@" >> "{self.open_calls}"; '
            f'[ -f "{self.tty_report}" ] && cp "{self.tty_report}" "{tmp_path}/log/cli_start.tty"; :; }}\n'
            f'osascript() {{ echo "$@" >> "{self.osascript_calls}"; cat >/dev/null; }}\n'
            'tail() { :; }\n')

    def config(self, text: str):
        """The install's config.yaml; none means a Keep server."""
        (self.dir / 'config.yaml').write_text(text)

    def comes_up_on(self, port: int):
        """The launched process listens on `port` from the moment the launch window opens — on
        Linux, from the unit's restart."""
        if self.system == LINUX:
            self.table.on_restart(f'echo "{port} 4242 {os.getuid()}" >> "{self.dir}/listeners"')
            return
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

    def run(self, *args, terminal: bool = False, **more_env) -> subprocess.CompletedProcess:
        env = dict(os.environ, BASH_ENV=str(self.env_file), KEEP_TEST_CALLS=str(self.tui_calls), KEEP_OS=self.system,
                   HOME=str(self.dir / 'home'), XDG_CONFIG_HOME=str(self.config_home))
        env.update(more_env)
        env = {key: value for key, value in env.items() if value is not None}
        command = [BASH, str(self.script), *map(str, args)]
        if terminal:
            return run_on_a_terminal(command, env)
        # capture_output also makes stdout a pipe, so [ -t 1 ] is false and the
        # interactive tail is skipped, matching a detached auto-restart.
        return subprocess.run(command, env=env, capture_output=True, text=True, close_fds=False, timeout=30)

    @property
    def restart_log(self) -> str:
        path = self.dir / 'log' / 'restart.log'
        return path.read_text() if path.exists() else ''

    @property
    def launches(self) -> int:
        """How often a process was launched: a Terminal window opened, or the unit restarted."""
        if self.system == LINUX:
            return sum(' restart ' in f' {call} ' for call in self.table.systemctl_calls)
        return len(self.open_calls.read_text().splitlines()) if self.open_calls.exists() else 0

    @property
    def launched(self) -> bool:
        return self.launches > 0


@pytest.fixture
def harness(tmp_path):
    """The Mac's branch."""
    return StartShHarness(tmp_path)


@pytest.fixture
def linux(tmp_path):
    return StartShHarness(tmp_path, LINUX)


@pytest.fixture(params=[MAC, LINUX])
def either(request, tmp_path):
    """What the two systems' branches share is asked of both."""
    return StartShHarness(tmp_path, request.param)


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
        subprocess.run(['bash', str(script)], env=dict(os.environ, KEEP_OS=MAC), capture_output=True, text=True,
                       timeout=30, check=True)
        deadline = time.time() + 5.0
        while not report.exists() and time.time() < deadline:
            time.sleep(0.02)
        assert report.exists(), "the fake server never ran"
        sid, pid, tty_seen_at_start = report.read_text().split()
        assert sid == pid, "the server is not a session leader of its own"
        assert sid != str(os.getsid(0))
        assert tty_seen_at_start == 'True', "the tty was reported after the server started"
        assert (tmp_path / 'log' / 'cli_start.tty').read_text().strip()

    def test_on_linux_it_starts_nothing_and_fails(self, tmp_path):
        (tmp_path / 'log').mkdir()
        (tmp_path / 'floxy_ai_server.py').write_text('import pathlib\npathlib.Path("log/ran").write_text("")\n')
        script = tmp_path / 'scripts' / 'cli_start.sh'
        script.parent.mkdir()
        script.write_text(CLI_START_SH.read_text())
        result = subprocess.run([BASH, str(script)], env=dict(os.environ, KEEP_OS=LINUX), capture_output=True, text=True,
                                close_fds=False, timeout=30)
        assert result.returncode != 0 and 'macOS' in result.stderr
        assert not (tmp_path / 'log' / 'cli_start.tty').exists() and not (tmp_path / 'log' / 'ran').exists()


class TestTrialFallback:
    """start.sh <pid> <branch> guards a PR trial (PullRequestTrial): the branch is checked out
    again and the server relaunched when the trial's server never listens, and left alone when it
    does. `git` is a recording stub; the launch — the window's `open`, the unit's restart — marks
    the server as up when the test says the PR boots. The watch is the same on both systems."""

    def _arm(self, harness, boots: bool):
        # Each second of the wait is a lookup of the listeners — on Linux three processes; one is
        # enough to see the loop give up (the default 120 would run 16 before the pgrep check ends it).
        with harness.env_file.open('a') as env:
            env.write('TRIAL_TIMEOUT=1\n'
                      f'git() {{ echo "$@" >> "{harness.dir}/git_calls"; }}\n')
        if boots:
            harness.comes_up_on(8888)

    def _git_calls(self, harness) -> list[str]:
        path = harness.dir / 'git_calls'
        return path.read_text().splitlines() if path.exists() else []

    def test_a_pr_whose_server_never_comes_up_is_rolled_back_and_the_branch_relaunched(self, either):
        harness = either
        self._arm(harness, boots=False)
        result = harness.run(DEAD_PID, 'release')
        assert result.returncode == 0, result.stderr
        assert self._git_calls(harness) == ['checkout release']
        assert harness.launches == 2, "the previous branch is launched again"
        note = (harness.dir / 'log' / 'trial_rollback.txt').read_text()
        assert 'back on release' in note
        assert 'going back to release' in harness.restart_log

    def test_a_pr_whose_server_listens_is_left_running(self, either):
        harness = either
        self._arm(harness, boots=True)
        result = harness.run(DEAD_PID, 'release')
        assert result.returncode == 0, result.stderr
        assert self._git_calls(harness) == []
        assert harness.launches == 1
        assert not (harness.dir / 'log' / 'trial_rollback.txt').exists()
        assert 'the server is up' in harness.restart_log

    def test_a_failed_checkout_leaves_the_server_down_and_says_so(self, either):
        harness = either
        self._arm(harness, boots=False)
        with harness.env_file.open('a') as env:
            env.write('git() { echo "error: pathspec did not match" >&2; return 1; }\n')
        result = harness.run(DEAD_PID, 'release')
        assert result.returncode == 1
        assert harness.launches == 1
        assert 'git checkout release failed' in harness.restart_log

    def test_an_ordinary_restart_waits_for_nothing(self, either):
        harness = either
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
            # (11 rounds: the first ten never ask whether the process is still there)
            env.write('seq() { if [ "$2" = 120 ]; then command seq 1 11; else command seq "$@"; fi; }\n')
        result = harness.run(terminal=True)
        assert result.returncode == 1
        assert 'process is gone' not in harness.restart_log
        assert harness.console_launches == []

    @pytest.mark.parametrize('system', [MAC, LINUX])
    def test_the_server_still_gets_its_console(self, tmp_path, system):
        harness = StartShHarness(tmp_path, system)
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
    def test_nothing_is_stopped_and_nothing_is_started(self, either, bystander, config):
        harness = either
        running = bystander()
        harness.config(config)
        harness.table.process(running)
        harness.table.listener(8888, running)
        result = harness.run(running)
        assert result.returncode == 1
        assert alive(running), "the running process was stopped on a config the new one cannot start on"
        assert not harness.launched
        assert 'config.yaml' in harness.restart_log

    def test_a_satellite_role_on_linux_is_refused_the_same_way(self, linux, bystander):
        """floxy_ai_server._launch refuses the role there: a satellite is a Mac."""
        running = bystander()
        linux.config(SATELLITE_CONFIG)
        linux.table.listener(SATELLITE_PORT, running)
        linux.table.process(running)
        result = linux.run(running)
        assert result.returncode == 1
        assert alive(running) and not linux.launched
        assert linux.table.systemctl_calls == [] and not linux.unit.exists()
        assert 'macOS' in linux.restart_log


def unit_settings(unit: Path) -> dict:
    return dict(line.split('=', 1) for line in unit.read_text().splitlines() if '=' in line and not line.startswith('#'))


class TestTheSystemdUnit:
    """On Linux a start is the user unit keep.service: written by scripts/service.sh, enabled and
    restarted — systemd is what stops the old server and starts the new one."""

    def test_a_first_start_writes_the_unit_enables_it_and_restarts_it(self, linux):
        result = linux.run()
        assert result.returncode == 0, linux.restart_log
        verbs = [call.split()[1] for call in linux.table.systemctl_calls]
        assert verbs == ['daemon-reload', 'enable', 'reset-failed', 'restart']
        assert all(call.startswith('--user ') for call in linux.table.systemctl_calls), "the system's manager was asked"

    def test_the_unit_runs_this_checkouts_server_a_contract_with_systemd(self, linux):
        linux.run()
        checkout = str(linux.dir.resolve())
        unit = unit_settings(linux.unit)
        assert unit['WorkingDirectory'] == checkout
        assert unit['ExecStart'] == f'{checkout}/.venv/bin/python floxy_ai_server.py'
        assert unit['StandardOutput'] == f'append:{checkout}/log/raw-floxy.log', "where start.sh's rollback note reads the output"
        assert unit['Restart'] == 'on-failure', "an explicit stop must stay stopped"
        assert unit['WantedBy'] == 'default.target'

    def test_the_unit_waits_for_no_target_of_the_system_manager(self, linux):
        """network-online.target is the system manager's: a user unit ordered after it is ordered
        after nothing (not-found on Ubuntu 24.04 and Debian 12), so the unit names none."""
        linux.run()
        ordering = [line for line in linux.unit.read_text().splitlines() if line.startswith(('After=', 'Wants=', 'Requires='))]
        assert ordering == []

    def test_the_unit_finds_what_the_user_installed_under_their_home_before_the_systems(self, linux):
        """A unit gets the user manager's PATH, which has no ~/.local/bin: the nightly run's
        `claude` lives there. The line replaces that PATH, so the system's folders — and Ubuntu's
        /snap/bin, where its Chromium is — must be on it too. %h is systemd's word for the home."""
        linux.run()
        (line,) = [line for line in linux.unit.read_text().splitlines() if 'PATH=' in line]
        assert line.startswith('Environment="PATH=') and line.endswith('"'), "unquoted, a home with a space splits it"
        path = line[len('Environment="PATH='):-1].split(':')
        assert path[0] == '%h/.local/bin'
        assert {'/usr/local/bin', '/usr/bin', '/bin', '/snap/bin'} <= set(path)
        assert all(folder.startswith(('/', '%h/')) for folder in path)

    def test_without_xdg_config_home_the_unit_lands_under_the_homes_config(self, linux):
        linux.run(XDG_CONFIG_HOME=None)
        assert (linux.dir / 'home' / '.config' / 'systemd' / 'user' / 'keep.service').exists()
        assert not linux.unit.exists()

    def test_a_second_start_leaves_the_unit_alone_and_only_restarts(self, linux):
        linux.run()
        written = linux.unit.stat().st_mtime_ns
        linux.run()
        verbs = [call.split()[1] for call in linux.table.systemctl_calls]
        assert verbs.count('daemon-reload') == 1 and verbs.count('restart') == 2
        assert linux.unit.stat().st_mtime_ns == written

    def test_a_unit_of_this_checkout_with_another_text_is_rewritten_and_reloaded(self, linux):
        linux.unit.parent.mkdir(parents=True)
        linux.unit.write_text(f'[Service]\nWorkingDirectory={linux.dir}\nExecStart=/usr/bin/true\n')
        result = linux.run()
        assert result.returncode == 0, linux.restart_log
        assert unit_settings(linux.unit)['ExecStart'].endswith('floxy_ai_server.py')
        assert '--user daemon-reload' in linux.table.systemctl_calls

    def test_a_unit_of_another_checkout_is_refused_with_both_folders_named(self, linux, tmp_path_factory):
        other = tmp_path_factory.mktemp('KeepElsewhere')
        linux.unit.parent.mkdir(parents=True)
        theirs = f'[Service]\nWorkingDirectory={other}\nExecStart={other}/.venv/bin/python floxy_ai_server.py\n'
        linux.unit.write_text(theirs)
        result = linux.run()
        assert result.returncode == 1
        assert linux.unit.read_text() == theirs, "the other checkout's unit was replaced"
        assert linux.table.systemctl_calls == [], "one Keep per user: the other one's unit was touched"
        assert str(other) in linux.restart_log and str(linux.dir.resolve()) in linux.restart_log

    def test_a_unit_whose_checkout_is_gone_is_taken_over(self, linux):
        linux.unit.parent.mkdir(parents=True)
        linux.unit.write_text(f'[Service]\nWorkingDirectory={linux.dir}/moved-away\n')
        result = linux.run()
        assert result.returncode == 0, linux.restart_log
        assert unit_settings(linux.unit)['WorkingDirectory'] == str(linux.dir.resolve())
        assert linux.launched

    def test_a_file_of_that_name_keep_did_not_write_is_not_replaced(self, linux):
        linux.unit.parent.mkdir(parents=True)
        linux.unit.write_text('[Service]\nExecStart=/usr/bin/true\n')
        result = linux.run()
        assert result.returncode == 1
        assert linux.unit.read_text() == '[Service]\nExecStart=/usr/bin/true\n' and not linux.launched

    @pytest.mark.parametrize('name', ['My Keep', 'Keep 100%', "Marcus' Keep"])
    def test_a_folder_systemd_would_read_otherwise_is_refused_and_no_unit_written(self, tmp_path, name):
        """A space splits ExecStart in two and % is expanded: the unit would run something else.
        (No case with a $: bash expands it in $BASH_ENV, and the run would be one without its stubs.)"""
        linux = StartShHarness(tmp_path / name, LINUX)
        result = linux.run()
        assert result.returncode == 1
        assert not linux.unit.exists() and linux.table.systemctl_calls == []
        assert name in linux.restart_log

    def test_a_folder_with_letters_outside_ascii_is_fine(self, tmp_path):
        linux = StartShHarness(tmp_path / 'Kęp-ö', LINUX)
        result = linux.run(LC_ALL='C')
        assert result.returncode == 0, linux.restart_log
        assert unit_settings(linux.unit)['WorkingDirectory'] == str(linux.dir.resolve())

    @pytest.mark.parametrize('verb', ['enable', 'daemon-reload', 'restart'])
    def test_a_systemd_that_does_not_answer_fails_the_start_and_says_so(self, linux, verb):
        """No user session — a container, a shell reached through su."""
        linux.table.systemctl_fails(verb)
        result = linux.run()
        assert result.returncode == 1
        assert 'ERROR' in linux.restart_log and verb in linux.restart_log
        assert linux.launches == (1 if verb == 'restart' else 0)

    def test_the_handed_over_pid_is_left_to_systemd(self, linux, bystander):
        """Updater.relaunch hands the server's pid over; on Linux the unit's restart is what ends it."""
        old = bystander()
        linux.table.process(old)
        linux.table.in_unit(old)
        result = linux.run(old)
        assert result.returncode == 0, linux.restart_log
        assert alive(old), "start.sh killed a process on Linux"
        assert linux.launched


class TestAListenerTheUnitDoesNotRun:
    """A server started by hand holds the port: the unit's server would die on it, five times."""

    def test_the_start_is_refused_and_nothing_is_touched(self, linux, bystander):
        by_hand = bystander()
        linux.table.process(by_hand)
        linux.table.listener(8888, by_hand)
        result = linux.run()
        assert result.returncode == 1
        assert alive(by_hand)
        assert linux.table.systemctl_calls == [] and not linux.unit.exists()
        assert str(by_hand) in linux.restart_log

    def test_one_that_asked_for_the_restart_itself_is_stopped_and_the_unit_takes_over(self, linux, bystander):
        """restart_server from a server started by hand: it hands its own pid over, and a refusal
        would leave the command answered with nothing."""
        by_hand = bystander()
        linux.table.process(by_hand)
        linux.table.listener(8888, by_hand)
        with linux.env_file.open('a') as env:     # the port is free once its holder is signalled
            env.write(f'kill() {{ builtin kill "$@"; [ "$1" = -0 ] || : > "{linux.dir}/listeners"; }}\n')
        result = linux.run(by_hand)
        assert result.returncode == 0, linux.restart_log
        assert wait_dead(by_hand), "the server that asked for the restart still runs"
        assert linux.launches == 1

    def test_a_handed_over_pid_that_holds_no_port_is_not_stopped(self, linux, bystander):
        """Only the listener the pid names is the server that asked; anything else is not this script's."""
        other = bystander()
        linux.table.process(other)
        result = linux.run(other)
        assert result.returncode == 0, linux.restart_log
        assert alive(other) and linux.launched

    def test_the_units_own_server_on_the_port_is_what_a_restart_is_for(self, linux, bystander):
        server = bystander()
        linux.table.process(server)
        linux.table.listener(8888, server)
        linux.table.in_unit(server)
        result = linux.run()
        assert result.returncode == 0, linux.restart_log
        assert linux.launched

    def test_the_other_users_server_on_the_same_port_is_not_this_installs_to_judge(self, linux, bystander):
        """ss names the pid here, as it does for root: whose it is must be checked."""
        theirs = bystander()
        linux.table.process(theirs, user=OTHER_USER, checkout=OTHER_CHECKOUT)
        linux.table.listener(8888, theirs, user=OTHER_USER)
        result = linux.run()
        assert result.returncode == 0, linux.restart_log
        assert linux.launched and alive(theirs)

    def test_the_outpost_port_is_not_a_linux_servers(self, linux, bystander):
        """A Linux server binds 8888 alone (VoiceInterface starts no Outpost TCP server there)."""
        other = bystander()
        linux.table.process(other)
        linux.table.listener(43523, other)
        result = linux.run()
        assert result.returncode == 0, linux.restart_log
        assert linux.launched


class TestStartedFromInsideTheUnit:
    """The server restarts itself by spawning start.sh, which is then in the unit's control group:
    the restart would end the script with the server, and the PR-trial watch with it."""

    def _trial(self, linux):
        with linux.env_file.open('a') as env:
            env.write(f'TRIAL_TIMEOUT=1\ngit() {{ echo "$@" >> "{linux.dir}/git_calls"; }}\n')

    def test_it_runs_again_in_a_scope_with_its_arguments_and_restarts_once(self, linux):
        self._trial(linux)
        linux.table.in_unit()
        linux.comes_up_on(8888)
        result = linux.run(DEAD_PID, 'release')
        assert result.returncode == 0, linux.restart_log
        scoped = [call for call in linux.table.systemd_run_calls if 'start.sh' in call]
        assert len(scoped) == 1, linux.table.systemd_run_calls
        assert scoped[0].startswith('--user --scope ') and scoped[0].endswith(f'start.sh {DEAD_PID} release')
        assert linux.launches == 1, "the copy left in the unit restarted it as well"

    def test_the_trial_watch_runs_in_the_scope(self, linux):
        self._trial(linux)
        linux.table.in_unit()
        linux.comes_up_on(8888)
        result = linux.run(DEAD_PID, 'release')
        assert result.returncode == 0, linux.restart_log
        assert 'the server is up' in linux.restart_log

    def test_a_start_outside_the_unit_asks_for_no_scope(self, linux):
        linux.run()
        assert linux.table.systemd_run_calls == []

    def test_a_scope_that_does_not_leave_the_unit_is_not_tried_twice(self, linux):
        (linux.dir / 'scope_stays_in_unit').write_text('')
        linux.table.in_unit()
        result = linux.run(DEAD_PID)
        assert result.returncode == 0, linux.restart_log
        assert len([c for c in linux.table.systemd_run_calls if 'start.sh' in c]) == 1, "start.sh ran itself in a loop"
        assert linux.launches == 1

    def test_without_a_scope_to_be_had_the_restart_still_happens(self, linux):
        (linux.dir / 'no_scope').write_text('')
        linux.table.in_unit()
        result = linux.run(DEAD_PID)
        assert result.returncode == 0, linux.restart_log
        assert linux.launches == 1
        assert 'WARNING' in linux.restart_log


class TestWhichUnitIsThisCheckouts:
    """The unit is this checkout's when its WorkingDirectory is this folder — by device and inode,
    as role.sh tells every folder, never by the name it was written under."""

    def test_a_unit_that_names_this_checkout_through_a_symlink_is_its_own_and_is_rewritten(self, linux, tmp_path_factory):
        """Read by name, ~/Keep -> /srv/keep would be "another Keep of this user" and every start refused."""
        link = tmp_path_factory.mktemp('links') / 'Keep'
        link.symlink_to(linux.dir)
        linux.unit.parent.mkdir(parents=True)
        linux.unit.write_text(f'[Service]\nWorkingDirectory={link}\n')
        result = linux.run()
        assert result.returncode == 0, linux.restart_log
        assert unit_settings(linux.unit)['WorkingDirectory'] == str(linux.dir.resolve())
        assert linux.launched

    def test_a_unit_folder_that_cannot_be_written_fails_the_start_before_systemd_is_asked(self, linux):
        linux.config_home.mkdir()
        (linux.config_home / 'systemd').write_text('')      # a file where the folder would be made
        result = linux.run()
        assert result.returncode == 1
        assert linux.table.systemctl_calls == [] and 'ERROR' in linux.restart_log


# Folder names systemd's unit syntax would misread, and the ones the harness cannot stage: its
# stubs are sourced through $BASH_ENV, which bash expands. install.sh refuses the same ones.
UNIT_UNSAFE_NAMES = ['Keep$HOME', 'Keep\\backup', 'Keep"quoted"', 'Keep\ttab', 'Keep\nline', 'My Keep', 'Keep 100%',
                     "Marcus' Keep"]
UNIT_SAFE_NAMES = ['Keep', 'Kęp-ö', 'keep_1.2+test@home']
ENSURE_UNIT = """
cd "$0" || exit 2
record=$1
rlog() { :; }
systemctl() { echo "$*" >> "$record/systemctl_calls"; }
. "$2"
if ensure_unit; then echo written; else echo "refused: $unit_problem"; fi
"""


def ensure_unit(tmp_path: Path, name: str) -> tuple[subprocess.CompletedProcess, Path, Path]:
    """scripts/service.sh's ensure_unit run in a checkout of that name: (the run, the unit, the
    record of what systemd was asked)."""
    checkout = tmp_path / 'checkouts' / name
    checkout.mkdir(parents=True)
    env = dict(os.environ, XDG_CONFIG_HOME=str(tmp_path / 'xdg'), HOME=str(tmp_path / 'home'), KEEP_OS=LINUX)
    out = subprocess.run([BASH, '-c', ENSURE_UNIT, str(checkout), str(tmp_path), str(ROLE_SH.parent / 'service.sh')],
                         env=env, capture_output=True, text=True, close_fds=False, timeout=10)
    return out, tmp_path / 'xdg' / 'systemd' / 'user' / 'keep.service', tmp_path / 'systemctl_calls'


class TestTheFoldersAUnitCanCarry:
    """ensure_unit asked directly: `$`, a backslash, a double quote and a control character are
    each read by systemd as something else than a letter of the path."""

    @pytest.mark.parametrize('name', UNIT_UNSAFE_NAMES, ids=[repr(name) for name in UNIT_UNSAFE_NAMES])
    def test_a_folder_systemd_would_misread_gets_no_unit_and_systemd_is_not_asked(self, tmp_path, name):
        out, unit, asked = ensure_unit(tmp_path, name)
        assert out.stdout.startswith('refused: '), out.stdout + out.stderr
        assert not unit.exists() and not asked.exists()

    @pytest.mark.parametrize('name', UNIT_SAFE_NAMES)
    def test_any_other_folder_is_written_into_the_unit_as_it_is(self, tmp_path, name):
        out, unit, asked = ensure_unit(tmp_path, name)
        assert out.stdout.strip() == 'written', out.stdout + out.stderr
        checkout = str((tmp_path / 'checkouts' / name).resolve())
        assert unit_settings(unit)['ExecStart'] == f'{checkout}/.venv/bin/python floxy_ai_server.py'
        assert asked.read_text().splitlines() == ['--user daemon-reload', '--user enable --quiet keep.service']


class TestWhatALinuxStartNeeds:
    @pytest.mark.parametrize('tool', ['ss', 'systemctl'])
    def test_without_the_tool_nothing_is_stopped_and_nothing_started(self, linux, bystander, tool):
        """Without ss no listener is ever seen: a held port would pass for free."""
        by_hand = bystander()
        linux.table.process(by_hand)
        linux.table.listener(8888, by_hand)
        with linux.env_file.open('a') as env:
            env.write(f'command() {{ if [ "$1" = -v ] && [ "$2" = {tool} ]; then return 1; fi; builtin command "$@"; }}\n')
        result = linux.run(by_hand)
        assert result.returncode == 1
        assert alive(by_hand) and not linux.unit.exists() and linux.table.systemctl_calls == []
        assert tool in linux.restart_log

    def test_a_server_started_by_hand_that_ignores_its_stop_is_force_killed_before_the_unit_starts(self, linux):
        """restart_server from a wedged server run in a terminal: left alive it holds the port,
        and the unit's server dies on it five times."""
        trap_ready = linux.dir / 'trap_ready'
        by_hand = spawn_orphan(f"""bash -c 'trap "" TERM; touch "{trap_ready}"; while :; do /bin/sleep 1; done'""")
        deadline = time.time() + 5.0
        while not trap_ready.exists() and time.time() < deadline:
            time.sleep(0.005)
        assert trap_ready.exists(), "the process never installed its TERM trap"
        linux.table.process(by_hand)
        linux.table.listener(8888, by_hand)
        with linux.env_file.open('a') as env:     # the port is free only once its holder is really gone
            env.write(f'kill() {{ builtin kill "$@"; [ "$1" != -9 ] || : > "{linux.dir}/listeners"; }}\n')
        result = linux.run(by_hand)
        assert wait_dead(by_hand), "the server that ignored SIGTERM still runs"
        assert result.returncode == 0, linux.restart_log
        assert linux.launches == 1

    def test_the_terminal_is_shown_the_log_from_where_it_ended_before_the_start(self, either):
        """A start appends to a log that holds every earlier run: the tail begins at the first new byte."""
        harness = either
        (harness.dir / 'log' / 'verbose-floxy.out').write_text('x' * 470)
        harness.with_a_console_to_launch()
        harness.comes_up_on(5151)
        with harness.env_file.open('a') as env:
            env.write(f'tail() {{ echo "$*" >> "{harness.dir}/tail_calls"; }}\n')
        harness.run(terminal=True)
        (shown,) = (harness.dir / 'tail_calls').read_text().splitlines()
        assert '+471 ' in shown and shown.endswith('log/verbose-floxy.out')
