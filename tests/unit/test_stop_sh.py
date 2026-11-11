"""stop.sh: SIGTERM for what this install runs — this user's floxy_ai_server processes of this
checkout — and for nothing else on the Mac: not the Keep of the Mac's other user (a satellite and
its server usually share a Mac), not another checkout's.

Run like start.sh's tests: a copy in a tmp tree, `lsof` and `pgrep` answering from a staged
process table (process_table.py), real processes as the ones to stop or to leave alone."""
import os
import subprocess
from pathlib import Path

import pytest

from .process_table import OTHER_CHECKOUT, OTHER_USER, ProcessTable, stubs
from .test_start_sh import BASH, alive, bystander, spawn_orphan, wait_dead  # noqa: F401  (bystander is a fixture)

STOP_SH = Path(__file__).resolve().parents[2] / 'stop.sh'
ROLE_SH = STOP_SH.parent / 'scripts' / 'role.sh'


class Harness:
    def __init__(self, tmp_path: Path):
        self.dir = tmp_path
        (tmp_path / 'stop.sh').write_text(STOP_SH.read_text())
        (tmp_path / 'scripts').mkdir()
        (tmp_path / 'scripts' / 'role.sh').write_text(ROLE_SH.read_text())
        self.table = ProcessTable(tmp_path)
        self.env_file = tmp_path / 'stubs.bash'
        self.env_file.write_text('date() { echo "2026-10-13 03:00:00"; }\n' + stubs(tmp_path))

    def run(self) -> subprocess.CompletedProcess:
        return subprocess.run([BASH, str(self.dir / 'stop.sh')], env=dict(os.environ, BASH_ENV=str(self.env_file)),
                              capture_output=True, text=True, close_fds=False, timeout=30)


@pytest.fixture
def harness(tmp_path):
    return Harness(tmp_path)


@pytest.mark.unit
class TestStopSh:
    @pytest.mark.parametrize('config', [None, 'role: imessage_satellite\nimessage_satellite:\n  port: 8901\n'],
                             ids=['a server', 'a satellite'])
    def test_this_installs_process_is_stopped(self, harness, config):
        victim = spawn_orphan('/bin/sleep 300')
        if config:
            (harness.dir / 'config.yaml').write_text(config)
        harness.table.process(victim)
        result = harness.run()
        assert result.returncode == 0, result.stderr
        assert wait_dead(victim), f"pid {victim} is still alive"

    def test_the_other_users_keep_is_left_alone(self, harness, bystander):
        theirs = bystander()
        harness.table.process(theirs, user=OTHER_USER, checkout=OTHER_CHECKOUT)
        harness.table.listener(8888, theirs, user=OTHER_USER)
        harness.run()
        assert alive(theirs)

    def test_the_same_users_other_checkout_is_left_alone(self, harness, bystander):
        """A server and a satellite under one user are both `python floxy_ai_server.py`."""
        mine, other = spawn_orphan('/bin/sleep 300'), bystander()
        harness.table.process(mine)
        harness.table.process(other, checkout=OTHER_CHECKOUT)
        harness.run()
        assert wait_dead(mine)
        assert alive(other), "stopping one install stopped the other"

    def test_a_process_whose_folder_cannot_be_read_is_found_by_the_port_it_holds(self, harness):
        """Without it a stop.sh that cannot match the working directory says "nothing to stop"
        and leaves the server up."""
        victim = spawn_orphan('/bin/sleep 300')
        harness.table.listener(8888, victim)        # in no process list: lsof cannot say where it runs
        result = harness.run()
        assert result.returncode == 0, result.stderr
        assert wait_dead(victim), "the holder of this install's port was left running"

    def test_the_same_users_other_checkout_is_left_alone_on_the_same_port(self, harness, bystander):
        other = bystander()
        harness.table.process(other, checkout=OTHER_CHECKOUT)
        harness.table.listener(8888, other)
        harness.run()
        assert alive(other), "a port match stopped another checkout's server"

    def test_a_satellite_stops_nothing_on_the_servers_ports(self, harness, bystander):
        server = bystander()
        (harness.dir / 'config.yaml').write_text('role: imessage_satellite\nimessage_satellite:\n  port: 8901\n')
        harness.table.listener(8888, server)
        harness.run()
        assert alive(server)

    def test_with_nothing_running_it_stops_nothing_and_succeeds(self, harness):
        result = harness.run()
        assert result.returncode == 0
        assert 'nothing to stop' in result.stdout

    @pytest.mark.parametrize('watchdog', [True, False], ids=['the watchdog on', 'the watchdog off'])
    def test_it_says_when_the_watchdog_will_start_the_process_again(self, harness, watchdog):
        victim = spawn_orphan('/bin/sleep 300')
        (harness.dir / 'config.yaml').write_text(f'role: imessage_satellite\nwatchdog: {str(watchdog).lower()}\n')
        harness.table.process(victim)
        said = harness.run().stdout
        assert wait_dead(victim)
        assert ('watchdog' in said) == watchdog
