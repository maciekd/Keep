"""trial_watchdog.sh: the way back when a tried PR does not start.

Runs with `sleep` as a no-op and `git` recorded (shell functions sourced through $BASH_ENV,
as in test_start_sh.py), so the timeout loop is instant and no repository is needed.
"""
import os
import subprocess
from pathlib import Path

import pytest

WATCHDOG_SH = Path(__file__).resolve().parents[2] / 'trial_watchdog.sh'


class Harness:
    def __init__(self, tmp_path: Path):
        self.dir = tmp_path
        (tmp_path / 'log').mkdir()
        self.trial = tmp_path / 'cache' / 'trial'
        self.trial.mkdir(parents=True)
        (self.trial / 'state.json').write_text('{}')
        (tmp_path / 'log' / 'raw-floxy.log').write_text('ModuleNotFoundError: lights\n')
        (tmp_path / 'start.sh').write_text('echo started >> start_calls\n')
        self.git_calls = tmp_path / 'git_calls'
        self.env_file = tmp_path / 'stubs.bash'
        self.git_status = 0
        self._write_stubs()

    def _write_stubs(self):
        self.env_file.write_text('sleep() { :; }\n'
                                 f'git() {{ echo "$@" >> "{self.git_calls}"; return {self.git_status}; }}\n')

    def fail_git(self):
        self.git_status = 1
        self._write_stubs()

    def run(self, *args) -> subprocess.CompletedProcess:
        return subprocess.run(['bash', str(WATCHDOG_SH), *args], cwd=self.dir, capture_output=True, text=True,
                              env=dict(os.environ, BASH_ENV=str(self.env_file)), timeout=30)

    @property
    def restarted(self) -> bool:
        return (self.dir / 'start_calls').exists()


@pytest.fixture
def harness(tmp_path):
    return Harness(tmp_path)


def test_a_server_that_never_comes_up_is_rolled_back_and_restarted(harness):
    assert harness.run('release', '3').returncode == 0
    assert harness.git_calls.read_text().split() == ['checkout', 'release']
    assert harness.restarted
    assert 'ModuleNotFoundError: lights' in (harness.trial / 'rolled_back').read_text()


def test_a_server_that_came_up_is_left_alone(harness):
    (harness.trial / 'started').write_text('1')
    assert harness.run('release', '3').returncode == 0
    assert not harness.git_calls.exists() and not harness.restarted


def test_a_trial_ended_meanwhile_is_left_alone(harness):
    (harness.trial / 'state.json').unlink()
    harness.run('release', '3')
    assert not harness.git_calls.exists() and not harness.restarted


def test_a_failed_checkout_does_not_restart_the_pr_code(harness):
    harness.fail_git()
    assert harness.run('release', '3').returncode != 0
    assert not harness.restarted
