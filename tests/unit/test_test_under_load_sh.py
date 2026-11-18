"""Tests for scripts/test_under_load.sh, which the nightly test health runs the unit suite with
on the server's own machine: busy loops beside the suite, both niced so the server keeps the CPU.

The script runs from a copy of the tree's shape in a temporary folder whose venv holds a fake
`python` (checked in under `tests/unit/data/test_under_load_sh/` — macOS assesses a freshly
written executable the first time it runs). The fake reports its arguments and, asked to, its niceness and the
script's other children (pgrep leaves out its own ancestors), which is how a test knows the loops and that they are gone afterwards.
One loop per run of the script, not one per core: the test is about the script, not the load.
"""
import os
import re
import subprocess
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / 'scripts' / 'test_under_load.sh'
FAKE_PYTHON = Path(__file__).parent / 'data' / 'test_under_load_sh' / 'python'


@pytest.fixture
def run(tmp_path):
    (tmp_path / 'scripts').mkdir()
    (tmp_path / 'scripts' / 'test_under_load.sh').symlink_to(SCRIPT)
    venv = tmp_path / '.venv' / 'bin'
    venv.mkdir(parents=True)
    (venv / 'python').symlink_to(FAKE_PYTHON)
    (venv / 'activate').write_text(f'PATH="{venv}:$PATH"\n')

    def run(*args, **env):
        return subprocess.run([str(tmp_path / 'scripts' / 'test_under_load.sh'), *args], cwd='/',
                              env={**os.environ, 'LOAD_PROCESSES': '1', **env}, capture_output=True, text=True)
    return run


def field(output: Path, name: str) -> str:
    return re.search(rf'^{name}: (.*)$', output.read_text(), re.M).group(1).strip()


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@pytest.mark.unit
class TestTestUnderLoad:

    def test_each_run_leaves_its_output_and_the_given_tests_replace_the_whole_suite(self, run, tmp_path):
        result = run('2', str(tmp_path / 'out'), 'tests/unit/test_one.py', '-k', 'slow')
        assert result.returncode == 0, result.stderr
        for name in ('run_1.txt', 'run_2.txt'):
            argv = field(tmp_path / 'out' / name, 'argv').split()
            assert argv[:3] == ['-m', 'pytest', 'tests/unit/test_one.py'] and argv[3:5] == ['-k', 'slow']
            assert 'tests/unit' not in argv
        assert result.stdout.count('3 passed') == 2

    def test_without_arguments_the_whole_unit_suite_runs_niced_beside_a_loop_that_ends_with_the_script(self, run,
                                                                                                         tmp_path):
        result = run('1', str(tmp_path / 'out'), REPORT_PROCESSES='1')
        assert result.returncode == 0, result.stderr
        output = tmp_path / 'out' / 'run_1.txt'
        assert field(output, 'argv').split()[:3] == ['-m', 'pytest', 'tests/unit']
        assert int(field(output, 'niceness')) > os.nice(0)
        loops = [int(pid) for pid in field(output, 'siblings').split()]
        assert loops, "the script ran the suite with no busy loop beside it"
        deadline = time.monotonic() + 2
        while any(map(alive, loops)) and time.monotonic() < deadline:
            time.sleep(0.002)
        assert not any(map(alive, loops)), "a busy loop outlived the script"

    def test_without_an_override_the_load_is_several_loops_for_every_core(self, run, tmp_path):
        """One loop per core passed a test that failed on a CI runner; the load follows the cores,
        several times over. The cores are staged, so the test costs the same on any machine."""
        def loops_on(cores: int) -> int:
            stubs = tmp_path / f'cores_{cores}.bash'
            stubs.write_text(f'getconf() {{ echo {cores}; }}\n')
            out = tmp_path / f'out_{cores}'
            result = run('1', str(out), LOAD_PROCESSES='', BASH_ENV=str(stubs), REPORT_PROCESSES='1')
            assert result.returncode == 0, result.stderr
            return len(field(out / 'run_1.txt', 'siblings').split())

        one, two = loops_on(1), loops_on(2)
        per_core = two - one
        assert per_core > 1, "a single loop per core is the load that hid the failing test"
        # procps' pgrep lists the suite's own process among the siblings, the Mac's leaves it out
        assert one - per_core in (0, 1)

    def test_a_missing_run_count_is_refused_before_anything_starts(self, run, tmp_path):
        result = run(str(tmp_path / 'out'))
        assert result.returncode == 2
        assert not (tmp_path / 'out').exists()
