"""Tests for scripts/local_omlx_start.sh, the script that starts oMLX serving the local model.

The script runs from a copy in a temporary tree (it cds to its parent folder and writes
`log/omlx.log` there). `omlx`, `hf`, `pgrep`, `pkill` and `sleep` are bash functions sourced
through $BASH_ENV: each records its call in one journal, so no process of this machine is looked
up or signalled, nothing is downloaded and no server starts.
"""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.unit.waiting import FAILURE_BOUND

SCRIPT = Path(__file__).resolve().parents[2] / 'scripts' / 'local_omlx_start.sh'
REPO = 'example-org/Tiny-Model-4bit'


class OmlxHarness:
    def __init__(self, tmp_path: Path, installed: bool = True):
        self.root = tmp_path / 'keep'
        (self.root / 'scripts').mkdir(parents=True)
        self.script = self.root / 'scripts' / SCRIPT.name
        shutil.copy(SCRIPT, self.script)
        self.models = tmp_path / 'models'
        self.journal = tmp_path / 'journal'
        self.running = tmp_path / 'running'     # exists while a (staged) omlx-server runs
        stubs = [f'pgrep() {{ [ -f "{self.running}" ]; }}',
                 f'pkill() {{ echo "pkill $*" >> "{self.journal}"; rm -f "{self.running}"; }}',
                 'sleep() { :; }',
                 f'hf() {{ echo "hf $*" >> "{self.journal}"; }}']
        if installed:
            stubs.append(f'omlx() {{ echo "omlx $*" >> "{self.journal}"; echo "serving"; }}')
        self.env_file = tmp_path / 'stubs.bash'
        self.env_file.write_text("\n".join(stubs) + "\n")

    def with_model(self):
        folder = self.models / REPO.split('/')[1]
        folder.mkdir(parents=True)
        (folder / 'config.json').write_text('{}')

    def run(self, *args, **env) -> subprocess.CompletedProcess:
        # A PATH of the system's own folders: an oMLX installed on this machine is not on it.
        full = dict(os.environ, PATH='/usr/bin:/bin', BASH_ENV=str(self.env_file), MODEL_REPO=REPO,
                    MODEL_DIR=str(self.models), **env)
        return subprocess.run(['/bin/bash', str(self.script), *args], env=full, capture_output=True, text=True,
                              timeout=FAILURE_BOUND)

    def calls(self) -> list[str]:
        return self.journal.read_text().splitlines() if self.journal.exists() else []


@pytest.fixture
def omlx(tmp_path):
    return OmlxHarness(tmp_path)


@pytest.mark.unit
class TestLocalOmlxStart:
    def test_without_omlx_nothing_starts_and_the_install_command_is_named(self, tmp_path):
        harness = OmlxHarness(tmp_path, installed=False)
        harness.with_model()
        result = harness.run()
        assert result.returncode == 1 and 'brew install' in result.stdout
        assert harness.calls() == []

    def test_the_server_gets_the_folder_address_and_memory_ceiling_it_was_started_with(self, omlx):
        omlx.with_model()
        result = omlx.run(HOST='0.0.0.0', PORT='8123', MEMORY_GB='30')
        assert result.returncode == 0, result.stderr
        [serve] = omlx.calls()
        words = serve.split()
        assert words[:2] == ['omlx', 'serve']
        options = dict(zip(words[2::2], words[3::2]))
        assert options == {'--model-dir': str(omlx.models), '--host': '0.0.0.0', '--port': '8123',
                           '--memory-guard-gb': '30'}

    def test_the_servers_output_is_kept_in_the_trees_log_folder(self, omlx):
        omlx.with_model()
        result = omlx.run()
        assert 'serving' in result.stdout
        assert (omlx.root / 'log' / 'omlx.log').read_text() == 'serving\n'

    def test_a_missing_model_is_downloaded_into_the_model_folder_before_the_start(self, omlx):
        result = omlx.run()
        assert result.returncode == 0, result.stderr
        download, serve = omlx.calls()
        assert download == f"hf download {REPO} --local-dir {omlx.models / 'Tiny-Model-4bit'}"
        assert serve.startswith('omlx serve')

    def test_a_model_already_there_is_not_downloaded_again(self, omlx):
        omlx.with_model()
        omlx.run()
        assert not any(call.startswith('hf ') for call in omlx.calls())

    def test_a_running_server_is_stopped_before_the_new_one_starts(self, omlx):
        omlx.with_model()
        omlx.running.touch()
        result = omlx.run()
        assert result.returncode == 0, result.stderr
        stop, serve = omlx.calls()
        assert stop.startswith('pkill') and 'omlx-server' in stop and serve.startswith('omlx serve')

    def test_stop_ends_a_running_server_and_starts_none(self, omlx):
        omlx.with_model()
        omlx.running.touch()
        result = omlx.run('--stop')
        assert result.returncode == 0
        assert len(omlx.calls()) == 1 and omlx.calls()[0].startswith('pkill') and not omlx.running.exists()

    def test_stop_with_nothing_running_signals_nothing_even_where_omlx_is_not_installed(self, tmp_path):
        harness = OmlxHarness(tmp_path, installed=False)
        result = harness.run('--stop')
        assert result.returncode == 0 and harness.calls() == []

    def test_a_server_that_ignores_the_stop_is_killed(self, omlx):
        omlx.with_model()
        omlx.running.touch()
        stubborn = omlx.env_file.read_text().replace(f'rm -f "{omlx.running}"; }}', '}')
        # the forced kill is the one that ends it
        stubborn += (f'pkill() {{ echo "pkill $*" >> "{omlx.journal}"; '
                     f'[ "$1" = "-9" ] && rm -f "{omlx.running}"; return 0; }}\n')
        omlx.env_file.write_text(stubborn)
        result = omlx.run('--stop')
        assert result.returncode == 0
        first, forced = omlx.calls()
        assert '-9' not in first.split() and '-9' in forced.split()
