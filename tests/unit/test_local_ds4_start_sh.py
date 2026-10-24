"""Tests for local_ds4_start.sh, the script that starts the local ds4-server for Keep.

The script runs against a fake ds4 checkout in a temporary folder: a `ds4-server` that records
its arguments and stays up (or dies at once), a model file and, per test, the vision encoder.
`lsof`, `curl` and `sleep` are bash functions sourced through $BASH_ENV, so no port of this
machine is probed and the ten-minute wait loop ends in milliseconds; the KV cache folder is pointed into
the temporary folder, so nothing under the real home is created.
"""
import os
import signal
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / 'local_ds4_start.sh'
MODEL = 'gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf'
VISION = 'gguf/DeepSeek-V4.1-Flash-Vision.gguf'
# The memory curator's nightly prompt of 2026-10-22, the request that failed at the old window.
CURATOR_PROMPT_TOKENS = 98557

FAKE_SERVER = '''#!/bin/bash
echo "$$" > "$PID_OUT"
if [ -n "${DS4_FAKE_EXIT:-}" ]; then echo "fatal: no such GPU"; exit 7; fi
printf '%s\\n' "$@" > "$ARGS_OUT"
exec /bin/sleep 3
'''


class Ds4Harness:
    def __init__(self, tmp_path: Path, fake_server: Path):
        self.dir = tmp_path
        self.ds4 = tmp_path / 'ds4'
        (self.ds4 / 'gguf').mkdir(parents=True)
        (self.ds4 / MODEL).write_text('model')
        (self.ds4 / 'ds4-server').symlink_to(fake_server)
        self.args_out = tmp_path / 'args'
        self.pid_out = tmp_path / 'pid'
        self.env_file = tmp_path / 'stubs.bash'
        # curl answers once the server has started (its arguments are on disk); lsof replays
        # the pids staged for the port, none by default. The wait loop's sleep is short, not
        # gone: the fork of the server takes longer than 600 instant iterations.
        self.env_file.write_text(
            'sleep() { /bin/sleep 0.01; }\n'
            f'lsof() {{ cat "{tmp_path}/lsof_out" 2>/dev/null; return 0; }}\n'
            f'curl() {{ [ -f "{self.args_out}" ] && echo "{{}}"; }}\n')

    def with_vision(self):
        (self.ds4 / VISION).write_text('encoder')

    def port_taken_by(self, pid: int):
        (self.dir / 'lsof_out').write_text(f'{pid}\n')

    def run(self, **env) -> subprocess.CompletedProcess:
        full = dict(os.environ, BASH_ENV=str(self.env_file), DS4_DIR=str(self.ds4),
                    KV_DIR=str(self.dir / 'kv'), ARGS_OUT=str(self.args_out), PID_OUT=str(self.pid_out), **env)
        return subprocess.run(['bash', str(SCRIPT)], env=full, capture_output=True, text=True, timeout=30)

    @property
    def launched(self) -> bool:
        return self.args_out.exists()

    def flags(self) -> dict[str, str]:
        """The server's arguments as `{flag: value}`; every flag of the script takes one."""
        lines = self.args_out.read_text().splitlines()
        return {lines[i]: lines[i + 1] for i in range(0, len(lines), 2)}

    def stop(self):
        if self.pid_out.exists():
            try:
                os.kill(int(self.pid_out.read_text()), signal.SIGKILL)
            except (ProcessLookupError, ValueError):
                pass


@pytest.fixture(scope='module')
def fake_server(tmp_path_factory) -> Path:
    # One executable for the module: macOS scans a freshly written executable on its first
    # exec (~110 ms), a symlink to a scanned one runs in 30 ms.
    server = tmp_path_factory.mktemp('ds4-bin') / 'ds4-server'
    server.write_text(FAKE_SERVER)
    server.chmod(0o755)
    return server


@pytest.fixture
def harness(tmp_path, fake_server):
    h = Ds4Harness(tmp_path, fake_server)
    yield h
    h.stop()


@pytest.mark.unit
class TestLocalDs4StartSh:
    def test_the_default_window_holds_the_curators_nightly_prompt_and_the_cold_ceiling_follows_it(self, harness):
        result = harness.run()
        assert result.returncode == 0, result.stdout + result.stderr
        flags = harness.flags()
        assert int(flags['--ctx']) > CURATOR_PROMPT_TOKENS
        assert flags['--kv-cache-cold-max-tokens'] == flags['--ctx']
        assert flags['--kv-disk-dir'] == str(harness.dir / 'kv') and (harness.dir / 'kv').is_dir()

    def test_an_overridden_window_moves_the_cold_ceiling_with_it(self, harness):
        assert harness.run(CTX='32768').returncode == 0
        flags = harness.flags()
        assert flags['--ctx'] == '32768' and flags['--kv-cache-cold-max-tokens'] == '32768'

    def test_an_overridden_cold_ceiling_stands_on_its_own(self, harness):
        assert harness.run(CTX='32768', COLD_MAX='30000').returncode == 0
        assert harness.flags()['--kv-cache-cold-max-tokens'] == '30000'

    def test_the_vision_encoder_rides_only_when_its_file_is_there(self, harness):
        without = harness.run()
        assert without.returncode == 0
        assert '--vision' not in harness.flags()
        assert 'without images' in without.stdout
        harness.stop()
        harness.args_out.unlink()

        harness.with_vision()
        with_it = harness.run()
        assert with_it.returncode == 0
        assert harness.flags()['--vision'] == VISION
        assert 'without images' not in with_it.stdout

    def test_an_empty_vision_override_drops_the_encoder_even_when_its_file_is_there(self, harness):
        harness.with_vision()
        assert harness.run(VISION='').returncode == 0
        assert '--vision' not in harness.flags()

    def test_an_empty_cold_anchor_drops_the_flag_an_upstream_ds4_refuses(self, harness):
        assert harness.run().returncode == 0
        assert '--kv-cache-cold-anchor' in harness.flags()
        harness.stop()
        harness.args_out.unlink()

        assert harness.run(COLD_ANCHOR='').returncode == 0
        assert '--kv-cache-cold-anchor' not in harness.flags()

    def test_a_taken_port_refuses_to_start_a_second_server(self, harness):
        harness.port_taken_by(4242)
        result = harness.run()
        assert result.returncode == 1
        assert '4242' in result.stdout
        assert not harness.launched

    def test_a_missing_model_is_refused_before_the_server_starts(self, harness):
        (harness.ds4 / MODEL).unlink()
        result = harness.run()
        assert result.returncode == 1
        assert MODEL in result.stdout
        assert not harness.launched

    def test_a_server_that_dies_during_startup_fails_with_its_log_tail(self, harness):
        result = harness.run(DS4_FAKE_EXIT='1')
        assert result.returncode == 1
        assert 'no such GPU' in result.stdout   # the server's last words, so the cause is in the terminal
        assert not harness.launched
