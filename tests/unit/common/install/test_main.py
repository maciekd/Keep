"""`python -m common.install`: the steps in order, the exit status install.sh reads, and what the
package may import."""
import os
import subprocess
import sys

import pytest
import yaml

from common.install import __main__ as entry
from common.install.Terminal import InstallFailed
from common.Paths import SRC_DIR


@pytest.fixture
def env(tmp_path):
    return {'KEEP_DIR': str(tmp_path), 'KEEP_INSTALL_BRANCH': 'release', 'KEEP_INSTALL_LOG': str(tmp_path / 'install.log'),
            'KEEP_INSTALL_TELEMETRY_ID': '11111111-2222-4333-8444-555555555555', 'KEEP_NO_TELEMETRY': '1',
            'KEEP_INSTALL_STEP': '7', 'KEEP_INSTALL_STEPS': '9', 'KEEP_LAN_ACCESS': '0'}


@pytest.fixture
def key_step(monkeypatch):
    """The key intake replaced: `asked` counts its runs, `outcome` is what it returns or raises."""
    state = {'asked': 0, 'outcome': ('sk-ant-typed-47L', 'wrkspc_01TEST')}

    class Intake:
        def __init__(self, terminal, env, keep_dir):
            pass

        def obtain(self):
            state['asked'] += 1
            if isinstance(state['outcome'], BaseException):
                raise state['outcome']
            return state['outcome']
    monkeypatch.setattr(entry, 'AnthropicKey', Intake)
    return state


def logged(tmp_path) -> list[str]:
    return (tmp_path / 'install.log').read_text().splitlines()


@pytest.mark.unit
class TestTheSteps:
    def test_a_server_gets_its_key_and_then_the_config_that_holds_it(self, tmp_path, env, key_step, capsys):
        assert entry.main(env) == 0
        config = yaml.safe_load((tmp_path / 'config.yaml').read_text())
        assert config['api_keys'] == {'anthropic': 'sk-ant-typed-47L', 'anthropic_workspace': 'wrkspc_01TEST'}
        headers = [line for line in logged(tmp_path) if line.startswith('=== ')]
        assert [header.split()[1] for header in headers] == ['[8/9]', '[9/9]'], "the counter does not go on from install.sh's"
        assert 'sk-ant-typed-47L' not in capsys.readouterr().out

    def test_a_satellite_is_asked_for_no_key(self, tmp_path, env, key_step):
        env.update(KEEP_INSTALL_ROLE='imessage_satellite', KEEP_INSTALL_STEP='6', KEEP_INSTALL_STEPS='7')
        assert entry.main(env) == 0
        assert key_step['asked'] == 0
        config = yaml.safe_load((tmp_path / 'config.yaml').read_text())
        assert config['role'] == 'imessage_satellite' and 'api_keys' not in config
        assert [line.split()[1] for line in logged(tmp_path) if line.startswith('=== ')] == ['[7/7]']

    def test_a_failure_is_shown_once_and_left_in_the_log_for_install_sh(self, tmp_path, env, key_step, capsys):
        """Contract with install.sh's `configure`: exit 1, the step and the reason on the log's
        last two lines."""
        key_step['outcome'] = InstallFailed('No valid API key after 3 attempts.', 'Re-run the installer when you have one.')
        assert entry.main(env) == 1
        assert not (tmp_path / 'config.yaml').exists()
        assert logged(tmp_path)[-2:] == ['FAILED in step: Anthropic API key', 'FAILED: No valid API key after 3 attempts.']
        said = capsys.readouterr().out
        assert said.count('No valid API key') == 1 and 'Re-run the installer when you have one.' in said

    def test_an_error_nobody_foresaw_fails_the_step_with_its_trace_in_the_log(self, tmp_path, env, key_step, capsys):
        key_step['outcome'] = RuntimeError('the socket fell off')
        assert entry.main(env) == 1
        lines = logged(tmp_path)
        assert lines[-2] == 'FAILED in step: Anthropic API key' and lines[-1].startswith('FAILED: ')
        assert any('Traceback' in line for line in lines)
        assert 'Traceback' not in capsys.readouterr().out

    def test_ctrl_c_ends_it_with_the_status_of_an_interrupt(self, tmp_path, env, key_step):
        """Contract with install.sh, which takes 130 for the user's Ctrl-C and reports no failure."""
        key_step['outcome'] = KeyboardInterrupt()
        assert entry.main(env) == 130
        assert not any(line.startswith('FAILED') for line in logged(tmp_path))


@pytest.mark.unit
def test_the_package_imports_nothing_of_the_assistant_and_nothing_that_reads_the_config(tmp_path):
    """It runs for a satellite too, and before there is a config.yaml: `ConfigManager` would write
    the server's skeleton where the installer is about to write the real file."""
    config = tmp_path / 'config.yaml'
    script = ('import sys\nimport common.install.__main__\n'
              'trees = ("assistant", "integrations", "interfaces")\n'
              'print(sorted(n for n in sys.modules if n.split(".")[0] in trees or n == "common.ConfigManager"))\n')
    result = subprocess.run([sys.executable, '-P', '-c', script], capture_output=True, text=True, timeout=60,
                            env={**os.environ, 'PYTHONPATH': str(SRC_DIR), 'FLOXY_CONFIG': str(config)})
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip() == '[]'
    assert not config.exists()
