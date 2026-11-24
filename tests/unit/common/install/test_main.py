"""`python -m common.install`: the steps in order, the exit status install.sh reads, and what the
package may import."""
import os
import subprocess
import sys

import pytest
import yaml

from common.install import __main__ as entry
from common.install.AiChoice import CLOUD, LOCAL
from common.install.LocalRunner import LocalModel
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


@pytest.fixture(autouse=True)
def ai(monkeypatch):
    """Where the install thinks, decided by the test and not by the machine it runs on: `choice`
    is the answer, `explicit` whether KEEP_AI=local gave it, `outcome` what the local setup returns
    or raises, `set_up` how often it ran."""
    state = {'choice': CLOUD, 'explicit': False, 'outcome': LocalModel('Qwen3.8-27B-oQ4e-mtp'), 'set_up': 0}

    class Choice:
        def __init__(self, terminal, env):
            self.explicit = state['explicit']
            self.lacks = ''
            self.answer = state['choice']

        def choose(self):
            return state['choice']

    class Runner:
        def __init__(self, terminal, env):
            pass

        def set_up(self):
            state['set_up'] += 1
            if isinstance(state['outcome'], BaseException):
                raise state['outcome']
            return state['outcome']
    monkeypatch.setattr(entry, 'AiChoice', Choice)
    monkeypatch.setattr(entry, 'LocalRunner', Runner)
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

    def test_a_satellite_is_not_asked_where_it_thinks(self, tmp_path, env, key_step, ai):
        ai['choice'] = LOCAL
        env.update(KEEP_INSTALL_ROLE='imessage_satellite')
        assert entry.main(env) == 0
        assert ai['set_up'] == 0 and 'local_backend' not in yaml.safe_load((tmp_path / 'config.yaml').read_text())

    def test_a_failure_is_shown_once_and_left_in_the_log_for_install_sh(self, tmp_path, env, key_step, capsys):
        """Contract with install.sh's `configure`: exit 1, the step and the reason on the log's
        last two lines."""
        key_step['outcome'] = InstallFailed('No valid API key after 3 attempts.', 'Re-run the installer when you have one.')
        assert entry.main(env) == 1
        assert not (tmp_path / 'config.yaml').exists()
        assert logged(tmp_path)[-2:] == ['FAILED in step: AI model', 'FAILED: No valid API key after 3 attempts.']
        said = capsys.readouterr().out
        assert said.count('No valid API key') == 1 and 'Re-run the installer when you have one.' in said

    def test_an_error_nobody_foresaw_fails_the_step_with_its_trace_in_the_log(self, tmp_path, env, key_step, capsys):
        key_step['outcome'] = RuntimeError('the socket fell off')
        assert entry.main(env) == 1
        lines = logged(tmp_path)
        assert lines[-2] == 'FAILED in step: AI model' and lines[-1].startswith('FAILED: ')
        assert any('Traceback' in line for line in lines)
        assert 'Traceback' not in capsys.readouterr().out

    def test_ctrl_c_ends_it_with_the_status_of_an_interrupt(self, tmp_path, env, key_step):
        """Contract with install.sh, which takes 130 for the user's Ctrl-C and reports no failure."""
        key_step['outcome'] = KeyboardInterrupt()
        assert entry.main(env) == 130
        assert not any(line.startswith('FAILED') for line in logged(tmp_path))


@pytest.mark.unit
class TestTheLocalModel:
    def config(self, tmp_path) -> dict:
        return yaml.safe_load((tmp_path / 'config.yaml').read_text())

    def test_a_local_install_is_asked_for_no_key_and_its_config_holds_none(self, tmp_path, env, key_step, ai):
        ai['choice'] = LOCAL
        assert entry.main(env) == 0
        assert key_step['asked'] == 0 and ai['set_up'] == 1
        config = self.config(tmp_path)
        assert config['local_backend'] == 'openai' and config['openai_compatible']['model'] == 'Qwen3.8-27B-oQ4e-mtp'
        assert 'api_keys' not in config
        assert [line.split()[1] for line in logged(tmp_path) if line.startswith('=== ')] == ['[8/9]', '[9/9]']

    def test_the_cloud_sets_up_nothing_on_this_mac(self, tmp_path, env, key_step, ai):
        assert entry.main(env) == 0
        assert ai['set_up'] == 0 and 'local_backend' not in self.config(tmp_path)

    def reported(self, tmp_path) -> dict:
        """The facts logged for install.sh's event — contract with its `configure`: "report: name=value"."""
        return dict(line[len('report: '):].split('=', 1) for line in logged(tmp_path) if line.startswith('report: '))

    def test_the_install_report_is_told_what_was_offered_and_answered(self, tmp_path, env, key_step, ai):
        ai['choice'] = LOCAL
        entry.main(env)
        report = self.reported(tmp_path)
        assert (report['ai_offered'], report['ai_choice']) == ('yes', LOCAL)
        assert report['ai_setup_s'].isdigit() and 'ai_fallback' not in report

    def test_a_fallback_to_claude_is_reported_with_its_reason(self, tmp_path, env, key_step, ai):
        ai.update(choice=LOCAL, outcome=InstallFailed('oMLX does not answer on port 8000.', 'Its log: var/log/omlx.log'))
        entry.main(env)
        report = self.reported(tmp_path)
        assert report['ai_fallback'] == 'oMLX does not answer on port 8000.' and report['ai_setup_s'].isdigit()

    def test_the_cloud_reports_no_local_setup(self, tmp_path, env, key_step, ai):
        entry.main(env)
        report = self.reported(tmp_path)
        assert report['ai_choice'] == CLOUD and 'ai_setup_s' not in report and 'ai_fallback' not in report

    def test_a_failed_install_still_ends_its_log_with_the_step_and_the_reason(self, tmp_path, env, key_step, ai):
        """install.sh reads the failure off the log's last two lines: no report line may follow them."""
        ai.update(choice=LOCAL, explicit=True, outcome=InstallFailed('oMLX does not answer on port 8000.'))
        assert entry.main(env) == 1
        lines = logged(tmp_path)
        assert lines[-2].startswith('FAILED in step: ') and lines[-1] == 'FAILED: oMLX does not answer on port 8000.'
        assert self.reported(tmp_path)['ai_choice'] == LOCAL

    @pytest.mark.parametrize('failure', [InstallFailed('oMLX does not answer on port 8000.', 'Its log: var/log/omlx.log'),
                                         RuntimeError('the tap moved')], ids=['a failure it names', 'an error nobody foresaw'])
    def test_a_local_setup_that_fails_costs_the_local_option_not_the_install(self, tmp_path, env, key_step, ai, capsys, failure):
        ai.update(choice=LOCAL, outcome=failure)
        assert entry.main(env) == 0
        assert key_step['asked'] == 1
        config = self.config(tmp_path)
        assert config['api_keys']['anthropic'] == 'sk-ant-typed-47L' and 'local_backend' not in config
        said = capsys.readouterr().out
        assert str(failure) in said and 'Traceback' not in said
        assert not any(line.startswith('FAILED') for line in logged(tmp_path))

    def test_the_unforeseen_error_leaves_its_trace_in_the_log(self, tmp_path, env, key_step, ai):
        ai.update(choice=LOCAL, outcome=RuntimeError('the tap moved'))
        entry.main(env)
        assert any('Traceback' in line for line in logged(tmp_path))

    @pytest.mark.parametrize('failure', [InstallFailed('oMLX does not answer on port 8000.'), RuntimeError('the tap moved')],
                             ids=['a failure it names', 'an error nobody foresaw'])
    def test_local_asked_for_by_name_fails_the_install_where_it_fails(self, tmp_path, env, key_step, ai, failure):
        """KEEP_AI=local: nobody is there to give a key instead."""
        ai.update(choice=LOCAL, explicit=True, outcome=failure)
        assert entry.main(env) == 1
        assert key_step['asked'] == 0 and not (tmp_path / 'config.yaml').exists()
        lines = logged(tmp_path)
        assert lines[-2] == 'FAILED in step: AI model' and str(failure) in lines[-1]

    def test_ctrl_c_during_the_local_setup_is_an_interrupt_not_a_reason_to_go_on_with_claude(self, tmp_path, env, key_step, ai):
        ai.update(choice=LOCAL, outcome=KeyboardInterrupt())
        assert entry.main(env) == 130
        assert key_step['asked'] == 0 and not (tmp_path / 'config.yaml').exists()


@pytest.mark.unit
def test_the_variable_reaches_the_real_choice(tmp_path, env, key_step, monkeypatch):
    """KEEP_AI=cloud through the real `AiChoice`: no question on any machine, and the key step."""
    monkeypatch.undo()
    monkeypatch.setattr(entry, 'AnthropicKey', type('Intake', (), {
        '__init__': lambda self, terminal, env, keep_dir: None, 'obtain': lambda self: ('sk-ant-typed-47L', '')}))
    env['KEEP_AI'] = 'cloud'
    assert entry.main(env) == 0
    assert yaml.safe_load((tmp_path / 'config.yaml').read_text())['api_keys']['anthropic'] == 'sk-ant-typed-47L'


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
