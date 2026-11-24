"""The config.yaml the installer writes: a Keep server's and an iMessage satellite's."""
import os
import re
import stat
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from common.ConfigManager import REQUIRED_KEYS, ConfigManager
from common.install import InstallConfig as module
from common.install.InstallConfig import InstallConfig
from common.install.Terminal import InstallFailed

from .fakes import ScriptedTerminal

INSTALL_SH = Path(__file__).resolve().parents[4] / 'install.sh'
INSTALL_ID = '11111111-2222-4333-8444-555555555555'


def installer(tmp_path, answers=None, branch='master', **env) -> InstallConfig:
    env = {'KEEP_DIR': str(tmp_path), 'KEEP_INSTALL_BRANCH': branch, 'KEEP_INSTALL_TELEMETRY_ID': INSTALL_ID, **env}
    return InstallConfig(ScriptedTerminal(answers, log_path=str(tmp_path / 'install.log')), env)


def written(tmp_path) -> dict:
    return yaml.safe_load((tmp_path / 'config.yaml').read_text())


def server(tmp_path, key='sk-ant-typed-47L', workspace='', **kwargs) -> dict:
    installer(tmp_path, **kwargs).write_server(key, workspace)
    return written(tmp_path)


def satellite(tmp_path, **kwargs) -> dict:
    installer(tmp_path, **kwargs).write_satellite()
    return written(tmp_path)


def loaded(path: Path) -> ConfigManager:
    """A ConfigManager on the file, as the process started on that config has it."""
    with patch.dict(os.environ, FLOXY_CONFIG=str(path)):
        return ConfigManager()


@pytest.mark.unit
class TestTheServersConfig:
    def test_writes_a_private_config_with_every_required_key(self, tmp_path):
        config = server(tmp_path)
        assert stat.S_IMODE((tmp_path / 'config.yaml').stat().st_mode) == 0o600
        assert loaded(tmp_path / 'config.yaml').missing_required(REQUIRED_KEYS) == []
        assert config['api_keys']['anthropic'] == 'sk-ant-typed-47L'
        assert 'anthropic_workspace' not in config['api_keys']   # a key tied to one workspace needs none
        assert re.fullmatch(r'[A-Za-z0-9]{32}', str(config['vault']['key']))
        assert 'role' not in config and 'imessage_satellite' not in config and 'watchdog' not in config

    def test_two_installs_get_two_vault_passphrases(self, tmp_path):
        assert server(tmp_path)['vault']['key'] != server(tmp_path)['vault']['key']

    def test_workspace_id_is_written_for_a_key_not_tied_to_one(self, tmp_path):
        """Contract with config.yaml: ClaudeLLM reads the id as api_keys.anthropic_workspace."""
        assert server(tmp_path, workspace='wrkspc_01TEST')['api_keys']['anthropic_workspace'] == 'wrkspc_01TEST'

    def test_developer_release_follows_the_branch_the_install_was_made_from(self, tmp_path):
        """restart_server pulls the branch `developer_release` names, so a master install
        must keep following master and a release install the release branch."""
        assert server(tmp_path, branch='master')['developer_release'] is True
        assert server(tmp_path, branch='release')['developer_release'] is False

    def test_the_install_id_is_written_as_the_servers(self, tmp_path):
        """Contract with config.yaml: the server reads the install id as telemetry.id."""
        assert server(tmp_path)['telemetry'] == {'enabled': True, 'id': INSTALL_ID}

    def test_the_environment_opt_out_is_written(self, tmp_path):
        assert server(tmp_path, KEEP_NO_TELEMETRY='1')['telemetry'] == {'enabled': False}

    def test_the_vault_passphrase_is_shown_to_need_a_backup_and_is_not_in_the_log(self, tmp_path):
        config = installer(tmp_path)
        config.write_server('sk-ant-typed-47L', '')
        assert 'back' in config.terminal.said
        logged = (tmp_path / 'install.log').read_text()
        assert written(tmp_path)['vault']['key'] not in logged and 'sk-ant-typed-47L' not in logged

    def test_a_value_yaml_would_misread_comes_back_as_it_was_given(self, tmp_path):
        """The values go through a YAML library: an id of digits alone stays a string."""
        config = server(tmp_path, workspace='wrkspc: #1', KEEP_INSTALL_TELEMETRY_ID='12345')
        assert config['api_keys']['anthropic_workspace'] == 'wrkspc: #1'
        assert config['telemetry']['id'] == '12345'

    def test_every_key_is_on_a_line_of_its_own_as_the_scripts_read_it(self, tmp_path):
        """scripts/role.sh and install.sh read the file with awk: block style, a key a line."""
        server(tmp_path, workspace='wrkspc_01TEST')
        lines = [line for line in (tmp_path / 'config.yaml').read_text().splitlines() if not line.lstrip().startswith('#')]
        assert all(re.match(r'^( {2})?[a-z_]+:( \S.*)?$', line) for line in lines), lines


@pytest.mark.unit
class TestTheSatellitesConfig:
    def test_the_role_and_the_port_are_the_ones_the_satellite_and_the_scripts_know(self):
        """A contract with config.yaml (common/Role.py, common/imessage/Satellite.py) and with
        install.sh, whose summary falls back on the same port."""
        from common import Role
        from common.imessage import Satellite
        assert module.SATELLITE_ROLE == Role.IMESSAGE_SATELLITE
        assert module.SATELLITE_PORT == Satellite.DEFAULT_PORT
        assert int(re.search(r'^SATELLITE_PORT=(\d+)', INSTALL_SH.read_text(), re.M).group(1)) == Satellite.DEFAULT_PORT

    def test_the_config_is_private_and_one_the_satellite_starts_on(self, tmp_path):
        """Contract with floxy_ai_server.py: the role it reads, and a section Satellite.py accepts."""
        from common import Role
        from common.imessage import Satellite
        config = satellite(tmp_path)
        path = tmp_path / 'config.yaml'
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        manager = loaded(path)
        with patch.object(Role, 'config', manager), patch.object(Satellite, 'config', manager):
            assert Role.current() == Role.IMESSAGE_SATELLITE
            assert Satellite.config_problem() == ''
            assert Satellite.bind_host() == Satellite.LOOPBACK, "with nobody asked, the satellite opens to this Mac only"
        assert re.fullmatch(r'[A-Za-z0-9]{32}', str(config['imessage_satellite']['token']))
        assert 'api_keys' not in config and 'vault' not in config

    def test_the_watchdog_is_on_for_a_satellite(self, tmp_path):
        """Contract with common/Watchdog.KEY, which the satellite applies at each start: nobody is
        at a satellite to start it again."""
        from common import Watchdog
        assert satellite(tmp_path)[Watchdog.KEY] is True

    def test_two_installs_get_two_tokens(self, tmp_path):
        assert satellite(tmp_path)['imessage_satellite']['token'] != satellite(tmp_path)['imessage_satellite']['token']

    def test_the_token_reaches_neither_the_screen_nor_the_log(self, tmp_path):
        config = installer(tmp_path)
        config.write_satellite()
        token = written(tmp_path)['imessage_satellite']['token']
        assert token not in config.terminal.said and token not in (tmp_path / 'install.log').read_text()

    def test_the_release_line_and_the_telemetry_are_written_as_for_a_server(self, tmp_path):
        config = satellite(tmp_path, branch='master')
        assert config['developer_release'] is True and config['telemetry'] == {'enabled': True, 'id': INSTALL_ID}
        config = satellite(tmp_path, branch='release', KEEP_NO_TELEMETRY='1')
        assert config['developer_release'] is False and config['telemetry'] == {'enabled': False}

    @pytest.mark.parametrize('env, answers, expected, asked', [
        ({}, None, False, 0),
        ({'KEEP_LAN_ACCESS': '1'}, ['2'], True, 0),
        ({'KEEP_LAN_ACCESS': '0'}, ['2'], False, 0),
        ({'KEEP_LAN_ACCESS': 'Yes'}, None, True, 0),
        ({}, [''], False, 1),
        ({}, ['1'], False, 1),
        ({}, ['2'], True, 1),
        ({}, ['yes', ' 2 '], True, 2),
        ({}, ['a', 'b', 'c', '2'], False, 3),
    ], ids=['no terminal', 'the variable on', 'the variable off', 'the variable as a word', 'Enter', 'this Mac',
            'another Mac', 'a wrong answer first', 'three wrong answers'])
    def test_whether_it_answers_the_network_is_asked_once_and_closed_by_default(self, tmp_path, env, answers, expected, asked):
        config = installer(tmp_path, answers=answers, **env)
        config.write_satellite()
        assert written(tmp_path)['imessage_satellite']['lan_access'] is expected
        assert len(config.terminal.prompts) == asked

    def test_a_lan_variable_that_is_neither_yes_nor_no_stops_the_install(self, tmp_path):
        with pytest.raises(InstallFailed) as failure:
            installer(tmp_path, KEEP_LAN_ACCESS='maybe').write_satellite()
        assert 'KEEP_LAN_ACCESS' in failure.value.reason
        assert not (tmp_path / 'config.yaml').exists()
