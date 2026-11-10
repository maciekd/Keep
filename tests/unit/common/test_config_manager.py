import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from common import Paths
from common.ConfigManager import ConfigManager


@pytest.mark.unit
class TestConfigManagerGet:

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        ConfigManager.reset_instance()
        yield
        ConfigManager.reset_instance()

    @pytest.fixture
    def config_dir(self, tmp_path):
        config_data = {
            'api_keys': {
                'openai': 'test-openai-key',
                'elevenlabs': 'test-elevenlabs-key',
            },
            'location': {
                'home': {
                    'latitude': 59.91,
                    'longitude': 10.75
                }
            },
            'voice_cache_size_in_gb': 3,
            'indigo': {
                'login': 'admin',
                'password': 'secret'
            }
        }
        config_file = tmp_path / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        return tmp_path

    def _create_config(self, config_dir):
        with patch.dict(os.environ, {'FLOXY_CONFIG': str(config_dir / 'config.yaml')}):
            return ConfigManager()

    def test_get_top_level_key(self, config_dir):
        cm = self._create_config(config_dir)
        assert cm.get('voice_cache_size_in_gb') == 3

    def test_get_deeply_nested_key(self, config_dir):
        cm = self._create_config(config_dir)
        assert cm.get('location.home.latitude') == 59.91

    def test_get_returns_dict_for_intermediate_key(self, config_dir):
        cm = self._create_config(config_dir)
        result = cm.get('location.home')
        assert isinstance(result, dict)
        assert result['latitude'] == 59.91
        assert result['longitude'] == 10.75

    def test_get_missing_key_with_default(self, config_dir):
        cm = self._create_config(config_dir)
        assert cm.get('nonexistent.key', 'fallback') == 'fallback'

    def test_none_is_a_default_like_any_other(self, config_dir):
        """Issue #174: `get(key, None)` raised for a missing key, so a reader whose "unset" is
        None failed on every install that had not written the key."""
        cm = self._create_config(config_dir)
        assert cm.get('nonexistent.key', None) is None
        assert cm.get('voice_cache_size_in_gb.deeper', None) is None

    def test_get_missing_key_without_default_raises_key_error(self, config_dir):
        cm = self._create_config(config_dir)
        with pytest.raises(KeyError, match="not found"):
            cm.get('nonexistent.key')

    def test_get_path_through_non_dict_with_default(self, config_dir):
        cm = self._create_config(config_dir)
        assert cm.get('voice_cache_size_in_gb.deeper', 'default') == 'default'

    def test_get_path_through_non_dict_without_default_raises(self, config_dir):
        cm = self._create_config(config_dir)
        with pytest.raises(KeyError, match="not found"):
            cm.get('voice_cache_size_in_gb.deeper')

@pytest.mark.unit
class TestConfigManagerApiKey:

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        ConfigManager.reset_instance()
        yield
        ConfigManager.reset_instance()

    @pytest.fixture
    def config_dir(self, tmp_path):
        config_data = {
            'api_keys': {
                'openai': 'test-openai-key',
            },
            'location': {
                'home': {
                    'latitude': 40.71,
                    'longitude': -74.00
                }
            },
        }
        config_file = tmp_path / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        return tmp_path

    def _create_config(self, config_dir):
        with patch.dict(os.environ, {'FLOXY_CONFIG': str(config_dir / 'config.yaml')}):
            return ConfigManager()

    def test_get_api_key(self, config_dir):
        cm = self._create_config(config_dir)
        assert cm.get_api_key('openai') == 'test-openai-key'

    def test_get_api_key_missing_service_is_none(self, config_dir):
        cm = ConfigManager()
        assert cm.get_api_key('nonexistent_service') is None

    def test_get_home_latitude(self, config_dir):
        cm = self._create_config(config_dir)
        assert cm.get_home_latitude() == 40.71

@pytest.mark.unit
class TestConfigManagerFileHandling:

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        ConfigManager.reset_instance()
        yield
        ConfigManager.reset_instance()

    def test_missing_config_file_gets_a_skeleton_asking_for_the_required_keys(self, tmp_path):
        path = tmp_path / 'config.yaml'
        with patch.dict(os.environ, {'FLOXY_CONFIG': str(path)}):
            cm = ConfigManager()
        assert cm.created_skeleton
        text = path.read_text()
        assert path.exists() and "anthropic:" in text and "vault:" in text
        assert "gemini" not in text   # set up during the first run, not needed for a boot
        assert cm.missing_required() == ["vault.key"]
        assert cm.missing_required(("api_keys.anthropic", "vault.key")) == ["api_keys.anthropic", "vault.key"]
        assert cm.get("interfaces.voice.enabled", True) is True   # the skeleton leaves the defaults alone

    def test_missing_required_spots_placeholders_and_blanks(self, tmp_path):
        path = tmp_path / 'config.yaml'
        path.write_text("api_keys:\n  anthropic: YOUR_ANTHROPIC_API_KEY_HERE\nvault:\n  key: CHANGE_ME\n")
        with patch.dict(os.environ, {'FLOXY_CONFIG': str(path)}):
            assert ConfigManager().missing_required(("api_keys.anthropic", "vault.key")) == ["api_keys.anthropic", "vault.key"]
        ConfigManager.reset_instance()
        path.write_text("api_keys:\n  anthropic: sk-ant-real\nvault:\n  key: hunter2\n")
        with patch.dict(os.environ, {'FLOXY_CONFIG': str(path)}):
            cm = ConfigManager()
        assert cm.missing_required() == [] and not cm.created_skeleton

    def test_malformed_yaml_raises_value_error(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("[unclosed bracket")
        with patch.dict(os.environ, {'FLOXY_CONFIG': str(tmp_path / 'config.yaml')}):
            with pytest.raises(ValueError, match="Invalid YAML"):
                ConfigManager()

@pytest.mark.unit
class TestConfigManagerSingleton:

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        ConfigManager.reset_instance()
        yield
        ConfigManager.reset_instance()

    def test_get_instance_returns_same_object(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("key: value\n")
        with patch.dict(os.environ, {'FLOXY_CONFIG': str(tmp_path / 'config.yaml')}):
            instance1 = ConfigManager.get_instance()
            instance2 = ConfigManager.get_instance()
            assert instance1 is instance2

    def test_reset_instance_allows_different_config(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("key: value1\n")
        with patch.dict(os.environ, {'FLOXY_CONFIG': str(tmp_path / 'config.yaml')}):
            cm1 = ConfigManager.get_instance()
            assert cm1.get('key') == 'value1'

        ConfigManager.reset_instance()

        config_file.write_text("key: value2\n")
        with patch.dict(os.environ, {'FLOXY_CONFIG': str(tmp_path / 'config.yaml')}):
            cm2 = ConfigManager.get_instance()
            assert cm2.get('key') == 'value2'


@pytest.mark.unit
class TestConfigPathResolution:
    def test_env_var_wins(self, tmp_path):
        with patch.dict(os.environ, {'FLOXY_CONFIG': str(tmp_path / 'other.yaml')}):
            assert ConfigManager.config_path() == str(tmp_path / 'other.yaml')

    def test_default_is_aiserver_config_yaml_regardless_of_argv_and_cwd(self, tmp_path, monkeypatch):
        monkeypatch.delenv('FLOXY_CONFIG', raising=False)
        monkeypatch.setattr('sys.argv', ['-'])
        monkeypatch.chdir(tmp_path)
        path = Path(ConfigManager.config_path())
        assert path.name == 'config.yaml'
        assert path.parent == Paths.AISERVER_DIR
        assert (path.parent / 'src' / 'common' / 'ConfigManager.py').exists()


@pytest.mark.unit
class TestEnabledIntegrations:
    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        ConfigManager.reset_instance()
        yield
        ConfigManager.reset_instance()

    def _cm(self, tmp_path, text):
        path = tmp_path / "config.yaml"
        path.write_text(text)
        with patch.dict(os.environ, {"FLOXY_CONFIG": str(path)}):
            return ConfigManager()

    def test_listed_names_match_loosely(self, tmp_path):
        cm = self._cm(tmp_path, "enabled_integrations:\n  - Tesla\n  - sonos & spotify\n  - Telegram Feed\n")
        assert cm.enabled_integrations() == ["Tesla", "sonos & spotify", "Telegram Feed"]
        assert cm.integration_enabled("tesla") and cm.integration_enabled("Sonos & Spotify")
        assert cm.integration_enabled("telegramfeed") and cm.integration_enabled("Telegram feed")
        assert not cm.integration_enabled("Ring")
        assert cm.integration_enabled("Web Access", default=True)
        assert not cm.integration_enabled("Web Access")

    def test_missing_or_odd_list(self, tmp_path):
        assert self._cm(tmp_path, "api_keys: {}\n").enabled_integrations() == []
        ConfigManager.reset_instance()
        assert self._cm(tmp_path, "enabled_integrations: Tesla\n").integration_enabled("Tesla")
        ConfigManager.reset_instance()
        assert self._cm(tmp_path, "enabled_integrations:\n").enabled_integrations() == []

    def test_integration_key(self):
        from common.ConfigManager import integration_key
        assert integration_key("Sonos & Spotify") == integration_key("sonos-and-spotify") \
            == integration_key("SonosSpotify") == integration_key("sonos_spotify") == "sonosspotify"
        assert integration_key("Reminders") == "reminders"   # 'and' inside a word stays
        assert integration_key("iCloud Email") == "icloudemail"
