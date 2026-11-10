import functools
import re
import yaml
import os

from common import ConfigMigrations
from common.Paths import CONFIG_PATH
from common.logging import info
from typing import Optional, Any

# The keys the server cannot start without, whatever the LLM setup: the vault passphrase
# encrypts every secret the assistant is handed later. The launcher adds the api_keys entry of
# the `smart` tier's backend (Claude by default, none at all on a local model — see
# `_required_keys` in floxy_ai_server.py). The Gemini key (fast, video, photo indexing and the
# cache auditor default to it) is set up during the first run; without it those tiers report
# the missing key instead of running. Everything else is an optional integration.
REQUIRED_KEYS = ("vault.key",)
# Written when config.yaml is missing (first start, or after //wipe), so the user has a file
# to put the keys into rather than a traceback.
SKELETON = """# Keep AI configuration. config.example.yaml documents every setting; the assistant can
# set up the optional integrations itself once it runs. These two values are required:
api_keys:
  anthropic:
vault:
  key:    # any passphrase of your choosing; it encrypts vault.bin, the secrets the assistant stores for you.
          # Lose it and the vault is lost; change it and the existing vault.bin can no longer be opened.
"""
_PLACEHOLDER_RE = re.compile(r"YOUR_|CHANGE_ME|_HERE\b")
# The one switch for every integration: a top-level list of names as list_integrations shows
# them. Unlisted = off, except the ones whose class says `enabled_by_default` (Web Access).
ENABLED_INTEGRATIONS_KEY = "enabled_integrations"
_RAISE = object()   # get() without a default: a missing key raises, so None can be a default


def integration_key(name: str) -> str:
    """'Sonos & Spotify', 'sonos-and-spotify' and 'SonosSpotify' all mean the same entry: the
    connector goes, whether written as '&' or 'and', then everything but letters and digits."""
    return re.sub(r"[^a-z0-9]", "", re.sub(r"&|\band\b", "", name.lower()))


class ConfigManager:
    """Loads config.yaml once and serves its values by dot-notation key path."""

    _instance: Optional['ConfigManager'] = None
    created_skeleton: bool = False

    @classmethod
    def get_instance(cls) -> 'ConfigManager':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls):
        cls._instance = None

    def __init__(self):
        self._config = None
        self.load_config()

    @staticmethod
    def config_path() -> str:
        """$FLOXY_CONFIG if set, else AIServer/config.yaml (common.Paths.CONFIG_PATH)."""
        env = os.environ.get("FLOXY_CONFIG")
        if env:
            return env
        return str(CONFIG_PATH)

    def load_config(self):
        """Load configuration from config.yaml file."""
        config_path = self.config_path()

        if not os.path.exists(config_path):
            info(f"No config file at {config_path} — writing a skeleton; the server will ask for the required keys")
            with open(config_path, 'w') as f:
                f.write(SKELETON)
            self.created_skeleton = True

        data = self._read(config_path)
        if ConfigMigrations.needed(data):
            # A release moved sections: rewrite the file before anything reads them, then read again.
            info(f"Config at {config_path} has sections a release moved — migrating")
            if ConfigMigrations.run(config_path):
                data = self._read(config_path)
        self._config = data if isinstance(data, dict) else {}

    @staticmethod
    def _read(config_path: str):
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file: {e}")

    def enabled_integrations(self) -> list[str]:
        """The `enabled_integrations` list as written in config.yaml (a scalar is taken as one name)."""
        value = self.get(ENABLED_INTEGRATIONS_KEY, [])
        if isinstance(value, str):
            value = [value]
        return [str(v).strip() for v in value if str(v).strip()] if isinstance(value, list) else []

    def integration_enabled(self, name: str, default: bool = False) -> bool:
        """Whether *name* is in `enabled_integrations`; *default* (the class's
        `enabled_by_default`) decides when the list does not mention it."""
        listed = {integration_key(n) for n in self.enabled_integrations()}
        return integration_key(name) in listed or default

    def missing_required(self, keys: tuple[str, ...] = REQUIRED_KEYS) -> list[str]:
        """The *keys* (REQUIRED_KEYS by default) that are absent, empty or still an example placeholder."""
        missing = []
        for key in keys:
            value = self.get(key, "")
            if not isinstance(value, str) or not value.strip() or _PLACEHOLDER_RE.search(value):
                missing.append(key)
        return missing

    def reload(self):
        """Re-read config.yaml (after ConfigStore wrote it) so every reader of this instance sees the new values."""
        self.load_config()
        info("Config reloaded")

    def get(self, key_path: str, default=_RAISE) -> Any:
        """
        Get a configuration value using dot notation.

        Example:
            config.get('api_keys.openai')
            config.get('tesla.tessie_access_token')
            config.get('voice_cache_size_in_gb', 2)  # Returns 2 if not found
            config.get('vault.key', None)            # Returns None if not found

        Without a default a missing (or empty) key raises KeyError.
        """
        if self._config is None:
            self.load_config()

        keys = key_path.split('.')
        value = self._config

        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    if default is not _RAISE:
                        return default
                    raise KeyError(f"Configuration key '{key_path}' not found")
            else:
                if default is not _RAISE:
                    return default
                raise KeyError(f"Configuration key '{key_path}' not found")

        return value

    def get_api_key(self, service: str) -> Optional[str]:
        """The key for `service`, or None when it is not configured: an integration without
        its key reports NOT_CONFIGURED instead of failing to construct."""
        return self.get(f'api_keys.{service}', "") or None

    def get_home_latitude(self) -> Optional[float]:
        """Home latitude, or None when location.home is not configured."""
        value = self.get('location.home.latitude', "")
        return None if value == "" else float(value)

    def get_home_longitude(self) -> Optional[float]:
        """Home longitude, or None when location.home is not configured."""
        value = self.get('location.home.longitude', "")
        return None if value == "" else float(value)

    def get_home_timezone(self) -> str:
        """The home's IANA timezone: `location.home.timezone` when set, else the zone the
        home coordinates lie in, else the server's own."""
        configured = self.get('location.home.timezone', "")
        if configured:
            return configured
        return _timezone_at(self.get_home_latitude(), self.get_home_longitude())

    def get_owner_name(self) -> str:
        """Display name of the household owner, used as the `user` of requests that
        carry no sender identity of their own (TUI chat, Telegram fallback)."""
        return self.get('owner_name', 'user')



@functools.lru_cache(maxsize=4)
def _timezone_at(latitude: Optional[float], longitude: Optional[float]) -> str:
    """Looked up once per location: loading timezonefinder's polygons takes a few seconds."""
    if latitude is not None and longitude is not None:
        from timezonefinder import TimezoneFinder
        zone = TimezoneFinder().timezone_at(lat=latitude, lng=longitude)
        if zone:
            info(f"Home timezone: {zone}, from the location.home coordinates")
            return zone
        info(f"Home timezone: no zone found at {latitude}, {longitude} — using the server's")
    from tzlocal import get_localzone_name
    zone = get_localzone_name() or "UTC"
    info(f"Home timezone: {zone}, the server's own (no usable location.home)")
    return zone


config = ConfigManager.get_instance()
