"""The settings of the podcasts section of config.yaml, read live on every use."""
import os
import re
from dataclasses import dataclass
from datetime import time as clock
from pathlib import Path
from typing import Optional

from common.ConfigManager import ConfigManager
from common.Paths import CACHE_DIR
from common.logging import info

SECTION = "podcasts"
SHOWS_KEY = f"{SECTION}.shows"
DEFAULT_CHECK_TIME = clock(5, 0)
DEFAULT_ARTIFACTS = ("explainer", "linkedin_ideas", "tldr")
DEFAULT_OPENAI_MODEL = "gpt-4o"
DEFAULT_LOCAL_ENDPOINT = "http://localhost:8000"
DEFAULT_LOCAL_MODEL = "qwen3-27b"
DEFAULT_STATE_DIR = CACHE_DIR / "podcasts"
PROVIDER_OPENAI = "openai"
PROVIDER_LOCAL = "local"


@dataclass(frozen=True)
class Show:
    name: str
    playlist_url: str
    output_folder: Path

    @property
    def slug_prefix(self) -> str:
        """"Moonshots" -> "moonshots", the start of every episode slug of the show."""
        return re.sub(r"[^a-z0-9]+", "-", self.name.lower()).strip("-") or "podcast"


@dataclass(frozen=True)
class LlmSettings:
    provider: str
    model: str
    base_url: Optional[str]     # None: the OpenAI API itself
    api_key: Optional[str]

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"


class PodcastsConfig:
    """The podcasts section: the shows, the daily check time, the report LLM and the artifacts."""

    def __init__(self, config: Optional[ConfigManager] = None):
        self._config = config or ConfigManager.get_instance()

    def get(self, key: str, default=None):
        return self._config.get(f"{SECTION}.{key}", default)

    def shows(self) -> list[Show]:
        shows = []
        for entry in self._config.get(SHOWS_KEY, []) or []:
            if not isinstance(entry, dict) or not entry.get("name") or not entry.get("playlist_url") \
                    or not entry.get("output_folder"):
                info(f"Podcasts: a show entry without name, playlist_url and output_folder is left out: {entry!r}")
                continue
            shows.append(Show(str(entry["name"]).strip(), str(entry["playlist_url"]).strip(),
                              Path(str(entry["output_folder"])).expanduser()))
        return shows

    def check_time(self) -> clock:
        value = str(self.get("check_time", "") or "").strip()
        match = re.fullmatch(r"(\d{1,2}):(\d{2})", value)
        if match and int(match.group(1)) < 24 and int(match.group(2)) < 60:
            return clock(int(match.group(1)), int(match.group(2)))
        if value:
            info(f"Podcasts: check_time {value!r} is not HH:MM, using {DEFAULT_CHECK_TIME:%H:%M}")
        return DEFAULT_CHECK_TIME

    def artifacts(self) -> list[str]:
        listed = self.get("artifacts") or list(DEFAULT_ARTIFACTS)
        known = [a for a in listed if a in DEFAULT_ARTIFACTS]
        if len(known) < len(listed):
            info(f"Podcasts: unknown artifacts left out: {sorted(set(listed) - set(known))}")
        return known

    def state_dir(self) -> Path:
        configured = self.get("state_dir")
        path = Path(str(configured)).expanduser() if configured else DEFAULT_STATE_DIR
        path.mkdir(parents=True, exist_ok=True)
        return path

    def llm(self) -> LlmSettings:
        llm = self.get("llm") or {}
        provider = str(llm.get("provider") or PROVIDER_OPENAI).strip().lower()
        if provider == PROVIDER_LOCAL:
            local = llm.get("local") or {}
            endpoint = str(local.get("endpoint") or DEFAULT_LOCAL_ENDPOINT).rstrip("/")
            if not endpoint.endswith("/v1"):
                endpoint += "/v1"
            return LlmSettings(PROVIDER_LOCAL, str(local.get("model") or DEFAULT_LOCAL_MODEL), endpoint,
                               str(local.get("api_key") or "local"))
        if provider != PROVIDER_OPENAI:
            info(f"Podcasts: llm.provider {provider!r} is unknown, using {PROVIDER_OPENAI}")
        return LlmSettings(PROVIDER_OPENAI, str(llm.get("model") or DEFAULT_OPENAI_MODEL), None, self.openai_api_key())

    def openai_api_key(self) -> Optional[str]:
        """podcasts.openai_api_key, else $OPENAI_API_KEY, else api_keys.openai; a placeholder counts as none."""
        for candidate in (self.get("openai_api_key"), os.environ.get("OPENAI_API_KEY"),
                          self._config.get("api_keys.openai")):
            key = str(candidate or "").strip()
            if key and "YOUR_" not in key:
                return key
        return None
