from datetime import time
from unittest.mock import patch

from integrations.podcasts.PodcastsConfig import PodcastsConfig

from .conftest import DictConfig


def test_the_api_key_is_the_sections_then_the_environments_then_api_keys_openai():
    config = PodcastsConfig(DictConfig({"api_keys.openai": "sk-from-api-keys"}))
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-from-env"}):
        assert config.openai_api_key() == "sk-from-env"
    with patch.dict("os.environ", {"OPENAI_API_KEY": ""}):
        assert config.openai_api_key() == "sk-from-api-keys"
        assert PodcastsConfig(DictConfig({"api_keys.openai": "YOUR_OPENAI_API_KEY_HERE"})).openai_api_key() is None
    section = PodcastsConfig(DictConfig({"podcasts.openai_api_key": "sk-section", "api_keys.openai": "sk-other"}))
    assert section.openai_api_key() == "sk-section"


def test_the_local_provider_points_at_the_v1_api_of_the_endpoint():
    llm = PodcastsConfig(DictConfig({"podcasts.llm": {"provider": "local", "local": {
        "endpoint": "http://omlx.local:8000/", "model": "qwen-local"}}})).llm()
    assert (llm.base_url, llm.model) == ("http://omlx.local:8000/v1", "qwen-local")


def test_a_check_time_that_is_no_time_of_day_falls_back_to_five():
    assert PodcastsConfig(DictConfig({"podcasts.check_time": "6:30"})).check_time() == time(6, 30)
    assert PodcastsConfig(DictConfig({"podcasts.check_time": "25:00"})).check_time() == time(5, 0)


def test_a_show_without_a_folder_is_left_out():
    shows = PodcastsConfig(DictConfig({"podcasts.shows": [
        {"name": "Moonshots", "playlist_url": "https://example.com/pl", "output_folder": "/tmp/x"},
        {"name": "Half"}]})).shows()
    assert [(s.name, s.slug_prefix) for s in shows] == [("Moonshots", "moonshots")]
