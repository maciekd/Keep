"""The Sonos-only command left in the folder, sonos_enqueue_and_play: a favorite by name, else a
playlist, never in a zone that is not there; and building the commands probes no speaker."""
from unittest.mock import Mock, patch

import pytest

from assistant.commands.Commands import Commands
from integrations.sonos.SonosIntegration import SonosIntegration


@pytest.fixture
def music_manager():
    manager = Mock()
    manager.zone_problem.return_value = ""
    with patch.object(SonosIntegration, "get_instance", return_value=manager):
        yield manager


class TestSonosEnqueueAndPlay:
    def test_a_favorite_by_name_plays_as_the_favorite(self, commands_instance, music_manager):
        music_manager.get_favorites.return_value = ["Morning Mix"]
        commands_instance.run("sonos_enqueue_and_play", ["Kitchen", "Morning Mix"])
        music_manager.play_favorite.assert_called_once_with("Kitchen", "Morning Mix")
        music_manager.play_playlist.assert_not_called()

    def test_any_other_name_plays_as_a_shuffled_playlist(self, commands_instance, music_manager):
        music_manager.get_favorites.return_value = ["Morning Mix"]
        commands_instance.run("sonos_enqueue_and_play", ["Kitchen", "Dinner"])
        music_manager.play_playlist.assert_called_once_with("Kitchen", "Dinner", shuffle=True)

    def test_an_unknown_zone_is_the_commands_error_and_nothing_plays(self, commands_instance, music_manager):
        music_manager.zone_problem.return_value = "no Sonos zone 'Garage'; the zones are: Bedroom"
        result = commands_instance.run("sonos_enqueue_and_play", ["Garage", "Dinner"])
        assert "sonos_enqueue_and_play" in result and "Bedroom" in result
        music_manager.play_favorite.assert_not_called()
        music_manager.play_playlist.assert_not_called()

    def test_building_commands_does_not_build_the_music_integration(self):
        # A disabled Sonos must not be probed at startup: the integration is built by the
        # IntegrationMonitor when enabled, or by the first Sonos command.
        with patch.object(SonosIntegration, "get_instance") as build:
            Commands(Mock())
        build.assert_not_called()
