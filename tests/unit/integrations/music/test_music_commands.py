from unittest.mock import Mock, patch

import pytest

from assistant.commands.Commands import Commands
from assistant.commands.Toolkits import TOOLKITS
from integrations.music.MusicCommands import MusicCommands
from integrations.music.SpotifyAuth import REDIRECT_URI, SignIn


@pytest.fixture
def music_manager():
    manager = Mock()
    with patch.object(MusicCommands, '_get_music_manager', return_value=manager):
        yield manager


class TestCmdMusicPlayback:
    def test_volume_up(self, commands_instance, music_manager):
        commands_instance.run("volume_up", ["Living Room"])
        music_manager.adjust_volume.assert_called_once_with("Living Room", 4)

    def test_volume_down(self, commands_instance, music_manager):
        commands_instance.run("volume_down", ["Living Room"])
        music_manager.adjust_volume.assert_called_once_with("Living Room", -4)

    def test_play_music_splits_and_delegates(self, commands_instance, music_manager):
        commands_instance.run("play_music", ["Living Room", "spotify:track:123"])
        music_manager.play_in_zone.assert_called_once_with("spotify:track:123", "Living Room")

    def test_sonos_enqueue_favorite(self, commands_instance, music_manager):
        music_manager.get_favorites.return_value = ["My Fav"]
        commands_instance.run("sonos_enqueue_and_play", ["Kitchen", "My Fav"])
        music_manager.play_favorite.assert_called_once_with("Kitchen", "My Fav")

    def test_sonos_enqueue_playlist(self, commands_instance, music_manager):
        music_manager.get_favorites.return_value = ["Other Fav"]
        commands_instance.run("sonos_enqueue_and_play", ["Kitchen", "My Playlist"])
        music_manager.play_playlist.assert_called_once_with("Kitchen", "My Playlist", shuffle=True)

    def test_building_commands_does_not_build_the_music_integration(self):
        # A disabled Sonos must not be probed at startup: the integration is built by the
        # IntegrationMonitor when enabled, or by the first music command.
        with patch.object(MusicCommands, '_get_music_manager') as build:
            Commands(Mock())
        build.assert_not_called()


MODULE = "integrations.music.MusicCommands"
URL = "https://accounts.spotify.com/authorize?client_id=cid&state=st-1"
PASTED = f"{REDIRECT_URI}?code=AQcode&state=st-1"


def _integration(finish="", playlists=3):
    integration = Mock()
    integration.login_begin.return_value = SignIn(URL, "st-1")
    integration.login_finish.return_value = finish
    integration.spotify_user_name.return_value = "Marcus"
    integration.get_spotify_playlists.return_value = [Mock()] * playlists
    return integration


def _monitor(integration):
    monitor = Mock()
    monitor.find.return_value = integration
    return monitor


class TestSpotifyLogin:
    def test_both_commands_are_in_the_setup_toolkit_and_owned_by_the_music_set(self, commands_instance):
        setup = next(t for t in TOOLKITS if t.name == "setup")
        assert {"spotify_login", "spotify_login_code"} <= set(setup.commands)
        assert isinstance(commands_instance._owner("spotify_login"), MusicCommands)
        assert isinstance(commands_instance._owner("spotify_login_code"), MusicCommands)

    def test_login_runs_in_the_config_lane(self, commands_instance):
        with patch.object(commands_instance, "background_command") as bg:
            assert commands_instance.run("spotify_login", [""]) == ""
        bg.assert_called_once_with(commands_instance._owner("spotify_login")._run_spotify_login, (), lane=Commands.CONFIG_LANE)

    def test_the_link_and_the_paste_instruction_come_back_as_the_result_and_a_note(self, commands_instance):
        integration = _integration()
        with patch.object(Commands, "_integration_monitor", return_value=_monitor(integration)), patch(f"{MODULE}.system_note") as note:
            reply = commands_instance._owner("spotify_login")._run_spotify_login()

        assert URL in reply and "spotify_login_code" in reply and "127.0.0.1:8888" in reply and "will not load" in reply
        assert note.call_args.args[1] == reply
        integration.login_begin.assert_called_once_with()

    def test_a_missing_client_and_a_disabled_integration_are_errors(self, commands_instance):
        integration = _integration()
        integration.login_begin.side_effect = RuntimeError("api_keys.spotify.client_id and api_keys.spotify.client_secret are not set")
        with patch.object(Commands, "_integration_monitor", return_value=_monitor(integration)), patch(f"{MODULE}.system_note") as note:
            text = commands_instance._owner("spotify_login")._run_spotify_login()
        assert text.startswith("spotify_login failed") and "client_id" in text
        note.assert_not_called()

        disabled = Mock(spec=["state"])
        disabled.state = Mock(value="disabled")
        with patch.object(Commands, "_integration_monitor", return_value=_monitor(disabled)):
            assert 'enable_integration("Sonos & Spotify")' in commands_instance._owner("spotify_login")._run_spotify_login()

    def test_the_pasted_address_goes_to_the_background_unquoted_and_whole(self, commands_instance):
        with patch.object(commands_instance, "background_command") as bg:
            assert commands_instance.run("spotify_login_code", [f'"{PASTED}"']) == ""
        bg.assert_called_once_with(commands_instance._owner("spotify_login_code")._run_spotify_login_code, (PASTED,), lane=Commands.CONFIG_LANE)

    def test_without_anything_pasted_it_refuses(self, commands_instance):
        with patch.object(commands_instance, "background_command") as bg:
            assert commands_instance.run("spotify_login_code", [""]).startswith("spotify_login_code failed")
        bg.assert_not_called()

    def test_a_finished_sign_in_names_the_user_and_the_playlist_count_in_the_result_and_a_note(self, commands_instance):
        integration = _integration(playlists=3)
        with patch.object(Commands, "_integration_monitor", return_value=_monitor(integration)), patch(f"{MODULE}.system_note") as note:
            text = commands_instance._owner("spotify_login_code")._run_spotify_login_code(PASTED)

        assert "signed in as Marcus" in text and "3 playlists" in text and "S-0" in text
        assert note.call_args.args[1] == text
        integration.login_finish.assert_called_once_with(PASTED)

    def test_a_failed_sign_in_comes_back_as_text_without_a_note(self, commands_instance):
        integration = _integration(finish="the code expired or was already used — run spotify_login() for a new link")
        with patch.object(Commands, "_integration_monitor", return_value=_monitor(integration)), patch(f"{MODULE}.system_note") as note:
            text = commands_instance._owner("spotify_login_code")._run_spotify_login_code("AQold")

        assert "did not complete" in text and "expired" in text
        note.assert_not_called()
