from unittest.mock import Mock

import pytest
from spotipy.exceptions import SpotifyException

from assistant.integration.music.SpotifyAuth import NotSignedIn, TokenUnavailable
from integrations.heos.SpotifyConnect import ConnectError, NotListed, SpotifyConnect

KITCHEN_ID = "a" * 40
DEN_ID = "b" * 40


def make(devices=None, device_names=None, album_uri="spotify:album:alb1"):
    user = Mock()
    user.devices.return_value = {"devices": devices if devices is not None else [
        {"id": KITCHEN_ID, "name": "Kitchen", "is_active": False},
        {"id": DEN_ID, "name": "Den Speaker", "is_active": True},
        {"id": "c" * 40, "name": "Marcus's iPhone", "is_active": False}]}
    app = Mock()
    if isinstance(album_uri, Exception):
        app.track.side_effect = album_uri
    else:
        app.track.return_value = {"album": {"uri": album_uri}}
    account = Mock()
    account.user_client.return_value = user
    account.app_client.return_value = app
    return SpotifyConnect(account, device_names), user


def test_a_device_matches_its_player_by_name_case_aside():
    connect, _ = make()
    assert connect.device_for("KITCHEN").id == KITCHEN_ID


def test_a_mapped_player_matches_its_connect_name():
    connect, _ = make(device_names={"Den": "Den Speaker"})
    assert connect.device_for("den").id == DEN_ID


def test_a_player_the_account_does_not_list_is_not_played_to():
    connect, user = make()
    with pytest.raises(NotListed):
        connect.play("Porch", "spotify:playlist:p1")
    user.start_playback.assert_not_called()


def test_a_playlist_plays_as_its_context_shuffled():
    connect, user = make()
    connect.play("Kitchen", "spotify:playlist:p1")
    user.start_playback.assert_called_once_with(device_id=KITCHEN_ID, context_uri="spotify:playlist:p1")
    user.shuffle.assert_called_once_with(True, device_id=KITCHEN_ID)


def test_an_artist_plays_as_its_context_in_order():
    connect, user = make()
    connect.play("Kitchen", "spotify:artist:ar1")
    assert user.start_playback.call_args.kwargs["context_uri"] == "spotify:artist:ar1"
    user.shuffle.assert_not_called()


def test_a_track_plays_inside_its_album_from_that_track():
    connect, user = make()
    connect.play("Kitchen", "spotify:track:t1")
    kwargs = user.start_playback.call_args.kwargs
    assert kwargs["context_uri"] == "spotify:album:alb1"
    assert kwargs["offset"] == {"uri": "spotify:track:t1"}
    user.shuffle.assert_not_called()   # what follows the asked-for track is its album, in order


def test_a_track_whose_album_cannot_be_read_plays_alone():
    connect, user = make(album_uri=RuntimeError("lookup failed"))
    connect.play("Kitchen", "spotify:track:t1")
    assert user.start_playback.call_args.kwargs["uris"] == ["spotify:track:t1"]


def test_a_premium_refusal_says_premium_is_needed():
    connect, user = make()
    user.start_playback.side_effect = SpotifyException(403, -1, "Player command failed", reason="PREMIUM_REQUIRED")
    with pytest.raises(ConnectError, match="Premium"):
        connect.play("Kitchen", "spotify:playlist:p1")


def test_a_404_says_the_player_is_not_in_the_list():
    connect, user = make()
    user.start_playback.side_effect = SpotifyException(404, -1, "Device not found")
    with pytest.raises(ConnectError, match="device list"):
        connect.play("Kitchen", "spotify:playlist:p1")


def test_an_account_not_signed_in_is_a_connect_error():
    connect, _ = make()
    connect.account.user_client.side_effect = NotSignedIn("not signed in — spotify_login()")
    with pytest.raises(ConnectError, match="not signed in"):
        connect.play("Kitchen", "spotify:playlist:p1")


def test_an_album_is_shuffled_like_a_playlist():
    connect, user = make()
    connect.play("Kitchen", "spotify:album:alb9")
    assert user.start_playback.call_args.kwargs == {"device_id": KITCHEN_ID, "context_uri": "spotify:album:alb9"}
    user.shuffle.assert_called_once_with(True, device_id=KITCHEN_ID)


def test_a_listed_entry_without_an_id_is_no_device_to_play_on():
    """Spotify lists a restricted device with a null id; its name must not match a player."""
    connect, user = make(devices=[{"id": None, "name": "Kitchen"}, {"id": DEN_ID, "name": "Den Speaker"}])
    assert [d.id for d in connect.devices()] == [DEN_ID]
    with pytest.raises(NotListed):
        connect.play("Kitchen", "spotify:playlist:p1")
    user.start_playback.assert_not_called()


def test_a_device_list_spotify_refuses_is_a_connect_error_and_nothing_plays():
    connect, user = make()
    user.devices.side_effect = SpotifyException(502, -1, "Bad gateway")
    with pytest.raises(ConnectError, match="502") as refused:
        connect.play("Kitchen", "spotify:playlist:p1")
    assert not isinstance(refused.value, NotListed), "an unread list is not a player the account does not list"
    user.start_playback.assert_not_called()


def test_a_token_that_cannot_be_had_for_now_is_a_connect_error_that_does_not_ask_for_a_sign_in():
    connect, _ = make()
    connect.account.user_client.side_effect = TokenUnavailable("the token endpoint could not be reached")
    with pytest.raises(ConnectError) as refused:
        connect.play("Kitchen", "spotify:playlist:p1")
    assert "token endpoint" in str(refused.value) and "not signed in" not in str(refused.value)


def test_another_refusal_names_the_player_and_spotifys_status():
    connect, user = make()
    user.start_playback.side_effect = SpotifyException(403, -1, "Restriction violated", reason="UNKNOWN")
    with pytest.raises(ConnectError) as refused:
        connect.play("Kitchen", "spotify:playlist:p1")
    assert "Kitchen" in str(refused.value) and "403" in str(refused.value) and "Premium" not in str(refused.value)

