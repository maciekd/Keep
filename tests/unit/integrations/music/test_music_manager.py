import time

import pytest
from unittest.mock import Mock, patch
import threading

from integrations.music.MusicIntegration import MusicManager, MusicIntegration, SpotifyPlaylist
from integrations.music.SpotifyAuth import NotSignedIn, TokenUnavailable


ROOM_MAPPING = {"Kitchen Front": "Kitchen", "Kitchen Back": ""}


def make_music_manager(room_mapping=ROOM_MAPPING):
    """Create a MusicManager bypassing __init__ to avoid network calls."""
    mm = MusicManager.__new__(MusicManager)
    mm.config = Mock()
    mm.config.get.side_effect = lambda key, default=None: room_mapping if key == 'sonos.room_mapping' else default
    mm._spotify_client = None
    mm._sonos_zones_cache = {}
    mm._sonos_zones_cache_time = None
    mm._playlists_cache = ["Chill Vibes", "Workout Mix", "Jazz Evening"]
    mm._favorites_cache = ["Liked Songs", "Daily Mix 1", "Discover Weekly"]
    mm._spotify_playlists_cache = [SpotifyPlaylist("Discover Weekly", "spotify:playlist:dw1", "Spotify"),
                                   SpotifyPlaylist("Road Trip", "spotify:playlist:rt2", "")]
    mm._spotify_playlists_total = 2
    mm._spotify_library_error = ""
    mm._sign_in_again_noted = False
    mm._spotify_auth = Mock()
    mm._spotify_auth.signed_in.return_value = True
    mm._spotify_auth.needs_sign_in_again.return_value = False
    mm._user_client = None
    mm._user_client_token = ""
    mm._cache_lock = threading.Lock()
    mm._spotify_search_cache = {}
    mm._next_cache_id = 0
    mm._refresh_timer = None
    mm._stopped = threading.Event()
    return mm


class TestMusicManager:
    def test_generate_short_id_sequence(self):
        mm = make_music_manager()
        assert mm._generate_short_id() == "AA"
        assert mm._generate_short_id() == "AB"
        assert mm._generate_short_id() == "AC"

    def test_generate_short_id_wraps_at_676(self):
        mm = make_music_manager()
        mm._next_cache_id = 675  # ZZ
        assert mm._generate_short_id() == "ZZ"
        assert mm._generate_short_id() == "AA"  # wrap

    def test_playlist_uri_resolution(self):
        mm = make_music_manager()
        assert mm._uri_to_playlist_name("P-0") == "Chill Vibes"
        assert mm._uri_to_playlist_name("P-2") == "Jazz Evening"

    def test_favorite_uri_resolution(self):
        mm = make_music_manager()
        assert mm._uri_to_favorite_name("F-0") == "Liked Songs"
        assert mm._uri_to_favorite_name("F-1") == "Daily Mix 1"

    def test_playlist_uri_out_of_range(self):
        mm = make_music_manager()
        assert mm._uri_to_playlist_name("P-99") is None

    def test_favorite_uri_out_of_range(self):
        mm = make_music_manager()
        assert mm._uri_to_favorite_name("F-99") is None

    def test_playlist_uri_invalid_prefix(self):
        mm = make_music_manager()
        assert mm._uri_to_playlist_name("X-0") is None

    def test_favorite_uri_invalid_prefix(self):
        mm = make_music_manager()
        assert mm._uri_to_favorite_name("X-0") is None

    def test_normalize_name_with_room_mapping(self):
        mm = make_music_manager()
        assert mm.normalize_name("Kitchen") == "Kitchen Front"

    def test_normalize_name_without_mapping(self):
        mm = make_music_manager()
        assert mm.normalize_name("Living Room") == "Living Room"

    def test_search_spotify_formats_results(self):
        mm = make_music_manager()
        mock_spotify = Mock()
        mm._get_spotify_client = Mock(return_value=mock_spotify)

        mock_spotify.search.return_value = {
            'tracks': {
                'items': [
                    {
                        'name': 'Song A',
                        'artists': [{'name': 'Artist 1'}],
                        'popularity': 85,
                        'uri': 'spotify:track:abc123'
                    }
                ]
            },
            'artists': {
                'items': [
                    {
                        'name': 'Artist 1',
                        'genres': ['pop', 'rock'],
                        'popularity': 90,
                        'uri': 'spotify:artist:def456'
                    }
                ]
            }
        }

        result = mm.search_spotify("test query")
        assert "Song A" in result
        assert "Artist 1" in result
        assert "85" in result
        assert "TRACKS:" in result
        assert "ARTISTS:" in result
        # Verify short IDs were generated
        assert len(mm._spotify_search_cache) == 2

    def test_search_spotify_lists_results_without_popularity(self):
        """A Development Mode app (created since February 2026) gets no popularity field at all."""
        mm = make_music_manager()
        mock_spotify = Mock()
        mm._get_spotify_client = Mock(return_value=mock_spotify)
        mock_spotify.search.return_value = {
            'tracks': {'items': [{'name': 'Song A', 'artists': [{'name': 'Artist 1'}], 'uri': 'spotify:track:abc123'}]},
            'artists': {'items': [{'name': 'Artist 1', 'genres': [], 'uri': 'spotify:artist:def456'}]},
        }

        result = mm.search_spotify("test query")

        assert not result.startswith("Error")
        assert "Song A - Artist 1 ID: " in result
        assert "Artist 1 - N/A ID: " in result
        assert "popularity" not in result
        assert set(mm._spotify_search_cache.values()) == {'spotify:track:abc123', 'spotify:artist:def456'}

    def test_get_playlists_returns_copy(self):
        mm = make_music_manager()
        playlists = mm.get_playlists()
        assert playlists == ["Chill Vibes", "Workout Mix", "Jazz Evening"]
        playlists.append("should not affect original")
        assert len(mm._playlists_cache) == 3

    def test_get_favorites_returns_copy(self):
        mm = make_music_manager()
        favorites = mm.get_favorites()
        assert favorites == ["Liked Songs", "Daily Mix 1", "Discover Weekly"]
        favorites.append("should not affect original")
        assert len(mm._favorites_cache) == 3

    def test_room_mapping_unset_maps_nothing(self):
        mm = make_music_manager(room_mapping=None)
        assert mm.room_mapping == {}
        assert mm.normalize_name("Kitchen") == "Kitchen"


class TestCheckHealth:
    def test_healthy_when_zones_and_spotify_ok(self):
        mm = make_music_manager()
        mock_zone = Mock()
        mm._sonos_zones_cache = {"Living Room": mock_zone}
        mm._sonos_zones_cache_time = time.time()
        mm._spotify_client = Mock()
        assert mm.check_health() is True

    def test_unhealthy_when_no_zones(self):
        mm = make_music_manager()
        mm._discover_sonos_zones = Mock(return_value={})
        assert mm.check_health() is False

    def test_unhealthy_when_spotify_fails(self):
        mm = make_music_manager()
        mock_zone = Mock()
        mm._sonos_zones_cache = {"Living Room": mock_zone}
        mm._sonos_zones_cache_time = time.time()
        mm._spotify_client = None
        mm.config.get.side_effect = ValueError("no creds")
        assert mm.check_health() is False

    def test_a_revoked_spotify_sign_in_fails_the_check_naming_the_cure(self):
        mm = make_music_manager()
        mm._sonos_zones_cache = {"Living Room": Mock()}
        mm._sonos_zones_cache_time = time.time()
        mm._spotify_client = Mock()
        mm._spotify_auth.needs_sign_in_again.return_value = True
        with pytest.raises(RuntimeError, match="spotify_login"):
            mm.check_health()


class TestSonosZonesCacheExpiry:
    @patch('integrations.music.MusicIntegration.soco.discover')
    def test_cache_hit_within_expiry(self, mock_discover):
        mm = make_music_manager()
        mock_zone = Mock()
        mm._sonos_zones_cache = {"Living Room": mock_zone}
        mm._sonos_zones_cache_time = time.time()

        result = mm._discover_sonos_zones()
        assert result == {"Living Room": mock_zone}
        mock_discover.assert_not_called()

    @patch('integrations.music.MusicIntegration.soco.discover')
    def test_cache_expired_triggers_rediscovery(self, mock_discover):
        mm = make_music_manager()
        old_zone = Mock()
        mm._sonos_zones_cache = {"Old Room": old_zone}
        mm._sonos_zones_cache_time = time.time() - MusicIntegration.CACHE_EXPIRY_SECONDS - 1

        new_zone = Mock()
        new_zone.player_name = "New Room"
        mock_discover.return_value = [new_zone]

        result = mm._discover_sonos_zones()
        assert "New Room" in result
        assert "Old Room" not in result
        mock_discover.assert_called_once()

    @patch('integrations.music.MusicIntegration.soco.discover')
    def test_empty_cache_triggers_discovery(self, mock_discover):
        mm = make_music_manager()
        zone = Mock()
        zone.player_name = "Bedroom"
        mock_discover.return_value = [zone]

        result = mm._discover_sonos_zones()
        assert "Bedroom" in result
        mock_discover.assert_called_once()


class TestPlayInZoneRouting:
    def test_short_id_resolves_from_cache(self):
        mm = make_music_manager()
        mm._spotify_search_cache = {"AA": "spotify:artist:abc123"}
        mock_zone = Mock()
        mock_coordinator = Mock()
        mm._get_zone = Mock(return_value=mock_zone)
        mm._get_coordinator = Mock(return_value=mock_coordinator)

        result = mm.play_in_zone("AA", "Living Room")
        assert result is True
        mock_coordinator.play_uri.assert_called_once()

    def test_unknown_short_id_returns_false(self):
        mm = make_music_manager()
        mm._spotify_search_cache = {}
        mock_zone = Mock()
        mm._get_zone = Mock(return_value=mock_zone)
        mm._get_coordinator = Mock(return_value=Mock())

        result = mm.play_in_zone("ZZ", "Living Room")
        assert result is False

    def test_artist_uri_plays_radio(self):
        mm = make_music_manager()
        mock_zone = Mock()
        mock_coordinator = Mock()
        mm._get_zone = Mock(return_value=mock_zone)
        mm._get_coordinator = Mock(return_value=mock_coordinator)

        result = mm.play_in_zone("spotify:artist:abc123", "Living Room")
        assert result is True
        played_uri = mock_coordinator.play_uri.call_args[0][0]
        assert "artistRadio" in played_uri
        assert "abc123" in played_uri

    def _track_radio_manager(self, top_tracks):
        from spotipy.exceptions import SpotifyException
        mm = make_music_manager()
        mm._spotify_search_cache = {"AA": "spotify:track:seed01"}
        coordinator = Mock()
        mm._get_zone = Mock(return_value=Mock())
        mm._get_coordinator = Mock(return_value=coordinator)
        spotify = Mock()
        spotify.track.return_value = {'artists': [{'id': 'art01', 'name': 'Artist 1'}]}
        if isinstance(top_tracks, int):
            spotify.artist_top_tracks.side_effect = SpotifyException(top_tracks, -1, "forbidden")
        else:
            spotify.artist_top_tracks.return_value = {'tracks': [{'id': t} for t in top_tracks]}
        mm._get_spotify_client = Mock(return_value=spotify)
        return mm, coordinator

    def queued_track_ids(self, coordinator):
        return [c.args[0].split("spotify:track:")[1].split("?")[0] for c in coordinator.add_uri_to_queue.call_args_list]

    def test_track_radio_queues_seed_then_top_tracks(self):
        mm, coordinator = self._track_radio_manager(["seed01", "top02", "top03"])

        assert mm.play_in_zone("AA", "Living Room") is True
        assert self.queued_track_ids(coordinator) == ["seed01", "top02", "top03"]
        coordinator.play_from_queue.assert_called_once_with(0)

    def test_track_radio_without_top_tracks_endpoint_plays_seed_from_queue(self):
        """Spotify removed artist top tracks for Development Mode apps (February 2026): a 403 is seed-only, not an error."""
        mm, coordinator = self._track_radio_manager(403)

        assert mm.play_in_zone("AA", "Living Room") is True
        assert self.queued_track_ids(coordinator) == ["seed01"]
        coordinator.play_from_queue.assert_called_once_with(0)
        coordinator.play_uri.assert_not_called()

    def test_track_radio_other_spotify_error_falls_back_to_single_track(self):
        mm, coordinator = self._track_radio_manager(500)

        assert mm.play_in_zone("AA", "Living Room") is True
        coordinator.play_from_queue.assert_not_called()
        assert "seed01" in coordinator.play_uri.call_args[0][0]

    def test_playlist_uri_delegates_to_play_playlist(self):
        mm = make_music_manager()
        mock_zone = Mock()
        mm._get_zone = Mock(return_value=mock_zone)
        mm._get_coordinator = Mock(return_value=Mock())
        mm.play_playlist = Mock(return_value=True)

        result = mm.play_in_zone("P-0", "Living Room")
        assert result is True
        mm.play_playlist.assert_called_once_with("Living Room", "Chill Vibes")

    def test_favorite_uri_delegates_to_play_favorite(self):
        mm = make_music_manager()
        mock_zone = Mock()
        mm._get_zone = Mock(return_value=mock_zone)
        mm._get_coordinator = Mock(return_value=Mock())
        mm.play_favorite = Mock(return_value=True)

        result = mm.play_in_zone("F-1", "Living Room")
        assert result is True
        mm.play_favorite.assert_called_once_with("Living Room", "Daily Mix 1")

    def test_unknown_spotify_uri_type_returns_false(self):
        mm = make_music_manager()
        mock_zone = Mock()
        mm._get_zone = Mock(return_value=mock_zone)
        mm._get_coordinator = Mock(return_value=Mock())

        result = mm.play_in_zone("spotify:show:abc123", "Living Room")
        assert result is False

    def test_zone_not_found_returns_false(self):
        mm = make_music_manager()
        mm._get_zone = Mock(return_value=None)

        result = mm.play_in_zone("spotify:track:abc", "Nonexistent")
        assert result is False


class TestGetAllZonesStatus:
    def test_status_queries_coordinator_transport_info(self):
        """Grouped speakers should report the coordinator's transport state, not their own."""
        mm = make_music_manager()

        zone = Mock()
        coordinator = Mock()

        zone.get_current_transport_info.return_value = {'current_transport_state': 'PLAYING'}
        coordinator.get_current_transport_info.return_value = {'current_transport_state': 'PAUSED_PLAYBACK'}
        coordinator.get_current_track_info.return_value = {'title': 'Test Song', 'artist': 'Test Artist'}

        mm._discover_sonos_zones = Mock(return_value={"Kitchen Front": zone})
        mm._get_coordinator = Mock(return_value=coordinator)

        result = mm.get_all_zones_status()

        mm._get_coordinator.assert_called_once_with(zone)
        zone.get_current_transport_info.assert_not_called()
        assert "Paused" in result
        assert "Test Song" in result

    def test_status_shows_playing_from_coordinator(self):
        mm = make_music_manager()

        zone = Mock()
        coordinator = Mock()

        coordinator.get_current_transport_info.return_value = {'current_transport_state': 'PLAYING'}
        coordinator.get_current_track_info.return_value = {'title': 'Now Playing', 'artist': 'Artist'}

        mm._discover_sonos_zones = Mock(return_value={"Kitchen Front": zone})
        mm._get_coordinator = Mock(return_value=coordinator)

        result = mm.get_all_zones_status()
        assert "Playing" in result
        assert "Kitchen" in result

    def test_status_skips_empty_mapped_rooms(self):
        mm = make_music_manager()

        zone = Mock()
        mm._discover_sonos_zones = Mock(return_value={"Kitchen Back": zone})

        result = mm.get_all_zones_status()
        assert result == ""

    def test_zone_error_propagates_instead_of_replacing_status(self):
        """Regression: a failing zone must raise so PromptSectionGenerator's
        retry/keep-stale logic can keep the last good zone list. The old code
        returned the error string (wiping the whole MusicZones prompt section)
        and passed the exception object to error(), which crashed the logger
        with AttributeError ('ConnectionError' object has no attribute 'split')
        — this test uses the real error(), so either regression fails it."""
        mm = make_music_manager()

        zone = Mock()
        mm._discover_sonos_zones = Mock(return_value={"Kitchen Front": zone})
        mm._get_coordinator = Mock(side_effect=ConnectionError("network blip"))

        with pytest.raises(ConnectionError):
            mm.get_all_zones_status()


class TestRefreshCycle:
    def test_a_tick_refreshes_and_schedules_the_next_one(self):
        mm = make_music_manager()
        mm._refresh_playlists_and_favorites = Mock()
        try:
            mm._refresh_tick()
            mm._refresh_playlists_and_favorites.assert_called_once()
            assert mm._refresh_timer is not None and mm._refresh_timer.is_alive()
        finally:
            mm._on_reset()

    def test_a_dropped_integration_stops_refreshing(self):
        mm = make_music_manager()
        mm._refresh_playlists_and_favorites = Mock()
        mm._schedule_refresh(3600)
        pending = mm._refresh_timer

        mm._on_reset()
        pending.join(timeout=1)
        mm._refresh_tick()   # a tick that was already running when the integration was dropped

        assert not pending.is_alive()
        mm._refresh_playlists_and_favorites.assert_not_called()
        assert mm._refresh_timer is pending, "no new timer may be scheduled after the drop"


class TestZoneStateParsingIsSerialised:
    """soco's zone-group XSLT is shared by every thread and not thread-safe: two threads in it
    at once abort the process. The integration module puts a lock on soco's entry point."""

    def test_soco_parses_one_zone_group_payload_at_a_time(self):
        from soco.zonegroupstate import ZoneGroupState
        from integrations.music import MusicIntegration as module

        done = threading.Event()

        def parse():
            try:
                ZoneGroupState().process_payload(payload="<not-a-zone-group-state/>", source="test",
                                                 source_ip="192.0.2.10")
            except Exception:
                pass   # the payload is junk; only when soco gets to it matters
            done.set()

        with module._zone_state_lock:
            thread = threading.Thread(target=parse, daemon=True)
            thread.start()
            assert not done.wait(0.05), "soco parsed a payload while another thread held the parsing lock"
        assert done.wait(2), "soco never got to the payload after the lock was released"


class TestSpotifyUserPlaylists:
    """The signed-in user's playlist library: S- ids, the fetch, and playback by URI through the Sonos Spotify service."""

    def test_s_ids_resolve_to_the_cached_playlists_in_order(self):
        mm = make_music_manager()
        assert mm._uri_to_spotify_playlist("S-0").uri == "spotify:playlist:dw1"
        assert mm._uri_to_spotify_playlist("S-1").name == "Road Trip"
        assert mm._uri_to_spotify_playlist("S-2") is None
        assert mm._uri_to_spotify_playlist("P-0") is None
        assert mm._uri_to_spotify_playlist("S-x") is None

    def test_get_spotify_playlists_returns_a_copy(self):
        mm = make_music_manager()
        playlists = mm.get_spotify_playlists()
        playlists.clear()
        assert len(mm.get_spotify_playlists()) == 2

    @patch('integrations.music.MusicIntegration.ShareLinkPlugin')
    def test_an_s_id_is_started_by_its_spotify_uri_through_the_sonos_spotify_service(self, share_cls):
        mm = make_music_manager()
        coordinator = Mock()
        mm._get_zone = Mock(return_value=Mock())
        mm._get_coordinator = Mock(return_value=coordinator)

        assert mm.play_in_zone("S-0", "Kitchen") is True

        share_cls.assert_called_once_with(coordinator)
        share_cls.return_value.add_share_link_to_queue.assert_called_once_with("spotify:playlist:dw1")
        coordinator.clear_queue.assert_called_once()
        coordinator.play_from_queue.assert_called_once_with(0)
        assert coordinator.play_mode == 'SHUFFLE'

    @patch('integrations.music.MusicIntegration.ShareLinkPlugin')
    def test_a_playlist_found_by_search_plays_the_same_way(self, share_cls):
        mm = make_music_manager()
        coordinator = Mock()
        mm._get_zone = Mock(return_value=Mock())
        mm._get_coordinator = Mock(return_value=coordinator)
        mm._spotify_search_cache["AB"] = "spotify:playlist:pub9"

        assert mm.play_in_zone("AB", "Kitchen") is True

        share_cls.return_value.add_share_link_to_queue.assert_called_once_with("spotify:playlist:pub9")
        coordinator.play_from_queue.assert_called_once_with(0)

    @patch('integrations.music.MusicIntegration.ShareLinkPlugin')
    def test_a_sonos_refusal_is_reported_as_failure(self, share_cls):
        mm = make_music_manager()
        mm._get_zone = Mock(return_value=Mock())
        mm._get_coordinator = Mock(return_value=Mock())
        share_cls.return_value.add_share_link_to_queue.side_effect = Exception("Unsupported URI")

        assert mm.play_in_zone("S-1", "Kitchen") is False

    def test_an_s_id_out_of_range_does_not_play(self):
        mm = make_music_manager()
        mm._get_zone = Mock(return_value=Mock())
        mm._get_coordinator = Mock(return_value=Mock())
        mm.play_spotify_playlist = Mock()

        assert mm.play_in_zone("S-7", "Kitchen") is False
        mm.play_spotify_playlist.assert_not_called()

    @patch('integrations.music.MusicIntegration.ShareLinkPlugin')
    def test_a_spotify_favorite_still_goes_through_the_share_link(self, share_cls):
        mm = make_music_manager()
        coordinator = Mock()
        mm._get_zone = Mock(return_value=Mock())
        mm._get_coordinator = Mock(return_value=coordinator)
        favorite = Mock()
        favorite.title = "Daily Mix 1"
        favorite.get_uri.return_value = "x-rincon-cpcontainer:1006206cspotify%3aplaylist%3a37i9dQZF1E3?sid=9"
        coordinator.music_library.get_sonos_favorites.return_value = [favorite]

        assert mm.play_favorite("Kitchen", "Daily Mix 1") is True

        share_cls.return_value.add_share_link_to_queue.assert_called_once_with("https://open.spotify.com/playlist/37i9dQZF1E3")
        coordinator.play_from_queue.assert_called_once_with(0)

    def test_the_fetch_takes_the_first_page_of_the_library_in_spotifys_order_skips_hidden_items_and_keeps_the_size(self):
        mm = make_music_manager()
        client = Mock()
        mm._get_user_client = Mock(return_value=client)
        client.current_user_playlists.return_value = {
            'items': [{'name': 'Road Trip', 'uri': 'spotify:playlist:rt2', 'owner': {}},
                      None,
                      {'name': 'Shared Dinner', 'uri': 'spotify:playlist:sd3', 'owner': {'display_name': 'Alice'}}],
            'next': 'https://api.spotify.com/v1/me/playlists?offset=20', 'total': 57}

        playlists, total = mm._fetch_spotify_playlists()

        assert [(p.name, p.uri, p.owner) for p in playlists] == [
            ("Road Trip", "spotify:playlist:rt2", ""),
            ("Shared Dinner", "spotify:playlist:sd3", "Alice")]
        assert total == 57
        client.current_user_playlists.assert_called_once_with(limit=MusicManager.SPOTIFY_PLAYLISTS_IN_PROMPT)
        client.next.assert_not_called()
        assert mm.spotify_library_error() == ""

    def test_without_a_signed_in_user_the_library_is_empty_and_spotify_is_not_asked(self):
        mm = make_music_manager()
        mm._spotify_auth.signed_in.return_value = False
        mm._get_user_client = Mock()

        mm._refresh_spotify_playlists()

        assert mm.get_spotify_playlists() == []
        mm._get_user_client.assert_not_called()

    def test_a_transient_failure_keeps_the_library_and_says_why(self):
        mm = make_music_manager()
        for failure in (TokenUnavailable("token endpoint could not be reached"), Exception("timed out")):
            mm._get_user_client = Mock(side_effect=failure)
            assert mm._refresh_spotify_playlists() is False
            assert len(mm.get_spotify_playlists()) == 2 and mm.spotify_playlists_total() == 2
            assert str(failure) in mm.spotify_library_error()

    @patch("integrations.music.MusicIntegration.system_note")
    def test_a_sign_in_the_user_must_renew_empties_the_library_and_is_noted_once(self, note):
        mm = make_music_manager()
        mm._get_user_client = Mock(side_effect=NotSignedIn("sign in again"))
        mm._spotify_auth.needs_sign_in_again.return_value = True

        assert mm._refresh_spotify_playlists() is True
        mm._refresh_spotify_playlists()

        assert mm.get_spotify_playlists() == [] and mm.spotify_playlists_total() == 0
        assert note.call_count == 1 and "spotify_login" in note.call_args.args[1]

    @patch("integrations.music.MusicIntegration.system_note")
    def test_a_token_that_was_never_there_is_not_a_revoked_one_and_gets_no_note(self, note):
        mm = make_music_manager()
        mm._get_user_client = Mock(side_effect=NotSignedIn("not signed in"))
        mm._refresh_spotify_playlists()
        note.assert_not_called()

    def test_a_refresh_that_raises_still_schedules_the_next_tick_and_the_sonos_lists(self):
        mm = make_music_manager()
        mm._spotify_auth.signed_in.side_effect = AttributeError("'str' object has no attribute 'get'")
        mm._discover_sonos_zones = Mock(return_value={"Kitchen": Mock()})
        mm._fetch_playlists_from_sonos = Mock(return_value=["Fresh"])
        mm._fetch_favorites_from_sonos = Mock(return_value=[])
        with patch.object(mm, "_schedule_refresh") as schedule:
            mm._refresh_tick()
        schedule.assert_called_once_with(mm.REFRESH_INTERVAL_SECONDS)
        assert mm.get_playlists() == ["Fresh"]

    def test_the_user_client_follows_the_access_token(self):
        mm = make_music_manager()
        mm._spotify_auth.token.return_value = "tok-1"
        first = mm._get_user_client()
        assert mm._get_user_client() is first
        mm._spotify_auth.token.return_value = "tok-2"
        assert mm._get_user_client() is not first

    def test_a_finished_sign_in_reloads_the_library_at_once(self):
        mm = make_music_manager()
        mm._spotify_auth.finish.return_value = ""
        mm._user_client = Mock()
        mm._fetch_spotify_playlists = Mock(return_value=([SpotifyPlaylist("New", "spotify:playlist:n1")], 1))

        assert mm.login_finish("http://127.0.0.1:8888/callback?code=AQ&state=st") == ""

        assert [p.name for p in mm.get_spotify_playlists()] == ["New"]
        assert mm.spotify_playlists_total() == 1
        assert mm._user_client is None

    def test_a_reloaded_library_wakes_the_listeners_but_a_failed_reload_does_not(self):
        mm = make_music_manager()
        mm._spotify_auth.finish.return_value = ""
        woken, broken = [], Mock(side_effect=RuntimeError("listener broke"))
        listener = lambda: woken.append(len(mm.get_spotify_playlists()))
        MusicManager.add_library_listener(broken)
        MusicManager.add_library_listener(listener)
        MusicManager.add_library_listener(listener)   # registered once however often it is added
        try:
            mm._fetch_spotify_playlists = Mock(return_value=([SpotifyPlaylist("New", "spotify:playlist:n1")], 1))
            assert mm.login_finish("http://127.0.0.1:8888/callback?code=AQ&state=st") == ""
            assert woken == [1], "woken after the cache holds the new list, and once, despite the listener before it raising"

            mm._fetch_spotify_playlists = Mock(return_value=None)
            mm._refresh_spotify_playlists()
            assert woken == [1]

            MusicManager.remove_library_listener(listener)
            mm._fetch_spotify_playlists = Mock(return_value=([], 0))
            mm._refresh_spotify_playlists()
            assert woken == [1]
        finally:
            MusicManager.remove_library_listener(listener)
            MusicManager.remove_library_listener(broken)
        assert MusicManager._library_listeners == []

    def test_a_failed_sign_in_comes_back_as_the_reason_and_touches_nothing(self):
        mm = make_music_manager()
        mm._spotify_auth.finish.return_value = "the code expired"
        mm._fetch_spotify_playlists = Mock()

        assert mm.login_finish("AQold") == "the code expired"
        mm._fetch_spotify_playlists.assert_not_called()
        assert len(mm.get_spotify_playlists()) == 2

    def test_search_lists_public_playlists_with_their_owner_and_skips_null_items(self):
        mm = make_music_manager()
        mock_spotify = Mock()
        mm._get_spotify_client = Mock(return_value=mock_spotify)
        mock_spotify.search.return_value = {
            'tracks': {'items': []},
            'artists': {'items': []},
            'playlists': {'items': [None, {'name': 'Summer Hits', 'uri': 'spotify:playlist:sh1', 'owner': {'display_name': 'Alice'}}]},
        }

        result = mm.search_spotify("summer")

        assert "PLAYLISTS:" in result
        assert "Summer Hits - by Alice ID: AA" in result
        assert mm._spotify_search_cache["AA"] == "spotify:playlist:sh1"
        assert mock_spotify.search.call_args.kwargs['type'] == 'track,artist,playlist'
