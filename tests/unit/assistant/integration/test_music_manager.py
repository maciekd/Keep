import time

import pytest
from unittest.mock import Mock, patch
import threading

from assistant.integration.integrations.MusicIntegration import MusicManager, MusicIntegration


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


class TestSonosZonesCacheExpiry:
    @patch('assistant.integration.integrations.MusicIntegration.soco.discover')
    def test_cache_hit_within_expiry(self, mock_discover):
        mm = make_music_manager()
        mock_zone = Mock()
        mm._sonos_zones_cache = {"Living Room": mock_zone}
        mm._sonos_zones_cache_time = time.time()

        result = mm._discover_sonos_zones()
        assert result == {"Living Room": mock_zone}
        mock_discover.assert_not_called()

    @patch('assistant.integration.integrations.MusicIntegration.soco.discover')
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

    @patch('assistant.integration.integrations.MusicIntegration.soco.discover')
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
        from assistant.integration.integrations import MusicIntegration as module

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
