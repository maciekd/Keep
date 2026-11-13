import time

import pytest
from unittest.mock import PropertyMock, Mock, patch
import threading

from assistant.integration.music.SpotifyAccount import SIGN_IN_REVOKED, SpotifyAccount
from integrations.sonos.SonosIntegration import SonosIntegration
from tests.unit.assistant.integration.music.spotify_fakes import install, make_account


ROOM_MAPPING = {"Kitchen Front": "Kitchen", "Kitchen Back": ""}


@pytest.fixture(autouse=True)
def spotify():
    """The shared Spotify account the integration reaches through `self.spotify`: a fake with two playlists."""
    account = install(make_account())
    yield account
    SpotifyAccount.reset_instance()


def make_sonos(room_mapping=ROOM_MAPPING):
    """Create a SonosIntegration bypassing __init__ to avoid network calls."""
    mm = SonosIntegration.__new__(SonosIntegration)
    mm.config = Mock()
    mm.config.get.side_effect = lambda key, default=None: room_mapping if key == 'sonos.room_mapping' else default
    mm._sonos_zones_cache = {}
    mm._sonos_zones_cache_time = None
    mm._playlists_cache = ["Chill Vibes", "Workout Mix", "Jazz Evening"]
    mm._favorites_cache = ["Liked Songs", "Daily Mix 1", "Discover Weekly"]
    mm._cache_lock = threading.Lock()
    mm._refresh_timer = None
    mm._stopped = threading.Event()
    return mm


class TestSonosIntegration:
    def test_playlist_uri_resolution(self):
        mm = make_sonos()
        assert mm._uri_to_playlist_name("P-0") == "Chill Vibes"
        assert mm._uri_to_playlist_name("P-2") == "Jazz Evening"

    def test_favorite_uri_resolution(self):
        mm = make_sonos()
        assert mm._uri_to_favorite_name("F-0") == "Liked Songs"
        assert mm._uri_to_favorite_name("F-1") == "Daily Mix 1"

    def test_playlist_uri_out_of_range(self):
        mm = make_sonos()
        assert mm._uri_to_playlist_name("P-99") is None

    def test_favorite_uri_out_of_range(self):
        mm = make_sonos()
        assert mm._uri_to_favorite_name("F-99") is None

    def test_playlist_uri_invalid_prefix(self):
        mm = make_sonos()
        assert mm._uri_to_playlist_name("X-0") is None

    def test_favorite_uri_invalid_prefix(self):
        mm = make_sonos()
        assert mm._uri_to_favorite_name("X-0") is None

    def test_normalize_name_with_room_mapping(self):
        mm = make_sonos()
        assert mm.normalize_name("Kitchen") == "Kitchen Front"

    def test_normalize_name_without_mapping(self):
        mm = make_sonos()
        assert mm.normalize_name("Living Room") == "Living Room"

    def test_get_playlists_returns_copy(self):
        mm = make_sonos()
        playlists = mm.get_playlists()
        assert playlists == ["Chill Vibes", "Workout Mix", "Jazz Evening"]
        playlists.append("should not affect original")
        assert len(mm._playlists_cache) == 3

    def test_get_favorites_returns_copy(self):
        mm = make_sonos()
        favorites = mm.get_favorites()
        assert favorites == ["Liked Songs", "Daily Mix 1", "Discover Weekly"]
        favorites.append("should not affect original")
        assert len(mm._favorites_cache) == 3

    def test_room_mapping_unset_maps_nothing(self):
        mm = make_sonos(room_mapping=None)
        assert mm.room_mapping == {}
        assert mm.normalize_name("Kitchen") == "Kitchen"


class TestCheckHealth:
    def test_healthy_when_zones_and_spotify_ok(self, spotify):
        mm = make_sonos()
        mock_zone = Mock()
        mm._sonos_zones_cache = {"Living Room": mock_zone}
        mm._sonos_zones_cache_time = time.time()
        spotify._app_client = Mock()
        assert mm.check_health() is True

    def test_unhealthy_when_no_zones(self):
        mm = make_sonos()
        mm._discover_sonos_zones = Mock(return_value={})
        assert mm.check_health() is False

    def test_unhealthy_without_a_spotify_client(self, spotify):
        mm = make_sonos()
        mock_zone = Mock()
        mm._sonos_zones_cache = {"Living Room": mock_zone}
        mm._sonos_zones_cache_time = time.time()
        assert mm.check_health() is False

    def test_a_revoked_spotify_sign_in_fails_the_check_naming_the_cure(self, spotify):
        mm = make_sonos()
        mm._sonos_zones_cache = {"Living Room": Mock()}
        mm._sonos_zones_cache_time = time.time()
        spotify._app_client = Mock()
        spotify._auth.needs_sign_in_again.return_value = True
        with pytest.raises(RuntimeError, match="spotify_login") as raised:
            mm.check_health()
        assert str(raised.value) == SIGN_IN_REVOKED


class TestSonosZonesCacheExpiry:
    @patch('integrations.sonos.SonosIntegration.soco.discover')
    def test_cache_hit_within_expiry(self, mock_discover):
        mm = make_sonos()
        mock_zone = Mock()
        mm._sonos_zones_cache = {"Living Room": mock_zone}
        mm._sonos_zones_cache_time = time.time()

        result = mm._discover_sonos_zones()
        assert result == {"Living Room": mock_zone}
        mock_discover.assert_not_called()

    @patch('integrations.sonos.SonosIntegration.soco.discover')
    def test_cache_expired_triggers_rediscovery(self, mock_discover):
        mm = make_sonos()
        old_zone = Mock()
        mm._sonos_zones_cache = {"Old Room": old_zone}
        mm._sonos_zones_cache_time = time.time() - SonosIntegration.CACHE_EXPIRY_SECONDS - 1

        new_zone = Mock()
        new_zone.player_name = "New Room"
        mock_discover.return_value = [new_zone]

        result = mm._discover_sonos_zones()
        assert "New Room" in result
        assert "Old Room" not in result
        mock_discover.assert_called_once()

    @patch('integrations.sonos.SonosIntegration.soco.discover')
    def test_empty_cache_triggers_discovery(self, mock_discover):
        mm = make_sonos()
        zone = Mock()
        zone.player_name = "Bedroom"
        mock_discover.return_value = [zone]

        result = mm._discover_sonos_zones()
        assert "Bedroom" in result
        mock_discover.assert_called_once()


    @patch('integrations.sonos.SonosIntegration.soco.discover')
    def test_an_offline_zone_is_skipped_and_the_reachable_ones_are_cached(self, mock_discover):
        """Reading a zone's name is an HTTP GET to the speaker: one offline speaker must not
        leave the cache empty and every poll rediscovering."""
        import requests
        mm = make_sonos()
        kitchen, offline = Mock(), Mock()
        kitchen.player_name = "Kitchen"
        type(offline).player_name = PropertyMock(side_effect=requests.ConnectionError("HTTPConnectionPool(host='192.0.2.7', port=1400)"))
        offline.ip_address = "192.0.2.7"
        mock_discover.return_value = [offline, kitchen]

        assert mm._discover_sonos_zones() == {"Kitchen": kitchen}
        assert mm._sonos_zones_cache_time is not None
        assert mm._discover_sonos_zones() == {"Kitchen": kitchen}
        mock_discover.assert_called_once()

    @patch('integrations.sonos.SonosIntegration.soco.discover')
    def test_a_discovery_that_reaches_no_zone_keeps_the_old_cache(self, mock_discover):
        mm = make_sonos()
        old_zone = Mock()
        mm._sonos_zones_cache = {"Bedroom": old_zone}
        mm._sonos_zones_cache_time = time.time() - SonosIntegration.CACHE_EXPIRY_SECONDS - 1
        mock_discover.return_value = None
        assert mm._discover_sonos_zones() == {"Bedroom": old_zone}



class TestZoneProblem:
    def _with_zones(self, zones):
        mm = make_sonos(room_mapping={})
        mm._sonos_zones_cache = zones
        mm._sonos_zones_cache_time = time.time()
        return mm

    def test_a_known_zone_has_no_problem_whatever_its_case(self):
        assert self._with_zones({"Living Room": Mock()}).zone_problem("living room") == ""

    def test_an_unknown_zone_names_the_zones_there_are(self):
        problem = self._with_zones({"Living Room": Mock(), "Bedroom": Mock()}).zone_problem("Garage")
        assert "Garage" in problem and "Bedroom" in problem and "Living Room" in problem

    @patch("integrations.sonos.SonosIntegration.soco.discover", return_value=None)
    def test_with_no_speaker_on_the_network_the_problem_says_so(self, _discover):
        problem = make_sonos(room_mapping={}).zone_problem("Garage")
        assert "Garage" in problem and "no Sonos speaker" in problem


class TestPlayInZoneRouting:
    def test_short_id_resolves_from_cache(self, spotify):
        mm = make_sonos()
        spotify._search_ids = {"AA": "spotify:artist:abc123"}
        mock_zone = Mock()
        mock_coordinator = Mock()
        mm._get_zone = Mock(return_value=mock_zone)
        mm._get_coordinator = Mock(return_value=mock_coordinator)

        result = mm.play_in_zone("AA", "Living Room")
        assert result is True
        mock_coordinator.play_uri.assert_called_once()

    def test_unknown_short_id_returns_false(self):
        mm = make_sonos()
        mock_zone = Mock()
        mm._get_zone = Mock(return_value=mock_zone)
        mm._get_coordinator = Mock(return_value=Mock())

        result = mm.play_in_zone("ZZ", "Living Room")
        assert result is False

    def test_artist_uri_plays_radio(self):
        mm = make_sonos()
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
        mm = make_sonos()
        account = SpotifyAccount.get_instance()
        account._search_ids = {"AA": "spotify:track:seed01"}
        coordinator = Mock()
        mm._get_zone = Mock(return_value=Mock())
        mm._get_coordinator = Mock(return_value=coordinator)
        spotify = Mock()
        spotify.track.return_value = {'artists': [{'id': 'art01', 'name': 'Artist 1'}]}
        if isinstance(top_tracks, int):
            spotify.artist_top_tracks.side_effect = SpotifyException(top_tracks, -1, "forbidden")
        else:
            spotify.artist_top_tracks.return_value = {'tracks': [{'id': t} for t in top_tracks]}
        account._app_client = spotify
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
        mm = make_sonos()
        mock_zone = Mock()
        mm._get_zone = Mock(return_value=mock_zone)
        mm._get_coordinator = Mock(return_value=Mock())
        mm.play_playlist = Mock(return_value=True)

        result = mm.play_in_zone("P-0", "Living Room")
        assert result is True
        mm.play_playlist.assert_called_once_with("Living Room", "Chill Vibes")

    def test_favorite_uri_delegates_to_play_favorite(self):
        mm = make_sonos()
        mock_zone = Mock()
        mm._get_zone = Mock(return_value=mock_zone)
        mm._get_coordinator = Mock(return_value=Mock())
        mm.play_favorite = Mock(return_value=True)

        result = mm.play_in_zone("F-1", "Living Room")
        assert result is True
        mm.play_favorite.assert_called_once_with("Living Room", "Daily Mix 1")

    def test_unknown_spotify_uri_type_returns_false(self):
        mm = make_sonos()
        mock_zone = Mock()
        mm._get_zone = Mock(return_value=mock_zone)
        mm._get_coordinator = Mock(return_value=Mock())

        result = mm.play_in_zone("spotify:show:abc123", "Living Room")
        assert result is False

    def test_zone_not_found_returns_false(self):
        mm = make_sonos()
        mm._get_zone = Mock(return_value=None)

        result = mm.play_in_zone("spotify:track:abc", "Nonexistent")
        assert result is False


class TestGetAllZonesStatus:
    def test_status_queries_coordinator_transport_info(self):
        """Grouped speakers should report the coordinator's transport state, not their own."""
        mm = make_sonos()

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
        mm = make_sonos()

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
        mm = make_sonos()

        zone = Mock()
        mm._discover_sonos_zones = Mock(return_value={"Kitchen Back": zone})

        result = mm.get_all_zones_status()
        assert result == ""

    def test_a_failing_zone_is_reported_unreachable_instead_of_replacing_the_status(self):
        """Regression: a failing zone once raised out of the whole status — the section then
        failed every poll for the outage's length — and before that its error string replaced
        the section. It is one unreachable line now. The exception once reached error() as an
        object and crashed the logger ('ConnectionError' object has no attribute 'split'), so
        this test uses the real error()."""
        mm = make_sonos()

        zone = Mock()
        mm._discover_sonos_zones = Mock(return_value={"Kitchen Front": zone})
        mm._get_coordinator = Mock(side_effect=ConnectionError("network blip"))

        assert mm.get_all_zones_status() == "Kitchen (unreachable)"


class TestRefreshCycle:
    def test_a_tick_refreshes_and_schedules_the_next_one(self):
        mm = make_sonos()
        mm._refresh_playlists_and_favorites = Mock()
        try:
            mm._refresh_tick()
            mm._refresh_playlists_and_favorites.assert_called_once()
            assert mm._refresh_timer is not None and mm._refresh_timer.is_alive()
        finally:
            mm._on_reset()

    def test_a_dropped_integration_stops_refreshing(self):
        mm = make_sonos()
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
        from integrations.sonos import SonosIntegration as module

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

    @patch('integrations.sonos.SonosIntegration.ShareLinkPlugin')
    def test_an_s_id_is_started_by_its_spotify_uri_through_the_sonos_spotify_service(self, share_cls):
        mm = make_sonos()
        coordinator = Mock()
        mm._get_zone = Mock(return_value=Mock())
        mm._get_coordinator = Mock(return_value=coordinator)

        assert mm.play_in_zone("S-0", "Kitchen") is True

        share_cls.assert_called_once_with(coordinator)
        share_cls.return_value.add_share_link_to_queue.assert_called_once_with("spotify:playlist:dw1")
        coordinator.clear_queue.assert_called_once()
        coordinator.play_from_queue.assert_called_once_with(0)
        assert coordinator.play_mode == 'SHUFFLE'

    @patch('integrations.sonos.SonosIntegration.ShareLinkPlugin')
    def test_a_playlist_found_by_search_plays_the_same_way(self, share_cls, spotify):
        mm = make_sonos()
        coordinator = Mock()
        mm._get_zone = Mock(return_value=Mock())
        mm._get_coordinator = Mock(return_value=coordinator)
        spotify._search_ids["AB"] = "spotify:playlist:pub9"

        assert mm.play_in_zone("AB", "Kitchen") is True

        share_cls.return_value.add_share_link_to_queue.assert_called_once_with("spotify:playlist:pub9")
        coordinator.play_from_queue.assert_called_once_with(0)

    @patch('integrations.sonos.SonosIntegration.ShareLinkPlugin')
    def test_a_sonos_refusal_is_reported_as_failure(self, share_cls):
        mm = make_sonos()
        mm._get_zone = Mock(return_value=Mock())
        mm._get_coordinator = Mock(return_value=Mock())
        share_cls.return_value.add_share_link_to_queue.side_effect = Exception("Unsupported URI")

        assert mm.play_in_zone("S-1", "Kitchen") is False

    def test_an_s_id_out_of_range_does_not_play(self):
        mm = make_sonos()
        mm._get_zone = Mock(return_value=Mock())
        mm._get_coordinator = Mock(return_value=Mock())
        mm.play_spotify_playlist = Mock()

        assert mm.play_in_zone("S-7", "Kitchen") is False
        mm.play_spotify_playlist.assert_not_called()

    @patch('integrations.sonos.SonosIntegration.ShareLinkPlugin')
    def test_a_spotify_favorite_still_goes_through_the_share_link(self, share_cls):
        mm = make_sonos()
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

    def test_a_refresh_that_raises_still_schedules_the_next_tick_and_the_sonos_lists(self, spotify):
        mm = make_sonos()
        spotify._auth.signed_in.side_effect = AttributeError("'str' object has no attribute 'get'")
        mm._discover_sonos_zones = Mock(return_value={"Kitchen": Mock()})
        mm._fetch_playlists_from_sonos = Mock(return_value=["Fresh"])
        mm._fetch_favorites_from_sonos = Mock(return_value=[])
        with patch.object(mm, "_schedule_refresh") as schedule:
            mm._refresh_tick()
        schedule.assert_called_once_with(mm.REFRESH_INTERVAL_SECONDS)
        assert mm.get_playlists() == ["Fresh"]

    def test_the_sonos_lists_are_cut_to_the_prompt_cap_in_the_apps_order_so_the_ids_match(self, spotify):
        mm = make_sonos()
        spotify._auth.signed_in.return_value = False
        mm._discover_sonos_zones = Mock(return_value={"Kitchen": Mock()})
        cap = SonosIntegration.PLAYLISTS_IN_PROMPT
        mm._fetch_playlists_from_sonos = Mock(return_value=[f"Playlist {n}" for n in range(cap + 5)])
        mm._fetch_favorites_from_sonos = Mock(return_value=[f"Favorite {n}" for n in range(cap - 1)])

        mm._refresh_playlists_and_favorites()

        assert mm.get_playlists() == [f"Playlist {n}" for n in range(cap)]
        assert mm.get_favorites() == [f"Favorite {n}" for n in range(cap - 1)]
        assert mm._uri_to_playlist_name(f"P-{cap - 1}") == f"Playlist {cap - 1}"
        assert mm._uri_to_playlist_name(f"P-{cap}") is None

    def test_the_refresh_reloads_the_shared_spotify_library_too(self, spotify):
        mm = make_sonos()
        spotify.refresh_library = Mock(return_value=True)
        mm._discover_sonos_zones = Mock(return_value={})
        mm._refresh_playlists_and_favorites()
        spotify.refresh_library.assert_called_once_with()



class TestZonesStatusWithAnUnreachableZone:
    """One offline speaker (soco answers no group for a zone the topology no longer lists, or
    its HTTP is refused) must not cost the other zones' status every poll."""

    @staticmethod
    def _zone(state="PLAYING", title="Song", artist="Band", group=True):
        zone = Mock()
        coordinator = Mock()
        coordinator.get_current_transport_info.return_value = {'current_transport_state': state}
        coordinator.get_current_track_info.return_value = {'title': title, 'artist': artist}
        zone.group = Mock(coordinator=coordinator) if group else None
        return zone

    def _manager(self, zones):
        mm = make_sonos(room_mapping={})
        mm._sonos_zones_cache = dict(zones)
        mm._sonos_zones_cache_time = time.time()
        return mm

    def test_a_zone_without_a_group_is_listed_unreachable_and_the_rest_keep_their_status(self):
        mm = self._manager({"Kitchen": self._zone(group=False), "Office": self._zone(state="PAUSED_PLAYBACK")})
        with patch("integrations.sonos.SonosIntegration.error"):
            status = mm.get_all_zones_status()
        assert status.splitlines() == ["Kitchen (unreachable)", "Office (Song by Band - Paused)"]

    def test_a_zone_whose_http_fails_is_listed_unreachable(self):
        broken = self._zone()
        broken.group.coordinator.get_current_transport_info.side_effect = ConnectionError("Host is down")
        mm = self._manager({"Kitchen": broken, "Office": self._zone()})
        with patch("integrations.sonos.SonosIntegration.error"):
            status = mm.get_all_zones_status()
        assert status.splitlines() == ["Kitchen (unreachable)", "Office (Song by Band - Playing)"]

    def test_an_unreachable_zone_leaves_the_cache_for_the_next_discovery_to_re_find(self):
        mm = self._manager({"Kitchen": self._zone(group=False), "Office": self._zone()})
        with patch("integrations.sonos.SonosIntegration.error") as err:
            mm.get_all_zones_status()
            mm.get_all_zones_status()
        assert list(mm._sonos_zones_cache) == ["Office"]
        assert err.call_count == 1   # the outage is logged once, not on every 3-second poll
        assert mm._sonos_zones_cache_time is not None   # no full rediscovery per poll

    def test_a_command_on_a_zone_without_a_group_fails_with_a_connection_error(self):
        mm = make_sonos()
        with pytest.raises(ConnectionError):
            mm._get_coordinator(self._zone(group=False))


class TestAsAMusicSystem:
    """Sonos as one MusicSystem: its zones under the names the LLM uses, its own lists with F-/P-
    ids, and every action's outcome as "" or the line the LLM reads."""

    @staticmethod
    def _with_zones(zones, room_mapping=ROOM_MAPPING):
        mm = make_sonos(room_mapping=room_mapping)
        mm._sonos_zones_cache = dict(zones)
        mm._sonos_zones_cache_time = time.time()
        return mm

    @staticmethod
    def _zone(coordinator=None):
        zone = Mock()
        zone.group = Mock(coordinator=coordinator or Mock(), members=[Mock(volume=20)])
        return zone

    def test_the_zones_carry_the_mapped_names_and_a_blank_mapping_hides_one(self):
        mm = self._with_zones({"Kitchen Front": Mock(), "Kitchen Back": Mock(), "Bedroom": Mock()})
        assert sorted(mm.zones()) == ["Bedroom", "Kitchen"]

    def test_the_library_numbers_the_favorites_and_the_playlists_each_from_zero(self):
        lines = make_sonos().library_text().splitlines()
        assert lines[0].endswith("(ID: F-0)") and "Liked Songs" in lines[0]
        assert lines[2].endswith("(ID: F-2)") and "Discover Weekly" in lines[2]
        assert lines[3].endswith("(ID: P-0)") and "Chill Vibes" in lines[3]
        assert len(lines) == 6

    def test_its_own_lists_are_its_ids_and_spotifys_are_not(self):
        mm = make_sonos()
        assert mm.owns_id("F-1") and mm.owns_id("P-0")
        assert not mm.owns_id("S-0") and not mm.owns_id("AA")

    @pytest.mark.parametrize("verb, args", [("play", ()), ("pause", ()), ("next_track", ()), ("adjust_volume", (4,))])
    def test_a_done_action_answers_nothing(self, verb, args):
        mm = self._with_zones({"Bedroom": self._zone()}, room_mapping={})
        assert getattr(mm, verb)("Bedroom", *args) == ""

    @pytest.mark.parametrize("verb, args", [("play", ()), ("pause", ()), ("next_track", ()), ("adjust_volume", (4,))])
    def test_an_unknown_zone_answers_with_the_zones_there_are(self, verb, args):
        mm = self._with_zones({"Bedroom": self._zone()}, room_mapping={})
        problem = getattr(mm, verb)("Garage", *args)
        assert "Garage" in problem and "Bedroom" in problem

    @pytest.mark.parametrize("verb, sonos_call", [("play", "play"), ("pause", "pause"), ("next_track", "next")])
    def test_a_speaker_that_refuses_is_the_answer_not_a_raise(self, verb, sonos_call):
        coordinator = Mock()
        getattr(coordinator, sonos_call).side_effect = ConnectionError("speaker offline")
        mm = self._with_zones({"Bedroom": self._zone(coordinator)}, room_mapping={})
        problem = getattr(mm, verb)("Bedroom")
        assert "Bedroom" in problem and "speaker offline" in problem

    def test_a_zone_the_topology_no_longer_lists_is_the_answer_not_a_raise(self):
        offline = Mock()
        offline.group = None
        mm = self._with_zones({"Bedroom": offline}, room_mapping={})
        assert "Bedroom" in mm.pause("Bedroom")

    def test_a_volume_the_speaker_refuses_is_the_answer(self):
        zone = Mock()
        member = Mock()
        type(member).volume = PropertyMock(side_effect=ConnectionError("speaker offline"))
        zone.group = Mock(members=[member])
        mm = self._with_zones({"Bedroom": zone}, room_mapping={})
        assert "speaker offline" in mm.adjust_volume("Bedroom", 4)

    def test_play_id_answers_nothing_when_it_started_and_names_the_id_when_not(self):
        mm = make_sonos()
        mm.play_in_zone = Mock(return_value=True)
        assert mm.play_id("Kitchen", "F-1") == ""
        mm.play_in_zone.assert_called_once_with("F-1", "Kitchen")
        mm.play_in_zone = Mock(return_value=False)
        problem = mm.play_id("Kitchen", "F-1")
        assert "F-1" in problem and "Kitchen" in problem
