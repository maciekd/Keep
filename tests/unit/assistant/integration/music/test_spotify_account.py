"""SpotifyAccount: search and its short ids, the user's playlist library (the fetch, its failures, the
429 hold-back, the dropped-connection retry, the reload gap), the sign-in's reload and the listeners,
the artist top tracks, and the scopes a Connect speaker system adds."""
import http.client
import time
from unittest.mock import Mock, patch

import pytest
import requests
import urllib3
from spotipy.exceptions import SpotifyException

from assistant.integration.music.SpotifyAccount import PLAYBACK_SCOPES, SpotifyAccount, SpotifyPlaylist
from assistant.integration.music.SpotifyAuth import PLAYLIST_SCOPES, NotSignedIn, TokenUnavailable
from tests.unit.assistant.integration.music.spotify_fakes import make_account

MODULE = "assistant.integration.music.SpotifyAccount"
LIBRARY_PAGE = {'items': [{'name': 'Road Trip', 'uri': 'spotify:playlist:rt2', 'owner': {}}], 'total': 1}


@pytest.fixture(autouse=True)
def class_state():
    """The listeners and the extra scopes live on the class: each test starts and ends without any."""
    SpotifyAccount._library_listeners.clear()
    SpotifyAccount._extra_scopes.clear()
    yield
    SpotifyAccount._library_listeners.clear()
    SpotifyAccount._extra_scopes.clear()
    SpotifyAccount.reset_instance()


class TestSearch:
    def test_search_lists_tracks_and_artists_with_popularity_and_gives_each_a_short_id(self):
        account = make_account()
        account._app_client = Mock()
        account._app_client.search.return_value = {
            'tracks': {'items': [{'name': 'Song A', 'artists': [{'name': 'Artist 1'}], 'popularity': 85,
                                  'uri': 'spotify:track:abc123'}]},
            'artists': {'items': [{'name': 'Artist 1', 'genres': ['pop', 'rock'], 'popularity': 90,
                                   'uri': 'spotify:artist:def456'}]},
        }

        result = account.search("test query")

        assert "Song A" in result and "Artist 1" in result and "85" in result
        assert "TRACKS:" in result and "ARTISTS:" in result
        assert account.resolve("AA") == 'spotify:track:abc123'
        assert account.resolve("AB") == 'spotify:artist:def456'

    def test_search_lists_results_without_popularity(self):
        """A Development Mode app (created since February 2026) gets no popularity field at all."""
        account = make_account()
        account._app_client = Mock()
        account._app_client.search.return_value = {
            'tracks': {'items': [{'name': 'Song A', 'artists': [{'name': 'Artist 1'}], 'uri': 'spotify:track:abc123'}]},
            'artists': {'items': [{'name': 'Artist 1', 'genres': [], 'uri': 'spotify:artist:def456'}]},
        }

        result = account.search("test query")

        assert not result.startswith("Error")
        assert "popularity" not in result
        assert {account.resolve("AA"), account.resolve("AB")} == {'spotify:track:abc123', 'spotify:artist:def456'}

    def test_search_lists_public_playlists_with_their_owner_and_skips_null_items(self):
        account = make_account()
        account._app_client = Mock()
        account._app_client.search.return_value = {
            'tracks': {'items': []}, 'artists': {'items': []},
            'playlists': {'items': [None, {'name': 'Summer Hits', 'uri': 'spotify:playlist:sh1',
                                           'owner': {'display_name': 'Alice'}}]},
        }

        result = account.search("summer")

        assert "Summer Hits" in result and "Alice" in result
        assert account.resolve("AA") == "spotify:playlist:sh1"
        assert account._app_client.search.call_args.kwargs['type'] == 'track,artist,playlist'

    def test_the_short_ids_run_through_the_two_letter_space_and_wrap(self):
        account = make_account()
        assert [account._new_search_id() for _ in range(3)] == ["AA", "AB", "AC"]
        account._next_search_id = 26 * 26 - 1
        assert [account._new_search_id() for _ in range(2)] == ["ZZ", "AA"]

    def test_an_unknown_short_id_resolves_to_nothing(self):
        assert make_account().resolve("QQ") is None

    def test_without_a_client_search_answers_the_error_instead_of_raising(self):
        account = make_account()
        assert account.search("anything").startswith("Error searching Spotify")

    @pytest.mark.parametrize("text,expected", [("AB", True), ("ab", False), ("A1", False), ("ABC", False),
                                               ("S-0", False)])
    def test_only_two_capital_letters_are_a_search_id(self, text, expected):
        assert SpotifyAccount.is_search_id(text) is expected


class TestArtistTopTracks:
    def _account(self, answer):
        account = make_account()
        account._app_client = Mock()
        account._app_client.track.return_value = {'artists': [{'id': 'art01', 'name': 'Artist 1'}]}
        if isinstance(answer, int):
            account._app_client.artist_top_tracks.side_effect = SpotifyException(answer, -1, "refused")
        else:
            account._app_client.artist_top_tracks.return_value = {'tracks': answer}
        return account

    def test_the_seed_tracks_artist_is_asked_for_its_top_tracks(self):
        account = self._account([{'id': 'top02'}])
        assert account.artist_top_tracks("seed01") == [{'id': 'top02'}]
        assert account._app_client.artist_top_tracks.call_args.args[0] == 'art01'

    @pytest.mark.parametrize("status", [403, 404])
    def test_an_endpoint_closed_to_development_mode_apps_is_no_top_tracks(self, status):
        assert self._account(status).artist_top_tracks("seed01") == []

    def test_another_spotify_error_is_raised(self):
        with pytest.raises(SpotifyException):
            self._account(500).artist_top_tracks("seed01")


class TestLibrary:
    def test_s_ids_resolve_to_the_cached_playlists_in_order(self):
        account = make_account()
        assert account.playlist("S-0").uri == "spotify:playlist:dw1"
        assert account.playlist("S-1").name == "Road Trip"
        assert account.playlist("S-2") is None
        assert account.playlist("P-0") is None
        assert account.playlist("S-x") is None

    def test_playlists_returns_a_copy(self):
        account = make_account()
        account.playlists().clear()
        assert len(account.playlists()) == 2

    def test_the_fetch_takes_the_first_page_of_the_library_in_spotifys_order_skips_hidden_items_and_keeps_the_size(self):
        account = make_account()
        client = Mock()
        account.user_client = Mock(return_value=client)
        client.current_user_playlists.return_value = {
            'items': [{'name': 'Road Trip', 'uri': 'spotify:playlist:rt2', 'owner': {}},
                      None,
                      {'name': 'Shared Dinner', 'uri': 'spotify:playlist:sd3', 'owner': {'display_name': 'Alice'}}],
            'next': 'https://api.spotify.com/v1/me/playlists?offset=20', 'total': 57}

        assert account.refresh_library() is True

        assert [(p.name, p.uri, p.owner) for p in account.playlists()] == [
            ("Road Trip", "spotify:playlist:rt2", ""),
            ("Shared Dinner", "spotify:playlist:sd3", "Alice")]
        assert account.playlists_total() == 57
        client.current_user_playlists.assert_called_once_with(limit=SpotifyAccount.PLAYLISTS_IN_PROMPT)
        client.next.assert_not_called()
        assert account.library_error() == ""

    def test_without_a_signed_in_user_the_library_is_empty_and_spotify_is_not_asked(self):
        account = make_account()
        account._auth.signed_in.return_value = False
        account.user_client = Mock()

        account.refresh_library()

        assert account.playlists() == []
        account.user_client.assert_not_called()

    def test_a_transient_failure_keeps_the_library_and_says_why(self):
        account = make_account()
        for failure in (TokenUnavailable("token endpoint could not be reached"), Exception("timed out")):
            account.user_client = Mock(side_effect=failure)
            assert account.refresh_library() is False
            assert len(account.playlists()) == 2 and account.playlists_total() == 2
            assert str(failure) in account.library_error()

    @patch(f"{MODULE}.system_note")
    def test_a_sign_in_the_user_must_renew_empties_the_library_and_is_noted_once(self, note):
        account = make_account()
        account.user_client = Mock(side_effect=NotSignedIn("sign in again"))
        account._auth.needs_sign_in_again.return_value = True

        assert account.refresh_library() is True
        account.refresh_library(force=True)

        assert account.playlists() == [] and account.playlists_total() == 0
        assert note.call_count == 1 and "spotify_login" in note.call_args.args[1]

    @patch(f"{MODULE}.system_note")
    def test_a_sign_in_from_before_the_scopes_grew_keeps_the_library_and_is_noted_once(self, note):
        """Turning on a system that starts Spotify over Connect: the playlists stay for every
        system, and the user hears that a new sign-in is needed — once."""
        account = make_account()
        account._auth.lacks_scopes.return_value = True
        client = Mock()
        client.current_user_playlists.return_value = {
            'items': [{'name': 'Road Trip', 'uri': 'spotify:playlist:rt2', 'owner': {}}], 'total': 1}
        account.user_client = Mock(return_value=client)

        assert account.refresh_library() is True
        account.refresh_library(force=True)

        assert [p.name for p in account.playlists()] == ["Road Trip"]
        assert note.call_count == 1 and "spotify_login" in note.call_args.args[1]

    @patch(f"{MODULE}.system_note")
    def test_a_token_that_was_never_there_is_not_a_revoked_one_and_gets_no_note(self, note):
        account = make_account()
        account.user_client = Mock(side_effect=NotSignedIn("not signed in"))
        account.refresh_library()
        note.assert_not_called()

    def test_a_reload_within_the_gap_is_skipped_unless_forced(self):
        """Two speaker systems each refresh every 30 minutes: the second within the gap costs no call."""
        account = make_account()
        client = Mock()
        client.current_user_playlists.return_value = LIBRARY_PAGE
        account.user_client = Mock(return_value=client)

        assert account.refresh_library() is True
        assert account.refresh_library() is False
        assert client.current_user_playlists.call_count == 1

        assert account.refresh_library(force=True) is True
        assert client.current_user_playlists.call_count == 2

        account._library_fetched_at = time.time() - SpotifyAccount.MIN_REFRESH_GAP - 1
        assert account.refresh_library() is True
        assert client.current_user_playlists.call_count == 3

    def test_a_failed_reload_is_tried_again_without_waiting_for_the_gap(self):
        account = make_account()
        client = Mock()
        client.current_user_playlists.side_effect = [Exception("timed out"), LIBRARY_PAGE]
        account.user_client = Mock(return_value=client)

        assert account.refresh_library() is False
        assert account.refresh_library() is True
        assert [p.uri for p in account.playlists()] == ["spotify:playlist:rt2"]

    def test_the_user_client_follows_the_access_token(self):
        account = make_account()
        account._auth.token.return_value = "tok-1"
        first = account.user_client()
        assert account.user_client() is first
        account._auth.token.return_value = "tok-2"
        assert account.user_client() is not first


class TestSignIn:
    def test_a_finished_sign_in_reloads_the_library_at_once_even_right_after_a_reload(self):
        account = make_account()
        account._auth.finish.return_value = ""
        account._user_client = Mock()
        account._library_fetched_at = time.time()   # a reload a moment ago: the sign-in's is forced anyway
        account._fetch_library = Mock(return_value=([SpotifyPlaylist("New", "spotify:playlist:n1")], 1))

        assert account.login_finish("http://127.0.0.1:8888/callback?code=AQ&state=st") == ""

        assert [p.name for p in account.playlists()] == ["New"]
        assert account.playlists_total() == 1
        assert account._user_client is None

    def test_a_failed_sign_in_comes_back_as_the_reason_and_touches_nothing(self):
        account = make_account()
        account._auth.finish.return_value = "the code expired"
        account._fetch_library = Mock()

        assert account.login_finish("AQold") == "the code expired"
        account._fetch_library.assert_not_called()
        assert len(account.playlists()) == 2

    def test_a_reloaded_library_wakes_the_listeners_but_a_failed_reload_does_not(self):
        account = make_account()
        account._auth.finish.return_value = ""
        woken, broken = [], Mock(side_effect=RuntimeError("listener broke"))
        listener = lambda: woken.append(len(account.playlists()))
        SpotifyAccount.add_library_listener(broken)
        SpotifyAccount.add_library_listener(listener)
        SpotifyAccount.add_library_listener(listener)   # registered once however often it is added

        account._fetch_library = Mock(return_value=([SpotifyPlaylist("New", "spotify:playlist:n1")], 1))
        assert account.login_finish("http://127.0.0.1:8888/callback?code=AQ&state=st") == ""
        assert woken == [1], "woken after the cache holds the new list, and once, despite the listener before it raising"

        account._fetch_library = Mock(return_value=None)
        account.refresh_library(force=True)
        assert woken == [1]

        SpotifyAccount.remove_library_listener(listener)
        account._fetch_library = Mock(return_value=([], 0))
        account.refresh_library(force=True)
        assert woken == [1]

    def test_the_users_name_is_read_from_the_profile_and_a_failure_is_no_name(self):
        account = make_account()
        account.user_client = Mock(return_value=Mock(current_user=Mock(return_value={'display_name': 'Marcus'})))
        assert account.user_name() == "Marcus"
        account.user_client = Mock(side_effect=NotSignedIn("not signed in"))
        assert account.user_name() == ""


class TestScopes:
    def test_a_connect_speaker_system_adds_the_playback_scopes_to_the_librarys(self):
        assert set(SpotifyAccount.scopes()) == set(PLAYLIST_SCOPES)
        SpotifyAccount.want_playback()
        SpotifyAccount.want_playback()
        scopes = SpotifyAccount.scopes()
        assert set(scopes) == set(PLAYLIST_SCOPES) | set(PLAYBACK_SCOPES)
        assert len(scopes) == len(set(scopes))

    def test_the_sign_in_asks_for_the_scopes_as_they_are_at_the_time(self, tmp_path):
        account = SpotifyAccount.get_instance()
        account._auth._client = lambda: ("cid-1234", "csecret")
        account._auth.path = tmp_path / "oauth" / "spotify.json"
        before = account.login_begin().url
        SpotifyAccount.want_playback()
        after = account.login_begin().url
        assert "user-modify-playback-state" not in before
        assert "user-modify-playback-state" in after


class TestDroppedConnection:
    def _account_on_clients(self, *clients):
        """An account whose successive Spotify(...) constructions hand out `clients` in turn."""
        account = make_account()
        account._auth.token.return_value = "tok-1"
        return account, patch(f"{MODULE}.Spotify", side_effect=list(clients))

    @pytest.mark.parametrize("dropped", [
        http.client.RemoteDisconnected("Remote end closed connection without response"),
        requests.exceptions.ConnectionError("('Connection aborted.', RemoteDisconnected(...))"),
        urllib3.exceptions.ProtocolError("Connection aborted."),
        ConnectionResetError(54, "Connection reset by peer"),
    ])
    def test_a_dropped_connection_is_retried_once_on_a_fresh_client(self, dropped):
        stale, fresh = Mock(), Mock()
        stale.current_user_playlists.side_effect = dropped
        fresh.current_user_playlists.return_value = LIBRARY_PAGE
        account, spotify_cls = self._account_on_clients(stale, fresh)

        with spotify_cls as built:
            assert account.refresh_library() is True

        assert [p.uri for p in account.playlists()] == ["spotify:playlist:rt2"] and account.playlists_total() == 1
        assert built.call_count == 2
        assert account._user_client is fresh
        assert account.library_error() == ""

    def test_a_connection_that_drops_twice_keeps_the_cached_library_and_is_not_tried_a_third_time(self):
        stale, fresh = Mock(), Mock()
        stale.current_user_playlists.side_effect = http.client.RemoteDisconnected("closed")
        fresh.current_user_playlists.side_effect = http.client.RemoteDisconnected("closed")
        account, spotify_cls = self._account_on_clients(stale, fresh)

        with spotify_cls as built:
            assert account.refresh_library() is False

        assert built.call_count == 2
        assert fresh.current_user_playlists.call_count == 1
        assert len(account.playlists()) == 2
        assert "RemoteDisconnected" in account.library_error()

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 429])
    def test_a_4xx_answer_is_not_retried(self, status):
        client = Mock()
        client.current_user_playlists.side_effect = SpotifyException(status, -1, "refused", headers={})
        account, spotify_cls = self._account_on_clients(client, Mock())

        with spotify_cls as built:
            assert account.refresh_library() is False

        assert built.call_count == 1 and client.current_user_playlists.call_count == 1
        assert str(status) in account.library_error()

    def test_a_429_holds_the_fetches_back_for_its_retry_after(self):
        limited, later = Mock(), Mock()
        limited.current_user_playlists.side_effect = SpotifyException(429, -1, "rate limited",
                                                                      headers={"Retry-After": "120"})
        later.current_user_playlists.return_value = LIBRARY_PAGE
        account, spotify_cls = self._account_on_clients(limited)

        with spotify_cls, patch(f"{MODULE}.time.time", return_value=1000.0):
            assert account.refresh_library(force=True) is False
        account._user_client, account._user_client_token = later, "tok-1"
        with patch(f"{MODULE}.time.time", return_value=1119.0):
            assert account.refresh_library(force=True) is False
        later.current_user_playlists.assert_not_called()
        with patch(f"{MODULE}.time.time", return_value=1121.0):
            assert account.refresh_library(force=True) is True

        assert [p.uri for p in account.playlists()] == ["spotify:playlist:rt2"]

    def test_a_429_without_a_retry_after_does_not_hold_the_next_fetch_back(self):
        limited = Mock()
        limited.current_user_playlists.side_effect = SpotifyException(429, -1, "rate limited", headers={})
        account, spotify_cls = self._account_on_clients(limited)

        with spotify_cls:
            assert account.refresh_library() is False

        assert account._not_before <= time.time()


def test_the_app_token_is_never_written_to_the_working_directory(tmp_path, monkeypatch):
    """spotipy's default cache is a `.cache` file in the working directory — the checkout's root, which no
    ignore rule covers, so a token would be one `git add -A` from a public commit."""
    monkeypatch.chdir(tmp_path)
    account = make_account()
    account.config.get.side_effect = lambda key, default=None: {"api_keys.spotify.client_id": "app-id",
                                                                "api_keys.spotify.client_secret": "app-secret"}.get(key)
    credentials = account.app_client().client_credentials_manager
    credentials.cache_handler.save_token_to_cache({"access_token": "t0ken", "expires_at": time.time() + 3600})
    assert list(tmp_path.iterdir()) == []
