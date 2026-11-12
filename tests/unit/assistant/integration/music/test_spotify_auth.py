"""SpotifyAuth: the authorization URL, the whole redirected address pasted back, the token file it
leaves (0600, deleted on sign-out), the refresh when the token ran out, the refused refresh told
apart from a failed one, and spotipy's interactive fallback cut off."""
import json
import stat
import time
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

import pytest
import requests

from assistant.integration.music.SpotifyAuth import (REDIRECT_URI, PLAYLIST_SCOPES as SCOPES, NotSignedIn, SpotifyAuth,
                                                     TokenUnavailable, _PasteOnlyOAuth)

CLIENT = ("cid-1234", "csecret")


class FakeSpotify:
    """Spotify's token endpoint: every code and refresh token it knows answers a token; the rest invalid_grant."""

    def __init__(self):
        self.codes = {"AQcode": "access-1"}
        self.refresh_tokens = {"refresh-1": "access-2"}
        self.posts = []
        self.outage = None   # a status code to answer, or an exception to raise, instead of a token

    def post(self, url, data=None, **kwargs):
        self.posts.append(data)
        if isinstance(self.outage, Exception):
            raise self.outage
        if self.outage:
            return _response(self.outage, {"error": "server_error"})
        if data.get("grant_type") == "authorization_code" and data.get("code") in self.codes:
            return _response(200, {"access_token": self.codes[data["code"]], "token_type": "Bearer", "expires_in": 3600,
                                   "refresh_token": "refresh-1", "scope": " ".join(SCOPES)})
        if data.get("grant_type") == "refresh_token" and data.get("refresh_token") in self.refresh_tokens:
            return _response(200, {"access_token": self.refresh_tokens[data["refresh_token"]], "token_type": "Bearer",
                                   "expires_in": 3600, "scope": " ".join(SCOPES)})
        return _response(400, {"error": "invalid_grant", "error_description": "Invalid authorization code"})


def _response(status: int, body: dict):
    response = requests.Response()
    response.status_code = status
    response._content = json.dumps(body).encode()
    return response


@pytest.fixture
def spotify():
    fake = FakeSpotify()
    with patch.object(requests.Session, "post", fake.post):
        yield fake


@pytest.fixture
def token_path(tmp_path):
    return tmp_path / "oauth" / "spotify.json"


def auth(token_path, client=CLIENT, scopes=None) -> SpotifyAuth:
    if scopes is None:
        return SpotifyAuth(lambda: client, token_path)
    return SpotifyAuth(lambda: client, token_path, scopes=lambda: scopes)


def token_file(path, access="access-1", refresh="refresh-1", expires_in=3600, scope=" ".join(SCOPES)):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"access_token": access, "refresh_token": refresh, "token_type": "Bearer",
                                "expires_in": expires_in, "expires_at": int(time.time()) + expires_in, "scope": scope}))


def query(url: str) -> dict:
    return {k: v[0] for k, v in parse_qs(urlsplit(url).query).items()}


def redirected(sign_in, code="AQcode", state=None) -> str:
    return f"{REDIRECT_URI}?code={code}&state={sign_in.state if state is None else state}"


@pytest.mark.unit
class TestBegin:
    def test_the_url_asks_for_the_playlist_scopes_at_the_registered_redirect_and_carries_a_fresh_state(self, token_path):
        first = auth(token_path).begin()
        second = auth(token_path).begin()

        params = query(first.url)
        assert first.url.startswith("https://accounts.spotify.com/authorize?")
        assert set(params["scope"].split()) == set(SCOPES)
        assert params["redirect_uri"] == REDIRECT_URI and params["client_id"] == "cid-1234"
        assert params["response_type"] == "code"
        assert params["state"] == first.state and len(first.state) >= 16 and first.state != second.state
        assert REDIRECT_URI in first.instructions and first.url in first.instructions and " as " not in first.instructions

    def test_without_a_client_it_refuses(self, token_path):
        with pytest.raises(RuntimeError, match="client_id"):
            auth(token_path, client=("", "")).begin()

    def test_spotipys_interactive_fallback_is_cut_off(self, token_path):
        flow = _PasteOnlyOAuth(client_id="cid", client_secret="cs", redirect_uri=REDIRECT_URI, open_browser=False)
        with pytest.raises(NotSignedIn):
            flow.get_auth_response()


@pytest.mark.unit
class TestFinish:
    def test_the_pasted_address_is_exchanged_and_the_token_file_written_read_only_to_the_owner(self, token_path, spotify):
        a = auth(token_path)
        sign_in = a.begin()
        assert not a.signed_in()

        assert a.finish(redirected(sign_in)) == ""

        assert a.signed_in() and not a.pending()
        assert spotify.posts[-1]["code"] == "AQcode" and spotify.posts[-1]["redirect_uri"] == REDIRECT_URI
        on_file = json.loads(token_path.read_text())
        assert on_file["refresh_token"] == "refresh-1" and on_file["access_token"] == "access-1"
        assert stat.S_IMODE(token_path.stat().st_mode) == 0o600
        assert a.token() == "access-1"

    def test_a_bare_code_the_bare_redirect_and_the_link_are_refused_without_an_exchange(self, token_path, spotify):
        a = auth(token_path)
        sign_in = a.begin()
        for pasted in ("AQcode", REDIRECT_URI, sign_in.url, f"{REDIRECT_URI}?code=AQcode"):
            assert "whole address" in a.finish(pasted), pasted
        assert spotify.posts == [] and not a.signed_in() and a.pending()

    def test_a_token_spotipy_could_not_write_is_a_failed_sign_in_not_a_silent_one(self, token_path, spotify):
        a = auth(token_path)
        sign_in = a.begin()
        with patch("spotipy.cache_handler.CacheFileHandler.save_token_to_cache"):
            failure = a.finish(redirected(sign_in))
        assert str(token_path) in failure and not a.signed_in() and a.pending()

    def test_another_flows_state_an_expired_code_and_a_refusal_are_reported_not_exchanged(self, token_path, spotify):
        a = auth(token_path)
        sign_in = a.begin()

        other = a.finish(redirected(sign_in, state="not-mine"))
        expired = a.finish(redirected(sign_in, code="AQold"))
        refused = a.finish(f"{REDIRECT_URI}?error=access_denied&state={sign_in.state}")

        assert "another sign-in" in other and "spotify_login()" in other
        assert "expired" in expired and "spotify_login()" in expired
        assert "access_denied" in refused
        assert [p["code"] for p in spotify.posts] == ["AQold"]
        assert not a.signed_in() and a.pending()

    def test_without_a_flow_in_progress_it_says_so(self, token_path, spotify):
        assert "spotify_login()" in auth(token_path).finish("AQcode")
        assert spotify.posts == []


@pytest.mark.unit
class TestToken:
    def test_a_token_that_has_not_run_out_is_handed_out_without_a_refresh(self, token_path, spotify):
        token_file(token_path)
        assert auth(token_path).token() == "access-1"
        assert spotify.posts == []

    def test_a_token_that_ran_out_is_refreshed_and_the_file_updated(self, token_path, spotify):
        token_file(token_path, expires_in=-10)
        assert auth(token_path).token() == "access-2"
        assert spotify.posts[-1]["grant_type"] == "refresh_token"
        on_file = json.loads(token_path.read_text())
        assert on_file["access_token"] == "access-2" and on_file["refresh_token"] == "refresh-1"
        assert stat.S_IMODE(token_path.stat().st_mode) == 0o600

    def test_a_refused_refresh_asks_for_a_new_sign_in_and_is_not_retried(self, token_path, spotify):
        token_file(token_path, refresh="revoked", expires_in=-10)
        a = auth(token_path)
        with pytest.raises(NotSignedIn, match="spotify_login"):
            a.token()
        assert a.needs_sign_in_again() and a.signed_in()

        with pytest.raises(NotSignedIn, match="spotify_login"):
            a.token()
        assert len(spotify.posts) == 1

        sign_in = a.begin()
        assert a.finish(redirected(sign_in)) == ""
        assert not a.needs_sign_in_again() and a.token() == "access-1"

    def test_an_unreachable_or_failing_token_endpoint_is_not_a_lost_sign_in(self, token_path, spotify):
        token_file(token_path, expires_in=-10)
        a = auth(token_path)
        for outage in (requests.ConnectionError("timed out"), 503):
            spotify.outage = outage
            with pytest.raises(TokenUnavailable):
                a.token()
            assert a.signed_in() and not a.needs_sign_in_again()

        spotify.outage = None
        assert a.token() == "access-2"

    def test_a_file_from_narrower_scopes_asks_for_a_new_sign_in(self, token_path, spotify):
        token_file(token_path, scope="playlist-read-private")
        with pytest.raises(NotSignedIn, match="scopes"):
            auth(token_path).token()

    def test_scopes_grown_since_the_sign_in_ask_for_a_new_one_and_the_url_asks_for_them(self, token_path, spotify):
        grown = SCOPES + ("user-modify-playback-state",)
        token_file(token_path)
        with pytest.raises(NotSignedIn, match="scopes"):
            auth(token_path, scopes=grown).token()
        params = query(auth(token_path, scopes=grown).begin().url)
        assert "user-modify-playback-state" in params["scope"].split()

    def test_a_caller_that_needs_only_the_scopes_on_file_is_served_after_they_grew(self, token_path, spotify):
        """A speaker system asking for playback must not take the playlist library away."""
        grown = SCOPES + ("user-modify-playback-state",)
        token_file(token_path, expires_in=-10)
        a = auth(token_path, scopes=grown)
        assert a.lacks_scopes()
        assert a.token(needs=SCOPES) == "access-2"
        assert set(json.loads(token_path.read_text())["scope"].split()) == set(SCOPES), \
            "the refresh does not claim scopes the sign-in never granted"
        with pytest.raises(NotSignedIn, match="scopes"):
            a.token()
        with pytest.raises(NotSignedIn, match="scopes"):
            a.token(needs=("user-modify-playback-state",))

    def test_a_token_carrying_every_scope_lacks_none(self, token_path, spotify):
        assert not auth(token_path).lacks_scopes(), "nobody signed in"
        token_file(token_path)
        assert not auth(token_path).lacks_scopes()

    def test_without_a_file_there_is_no_token(self, token_path, spotify):
        with pytest.raises(NotSignedIn, match="spotify_login"):
            auth(token_path).token()
        assert spotify.posts == []

    def test_sign_out_deletes_the_file(self, token_path, spotify):
        token_file(token_path)
        a = auth(token_path)
        a.sign_out()
        assert not token_path.exists() and not a.signed_in()
