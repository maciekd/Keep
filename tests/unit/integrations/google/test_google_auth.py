"""GoogleAuth: the authorization URL, the code pasted back as a URL or bare, the token file it
leaves (0600, deleted on sign-out), the refresh near expiry and the refused refresh."""
import json
import stat
import time
from unittest.mock import Mock, patch
from urllib.parse import parse_qs, urlsplit

import pytest
import requests

from integrations.google import GoogleAuth as auth_module
from integrations.google.GoogleAuth import (REDIRECT_URI, SCOPE_CALENDAR, SCOPE_MAIL, GoogleAuth, NotSignedIn,
                                            delete_all_token_files, token_path)

from .google_fakes import FakeGoogle, token_file, token_grant

POST = "integrations.google.GoogleAuth.requests.post"


@pytest.fixture
def token_dir(tmp_path):
    with patch.object(auth_module, "TOKEN_DIR", tmp_path / "oauth"):
        yield tmp_path / "oauth"


@pytest.fixture
def google():
    fake = FakeGoogle()
    with patch(POST, fake.post):
        yield fake


def auth(name="Personal", mail=True, calendar=True, address="marcus@example.com", client_id="cid.apps.googleusercontent.com",
         client_secret="csecret") -> GoogleAuth:
    scopes = [s for s, on in ((SCOPE_MAIL, mail), (SCOPE_CALENDAR, calendar)) if on]
    return GoogleAuth(name, lambda: (client_id, client_secret), lambda: scopes, lambda: address)


def query(url: str) -> dict:
    return {k: v[0] for k, v in parse_qs(urlsplit(url).query).items()}


def redirected(sign_in, code="4/0Acode", state=None) -> str:
    return f"{REDIRECT_URI}?state={sign_in.state if state is None else state}&code={code.replace('/', '%2F')}&scope=email"


@pytest.mark.unit
class TestBegin:
    def test_the_url_asks_for_the_scopes_of_the_flags_offline_access_consent_and_carries_the_state(self, token_dir):
        both = auth().begin()
        mail_only = auth(name="Mail", calendar=False).begin()

        params = query(both.url)
        assert both.url.startswith("https://accounts.google.com/o/oauth2/v2/auth?")
        assert set(params["scope"].split()) == {SCOPE_MAIL, SCOPE_CALENDAR}
        assert set(query(mail_only.url)["scope"].split()) == {SCOPE_MAIL}
        assert (params["access_type"], params["prompt"], params["response_type"]) == ("offline", "consent", "code")
        assert params["redirect_uri"] == REDIRECT_URI and params["client_id"] == "cid.apps.googleusercontent.com"
        assert params["state"] == both.state and len(both.state) >= 16 and both.state != mail_only.state
        assert params["login_hint"] == "marcus@example.com"
        assert "marcus@example.com" in both.instructions and "localhost:18722" in both.instructions

    def test_without_a_client_id_or_a_flag_nothing_starts(self, token_dir):
        with pytest.raises(RuntimeError, match="client_id"):
            auth(client_id="").begin()
        with pytest.raises(RuntimeError):
            auth(mail=False, calendar=False).begin()


@pytest.mark.unit
class TestFinish:
    def test_the_redirected_url_is_exchanged_and_leaves_a_0600_token_file(self, token_dir, google):
        google.answer_token(token_grant())
        a = auth()
        sign_in = a.begin()
        assert not a.signed_in()

        assert a.finish(redirected(sign_in)) == ""

        post, = google.token_posts()
        assert post.data["code"] == "4/0Acode" and post.data["grant_type"] == "authorization_code"
        assert post.data["redirect_uri"] == REDIRECT_URI and post.data["client_secret"] == "csecret"
        assert a.signed_in() and a.signed_in_as() == "marcus@example.com" and not a.pending()
        path = token_path("Personal")
        assert path == token_dir / "google-personal.json" and stat.S_IMODE(path.stat().st_mode) == 0o600
        stored = json.loads(path.read_text())
        assert stored["refresh_token"] == "1//refresh" and 3500 < stored["expiry"] - time.time() <= 3599
        assert a.token() == "ya29.access" and len(google.token_posts()) == 1      # fresh: no refresh yet

    def test_a_bare_code_the_bare_redirect_and_the_link_are_refused_without_an_exchange(self, token_dir, google):
        a = auth()
        sign_in = a.begin()
        for pasted in ("  4/0Abare  ", REDIRECT_URI, sign_in.url):
            assert "whole address" in a.finish(pasted), pasted
        assert google.token_posts() == [] and not a.signed_in() and a.pending()

    def test_a_wrong_state_or_no_flow_exchanges_nothing(self, token_dir, google):
        a = auth()
        assert "google_login" in a.finish("4/0Acode")          # nothing begun
        sign_in = a.begin()
        refusal = a.finish(redirected(sign_in, state="someone-elses"))
        assert "another sign-in" in refusal and not a.signed_in()
        assert google.token_posts() == []
        assert a.matches(sign_in.state) and not a.matches("someone-elses") and not a.matches("")

    def test_an_expired_code_and_a_refused_consent_say_why(self, token_dir, google):
        google.answer_token({"error": "invalid_grant", "error_description": "Bad Request"}, status=400)
        a = auth()
        sign_in = a.begin()
        expired = a.finish(redirected(sign_in))
        denied = a.finish(f"{REDIRECT_URI}?error=access_denied&state={sign_in.state}")
        assert "expired" in expired and 'google_login("Personal")' in expired
        assert "access_denied" in denied
        assert not a.signed_in() and not token_path("Personal").exists()

    def test_an_answer_without_a_refresh_token_is_not_a_sign_in(self, token_dir, google):
        google.answer_token(token_grant(refresh_token=""))
        a = auth()
        sign_in = a.begin()
        outcome = a.finish(redirected(sign_in))
        assert "refresh token" in outcome and not a.signed_in()


@pytest.mark.unit
class TestToken:
    def _signed_in(self, token_dir, **file):
        token_dir.mkdir()
        token_path("Personal").write_text(json.dumps(token_file(**file)))

    def test_a_token_file_from_an_earlier_run_means_signed_in_and_a_fresh_token_is_used_as_is(self, token_dir, google):
        self._signed_in(token_dir, expires_in=3000)
        a = auth()
        assert a.signed_in() and a.signed_in_as() == "marcus@example.com"
        assert a.token() == "ya29.old" and google.token_posts() == []
        assert 2990 < (a.expires_at().timestamp() - time.time()) <= 3000

    def test_near_its_expiry_the_token_is_refreshed_with_the_refresh_token_and_the_file_updated(self, token_dir, google):
        self._signed_in(token_dir, expires_in=30)
        google.answer_token(token_grant(access_token="ya29.new", refresh_token=""))
        a = auth()

        assert a.token() == "ya29.new"

        post, = google.token_posts()
        assert post.data["grant_type"] == "refresh_token" and post.data["refresh_token"] == "1//refresh"
        assert post.data["client_id"] == "cid.apps.googleusercontent.com" and post.data["client_secret"] == "csecret"
        stored = json.loads(token_path("Personal").read_text())
        assert stored["access_token"] == "ya29.new" and stored["refresh_token"] == "1//refresh"
        assert stored["expiry"] - time.time() > 3000
        assert a.token() == "ya29.new" and len(google.token_posts()) == 1

    def test_a_refresh_refused_with_invalid_grant_keeps_the_file_and_asks_to_sign_in_again(self, token_dir, google):
        self._signed_in(token_dir, expires_in=0)
        google.answer_token({"error": "invalid_grant", "error_description": "Token has been expired or revoked."}, status=400)
        a = auth()

        with pytest.raises(NotSignedIn) as first:
            a.token()
        with pytest.raises(NotSignedIn) as second:
            a.token()

        assert 'sign in again — google_login("Personal")' in str(first.value) and "revoked" in str(first.value)
        assert 'google_login("Personal")' in str(second.value) and len(google.token_posts()) == 1
        assert a.needs_sign_in_again() and a.signed_in() and token_path("Personal").exists()

    def test_a_new_sign_in_clears_the_refused_state(self, token_dir, google):
        self._signed_in(token_dir, expires_in=0)
        google.answer_token({"error": "invalid_grant"}, status=400)
        google.answer_token(token_grant(access_token="ya29.again"))
        a = auth()
        with pytest.raises(NotSignedIn):
            a.token()
        sign_in = a.begin()
        assert a.finish(redirected(sign_in)) == ""
        assert not a.needs_sign_in_again() and a.token() == "ya29.again"

    def test_without_a_token_asking_for_one_says_to_sign_in(self, token_dir, google):
        with pytest.raises(NotSignedIn) as raised:
            auth().token()
        assert 'google_login("Personal")' in str(raised.value) and google.token_posts() == []

    def test_sign_out_and_losing_the_config_delete_the_files(self, token_dir):
        token_dir.mkdir()
        token_path("Personal").write_text(json.dumps(token_file()))
        token_path("Charlotte's").write_text(json.dumps(token_file("charlotte@example.com")))
        a = auth()
        assert a.signed_in()
        a.sign_out()
        assert not a.signed_in() and not token_path("Personal").exists()
        assert token_path("Charlotte's").name == "google-charlotte-s.json" and token_path("Charlotte's").exists()
        assert delete_all_token_files() == 1 and list(token_dir.iterdir()) == []
        assert delete_all_token_files() == 0


@pytest.mark.unit
class TestRefusedRefresh:
    def _signed_in(self, token_dir, **file):
        token_dir.mkdir()
        token_path("Personal").write_text(json.dumps(token_file(**file)))

    def test_a_refusal_that_is_not_invalid_grant_is_tried_again_on_the_next_call(self, token_dir, google):
        """Only invalid_grant means the consent is gone; a bad client or a server error is
        reported but not latched, so the next call asks Google again."""
        self._signed_in(token_dir, expires_in=0)
        google.answer_token({"error": "invalid_client", "error_description": "The OAuth client was not found."}, status=401)
        google.answer_token(None, status=500)
        google.answer_token(token_grant(access_token="ya29.new", refresh_token=""))
        a = auth()

        with pytest.raises(NotSignedIn) as bad_client:
            a.token()
        with pytest.raises(NotSignedIn) as server_error:
            a.token()

        assert "The OAuth client was not found." in str(bad_client.value) and "500" in str(server_error.value)
        assert not a.needs_sign_in_again() and a.signed_in()
        assert a.token() == "ya29.new" and len(google.token_posts()) == 3

    def test_an_unreachable_token_endpoint_is_a_not_signed_in_that_leaves_the_file_alone(self, token_dir):
        self._signed_in(token_dir, expires_in=0)
        before = token_path("Personal").read_text()
        with patch(POST, Mock(side_effect=requests.ConnectionError("name resolution failed"))), pytest.raises(NotSignedIn) as raised:
            auth().token()
        assert "reached" in str(raised.value) and "name resolution failed" in str(raised.value)
        assert token_path("Personal").read_text() == before

    def test_an_unreadable_token_file_counts_as_not_signed_in(self, token_dir, google):
        token_dir.mkdir()
        token_path("Personal").write_text("{refresh_token: not json")
        a = auth()
        assert not a.signed_in() and a.signed_in_as() == "" and a.expires_at() is None
        with pytest.raises(NotSignedIn):
            a.token()
        assert google.token_posts() == []

    def test_a_pasted_address_with_the_state_but_no_code_exchanges_nothing(self, token_dir, google):
        a = auth()
        sign_in = a.begin()
        reply = a.finish(f"{REDIRECT_URI}?state={sign_in.state}&scope=email")
        assert "code" in reply.lower() and google.token_posts() == []
        assert a.pending() and not a.signed_in()
