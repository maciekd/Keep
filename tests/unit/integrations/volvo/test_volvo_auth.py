"""VolvoAuth: the authorization URL (PKCE on top of the client secret), the address pasted back,
the token file it leaves (0600, deleted on sign-out), the refresh near expiry with Volvo's
rotating refresh tokens, and the refused refresh that asks for a new sign-in."""
import base64
import hashlib
import json
import stat
import time
from datetime import datetime, timezone
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

import pytest
import requests

from integrations.volvo import VolvoAuth as auth_module
from integrations.volvo.VolvoAuth import (AUTHORIZE_URL, REDIRECT_URI, SCOPES, NotSignedIn, VolvoAuth,
                                          delete_token_file, token_path)

from .volvo_fakes import FakeResponse, FakeVolvo, token_file, token_grant

POST = "integrations.volvo.VolvoAuth.requests.post"


@pytest.fixture
def token_dir(tmp_path):
    with patch.object(auth_module, "TOKEN_DIR", tmp_path / "oauth"):
        yield tmp_path / "oauth"


@pytest.fixture
def volvo():
    fake = FakeVolvo()
    with patch(POST, fake.post):
        yield fake


def auth(client_id="app-id", client_secret="app-secret") -> VolvoAuth:
    return VolvoAuth(lambda: (client_id, client_secret))


def query(url: str) -> dict:
    return {k: v[0] for k, v in parse_qs(urlsplit(url).query).items()}


def redirected(sign_in, code="c0de", state=None) -> str:
    return f"{REDIRECT_URI}?code={code}&state={sign_in.state if state is None else state}"


def basic_credentials(call) -> str:
    return base64.b64decode(call.headers["Authorization"].removeprefix("Basic ")).decode()


def on_file() -> dict:
    return json.loads(token_path().read_text())


@pytest.mark.unit
class TestBegin:
    def test_the_url_is_the_code_flow_with_an_s256_challenge_the_scopes_and_the_loopback_redirect(self, token_dir):
        a = auth()
        sign_in = a.begin()

        assert sign_in.url.startswith(AUTHORIZE_URL + "?")
        params = query(sign_in.url)
        assert params["response_type"] == "code" and params["client_id"] == "app-id"
        assert params["redirect_uri"] == REDIRECT_URI == sign_in.redirect_uri
        assert params["code_challenge_method"] == "S256" and len(params["code_challenge"]) == 43
        assert params["state"] == sign_in.state and len(sign_in.state) >= 16
        assert set(params["scope"].split()) == set(SCOPES) and "location:read" in SCOPES and "conve:climatization_start_stop" in SCOPES
        assert a.pending()

    def test_the_challenge_is_the_hash_of_the_verifier_sent_at_the_exchange(self, token_dir, volvo):
        a = auth()
        sign_in = a.begin()
        volvo.answer_token(token_grant())

        assert a.finish(redirected(sign_in)) == ""

        exchange = volvo.calls[-1].data
        expected = base64.urlsafe_b64encode(hashlib.sha256(exchange["code_verifier"].encode()).digest()).decode().rstrip("=")
        assert query(sign_in.url)["code_challenge"] == expected

    def test_without_a_client_id_or_secret_it_refuses_to_start(self, token_dir):
        with pytest.raises(RuntimeError, match="client_id"):
            auth(client_id="").begin()
        with pytest.raises(RuntimeError, match="client_secret"):
            auth(client_secret="").begin()
        assert not auth().pending()


@pytest.mark.unit
class TestFinish:
    def test_the_code_is_exchanged_with_the_client_as_basic_auth_and_the_tokens_land_in_a_private_file(self, token_dir, volvo):
        a = auth()
        sign_in = a.begin()
        volvo.answer_token(token_grant(access_token="at.1", refresh_token="rt.1", expires_in=1799))
        before = time.time()

        assert a.finish(redirected(sign_in, code="4%2Fabc")) == ""

        exchange = volvo.calls[-1]
        assert exchange.data["grant_type"] == "authorization_code" and exchange.data["code"] == "4/abc"
        assert exchange.data["redirect_uri"] == REDIRECT_URI and exchange.data["code_verifier"]
        assert basic_credentials(exchange) == "app-id:app-secret"
        assert "client_secret" not in exchange.data
        saved = on_file()
        assert saved["refresh_token"] == "rt.1" and saved["access_token"] == "at.1"
        assert before + 1700 < saved["expiry"] <= time.time() + 1799
        assert stat.S_IMODE(token_path().stat().st_mode) == 0o600
        assert a.signed_in() and not a.pending() and not a.needs_sign_in_again()

    def test_an_address_of_another_flow_is_refused_and_nothing_exchanged(self, token_dir, volvo):
        a = auth()
        sign_in = a.begin()

        problem = a.finish(redirected(sign_in, state="someone-elses"))

        assert "another sign-in" in problem and volvo.calls == [] and not a.signed_in()

    def test_a_paste_while_no_flow_is_in_progress_points_at_the_login_command(self, token_dir, volvo):
        problem = auth().finish(f"{REDIRECT_URI}?code=x&state=y")
        assert "volvo_login()" in problem and volvo.calls == []

    def test_a_refused_exchange_says_why_and_leaves_no_file(self, token_dir, volvo):
        a = auth()
        sign_in = a.begin()
        volvo.answer_token({"error": "invalid_grant", "error_description": "Authorization code is invalid or expired."}, status=400)

        problem = a.finish(redirected(sign_in))

        assert "expired" in problem and "volvo_login()" in problem
        assert not token_path().exists() and not a.signed_in()

    def test_a_client_volvo_does_not_know_points_at_the_oauth_config(self, token_dir, volvo):
        a = auth()
        sign_in = a.begin()
        volvo.answer_token({"error": "invalid_client", "error_description": "Invalid client or client credentials."}, status=401)

        problem = a.finish(redirected(sign_in))

        assert "volvo.oauth" in problem and not a.signed_in()

    def test_a_grant_without_a_refresh_token_is_not_a_sign_in(self, token_dir, volvo):
        a = auth()
        sign_in = a.begin()
        volvo.answer_token(token_grant(refresh_token=None))

        problem = a.finish(redirected(sign_in))

        assert "refresh token" in problem and not a.signed_in()


@pytest.mark.unit
class TestToken:
    def _signed_in(self, token_dir, **more) -> VolvoAuth:
        token_dir.mkdir(parents=True)
        token_path().write_text(json.dumps(token_file(**more)))
        return auth()

    def test_a_token_with_more_than_a_minute_left_is_served_without_a_refresh(self, token_dir, volvo):
        a = self._signed_in(token_dir, expires_in=120)
        assert a.token() == "at.old" and volvo.calls == []

    def test_a_token_within_a_minute_of_its_expiry_is_refreshed_and_the_rotated_refresh_token_kept(self, token_dir, volvo):
        a = self._signed_in(token_dir, expires_in=30)
        volvo.answer_token(token_grant(access_token="at.new", refresh_token="rt.rotated", expires_in=1799))

        assert a.token() == "at.new"

        refresh = volvo.calls[-1]
        assert refresh.data == {"grant_type": "refresh_token", "refresh_token": "rt.old"}
        assert basic_credentials(refresh) == "app-id:app-secret"
        assert on_file()["refresh_token"] == "rt.rotated" and on_file()["access_token"] == "at.new"
        assert a.token() == "at.new" and len(volvo.calls) == 1

    def test_a_refresh_answered_without_a_refresh_token_keeps_the_one_on_file(self, token_dir, volvo):
        a = self._signed_in(token_dir, expires_in=0)
        volvo.answer_token(token_grant(access_token="at.new", refresh_token=None))

        assert a.token() == "at.new"
        assert on_file()["refresh_token"] == "rt.old" and a.signed_in()

    def test_a_refused_refresh_keeps_the_file_marks_the_sign_in_as_needed_and_stops_retrying(self, token_dir, volvo):
        a = self._signed_in(token_dir, expires_in=0)
        volvo.answer_token({"error": "invalid_grant", "error_description": "unknown, invalid, or expired refresh token"}, status=400)

        with pytest.raises(NotSignedIn, match="volvo_login"):
            a.token()
        assert a.needs_sign_in_again() and a.signed_in() and token_path().exists()

        with pytest.raises(NotSignedIn, match="lapsed"):
            a.token()
        assert len(volvo.calls) == 1

    def test_a_token_endpoint_outage_is_not_a_lapsed_sign_in(self, token_dir, volvo):
        a = self._signed_in(token_dir, expires_in=0)
        volvo.answer_token({"error": "server_error"}, status=503)

        with pytest.raises(NotSignedIn, match="server_error"):
            a.token()
        assert not a.needs_sign_in_again()

    def test_a_new_sign_in_clears_the_lapsed_mark(self, token_dir, volvo):
        a = self._signed_in(token_dir, expires_in=0)
        volvo.answer_token({"error": "invalid_grant"}, status=400)
        with pytest.raises(NotSignedIn):
            a.token()
        sign_in = a.begin()
        volvo.answer_token(token_grant(access_token="at.fresh", refresh_token="rt.fresh"))

        assert a.finish(redirected(sign_in)) == ""
        assert not a.needs_sign_in_again() and a.token() == "at.fresh"

    def test_without_a_file_there_is_no_token_only_the_hint(self, token_dir, volvo):
        with pytest.raises(NotSignedIn, match="volvo_login"):
            auth().token()
        assert volvo.calls == []


@pytest.mark.unit
class TestSignOut:
    def test_sign_out_deletes_the_file_and_forgets_a_pending_flow(self, token_dir, volvo):
        token_dir.mkdir(parents=True)
        token_path().write_text(json.dumps(token_file()))
        a = auth()
        a.begin()

        a.sign_out()

        assert not token_path().exists() and not a.signed_in() and not a.pending()

    def test_delete_token_file_says_whether_there_was_one(self, token_dir):
        assert delete_token_file() is False
        token_dir.mkdir(parents=True)
        token_path().write_text("{}")
        assert delete_token_file() is True and not token_path().exists()


@pytest.mark.unit
class TestTokenFile:
    def _on_disk(self, token_dir, **more):
        token_dir.mkdir(parents=True)
        token_path().write_text(json.dumps(token_file(**more)))

    def test_an_unreadable_file_means_not_signed_in_and_no_refresh_is_tried(self, token_dir, volvo):
        token_dir.mkdir(parents=True)
        token_path().write_text("{not json")
        a = auth()
        assert not a.signed_in()
        with pytest.raises(NotSignedIn, match="volvo_login"):
            a.token()
        assert volvo.calls == []

    def test_expires_at_is_the_files_expiry_in_utc_and_none_without_a_file(self, token_dir):
        assert auth().expires_at() is None
        self._on_disk(token_dir, expires_in=600)
        expires = auth().expires_at()
        assert expires.tzinfo == timezone.utc
        assert 590 < (expires - datetime.now(timezone.utc)).total_seconds() <= 600

    def test_a_refresh_token_without_an_access_token_refreshes_at_once(self, token_dir, volvo):
        self._on_disk(token_dir, expires_in=3600, access_token="")
        volvo.answer_token(token_grant(access_token="at.new"))
        assert auth().token() == "at.new" and len(volvo.calls) == 1


@pytest.mark.unit
class TestTokenEndpointFailures:
    def _expired(self, token_dir) -> VolvoAuth:
        token_dir.mkdir(parents=True)
        token_path().write_text(json.dumps(token_file(expires_in=0)))
        return auth()

    def test_an_unreachable_endpoint_is_not_a_lapsed_sign_in(self, token_dir):
        a = self._expired(token_dir)

        def unreachable(*args, **kwargs):
            raise requests.ConnectionError("dns")
        with patch(POST, unreachable):
            with pytest.raises(NotSignedIn, match="could not be reached"):
                a.token()
        assert not a.needs_sign_in_again() and a.signed_in()

    def test_a_gateway_error_without_a_json_body_is_named_by_its_status_and_not_lapsed(self, token_dir, volvo):
        a = self._expired(token_dir)
        volvo.token_answers.append(FakeResponse(502, None))
        with pytest.raises(NotSignedIn, match="HTTP 502"):
            a.token()
        assert not a.needs_sign_in_again()

    @pytest.mark.parametrize("status", [400, 401, 403])
    def test_a_refusal_the_owner_must_fix_marks_the_sign_in_lapsed(self, token_dir, volvo, status):
        a = self._expired(token_dir)
        volvo.answer_token({"error": "invalid_client"}, status=status)
        with pytest.raises(NotSignedIn):
            a.token()
        assert a.needs_sign_in_again()
