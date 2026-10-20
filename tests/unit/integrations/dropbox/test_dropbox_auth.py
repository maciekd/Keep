"""DropboxAuth: the authorization URL (PKCE, offline access, no secret), the address pasted
back, the token file it leaves (0600, deleted on sign-out), the refresh without a secret near
expiry, and the refused refresh that asks for a new sign-in."""
import base64
import hashlib
import json
import stat
import time
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

import pytest
import requests

from integrations.dropbox import DropboxAuth as auth_module
from integrations.dropbox.DropboxAuth import (AUTHORIZE_URL, REDIRECT_URI, SCOPES, DropboxAuth, NotSignedIn,
                                              delete_token_file, token_path)

from .dropbox_fakes import FakeDropbox, FakeResponse, token_file, token_grant

POST = "integrations.dropbox.DropboxAuth.requests.post"


@pytest.fixture(autouse=True)
def no_pending_flow():
    auth_module._set_pending(None)
    yield
    auth_module._set_pending(None)


@pytest.fixture
def token_dir(tmp_path):
    with patch.object(auth_module, "TOKEN_DIR", tmp_path / "oauth"):
        yield tmp_path / "oauth"


@pytest.fixture
def dropbox():
    fake = FakeDropbox()
    with patch(POST, fake.post):
        yield fake


def auth(app_key="app-key") -> DropboxAuth:
    return DropboxAuth(lambda: app_key)


def query(url: str) -> dict:
    return {k: v[0] for k, v in parse_qs(urlsplit(url).query).items()}


def redirected(sign_in, code="c0de", state=None) -> str:
    return f"{REDIRECT_URI}?code={code}&state={sign_in.state if state is None else state}"


def on_file() -> dict:
    return json.loads(token_path().read_text())


@pytest.mark.unit
class TestBegin:
    def test_the_url_asks_for_a_code_with_an_s256_challenge_offline_access_and_the_read_scopes(self, token_dir):
        a = auth()
        sign_in = a.begin()

        assert sign_in.url.startswith(AUTHORIZE_URL + "?")
        params = query(sign_in.url)
        assert params["response_type"] == "code" and params["client_id"] == "app-key"
        assert params["redirect_uri"] == REDIRECT_URI == sign_in.redirect_uri
        assert params["token_access_type"] == "offline"
        assert params["code_challenge_method"] == "S256" and len(params["code_challenge"]) == 43
        assert params["state"] == sign_in.state and len(sign_in.state) >= 16
        assert set(params["scope"].split()) == set(SCOPES)
        assert all(scope.endswith(".read") for scope in SCOPES), "the integration is read-only"
        assert "client_secret" not in sign_in.url
        assert a.pending()

    def test_the_challenge_is_the_hash_of_the_verifier_sent_at_the_exchange_with_no_secret(self, token_dir, dropbox):
        a = auth()
        sign_in = a.begin()
        challenge = query(sign_in.url)["code_challenge"]
        dropbox.answer_token(token_grant())

        assert a.finish(redirected(sign_in)) == ""

        exchange = dropbox.calls[-1].data
        assert exchange["grant_type"] == "authorization_code" and exchange["code"] == "c0de"
        assert exchange["client_id"] == "app-key" and exchange["redirect_uri"] == REDIRECT_URI
        assert "client_secret" not in exchange and "Authorization" not in dropbox.calls[-1].headers
        digest = hashlib.sha256(exchange["code_verifier"].encode()).digest()
        assert base64.urlsafe_b64encode(digest).decode().rstrip("=") == challenge

    def test_without_an_app_key_there_is_no_link(self, token_dir):
        with pytest.raises(RuntimeError, match="dropbox.app_key"):
            auth("").begin()
        assert not auth("").pending()


@pytest.mark.unit
class TestFinish:
    def test_a_signed_in_account_has_its_tokens_on_a_private_file(self, token_dir, dropbox):
        a = auth()
        sign_in = a.begin()
        dropbox.answer_token(token_grant(access_token="at.1", refresh_token="rt.1", expires_in=14400))

        assert a.finish(redirected(sign_in)) == ""

        assert a.signed_in() and not a.pending()
        assert stat.S_IMODE(token_path().stat().st_mode) == 0o600
        tokens = on_file()
        assert tokens["refresh_token"] == "rt.1" and tokens["access_token"] == "at.1"
        assert time.time() + 14000 < tokens["expiry"] <= time.time() + 14400
        assert a.token() == "at.1" and len(dropbox.calls) == 1, "a fresh token is used without a refresh"

    def test_a_paste_with_another_flows_state_is_refused_and_nothing_exchanged(self, token_dir, dropbox):
        a = auth()
        sign_in = a.begin()

        problem = a.finish(redirected(sign_in, state="someone-elses"))

        assert "another sign-in" in problem
        assert not dropbox.calls and not a.signed_in() and a.pending()

    def test_a_bare_code_is_refused(self, token_dir, dropbox):
        a = auth()
        a.begin()
        assert "whole address" in a.finish("c0de")
        assert not dropbox.calls

    def test_without_a_flow_the_paste_says_to_start_one(self, token_dir, dropbox):
        problem = auth().finish(f"{REDIRECT_URI}?code=c0de&state=x")
        assert "dropbox_login()" in problem and not dropbox.calls

    def test_an_expired_code_and_an_unknown_app_key_are_told_apart(self, token_dir, dropbox):
        a = auth()
        sign_in = a.begin()
        dropbox.answer_token({"error": "invalid_grant", "error_description": "code has expired"}, status=400)
        assert "expired" in a.finish(redirected(sign_in)) and a.pending()

        dropbox.answer_token({"error": "invalid_client", "error_description": "Invalid client_id"}, status=400)
        assert "dropbox.app_key" in a.finish(redirected(sign_in))
        assert not a.signed_in()

    def test_a_grant_short_of_a_file_scope_is_refused_naming_the_scope_and_the_tab(self, token_dir, dropbox):
        a = auth()
        sign_in = a.begin()
        dropbox.answer_token({**token_grant(), "scope": "account_info.read"})

        problem = a.finish(redirected(sign_in))

        assert "files.metadata.read" in problem and "files.content.read" in problem
        assert "Permissions" in problem and "Submit" in problem and "dropbox_login()" in problem
        assert not a.signed_in() and not token_path().exists(), "a token that fails every file call is not kept"
        assert not a.pending(), "the code is spent: the user fixes the app, then starts anew"
        assert a.missing_scopes() == []

    def test_a_grant_that_names_no_scopes_is_taken_as_whole(self, token_dir, dropbox):
        a = auth()
        sign_in = a.begin()
        dropbox.answer_token({**token_grant(), "scope": ""})
        assert a.finish(redirected(sign_in)) == "" and a.signed_in()

    def test_a_sign_in_on_file_from_before_the_check_shows_its_missing_scopes(self, token_dir):
        a = auth()
        a._save({**token_file(), "scope": "account_info.read files.metadata.read"})
        assert a.missing_scopes() == ["files.content.read"]
        a._save(token_file())
        assert a.missing_scopes() == [], "an old file without the field is not called short"

    def test_a_grant_without_a_refresh_token_is_not_a_sign_in(self, token_dir, dropbox):
        a = auth()
        sign_in = a.begin()
        dropbox.answer_token(token_grant(refresh_token=None))
        assert "refresh token" in a.finish(redirected(sign_in))
        assert not a.signed_in()

    def test_the_flow_survives_a_rebuilt_auth_but_not_a_changed_app_key(self, token_dir, dropbox):
        sign_in = auth().begin()
        rebuilt = auth()
        assert rebuilt.pending()

        changed = auth("other-key")
        problem = changed.finish(redirected(sign_in))
        assert "app key changed" in problem and not changed.pending() and not dropbox.calls


@pytest.mark.unit
class TestToken:
    def test_near_expiry_the_token_is_refreshed_with_the_app_key_alone(self, token_dir, dropbox):
        a = auth()
        a._save(token_file(expires_in=30))
        dropbox.answer_token(token_grant(access_token="at.2", refresh_token=None))

        assert a.token() == "at.2"

        refresh = dropbox.calls[-1].data
        assert refresh == {"grant_type": "refresh_token", "refresh_token": "rt.old", "client_id": "app-key"}
        assert on_file()["access_token"] == "at.2" and on_file()["refresh_token"] == "rt.old"

    def test_a_refused_refresh_keeps_the_file_and_asks_for_a_new_sign_in_until_one_succeeds(self, token_dir, dropbox):
        a = auth()
        a._save(token_file(expires_in=0))
        dropbox.answer_token({"error": "invalid_grant", "error_description": "refresh token is invalid or revoked"}, status=400)

        with pytest.raises(NotSignedIn, match="dropbox_login"):
            a.token()
        assert a.needs_sign_in_again() and token_path().exists()
        with pytest.raises(NotSignedIn):
            a.token()
        assert len(dropbox.calls) == 1, "the refused refresh is not retried on every call"

        sign_in = a.begin()
        dropbox.answer_token(token_grant(access_token="at.3"))
        assert a.finish(redirected(sign_in)) == ""
        assert not a.needs_sign_in_again() and a.token() == "at.3"

    def test_a_server_error_at_the_refresh_is_not_a_lapsed_sign_in(self, token_dir, dropbox):
        a = auth()
        a._save(token_file(expires_in=0))
        dropbox.answer_token({}, status=503)
        with pytest.raises(NotSignedIn):
            a.token()
        assert not a.needs_sign_in_again()

    def test_an_unreachable_endpoint_is_reported_not_raised_bare(self, token_dir, dropbox):
        a = auth()
        a._save(token_file(expires_in=0))
        with patch(POST, side_effect=requests.ConnectionError("down")):
            with pytest.raises(NotSignedIn, match="could not be reached"):
                a.token()

    def test_never_signed_in_says_so(self, token_dir):
        with pytest.raises(NotSignedIn, match="not signed in"):
            auth().token()


@pytest.mark.unit
class TestSignOut:
    def test_sign_out_deletes_the_file_and_drops_the_flow(self, token_dir, dropbox):
        a = auth()
        a._save(token_file())
        a.begin()
        a.sign_out()
        assert not token_path().exists() and not a.signed_in() and not a.pending()

    def test_delete_token_file_reports_whether_there_was_one(self, token_dir):
        assert delete_token_file() is False
        auth()._save(token_file())
        assert delete_token_file() is True and not token_path().exists()
