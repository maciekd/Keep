"""MicrosoftAuth: the device-code flow in two steps over a faked MSAL application, and the
token cache file it keeps — written 0600, read for the signed-in state without the network,
deleted on sign-out and when the integration loses its config."""
import json
import os
import stat
import time
from unittest.mock import patch

from http.client import RemoteDisconnected

import pytest
import requests

from integrations.microsoft import MicrosoftAuth as auth_module
from integrations.microsoft.GraphClient import GraphClient
from integrations.microsoft.MicrosoftAuth import MicrosoftAuth, NotSignedIn, delete_all_token_files, token_path

MSAL_APP = "integrations.microsoft.MicrosoftAuth.msal.PublicClientApplication"
EXPIRES = int(time.time()) + 3000


def cache_state(username="marcus@example.com", expires_on=EXPIRES) -> str:
    """A serialized MSAL cache holding one account and one access token."""
    return json.dumps({
        "Account": {"uid.utid-login.microsoftonline.com-common": {
            "home_account_id": "uid.utid", "environment": "login.microsoftonline.com", "realm": "common",
            "local_account_id": "uid", "username": username, "authority_type": "MSSTS"}},
        "AccessToken": {"uid.utid-login.microsoftonline.com-accesstoken-client-common-scopes": {
            "credential_type": "AccessToken", "secret": "at", "home_account_id": "uid.utid",
            "environment": "login.microsoftonline.com", "client_id": "client", "target": "Mail.Read",
            "realm": "common", "expires_on": str(expires_on), "cached_at": str(int(time.time()))}}})


class FakeApp:
    """What MSAL does with the cache it was handed: the device flow writes an account into it."""

    def __init__(self, client_id, authority=None, token_cache=None, timeout=None):
        self.client_id, self.authority, self.cache, self.timeout = client_id, authority, token_cache, timeout
        self.flow = {"user_code": "ABCD-1234", "verification_uri": "https://microsoft.com/devicelogin",
                     "expires_at": time.time() + 900, "device_code": "dc", "message": "..."}
        self.finish_result = {"access_token": "at"}

    def initiate_device_flow(self, scopes=None):
        self.scopes = scopes
        return dict(self.flow)

    def acquire_token_by_device_flow(self, flow):
        if "access_token" in self.finish_result:
            self.cache.deserialize(cache_state())
            self.cache.has_state_changed = True
        return self.finish_result

    def get_accounts(self, username=None):
        return list(self.cache.search("Account"))

    def acquire_token_silent(self, scopes, account=None):
        return {"access_token": "fresh-token"}

    def remove_account(self, account):
        self.cache.deserialize("{}")
        self.cache.has_state_changed = True


@pytest.fixture
def token_dir(tmp_path):
    with patch.object(auth_module, "TOKEN_DIR", tmp_path / "oauth"):
        yield tmp_path / "oauth"


def auth(name="Work", client_id="client-1", tenant="common") -> MicrosoftAuth:
    return MicrosoftAuth(name, lambda: (client_id, tenant))


@pytest.mark.unit
class TestDeviceFlow:
    def test_begin_hands_over_the_code_and_the_url_and_finish_leaves_a_0600_token_file(self, token_dir):
        with patch(MSAL_APP, FakeApp):
            a = auth()
            assert not a.signed_in()
            flow = a.begin()
            assert (flow.user_code, flow.verification_uri) == ("ABCD-1234", "https://microsoft.com/devicelogin")
            assert "ABCD-1234" in flow.instructions and "devicelogin" in flow.instructions
            assert a.finish(flow) == ""
            assert a.signed_in() and a.signed_in_as() == "marcus@example.com"
            assert a.token() == "fresh-token"
            assert a._application.authority == "https://login.microsoftonline.com/common"
            assert "offline_access" not in a._application.scopes and "Mail.Send" in a._application.scopes
        path = token_path("Work")
        assert path == token_dir / "microsoft-work.json" and path.exists()
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert int(a.expires_at().timestamp()) == EXPIRES

    def test_a_flow_that_expires_says_so(self, token_dir):
        with patch(MSAL_APP, FakeApp):
            a = auth()
            flow = a.begin()
            a._application.finish_result = {"error": "expired_token", "error_description": "AADSTS70020: The provided value for the input parameter 'device_code' is not valid. The code has expired."}
            assert a.finish(flow) == "the code expired before it was entered"
            assert not a.signed_in() and not token_path("Work").exists()

    def test_a_registration_entra_refuses_cannot_start(self, token_dir):
        with patch(MSAL_APP, FakeApp):
            a = auth()
            a._app().flow = {"error": "invalid_client", "error_description": "AADSTS7000218: client assertion needed"}
            with pytest.raises(RuntimeError) as raised:
                a.begin()
        assert "AADSTS7000218" in str(raised.value)

    def test_without_a_client_id_nothing_is_built(self, token_dir):
        with patch(MSAL_APP, FakeApp) as app:
            a = auth(client_id="")
            assert not a.signed_in()
            with pytest.raises(NotSignedIn):
                a.begin()


@pytest.mark.unit
class TestTokenFile:
    def test_a_token_file_from_an_earlier_run_means_signed_in_without_touching_msal(self, token_dir):
        token_dir.mkdir()
        token_path("Work").write_text(cache_state())
        with patch(MSAL_APP, side_effect=AssertionError("built the application")):
            a = auth()
            assert a.signed_in() and a.signed_in_as() == "marcus@example.com"

    def test_without_a_token_asking_for_one_says_to_sign_in(self, token_dir):
        with patch(MSAL_APP, FakeApp):
            with pytest.raises(NotSignedIn) as raised:
                auth().token()
        assert 'microsoft_login("Work")' in str(raised.value)

    def test_sign_out_and_losing_the_config_delete_the_files(self, token_dir):
        token_dir.mkdir()
        token_path("Work").write_text(cache_state())
        token_path("Charlotte's").write_text(cache_state("charlotte@example.com"))
        with patch(MSAL_APP, FakeApp):
            a = auth()
            a.sign_out()
            assert not a.signed_in() and not token_path("Work").exists()
        assert token_path("Charlotte's").name == "microsoft-charlotte-s.json" and token_path("Charlotte's").exists()
        assert delete_all_token_files() == 1 and list(token_dir.iterdir()) == []
        assert delete_all_token_files() == 0

    def test_a_silent_refresh_that_fails_is_a_not_signed_in(self, token_dir):
        token_dir.mkdir()
        token_path("Work").write_text(cache_state())
        app = FakeApp("client-1")
        with patch(MSAL_APP, lambda *a, **k: (setattr(app, "cache", k["token_cache"]) or app)):
            app.acquire_token_silent = lambda scopes, account=None: {"error": "interaction_required", "error_description": "consent revoked"}
            with pytest.raises(NotSignedIn) as raised:
                auth().token()
        assert "expired" in str(raised.value) and "microsoft_login" in str(raised.value)


class RefreshingApp(FakeApp):
    """A silent refresh that MSAL wrote a new access token into the cache for."""
    renewed = EXPIRES + 1800

    def acquire_token_silent(self, scopes, account=None):
        self.cache.deserialize(cache_state(expires_on=self.renewed))
        self.cache.has_state_changed = True
        return {"access_token": "renewed-token"}


@pytest.mark.unit
class TestCacheAndApplication:
    def test_a_silent_refresh_that_changed_the_cache_is_written_0600_and_an_unchanged_one_is_not(self, token_dir):
        token_dir.mkdir()
        path = token_path("Work")
        path.write_text(cache_state())
        os.chmod(path, 0o644)
        with patch(MSAL_APP, RefreshingApp):
            a = auth()
            assert a.token() == "renewed-token"
            assert stat.S_IMODE(path.stat().st_mode) == 0o600
            # the file holds the renewed token: a fresh instance reads its expiry without MSAL
            with patch(MSAL_APP, side_effect=AssertionError("built the application")):
                assert int(auth().expires_at().timestamp()) == RefreshingApp.renewed

            a._application.acquire_token_silent = lambda scopes, account=None: {"access_token": "same-token"}
            os.utime(path, ns=(1_000_000_000, 1_000_000_000))
            assert a.token() == "same-token"
        assert path.stat().st_mtime_ns == 1_000_000_000

    def test_the_application_follows_the_registration_and_an_empty_tenant_is_common(self, token_dir):
        settings = {"app": ("client-1", "")}
        built = []

        def build(client_id, authority=None, token_cache=None, timeout=None):
            built.append(FakeApp(client_id, authority, token_cache, timeout))
            return built[-1]

        with patch(MSAL_APP, build):
            a = MicrosoftAuth("Work", lambda: settings["app"])
            first = a._app()
            assert first.authority.endswith("/common") and a._app() is first
            settings["app"] = ("client-1", "contoso.example")
            second = a._app()
            assert second is not first and second.authority.endswith("/contoso.example")
            settings["app"] = ("client-2", "contoso.example")
            third = a._app()
            assert third is not second and third.client_id == "client-2" and a._app() is third
        assert len(built) == 3 and all(app.cache is a._cache for app in built)
        assert all(app.timeout and app.timeout > 0 for app in built)

    def test_a_declined_consent_carries_entras_description(self, token_dir):
        with patch(MSAL_APP, FakeApp):
            a = auth()
            flow = a.begin()
            a._application.finish_result = {"error": "authorization_declined",
                                            "error_description": "AADSTS65004: User declined to consent to access the app."}
            outcome = a.finish(flow)
            assert "AADSTS65004" in outcome and "expired" not in outcome.lower()
            assert not a.signed_in() and not token_path("Work").exists()


class DroppingApp(FakeApp):
    """A refresh POST that lands on a keep-alive connection the identity platform has closed, `drops` times."""
    drops = 1

    def acquire_token_silent(self, scopes, account=None):
        self.attempts = getattr(self, "attempts", 0) + 1
        if self.attempts <= self.drops:
            raise requests.ConnectionError(("Connection aborted.", RemoteDisconnected("Remote end closed connection without response")))
        return {"access_token": "fresh-token"}


@pytest.mark.unit
class TestDroppedRefresh:
    @pytest.fixture
    def signed_in(self, token_dir):
        token_dir.mkdir()
        token_path("Jobb").write_text(cache_state())

    def test_a_refresh_whose_connection_drops_is_asked_once_more_and_the_fetch_succeeds(self, signed_in):
        sent = []

        def graph(method, url, params=None, headers=None, json=None, timeout=None):
            sent.append(headers["Authorization"])
            return type("Response", (), {"status_code": 200, "content": b"{}", "json": lambda self: {"value": []}})()

        with patch(MSAL_APP, DroppingApp), patch.object(auth_module, "info") as logged, \
                patch("integrations.microsoft.GraphClient.requests.request", graph):
            a = auth("Jobb")
            assert GraphClient(a).get("/me/calendars") == {"value": []}
        assert a._application.attempts == 2 and sent == ["Bearer fresh-token"]
        assert any("Jobb" in c.args[0] and "RemoteDisconnected" in c.args[0] for c in logged.call_args_list)

    def test_a_second_drop_is_raised(self, signed_in):
        with patch(MSAL_APP, type("TwiceDropping", (DroppingApp,), {"drops": 2})):
            a = auth("Jobb")
            with pytest.raises(requests.ConnectionError):
                a.token()
        assert a._application.attempts == 2
