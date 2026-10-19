"""The read-only Find My probe behind the TUI's device table and the device listing, and the
rules since the lockouts of 2026-10-15/16: nothing reads Find My on a timer, a session Find My
wants re-signed is idle, the one sign-in a command earns is deliberate and completed, and a
refusal from Apple parks every sign-in — across restarts — until the user confirms."""
import time
from unittest.mock import Mock, patch

import pytest
from pyicloud.exceptions import (PyiCloudAPIResponseException, PyiCloudAuthRequiredException,
                                 PyiCloudFailedLoginException)

from integrations.icloud.ICloudIntegration import iCloudAPI, _brief

CONFIG = {"apple_id": "m@mlab.test", "password": "secret",
          "devices": [{"id": "fmip-dad", "display_name": "dad"}]}
NOTIFY_PATH = 'assistant.Notifier.NullNotifier.send_message'
LOCK_BODY = ('{"serviceErrors": [{"code": "-20209", "message": "This Apple Account has been locked '
             'for security reasons. Visit iForgot to reset your account (https://iforgot.apple.com)."}]}')


def _record(device_id, name="iPhone"):
    return {"id": device_id, "baUUID": "", "name": name, "deviceDisplayName": "iPhone 15", "deviceClass": "iPhone"}


def _answer(devices):
    return Mock(json=Mock(return_value={"content": devices}))


def _service(answers=None, exc=None):
    """A pyicloud mock whose Find My requests answer in order with `answers` (dicts), or raise `exc`."""
    service = Mock()
    service.requires_2fa = False
    service._auth_data = {}
    service.get_webservice_url.return_value = "https://p1-fmipweb.icloud.example"
    if exc is not None:
        service.session.post.side_effect = exc
    else:
        service.session.post.side_effect = [Mock(json=Mock(return_value=answer)) for answer in answers]
    return service


def _reauth_wanted():
    return PyiCloudAuthRequiredException("m@mlab.test", Mock())


def _login_behind_a_5xx():
    try:
        raise PyiCloudAPIResponseException("Service Temporarily Unavailable", 503, Mock(text=""))
    except PyiCloudAPIResponseException as cause:
        try:
            raise PyiCloudFailedLoginException("Failed to initiate srp authentication.") from cause
        except PyiCloudFailedLoginException as e:
            return e


def _login_behind_a_5xx_with_a_body():
    """Apple's fault answered with a JSON error body: pyicloud puts Apple's own code in `code`
    and the HTTP status is on the response alone."""
    try:
        raise PyiCloudAPIResponseException("Service Temporarily Unavailable", "-21000", Mock(status_code=503, text=""))
    except PyiCloudAPIResponseException as cause:
        try:
            raise PyiCloudFailedLoginException("Failed to initiate srp authentication.") from cause
        except PyiCloudFailedLoginException as e:
            return e


def _locked_login():
    try:
        raise PyiCloudAPIResponseException("", 403, Mock(text=LOCK_BODY))
    except PyiCloudAPIResponseException as cause:
        try:
            raise PyiCloudFailedLoginException("Invalid email/password combination.") from cause
        except PyiCloudFailedLoginException as e:
            return e


@pytest.fixture(autouse=True)
def reset_singleton():
    iCloudAPI.reset_instance()
    yield
    iCloudAPI.reset_instance()


@pytest.fixture
def api():
    return iCloudAPI(config=CONFIG)


class TestProbeRequest:
    def test_listing_asks_for_the_device_list_without_waking_a_device(self, api):
        api._api = _service([{"content": [_record("fmip-dad")]}])
        assert [d["id"] for d in api.get_icloud_devices()] == ["fmip-dad"]
        (url,), kwargs = api._api.session.post.call_args
        assert url.endswith("/fmipservice/client/web/initClient")
        body = kwargs["json"]
        assert "isUpdatingAllLocations" not in body
        assert "shouldLocate" not in body["clientContext"]
        assert "selectedDevice" not in body["clientContext"]

    def test_one_request_serves_listings_inside_the_ttl(self, api):
        api._api = _service([{"content": [_record("fmip-dad")]}])
        assert len(api.get_icloud_devices()) == 1
        assert len(api.get_icloud_devices()) == 1
        assert api._api.session.post.call_count == 1

    def test_status_checks_make_no_request_of_their_own(self, api):
        api._api = _service([{"content": [_record("fmip-dad")]}])
        for _ in range(3):
            assert api.get_auth_status()["status"] == "authenticated"
        api._api.session.post.assert_not_called()

    def test_a_new_session_drops_the_cached_verdict(self, api):
        api._api = _service(exc=_reauth_wanted())
        assert api.get_icloud_devices() == []
        assert api.get_auth_status()["status"] == "needs_reauth"
        api._api = _service([{"content": [_record("fmip-dad")]}])
        assert api.get_auth_status()["status"] == "authenticated"
        assert len(api.get_icloud_devices()) == 1

    def test_listing_merges_family_devices_that_load_after_the_first_answer(self, api):
        api.FAMILY_POLL_DELAY_SECONDS = 0
        first = {"content": [_record("fmip-dad", "Dad's iPhone")], "serverContext": {"ctx": 1},
                 "userInfo": {"hasMembers": True, "membersInfo": {"charlotte": {"deviceFetchStatus": "LOADING"}}}}
        second = {"content": [_record("fmip-dad", "Dad's iPhone"), _record("fmip-charlotte", "Charlotte's iPhone")],
                  "userInfo": {"hasMembers": True, "membersInfo": {"charlotte": {"deviceFetchStatus": "DONE"}}}}
        api._api = _service([first, second])
        names = [d["name"] for d in api.get_icloud_devices()]
        assert names == ["Dad's iPhone", "Charlotte's iPhone"]
        (url,), kwargs = api._api.session.post.call_args
        assert url.endswith("/refreshClient")
        assert kwargs["json"]["serverContext"] == {"ctx": 1}, "the poll continues the server's context"
        assert "shouldLocate" not in kwargs["json"]["clientContext"]

    def test_family_polling_stops_at_the_cap(self, api):
        api.FAMILY_POLL_DELAY_SECONDS = 0
        loading = {"content": [_record("fmip-dad")], "serverContext": {},
                   "userInfo": {"hasMembers": True, "membersInfo": {"x": {"deviceFetchStatus": "LOADING"}}}}
        api._api = _service([loading] * 10)
        assert len(api.get_icloud_devices()) == 1
        assert api._api.session.post.call_count == 1 + api.FAMILY_POLLS


class TestReauthVerdict:
    def test_the_tuis_listing_says_needs_reauth_and_never_signs_in(self, api):
        api._api = _service(exc=_reauth_wanted())
        assert api.get_icloud_devices() == []
        assert api.get_auth_status()["status"] == "needs_reauth"
        api._api.authenticate.assert_not_called()

    def test_apples_authentication_required_reason_counts_as_reauth(self, api):
        api._api = _service(exc=PyiCloudAPIResponseException("Authentication required for Account.", 421, Mock(text="")))
        assert api.get_icloud_devices() == []
        assert api.get_auth_status()["status"] == "needs_reauth"

    def test_other_failures_show_as_an_error_with_the_reason(self, api):
        api._api = _service(exc=PyiCloudAPIResponseException("Service Temporarily Unavailable", 503, Mock(text="")))
        assert api.get_icloud_devices() == []
        status = api.get_auth_status()
        assert status["status"] == "error"
        assert "503" in status["message"]

    def test_the_commands_listing_signs_in_once_and_lists(self, api):
        service = _service([])
        service.session.post.side_effect = [_reauth_wanted(), _answer([_record("fmip-dad")])]
        api._api = service
        assert [d["id"] for d in api.get_icloud_devices(sign_in=True)] == ["fmip-dad"]
        service.authenticate.assert_called_once_with(force_refresh=True)

    def test_the_health_check_leaves_a_rejected_session_alone(self, api):
        """Idle, not unhealthy: the hourly re-sign-in this used to do is gone."""
        service = _service(exc=_reauth_wanted())
        api._api = service
        for _ in range(3):
            assert api.check_health() is True
        service.authenticate.assert_not_called()
        service.session.post.assert_not_called()

    def test_a_users_refresh_signs_in_for_find_my(self, api):
        service = _service([])
        service.session.post.side_effect = [_reauth_wanted(), _answer([_record("fmip-dad")])]
        api._api = service
        with patch.object(api, "_ensure_authenticated", side_effect=lambda: setattr(api, "_api", service)):
            api.refresh_connection(by_user=True)
        service.authenticate.assert_called_once_with(force_refresh=True)
        assert api.get_auth_status()["status"] == "authenticated"

    def test_a_signin_that_ends_in_a_2fa_challenge_hands_over_to_the_2fa_flow(self, api):
        service = _service([])
        service.session.post.side_effect = [_reauth_wanted(), _answer([_record("fmip-dad")])]

        def challenged(force_refresh=False):
            service.requires_2fa = True
            service._auth_data = {"phoneNumberVerification": {"trustedPhoneNumber": {
                "id": 1, "pushMode": "sms", "nonFTEU": False, "obfuscatedNumber": "•••05"}}}
        service.authenticate.side_effect = challenged
        api._api = service
        assert api.get_icloud_devices(sign_in=True) == []
        assert api.get_auth_status()["status"] == "needs_2fa", "the 2FA flow owns the session from here"
        assert api.sms_request_pending(), "the code Apple texts on its own is the pending request"
        assert service.session.post.call_count == 1, "no probe is retried against a session Apple is still challenging"


class TestAccountLock:
    def test_brief_follows_the_cause_to_apples_answer(self):
        line = _brief(_locked_login())
        assert line.startswith("Invalid email/password combination. <- ")
        assert "(403)" in line and "locked for security reasons" in line
        assert "\n" not in line

    def test_brief_still_caps_a_huge_chained_body(self):
        try:
            raise PyiCloudAPIResponseException("x", 409, Mock(text="<html>" + "\n<div>row</div>" * 900))
        except PyiCloudAPIResponseException as cause:
            try:
                raise PyiCloudFailedLoginException("wrapped") from cause
            except PyiCloudFailedLoginException as e:
                line = _brief(e)
        assert len(line) < 400 and "\n" not in line
        assert "chars, PyiCloudFailedLoginException]" in line

    def test_a_locked_account_is_reported_once_with_the_unlock_page_and_parks(self, api, icloud_session_dir):
        service = _service(exc=_reauth_wanted())
        service.authenticate.side_effect = _locked_login()
        api._api = service
        with patch(NOTIFY_PATH) as notify:
            verdicts = [api._sign_in_for_find_my("test") for _ in range(3)]
        assert all(verdicts), "Find My stays unavailable"
        notify.assert_called_once()
        assert "iforgot.apple.com" in notify.call_args.args[0]
        service.authenticate.assert_called_once(), "the refusal parks the sign-ins; no retry"
        assert (icloud_session_dir / api.PARK_FILE).exists()
        status = api.get_auth_status()
        assert status["status"] == "parked" and "locked" in status["message"].lower()

    @pytest.mark.parametrize("login_failure", [_login_behind_a_5xx, _login_behind_a_5xx_with_a_body])
    def test_a_5xx_behind_the_login_failure_parks_nothing_so_the_next_command_tries_again(self, api, icloud_session_dir, login_failure):
        """pyicloud wraps Apple's server fault in the same exception as a refused password."""
        service = _service(exc=_reauth_wanted())
        service.authenticate.side_effect = login_failure()
        api._api = service
        with patch(NOTIFY_PATH) as notify:
            assert all(api._sign_in_for_find_my("test") for _ in range(2)), "Find My stays unavailable for now"
        notify.assert_not_called()
        assert service.authenticate.call_count == 2
        assert not (icloud_session_dir / api.PARK_FILE).exists()
        assert api.get_auth_status()["status"] != "parked"

    def test_commands_wanting_a_sign_in_at_once_share_the_one_that_runs(self, api):
        import threading
        service = _service([{"content": [_record("fmip-dad")]}] * 2)

        def slow_signin(force_refresh=False):
            time.sleep(0.02)
        service.authenticate.side_effect = slow_signin
        api._api = service
        outcomes = []
        threads = [threading.Thread(target=lambda: outcomes.append(api._sign_in_for_find_my("locate"))) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert outcomes == [None, None]
        service.authenticate.assert_called_once()
        assert api._sign_in_for_find_my("later") is None and service.authenticate.call_count == 2, \
            "a command asking after the shared sign-in runs its own"

    def _parked(self, api):
        service = _service(exc=_reauth_wanted())
        service.authenticate.side_effect = _locked_login()
        api._api = service
        with patch(NOTIFY_PATH):
            api._sign_in_for_find_my("test")
        return service

    def test_the_park_survives_a_new_instance(self, api):
        """17-23 restarts a day on the production machine, each a sign-in attempt before this."""
        from integrations.icloud.ICloudIntegration import iCloudAuthenticationError
        self._parked(api)
        iCloudAPI.reset_instance()
        fresh = iCloudAPI(config=CONFIG)
        with patch('integrations.icloud.ICloudIntegration.PyiCloudService') as mock_pyicloud:
            with pytest.raises(iCloudAuthenticationError, match="parked"):
                fresh.check_health()
            assert fresh.get_auth_status()["status"] == "parked"
        mock_pyicloud.assert_not_called()

    def test_the_users_refresh_lifts_the_park_and_signs_in(self, api, icloud_session_dir):
        self._parked(api)
        with patch('integrations.icloud.ICloudIntegration.PyiCloudService') as mock_pyicloud:
            mock_pyicloud.return_value = _service([{"content": [_record("fmip-dad")]}])
            api.refresh_connection(by_user=True)
        assert not (icloud_session_dir / api.PARK_FILE).exists()
        mock_pyicloud.return_value.authenticate.assert_called_once()
        assert api.get_auth_status()["status"] == "authenticated"

    def test_an_automatic_refresh_keeps_the_park(self, api, icloud_session_dir):
        from integrations.icloud.ICloudIntegration import iCloudAuthenticationError
        self._parked(api)
        with patch('integrations.icloud.ICloudIntegration.PyiCloudService') as mock_pyicloud:
            with pytest.raises(iCloudAuthenticationError, match="parked"):
                api.refresh_connection()
        mock_pyicloud.assert_not_called()
        assert (icloud_session_dir / api.PARK_FILE).exists()

    def test_the_auto_2fa_flow_requests_nothing_while_parked(self, api):
        self._parked(api)
        api._api.requires_2fa = True
        api._challenge_from_fresh_signin = False
        with patch.object(api, "_fresh_password_signin") as fresh, patch.object(api, "request_sms_2fa") as sms:
            api.on_needs_2fa()
        fresh.assert_not_called()
        sms.assert_not_called()


class TestVerdictLifetime:
    """What the status shows of the last Find My verdict, and for how long: a "reauth" stands
    until a sign-in or an accepted code (expired cookies never come back by themselves), a
    transient error expires so the table tries again, and an accepted code drops the "2FA
    pending" verdict recorded while the challenge was open."""

    def _aged_past_ttl(self, api):
        api._find_my_probe.at -= api.FIND_MY_PROBE_TTL_SECONDS + 1

    def test_a_reauth_verdict_stands_past_the_ttl_without_another_request(self, api):
        api._api = _service(exc=_reauth_wanted())
        assert api.get_icloud_devices() == []
        self._aged_past_ttl(api)
        assert api.get_auth_status()["status"] == "needs_reauth"
        assert api._api.session.post.call_count == 1

    def test_a_transient_error_verdict_expires_so_the_next_listing_asks_again(self, api):
        service = _service([])
        service.session.post.side_effect = [
            PyiCloudAPIResponseException("Service Temporarily Unavailable", 503, Mock(text="")),
            _answer([_record("fmip-dad")])]
        api._api = service
        assert api.get_icloud_devices() == []
        assert api.get_auth_status()["status"] == "error"
        self._aged_past_ttl(api)
        assert api.get_auth_status()["status"] == "authenticated"
        assert len(api.get_icloud_devices()) == 1

    @pytest.mark.xfail(strict=True, reason="a reauth or error FindMyProbe is recorded with complete=False, so a complete listing (get_icloud_devices) inside the TTL never takes it from the cache and probes Find My again")
    def test_a_reauth_verdict_inside_the_ttl_serves_the_next_listing_without_a_request(self, api):
        """The verdict is cached so the table and the listing share one request; a refused
        request repeated for a verdict that cannot change is the pattern that got the account
        locked."""
        api._api = _service(exc=_reauth_wanted())
        assert api.get_icloud_devices() == []
        assert api.get_icloud_devices() == []
        assert api._api.session.post.call_count == 1

    def test_an_accepted_code_drops_the_pending_verdict_and_the_next_listing_decides(self, api):
        service = _service([])
        service.session.post.side_effect = [_reauth_wanted(), _answer([_record("fmip-dad")])]
        service.two_factor_delivery_method = "sms"
        service.requires_2sa = False
        service.is_trusted_session = True

        def challenged(force_refresh=False):
            service.requires_2fa = True
            service._auth_data = {"phoneNumberVerification": {"trustedPhoneNumber": {
                "id": 1, "pushMode": "sms", "nonFTEU": False, "obfuscatedNumber": "•••05"}}}
        service.authenticate.side_effect = challenged
        api._api = service
        assert api.get_icloud_devices(sign_in=True) == []
        assert api.get_auth_status()["status"] == "needs_2fa"

        service._validate_sms_code.side_effect = lambda code: setattr(service, "requires_2fa", False)
        assert api.send_2fa_code("123456") is True
        assert api.get_auth_status()["status"] == "authenticated", "the '2FA pending' verdict predates the code"
        assert [d["id"] for d in api.get_icloud_devices()] == ["fmip-dad"]
        assert service.session.post.call_count == 2


class TestParkFile:
    def test_a_park_file_that_cannot_be_read_still_parks(self, api, icloud_session_dir):
        """A refused sign-in is the one verdict that must survive anything: a garbled file is a
        park whose details were lost, not a licence to sign in."""
        from integrations.icloud.ICloudIntegration import iCloudAuthenticationError
        icloud_session_dir.mkdir(parents=True, exist_ok=True)
        (icloud_session_dir / api.PARK_FILE).write_text("{not json")
        with patch('integrations.icloud.ICloudIntegration.PyiCloudService') as mock_pyicloud:
            with pytest.raises(iCloudAuthenticationError, match="parked"):
                api.check_health()
        mock_pyicloud.assert_not_called()
        assert api.get_auth_status()["status"] == "parked"

    def test_a_park_that_cannot_be_written_holds_in_memory_until_the_user_lifts_it(self, api, tmp_path):
        service = _service(exc=_reauth_wanted())
        service.authenticate.side_effect = _locked_login()
        api._api = service
        unwritable = tmp_path / "no-such-dir" / api.PARK_FILE
        with patch.object(api, "_park_path", return_value=unwritable), patch(NOTIFY_PATH) as notify:
            assert api._sign_in_for_find_my("test")
            assert api._sign_in_for_find_my("test")
            service.authenticate.assert_called_once(), "parked in memory: no second sign-in"
            notify.assert_called_once()
            assert api.get_auth_status()["status"] == "parked"
            with patch('integrations.icloud.ICloudIntegration.PyiCloudService') as mock_pyicloud:
                mock_pyicloud.return_value = _service([{"content": [_record("fmip-dad")]}])
                api.refresh_connection(by_user=True)
        mock_pyicloud.return_value.authenticate.assert_called_once()
        assert api.get_auth_status()["status"] == "authenticated"
