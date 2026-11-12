import json
import threading
import time

import pytest
from unittest.mock import Mock, patch, MagicMock, PropertyMock

from integrations.icloud.ICloudIntegration import iCloudAPI, iCloudAPIError, iCloudAuthenticationError, iCloudConfigurationError, ICloudIntegration
from assistant.integration.Integration import IntegrationState


VALID_CONFIG = {
    "apple_id": "test@icloud.com",
    "password": "secret",
    "devices": [
        {"id": "device-1-uuid", "display_name": "iPhone"},
        {"id": "device-2-uuid", "display_name": "iPad"},
    ]
}


def _find_my_answers(service, devices=None, exc=None):
    """Make `service`'s Find My probe (one `session.post` to initClient) answer with these
    device records, or raise `exc`."""
    service.get_webservice_url.return_value = "https://p1-fmipweb.icloud.example"
    if exc is not None:
        service.session.post.side_effect = exc
    else:
        service.session.post.return_value.json.return_value = {"content": list(devices or [])}
    return service


@pytest.fixture(autouse=True)
def reset_singleton():
    iCloudAPI.reset_instance()
    yield
    iCloudAPI.reset_instance()


class TestiCloudAPI:
    def test_starts_without_devices(self):
        """apple_id + password are enough: photos and the device listing work before any
        device is configured, and the Find My commands say what to do."""
        api = iCloudAPI(config={"apple_id": "test@icloud.com", "password": "secret", "devices": []})
        assert api._device_specs == []
        assert api.get_devices_list() == ""
        with pytest.raises(iCloudConfigurationError, match="icloud_devices\\(\\)"):
            api._select_device_spec("iPhone")
        with pytest.raises(iCloudConfigurationError, match="No iCloud devices are configured"):
            api._select_device_spec(None)

    def test_starts_without_devices_key(self):
        api = iCloudAPI(config={"apple_id": "test@icloud.com", "password": "secret"})
        assert api._device_specs == []

    def test_get_devices_list_returns_formatted_names(self):
        api = iCloudAPI(config=VALID_CONFIG)
        result = api.get_devices_list()
        assert "iPhone" in result
        assert "iPad" in result

    def test_select_device_spec_exact_match(self):
        api = iCloudAPI(config=VALID_CONFIG)
        spec = api._select_device_spec("iPhone")
        assert spec.id == "device-1-uuid"
        assert spec.display_name == "iPhone"

    def test_select_device_spec_no_match_raises(self):
        api = iCloudAPI(config=VALID_CONFIG)
        with pytest.raises(iCloudConfigurationError, match="No iCloud device matches"):
            api._select_device_spec("NonExistentDevice")

    def test_select_device_spec_none_returns_first(self):
        api = iCloudAPI(config=VALID_CONFIG)
        spec = api._select_device_spec(None)
        assert spec.id == "device-1-uuid"

    def test_get_auth_status_not_initialized(self):
        api = iCloudAPI(config=VALID_CONFIG)
        status = api.get_auth_status()
        assert status['status'] == 'not_initialized'

    def test_get_auth_status_error(self):
        api = iCloudAPI(config=VALID_CONFIG)
        api._auth_error = "Login failed"
        status = api.get_auth_status()
        assert status['status'] == 'error'
        assert "Login failed" in status['message']

    @patch('integrations.icloud.ICloudIntegration.PyiCloudService')
    def test_requires_2fa_when_the_signin_ended_on_a_challenge(self, mock_pyicloud):
        mock_pyicloud.return_value = _service_with_sms_challenge()

        api = iCloudAPI(config=VALID_CONFIG)
        assert api.requires_2fa() is True

    @patch('integrations.icloud.ICloudIntegration.PyiCloudService')
    def test_an_untrusted_token_session_is_not_a_2fa_case(self, mock_pyicloud):
        """pyicloud flags a token-restored session Apple calls untrusted as requires_2fa, but it
        holds no challenge a code could answer — and no Find My request decides it any more."""
        mock_service = Mock()
        mock_service.requires_2fa = True
        mock_service._auth_data = {}
        mock_pyicloud.return_value = mock_service

        api = iCloudAPI(config=VALID_CONFIG)
        assert api.requires_2fa() is False
        mock_service.session.post.assert_not_called()


class TestDeviceSpecs:
    def test_non_dict_entry_skipped(self):
        config = {**VALID_CONFIG, "devices": ["not-a-dict", {"id": "x", "display_name": "Phone"}]}
        api = iCloudAPI(config=config)
        specs = api._device_specs
        assert len(specs) == 1
        assert specs[0].display_name == "Phone"

    def test_entry_missing_id_skipped(self):
        config = {**VALID_CONFIG, "devices": [{"display_name": "Phone"}, {"id": "x", "display_name": "Tablet"}]}
        api = iCloudAPI(config=config)
        specs = api._device_specs
        assert len(specs) == 1
        assert specs[0].display_name == "Tablet"

    def test_entry_missing_display_name_skipped(self):
        config = {**VALID_CONFIG, "devices": [{"id": "x"}, {"id": "y", "display_name": "Tablet"}]}
        api = iCloudAPI(config=config)
        specs = api._device_specs
        assert len(specs) == 1
        assert specs[0].id == "y"

    def test_all_entries_invalid_counts_as_none_configured(self):
        config = {**VALID_CONFIG, "devices": [{"id": "x"}, {"display_name": "Phone"}]}
        api = iCloudAPI(config=config)
        assert api._device_specs == []
        with pytest.raises(iCloudConfigurationError, match="No iCloud devices are configured"):
            api._select_device_spec("Phone")


class TestEnsureAuthenticated:
    @patch('integrations.icloud.ICloudIntegration.PyiCloudService')
    def test_already_authenticated_is_noop(self, mock_pyicloud):
        api = iCloudAPI(config=VALID_CONFIG)
        api._api = Mock()
        api._ensure_authenticated()
        mock_pyicloud.assert_not_called()

    @patch('integrations.icloud.ICloudIntegration.PyiCloudService')
    def test_failed_login_sets_auth_error_and_raises(self, mock_pyicloud):
        from pyicloud.exceptions import PyiCloudFailedLoginException
        mock_pyicloud.side_effect = PyiCloudFailedLoginException("bad creds")
        api = iCloudAPI(config=VALID_CONFIG)
        with pytest.raises(iCloudAuthenticationError, match="Failed to authenticate"):
            api._ensure_authenticated()
        assert api._auth_error is not None

    @patch('integrations.icloud.ICloudIntegration.PyiCloudService')
    def test_successful_signin_resets_fresh_signin_flag(self, mock_pyicloud):
        # A normal (token/persisted-session) signin carries no live MFA
        # challenge, so a flag left over from an earlier fresh signin must be
        # cleared — otherwise _auto_request_sms trusts the restored challenge
        # and burns an SMS on a session whose codes Apple rejects.
        api = iCloudAPI(config=VALID_CONFIG)
        api._challenge_from_fresh_signin = True
        api._ensure_authenticated()
        assert api._api is mock_pyicloud.return_value
        assert api._challenge_from_fresh_signin is False

    @patch('integrations.icloud.ICloudIntegration.PyiCloudService')
    def test_successful_signin_resets_fresh_signin_budget(self, mock_pyicloud):
        # A successful login opens a fresh recovery budget
        # (MAX_FRESH_SIGNINS_PER_RECOVERY). If the counter survived it, the
        # next recovery would refuse its one allowed fresh signin and jump
        # straight to the 2h failure cooldown.
        api = iCloudAPI(config=VALID_CONFIG)
        api._fresh_signins_this_recovery = 1
        api._ensure_authenticated()
        assert api._fresh_signins_this_recovery == 0


class TestSend2faCode:
    def test_validation_succeeds_trusts_untrusted_session(self):
        api = iCloudAPI(config=VALID_CONFIG)
        mock_service = Mock()
        mock_service.validate_2fa_code.return_value = True
        mock_service.is_trusted_session = False
        api._api = mock_service

        result = api.send_2fa_code("123456")

        assert result is True
        mock_service.trust_session.assert_called_once()

    def test_validation_fails_returns_false(self):
        api = iCloudAPI(config=VALID_CONFIG)
        mock_service = Mock()
        mock_service.validate_2fa_code.return_value = False
        api._api = mock_service

        result = api.send_2fa_code("000000")

        assert result is False

    def test_stale_rejection_flag_from_earlier_attempt_is_cleared_on_entry(self):
        # A 409 stale-session flag left over from a previous attempt must not
        # leak into this one: the generic failure path below never assigns the
        # flag, so without the entry reset submit_auto_2fa_code would misroute
        # an ordinary rejection into an unnecessary fresh password signin.
        api = iCloudAPI(config=VALID_CONFIG)
        api._stale_session_rejection = True
        mock_service = Mock()
        mock_service.two_factor_delivery_method = "sms"
        mock_service._validate_sms_code.side_effect = RuntimeError("network blip")
        api._api = mock_service

        assert api.send_2fa_code("123456") is False
        assert api._stale_session_rejection is False


class TestValidateSmsCodeVerbose:
    """SMS codes bypass pyicloud's validate_2fa_code() (which swallows the HTTP
    error) and go through the verbose mirror so the logs can distinguish
    'Apple rejected the code' from 'code accepted but trust failed'."""

    def _sms_service(self):
        mock_service = Mock()
        mock_service.two_factor_delivery_method = "sms"
        return mock_service

    def test_rejected_code_returns_false_without_trusting(self):
        from pyicloud.exceptions import PyiCloudAPIResponseException
        api = iCloudAPI(config=VALID_CONFIG)
        mock_service = self._sms_service()
        mock_service._validate_sms_code.side_effect = PyiCloudAPIResponseException("wrong code", 400)
        api._api = mock_service

        assert api.send_2fa_code("000000") is False
        mock_service.trust_session.assert_not_called()
        mock_service.validate_2fa_code.assert_not_called()

    def test_accepted_code_trusts_session_and_succeeds(self):
        api = iCloudAPI(config=VALID_CONFIG)
        mock_service = self._sms_service()
        mock_service.trust_session.return_value = True
        mock_service.requires_2sa = False
        mock_service.is_trusted_session = True
        api._api = mock_service

        assert api.send_2fa_code("123456") is True
        mock_service._validate_sms_code.assert_called_once_with("123456")
        mock_service.trust_session.assert_called_once()
        mock_service.validate_2fa_code.assert_not_called()

    def test_accepted_code_with_failed_trust_reports_session_state(self):
        # The challenge is consumed even when trust fails — the result must
        # reflect requires_2sa, not pretend the code was wrong.
        api = iCloudAPI(config=VALID_CONFIG)
        mock_service = self._sms_service()
        mock_service.trust_session.return_value = False
        mock_service.requires_2sa = True
        api._api = mock_service

        assert api.send_2fa_code("123456") is False
        mock_service._validate_sms_code.assert_called_once_with("123456")


class TestSubmitAuto2faCode:
    """Auto-detected iMessage 2FA submissions (added in d6574333) must alert
    the user over Telegram. Manual TUI entry — which calls send_2fa_code
    directly — must NOT fire the alert."""

    _NOTIFY_PATH = 'assistant.Notifier.NullNotifier.send_message'
    _MONITOR_PATH = 'assistant.integration.IntegrationMonitor.IntegrationMonitor'

    def test_success_fires_telegram_notification(self):
        api = iCloudAPI(config=VALID_CONFIG)
        mock_service = Mock()
        mock_service.requires_2fa = True
        mock_service.validate_2fa_code.return_value = True
        mock_service.is_trusted_session = True
        api._api = mock_service
        api._sms_requested_at = time.time()

        with patch(self._NOTIFY_PATH) as mock_notify, patch(self._MONITOR_PATH):
            api.submit_auto_2fa_code("123456", received_at=time.time())

        mock_notify.assert_called_once()
        message = mock_notify.call_args.args[0]
        assert "iCloud" in message
        assert "auto-detected" in message

    @staticmethod
    def _api_awaiting_a_code():
        api = iCloudAPI(config=VALID_CONFIG)
        mock_service = Mock()
        mock_service.requires_2fa = True
        mock_service.validate_2fa_code.return_value = True
        mock_service.is_trusted_session = True
        api._api = mock_service
        api._sms_requested_at = time.time()
        return api

    _OPEN_PATH = 'assistant.Notifier.NullNotifier.open_conversation'
    _FTUE_PATH = 'assistant.FirstRun.in_ftue'

    def test_during_the_setup_the_verified_code_opens_the_conversation_after_the_health_check(self):
        """The user is waiting on this sign-in (the brief's iCloud step), so the assistant
        speaks first, as after the Full Disk Access grant — and the turn finds iCloud signed
        in because the health check ran before it."""
        api = self._api_awaiting_a_code()
        order = []
        with patch(self._MONITOR_PATH) as monitor, patch(self._FTUE_PATH, return_value=True), \
             patch(self._OPEN_PATH, side_effect=lambda q: order.append("turn") or True) as opened, \
             patch(NOTE_PATH) as note:
            monitor.get_instance.return_value.check_now.side_effect = lambda name: order.append("check")
            api.submit_auto_2fa_code("123456", received_at=time.time())
        assert order == ["check", "turn"]
        assert "verified" in opened.call_args.args[0] and "iCloud" in opened.call_args.args[0]
        note.assert_not_called()

    def test_a_turn_no_chat_takes_falls_back_to_the_system_note(self):
        api = self._api_awaiting_a_code()
        with patch(self._MONITOR_PATH), patch(self._FTUE_PATH, return_value=True), \
             patch(self._OPEN_PATH, return_value=False), patch(NOTE_PATH) as note:
            api.submit_auto_2fa_code("123456", received_at=time.time())
        note.assert_called_once()
        assert "verified" in note.call_args.args[1]

    def test_outside_the_setup_a_re_auth_is_a_system_note_only(self):
        api = self._api_awaiting_a_code()
        with patch(self._MONITOR_PATH), patch(self._FTUE_PATH, return_value=False), \
             patch(self._OPEN_PATH) as opened, patch(NOTE_PATH) as note:
            api.submit_auto_2fa_code("123456", received_at=time.time())
        opened.assert_not_called()
        note.assert_called_once()

    def test_validation_failure_notes_the_retry(self):
        # The LLM saw "attempting automatic SMS 2FA"; a rejected code must not leave it
        # thinking the flow stalled — one system note says a fresh code is on its way.
        api = iCloudAPI(config=VALID_CONFIG)
        mock_service = Mock()
        mock_service.requires_2fa = True
        mock_service.validate_2fa_code.return_value = False
        api._api = mock_service
        api._sms_requested_at = time.time()

        with patch(self._NOTIFY_PATH) as mock_notify, patch(self._MONITOR_PATH), \
             patch.object(api, '_auto_request_sms'):
            api.submit_auto_2fa_code("000000", received_at=time.time())

        mock_notify.assert_called_once()
        assert "rejected" in mock_notify.call_args.args[0]
        assert "fresh code" in mock_notify.call_args.args[0]

    def test_validation_failure_requests_fresh_code(self):
        # A rejected code usually means it was superseded — the flow must
        # self-heal by requesting a fresh one.
        api = iCloudAPI(config=VALID_CONFIG)
        mock_service = Mock()
        mock_service.requires_2fa = True
        mock_service.validate_2fa_code.return_value = False
        api._api = mock_service
        api._sms_requested_at = time.time()

        with patch(self._NOTIFY_PATH), patch(self._MONITOR_PATH), \
             patch.object(api, '_auto_request_sms') as mock_request:
            api.submit_auto_2fa_code("000000", received_at=time.time())

        mock_request.assert_called_once()
        assert mock_request.call_args.kwargs.get('fresh_signin', False) is False

    def test_stale_session_rejection_recovers_with_fresh_signin(self):
        # Apple's 409 'Authentication required for Account' means the session
        # behind the challenge is dead — the recovery must be a fresh password
        # signin, not another code request on the doomed session (which is what
        # SMS-bombed the phone on 2026-08-20).
        api = iCloudAPI(config=VALID_CONFIG)
        mock_service = Mock()
        mock_service.requires_2fa = True
        api._api = mock_service
        api._sms_requested_at = time.time()

        def reject_as_stale(code):
            api._stale_session_rejection = True
            return False
        with patch(self._NOTIFY_PATH), patch(self._MONITOR_PATH), \
             patch.object(api, 'send_2fa_code', side_effect=reject_as_stale), \
             patch.object(api, '_auto_request_sms') as mock_request:
            api.submit_auto_2fa_code("000000", received_at=time.time())

        mock_request.assert_called_once()
        assert mock_request.call_args.kwargs.get('fresh_signin') is True
        assert api._stale_session_rejection is False

    def test_stale_409_on_fresh_signin_challenge_does_not_retry(self):
        # 2026-08-24: every code submitted against a challenge raised by our own
        # fresh password signin got 409 'Authentication required' — 'stale
        # session' is the wrong diagnosis there, and retrying the same recipe
        # looped 4x (4 SRP signins + 4 SMS in 63s). Must give up + cool down.
        api = iCloudAPI(config=VALID_CONFIG)
        mock_service = Mock()
        mock_service.requires_2fa = True
        api._api = mock_service
        api._sms_requested_at = time.time()
        api._challenge_from_fresh_signin = True

        def reject_as_stale(code):
            api._stale_session_rejection = True
            return False
        with patch(self._NOTIFY_PATH) as mock_notify, patch(self._MONITOR_PATH), \
             patch.object(api, 'send_2fa_code', side_effect=reject_as_stale), \
             patch.object(api, '_auto_request_sms') as mock_request:
            api.submit_auto_2fa_code("000000", received_at=time.time())

        mock_request.assert_not_called()
        mock_notify.assert_called_once()
        assert api._auto_sms_last_failure_ts > 0
        assert api._stale_session_rejection is False

    def test_stale_409_logs_response_body(self):
        from pyicloud.exceptions import PyiCloudAPIResponseException
        api = iCloudAPI(config=VALID_CONFIG)
        mock_service = Mock()
        exc = PyiCloudAPIResponseException("Authentication required for Account.", 409)
        exc.response = Mock(text='{"serviceErrors":[{"code":"-20101"}]}')
        mock_service._validate_sms_code.side_effect = exc
        with patch('integrations.icloud.ICloudIntegration.error') as mock_error:
            assert api._validate_sms_code_verbose(mock_service, "123456") is False
        assert '-20101' in mock_error.call_args.args[0]

    def test_sms_validation_classifies_stale_session_409(self):
        # The verbose SMS validator must distinguish 'session is stale' (409
        # Authentication required) from 'Apple rejected this code'.
        from pyicloud.exceptions import PyiCloudAPIResponseException
        api = iCloudAPI(config=VALID_CONFIG)
        mock_service = Mock()
        mock_service._validate_sms_code.side_effect = PyiCloudAPIResponseException(
            "Authentication required for Account.", 409)
        assert api._validate_sms_code_verbose(mock_service, "123456") is False
        assert api._stale_session_rejection is True

        mock_service._validate_sms_code.side_effect = PyiCloudAPIResponseException(
            "Incorrect verification code", 400)
        assert api._validate_sms_code_verbose(mock_service, "123456") is False
        assert api._stale_session_rejection is False

    def test_not_awaiting_2fa_does_not_notify(self):
        api = iCloudAPI(config=VALID_CONFIG)
        mock_service = Mock()
        mock_service.requires_2fa = False
        api._api = mock_service

        with patch(self._NOTIFY_PATH) as mock_notify, patch(self._MONITOR_PATH):
            api.submit_auto_2fa_code("123456")

        mock_notify.assert_not_called()

    def test_no_pending_sms_request_drops_code(self):
        # Codes that arrive without a request_sms_2fa() call (signin auto-sends,
        # codes triggered from another device) must never be submitted.
        api = iCloudAPI(config=VALID_CONFIG)
        mock_service = Mock()
        mock_service.requires_2fa = True
        api._api = mock_service

        with patch(self._NOTIFY_PATH), patch(self._MONITOR_PATH):
            api.submit_auto_2fa_code("123456", received_at=time.time())

        mock_service.validate_2fa_code.assert_not_called()

    def test_code_received_before_request_drops_code(self):
        # A code that arrived before the current SMS request belongs to an
        # earlier, superseded attempt and would fail (and burn the challenge).
        api = iCloudAPI(config=VALID_CONFIG)
        mock_service = Mock()
        mock_service.requires_2fa = True
        api._api = mock_service
        api._sms_requested_at = time.time()

        with patch(self._NOTIFY_PATH), patch(self._MONITOR_PATH):
            api.submit_auto_2fa_code("123456", received_at=time.time() - 30)

        mock_service.validate_2fa_code.assert_not_called()

    def test_single_shot_per_sms_request(self):
        # After one code has been submitted for a request, further codes are
        # dropped until a new request is made — never guess twice.
        api = iCloudAPI(config=VALID_CONFIG)
        mock_service = Mock()
        mock_service.requires_2fa = True
        mock_service.validate_2fa_code.return_value = False
        api._api = mock_service
        api._sms_requested_at = time.time()

        with patch(self._NOTIFY_PATH), patch(self._MONITOR_PATH), \
             patch.object(api, '_auto_request_sms'):
            api.submit_auto_2fa_code("111111", received_at=time.time())
            api.submit_auto_2fa_code("222222", received_at=time.time())

        mock_service.validate_2fa_code.assert_called_once_with("111111")

    def test_refresh_connection_clears_pending_sms_request(self):
        # A new signin is a new auth challenge — codes from the old session are dead.
        api = iCloudAPI(config=VALID_CONFIG)
        api._api = Mock()
        api._api.requires_2fa = False
        api._sms_requested_at = time.time()

        with patch.object(api, '_ensure_authenticated'), \
             patch.object(api, 'requires_2fa', return_value=True):
            api.refresh_connection()

        assert api._sms_requested_at is None

    def test_refresh_connection_auto_requests_sms_when_still_untrusted(self):
        # The hands-off flow: refresh leaves the session untrusted → request an
        # SMS code automatically so the iMessage poller can complete 2FA.
        api = iCloudAPI(config=VALID_CONFIG)
        mock_service = Mock()
        mock_service.requires_2fa = True
        with patch.object(api, '_ensure_authenticated',
                          side_effect=lambda: setattr(api, '_api', mock_service)), \
             patch.object(api, 'requires_2fa', return_value=True), \
             patch.object(api, '_auto_request_sms') as mock_request:
            api.refresh_connection()
        mock_request.assert_called_once()

    def test_manual_send_2fa_code_does_not_notify(self):
        # The TUI calls send_2fa_code() directly when the user types the code.
        # Only the auto-detection wrapper (submit_auto_2fa_code) should alert.
        api = iCloudAPI(config=VALID_CONFIG)
        mock_service = Mock()
        mock_service.validate_2fa_code.return_value = True
        mock_service.is_trusted_session = True
        api._api = mock_service

        with patch(self._NOTIFY_PATH) as mock_notify:
            api.send_2fa_code("123456")

        mock_notify.assert_not_called()


class TestAutoSmsRequest:
    """The unattended 2FA flow: on_needs_2fa / refresh trigger an SMS request
    on their own, rate-limited, with the TUI able to see a pending request."""

    def test_on_needs_2fa_requests_code(self):
        api = iCloudAPI(config=VALID_CONFIG)
        mock_service = Mock()
        mock_service.requires_2fa = True
        api._api = mock_service
        api._challenge_from_fresh_signin = True
        with patch.object(api, 'request_sms_2fa', return_value="•••05") as mock_request, \
             patch.object(api, '_ensure_authenticated'):
            api.on_needs_2fa()
        mock_request.assert_called_once()

    def test_on_needs_2fa_waits_while_request_pending(self):
        api = iCloudAPI(config=VALID_CONFIG)
        api._api = Mock()
        api._sms_requested_at = time.time()
        with patch.object(api, 'request_sms_2fa') as mock_request:
            api.on_needs_2fa()
        mock_request.assert_not_called()

    def test_on_needs_2fa_rerequests_after_code_never_arrived(self):
        api = iCloudAPI(config=VALID_CONFIG)
        mock_service = Mock()
        mock_service.requires_2fa = True
        api._api = mock_service
        api._challenge_from_fresh_signin = True
        api._sms_requested_at = time.time() - iCloudAPI.SMS_CODE_WAIT_SECONDS - 1
        with patch.object(api, 'request_sms_2fa', return_value="•••05") as mock_request, \
             patch.object(api, '_ensure_authenticated'):
            api.on_needs_2fa()
        mock_request.assert_called_once()

    def test_auto_request_rate_limited(self):
        api = iCloudAPI(config=VALID_CONFIG)
        mock_service = Mock()
        mock_service.requires_2fa = True
        api._api = mock_service
        api._challenge_from_fresh_signin = True
        with patch.object(api, 'request_sms_2fa', return_value="•••05") as mock_request, \
             patch.object(api, '_ensure_authenticated'):
            for _ in range(iCloudAPI.AUTO_SMS_MAX_REQUESTS + 2):
                api._auto_request_sms("test")
                api._sms_requested_at = None  # pretend each code was consumed
        assert mock_request.call_count == iCloudAPI.AUTO_SMS_MAX_REQUESTS

    def test_auto_request_skips_when_no_challenge(self):
        api = iCloudAPI(config=VALID_CONFIG)
        mock_service = Mock()
        mock_service.requires_2fa = False
        api._api = mock_service
        api._challenge_from_fresh_signin = True
        with patch.object(api, 'request_sms_2fa') as mock_request, \
             patch.object(api, '_ensure_authenticated'):
            api._auto_request_sms("test")
        mock_request.assert_not_called()

    def test_auto_request_falls_back_to_fresh_signin_when_request_fails(self):
        # The stale-challenge path: Apple rejects the SMS request on the current
        # session, so the flow does a from-scratch password signin and retries.
        api = iCloudAPI(config=VALID_CONFIG)
        mock_service = Mock()
        mock_service.requires_2fa = True
        api._api = mock_service
        api._challenge_from_fresh_signin = True
        with patch.object(api, 'request_sms_2fa',
                          side_effect=[Exception("challenge expired"), "•••05"]) as mock_request, \
             patch.object(api, '_ensure_authenticated'), \
             patch.object(api, '_fresh_password_signin',
                          side_effect=lambda: setattr(api, '_api', mock_service)) as mock_fresh:
            api._auto_request_sms("test")
        assert mock_request.call_count == 2
        mock_fresh.assert_called_once()

    def test_auto_request_untrusted_challenge_goes_straight_to_fresh_signin(self):
        # A challenge that did not come from our own fresh password signin
        # (e.g. restored from a persisted session token at startup) gets its
        # codes rejected by Apple — requesting on it just wastes an SMS, so the
        # auto flow must skip it and fresh-signin first (observed 2026-08-20).
        api = iCloudAPI(config=VALID_CONFIG)
        mock_service = Mock()
        mock_service.requires_2fa = True
        api._api = mock_service
        assert api._challenge_from_fresh_signin is False
        with patch.object(api, 'request_sms_2fa', return_value="•••05") as mock_request, \
             patch.object(api, '_ensure_authenticated') as mock_auth, \
             patch.object(api, '_fresh_password_signin',
                          side_effect=lambda: setattr(api, '_api', mock_service)) as mock_fresh:
            api._auto_request_sms("test")
        mock_fresh.assert_called_once()
        mock_auth.assert_not_called()
        mock_request.assert_called_once()

    def test_auto_request_fresh_signin_param_skips_existing_session(self):
        # fresh_signin=True (set after a stale-session code rejection) must not
        # request a code on the existing session first — that code would be
        # rejected too, wasting an SMS.
        api = iCloudAPI(config=VALID_CONFIG)
        mock_service = Mock()
        mock_service.requires_2fa = True
        api._api = mock_service
        with patch.object(api, 'request_sms_2fa', return_value="•••05") as mock_request, \
             patch.object(api, '_ensure_authenticated') as mock_auth, \
             patch.object(api, '_fresh_password_signin',
                          side_effect=lambda: setattr(api, '_api', mock_service)) as mock_fresh:
            api._auto_request_sms("test", fresh_signin=True)
        mock_fresh.assert_called_once()
        mock_auth.assert_not_called()
        mock_request.assert_called_once()

    def test_auto_request_total_failure_does_not_consume_rate_limit(self):
        # Both the initial request and the fresh-signin retry fail. Must not raise
        # (runs on monitor/poller threads), must not burn a rate-limit slot so a
        # later recovery attempt is still allowed, and must alert via Telegram
        # since automatic recovery is out of options.
        api = iCloudAPI(config=VALID_CONFIG)
        mock_service = Mock()
        mock_service.requires_2fa = True
        api._api = mock_service
        api._challenge_from_fresh_signin = True
        notify_path = 'assistant.Notifier.NullNotifier.send_message'
        with patch.object(api, 'request_sms_2fa', side_effect=Exception("still broken")), \
             patch.object(api, '_ensure_authenticated'), \
             patch.object(api, '_fresh_password_signin',
                          side_effect=lambda: setattr(api, '_api', mock_service)), \
             patch(notify_path) as mock_notify, \
             patch.object(api._auto_sms_limiter, 'record') as mock_record:
            api._auto_request_sms("test")
        mock_record.assert_not_called()
        mock_notify.assert_called_once()

    def test_auto_request_total_failure_enters_cooldown(self):
        # A total failure must arm the failure cooldown: the monitor retries
        # every 15 min, which the 4-per-30-min limiter never catches, so
        # without it a throttling Apple (-28248) means a full SRP signin plus
        # a Telegram alert on every tick, indefinitely.
        api = iCloudAPI(config=VALID_CONFIG)
        mock_service = Mock()
        mock_service.requires_2fa = True
        api._api = mock_service
        api._challenge_from_fresh_signin = True
        notify_path = 'assistant.Notifier.NullNotifier.send_message'
        with patch.object(api, 'request_sms_2fa', side_effect=Exception("throttled")), \
             patch.object(api, '_ensure_authenticated'), \
             patch.object(api, '_fresh_password_signin',
                          side_effect=lambda: setattr(api, '_api', mock_service)) as mock_fresh, \
             patch(notify_path) as mock_notify:
            api._auto_request_sms("monitor tick 1")
            api._auto_request_sms("monitor tick 2")
            api._auto_request_sms("monitor tick 3")
        assert api._auto_sms_last_failure_ts > 0
        mock_fresh.assert_called_once()
        mock_notify.assert_called_once()

    def test_auto_request_caps_fresh_signins_per_recovery(self):
        # Independent of the SMS limiter: a second from-scratch password signin
        # within one recovery is refused and the failure cooldown armed.
        api = iCloudAPI(config=VALID_CONFIG)
        mock_service = Mock()
        mock_service.requires_2fa = True
        notify_path = 'assistant.Notifier.NullNotifier.send_message'
        with patch.object(api, 'request_sms_2fa', return_value="•••05") as mock_request, \
             patch.object(api, '_fresh_password_signin',
                          side_effect=lambda: setattr(api, '_api', mock_service)) as mock_fresh, \
             patch(notify_path) as mock_notify:
            api._auto_request_sms("first", fresh_signin=True)
            api._sms_requested_at = None
            api._auto_request_sms("second", fresh_signin=True)
        mock_fresh.assert_called_once()
        mock_request.assert_called_once()
        mock_notify.assert_called_once()
        assert api._auto_sms_last_failure_ts > 0

    def test_failed_recovery_resets_fresh_signin_budget_for_next_recovery(self):
        # A failed recovery must hand the NEXT recovery (after the cooldown) a
        # fresh signin budget. If the counter survived, the cap check would trip
        # before any work, call _recovery_failed again, re-arm the 2h cooldown,
        # and loop alerts forever without ever attempting another signin.
        api = iCloudAPI(config=VALID_CONFIG)
        mock_service = Mock()
        mock_service.requires_2fa = True
        notify_path = 'assistant.Notifier.NullNotifier.send_message'
        with patch.object(api, 'request_sms_2fa', return_value="•••05") as mock_request, \
             patch.object(api, '_fresh_password_signin',
                          side_effect=lambda: setattr(api, '_api', mock_service)) as mock_fresh, \
             patch(notify_path) as mock_notify:
            api._auto_request_sms("first", fresh_signin=True)
            api._sms_requested_at = None
            api._auto_request_sms("second", fresh_signin=True)  # cap trips → recovery failed
            api._auto_sms_last_failure_ts = (
                time.time() - iCloudAPI.AUTO_SMS_FAILURE_COOLDOWN_SECONDS - 1)
            api._auto_request_sms("after cooldown", fresh_signin=True)
        assert mock_fresh.call_count == 2
        assert mock_request.call_count == 2
        mock_notify.assert_called_once()
        assert api._auto_sms_last_failure_ts == 0.0

    def test_fresh_signin_counter_resets_on_2fa_success(self):
        api = iCloudAPI(config=VALID_CONFIG)
        mock_service = Mock()
        mock_service.requires_2fa = True
        mock_service.is_trusted_session = True
        mock_service.two_factor_delivery_method = "sms"
        api._api = mock_service
        api._fresh_signins_this_recovery = 1
        with patch.object(api, '_validate_sms_code_verbose', return_value=True):
            assert api.send_2fa_code("123456") is True
        assert api._fresh_signins_this_recovery == 0

    def test_auto_request_cooldown_expiry_allows_a_new_attempt_and_success_clears_it(self):
        api = iCloudAPI(config=VALID_CONFIG)
        mock_service = Mock()
        mock_service.requires_2fa = True
        api._api = mock_service
        api._challenge_from_fresh_signin = True
        api._auto_sms_last_failure_ts = (
            time.time() - iCloudAPI.AUTO_SMS_FAILURE_COOLDOWN_SECONDS - 1)
        with patch.object(api, 'request_sms_2fa', return_value="•••05") as mock_request, \
             patch.object(api, '_ensure_authenticated'):
            api._auto_request_sms("test")
        mock_request.assert_called_once()
        assert api._auto_sms_last_failure_ts == 0.0

    def test_auto_request_burst_from_worker_threads_sends_one_sms(self):
        # A batch of photo download workers can all hit on_needs_2fa at once.
        # The lock must be held ACROSS the request — not just around the pending
        # check — so the second thread sees _sms_requested_at already set and
        # skips; concurrent requests would each fire an SMS and get the number
        # throttled by Apple.
        import threading
        api = iCloudAPI(config=VALID_CONFIG)
        mock_service = Mock()
        mock_service.requires_2fa = True
        api._api = mock_service
        api._challenge_from_fresh_signin = True

        both_asking = threading.Event()

        class WatchedLock:
            """The real lock, telling when the second worker has come for it."""

            def __init__(self, lock):
                self.lock = lock
                self.askers = set()

            def __enter__(self):
                self.askers.add(threading.get_ident())
                if len(self.askers) == 2:
                    both_asking.set()
                return self.lock.__enter__()

            def __exit__(self, *exc):
                return self.lock.__exit__(*exc)

        api._auto_sms_lock = WatchedLock(api._auto_sms_lock)

        def slow_request():
            # Still in flight when the second worker asks: with the lock released before the
            # request, that worker would find nothing pending and send an SMS of its own.
            assert both_asking.wait(5)
            api._sms_requested_at = time.time()
            return "•••05"

        barrier = threading.Barrier(2)

        def attempt():
            barrier.wait()
            api._auto_request_sms("burst")

        with patch.object(api, 'request_sms_2fa', side_effect=slow_request) as mock_request, \
             patch.object(api, '_ensure_authenticated'):
            threads = [threading.Thread(target=attempt) for _ in range(2)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        assert mock_request.call_count == 1

    def test_fresh_password_signin_failure_raises_and_clears_api(self):
        # The raise is what routes _auto_request_sms into the Telegram
        # manual-attention alert; the old token-restored session was already
        # discarded on entry, so _api must be None afterwards — not a zombie
        # session that later calls would treat as authenticated.
        from pyicloud.exceptions import PyiCloudFailedLoginException
        api = iCloudAPI(config=VALID_CONFIG)
        api._api = Mock()
        with patch('integrations.icloud.ICloudIntegration.PyiCloudService') as mock_cls:
            mock_cls.return_value.authenticate.side_effect = \
                PyiCloudFailedLoginException("bad password")
            with pytest.raises(iCloudAuthenticationError, match="Fresh signin failed"):
                api._fresh_password_signin()
        assert api._api is None
        assert api._auth_error is not None

    def test_auto_request_skips_sms_when_fresh_signin_resolves_2fa(self):
        # Fresh signin coming back with requires_2fa=False IS the good outcome.
        # Requesting a code anyway would raise (no phone challenge on the new
        # session) and fire the could-not-recover Telegram alert on what was a
        # successful recovery; the rate-limit slot must not be burned either.
        api = iCloudAPI(config=VALID_CONFIG)
        api._api = Mock()
        mock_service = Mock()
        mock_service.requires_2fa = False
        with patch.object(api, 'request_sms_2fa') as mock_request, \
             patch.object(api, '_fresh_password_signin',
                          side_effect=lambda: setattr(api, '_api', mock_service)) as mock_fresh, \
             patch.object(api._auto_sms_limiter, 'record') as mock_record:
            api._auto_request_sms("test", fresh_signin=True)
        mock_fresh.assert_called_once()
        mock_request.assert_not_called()
        mock_record.assert_not_called()

    def test_fresh_password_signin_clears_persistence_and_reauthenticates(self):
        # The whole point of the fresh signin: drop the persisted session token
        # (in-process, no manual cookie removal) so authenticate() is forced into
        # an SRP password signin that raises a live MFA challenge.
        api = iCloudAPI(config=VALID_CONFIG)
        api._api = Mock()
        api._sms_requested_at = time.time()
        api._auth_error = "old failure"
        with patch('integrations.icloud.ICloudIntegration.PyiCloudService') as mock_cls:
            mock_new_api = mock_cls.return_value
            mock_new_api.requires_2fa = True
            mock_new_api.is_trusted_session = False
            api._fresh_password_signin()
        assert mock_cls.call_args.kwargs.get('authenticate') is False
        mock_new_api.session.clear_persistence.assert_called_once()
        mock_new_api.authenticate.assert_called_once()
        assert api._api is mock_new_api
        assert api._sms_requested_at is None
        # Success must clear the stale auth error (get_auth_status reports it) and
        # mark the challenge trusted — this is the flag's ONLY True-producer; without
        # it every pending-code re-request would burn a redundant full SRP signin.
        assert api._auth_error is None
        assert api._challenge_from_fresh_signin is True

    def test_sms_request_pending_expires(self):
        api = iCloudAPI(config=VALID_CONFIG)
        api._sms_requested_at = time.time()
        assert api.sms_request_pending() is True
        api._sms_requested_at = time.time() - iCloudAPI.SMS_CODE_WAIT_SECONDS - 1
        assert api.sms_request_pending() is False
        api._sms_requested_at = None
        assert api.sms_request_pending() is False

    def test_manual_2fa_success_clears_pending_request(self):
        # After a successful manual code entry, a late-arriving stray code must
        # not find a pending request to attach to.
        api = iCloudAPI(config=VALID_CONFIG)
        mock_service = Mock()
        mock_service.validate_2fa_code.return_value = True
        mock_service.is_trusted_session = True
        api._api = mock_service
        api._sms_requested_at = time.time()
        api.send_2fa_code("123456")
        assert api._sms_requested_at is None


class TestGetAuthStatus:
    def test_needs_2fa_branch(self):
        api = iCloudAPI(config=VALID_CONFIG)
        api._api = _service_with_sms_challenge()
        api._challenge_from_fresh_signin = True
        status = api.get_auth_status()
        assert status['status'] == 'needs_2fa'

    def test_untrusted_token_session_is_partly_verified(self):
        api = iCloudAPI(config=VALID_CONFIG)
        mock_service = Mock()
        mock_service.requires_2fa = True
        api._api = mock_service
        status = api.get_auth_status()
        assert status['status'] == 'partial_auth'
        mock_service.session.post.assert_not_called(), "the status check makes no Find My request"

    def test_parked_signins_show_as_parked(self):
        from pyicloud.exceptions import PyiCloudFailedLoginException
        api = iCloudAPI(config=VALID_CONFIG)
        with patch('assistant.Notifier.NullNotifier.send_message'):
            api._park(PyiCloudFailedLoginException("Invalid email/password combination."))
        status = api.get_auth_status()
        assert status['status'] == 'parked'
        assert "Invalid email/password" in status['message']

    def test_authenticated_branch(self):
        api = iCloudAPI(config=VALID_CONFIG)
        mock_service = _find_my_answers(Mock(), [{"id": "fmip-1", "name": "iPhone"}])
        mock_service.requires_2fa = False
        api._api = mock_service
        status = api.get_auth_status()
        assert status['status'] == 'authenticated'


class TestGetICloudDevices:
    @patch('integrations.icloud.ICloudIntegration.PyiCloudService')
    def test_happy_path_returns_device_list(self, mock_pyicloud):
        record = {"id": "fmip-1", "baUUID": "uuid1", "name": "iPhone",
                  "deviceDisplayName": "iPhone 15", "deviceClass": "iPhone"}
        mock_service = _find_my_answers(Mock(), [record])
        mock_service.requires_2fa = False
        mock_pyicloud.return_value = mock_service

        api = iCloudAPI(config=VALID_CONFIG)
        devices = api.get_icloud_devices()

        assert len(devices) == 1
        assert devices[0]["id"] == "fmip-1", "the listing shows Find My's id, the one config entries should hold"
        assert devices[0]["name"] == "iPhone"

    @patch('integrations.icloud.ICloudIntegration.PyiCloudService')
    def test_requires_2fa_returns_empty_list(self, mock_pyicloud):
        mock_service = Mock()
        mock_service.requires_2fa = True
        mock_pyicloud.return_value = mock_service

        api = iCloudAPI(config=VALID_CONFIG)
        devices = api.get_icloud_devices()

        assert devices == []

    @patch('integrations.icloud.ICloudIntegration.PyiCloudService')
    def test_pyicloud_error_is_logged_on_one_short_line(self, mock_pyicloud):
        """2026-09-06: Apple served the 2FA page with a 409, pyicloud put all 940 lines
        of it into the exception, and the error line dumped them into floxy.out."""
        from pyicloud.exceptions import PyiCloudAPIResponseException
        html = "<html>\n" + "\n".join(f"  <div>{i}</div>" for i in range(940)) + "\n</html>"
        exc = PyiCloudAPIResponseException("Service Temporarily Unavailable", 503, Mock(text=html))
        mock_service = _find_my_answers(Mock(), exc=exc)
        mock_service.requires_2fa = False
        mock_pyicloud.return_value = mock_service

        api = iCloudAPI(config=VALID_CONFIG)
        with patch('integrations.icloud.ICloudIntegration.error') as mock_error:
            assert api.get_icloud_devices() == []

        line = mock_error.call_args.args[0]
        assert line.startswith("iCloud: Find My probe failed: Service Temporarily Unavailable (503): <html>")
        assert "\n" not in line
        assert len(line) < 400
        assert "chars, PyiCloudAPIResponseException]" in line


class TestPlaySound:
    def _make_device(self, ba_uuid, name="Device"):
        device = Mock()
        device.data = {"baUUID": ba_uuid, "name": name}
        device.play_sound = Mock()
        return device

    @patch('integrations.icloud.ICloudIntegration.PyiCloudService')
    def test_happy_path(self, mock_pyicloud):
        device = self._make_device("device-1-uuid", "iPhone")
        mock_service = Mock()
        mock_service.requires_2fa = False
        mock_service.devices = [device]
        mock_pyicloud.return_value = mock_service

        api = iCloudAPI(config=VALID_CONFIG)
        result = api.play_sound("iPhone")

        assert "Playing sound on iPhone" in result
        device.play_sound.assert_called_once()

    @patch('integrations.icloud.ICloudIntegration.PyiCloudService')
    def test_a_refused_signin_propagates_and_parks(self, mock_pyicloud, icloud_session_dir):
        from pyicloud.exceptions import PyiCloudFailedLoginException
        mock_pyicloud.side_effect = PyiCloudFailedLoginException("bad")

        api = iCloudAPI(config=VALID_CONFIG)
        with patch('assistant.Notifier.NullNotifier.send_message') as notify:
            with pytest.raises(iCloudAuthenticationError):
                api.play_sound("iPhone")
        notify.assert_called_once()
        assert (icloud_session_dir / api.PARK_FILE).exists()

    @patch('integrations.icloud.ICloudIntegration.PyiCloudService')
    def test_an_open_challenge_raises_and_keeps_the_session(self, mock_pyicloud):
        """Dropping the session here (as until 2026-10-16) threw away the challenge the
        auto-2FA flow was completing, and the next call opened another one. The ring waits
        out the challenge since 2026-11-02 (`TestFindMyWaitsOutTheChallenge`); here the wait
        budget is spent at once, so the refusal is what is checked, not the minute's wait."""
        mock_service = _service_with_sms_challenge()
        mock_pyicloud.return_value = mock_service

        api = iCloudAPI(config=VALID_CONFIG)
        api._api = mock_service
        api._challenge_from_fresh_signin = True

        with patch('assistant.Notifier.NullNotifier.send_message') as notify, \
             patch.object(iCloudAPI, "CHALLENGE_WAIT_SECONDS", 0):
            with pytest.raises(iCloudAuthenticationError, match="2FA code"):
                api.play_sound("iPhone")
        notify.assert_not_called()
        assert api._api is mock_service
        assert api.state == IntegrationState.NEEDS_2FA
        mock_service.authenticate.assert_not_called()

    @patch('integrations.icloud.ICloudIntegration.PyiCloudService')
    def test_device_not_found_raises_with_available_names_and_models(self, mock_pyicloud):
        """2026-10-08: a phone whose baUUID went blank was listed by name only, so the error
        hid that the row was there — the model tells the rows apart."""
        other_device = self._make_device("other-uuid", "MacBook")
        other_device.data["deviceDisplayName"] = "MacBook Pro"
        mock_service = Mock()
        mock_service.requires_2fa = False
        mock_service.devices = [other_device]
        mock_pyicloud.return_value = mock_service

        api = iCloudAPI(config=VALID_CONFIG)
        with pytest.raises(iCloudAPIError, match="not found in iCloud.*MacBook \\(MacBook Pro"):
            api.play_sound("iPhone")

    @patch('integrations.icloud.ICloudIntegration.PyiCloudService')
    def test_device_is_found_by_find_my_id_when_its_bauuid_is_blank(self, mock_pyicloud):
        """2026-10-08: a phone re-enrolled in Find My kept its record but lost its baUUID."""
        device = Mock()
        device.data = {"id": "device-1-uuid", "baUUID": "", "name": "iPhone"}
        device.play_sound = Mock()
        mock_service = Mock()
        mock_service.requires_2fa = False
        mock_service.devices = [device]
        mock_pyicloud.return_value = mock_service

        api = iCloudAPI(config=VALID_CONFIG)
        api.play_sound("iPhone")
        device.play_sound.assert_called_once()

    @patch('integrations.icloud.ICloudIntegration.PyiCloudService')
    def test_an_entry_holding_the_old_bauuid_still_matches(self, mock_pyicloud):
        device = Mock()
        device.data = {"id": "fmip-long-id", "baUUID": "device-1-uuid", "name": "iPhone"}
        device.play_sound = Mock()
        blank = Mock()
        blank.data = {"id": "fmip-other", "baUUID": "", "name": "AirPods"}
        mock_service = Mock()
        mock_service.requires_2fa = False
        mock_service.devices = [blank, device]
        mock_pyicloud.return_value = mock_service

        api = iCloudAPI(config=VALID_CONFIG)
        api.play_sound("iPhone")
        device.play_sound.assert_called_once()

    @patch('integrations.icloud.ICloudIntegration.PyiCloudService')
    def test_device_list_reauth_signs_in_on_the_live_session_and_retries(self, mock_pyicloud):
        """Find My rejecting the cookies earns one deliberate sign-in on the session that holds
        the trust token (force_refresh: the cached token alone would hand the same cookies back),
        never a second PyiCloudService."""
        device = self._make_device("device-1-uuid", "iPhone")

        class FailingDevices:
            def __iter__(self):
                raise Exception("Re-authentication required")

        service = Mock()
        service.requires_2fa = False
        type(service).devices = PropertyMock(side_effect=[FailingDevices(), [device]])

        api = iCloudAPI(config=VALID_CONFIG)
        api._api = service

        result = api.play_sound("iPhone")

        assert "Playing sound on iPhone" in result
        device.play_sound.assert_called_once()
        service.authenticate.assert_called_once_with(force_refresh=True)
        mock_pyicloud.assert_not_called()
        assert service._devices is None, "the device manager of the rejected cookies is dropped"

    def test_play_sound_call_reauth_retries_successfully(self):
        device_fail = self._make_device("device-1-uuid", "iPhone")
        device_fail.play_sound.side_effect = Exception("Re-authentication required")
        device_ok = self._make_device("device-1-uuid", "iPhone")

        service = Mock()
        service.requires_2fa = False
        type(service).devices = PropertyMock(side_effect=[[device_fail], [device_ok]])

        api = iCloudAPI(config=VALID_CONFIG)
        api._api = service
        result = api.play_sound("iPhone")

        assert "Playing sound on iPhone" in result
        device_ok.play_sound.assert_called_once()
        service.authenticate.assert_called_once_with(force_refresh=True)

    def test_a_reauth_that_apple_refuses_fails_the_command_and_parks(self, icloud_session_dir):
        from pyicloud.exceptions import PyiCloudFailedLoginException

        class FailingDevices:
            def __iter__(self):
                raise Exception("Re-authentication required")

        service = Mock()
        service.requires_2fa = False
        service.devices = FailingDevices()
        service.authenticate.side_effect = PyiCloudFailedLoginException("Invalid email/password combination.")

        api = iCloudAPI(config=VALID_CONFIG)
        api._api = service
        with patch('assistant.Notifier.NullNotifier.send_message'):
            with pytest.raises(iCloudAuthenticationError, match="Find My is not available"):
                api.play_sound("iPhone")
        assert (icloud_session_dir / api.PARK_FILE).exists()
        with pytest.raises(iCloudAuthenticationError, match="parked"):
            api.play_sound("iPhone")
        service.authenticate.assert_called_once(), "parked: the second command does not sign in"


class TestGetLocation:
    def _make_device(self, ba_uuid, name="Device", location=None, location_available=True):
        device = Mock()
        device.data = {"baUUID": ba_uuid, "name": name}
        # AppleDevice.location is a property; on real instances it returns the dict
        # only when location_available is True. Bypass that check by attaching the
        # value directly to the Mock — _location_for_spec only reads .location.
        if location_available:
            device.location = location
        else:
            device.location = None
        return device

    @patch('integrations.icloud.ICloudIntegration.reverse_geocode')
    @patch('integrations.icloud.ICloudIntegration.PyiCloudService')
    def test_happy_path(self, mock_pyicloud, mock_geocode):
        import time as _time
        recent_ts_ms = (_time.time() - 60) * 1000  # 1 minute ago
        device = self._make_device(
            "device-1-uuid",
            name="Charlotte's iPhone",
            location={
                "latitude": 57.7012,
                "longitude": 11.9730,
                "horizontalAccuracy": 25.4,
                "timeStamp": recent_ts_ms,
                "isOld": False,
                "isInaccurate": False,
            },
        )
        mock_service = Mock()
        mock_service.requires_2fa = False
        mock_service.devices = [device]
        mock_pyicloud.return_value = mock_service
        mock_geocode.return_value = "Parkgatan, Centrum"

        api = iCloudAPI(config=VALID_CONFIG)
        result = api.get_location("iPhone")

        mock_geocode.assert_called_once_with(57.7012, 11.9730)
        assert "iPhone: Parkgatan, Centrum" in result
        assert "57.70120" in result
        assert "11.97300" in result
        assert "±25m" in result
        assert "minute" in result  # freshness line present

    @patch('integrations.icloud.ICloudIntegration.reverse_geocode')
    @patch('integrations.icloud.ICloudIntegration.PyiCloudService')
    def test_location_dict_none_returns_unavailable(self, mock_pyicloud, mock_geocode):
        device = self._make_device("device-1-uuid", name="iPhone", location=None)
        mock_service = Mock()
        mock_service.requires_2fa = False
        mock_service.devices = [device]
        mock_pyicloud.return_value = mock_service

        api = iCloudAPI(config=VALID_CONFIG)
        result = api.get_location("iPhone")

        assert "Location unavailable" in result
        assert "iPhone" in result
        mock_geocode.assert_not_called()

    @patch('integrations.icloud.ICloudIntegration.reverse_geocode')
    @patch('integrations.icloud.ICloudIntegration.PyiCloudService')
    def test_missing_coordinates_returns_unavailable(self, mock_pyicloud, mock_geocode):
        device = self._make_device(
            "device-1-uuid",
            name="iPhone",
            location={"latitude": None, "longitude": None, "horizontalAccuracy": 100},
        )
        mock_service = Mock()
        mock_service.requires_2fa = False
        mock_service.devices = [device]
        mock_pyicloud.return_value = mock_service

        api = iCloudAPI(config=VALID_CONFIG)
        result = api.get_location("iPhone")

        assert "Location unavailable" in result
        mock_geocode.assert_not_called()

    @patch('integrations.icloud.ICloudIntegration.reverse_geocode')
    @patch('integrations.icloud.ICloudIntegration.PyiCloudService')
    def test_geocode_failure_still_returns_gps(self, mock_pyicloud, mock_geocode):
        device = self._make_device(
            "device-1-uuid",
            name="iPhone",
            location={
                "latitude": 57.0,
                "longitude": 12.0,
                "horizontalAccuracy": 50,
                "timeStamp": None,
                "isOld": False,
                "isInaccurate": False,
            },
        )
        mock_service = Mock()
        mock_service.requires_2fa = False
        mock_service.devices = [device]
        mock_pyicloud.return_value = mock_service
        mock_geocode.return_value = None

        api = iCloudAPI(config=VALID_CONFIG)
        result = api.get_location("iPhone")

        assert "57.00000" in result
        assert "12.00000" in result
        assert "±50m" in result

    @patch('integrations.icloud.ICloudIntegration.reverse_geocode')
    @patch('integrations.icloud.ICloudIntegration.PyiCloudService')
    def test_stale_and_inaccurate_flags_surfaced(self, mock_pyicloud, mock_geocode, monkeypatch):
        # Timeout 0: a fix that stays stale is returned immediately as last-known
        # rather than polled — see test_stale_location_triggers_active_locate for
        # the polling path.
        monkeypatch.setattr(ICloudIntegration, "_LOCATE_TIMEOUT_SECONDS", 0)
        device = self._make_device(
            "device-1-uuid",
            name="iPhone",
            location={
                "latitude": 57.0,
                "longitude": 12.0,
                "horizontalAccuracy": 1500,
                "timeStamp": None,
                "isOld": True,
                "isInaccurate": True,
            },
        )
        mock_service = Mock()
        mock_service.requires_2fa = False
        mock_service.devices = [device]
        mock_pyicloud.return_value = mock_service
        mock_geocode.return_value = "Somewhere"

        api = iCloudAPI(config=VALID_CONFIG)
        result = api.get_location("iPhone")

        assert "STALE" in result
        assert "inaccurate" in result

    @patch('integrations.icloud.ICloudIntegration.time.sleep')
    @patch('integrations.icloud.ICloudIntegration.reverse_geocode')
    @patch('integrations.icloud.ICloudIntegration.PyiCloudService')
    def test_stale_location_triggers_active_locate(self, mock_pyicloud, mock_geocode, mock_sleep):
        # A stale fix should trigger devices.refresh(locate=True); once the
        # device reports a fresh fix the loop returns it.
        stale = {
            "latitude": 57.0, "longitude": 12.0, "horizontalAccuracy": 1500,
            "timeStamp": None, "isOld": True, "isInaccurate": False,
        }
        fresh = {
            "latitude": 57.5, "longitude": 12.5, "horizontalAccuracy": 20,
            "timeStamp": None, "isOld": False, "isInaccurate": False,
        }
        device = self._make_device("device-1-uuid", name="iPhone", location=stale)

        devices = MagicMock()
        devices.__iter__.side_effect = lambda: iter([device])

        def do_refresh(locate=False):
            device.location = fresh

        devices.refresh.side_effect = do_refresh

        mock_service = Mock()
        mock_service.requires_2fa = False
        mock_service.devices = devices
        mock_pyicloud.return_value = mock_service
        mock_geocode.return_value = "Fresh Place"

        api = iCloudAPI(config=VALID_CONFIG)
        result = api.get_location("iPhone")

        devices.refresh.assert_called_with(locate=True)
        assert "Fresh Place" in result
        assert "57.50000" in result
        assert "STALE" not in result

    @patch('integrations.icloud.ICloudIntegration.time.sleep')
    @patch('integrations.icloud.ICloudIntegration.time.time')
    @patch('integrations.icloud.ICloudIntegration.reverse_geocode')
    @patch('integrations.icloud.ICloudIntegration.PyiCloudService')
    def test_persistently_stale_location_falls_back_after_timeout(
        self, mock_pyicloud, mock_geocode, mock_time, mock_sleep
    ):
        # Device never reports a fresh fix: after the timeout budget we fall
        # back to the last-known (stale) position rather than looping forever.
        # Fake clock: signin timestamp, then deadline=1010; loop checks at
        # 1000, 1005, then 1011 (over).
        clock = [1000.0, 1000.0, 1000.0, 1005.0, 1011.0]
        # The patch replaces the process-wide time.time: another thread's call (a logger, a
        # worker an earlier test left behind) must not eat a tick of this clock.
        test_thread, real_time = threading.current_thread(), time.time

        def fake_time():
            if threading.current_thread() is not test_thread:
                return real_time()
            return clock.pop(0) if len(clock) > 1 else clock[0]

        mock_time.side_effect = fake_time
        stale = {
            "latitude": 57.0, "longitude": 12.0, "horizontalAccuracy": 1500,
            "timeStamp": None, "isOld": True, "isInaccurate": False,
        }
        device = self._make_device("device-1-uuid", name="iPhone", location=stale)

        devices = MagicMock()
        devices.__iter__.side_effect = lambda: iter([device])
        mock_service = Mock()
        mock_service.requires_2fa = False
        mock_service.devices = devices
        mock_pyicloud.return_value = mock_service
        mock_geocode.return_value = "Old Place"

        api = iCloudAPI(config=VALID_CONFIG)
        result = api.get_location("iPhone")

        assert "STALE" in result
        assert devices.refresh.call_count == 2  # polled until the 10s budget ran out

    @patch('integrations.icloud.ICloudIntegration.reverse_geocode')
    @patch('integrations.icloud.ICloudIntegration.PyiCloudService')
    def test_reauth_retries_successfully(self, mock_pyicloud, mock_geocode):
        # First device raises a re-auth error when location is read; after the deliberate
        # sign-in on the same session the device list is read again and works.
        device_fail = self._make_device("device-1-uuid", name="iPhone")
        type(device_fail).location = PropertyMock(side_effect=Exception("Re-authentication required"))

        device_ok = self._make_device(
            "device-1-uuid",
            name="iPhone",
            location={
                "latitude": 57.0,
                "longitude": 12.0,
                "horizontalAccuracy": 25,
                "timeStamp": None,
                "isOld": False,
                "isInaccurate": False,
            },
        )

        service = Mock()
        service.requires_2fa = False
        type(service).devices = PropertyMock(side_effect=[[device_fail], [device_ok]])
        mock_pyicloud.return_value = service
        mock_geocode.return_value = "Some Place"

        api = iCloudAPI(config=VALID_CONFIG)
        result = api.get_location("iPhone")

        assert "Some Place" in result
        service.authenticate.assert_called_with(force_refresh=True)
        assert mock_pyicloud.call_count == 1, "the sign-in happens on the live session, not a new one"


class TestIsReauthenticationError:
    def test_matches_re_authentication_required(self):
        assert iCloudAPI._is_reauthentication_error(Exception("Re-authentication required")) is True

    def test_matches_authentication_required(self):
        assert iCloudAPI._is_reauthentication_error(Exception("Authentication required")) is True

    def test_case_insensitive(self):
        assert iCloudAPI._is_reauthentication_error(Exception("RE-AUTHENTICATION REQUIRED")) is True

    def test_no_match_timeout(self):
        assert iCloudAPI._is_reauthentication_error(Exception("Connection timed out")) is False

    def test_partial_keyword_not_enough(self):
        assert iCloudAPI._is_reauthentication_error(Exception("authentication failed")) is False


class TestFindMyManagerNeverSignsIn:
    """pyicloud's device manager answers Find My's 'Re-authentication required' with a password
    sign-in of its own, inside the device read — the Oct 6–13 sign-ins whose 2FA challenge
    nothing completed. The module replaces that method: the refusal propagates."""

    def test_a_reauth_refusal_propagates_instead_of_a_signin(self):
        from pyicloud.exceptions import PyiCloudAuthRequiredException
        from pyicloud.services.findmyiphone import FindMyiPhoneServiceManager
        manager = Mock()
        manager._refresh_client.side_effect = PyiCloudAuthRequiredException("m@mlab.test", Mock())
        with pytest.raises(PyiCloudAuthRequiredException):
            FindMyiPhoneServiceManager._refresh_client_with_reauth(manager, locate=False)
        manager.session.service.authenticate.assert_not_called()
        manager._initialize_devices.assert_not_called()

    def test_a_working_refresh_initializes_the_devices_without_a_monitor_thread(self):
        from pyicloud.services.findmyiphone import FindMyiPhoneServiceManager
        manager = Mock()
        manager._refresh_interval = None
        FindMyiPhoneServiceManager._refresh_client_with_reauth(manager, locate=True)
        manager._refresh_client.assert_called_once_with(locate=True)
        manager._initialize_devices.assert_called_once_with(locate=True)
        manager._start_monitor_thread.assert_not_called()

    @patch('integrations.icloud.ICloudIntegration.PyiCloudService')
    def test_the_service_is_built_without_a_refresh_interval(self, mock_pyicloud):
        """A manager with the default interval re-reads Find My every five minutes for the rest
        of the process once a command has used it."""
        api = iCloudAPI(config=VALID_CONFIG)
        api._ensure_authenticated()
        assert mock_pyicloud.call_args.kwargs["refresh_interval"] is None


class TestRequestSms2fa:
    def test_raises_when_no_trusted_phone(self):
        api = iCloudAPI(config=VALID_CONFIG)
        mock_service = Mock()
        mock_service._auth_data = {"phoneNumberVerification": {}}
        api._api = mock_service
        with pytest.raises(iCloudAuthenticationError, match="No trusted phone"):
            api.request_sms_2fa()

    @patch('integrations.icloud.ICloudIntegration.ICloudIntegration._ensure_authenticated')
    def test_raises_on_http_failure(self, _mock_auth):
        api = iCloudAPI(config=VALID_CONFIG)
        mock_service = Mock()
        mock_service._auth_data = {
            "phoneNumberVerification": {
                "trustedPhoneNumber": {"id": 1, "obfuscatedNumber": "(***) ***-42"}
            }
        }
        mock_service._get_auth_headers.return_value = {}
        mock_response = Mock()
        mock_response.status_code = 403
        mock_service.session.put.return_value = mock_response
        api._api = mock_service

        with pytest.raises(iCloudAuthenticationError, match="SMS request failed"):
            api.request_sms_2fa()

    @patch('integrations.icloud.ICloudIntegration.ICloudIntegration._ensure_authenticated')
    def test_happy_path_returns_obfuscated_number(self, _mock_auth):
        api = iCloudAPI(config=VALID_CONFIG)
        mock_service = Mock()
        mock_service._auth_data = {
            "phoneNumberVerification": {
                "trustedPhoneNumber": {"id": 1, "obfuscatedNumber": "(***) ***-42"}
            }
        }
        mock_service._get_auth_headers.return_value = {}
        mock_response = Mock()
        mock_response.status_code = 200
        mock_service.session.put.return_value = mock_response
        api._api = mock_service

        result = api.request_sms_2fa()

        assert result == "(***) ***-42"
        assert mock_service._auth_data["mode"] == "sms"
        mock_service.session.put.assert_called_once()


class TestCheckHealth:
    @patch('integrations.icloud.ICloudIntegration.PyiCloudService')
    def test_returns_true_when_no_2fa_needed(self, mock_pyicloud):
        mock_service = _find_my_answers(Mock(), [{"id": "fmip-1"}])
        mock_service.requires_2fa = False
        mock_pyicloud.return_value = mock_service

        api = iCloudAPI(config=VALID_CONFIG)
        assert api.check_health() is True

    @patch('integrations.icloud.ICloudIntegration.PyiCloudService')
    def test_an_untrusted_token_session_is_healthy(self, mock_pyicloud):
        """pyicloud's requires_2fa alone is not the verdict: a token-restored session Apple calls
        untrusted has no open challenge, and nothing asks Find My about it."""
        mock_service = Mock()
        mock_service.requires_2fa = True
        mock_service._auth_data = {}
        mock_pyicloud.return_value = mock_service

        api = iCloudAPI(config=VALID_CONFIG)
        assert api.check_health() is True
        mock_service.session.post.assert_not_called()

    @patch('integrations.icloud.ICloudIntegration.PyiCloudService')
    def test_returns_false_while_a_challenge_is_open(self, mock_pyicloud):
        mock_pyicloud.return_value = _service_with_sms_challenge()

        api = iCloudAPI(config=VALID_CONFIG)
        assert api.check_health() is False

    def test_a_session_find_my_rejects_is_idle_not_unhealthy(self):
        """Until 2026-10-16 the check read Find My every 15 minutes and re-signed in hourly when
        the cookies had expired — the stream of refusals Apple read as an attack."""
        from pyicloud.exceptions import PyiCloudAuthRequiredException
        service = _find_my_answers(Mock(), exc=PyiCloudAuthRequiredException("m@mlab.test", Mock()))
        service.requires_2fa = False
        api = iCloudAPI(config=VALID_CONFIG)
        api._api = service
        for _ in range(3):
            assert api.check_health() is True
        service.session.post.assert_not_called()
        service.authenticate.assert_not_called()


class TestApiProperty:
    @patch('integrations.icloud.ICloudIntegration.PyiCloudService')
    def test_returns_api_when_authenticated_no_2fa(self, mock_pyicloud):
        mock_service = Mock()
        mock_service.requires_2fa = False
        mock_pyicloud.return_value = mock_service

        api = iCloudAPI(config=VALID_CONFIG)
        assert api.api is mock_service

    @patch('integrations.icloud.ICloudIntegration.PyiCloudService')
    def test_returns_api_for_an_untrusted_token_session(self, mock_pyicloud):
        """Photos reads CloudKit through it and asks for the sign-in itself when the trust is missing."""
        mock_service = Mock()
        mock_service.requires_2fa = True
        mock_service._auth_data = {}
        mock_pyicloud.return_value = mock_service

        api = iCloudAPI(config=VALID_CONFIG)
        assert api.api is mock_service
        mock_service.session.post.assert_not_called()

    @patch('integrations.icloud.ICloudIntegration.PyiCloudService')
    def test_returns_none_while_a_challenge_is_open(self, mock_pyicloud):
        mock_pyicloud.return_value = _service_with_sms_challenge()

        api = iCloudAPI(config=VALID_CONFIG)
        assert api.api is None

    @patch('integrations.icloud.ICloudIntegration.PyiCloudService')
    def test_returns_none_when_not_authenticated(self, mock_pyicloud):
        # Make re-authentication fail so the api property falls into its except branch
        # without actually hitting Apple's servers (which is what made this test slow).
        from pyicloud.exceptions import PyiCloudFailedLoginException
        mock_pyicloud.side_effect = PyiCloudFailedLoginException("auth failed")

        api = iCloudAPI(config=VALID_CONFIG)
        api._api = None
        assert api.api is None


class TestRefreshConnection:
    @patch('integrations.icloud.ICloudIntegration.PyiCloudService')
    def test_clears_state_and_reauthenticates(self, mock_pyicloud):
        mock_service = Mock()
        mock_pyicloud.return_value = mock_service

        api = iCloudAPI(config=VALID_CONFIG)
        api._api = Mock()
        api._auth_error = "old error"

        api.refresh_connection()

        assert api._auth_error is None
        assert api._api is mock_service

    @patch('integrations.icloud.ICloudIntegration.PyiCloudService')
    def test_a_users_refresh_checks_find_my_and_sets_state_ok(self, mock_pyicloud):
        mock_service = _find_my_answers(Mock(), [{"id": "fmip-1"}])
        mock_service.requires_2fa = False
        mock_pyicloud.return_value = mock_service

        api = iCloudAPI(config=VALID_CONFIG)
        api.state = IntegrationState.NEEDS_2FA

        api.refresh_connection(by_user=True)

        assert api.state == IntegrationState.OK
        assert mock_service.session.post.call_count == 1

    @patch('integrations.icloud.ICloudIntegration.PyiCloudService')
    def test_an_automatic_refresh_leaves_find_my_alone(self, mock_pyicloud):
        """Photos' refresh on an auth error needs CloudKit, not Find My."""
        mock_service = _find_my_answers(Mock(), [{"id": "fmip-1"}])
        mock_service.requires_2fa = False
        mock_pyicloud.return_value = mock_service

        api = iCloudAPI(config=VALID_CONFIG)
        api.refresh_connection()

        assert api._api is mock_service
        mock_service.session.post.assert_not_called()
        assert all(call.kwargs.get("force_refresh") is not True for call in mock_service.authenticate.call_args_list), \
            "the session sign-in happens, the Find My one does not"

    def test_an_automatic_refresh_keeps_a_session_with_an_open_challenge(self):
        service = _service_with_sms_challenge()
        api = iCloudAPI(config=VALID_CONFIG)
        api._api = service
        api._challenge_from_fresh_signin = True
        with patch.object(api, "on_needs_2fa") as nudge, \
                patch('integrations.icloud.ICloudIntegration.PyiCloudService') as mock_pyicloud:
            api.refresh_connection()
        assert api._api is service
        mock_pyicloud.assert_not_called()
        nudge.assert_called_once()


class TestHumanizeAge:
    def test_hours_under_ten(self):
        result = ICloudIntegration._humanize_age(7200)
        assert "2.0 hours" == result

    def test_hours_over_ten(self):
        assert ICloudIntegration._humanize_age(43200) == "12 hours"

    def test_days_under_ten(self):
        result = ICloudIntegration._humanize_age(172800)
        assert "2.0 days" == result

    def test_days_over_ten(self):
        assert ICloudIntegration._humanize_age(1_209_600) == "14 days"

    def test_boundary_59_seconds(self):
        assert "seconds" in ICloudIntegration._humanize_age(59)

    def test_boundary_60_seconds_is_minutes(self):
        assert "minute" in ICloudIntegration._humanize_age(60)


class TestFormatLocation:
    def test_basic_with_place(self):
        import time as _time
        ts_ms = (_time.time() - 30) * 1000
        result = ICloudIntegration._format_location(
            "iPhone", "Parkgatan, Centrum", 57.7, 11.96, 25.0, ts_ms, False, False,
        )
        assert result.startswith("iPhone: Parkgatan, Centrum")
        assert "57.70000" in result
        assert "±25m" in result

    def test_no_place_name(self):
        result = ICloudIntegration._format_location(
            "iPad", None, 57.7, 11.96, 10.0, None, False, False,
        )
        assert "location unavailable as a place name" in result
        assert "57.70000" in result

    def test_accuracy_unknown(self):
        result = ICloudIntegration._format_location(
            "Phone", "Place", 57.0, 12.0, None, None, False, False,
        )
        assert "accuracy unknown" in result

class TestPersistentSessionDir:
    """pyicloud's default session store is the temp dir, which macOS wipes on
    reboot — that forced a password signin + 2FA on every reboot (2026-08-30)."""

    @patch('integrations.icloud.ICloudIntegration.PyiCloudService')
    def test_service_uses_persistent_cookie_directory(self, mock_pyicloud, tmp_path):
        with patch('integrations.icloud.ICloudIntegration.ICLOUD_SESSION_DIR',
                   tmp_path / "icloud_session"), \
             patch('integrations.icloud.ICloudIntegration.resolve_cookie_directory',
                   return_value=str(tmp_path / "missing")):
            api = iCloudAPI(config=VALID_CONFIG)
            api._ensure_authenticated()
        assert mock_pyicloud.call_args.kwargs['cookie_directory'] == str(tmp_path / "icloud_session")
        assert (tmp_path / "icloud_session").is_dir()

    @patch('integrations.icloud.ICloudIntegration.PyiCloudService')
    def test_adopts_legacy_temp_dir_session_once(self, mock_pyicloud, tmp_path):
        # The switch itself must not cost a signin: the live temp-dir session is
        # copied over when the persistent dir is empty — and only then.
        legacy = tmp_path / "legacy"
        legacy.mkdir()
        (legacy / "acct.session").write_text("token")
        (legacy / "acct.cookiejar").write_text("cookies")
        target = tmp_path / "icloud_session"
        with patch('integrations.icloud.ICloudIntegration.ICLOUD_SESSION_DIR', target), \
             patch('integrations.icloud.ICloudIntegration.resolve_cookie_directory',
                   return_value=str(legacy)):
            iCloudAPI(config=VALID_CONFIG)._ensure_authenticated()
            assert (target / "acct.session").read_text() == "token"
            assert (target / "acct.cookiejar").read_text() == "cookies"
            (legacy / "acct.session").write_text("newer")
            iCloudAPI.reset_instance()
            iCloudAPI(config=VALID_CONFIG)._fresh_password_signin()
        assert (target / "acct.session").read_text() == "token"


def _service_with_sms_challenge(requires_2fa=True):
    service = Mock()
    service.requires_2fa = requires_2fa
    service.is_trusted_session = False
    service._auth_data = {"phoneNumberVerification": {"trustedPhoneNumber": {
        "id": 1, "pushMode": "sms", "nonFTEU": False, "obfuscatedNumber": "•••05"}}}
    return service


class TestAutoSentCode:
    """A password signin makes Apple push an SMS code on its own. Until
    2026-08-30 the flow then PUT a second /verify/phone request on top of it —
    one extra text per signin, and Apple 409'd every code submitted after it.
    The auto-sent code is now the pending request."""

    @patch('integrations.icloud.ICloudIntegration.PyiCloudService')
    def test_password_signin_at_startup_arms_auto_sent_code(self, mock_pyicloud):
        mock_pyicloud.return_value = _service_with_sms_challenge()
        api = iCloudAPI(config=VALID_CONFIG)
        before = time.time()
        with patch.object(api, '_session_dir', return_value="/tmp/x"):
            api._ensure_authenticated()
        assert api.sms_request_pending()
        assert before <= api._sms_requested_at <= time.time()
        assert api._challenge_from_fresh_signin is True
        assert api._api._two_factor_delivery_method == "sms"
        assert api._api._auth_data["trustedPhoneNumber"]["id"] == 1

    @patch('integrations.icloud.ICloudIntegration.PyiCloudService')
    def test_token_restored_session_arms_nothing(self, mock_pyicloud):
        service = Mock()
        service.requires_2fa = True
        service._auth_data = {}
        mock_pyicloud.return_value = service
        api = iCloudAPI(config=VALID_CONFIG)
        with patch.object(api, '_session_dir', return_value="/tmp/x"):
            api._ensure_authenticated()
        assert api._sms_requested_at is None
        assert api._challenge_from_fresh_signin is False

    @patch('integrations.icloud.ICloudIntegration.PyiCloudService')
    def test_fresh_password_signin_arms_auto_sent_code(self, mock_pyicloud):
        mock_pyicloud.return_value = _service_with_sms_challenge()
        api = iCloudAPI(config=VALID_CONFIG)
        with patch.object(api, '_session_dir', return_value="/tmp/x"):
            api._fresh_password_signin()
        assert api.sms_request_pending()

    def test_auto_request_after_fresh_signin_skips_extra_sms_request(self):
        api = iCloudAPI(config=VALID_CONFIG)
        service = _service_with_sms_challenge()

        def fresh():
            api._api = service
            api._sms_requested_at = time.time()
        with patch.object(api, '_fresh_password_signin', side_effect=fresh), \
             patch.object(api, 'request_sms_2fa') as mock_request:
            api._auto_request_sms("test", fresh_signin=True)
        mock_request.assert_not_called()
        assert api._auto_sms_limiter.check() is None
        assert api.sms_request_pending()

    def test_auto_request_after_fresh_signin_without_pushed_code_still_requests(self):
        api = iCloudAPI(config=VALID_CONFIG)
        service = Mock()
        service.requires_2fa = True
        with patch.object(api, '_fresh_password_signin',
                          side_effect=lambda: setattr(api, '_api', service)), \
             patch.object(api, 'request_sms_2fa', return_value="•••05") as mock_request:
            api._auto_request_sms("test", fresh_signin=True)
        mock_request.assert_called_once()

    def test_auto_sent_code_is_submitted_by_the_poller(self):
        api = iCloudAPI(config=VALID_CONFIG)
        api._api = _service_with_sms_challenge()
        api._arm_auto_sent_code(time.time() - 5)
        with patch('assistant.Notifier.NullNotifier.send_message'), \
             patch('assistant.integration.IntegrationMonitor.IntegrationMonitor'), \
             patch.object(api, 'send_2fa_code', return_value=True) as mock_send:
            api.submit_auto_2fa_code("123456", received_at=time.time())
        mock_send.assert_called_once_with("123456")

    def test_manual_sms_request_sends_non_fteu_flag(self):
        # Apple's FTEU flow expects nonFTEU echoed back (pyicloud 2.6.5 does the
        # same in its own SMS requests and code validation).
        api = iCloudAPI(config=VALID_CONFIG)
        api._api = _service_with_sms_challenge()
        api._api.session.put.return_value = Mock(status_code=200)
        api.request_sms_2fa()
        payload = api._api.session.put.call_args.kwargs['json']
        assert payload == {"phoneNumber": {"id": 1, "nonFTEU": False}, "mode": "sms"}


class TestAcceptedCodeWith409:
    """2026-08-30 probe: on this account Apple answers an ACCEPTED SMS code with
    HTTP 409 (challenge form echoed, no service_errors) plus a fresh
    X-Apple-Session-Token, and 2sv/trust then succeeds. Treating the 409 as a
    rejection is what kept every 2FA round since Aug 17 from ever trusting."""

    @staticmethod
    def _exc(status, headers, body):
        from pyicloud.exceptions import PyiCloudAPIResponseException
        resp = Mock(status_code=status, headers=headers, text=json.dumps(body))
        resp.json.return_value = body
        return PyiCloudAPIResponseException("Authentication required for Account.", status, resp)

    def _api(self):
        api = iCloudAPI(config=VALID_CONFIG)
        service = Mock()
        service.two_factor_delivery_method = "sms"
        service.is_trusted_session = False
        service.requires_2sa = False
        service.trust_session.side_effect = lambda: setattr(service, "is_trusted_session", True) or True
        api._api = service
        return api, service

    def test_409_with_session_token_is_accepted_and_trusted(self):
        api, service = self._api()
        service._validate_sms_code.side_effect = self._exc(
            409, {"X-Apple-Session-Token": "tok", "scnt": "x"},
            {"securityCode": {"code": "123456", "valid": True}, "mode": "sms"})
        assert api.send_2fa_code("123456") is True
        service.trust_session.assert_called_once()
        assert api._stale_session_rejection is False
        assert api.state == IntegrationState.OK

    def test_400_incorrect_code_is_a_rejection(self):
        api, service = self._api()
        service._validate_sms_code.side_effect = self._exc(
            400, {"scnt": "x"},
            {"service_errors": [{"code": "-21669", "title": "Incorrect Verification Code"}],
             "hasError": True})
        assert api.send_2fa_code("000000") is False
        service.trust_session.assert_not_called()
        assert api._stale_session_rejection is False

    def test_409_without_session_token_is_still_stale_session(self):
        api, service = self._api()
        service._validate_sms_code.side_effect = self._exc(409, {"scnt": "x"}, {"mode": "sms"})
        assert api.send_2fa_code("123456") is False
        service.trust_session.assert_not_called()
        assert api._stale_session_rejection is True

    def test_service_errors_win_over_session_token(self):
        api, service = self._api()
        service._validate_sms_code.side_effect = self._exc(
            409, {"X-Apple-Session-Token": "tok"}, {"hasError": True, "service_errors": [{}]})
        assert api.send_2fa_code("123456") is False

    def test_409_with_token_and_non_json_body_is_accepted(self):
        # Apple echoes the challenge form back on acceptance — that body need not
        # be JSON; the fresh session token alone marks the code as accepted.
        api, service = self._api()
        exc = self._exc(409, {"X-Apple-Session-Token": "tok", "scnt": "x"}, {})
        exc.response.json.side_effect = ValueError("not json")
        exc.response.text = "<html>challenge form</html>"
        service._validate_sms_code.side_effect = exc
        assert api.send_2fa_code("123456") is True
        service.trust_session.assert_called_once()

    def test_non_str_session_token_is_not_accepted(self):
        # The token must be a real str: a bare Mock for headers (every .get() a
        # truthy Mock) must classify as a rejection, not as accepted.
        api, service = self._api()
        exc = self._exc(409, {}, {"mode": "sms"})
        exc.response.headers = Mock()
        service._validate_sms_code.side_effect = exc
        assert api.send_2fa_code("123456") is False
        service.trust_session.assert_not_called()
        assert api._stale_session_rejection is True


class TestSmsOnlyCodeRequest:
    """pyicloud 2.6.x GETs /verify/trusteddevice after signin, which pops an
    approval dialog on every Apple device; Floxy submits the SMS code only."""

    @patch('integrations.icloud.ICloudIntegration.PyiCloudService')
    def test_service_gets_sms_only_request_hook_and_authenticates(self, mock_pyicloud):
        api = iCloudAPI(config=VALID_CONFIG)
        with patch.object(api, '_session_dir', return_value="/tmp/x"):
            service = api._new_service()
        assert mock_pyicloud.call_args.kwargs['authenticate'] is False
        service.authenticate.assert_called_once()
        # pyicloud 2026.10 calls request_2fa_code (public) after an SRP signin that ends in a challenge
        assert service.request_2fa_code.__func__ == api._signin_2fa_code_hook

    def test_sms_only_request_puts_phone_without_device_push(self):
        from integrations.icloud.ICloudIntegration import _request_sms_code_only
        service = Mock()
        service._auth_endpoint = "https://idmsa.apple.com/appleauth/auth"
        phone = Mock(device_id=1)
        phone.as_phone_number_payload.return_value = {"id": 1, "nonFTEU": False}
        service._trusted_phone_number.return_value = phone
        _request_sms_code_only(service)
        service.session.get.assert_not_called()
        url = service.session.put.call_args.args[0]
        assert url.endswith("/verify/phone")
        assert service.session.put.call_args.kwargs['json'] == {
            "phoneNumber": {"id": 1, "nonFTEU": False}, "mode": "sms"}

    def test_sms_only_request_without_phone_does_nothing(self):
        from integrations.icloud.ICloudIntegration import _request_sms_code_only
        service = Mock()
        service._trusted_phone_number.return_value = None
        _request_sms_code_only(service)
        service.session.put.assert_not_called()

    def test_sms_only_request_failure_does_not_raise(self):
        # The hook runs inside pyicloud's authenticate(); a PUT failure escaping
        # here would fail the whole signin instead of just skipping the SMS ask.
        from integrations.icloud.ICloudIntegration import _request_sms_code_only
        service = Mock()
        service._auth_endpoint = "https://idmsa.apple.com/appleauth/auth"
        service._trusted_phone_number.return_value = Mock(device_id=1)
        service.session.put.side_effect = RuntimeError("network down")
        _request_sms_code_only(service)


class TestTolerantMfaPage:
    """2026-09-06: Apple served GET /appleauth/auth with HTTP 409 and the full 2FA
    page; pyicloud raised on the status and the signin never reached the SMS step."""

    BOOT_ARGS = (
        '{"direct": {"authInitialRoute": "verifyPhone", "hasTrustedDevices": false, '
        '"twoSV": {"authFactors": ["sms"], "bridgeInitiateData": {"phoneNumberVerification": '
        '{"trustedPhoneNumber": {"id": 1, "obfuscatedNumber": "•••-••• •• 05", "pushMode": "sms"}}}}}}'
    )
    HTML = ('<!DOCTYPE html><html><head><script type="application/json" class="boot_args">'
            + BOOT_ARGS + '</script></head><body></body></html>')

    @staticmethod
    def _service(status, text="", json_value=None):
        service = Mock()
        service._auth_endpoint = "https://idmsa.apple.com/appleauth/auth"
        service._get_auth_headers.return_value = {"Accept": "text/html", "scnt": "abc"}
        response = Mock(status_code=status, text=text, reason="Conflict")
        if json_value is None:
            response.json.side_effect = ValueError("not json")
        else:
            response.json.return_value = json_value
        service.session.request_raw.return_value = response
        return service

    @patch('integrations.icloud.ICloudIntegration.PyiCloudService')
    def test_service_gets_tolerant_page_hook(self, mock_pyicloud):
        from integrations.icloud.ICloudIntegration import _get_mfa_auth_options_tolerant
        api = iCloudAPI(config=VALID_CONFIG)
        with patch.object(api, '_session_dir', return_value="/tmp/x"):
            service = api._new_service()
        assert service._get_mfa_auth_options.__func__ is _get_mfa_auth_options_tolerant

    def test_409_html_page_is_parsed(self):
        from integrations.icloud.ICloudIntegration import _get_mfa_auth_options_tolerant
        service = self._service(409, text=self.HTML)
        options = _get_mfa_auth_options_tolerant(service)
        method, url = service.session.request_raw.call_args.args
        assert (method, url) == ("GET", "https://idmsa.apple.com/appleauth/auth")
        assert service.session.request_raw.call_args.kwargs["headers"]["Accept"] == "text/html"
        assert options["authInitialRoute"] == "verifyPhone"
        assert options["authFactors"] == ["sms"]
        assert options["trustedPhoneNumber"]["obfuscatedNumber"] == "•••-••• •• 05"
        assert service._hsa2_boot_context.auth_initial_route == "verifyPhone"
        service._clear_trusted_device_bridge_state.assert_called_once()
        service._set_two_factor_delivery_state.assert_called_once_with("unknown")

    def test_json_page_keeps_apple_fields(self):
        from integrations.icloud.ICloudIntegration import _get_mfa_auth_options_tolerant
        payload = {"authInitialRoute": "verifyPhone", "authFactors": ["sms"], "extra": 1,
                   "phoneNumberVerification": {"trustedPhoneNumber": {"id": 2, "pushMode": "sms"}}}
        service = self._service(200, json_value=payload)
        options = _get_mfa_auth_options_tolerant(service)
        assert options["extra"] == 1
        assert options["trustedPhoneNumber"]["id"] == 2

    def test_other_status_raises(self):
        from pyicloud.exceptions import PyiCloudAPIResponseException
        from integrations.icloud.ICloudIntegration import _get_mfa_auth_options_tolerant
        service = self._service(503, text="<html>down</html>")
        with pytest.raises(PyiCloudAPIResponseException):
            _get_mfa_auth_options_tolerant(service)
        service._clear_trusted_device_bridge_state.assert_not_called()

    def test_409_without_boot_args_raises(self):
        from pyicloud.exceptions import PyiCloudTrustedDevicePromptException
        from integrations.icloud.ICloudIntegration import _get_mfa_auth_options_tolerant
        service = self._service(409, text="<html><body>Something went wrong</body></html>")
        with pytest.raises(PyiCloudTrustedDevicePromptException):
            _get_mfa_auth_options_tolerant(service)


class TestTokenLoginLogged:
    """The token login is the first step of every sign-in; on 2026-10-16 two Find My
    sign-ins fell through it to a password sign-in Apple challenged, and the log could
    not say whether Apple had refused the session token or distrusted the browser."""

    @staticmethod
    def _service(trust_token=None, data=None):
        service = Mock()
        service.session.data = {"trust_token": trust_token} if trust_token else {}
        service.data = data or {}
        return service

    @patch('integrations.icloud.ICloudIntegration.PyiCloudService')
    def test_service_gets_logged_token_login(self, mock_pyicloud):
        from integrations.icloud.ICloudIntegration import _authenticate_with_token_logged
        from pyicloud import PyiCloudService
        assert PyiCloudService._authenticate_with_token is _authenticate_with_token_logged

    def test_untrusted_answer_is_logged_with_apples_verdict_and_reraised(self):
        from pyicloud.exceptions import PyiCloud2FARequiredException
        from integrations.icloud import ICloudIntegration as module
        service = self._service(trust_token="tok", data={"hsaTrustedBrowser": False})
        with patch.object(module, '_original_authenticate_with_token',
                          side_effect=PyiCloud2FARequiredException("m@mlab.test", Mock())), \
                patch.object(module, 'info') as mock_info:
            with pytest.raises(PyiCloud2FARequiredException):
                module._authenticate_with_token_logged(service)
        line = mock_info.call_args.args[0]
        assert "hsaTrustedBrowser=False" in line
        assert "trust token sent: True" in line

    def test_refused_token_is_logged_with_http_status(self):
        from pyicloud.exceptions import PyiCloudFailedLoginException, PyiCloudAPIResponseException
        from integrations.icloud import ICloudIntegration as module
        service = self._service()

        def refuse(api, require_trust=True):
            try:
                raise PyiCloudAPIResponseException("Authentication required", 421)
            except PyiCloudAPIResponseException as e:
                raise PyiCloudFailedLoginException("Invalid authentication token.", e) from e

        with patch.object(module, '_original_authenticate_with_token', side_effect=refuse), \
                patch.object(module, 'info') as mock_info:
            with pytest.raises(PyiCloudFailedLoginException):
                module._authenticate_with_token_logged(service)
        line = mock_info.call_args.args[0]
        assert "HTTP 421" in line
        assert "trust token sent: False" in line

    def test_trusted_answer_logs_nothing_at_info(self):
        from integrations.icloud import ICloudIntegration as module
        service = self._service(trust_token="tok", data={"hsaTrustedBrowser": True})
        with patch.object(module, '_original_authenticate_with_token', return_value=None), \
                patch.object(module, 'info') as mock_info:
            module._authenticate_with_token_logged(service)
        mock_info.assert_not_called()


class TestSigninSmsGate:
    """pyicloud calls _request_2fa_code on its own after ANY SRP signin ending in
    a 2FA challenge — including the find-my-iphone re-auth that the devices
    health probe forces on every restart of a working-but-untrusted session,
    which sent one unused SMS per server restart (Aug 31–Sep 1 2026). The hook
    only texts while Floxy runs a signin it intends to complete with a code."""

    def test_hook_is_suppressed_outside_deliberate_signins(self):
        api = iCloudAPI(config=VALID_CONFIG)
        service = Mock()
        api._signin_2fa_code_hook(service)
        service.session.put.assert_not_called()
        service._trusted_phone_number.assert_not_called()

    def test_hook_requests_sms_during_deliberate_signin(self):
        api = iCloudAPI(config=VALID_CONFIG)
        service = Mock()
        service._auth_endpoint = "https://idmsa.apple.com/appleauth/auth"
        phone = Mock(device_id=1)
        phone.as_phone_number_payload.return_value = {"id": 1, "nonFTEU": False}
        service._trusted_phone_number.return_value = phone
        api._signin_sms_wanted = True
        api._signin_2fa_code_hook(service)
        assert service.session.put.call_args.args[0].endswith("/verify/phone")

    @patch('integrations.icloud.ICloudIntegration.PyiCloudService')
    def test_startup_authenticate_opens_the_gate_and_closes_it_after(self, mock_pyicloud):
        service = _service_with_sms_challenge()
        seen = []
        service.authenticate.side_effect = lambda: seen.append(api._signin_sms_wanted)
        mock_pyicloud.return_value = service
        api = iCloudAPI(config=VALID_CONFIG)
        with patch.object(api, '_session_dir', return_value="/tmp/x"):
            api._ensure_authenticated()
        assert seen == [True]
        assert api._signin_sms_wanted is False

    @patch('integrations.icloud.ICloudIntegration.PyiCloudService')
    def test_fresh_signin_opens_the_gate_and_closes_it_even_on_failure(self, mock_pyicloud):
        from pyicloud.exceptions import PyiCloudFailedLoginException
        service = _service_with_sms_challenge()
        seen = []
        def fail():
            seen.append(api._signin_sms_wanted)
            raise PyiCloudFailedLoginException("bad password")
        service.authenticate.side_effect = fail
        mock_pyicloud.return_value = service
        api = iCloudAPI(config=VALID_CONFIG)
        with patch.object(api, '_session_dir', return_value="/tmp/x"), \
             pytest.raises(iCloudAuthenticationError):
            api._fresh_password_signin()
        assert seen == [True]
        assert api._signin_sms_wanted is False

    @patch('integrations.icloud.ICloudIntegration.PyiCloudService')
    def test_service_reauth_after_startup_sends_no_sms(self, mock_pyicloud):
        # The restart regression: after startup finishes, a service re-auth
        # (e.g. the devices probe) making pyicloud call _request_2fa_code
        # must not text a code.
        service = Mock()
        service.requires_2fa = True
        service._auth_data = {}
        mock_pyicloud.return_value = service
        api = iCloudAPI(config=VALID_CONFIG)
        with patch.object(api, '_session_dir', return_value="/tmp/x"):
            api._ensure_authenticated()
        service._request_2fa_code()
        service.session.put.assert_not_called()


class TestBrief:
    def test_short_message_is_unchanged(self):
        from integrations.icloud.ICloudIntegration import _brief
        assert _brief(ValueError("Invalid email/password combination.")) == "Invalid email/password combination."

    def test_whitespace_collapses_and_long_tail_is_cut(self):
        from integrations.icloud.ICloudIntegration import _brief
        text = "reason (409):\n\n  <html>\n" + "x" * 1000
        out = _brief(RuntimeError(text), limit=50)
        assert out.startswith("reason (409): <html> xxx")
        assert out.endswith("chars, RuntimeError]")
        assert len(out) < 100


def _seen(name, uuid, model="iPhone 15", ba_uuid=""):
    """A Find My device record as initClient returns it."""
    return {"id": uuid, "baUUID": ba_uuid, "name": name, "deviceDisplayName": model, "deviceClass": "iPhone"}


class TestDeviceListing:
    @pytest.fixture
    def api(self):
        api = iCloudAPI(config={"apple_id": "a@b.c", "password": "pw", "devices": [
            {"id": "uuid-dad", "display_name": "dad"}]})
        api._api = _find_my_answers(Mock(), [_seen("Dad's iPhone", "uuid-dad"), _seen("Anna's iPhone", "uuid-anna")])
        with patch.object(api, "_ensure_authenticated"), patch.object(api, "requires_2fa", return_value=False):
            yield api

    def test_listing_marks_configured_devices_with_their_slot(self, api):
        devices = api.get_icloud_devices()
        assert [(d["name"], d["configured_as"], d["slot"]) for d in devices] == \
            [("Dad's iPhone", "dad", 0), ("Anna's iPhone", None, None)]

    def test_listing_marks_an_entry_configured_by_the_old_bauuid(self, api):
        _find_my_answers(api._api, [_seen("Dad's iPhone", "fmip-dad", ba_uuid="uuid-dad")])
        devices = api.get_icloud_devices()
        assert [(d["id"], d["configured_as"]) for d in devices] == [("fmip-dad", "dad")]

    def test_listing_survives_a_broken_devices_entry(self):
        """A half-written entry (id set, display_name not yet) must not hide the listing the
        LLM needs to finish it."""
        api = iCloudAPI(config={"apple_id": "a@b.c", "password": "pw", "devices": [{"id": "uuid-dad"}]})
        api._api = _find_my_answers(Mock(), [_seen("Dad's iPhone", "uuid-dad")])
        with patch.object(api, "_ensure_authenticated"), patch.object(api, "requires_2fa", return_value=False):
            devices = api.get_icloud_devices()
        assert [(d["name"], d["configured_as"]) for d in devices] == [("Dad's iPhone", None)]


class TestTermsOfServiceNotification:
    """Apple blocking the account behind new terms is reported to the user once."""

    _NOTIFY_PATH = 'assistant.Notifier.NullNotifier.send_message'
    _TERMS_ERROR = Exception("You must accept the updated terms of service to continue. "
                             "Set --accept-terms to accept them.")

    def _api_whose_reauth_fails(self, exc):
        """A session Find My rejects, whose password re-sign-in Apple answers with `exc`."""
        from pyicloud.exceptions import PyiCloudAuthRequiredException, PyiCloudFailedLoginException
        api = iCloudAPI(config=VALID_CONFIG)
        mock_service = _find_my_answers(Mock(), exc=PyiCloudAuthRequiredException("m@mlab.test", Mock()))
        mock_service.requires_2fa = False
        mock_service.authenticate.side_effect = PyiCloudFailedLoginException(str(exc))
        api._api = mock_service
        return api

    def test_terms_error_notifies_user_once_and_parks(self):
        """The terms block surfaces where the sign-in happens: the deliberate Find My re-auth.
        The second attempt is not made — sign-ins are parked until the user confirms."""
        api = self._api_whose_reauth_fails(self._TERMS_ERROR)
        with patch(self._NOTIFY_PATH) as mock_notify:
            for _ in range(2):
                assert api._sign_in_for_find_my("test")
        mock_notify.assert_called_once()
        assert "icloud.com" in mock_notify.call_args.args[0]
        api._api.authenticate.assert_called_once()

    def test_a_connection_failure_neither_notifies_nor_parks(self):
        from pyicloud.exceptions import PyiCloudConnectionException
        api = self._api_whose_reauth_fails(None)
        api._api.authenticate.side_effect = PyiCloudConnectionException("connection reset")
        with patch(self._NOTIFY_PATH) as mock_notify:
            for _ in range(2):
                assert "connection reset" in api._sign_in_for_find_my("test")
        mock_notify.assert_not_called()
        assert api._parked_reason() is None
        assert api._api.authenticate.call_count == 2

    @patch('integrations.icloud.ICloudIntegration.PyiCloudService')
    def test_a_terms_block_at_signin_is_reported_once(self, mock_pyicloud):
        """A gated account fails at signin, before any device listing — the path a first run hits."""
        from pyicloud.exceptions import PyiCloudFailedLoginException
        mock_pyicloud.side_effect = PyiCloudFailedLoginException(str(self._TERMS_ERROR))
        api = iCloudAPI(config=VALID_CONFIG)
        with patch(self._NOTIFY_PATH) as mock_notify:
            for _ in range(2):
                with pytest.raises(iCloudAuthenticationError):
                    api._ensure_authenticated()
        mock_notify.assert_called_once()
        assert "icloud.com" in mock_notify.call_args.args[0]


NOTE_PATH = 'integrations.icloud.ICloudIntegration.system_note'
LIVE_PATH = 'integrations.icloud.ICloudIntegration.SmsCodeReader.live'


class TestReaderGate:
    """A code requested on a timer while nothing reads the Mac's text messages only SMS-bombs
    the phone (a beta tester, 2026-10-16): with no live reader the flow requests nothing and
    tells the chats once that the user enters the code."""

    def _api_with_open_challenge(self):
        api = iCloudAPI(config=VALID_CONFIG)
        api._api = _service_with_sms_challenge()
        api._challenge_from_fresh_signin = True
        return api

    def test_no_reader_means_no_sms_request_and_no_fresh_signin(self):
        api = self._api_with_open_challenge()
        with patch(LIVE_PATH, return_value=[]), patch(NOTE_PATH), \
             patch.object(api, 'request_sms_2fa') as mock_request, \
             patch.object(api, '_fresh_password_signin') as mock_fresh, \
             patch.object(api, '_ensure_authenticated'):
            api._auto_request_sms("test")
            api._auto_request_sms("test", fresh_signin=True)
        mock_request.assert_not_called()
        mock_fresh.assert_not_called()
        assert api._auto_sms_limiter.check() is None, "nothing was requested, so nothing counts"

    def test_the_chats_hear_once_per_challenge_that_the_user_enters_the_code(self):
        api = self._api_with_open_challenge()
        with patch(LIVE_PATH, return_value=[]), patch(NOTE_PATH) as note, \
             patch.object(api, 'request_sms_2fa'), patch.object(api, '_ensure_authenticated'):
            api._auto_request_sms("test")
            api.on_needs_2fa()
            api._auto_request_sms("test")
        note.assert_called_once()
        text = note.call_args.args[1]
        assert "icloud_login_code" in text and "iCloud tab" in text

    def test_no_note_without_an_open_challenge(self):
        api = iCloudAPI(config=VALID_CONFIG)
        api._api = None
        with patch(LIVE_PATH, return_value=[]), patch(NOTE_PATH) as note:
            api._auto_request_sms("test")
        note.assert_not_called()

    def test_an_untrusted_token_session_is_not_told_a_code_was_texted(self):
        """pyicloud flags a token-restored session untrusted too, with no challenge behind it:
        Apple texted nothing, and "enter the code" sends the user looking for a text that never
        comes (2026-10-19, the install with no reader)."""
        api = iCloudAPI(config=VALID_CONFIG)
        api._api = _service_with_sms_challenge()
        api._challenge_from_fresh_signin = False
        with patch(LIVE_PATH, return_value=[]), patch(NOTE_PATH) as note, patch.object(api, 'request_sms_2fa') as request:
            api._auto_request_sms("test")
        note.assert_not_called()
        request.assert_not_called()

    def test_a_live_reader_lets_the_request_through(self):
        api = self._api_with_open_challenge()
        with patch(NOTE_PATH), patch.object(api, 'request_sms_2fa', return_value="•••05") as mock_request, \
             patch.object(api, '_ensure_authenticated'):
            api._auto_request_sms("test")
        mock_request.assert_called_once()


class TestChallengeNote:
    """The sign-in that ends on a 2FA challenge tells the chats what happens next: the server
    reads the texted code itself, or the user has to enter it."""

    def test_with_a_reader_the_note_says_the_server_reads_the_code(self):
        api = iCloudAPI(config=VALID_CONFIG)
        api._api = _service_with_sms_challenge()
        with patch(NOTE_PATH) as note:
            api._arm_auto_sent_code(1000.0)
        note.assert_called_once()
        text = note.call_args.args[1]
        assert "by itself" in text and "icloud_login_code" in text

    def test_without_a_reader_the_note_asks_the_user_for_the_code(self):
        api = iCloudAPI(config=VALID_CONFIG)
        api._api = _service_with_sms_challenge()
        with patch(LIVE_PATH, return_value=[]), patch(NOTE_PATH) as note:
            api._arm_auto_sent_code(1000.0)
        text = note.call_args.args[1]
        assert "iMessage feed is not running" in text and "icloud_login_code" in text and "Ctrl-E" in text

    def test_the_login_hint_follows_the_reader(self):
        """The setup commands answer "needs 2FA" with the hint: with a reader live the user has
        nothing to type, without one the hint names the two ways in."""
        api = iCloudAPI(config=VALID_CONFIG)
        with patch(LIVE_PATH, return_value=["iMessage"]):
            live = api.current_login_hint()
        with patch(LIVE_PATH, return_value=[]):
            alone = api.current_login_hint()
        assert "by itself" in live and "Ctrl-E" not in live
        assert "icloud_login_code" in alone and "Ctrl-E" in alone and "by itself" not in alone

    @pytest.mark.parametrize("in_ftue, pushed", [(True, False), (False, True)])
    def test_during_the_setup_the_note_is_a_history_row_only(self, in_ftue, pushed):
        """The assistant's reply to the setup command that raised the challenge tells the user;
        outside the setup nothing else would, so the chats hear the note."""
        api = iCloudAPI(config=VALID_CONFIG)
        api._api = _service_with_sms_challenge()
        with patch(NOTE_PATH) as note, patch('assistant.FirstRun.in_ftue', return_value=in_ftue):
            api._arm_auto_sent_code(1000.0)
        assert note.call_args.kwargs["push"] is pushed

    def test_one_note_per_signin_a_new_signin_gets_its_own(self):
        api = iCloudAPI(config=VALID_CONFIG)
        api._api = _service_with_sms_challenge()
        with patch(NOTE_PATH) as note:
            api._arm_auto_sent_code(1000.0)
            api._arm_auto_sent_code(1000.0)
            assert note.call_count == 1
            api._arm_auto_sent_code(2000.0)
        assert note.call_count == 2

    def test_a_signin_without_a_challenge_says_nothing(self):
        api = iCloudAPI(config=VALID_CONFIG)
        api._api = _service_with_sms_challenge(requires_2fa=False)
        with patch(NOTE_PATH) as note:
            api._arm_auto_sent_code(1000.0)
        note.assert_not_called()


class TestParkFingerprint:
    """A park is tied to the credentials Apple refused: new ones lift it (a typo on the first
    set-up is not a security refusal — a beta tester's corrected password never reached Apple
    on 2026-10-16), the same ones keep it."""

    _NOTIFY_PATH = 'assistant.Notifier.NullNotifier.send_message'

    def _parked(self, config):
        from pyicloud.exceptions import PyiCloudFailedLoginException
        api = iCloudAPI(config=config)
        with patch(self._NOTIFY_PATH):
            api._park(PyiCloudFailedLoginException("Invalid email/password combination."))
        return api

    def test_the_same_credentials_keep_the_park_across_instances(self, icloud_session_dir):
        self._parked(VALID_CONFIG)
        again = iCloudAPI(config=dict(VALID_CONFIG))
        assert "Invalid email/password" in again._parked_reason()
        assert (icloud_session_dir / again.PARK_FILE).exists()

    @pytest.mark.parametrize("changed", [{"password": "corrected"}, {"apple_id": "other@example.com"}])
    def test_changed_credentials_lift_the_park_and_remove_the_file(self, icloud_session_dir, changed):
        self._parked(VALID_CONFIG)
        corrected = iCloudAPI(config={**VALID_CONFIG, **changed})
        assert corrected._parked_reason() is None
        assert not (icloud_session_dir / corrected.PARK_FILE).exists()

    def test_the_park_file_holds_a_digest_not_the_password(self, icloud_session_dir):
        api = self._parked(VALID_CONFIG)
        data = json.loads((icloud_session_dir / api.PARK_FILE).read_text())
        assert data["credentials"] and VALID_CONFIG["password"] not in json.dumps(data)
        assert VALID_CONFIG["apple_id"] not in json.dumps(data)

    def test_a_park_file_from_before_the_fingerprint_still_parks(self, icloud_session_dir):
        api = iCloudAPI(config=VALID_CONFIG)
        icloud_session_dir.mkdir(parents=True, exist_ok=True)
        (icloud_session_dir / api.PARK_FILE).write_text(json.dumps({"at": time.time(), "reason": "locked for security reasons"}))
        assert "locked for security reasons" in api._parked_reason()

    def test_a_lifted_park_lets_the_signin_run(self, icloud_session_dir):
        self._parked(VALID_CONFIG)
        corrected = iCloudAPI(config={**VALID_CONFIG, "password": "corrected"})
        with patch.object(corrected, '_new_service', return_value=_service_with_sms_challenge(requires_2fa=False)) as new_service, \
             patch(NOTE_PATH):
            corrected._ensure_authenticated()
        new_service.assert_called_once()


class TestSubmitUserCode:
    """The code the user read off the phone and told the assistant (icloud_login_code)."""

    _MONITOR_PATH = 'assistant.integration.IntegrationMonitor.IntegrationMonitor'

    def _api_awaiting_code(self):
        api = iCloudAPI(config=VALID_CONFIG)
        api._api = _service_with_sms_challenge()
        api._challenge_from_fresh_signin = True
        api._sms_requested_at = time.time()
        return api

    @pytest.mark.parametrize("code", ["12345", "1234567", "abcdef", ""])
    def test_a_non_six_digit_code_is_refused_before_apple_is_asked(self, code):
        api = self._api_awaiting_code()
        with patch.object(api, 'send_2fa_code') as send:
            text = api.submit_user_code(code)
        send.assert_not_called()
        assert "6 digits" in text

    def test_without_an_open_challenge_it_says_so(self):
        api = iCloudAPI(config=VALID_CONFIG)
        api._api = None
        with patch.object(api, 'send_2fa_code') as send:
            text = api.submit_user_code("123456")
        send.assert_not_called()
        assert "not waiting" in text

    def test_a_code_for_an_untrusted_token_session_is_not_sent_to_apple(self):
        """No challenge is open on such a session, so Apple would answer "rejected" to any
        code; the user hears that iCloud is not waiting for one instead."""
        api = iCloudAPI(config=VALID_CONFIG)
        api._api = _service_with_sms_challenge()
        api._challenge_from_fresh_signin = False
        with patch.object(api, 'send_2fa_code') as send:
            text = api.submit_user_code("123456")
        send.assert_not_called()
        assert "not waiting" in text and "rejected" not in text

    def test_an_accepted_code_signs_in_clears_the_pending_request_and_rechecks(self):
        api = self._api_awaiting_code()
        with patch.object(api, 'send_2fa_code', return_value=True) as send, patch(self._MONITOR_PATH) as monitor:
            text = api.submit_user_code("123 456")
        send.assert_called_once_with("123456")
        assert api._sms_requested_at is None
        monitor.get_instance.return_value.check_now.assert_called_once_with("iCloud")
        assert "accepted" in text

    def test_a_rejected_code_names_the_way_out(self):
        api = self._api_awaiting_code()
        with patch.object(api, 'send_2fa_code', return_value=False), patch(self._MONITOR_PATH) as monitor:
            text = api.submit_user_code("123456")
        monitor.get_instance.return_value.check_now.assert_not_called()
        assert "rejected" in text and "icloud_unlocked" in text
        assert api._sms_requested_at is not None, "a rejected manual code does not consume the auto-submit's shot"


class TestFindMyWaitsOutTheChallenge:
    """A ring or a location lookup asked while the session's re-sign-in has a 2FA challenge open
    waits for the auto-2FA flow to answer it (2026-10-24: the command failed at once, the code was
    accepted 21 s later, and the phone never rang) — unless nothing will answer it, or the
    budget runs out."""

    def _device(self, name="iPhone"):
        device = Mock()
        device.data = {"baUUID": "device-1-uuid", "name": name}
        return device

    def _challenged_api(self, device):
        service = Mock()
        service.requires_2fa = True
        service._auth_data = {"phoneNumberVerification": {}}
        service.devices = [device]
        api = iCloudAPI(config=VALID_CONFIG)
        api._api = service
        api._challenge_from_fresh_signin = True
        api._sms_requested_at = time.time()
        return api, service

    def test_the_ring_happens_once_the_code_is_accepted(self):
        device = self._device()
        api, service = self._challenged_api(device)

        def code_accepted(timeout=None):
            service.requires_2fa = False
            api._challenge_answered.set()
            return True

        with patch.object(api._challenge_answered, "wait", side_effect=code_accepted) as wait:
            result = api.play_sound("iPhone")
        assert "Playing sound on iPhone" in result
        device.play_sound.assert_called_once()
        assert wait.called

    def test_a_location_lookup_waits_the_same_way(self):
        device = self._device()
        device.location = {"isOld": False, "latitude": 59.3, "longitude": 18.1}
        api, service = self._challenged_api(device)

        def code_accepted(timeout=None):
            service.requires_2fa = False
            return True

        with patch.object(api._challenge_answered, "wait", side_effect=code_accepted), \
             patch.object(api, "_location_for_spec", return_value="iPhone is at home") as locate:
            assert api.get_location("iPhone") == "iPhone is at home"
        locate.assert_called_once()

    def test_a_challenge_that_closed_before_the_sign_in_finished_is_given_time_to_finish(self):
        """2026-11-11: pyicloud dropped its 2FA flag a second before the session was trusted, the
        lookup went ahead in between and Find My refused it."""
        device = self._device()
        api, service = self._challenged_api(device)
        waits = []

        def sign_in_progresses(timeout=None):
            waits.append(service.requires_2fa)
            if service.requires_2fa:
                service.requires_2fa = False    # the code is accepted, the trust step still runs
                return False
            api._challenge_answered.set()       # trusted
            return True

        device.play_sound.side_effect = lambda: None if api._challenge_answered.is_set() else pytest.fail("too early")
        with patch.object(api._challenge_answered, "wait", side_effect=sign_in_progresses):
            assert "Playing sound on iPhone" in api.play_sound("iPhone")
        assert waits == [True, False]

    def test_find_my_refusing_right_after_the_code_is_asked_once_more_without_a_new_sign_in(self):
        device_fail = self._device()
        device_fail.play_sound.side_effect = Exception("Re-authentication required")
        device_late = self._device()
        device_late.play_sound.side_effect = Exception("Re-authentication required for account: m@mlab.test")
        device_ok = self._device()
        service = Mock()
        service.requires_2fa = False
        service._auth_data = {}
        type(service).devices = PropertyMock(side_effect=[[device_fail], [device_late], [device_ok]])

        def reauth_opens_a_challenge(force_refresh=False):
            service.requires_2fa = True
            service._auth_data = {"phoneNumberVerification": {}}
        service.authenticate.side_effect = reauth_opens_a_challenge

        api = iCloudAPI(config=VALID_CONFIG)
        api._api = service
        api._sms_requested_at = time.time()

        def code_accepted(timeout=None):
            service.requires_2fa = False
            api._challenge_answered.set()
            return True

        with patch.object(api._challenge_answered, "wait", side_effect=code_accepted), \
                patch.object(iCloudAPI, "SESSION_SETTLE_SECONDS", 0):
            assert "Playing sound on iPhone" in api.play_sound("iPhone")
        device_ok.play_sound.assert_called_once()
        assert service.authenticate.call_count == 1

    def test_a_challenge_nothing_reads_the_code_for_fails_at_once(self):
        device = self._device()
        api, _ = self._challenged_api(device)
        api._sms_requested_at = None
        with patch(LIVE_PATH, return_value=[]), patch.object(api._challenge_answered, "wait") as wait:
            with pytest.raises(iCloudAuthenticationError):
                api.play_sound("iPhone")
        wait.assert_not_called()
        device.play_sound.assert_not_called()

    def test_the_wait_gives_up_after_its_budget(self):
        device = self._device()
        api, _ = self._challenged_api(device)
        with patch.object(iCloudAPI, "CHALLENGE_WAIT_SECONDS", 0.01):
            with pytest.raises(iCloudAuthenticationError, match="still waiting"):
                api.play_sound("iPhone")
        device.play_sound.assert_not_called()

    def test_a_re_sign_in_that_opens_a_challenge_is_waited_out_too(self):
        """Find My rejects the cookies mid-ring, the deliberate re-sign-in ends on a challenge
        (the 2026-10-24 case): the ring waits for the code instead of failing."""
        device_fail = self._device()
        device_fail.play_sound.side_effect = Exception("Re-authentication required")
        device_ok = self._device()
        service = Mock()
        service.requires_2fa = False
        service._auth_data = {}
        type(service).devices = PropertyMock(side_effect=[[device_fail], [device_ok]])

        def reauth_opens_a_challenge(force_refresh=False):
            service.requires_2fa = True
            service._auth_data = {"phoneNumberVerification": {}}
        service.authenticate.side_effect = reauth_opens_a_challenge

        api = iCloudAPI(config=VALID_CONFIG)
        api._api = service
        api._sms_requested_at = time.time()

        def code_accepted(timeout=None):
            service.requires_2fa = False
            return True

        with patch.object(api._challenge_answered, "wait", side_effect=code_accepted) as wait:
            assert "Playing sound on iPhone" in api.play_sound("iPhone")
        assert wait.called
        device_ok.play_sound.assert_called_once()

    def test_a_new_challenge_restarts_the_wait_and_an_accepted_code_ends_it(self):
        api = iCloudAPI(config=VALID_CONFIG)
        api._challenge_answered.set()
        api._challenge_from_fresh_signin = True
        assert not api._challenge_answered.is_set()
        service = Mock()
        service.requires_2fa = True
        service.is_trusted_session = True
        service.validate_2fa_code.return_value = True
        api._api = service
        with patch.object(api, "_validate_sms_code_verbose", return_value=True):
            api.send_2fa_code("123456")
        assert api._challenge_answered.is_set()
