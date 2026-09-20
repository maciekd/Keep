import json
import threading
from unittest.mock import AsyncMock, Mock, patch

import pytest

from assistant.integration.integrations.cameras.RingIntegration import (
    RingIntegration,
    RingCameraError,
    RingConfigurationError,
    RingAuthenticationError,
    _RingTfaHandler,
)


@pytest.fixture(autouse=True)
def reset_singleton():
    RingIntegration.reset_instance()
    yield
    RingIntegration.reset_instance()


VALID_CONFIG = {
    "username": "test@example.com",
    "password": "secret",
    "token_path": "/tmp/test_ring/token.cache",
}


class TestRingIntegrationInit:
    def test_missing_ring_section_raises(self):
        mock_cm = Mock()
        mock_cm.get.side_effect = KeyError("ring")
        with pytest.raises(RingConfigurationError, match="Missing 'ring' section"):
            RingIntegration(config_manager=mock_cm)

    def test_missing_username_raises(self):
        with pytest.raises(RingConfigurationError, match="username"):
            RingIntegration(config={"password": "secret"})

    def test_missing_password_raises(self):
        with pytest.raises(RingConfigurationError, match="password"):
            RingIntegration(config={"username": "user@example.com"})

    def test_valid_config_creates_instance(self):
        cam = RingIntegration(config=VALID_CONFIG)
        assert cam is not None
        assert cam.get_auth_status()["status"] == "not_initialized"

    def test_singleton(self):
        mock_cm = Mock()
        mock_cm.get.return_value = VALID_CONFIG
        a = RingIntegration.get_instance(config_manager=mock_cm)
        b = RingIntegration.get_instance()
        assert a is b

    def test_reset_instance(self):
        mock_cm = Mock()
        mock_cm.get.return_value = VALID_CONFIG
        a = RingIntegration.get_instance(config_manager=mock_cm)
        RingIntegration.reset_instance()
        b = RingIntegration.get_instance(config_manager=mock_cm)
        assert a is not b


class TestRingIntegrationAuthStatus:
    def test_not_initialized(self):
        cam = RingIntegration(config=VALID_CONFIG)
        status = cam.get_auth_status()
        assert status["status"] == "not_initialized"

    def test_error_status(self):
        cam = RingIntegration(config=VALID_CONFIG)
        cam._auth_error = "Connection refused"
        status = cam.get_auth_status()
        assert status["status"] == "error"
        assert "Connection refused" in status["message"]

    def test_authenticated_status(self):
        cam = RingIntegration(config=VALID_CONFIG)
        cam._ring = Mock()
        status = cam.get_auth_status()
        assert status["status"] == "authenticated"


class TestRingIntegrationTokenPersistence:
    def test_token_updated_writes_file(self, tmp_path):
        config = {**VALID_CONFIG, "token_path": str(tmp_path / "token.cache")}
        cam = RingIntegration(config=config)
        token = {"access_token": "abc123"}
        cam._token_updated(token)
        content = json.loads((tmp_path / "token.cache").read_text())
        assert content == token

    def test_load_cached_token_success(self, tmp_path):
        token_file = tmp_path / "token.cache"
        token = {"access_token": "cached"}
        token_file.write_text(json.dumps(token))
        config = {**VALID_CONFIG, "token_path": str(token_file)}
        cam = RingIntegration(config=config)
        loaded = cam._load_cached_token()
        assert loaded == token

    def test_load_cached_token_missing_file(self, tmp_path):
        config = {**VALID_CONFIG, "token_path": str(tmp_path / "nonexistent.cache")}
        cam = RingIntegration(config=config)
        assert cam._load_cached_token() is None

    def test_load_cached_token_bad_json(self, tmp_path):
        token_file = tmp_path / "token.cache"
        token_file.write_text("not json")
        config = {**VALID_CONFIG, "token_path": str(token_file)}
        cam = RingIntegration(config=config)
        assert cam._load_cached_token() is None


class TestRingIntegrationConnect:
    """A cached token the Ring API rejects must end in a normal re-login, not a lock-out."""

    @staticmethod
    def _cam(tmp_path, token=None):
        token_file = tmp_path / "token.cache"
        if token:
            token_file.write_text(json.dumps(token))
        return RingIntegration(config={**VALID_CONFIG, "token_path": str(token_file)}), token_file

    @staticmethod
    def _ring(*create_session_results):
        ring = Mock()
        ring.async_create_session = AsyncMock(side_effect=list(create_session_results))
        ring.async_update_data = AsyncMock()
        return ring

    def test_hardware_id_stays_the_one_the_first_tokens_were_issued_under(self):
        from ring_doorbell import Auth
        with patch("uuid.getnode", return_value=0x0242AC110002):
            legacy = Auth("Floxy/1.0").hardware_id
            assert RingIntegration._hardware_id() == legacy
            assert RingIntegration(config=VALID_CONFIG)._new_auth(None).hardware_id == legacy

    def test_rejected_cached_token_is_dropped_and_login_follows(self, tmp_path):
        from ring_doorbell import AuthenticationError
        cam, token_file = self._cam(tmp_path, {"access_token": "stale"})
        stale_auth, fresh_auth = Mock(async_close=AsyncMock()), Mock(async_fetch_token=AsyncMock())
        ring = self._ring(AuthenticationError("invalid_grant"), None)
        with patch.object(cam, "_new_auth", side_effect=[stale_auth, fresh_auth]) as new_auth, \
                patch("ring_doorbell.Ring", return_value=ring), \
                patch.object(cam._tfa_handler, "start"):
            cam._ensure_connected()
        assert [c.args[0] for c in new_auth.call_args_list] == [{"access_token": "stale"}, None]
        assert not token_file.exists()
        stale_auth.async_close.assert_awaited_once()
        fresh_auth.async_fetch_token.assert_awaited_once_with("test@example.com", "secret")
        assert cam._ring is ring
        assert cam.get_auth_status()["status"] == "authenticated"

    def test_network_failure_keeps_the_cached_token(self, tmp_path):
        from ring_doorbell import RingError
        cam, token_file = self._cam(tmp_path, {"access_token": "good"})
        with patch.object(cam, "_new_auth", return_value=Mock()) as new_auth, \
                patch("ring_doorbell.Ring", return_value=self._ring(RingError("connection reset"))), \
                patch.object(cam._tfa_handler, "start") as tfa_start:
            with pytest.raises(RingAuthenticationError):
                cam._ensure_connected()
        assert token_file.exists()
        assert new_auth.call_count == 1
        tfa_start.assert_not_called()
        assert cam._ring is None

    def test_valid_cached_token_connects_without_login(self, tmp_path):
        cam, token_file = self._cam(tmp_path, {"access_token": "good"})
        auth = Mock(async_fetch_token=AsyncMock())
        with patch.object(cam, "_new_auth", return_value=auth), \
                patch("ring_doorbell.Ring", return_value=self._ring(None)):
            cam._ensure_connected()
        auth.async_fetch_token.assert_not_awaited()
        assert token_file.exists()

    def test_token_is_dropped_even_when_closing_the_rejected_session_fails(self, tmp_path):
        # A close() that raises must not leave the stale cache behind — that is the lock-out again.
        from ring_doorbell import AuthenticationError
        cam, token_file = self._cam(tmp_path, {"access_token": "stale"})
        stale_auth = Mock(async_close=AsyncMock(side_effect=OSError("session already closed")))
        fresh_auth = Mock(async_fetch_token=AsyncMock())
        ring = self._ring(AuthenticationError("invalid_grant"), None)
        with patch.object(cam, "_new_auth", side_effect=[stale_auth, fresh_auth]), \
                patch("ring_doorbell.Ring", return_value=ring), \
                patch.object(cam._tfa_handler, "start"):
            cam._ensure_connected()
        assert not token_file.exists()
        assert cam._ring is ring

    def test_login_retries_with_the_2fa_code(self, tmp_path):
        from ring_doorbell import Requires2FAError
        cam, _ = self._cam(tmp_path)
        auth = Mock(async_fetch_token=AsyncMock(side_effect=[Requires2FAError(), None]))
        ring = self._ring(None)
        with patch.object(cam, "_new_auth", return_value=auth), \
                patch("ring_doorbell.Ring", return_value=ring), \
                patch.object(cam._tfa_handler, "start"), \
                patch.object(cam._tfa_handler, "get", return_value="654321"):
            cam._ensure_connected()
        assert [c.args for c in auth.async_fetch_token.await_args_list] == [
            ("test@example.com", "secret"), ("test@example.com", "secret", "654321")]
        assert cam._ring is ring

    def test_rejected_token_without_a_2fa_code_ends_in_needs_2fa(self, tmp_path):
        # The re-login after a rejected token is a full login: no code in time leaves the
        # integration asking for one, with no session and no stale cache.
        from ring_doorbell import AuthenticationError, Requires2FAError
        from assistant.integration.Integration import IntegrationState
        cam, token_file = self._cam(tmp_path, {"access_token": "stale"})
        stale_auth = Mock(async_close=AsyncMock())
        fresh_auth = Mock(async_fetch_token=AsyncMock(side_effect=Requires2FAError()))
        with patch.object(cam, "_new_auth", side_effect=[stale_auth, fresh_auth]), \
                patch("ring_doorbell.Ring", return_value=self._ring(AuthenticationError("invalid_grant"))), \
                patch.object(cam._tfa_handler, "start"), \
                patch.object(cam._tfa_handler, "get", return_value=None):
            with pytest.raises(RingAuthenticationError, match="No 2FA code"):
                cam._ensure_connected()
        assert cam.state == IntegrationState.NEEDS_2FA
        assert not token_file.exists()
        assert cam._ring is None
        assert cam.get_auth_status()["status"] == "error"


class TestRingIntegrationFetchSnapshot:
    def test_fetch_snapshot_not_connected_raises(self):
        cam = RingIntegration(config=VALID_CONFIG)
        cam._ensure_connected = Mock(return_value=None)
        cam._ring = None
        with pytest.raises(RingCameraError, match="Not connected"):
            cam.fetch_snapshot()

    def test_fetch_snapshot_no_doorbells(self):
        cam = RingIntegration(config=VALID_CONFIG)
        cam._ring = Mock()
        mock_devices = Mock()
        mock_devices.doorbells = []
        cam._ring.devices.return_value = mock_devices
        cam._ensure_connected = Mock()
        result = cam.fetch_snapshot()
        assert result == []

    def test_fetch_snapshot_returns_content_blocks(self):
        cam = RingIntegration(config=VALID_CONFIG)
        cam._ring = Mock()

        mock_doorbell = Mock()
        mock_doorbell.name = "Front Door"
        mock_devices = Mock()
        mock_devices.doorbells = [mock_doorbell]
        cam._ring.devices.return_value = mock_devices
        cam._ensure_connected = Mock()

        fake_blocks = [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "abc"}},
            {"type": "text", "text": "Camera: Front Door"},
        ]
        cam._fetch_doorbell_snapshot = Mock(return_value=fake_blocks)

        result = cam.fetch_snapshot()
        assert len(result) == 2
        assert result[0]["type"] == "image"
        assert result[1]["text"] == "Camera: Front Door"

    def test_fetch_snapshot_handles_single_doorbell_failure(self):
        cam = RingIntegration(config=VALID_CONFIG)
        cam._ring = Mock()

        good = Mock()
        good.name = "Front"
        bad = Mock()
        bad.name = "Back"
        mock_devices = Mock()
        mock_devices.doorbells = [good, bad]
        cam._ring.devices.return_value = mock_devices
        cam._ensure_connected = Mock()

        def side_effect(doorbell):
            if doorbell is bad:
                raise RuntimeError("Camera offline")
            return [{"type": "text", "text": f"Camera: {doorbell.name}"}]

        cam._fetch_doorbell_snapshot = Mock(side_effect=side_effect)

        result = cam.fetch_snapshot()
        assert any("Front" in str(b) for b in result)
        assert any("error" in str(b) for b in result)


class TestRingTfaHandler:
    def test_submit_and_get(self):
        handler = _RingTfaHandler()
        handler.start()
        handler.submit_code("123456")
        assert handler.get(timeout=1) == "123456"

    def test_get_returns_none_on_timeout(self):
        handler = _RingTfaHandler()
        handler.start()
        assert handler.get(timeout=0.01) is None

    def test_is_waiting_false_before_start(self):
        # Regression: a freshly constructed handler must NOT report is_waiting.
        # The old implementation derived is_waiting from an unset threading.Event,
        # so it returned True at construction — causing the iMessage poller to
        # "submit" stray, unrequested Ring 2FA codes into a phantom auth flow.
        handler = _RingTfaHandler()
        assert handler.is_waiting is False

    def test_is_waiting_true_after_start(self):
        handler = _RingTfaHandler()
        handler.start()
        assert handler.is_waiting is True

    def test_is_waiting_false_after_submit(self):
        handler = _RingTfaHandler()
        handler.start()
        handler.submit_code("123456")
        assert handler.is_waiting is False


class TestAutoTfa:
    def test_submit_auto_2fa_code_when_waiting(self):
        cam = RingIntegration(config=VALID_CONFIG)
        cam._tfa_handler.start()
        cam.submit_auto_2fa_code("654321")
        assert cam._tfa_handler._code == "654321"
        assert cam.auto_2fa_message is not None

    def test_submit_auto_2fa_code_when_not_waiting(self):
        cam = RingIntegration(config=VALID_CONFIG)
        cam._tfa_handler.start()
        cam._tfa_handler.submit_code("dummy")
        cam.submit_auto_2fa_code("654321")
        assert cam.auto_2fa_message is None
        assert cam._tfa_handler._code == "dummy"  # stray code not stored

    def test_submit_auto_2fa_code_ignores_code_from_before_flow_started(self):
        # A code that arrived before this 2FA flow began belongs to an earlier
        # attempt and must not be submitted into the waiting handler.
        cam = RingIntegration(config=VALID_CONFIG)
        cam._tfa_handler.start()
        cam.submit_auto_2fa_code("654321", received_at=cam._tfa_handler.waiting_since - 30)
        assert cam._tfa_handler._code is None
        assert cam.auto_2fa_message is None

    def test_submit_auto_2fa_code_accepts_code_after_flow_started(self):
        cam = RingIntegration(config=VALID_CONFIG)
        cam._tfa_handler.start()
        cam.submit_auto_2fa_code("654321", received_at=cam._tfa_handler.waiting_since + 5)
        assert cam._tfa_handler._code == "654321"

    def test_submit_auto_2fa_code_ignored_when_never_started(self):
        # Regression for the phantom-submit bug: with no auth flow ever started,
        # an unrequested Ring 2FA arriving via iMessage must be ignored.
        cam = RingIntegration(config=VALID_CONFIG)
        cam.submit_auto_2fa_code("654321")
        assert cam.auto_2fa_message is None
        assert cam._tfa_handler._code is None

    def test_auto_2fa_message_read_and_clear(self):
        cam = RingIntegration(config=VALID_CONFIG)
        cam._auto_2fa_message = "2FA code auto-detected!"
        assert cam.auto_2fa_message == "2FA code auto-detected!"
        assert cam.auto_2fa_message is None

    def test_on_needs_2fa_starts_background_thread(self):
        cam = RingIntegration(config=VALID_CONFIG)
        with patch.object(cam, '_ensure_connected'):
            cam.on_needs_2fa()
            cam._auto_2fa_thread.join(timeout=2)
            assert not cam._auto_2fa_thread.is_alive()

    def test_on_needs_2fa_handles_auth_error(self):
        cam = RingIntegration(config=VALID_CONFIG)
        with patch.object(cam, '_ensure_connected', side_effect=RingAuthenticationError("fail")):
            cam.on_needs_2fa()
            cam._auto_2fa_thread.join(timeout=2)

    def test_on_needs_2fa_skips_if_thread_alive(self):
        cam = RingIntegration(config=VALID_CONFIG)
        blocker = threading.Event()

        def block():
            blocker.wait(timeout=5)

        cam._auto_2fa_thread = threading.Thread(target=block, daemon=True)
        cam._auto_2fa_thread.start()
        try:
            first_thread = cam._auto_2fa_thread
            cam.on_needs_2fa()
            assert cam._auto_2fa_thread is first_thread
        finally:
            blocker.set()
            cam._auto_2fa_thread.join(timeout=2)


class TestAuto2faTelegramNotification:
    """The auto-2FA path must alert the user over Telegram (added in d6574333) so
    they know a tab was authenticated without their manual involvement. Manual
    TUI entry must NOT trigger the alert — only the auto-detection path does."""

    _NOTIFY_PATH = 'assistant.Notifier.NullNotifier.send_message'

    def test_auto_detection_success_notifies_telegram(self):
        cam = RingIntegration(config=VALID_CONFIG)
        cam._tfa_handler.start()
        with patch(self._NOTIFY_PATH) as mock_notify:
            cam.submit_auto_2fa_code("654321")
        mock_notify.assert_called_once()
        message = mock_notify.call_args.args[0]
        assert "Ring" in message
        assert "auto-detected" in message

    def test_manual_tfa_handler_submit_does_not_notify(self):
        # TUI manual entry calls _tfa_handler.submit_code() directly. That path
        # must stay silent — only auto-detection (submit_auto_2fa_code) alerts.
        cam = RingIntegration(config=VALID_CONFIG)
        cam._tfa_handler.start()
        with patch(self._NOTIFY_PATH) as mock_notify:
            cam._tfa_handler.submit_code("654321")
        mock_notify.assert_not_called()

    def test_auto_detection_warns_when_not_waiting(self):
        # An unrequested Ring 2FA (not awaiting one) means someone signed in to
        # the Ring account elsewhere — surface it as a security heads-up rather
        # than silently dropping it, and do not submit the stray code.
        cam = RingIntegration(config=VALID_CONFIG)
        cam._tfa_handler.start()
        cam._tfa_handler.submit_code("first")
        with patch(self._NOTIFY_PATH) as mock_notify:
            cam.submit_auto_2fa_code("second")
        mock_notify.assert_called_once()
        message = mock_notify.call_args.args[0]
        assert "did not request" in message
        assert cam._tfa_handler._code == "first"


class TestSystemRows:
    """The 2FA lifecycle lands in the dialog history as `system` rows sourced "Ring", so the
    LLM knows a code is being fetched or was submitted — not only the chats."""

    @pytest.fixture
    def history(self):
        from assistant import SystemNote
        h = Mock()
        SystemNote.set_history(h)
        yield h
        SystemNote.set_history(None)

    @staticmethod
    def _rows(history):
        return [(c.args[0], c.args[1], c.kwargs.get("source")) for c in history.add_context_message.call_args_list]

    def test_starting_the_handler_records_that_a_code_is_being_fetched(self, history):
        _RingTfaHandler().start()
        rows = self._rows(history)
        assert len(rows) == 1
        role, text, source = rows[0]
        assert (role, source) == ("system", "Ring")
        assert text.startswith("Ring 2FA required") and "text message" in text

    def test_an_auto_submitted_code_is_recorded(self, history):
        cam = RingIntegration(config=VALID_CONFIG)
        cam._tfa_handler.start()
        cam.submit_auto_2fa_code("654321")
        rows = self._rows(history)
        assert [r[2] for r in rows] == ["Ring", "Ring"]
        assert rows[-1][1].startswith("Ring 2FA code auto-detected")
        assert cam._tfa_handler._code == "654321"

    def test_a_stray_code_is_recorded_as_a_warning_and_not_submitted(self, history):
        cam = RingIntegration(config=VALID_CONFIG)
        cam.submit_auto_2fa_code("999999")
        rows = self._rows(history)
        assert len(rows) == 1 and rows[0][2] == "Ring"
        assert "did not request" in rows[0][1]
        assert cam._tfa_handler._code is None
