import threading
import time
from unittest.mock import Mock, patch, MagicMock

import pytest

from io import BytesIO

from PIL import Image

from assistant.integration.Integration import Integration, IntegrationState
from assistant.integration.integrations.cameras.ArloCameraIntegration import (
    ArloCameraIntegration,
    ArloAuthenticationError,
    _ArloTfaHandler,
    ArloConfigurationError,
    ARLO_2FA_PATTERN,
)
from assistant.integration.integrations.cameras.jpeg_freshness import looks_like_cached_thumbnail

VALID_CONFIG = {
    "username": "test@example.com",
    "password": "secret",
    "storage_dir": "/tmp/test_arlo_floxy",
    "ignore_cameras": [],
}


@pytest.fixture(autouse=True)
def reset_singleton():
    ArloCameraIntegration.reset_instance()
    yield
    ArloCameraIntegration.reset_instance()


@pytest.fixture
def arlo(tmp_path):
    config = {**VALID_CONFIG, "storage_dir": str(tmp_path)}
    return ArloCameraIntegration(config=config)


class TestExponentialBackoff:
    def test_no_cooldown_before_first_attempt(self, arlo):
        arlo._consecutive_auth_failures = 0
        assert arlo._current_cooldown == 0

    def test_first_failure_gives_base_cooldown(self, arlo):
        arlo._consecutive_auth_failures = 1
        assert arlo._current_cooldown == ArloCameraIntegration.BASE_AUTH_COOLDOWN

    def test_second_failure_triples_cooldown(self, arlo):
        arlo._consecutive_auth_failures = 2
        expected = ArloCameraIntegration.BASE_AUTH_COOLDOWN * ArloCameraIntegration.BACKOFF_FACTOR
        assert arlo._current_cooldown == expected

    def test_cooldown_capped_at_max(self, arlo):
        arlo._consecutive_auth_failures = 8
        assert arlo._current_cooldown <= ArloCameraIntegration.MAX_AUTH_COOLDOWN

    def test_max_failures_gives_lockout(self, arlo):
        arlo._consecutive_auth_failures = ArloCameraIntegration.MAX_CONSECUTIVE_FAILURES
        assert arlo._current_cooldown == ArloCameraIntegration.LOCKOUT_COOLDOWN

    def test_auth_cooldown_remaining_zero_when_no_failures(self, arlo):
        arlo._consecutive_auth_failures = 0
        arlo._last_auth_attempt_time = time.time()
        assert arlo.auth_cooldown_remaining == 0.0

    def test_auth_cooldown_remaining_positive_during_backoff(self, arlo):
        arlo._consecutive_auth_failures = 1
        arlo._last_auth_attempt_time = time.time()
        assert arlo.auth_cooldown_remaining > 0

    def test_auth_cooldown_remaining_zero_after_elapsed(self, arlo):
        arlo._consecutive_auth_failures = 1
        arlo._last_auth_attempt_time = time.time() - ArloCameraIntegration.BASE_AUTH_COOLDOWN - 1
        assert arlo.auth_cooldown_remaining == 0.0


class TestRateLimitDetection:
    def test_429_detected_as_rate_limit(self):
        assert ArloCameraIntegration._is_rate_limit_error("login failed: 429 - Too Many Requests")

    def test_400_detected_as_rate_limit(self):
        assert ArloCameraIntegration._is_rate_limit_error("login failed: start failed: 400 - None")

    def test_401_not_rate_limit(self):
        assert not ArloCameraIntegration._is_rate_limit_error("login failed: 401 - Unauthorized")

class TestThreadingLock:
    def test_ensure_connected_uses_lock(self, arlo):
        """Verify that _ensure_connected acquires the lock."""
        lock_acquired = []

        original_do_connect = arlo._do_connect

        def mock_do_connect():
            lock_acquired.append(arlo._connect_lock.locked())

        arlo._do_connect = mock_do_connect
        arlo._ensure_connected()
        assert lock_acquired == [True]

    def test_refresh_connection_uses_lock(self, arlo):
        """Verify that refresh_connection acquires the lock."""
        lock_acquired = []

        original_do_connect = arlo._do_connect

        def mock_do_connect():
            lock_acquired.append(arlo._connect_lock.locked())

        arlo._do_connect = mock_do_connect
        try:
            arlo.refresh_connection()
        except Exception:
            pass
        assert lock_acquired == [True]

    def test_concurrent_ensure_connected_serialized(self, arlo):
        """Two threads calling _ensure_connected should not overlap."""
        call_log = []
        barrier = threading.Barrier(2, timeout=5)

        original_do_connect = arlo._do_connect

        connecting = threading.Event()

        def slow_do_connect():
            call_log.append(("enter", threading.current_thread().name))
            connecting.set()
            time.sleep(0.01)
            call_log.append(("exit", threading.current_thread().name))

        arlo._do_connect = slow_do_connect

        t1 = threading.Thread(target=arlo._ensure_connected, name="T1")
        t2 = threading.Thread(target=arlo._ensure_connected, name="T2")
        t1.start()
        assert connecting.wait(5)  # T1 holds the lock and is mid-connect when T2 arrives
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        # Both should have entered and exited, but not overlapping
        enters = [e for e in call_log if e[0] == "enter"]
        exits = [e for e in call_log if e[0] == "exit"]
        assert len(enters) == 2
        assert len(exits) == 2
        # First exit should come before second enter
        first_exit_idx = call_log.index(exits[0])
        second_enter_idx = call_log.index(enters[1])
        assert first_exit_idx < second_enter_idx


class TestConnectionFailureStates:
    @patch('assistant.integration.integrations.cameras.ArloCameraIntegration.pyaarlo.PyArlo')
    @patch('assistant.integration.integrations.cameras.ArloCameraIntegration.pyaarlo.constant')
    def test_rate_limit_429_sets_rate_limited_state(self, mock_constant, mock_pyarlo, arlo):
        mock_constant.ECDH_CURVES = ['secp384r1', 'prime256v1']
        mock_instance = Mock()
        mock_instance._be.is_connected = False
        mock_instance._last_error = "login failed: 429 - Too Many Requests"
        mock_pyarlo.return_value = mock_instance

        with pytest.raises(ArloAuthenticationError):
            arlo._do_connect()

        assert arlo.state == IntegrationState.RATE_LIMITED
        assert arlo._consecutive_auth_failures == 1

    @patch('assistant.integration.integrations.cameras.ArloCameraIntegration.pyaarlo.PyArlo')
    @patch('assistant.integration.integrations.cameras.ArloCameraIntegration.pyaarlo.constant')
    def test_rate_limit_400_sets_rate_limited_state(self, mock_constant, mock_pyarlo, arlo):
        mock_constant.ECDH_CURVES = ['secp384r1', 'prime256v1']
        mock_instance = Mock()
        mock_instance._be.is_connected = False
        mock_instance._last_error = "login failed: start failed: 400 - None"
        mock_pyarlo.return_value = mock_instance

        with pytest.raises(ArloAuthenticationError):
            arlo._do_connect()

        assert arlo.state == IntegrationState.RATE_LIMITED

    @patch('assistant.integration.integrations.cameras.ArloCameraIntegration.pyaarlo.PyArlo')
    @patch('assistant.integration.integrations.cameras.ArloCameraIntegration.pyaarlo.constant')
    def test_non_rate_limit_error_sets_needs_2fa(self, mock_constant, mock_pyarlo, arlo):
        mock_constant.ECDH_CURVES = ['secp384r1', 'prime256v1']
        mock_instance = Mock()
        mock_instance._be.is_connected = False
        mock_instance._last_error = "login failed: 2fa: code retrieval failed"
        mock_pyarlo.return_value = mock_instance

        with pytest.raises(ArloAuthenticationError):
            arlo._do_connect()

        assert arlo.state == IntegrationState.NEEDS_2FA

    @patch('assistant.integration.integrations.cameras.ArloCameraIntegration.pyaarlo.PyArlo')
    @patch('assistant.integration.integrations.cameras.ArloCameraIntegration.pyaarlo.constant')
    def test_successful_connection_resets_failures(self, mock_constant, mock_pyarlo, arlo):
        mock_constant.ECDH_CURVES = ['secp384r1', 'prime256v1']
        arlo._consecutive_auth_failures = 5

        mock_instance = Mock()
        mock_instance._be.is_connected = True
        mock_instance._devices = []
        mock_instance.cameras = []
        mock_pyarlo.return_value = mock_instance

        arlo._do_connect()

        assert arlo._consecutive_auth_failures == 0
        assert arlo.state == IntegrationState.OK

class TestMaxConsecutiveFailures:
    def test_lockout_after_max_failures(self, arlo):
        arlo._consecutive_auth_failures = ArloCameraIntegration.MAX_CONSECUTIVE_FAILURES
        arlo._last_auth_attempt_time = time.time()

        with pytest.raises(ArloAuthenticationError, match="Locked out"):
            arlo._do_connect()

    @patch('assistant.integration.integrations.cameras.ArloCameraIntegration.pyaarlo.PyArlo')
    @patch('assistant.integration.integrations.cameras.ArloCameraIntegration.pyaarlo.constant')
    def test_lockout_expires_after_24h(self, mock_constant, mock_pyarlo, arlo):
        mock_constant.ECDH_CURVES = ['secp384r1', 'prime256v1']
        mock_instance = Mock()
        mock_instance._be.is_connected = False
        mock_instance._last_error = "login failed"
        mock_pyarlo.return_value = mock_instance

        arlo._consecutive_auth_failures = ArloCameraIntegration.MAX_CONSECUTIVE_FAILURES
        arlo._last_auth_attempt_time = time.time() - ArloCameraIntegration.LOCKOUT_COOLDOWN - 1

        # Should pass the lockout check and proceed into the (mocked) connection attempt,
        # which raises a non-lockout auth error.
        with pytest.raises(Exception) as exc_info:
            arlo._do_connect()
        assert "Locked out" not in str(exc_info.value)


class TestRefreshConnection:
    def test_refresh_resets_failure_counter(self, arlo):
        arlo._consecutive_auth_failures = 5

        def mock_do_connect():
            pass  # don't actually connect

        arlo._do_connect = mock_do_connect
        arlo.refresh_connection()
        assert arlo._consecutive_auth_failures == 0

    def test_refresh_clears_arlo_and_error(self, arlo):
        arlo._arlo = Mock()
        arlo._auth_error = "some error"

        def mock_do_connect():
            pass

        arlo._do_connect = mock_do_connect
        arlo.refresh_connection()
        assert arlo._arlo is None
        assert arlo._auth_error is None


class TestCooldownPersistence:
    def test_save_and_load_state(self, arlo, tmp_path):
        arlo._last_auth_attempt_time = 1234567890.5
        arlo._consecutive_auth_failures = 3
        arlo._save_persisted_state()

        arlo._last_auth_attempt_time = 0
        arlo._consecutive_auth_failures = 0
        arlo._load_persisted_state()

        assert arlo._last_auth_attempt_time == 1234567890.5
        assert arlo._consecutive_auth_failures == 3

    def test_load_handles_old_format(self, arlo, tmp_path):
        """Backward compatibility with old cooldown file (timestamp only)."""
        cooldown_file = arlo._cooldown_file
        import os
        os.makedirs(os.path.dirname(cooldown_file), exist_ok=True)
        with open(cooldown_file, 'w') as f:
            f.write("1234567890.5")

        arlo._load_persisted_state()
        assert arlo._last_auth_attempt_time == 1234567890.5
        assert arlo._consecutive_auth_failures == 0

class TestCredentialRejection:
    """A 401 never reached the 2FA step: it must read as an error naming the credentials,
    not as a wait for a code, and it must survive the restart that follows a fix."""

    @staticmethod
    def _reject(mock_pyarlo, message="login failed: 401 - Password not correct"):
        mock_instance = Mock()
        mock_instance._be.is_connected = False
        mock_instance._last_error = message
        mock_pyarlo.return_value = mock_instance

    @patch('assistant.integration.integrations.cameras.ArloCameraIntegration.pyaarlo.PyArlo')
    @patch('assistant.integration.integrations.cameras.ArloCameraIntegration.pyaarlo.constant')
    def test_a_401_is_an_error_about_the_credentials_not_a_2fa_wait(self, mock_constant, mock_pyarlo, arlo):
        self._reject(mock_pyarlo)
        with pytest.raises(ArloAuthenticationError):
            arlo._do_connect()
        assert arlo.state == IntegrationState.ERROR
        assert arlo.needs_2fa() is False
        assert "401" in arlo.last_error and "arlo.password" in arlo.last_error

    @patch('assistant.integration.integrations.cameras.ArloCameraIntegration.pyaarlo.PyArlo')
    @patch('assistant.integration.integrations.cameras.ArloCameraIntegration.pyaarlo.constant')
    def test_the_cooldown_message_names_the_last_rejection(self, mock_constant, mock_pyarlo, arlo):
        self._reject(mock_pyarlo)
        with pytest.raises(ArloAuthenticationError):
            arlo._do_connect()
        with pytest.raises(ArloAuthenticationError, match="cooldown.*Password not correct"):
            arlo._do_connect()

    @patch('assistant.integration.integrations.cameras.ArloCameraIntegration.pyaarlo.PyArlo')
    @patch('assistant.integration.integrations.cameras.ArloCameraIntegration.pyaarlo.constant')
    def test_the_rejection_survives_a_rebuild_with_the_same_credentials(self, mock_constant, mock_pyarlo, arlo, tmp_path):
        self._reject(mock_pyarlo)
        with pytest.raises(ArloAuthenticationError):
            arlo._do_connect()
        rebuilt = ArloCameraIntegration(config={**VALID_CONFIG, "storage_dir": str(tmp_path)})
        assert rebuilt.needs_2fa() is False
        assert rebuilt._consecutive_auth_failures == 1
        assert rebuilt.auth_cooldown_remaining > 0

    @patch('assistant.integration.integrations.cameras.ArloCameraIntegration.pyaarlo.PyArlo')
    @patch('assistant.integration.integrations.cameras.ArloCameraIntegration.pyaarlo.constant')
    def test_changed_credentials_clear_the_rejection_and_the_cooldown(self, mock_constant, mock_pyarlo, arlo, tmp_path):
        self._reject(mock_pyarlo)
        with pytest.raises(ArloAuthenticationError):
            arlo._do_connect()
        rebuilt = ArloCameraIntegration(config={**VALID_CONFIG, "password": "the right one", "storage_dir": str(tmp_path)})
        assert rebuilt.needs_2fa() is True
        assert rebuilt._consecutive_auth_failures == 0
        assert rebuilt.auth_cooldown_remaining == 0

    @patch('assistant.integration.integrations.cameras.ArloCameraIntegration.pyaarlo.PyArlo')
    @patch('assistant.integration.integrations.cameras.ArloCameraIntegration.pyaarlo.constant')
    def test_a_refresh_forgets_the_rejection(self, mock_constant, mock_pyarlo, arlo):
        self._reject(mock_pyarlo)
        with pytest.raises(ArloAuthenticationError):
            arlo._do_connect()
        with pytest.raises(ArloAuthenticationError):
            arlo.refresh_connection()
        assert mock_pyarlo.call_count == 2

    @patch('assistant.integration.integrations.cameras.ArloCameraIntegration.pyaarlo.PyArlo')
    @patch('assistant.integration.integrations.cameras.ArloCameraIntegration.pyaarlo.constant')
    def test_a_2fa_failure_still_reads_as_needs_2fa(self, mock_constant, mock_pyarlo, arlo):
        self._reject(mock_pyarlo, "login failed: 2fa: code retrieval failed")
        with pytest.raises(ArloAuthenticationError):
            arlo._do_connect()
        assert arlo.state == IntegrationState.NEEDS_2FA
        assert arlo.needs_2fa() is True


class TestUnconfiguredConnect:
    def test_connecting_without_credentials_is_a_config_error_and_counts_no_failure(self, tmp_path):
        arlo = ArloCameraIntegration(config={"storage_dir": str(tmp_path)})
        with pytest.raises(ArloConfigurationError, match="username"):
            arlo.refresh_connection()
        assert arlo._consecutive_auth_failures == 0
        assert arlo.auth_cooldown_remaining == 0


class TestOnNotConfigured:
    def test_clears_session_files(self, tmp_path):
        (tmp_path / "session.pickle").write_bytes(b"stale")
        (tmp_path / "cookies.txt").write_text("stale")
        (tmp_path / "auth_cooldown.txt").write_text("123\n2")

        arlo = ArloCameraIntegration(config={"storage_dir": str(tmp_path)})
        arlo.on_not_configured()

        assert not (tmp_path / "session.pickle").exists()
        assert not (tmp_path / "cookies.txt").exists()
        assert not (tmp_path / "auth_cooldown.txt").exists()

    def test_register_calls_on_not_configured(self):
        from assistant.integration.IntegrationMonitor import IntegrationMonitor

        monitor = IntegrationMonitor()
        mock_cm = Mock()
        mock_cm.get.side_effect = KeyError("arlo")
        monitor._config_manager = mock_cm

        arlo = ArloCameraIntegration(config={"storage_dir": "/tmp/test_unused"})
        with patch.object(arlo, 'on_not_configured') as mock_cleanup:
            monitor.register(arlo)
            mock_cleanup.assert_called_once()
        assert arlo.state == IntegrationState.NOT_CONFIGURED


class TestOnNeeds2FA:
    def test_on_needs_2fa_starts_background_connect(self, arlo):
        with patch.object(arlo, '_ensure_connected') as mock_connect:
            arlo.on_needs_2fa()
            arlo._auto_2fa_thread.join(timeout=2)
            mock_connect.assert_called_once()

    def test_on_needs_2fa_skips_if_thread_alive(self, arlo):
        blocker = threading.Event()
        call_count = [0]

        def slow_connect():
            call_count[0] += 1
            blocker.wait(timeout=2)

        with patch.object(arlo, '_ensure_connected', side_effect=slow_connect):
            arlo.on_needs_2fa()
            arlo.on_needs_2fa()
            arlo.on_needs_2fa()
            blocker.set()
            arlo._auto_2fa_thread.join(timeout=2)
        assert call_count[0] == 1

    def test_register_calls_on_needs_2fa_when_2fa_needed(self, tmp_path):
        from assistant.integration.IntegrationMonitor import IntegrationMonitor

        monitor = IntegrationMonitor()
        mock_cm = Mock()
        mock_cm.get.return_value = "value"
        monitor._config_manager = mock_cm

        config = {**VALID_CONFIG, "storage_dir": str(tmp_path)}
        arlo = ArloCameraIntegration(config=config)
        with patch.object(arlo, 'on_needs_2fa') as mock_2fa:
            with patch.object(arlo, 'needs_2fa', return_value=True):
                monitor.register(arlo)
                mock_2fa.assert_called_once()
        assert arlo.state == IntegrationState.NEEDS_2FA


class TestECDHCurvePinning:
    @patch('assistant.integration.integrations.cameras.ArloCameraIntegration.pyaarlo.PyArlo')
    def test_create_pyarlo_pins_single_curve(self, mock_pyarlo, arlo, tmp_path):
        import pyaarlo.constant
        original_curves = pyaarlo.constant.ECDH_CURVES[:]

        mock_pyarlo.return_value = Mock()
        arlo._create_pyarlo(str(tmp_path))

        assert pyaarlo.constant.ECDH_CURVES == ['prime256v1']

        # Restore
        pyaarlo.constant.ECDH_CURVES[:] = original_curves


class TestTfaHandler:
    @pytest.fixture(autouse=True)
    def no_email_polling(self):
        """get() starts a Gmail poll thread; a unit test must never reach the real mailbox."""
        with patch.object(_ArloTfaHandler, '_start_email_polling'):
            yield

    def test_start_does_not_notify(self):
        """start() should never send notifications — they are deferred to get()."""
        handler = _ArloTfaHandler()
        with patch.object(_ArloTfaHandler, '_notify_2fa_needed') as mock_notify:
            handler.start()
            handler.start()
            handler.start()
            assert mock_notify.call_count == 0

    def test_get_notifies_only_once(self):
        """get() should send notification on first call, but not on subsequent calls."""
        handler = _ArloTfaHandler()
        with patch.object(_ArloTfaHandler, '_notify_2fa_needed') as mock_notify:
            handler.start()
            handler._event.set()  # unblock after start() clears it
            handler.get()
            handler.get()
            handler.get()
            assert mock_notify.call_count == 1

    def test_reset_allows_re_notification_on_get(self):
        handler = _ArloTfaHandler()
        with patch.object(_ArloTfaHandler, '_notify_2fa_needed') as mock_notify:
            handler.start()
            handler._event.set()
            handler.get()
            assert mock_notify.call_count == 1
            handler.reset()
            handler.get()
            assert mock_notify.call_count == 2

    def test_submit_code_unblocks_get(self):
        handler = _ArloTfaHandler()
        handler.start()
        result = [None]

        def wait_for_code():
            result[0] = handler.get()

        with patch.object(_ArloTfaHandler, '_notify_2fa_needed'):
            t = threading.Thread(target=wait_for_code)
            t.start()
            handler.submit_code("123456")
            t.join(timeout=2)
        assert result[0] == "123456"


class TestIntegrationMonitorRateLimitHandling:
    """Test that the IntegrationMonitor correctly handles RATE_LIMITED state."""

    def test_record_failure_skips_rate_limited(self):
        from assistant.integration.IntegrationMonitor import IntegrationMonitor

        monitor = IntegrationMonitor()
        mock_cm = Mock()
        mock_cm.get.return_value = "value"
        monitor._config_manager = mock_cm

        integration = Mock(spec=Integration)
        integration.state = IntegrationState.RATE_LIMITED
        integration._consecutive_failures = 0

        monitor._record_failure(integration, "some error")

        # Should not increment failures when RATE_LIMITED
        assert integration._consecutive_failures == 0

    def test_record_failure_increments_for_non_rate_limited(self):
        from assistant.integration.IntegrationMonitor import IntegrationMonitor

        monitor = IntegrationMonitor()
        mock_cm = Mock()
        mock_cm.get.return_value = "value"
        monitor._config_manager = mock_cm

        integration = Mock(spec=Integration)
        integration.state = IntegrationState.OK
        integration._consecutive_failures = 0

        monitor._record_failure(integration, "some error")

        assert integration._consecutive_failures == 1


class TestArlo2FACodePattern:
    def test_extracts_from_arlo_email_body(self):
        body = "Your Arlo verification code is 123456. It expires in 10 minutes."
        assert ARLO_2FA_PATTERN.search(body).group(1) == "123456"

    def test_no_match_for_non_6_digit(self):
        assert ARLO_2FA_PATTERN.search("code: 12345") is None
        assert ARLO_2FA_PATTERN.search("code: 1234567") is None

    def test_extracts_first_code(self):
        body = "Code: 111111. Ref: 222222."
        assert ARLO_2FA_PATTERN.search(body).group(1) == "111111"


class TestEmailAutoDetection:
    def _make_mock_message(self, text="", html="", from_="noreply@arlo.com", subject="Verification"):
        msg = Mock()
        msg.text = text
        msg.html = html
        msg.from_ = from_
        msg.subject = subject
        return msg

    def test_extract_code_from_plain_text(self):
        handler = _ArloTfaHandler()
        gmail = Mock()
        gmail.fetch_recent_raw.return_value = [
            self._make_mock_message(text="Your Arlo verification code is 847291.")
        ]
        code = handler._try_extract_code_from_email(gmail)
        assert code == "847291"

    def test_extract_code_from_html_fallback(self):
        handler = _ArloTfaHandler()
        gmail = Mock()
        gmail.fetch_recent_raw.return_value = [
            self._make_mock_message(text="", html="<p>Code: 654321</p>")
        ]
        gmail.strip_html_plain.return_value = "Code: 654321"
        code = handler._try_extract_code_from_email(gmail)
        assert code == "654321"

    def test_html_fallback_uses_link_free_extraction(self):
        # Regression: the real link-preserving strip_html would surface URL
        # digits (e.g. .../track/482910) that could be matched ahead of the
        # actual code. Code extraction must use the plain-text variant.
        from assistant.integration.integrations.email.EmailProvider import EmailProvider
        handler = _ArloTfaHandler()
        gmail = Mock()
        html = '<p><a href="https://arlo.com/track/482910">View</a> Code: 654321</p>'
        gmail.fetch_recent_raw.return_value = [
            self._make_mock_message(text="", html=html)
        ]
        gmail.strip_html_plain.side_effect = EmailProvider.strip_html_plain
        code = handler._try_extract_code_from_email(gmail)
        assert code == "654321"

    def test_returns_none_when_no_code_in_body(self):
        handler = _ArloTfaHandler()
        gmail = Mock()
        gmail.fetch_recent_raw.return_value = [
            self._make_mock_message(text="Welcome to Arlo! No code here.")
        ]
        assert handler._try_extract_code_from_email(gmail) is None

            # Should return without error

    def test_email_polling_calls_submit_code(self):
        handler = _ArloTfaHandler()
        handler._stop_email_polling = False
        gmail = Mock()
        gmail.fetch_recent_raw.return_value = [
            self._make_mock_message(text="Code: 999888")
        ]
        with patch.object(_ArloTfaHandler, '_email_providers', return_value=[gmail]):
            handler._poll_email_for_code()
        assert handler._code == "999888"
        assert handler._event.is_set()

    def test_polls_every_configured_provider(self):
        """The Arlo account's address may be on iCloud rather than Gmail: a code found by
        the second provider is submitted, the same attempt, without waiting a poll interval."""
        handler = _ArloTfaHandler()
        handler._stop_email_polling = False
        gmail = Mock(name="Gmail")
        gmail.fetch_recent_raw.return_value = []
        icloud = Mock(name="iCloud Email")
        icloud.name = "iCloud Email"
        icloud.fetch_recent_raw.return_value = [self._make_mock_message(text="Code: 123456", from_="noreply@arlo.com")]
        with patch.object(_ArloTfaHandler, '_email_providers', return_value=[gmail, icloud]), \
             patch('time.sleep') as mock_sleep:
            handler._poll_email_for_code()
        assert handler._code == "123456"
        gmail.fetch_recent_raw.assert_called_once()
        icloud.fetch_recent_raw.assert_called_once()
        mock_sleep.assert_not_called()

    def test_email_providers_follow_the_config(self):
        """Gmail comes from the `email` section, iCloud Email from `icloud_email`; an
        unconfigured account is not built."""
        from assistant.integration.integrations.email.GmailIntegration import GmailIntegration
        from assistant.integration.integrations.email.ICloudEmailIntegration import ICloudEmailIntegration
        config = Mock()
        config.get.side_effect = lambda key, default="": {"icloud_email.address": "me@icloud.com"}.get(key, default)
        with patch('common.ConfigManager.ConfigManager.get_instance', return_value=config), \
             patch.object(GmailIntegration, 'get_instance') as gmail_instance, \
             patch.object(ICloudEmailIntegration, 'get_instance', return_value=Mock(name="icloud")) as icloud_instance:
            providers = _ArloTfaHandler._email_providers()
        assert providers == [icloud_instance.return_value]
        gmail_instance.assert_not_called()

class TestAutoDetectionTelegramNotification:
    """Email-based auto-detection (added in d6574333) must surface a Telegram
    alert so the user knows a tab was authenticated unattended. Manual TUI
    code entry must NOT fire the alert."""

    _NOTIFY_PATH = 'assistant.Notifier.NullNotifier.send_message'

    @staticmethod
    def _email(text):
        msg = Mock()
        msg.text = text
        msg.html = ""
        msg.from_ = "noreply@arlo.com"
        msg.subject = "Verification"
        return msg

    def test_email_auto_detection_notifies_telegram(self):
        handler = _ArloTfaHandler()
        handler._stop_email_polling = False
        gmail = Mock()
        gmail.fetch_recent_raw.return_value = [self._email("Your code is 555444")]

        with patch.object(_ArloTfaHandler, '_email_providers', return_value=[gmail]), \
             patch(self._NOTIFY_PATH) as mock_notify:
            handler._poll_email_for_code()

        mock_notify.assert_called_once()
        message = mock_notify.call_args.args[0]
        assert "Arlo" in message
        assert "auto-detected" in message

    def test_no_code_in_email_notes_the_timeout_once(self):
        # The user (and the LLM) were told a code is being fetched; when the polling
        # window closes empty, one note says the TUI tab is the way forward.
        handler = _ArloTfaHandler()
        handler._stop_email_polling = False
        gmail = Mock()
        gmail.fetch_recent_raw.return_value = [self._email("Welcome to Arlo!")]

        with patch.object(_ArloTfaHandler, '_email_providers', return_value=[gmail]), \
             patch('time.sleep'), \
             patch(self._NOTIFY_PATH) as mock_notify:
            handler._poll_email_for_code()

        mock_notify.assert_called_once()
        assert "not found in email" in mock_notify.call_args.args[0]
        assert "Arlo tab" in mock_notify.call_args.args[0]

    def test_code_entered_manually_before_the_timeout_is_not_a_timeout(self):
        handler = _ArloTfaHandler()
        handler._stop_email_polling = False
        gmail = Mock()
        gmail.fetch_recent_raw.return_value = [self._email("Welcome to Arlo!")]
        handler.submit_code("111222")

        with patch.object(_ArloTfaHandler, '_email_providers', return_value=[gmail]), \
             patch('time.sleep'), \
             patch(self._NOTIFY_PATH) as mock_notify:
            handler._poll_email_for_code()

        mock_notify.assert_not_called()

def _make_jpeg(width=1920, height=1024, comment=None) -> bytes:
    img = Image.new("RGB", (width, height), color=(128, 128, 128))
    buf = BytesIO()
    save_kwargs = {"format": "JPEG"}
    if comment is not None:
        save_kwargs["comment"] = comment
    img.save(buf, **save_kwargs)
    return buf.getvalue()


class TestSnapshotFreshness:
    def test_native_resolution_no_comment_is_fresh(self):
        is_stale, reason = looks_like_cached_thumbnail(_make_jpeg(1920, 1024))
        assert is_stale is False
        assert reason == ""

    def test_low_resolution_flagged(self):
        is_stale, reason = looks_like_cached_thumbnail(_make_jpeg(640, 357))
        assert is_stale is True
        assert "640x357" in reason

    def test_lavc_comment_flagged_even_at_native_resolution(self):
        jpeg = _make_jpeg(1920, 1024, comment=b"Lavc58.91.100\x00")
        is_stale, reason = looks_like_cached_thumbnail(jpeg)
        assert is_stale is True
        assert "Lavc58.91.100" in reason

    def test_low_res_and_lavc_combines_reasons(self):
        jpeg = _make_jpeg(640, 357, comment=b"Lavc61.19.101\x00")
        is_stale, reason = looks_like_cached_thumbnail(jpeg)
        assert is_stale is True
        assert "640x357" in reason
        assert "Lavc61.19.101" in reason

    def test_empty_bytes_not_stale(self):
        is_stale, reason = looks_like_cached_thumbnail(b"")
        assert is_stale is False
        assert reason == ""

    def test_unparseable_bytes_not_stale(self):
        is_stale, reason = looks_like_cached_thumbnail(b"not a jpeg at all")
        assert is_stale is False
        assert "unparseable" in reason

    def test_just_below_threshold_flagged(self):
        is_stale, _ = looks_like_cached_thumbnail(_make_jpeg(1279, 720))
        assert is_stale is True

    def test_at_threshold_is_fresh(self):
        is_stale, _ = looks_like_cached_thumbnail(_make_jpeg(1280, 720))
        assert is_stale is False


class TestFetchSingleSnapshot:
    @staticmethod
    def _mock_cam(name="Garden", snapshots=(), previous=None):
        cam = MagicMock()
        cam.name = name
        cam.last_image_from_cache = previous
        cam.get_snapshot.side_effect = list(snapshots)
        return cam

    def test_fresh_snapshot_passes_through(self):
        fresh = _make_jpeg(1920, 1024)
        cam = self._mock_cam(snapshots=[fresh])
        with patch("assistant.integration.integrations.cameras.ArloCameraIntegration.time.sleep") as mock_sleep:
            blocks = ArloCameraIntegration._fetch_single_snapshot(cam)
        assert cam.get_snapshot.call_count == 1
        mock_sleep.assert_not_called()
        assert blocks[0]["type"] == "image"
        assert blocks[1]["text"] == "Camera: Garden"
        assert "⚠️" not in blocks[1]["text"]

    def test_stale_thumbnail_triggers_retry_and_uses_fresh(self):
        stale = _make_jpeg(640, 357)
        fresh = _make_jpeg(1920, 1024)
        cam = self._mock_cam(snapshots=[stale, fresh])
        with patch("assistant.integration.integrations.cameras.ArloCameraIntegration.time.sleep") as mock_sleep:
            blocks = ArloCameraIntegration._fetch_single_snapshot(cam)
        assert cam.get_snapshot.call_count == 2
        mock_sleep.assert_called_once_with(10)
        assert blocks[0]["type"] == "image"
        # Fresh image bytes were used, not the stale ones.
        import base64
        assert base64.b64decode(blocks[0]["source"]["data"]) == fresh
        assert blocks[1]["text"] == "Camera: Garden"

    def test_retry_still_stale_surfaces_warning(self):
        stale1 = _make_jpeg(640, 357)
        stale2 = _make_jpeg(720, 720)
        cam = self._mock_cam(snapshots=[stale1, stale2])
        with patch("assistant.integration.integrations.cameras.ArloCameraIntegration.time.sleep"):
            blocks = ArloCameraIntegration._fetch_single_snapshot(cam)
        assert cam.get_snapshot.call_count == 2
        text = blocks[1]["text"]
        assert "⚠️" in text
        assert "STALE" in text
        assert "old cached frame" in text
        assert "720x720" in text  # reason from the retry attempt is what we surface

    def test_retry_returns_none_keeps_first_stale_with_warning(self):
        stale = _make_jpeg(640, 357)
        cam = self._mock_cam(snapshots=[stale, None])
        with patch("assistant.integration.integrations.cameras.ArloCameraIntegration.time.sleep"):
            blocks = ArloCameraIntegration._fetch_single_snapshot(cam)
        assert cam.get_snapshot.call_count == 2
        import base64
        assert base64.b64decode(blocks[0]["source"]["data"]) == stale
        text = blocks[1]["text"]
        assert "⚠️" in text
        assert "640x357" in text

    def test_first_call_none_returns_timeout_error(self):
        cam = self._mock_cam(snapshots=[None])
        with patch("assistant.integration.integrations.cameras.ArloCameraIntegration.time.sleep") as mock_sleep:
            blocks = ArloCameraIntegration._fetch_single_snapshot(cam)
        assert cam.get_snapshot.call_count == 1
        mock_sleep.assert_not_called()
        assert len(blocks) == 1
        assert "timeout" in blocks[0]["text"]

    def test_exception_in_get_snapshot_returns_error_block(self):
        cam = MagicMock()
        cam.name = "Garden"
        cam.last_image_from_cache = None
        cam.get_snapshot.side_effect = RuntimeError("boom")
        blocks = ArloCameraIntegration._fetch_single_snapshot(cam)
        assert len(blocks) == 1
        assert "boom" in blocks[0]["text"]


class TestCheckHealth:
    """The check connects itself rather than through get_camera_names, which swallows the
    error into a plain False: the monitor records the exception's text, so a rejected
    password shows up as such in list_integrations."""

    @patch('assistant.integration.integrations.cameras.ArloCameraIntegration.pyaarlo.PyArlo')
    @patch('assistant.integration.integrations.cameras.ArloCameraIntegration.pyaarlo.constant')
    def test_a_rejected_password_reaches_the_monitor_as_the_error_text(self, mock_constant, mock_pyarlo, arlo):
        TestCredentialRejection._reject(mock_pyarlo)
        with pytest.raises(ArloAuthenticationError, match="401"):
            arlo.check_health()
        with pytest.raises(ArloAuthenticationError, match="cooldown.*Password not correct"):
            arlo.check_health()

    def test_between_real_checks_a_healthy_integration_is_not_reconnected(self, arlo):
        arlo.state = IntegrationState.OK
        with patch.object(arlo, "_ensure_connected") as connect, \
             patch.object(arlo, "get_camera_names", return_value=["Front"]):
            results = [arlo.check_health() for _ in range(5)]
        assert results == [True] * 5
        assert connect.call_count == 2, "the first check and every REAL_CHECK_EVERY-th one connect"

    def test_an_unhealthy_integration_is_checked_for_real_every_time(self, arlo):
        arlo.state = IntegrationState.ERROR
        with patch.object(arlo, "_ensure_connected") as connect, \
             patch.object(arlo, "get_camera_names", return_value=[]):
            assert [arlo.check_health() for _ in range(3)] == [False] * 3
        assert connect.call_count == 3
