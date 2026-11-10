import threading
import time
from typing import List
from unittest.mock import patch, Mock

import pytest

from assistant.integration.Feed import Feed
from assistant.integration.Integration import HealthWarning, Integration, IntegrationState
from assistant.integration.IntegrationMonitor import IntegrationMonitor, DisabledIntegration


class FakeIntegration(Integration):
    def __init__(self, integration_name="FakeService", keys=None, healthy=True):
        super().__init__()
        self._name = integration_name
        self._keys = keys or ["fake"]
        self._healthy = healthy

    @property
    def name(self) -> str:
        return self._name

    @property
    def config_keys(self) -> List[str]:
        return self._keys

    def check_health(self) -> bool:
        return self._healthy


class Fake2FAIntegration(Integration):
    def __init__(self, integration_name="TwoFAService", keys=None, healthy=True, requires_2fa=True):
        super().__init__()
        self._name = integration_name
        self._keys = keys or ["twofa"]
        self._healthy = healthy
        self._requires_2fa = requires_2fa

    @property
    def name(self) -> str:
        return self._name

    @property
    def config_keys(self) -> List[str]:
        return self._keys

    def check_health(self) -> bool:
        return self._healthy

    def needs_2fa(self) -> bool:
        return self._requires_2fa


class FailingIntegration(Integration):
    def __init__(self, integration_name="FailService", keys=None):
        super().__init__()
        self._name = integration_name
        self._keys = keys or ["fail"]

    @property
    def name(self) -> str:
        return self._name

    @property
    def config_keys(self) -> List[str]:
        return self._keys

    def check_health(self) -> bool:
        raise ConnectionError("Connection refused")


@pytest.fixture(autouse=True)
def reset_singleton():
    IntegrationMonitor.reset_instance()
    yield
    IntegrationMonitor.reset_instance()


@pytest.fixture
def mock_config():
    monitor = IntegrationMonitor()
    mock_cm = Mock()
    mock_cm.get.return_value = "some_value"
    monitor._config_manager = mock_cm
    return monitor


class TestRegistration:
    def test_register_not_configured_when_key_missing(self, mock_config):
        mock_config._config_manager.get.side_effect = KeyError("missing")
        integration = FakeIntegration()
        mock_config.register(integration)
        assert integration.state == IntegrationState.NOT_CONFIGURED

    def test_register_not_configured_when_key_empty(self, mock_config):
        """An empty value (`bot_token:` left blank) is as missing as an absent key."""
        mock_config._config_manager.get.return_value = ""
        integration = FakeIntegration()
        mock_config.register(integration)
        assert integration.state == IntegrationState.NOT_CONFIGURED

    @pytest.mark.parametrize("empty", [[], {}])
    def test_register_not_configured_when_the_accounts_list_or_section_is_empty(self, mock_config, empty):
        """`imap: accounts: []` (what the migration leaves behind a never-filled section) and a
        bare `tesla: {}` hold nothing to run on: not configured, not an error of a health check."""
        mock_config._config_manager.get.return_value = empty
        integration = FakeIntegration(keys=["imap.accounts"])
        mock_config.register(integration)
        assert integration.state == IntegrationState.NOT_CONFIGURED

class TestStateTransitions:
    def test_healthy_check_stays_ok(self, mock_config):
        integration = FakeIntegration(healthy=True)
        mock_config.register(integration)
        mock_config._check_integration(integration)
        assert integration.state == IntegrationState.OK
        assert integration.consecutive_failures == 0

    def test_single_failure_transitions_to_warning(self, mock_config):
        integration = FailingIntegration()
        mock_config._config_manager.get.return_value = "value"
        mock_config.register(integration)
        mock_config._check_integration(integration)
        assert integration.state == IntegrationState.WARNING
        assert integration.consecutive_failures == 1

    def test_three_failures_transitions_to_error(self, mock_config):
        integration = FailingIntegration()
        mock_config._config_manager.get.return_value = "value"
        mock_config.register(integration)
        for _ in range(3):
            mock_config._check_integration(integration)
        assert integration.state == IntegrationState.ERROR
        assert integration.consecutive_failures == 3

    def test_recovery_from_warning_to_ok(self, mock_config):
        integration = FakeIntegration(healthy=False)
        mock_config.register(integration)
        mock_config._check_integration(integration)
        assert integration.state == IntegrationState.WARNING

        integration._healthy = True
        mock_config._check_integration(integration)
        assert integration.state == IntegrationState.OK
        assert integration.consecutive_failures == 0

    def test_recovery_from_error_to_ok(self, mock_config):
        integration = FailingIntegration()
        mock_config._config_manager.get.return_value = "value"
        mock_config.register(integration)
        for _ in range(3):
            mock_config._check_integration(integration)
        assert integration.state == IntegrationState.ERROR

        healthy = FakeIntegration(integration_name="FailService", healthy=True)
        healthy._state = integration._state
        healthy._consecutive_failures = integration._consecutive_failures
        mock_config._check_integration(healthy)
        assert healthy.state == IntegrationState.OK

    def test_check_records_last_check_time(self, mock_config):
        integration = FakeIntegration()
        mock_config.register(integration)
        before = time.time()
        mock_config._check_integration(integration)
        assert integration.last_check_time >= before

    def test_failure_records_last_error(self, mock_config):
        integration = FailingIntegration()
        mock_config._config_manager.get.return_value = "value"
        mock_config.register(integration)
        mock_config._check_integration(integration)
        assert "Connection refused" in integration.last_error

    def test_false_health_check_records_failure(self, mock_config):
        integration = FakeIntegration(healthy=False)
        mock_config.register(integration)
        mock_config._check_integration(integration)
        assert integration.state == IntegrationState.WARNING
        assert integration.last_error == "Health check returned False"

    def test_second_failure_stays_warning(self, mock_config):
        integration = FakeIntegration(healthy=False)
        mock_config.register(integration)
        mock_config._check_integration(integration)
        assert integration.state == IntegrationState.WARNING
        mock_config._check_integration(integration)
        assert integration.state == IntegrationState.WARNING
        assert integration.consecutive_failures == 2


class TestAWarningFromAWorkingIntegration:
    """`HealthWarning`: the integration works and has something for the owner — shown, never an error."""

    @pytest.fixture
    def integration(self, mock_config):
        integration = FakeIntegration()
        mock_config.register(integration)
        integration.check_health = Mock(side_effect=HealthWarning("the satellite could not update itself"))
        return integration

    def test_it_is_shown_as_a_warning_with_its_text(self, mock_config, integration):
        mock_config._check_integration(integration)
        assert integration.state == IntegrationState.WARNING
        assert integration.last_error == "the satellite could not update itself"

    def test_however_often_it_repeats_it_never_becomes_an_error(self, mock_config, integration):
        for _ in range(IntegrationMonitor.FAILURE_THRESHOLD + 3):
            mock_config._check_integration(integration)
        assert integration.state == IntegrationState.WARNING and integration.consecutive_failures == 0

    def test_it_does_not_hide_a_real_failure_that_follows(self, mock_config, integration):
        mock_config._check_integration(integration)
        integration.check_health = Mock(side_effect=ConnectionError("no route to host"))
        for _ in range(IntegrationMonitor.FAILURE_THRESHOLD):
            mock_config._check_integration(integration)
        assert integration.state == IntegrationState.ERROR and integration.last_error == "no route to host"

    def test_a_clean_check_clears_it(self, mock_config, integration):
        mock_config._check_integration(integration)
        integration.check_health = Mock(return_value=True)
        mock_config._check_integration(integration)
        assert integration.state == IntegrationState.OK and integration.last_error is None


class TestCheckNow:
    def test_check_now_returns_state(self, mock_config):
        integration = FakeIntegration()
        mock_config.register(integration)
        state = mock_config.check_now("FakeService")
        assert state == IntegrationState.OK

    def test_check_now_unknown_returns_none(self, mock_config):
        assert mock_config.check_now("Nonexistent") is None

    def test_check_now_not_configured_skips_check(self, mock_config):
        mock_config._config_manager.get.side_effect = KeyError("missing")
        integration = FakeIntegration()
        mock_config.register(integration)
        state = mock_config.check_now("FakeService")
        assert state == IntegrationState.NOT_CONFIGURED


class TestFeedStatus:
    def test_a_feed_integration_reports_its_feed_state(self, mock_config):
        from assistant.integration.Feed import Feed, FeedBatch
        from assistant.integration.FeedRunner import Feeds

        class FakeFeed(FakeIntegration, Feed):
            feed_name = "Fake feed"
            config_slug = "fake"

            def initial_cursor(self, horizon_ts):
                return None

            def fetch(self, cursor, backfill):
                return FeedBatch()

        mock_config.register(FakeFeed())
        mock_config.register(FakeIntegration("Plain", ["p"]))
        feed_status = {"state": "live", "progress": None, "remaining": 0, "last_error": None}
        with patch.object(Feeds, "get_instance") as feeds:
            feeds.return_value.status.side_effect = lambda name: feed_status if name == "Fake feed" else None
            statuses = {s["name"]: s for s in mock_config.get_all_statuses()}
        assert statuses["FakeService"]["feed"] == feed_status
        assert "feed" not in statuses["Plain"]


class TestNeeds2FA:
    def test_status_carries_the_prerequisite_of_a_built_and_of_a_disabled_integration(self, mock_config):
        class Follower(FakeIntegration):
            setup_after = "FakeService"
            setup_after_reason = "its data downloads through it"

        mock_config.register(Follower("Built"))
        placeholder = DisabledIntegration(Follower)
        mock_config._integrations[placeholder.name] = placeholder
        statuses = {s["name"]: s for s in mock_config.get_all_statuses()}
        for name in ("Built", placeholder.name):
            assert (statuses[name]["setup_after"], statuses[name]["setup_after_reason"]) \
                == ("FakeService", "its data downloads through it")

    def test_registration_sets_needs_2fa_when_true(self, mock_config):
        integration = Fake2FAIntegration(requires_2fa=True)
        mock_config.register(integration)
        assert integration.state == IntegrationState.NEEDS_2FA

    def test_check_now_returns_needs_2fa_without_health_check(self, mock_config):
        integration = Fake2FAIntegration(requires_2fa=True, healthy=False)
        mock_config.register(integration)
        state = mock_config.check_now("TwoFAService")
        assert state == IntegrationState.NEEDS_2FA
        assert integration.last_check_time is None

    def test_check_now_resumes_after_2fa_resolved(self, mock_config):
        integration = Fake2FAIntegration(requires_2fa=True)
        mock_config.register(integration)
        assert integration.state == IntegrationState.NEEDS_2FA

        integration._requires_2fa = False
        state = mock_config.check_now("TwoFAService")
        assert state == IntegrationState.OK


class TestRunLoop:
    """Tests for _run_loop startup grace period and periodic health check logic."""

    def _make_monitor(self):
        monitor = IntegrationMonitor()
        mock_cm = Mock()
        mock_cm.get.return_value = "val"
        monitor._config_manager = mock_cm
        return monitor

    @staticmethod
    def _drive(monitor, on_wait):
        """Run the loop with the stop event's wait answered by on_wait(seconds) — True stops it."""
        monitor._stop_event = Mock()
        monitor._stop_event.wait.side_effect = on_wait
        monitor._run_loop()

    def test_startup_grace_period_waits_before_first_check(self):
        monitor = self._make_monitor()
        monitor.register(FakeIntegration(healthy=True))

        call_order = []
        original_check = monitor._check_integration
        def tracking_check(i):
            call_order.append(("check", i.name))
            original_check(i)
        monitor._check_integration = tracking_check

        def wait(s):
            call_order.append(("wait", s))
            return s == IntegrationMonitor.CHECK_INTERVAL
        self._drive(monitor, wait)

        assert call_order[0] == ("wait", IntegrationMonitor.STARTUP_GRACE_PERIOD)
        assert ("check", "FakeService") in call_order

    def test_stop_during_grace_period_runs_no_check(self):
        monitor = self._make_monitor()
        monitor.register(FakeIntegration(healthy=True))
        monitor._check_integration = Mock()

        self._drive(monitor, lambda s: True)

        monitor._check_integration.assert_not_called()

    def test_stop_wakes_the_loop_at_once(self):
        monitor = self._make_monitor()
        monitor._running = True
        monitor._thread = threading.Thread(target=monitor._run_loop, daemon=True)
        monitor._thread.start()
        thread = monitor._thread

        started = time.monotonic()
        monitor.stop()

        assert not thread.is_alive()
        assert time.monotonic() - started < 1

    def test_initial_check_skips_not_configured(self):
        monitor = self._make_monitor()

        healthy = FakeIntegration("Healthy", ["h"], healthy=True)
        monitor.register(healthy)

        not_configured = FakeIntegration("Unconfigured", ["nc"])
        monitor._config_manager.get.side_effect = lambda k: (_ for _ in ()).throw(KeyError) if k == "nc" else "val"
        monitor.register(not_configured)
        assert not_configured.state == IntegrationState.NOT_CONFIGURED

        checked = []
        original_check = monitor._check_integration
        def tracking_check(i):
            checked.append(i.name)
            original_check(i)
        monitor._check_integration = tracking_check

        self._drive(monitor, lambda s: s == IntegrationMonitor.CHECK_INTERVAL)

        assert "Healthy" in checked
        assert "Unconfigured" not in checked

    def test_initial_check_skips_needs_2fa(self):
        monitor = self._make_monitor()
        integration = Fake2FAIntegration(requires_2fa=True)
        monitor.register(integration)
        assert integration.state == IntegrationState.NEEDS_2FA

        checked = []
        original_check = monitor._check_integration
        monitor._check_integration = lambda i: checked.append(i.name) or original_check(i)

        self._drive(monitor, lambda s: s == IntegrationMonitor.CHECK_INTERVAL)

        assert "TwoFAService" not in checked

    def test_periodic_check_resolves_2fa(self):
        monitor = self._make_monitor()
        integration = Fake2FAIntegration(requires_2fa=True, healthy=True)
        monitor.register(integration)
        assert integration.state == IntegrationState.NEEDS_2FA

        checked = []
        original_check = monitor._check_integration
        def tracking_check(i):
            checked.append(i.name)
            original_check(i)
        monitor._check_integration = tracking_check

        cycle = [0]
        def wait(s):
            if s != IntegrationMonitor.CHECK_INTERVAL:
                return False
            cycle[0] += 1
            if cycle[0] == 1:
                integration._requires_2fa = False
            return cycle[0] >= 2
        self._drive(monitor, wait)

        assert "TwoFAService" in checked
        assert integration.state == IntegrationState.OK

    def test_periodic_check_detects_new_2fa_requirement(self):
        monitor = self._make_monitor()
        integration = Fake2FAIntegration(requires_2fa=False, healthy=True)
        monitor.register(integration)

        checked = []
        original_check = monitor._check_integration
        monitor._check_integration = lambda i: checked.append(i.name) or original_check(i)

        cycle = [0]
        def wait(s):
            if s != IntegrationMonitor.CHECK_INTERVAL:
                return False
            cycle[0] += 1
            if cycle[0] == 1:
                integration._requires_2fa = True
            return cycle[0] >= 2
        self._drive(monitor, wait)

        assert integration.state == IntegrationState.NEEDS_2FA

    def test_periodic_check_rechecks_rate_limited(self):
        monitor = self._make_monitor()
        integration = FakeIntegration(healthy=True)
        monitor.register(integration)
        integration._state = IntegrationState.RATE_LIMITED

        checked = []
        original_check = monitor._check_integration
        def tracking_check(i):
            checked.append(i.name)
            original_check(i)
        monitor._check_integration = tracking_check

        cycle = [0]
        def wait(s):
            if s != IntegrationMonitor.CHECK_INTERVAL:
                return False
            cycle[0] += 1
            return cycle[0] >= 2
        self._drive(monitor, wait)

        periodic_checks = [c for c in checked if c == "FakeService"]
        assert len(periodic_checks) >= 1
        assert integration.state == IntegrationState.OK


# ---------------------------------------------------------------------------
# register_class / restart / describe_all (plans/SELF_SETUP_PLAN.md §3.3)
# ---------------------------------------------------------------------------

from assistant.integration.SingletonMixin import SingletonMixin
from assistant.integration.IntegrationMonitor import UnavailableIntegration


class RebuildableIntegration(FakeIntegration, SingletonMixin):
    """Counts constructions and resets so a restart can be seen from the outside."""
    built = 0
    resets = 0
    healthy = True

    def __init__(self):
        super().__init__("Rebuildable", keys=["rebuildable"], healthy=type(self).healthy)
        type(self).built += 1

    def _on_reset(self):
        type(self).resets += 1


class BrokenIntegration(Integration, SingletonMixin):
    """Raises in its constructor until `ready` is set, like Tesla without a `tesla` section."""
    ready = False

    def __init__(self):
        if not type(self).ready:
            raise RuntimeError("Missing 'broken' section in configuration.")
        super().__init__()

    @property
    def name(self) -> str:
        return "Broken Service"

    @property
    def config_keys(self) -> List[str]:
        return ["broken"]

    def check_health(self) -> bool:
        return True


@pytest.fixture(autouse=True)
def reset_fakes():
    for cls in (RebuildableIntegration, BrokenIntegration):
        cls.reset_instance()
    RebuildableIntegration.built = RebuildableIntegration.resets = 0
    RebuildableIntegration.healthy = True
    BrokenIntegration.ready = False
    yield
    for cls in (RebuildableIntegration, BrokenIntegration):
        cls.reset_instance()


class TestRegisterClass:
    def test_builds_registers_and_runs_the_start_hook(self, mock_config):
        started = []
        integration = mock_config.register_class(RebuildableIntegration, start=started.append)
        assert isinstance(integration, RebuildableIntegration)
        assert started == [integration]
        assert mock_config.find("rebuildable") is integration
        assert mock_config.get_all_statuses()[0]["restartable"] is True

    def test_a_raising_constructor_lands_as_not_configured_with_the_error(self, mock_config):
        started = []
        placeholder = mock_config.register_class(BrokenIntegration, start=started.append)
        assert isinstance(placeholder, UnavailableIntegration)
        assert placeholder.name == "Broken"
        assert placeholder.state == IntegrationState.NOT_CONFIGURED
        assert placeholder.last_error == "RuntimeError: Missing 'broken' section in configuration."
        assert started == []
        assert mock_config.get_all_statuses()[0]["restartable"] is True

    def test_register_by_instance_is_not_restartable(self, mock_config):
        mock_config.register(FakeIntegration())
        assert mock_config.get_all_statuses()[0]["restartable"] is False
        assert mock_config.restart("FakeService") == {"restartable": False,
                                                      "status": mock_config.get_all_statuses()[0]}

    def test_find_is_case_insensitive_and_none_for_unknown(self, mock_config):
        mock_config.register(FakeIntegration("iCloud Email"))
        assert mock_config.find("icloud email").name == "iCloud Email"
        assert mock_config.find("gmail") is None

    def test_folder_named_knows_the_folder_of_a_disabled_integration_too(self, mock_config):
        """improve_integration asks for the folder of the integration the user names; a disabled
        one is never built, but a fix to it must still land in its folder and not be opened
        as a new integration."""
        class FolderedIntegration(FakeIntegration, SingletonMixin):
            def __init__(self):
                super().__init__("Tesla", keys=["tesla"])

            @property
            def name(self) -> str:
                return "Tesla"   # constant, so the name is known before an instance exists
        FolderedIntegration.__module__ = "integrations.tesla.TeslaIntegration"

        mock_config.register_class(FolderedIntegration)
        assert mock_config.folder_named("tesla") == "tesla"

        disabled = IntegrationMonitor()
        disabled._config_manager = Mock()
        with patch.object(Integration, "enabled", classmethod(lambda cls, config: False)):   # the suite stubs it on
            placeholder = disabled.register_class(FolderedIntegration)
        assert isinstance(placeholder, DisabledIntegration)
        assert disabled.folder_named("TESLA") == "tesla"
        assert disabled.folder_named("ring") is None
        mock_config.register(FakeIntegration())
        assert mock_config.folder_named("FakeService") is None, "a class outside src/integrations has no folder"

    def test_an_integration_is_found_by_its_folder_when_its_name_differs(self, mock_config):
        """2026-10-26: improve_integration("music", ...) found no integration — the folder's
        is called "Sonos & Spotify" — and the fix ran as a new integration. The folder name is
        what a report calls the integration, so it finds it too; an exact name still wins."""
        class MusicIntegration(FakeIntegration):
            def __init__(self):
                super().__init__("Sonos & Spotify", keys=["music"])
        MusicIntegration.__module__ = "integrations.music.MusicIntegration"

        class SonosIntegration(FakeIntegration):
            def __init__(self):
                super().__init__("Music", keys=["sonos"])
        SonosIntegration.__module__ = "integrations.sonos.SonosIntegration"

        mock_config.register(MusicIntegration())
        assert mock_config.folder_named("music") == "music"
        assert mock_config.find("MUSIC").name == "Sonos & Spotify"
        mock_config.register(SonosIntegration())
        assert mock_config.folder_named("music") == "sonos", "the integration named so comes before the folder"

    def test_a_near_miss_names_the_integration_meant(self, mock_config):
        class MusicIntegration(FakeIntegration):
            def __init__(self):
                super().__init__("Sonos & Spotify", keys=["music"])
        MusicIntegration.__module__ = "integrations.music.MusicIntegration"

        class XIntegration(FakeIntegration):
            def __init__(self):
                super().__init__("X", keys=["x"])
        XIntegration.__module__ = "integrations.x.XIntegration"

        mock_config.register(MusicIntegration())
        mock_config.register(XIntegration())
        mock_config.register(FakeIntegration("Microsoft 365"))
        assert mock_config.resembling("Spotify") == [("Sonos & Spotify", "music")]
        assert mock_config.resembling("the music player") == [("Sonos & Spotify", "music")]
        assert mock_config.resembling("microsoft") == [("Microsoft 365", None)]
        assert mock_config.resembling("Ring") == [], "a new service resembles nothing"
        assert mock_config.resembling("Dropbox") == [], "a one-letter name is in every other word and is no near miss"
        assert mock_config.resembling("x") == [], "nor is a one-letter name asked for"


class TestRestart:
    def test_rebuilds_the_singleton_reruns_start_and_checks_health(self, mock_config):
        started = []
        old = mock_config.register_class(RebuildableIntegration, start=started.append)
        result = mock_config.restart("Rebuildable")
        new = mock_config.find("Rebuildable")
        assert result["restartable"] is True and result["status"]["state"] == "ok"
        assert new is not old and RebuildableIntegration.get_instance() is new
        assert RebuildableIntegration.built == 2 and RebuildableIntegration.resets == 1
        assert started == [old, new]
        assert len(mock_config.get_all_statuses()) == 1
        mock_config._config_manager.reload.assert_called_once()

    def test_a_failing_health_check_after_restart_is_reported_not_raised(self, mock_config):
        mock_config.register_class(RebuildableIntegration)
        RebuildableIntegration.healthy = False
        result = mock_config.restart("rebuildable")
        assert result["status"]["state"] == "warning"
        assert result["status"]["last_error"] == "Health check returned False"

    def test_missing_config_after_restart_stays_not_configured_without_a_check(self, mock_config):
        mock_config.register_class(RebuildableIntegration)
        mock_config._config_manager.get.side_effect = KeyError("rebuildable")
        result = mock_config.restart("Rebuildable")
        assert result["status"]["state"] == "not_configured"
        assert result["status"]["last_check_time"] is None

    def test_a_broken_integration_comes_alive_once_its_config_exists(self, mock_config):
        mock_config.register_class(BrokenIntegration)
        BrokenIntegration.ready = True
        result = mock_config.restart("Broken")
        assert result["status"] == {**result["status"], "name": "Broken Service", "state": "ok", "restartable": True}
        assert [s["name"] for s in mock_config.get_all_statuses()] == ["Broken Service"]
        assert mock_config.find("Broken") is None

    def test_a_reset_hook_that_raises_does_not_stop_the_restart(self, mock_config):
        mock_config.register_class(RebuildableIntegration)
        with patch.object(RebuildableIntegration, "_on_reset", side_effect=RuntimeError("stuck thread")):
            result = mock_config.restart("Rebuildable")
        assert result["status"]["state"] == "ok"
        assert RebuildableIntegration.built == 2


class RebuildableFeed(RebuildableIntegration, Feed):
    """A Feed integration: its health lives on the feed thread open_feed runs on."""

    @property
    def feed_name(self):
        return "Rebuildable feed"

    @property
    def config_slug(self):
        return "rebuildable"

    def open_feed(self):
        return True

    def initial_cursor(self, horizon_ts):
        return None

    def fetch(self, cursor, backfill):
        raise NotImplementedError


class TestRestartWaitsForTheFeed:
    """The start hook only spawns the feed thread; checked a millisecond later the restart
    reported a warning beside a live feed for the monitor's next 15 minutes."""

    @pytest.fixture(autouse=True)
    def reset(self):
        RebuildableFeed.reset_instance()
        yield
        RebuildableFeed.reset_instance()

    def test_the_health_check_waits_for_open_feed(self, mock_config):
        mock_config.register_class(RebuildableFeed)
        runner = Mock()
        order = []
        runner.wait_opened.side_effect = lambda timeout: order.append(("wait", timeout)) or True
        with patch.object(RebuildableFeed, "check_health", side_effect=lambda: order.append("check") or True), \
             patch("assistant.integration.IntegrationMonitor.Feeds.get_instance") as feeds:
            feeds.return_value.runner.return_value = runner
            result = mock_config.restart("Rebuildable")
        feeds.return_value.runner.assert_called_with("Rebuildable feed")
        assert order == [("wait", IntegrationMonitor.READY_TIMEOUT), "check"]
        assert result["status"]["state"] == "ok"

    def test_a_feed_still_opening_is_checked_anyway(self, mock_config):
        mock_config.register_class(RebuildableFeed)
        runner = Mock()
        runner.wait_opened.return_value = False
        with patch("assistant.integration.IntegrationMonitor.Feeds.get_instance") as feeds:
            feeds.return_value.runner.return_value = runner
            result = mock_config.restart("Rebuildable")
        assert result["status"]["state"] == "ok"

    def test_no_runner_and_a_plain_integration_do_not_wait(self, mock_config):
        mock_config.register_class(RebuildableFeed)
        with patch("assistant.integration.IntegrationMonitor.Feeds.get_instance") as feeds:
            feeds.return_value.runner.return_value = None
            assert mock_config.restart("Rebuildable")["status"]["state"] == "ok"
        RebuildableFeed.reset_instance()
        mock_config.register_class(RebuildableIntegration)
        with patch("assistant.integration.IntegrationMonitor.Feeds.get_instance") as feeds:
            mock_config.restart("Rebuildable")
        feeds.return_value.runner.assert_not_called()

    def test_an_integration_starting_on_its_own_thread_is_checked_once_ready(self, mock_config):
        """Web Access's Chrome comes up on a loop thread after the constructor returns."""
        RebuildableIntegration.reset_instance()
        mock_config.register_class(RebuildableIntegration)
        RebuildableIntegration.healthy = False

        def comes_up(self, timeout):
            assert timeout > 0
            self._healthy = True
            return True

        with patch.object(RebuildableIntegration, "wait_ready", autospec=True, side_effect=comes_up):
            result = mock_config.restart("Rebuildable")
        assert result["status"]["state"] == "ok"

    def test_an_integration_that_never_comes_up_is_checked_anyway(self, mock_config):
        RebuildableIntegration.reset_instance()
        mock_config.register_class(RebuildableIntegration)
        RebuildableIntegration.healthy = False
        with patch.object(RebuildableIntegration, "wait_ready", return_value=False):
            result = mock_config.restart("Rebuildable")
        assert result["status"]["state"] == "warning"


    def test_a_ready_hook_that_raises_still_gets_the_health_check(self, mock_config):
        """The wait is a diagnostic's helper: a bug in it must not leave the restart without a verdict."""
        RebuildableIntegration.reset_instance()
        mock_config.register_class(RebuildableIntegration)
        with patch.object(RebuildableIntegration, "wait_ready", side_effect=RuntimeError("no loop thread")), \
                patch.object(RebuildableIntegration, "check_health", return_value=True) as check:
            result = mock_config.restart("Rebuildable")
        check.assert_called_once()
        assert result["status"]["state"] == "ok"

    def test_the_ready_hook_gets_what_the_feed_wait_left_of_the_one_budget(self, mock_config):
        """A feed integration waits for open_feed first; the hook is given the rest of READY_TIMEOUT,
        so the two waits together never hold a restart past it."""
        mock_config.register_class(RebuildableFeed)
        runner = Mock()
        runner.wait_opened.return_value = True
        clock = iter([1000.0])            # the start; every later look is ten seconds on
        timeouts = []
        with patch.object(RebuildableFeed, "wait_ready", side_effect=lambda timeout: timeouts.append(timeout) or True), \
                patch("assistant.integration.IntegrationMonitor.Feeds.get_instance") as feeds, \
                patch("assistant.integration.IntegrationMonitor.time.time", side_effect=lambda: next(clock, 1010.0)):
            feeds.return_value.runner.return_value = runner
            mock_config.restart("Rebuildable")
        assert timeouts == [pytest.approx(IntegrationMonitor.READY_TIMEOUT - 10)]


class TestDescribeAll:
    def test_adds_per_key_set_flags_without_values(self, mock_config):
        mock_config.register(FakeIntegration("Mail", keys=["email"]))
        with patch("common.ConfigStore.describe_keys", side_effect=lambda keys: [{"key": f"{k}.x", "set": False} for k in keys]) as describe:
            described = mock_config.describe_all()
        describe.assert_called_once_with(["email"])
        assert described[0]["keys"] == [{"key": "email.x", "set": False}]
        assert described[0]["config_keys"] == ["email"]
        assert described[0]["restartable"] is False


@pytest.mark.unit
class TestDisabledIntegrations:
    class Switchable(Integration):
        built = 0

        def __init__(self):
            super().__init__()
            type(self).built += 1

        @property
        def name(self) -> str:
            return "Switchable Thing"

        @property
        def config_keys(self) -> List[str]:
            return []

        def check_health(self) -> bool:
            return True

    @pytest.fixture(autouse=True)
    def fresh(self):
        IntegrationMonitor.reset_instance()
        self.Switchable.reset_instance()
        self.Switchable.built = 0
        yield
        IntegrationMonitor.reset_instance()
        self.Switchable.reset_instance()

    def test_config_name_is_readable_without_an_instance(self):
        assert self.Switchable.config_name() == "Switchable Thing"

        class NoConstantName(Integration):
            @property
            def name(self):
                return self._n
            @property
            def config_keys(self):
                return []
            def check_health(self):
                return True
        assert NoConstantName.config_name() == "NoConstantName"

    def test_flag_reads_the_list_with_the_class_default(self):
        from common.ConfigManager import ConfigManager
        cfg = Mock(spec=ConfigManager)
        cfg.integration_enabled.side_effect = lambda name, default=False: default
        with patch.object(Integration, "enabled", Integration.__dict__["enabled"].original):   # undo the suite-wide stub
            assert self.Switchable.enabled(cfg) is False
            self.Switchable.enabled_by_default = True
            try:
                assert self.Switchable.enabled(cfg) is True
            finally:
                self.Switchable.enabled_by_default = False
        assert [c.args for c in cfg.integration_enabled.call_args_list] == [("Switchable Thing", False), ("Switchable Thing", True)]

    def test_disabled_class_is_listed_but_never_built(self):
        monitor = IntegrationMonitor.get_instance()
        started = []
        with patch.object(Integration, "enabled", classmethod(lambda cls, config: False)):
            placeholder = monitor.register_class(self.Switchable, start=started.append)
        assert placeholder.state == IntegrationState.DISABLED and placeholder.name == "Switchable Thing"
        assert self.Switchable.built == 0 and started == []
        assert monitor.find("switchable thing") is placeholder
        status = monitor._status(placeholder)
        assert status["state"] == "disabled" and status["restartable"] is True
        # the recurring health cycle and check_now leave it alone: a check would fail it into
        # ERROR and list_integrations would lose the enable line it keys on DISABLED
        with patch.object(monitor, "_check_integration") as check:
            monitor._check_cycle()
            assert monitor.check_now("Switchable Thing") is IntegrationState.DISABLED
        check.assert_not_called()
        assert placeholder.state is IntegrationState.DISABLED and placeholder.consecutive_failures == 0

    def test_a_sensitive_integration_is_flagged_while_still_disabled(self):
        """The warning before enabling a private-data integration (list_integrations, the beta
        telemetry's transcript latch) is wanted exactly when it is still a placeholder — the
        flag must come from the class behind the placeholder, which is not itself sensitive."""
        class Mailbox(self.Switchable):
            sensitive = True

            @property
            def name(self) -> str:
                return "Mailbox"

        monitor = IntegrationMonitor.get_instance()
        try:
            with patch.object(Integration, "enabled", classmethod(lambda cls, config: False)):
                monitor.register_class(Mailbox)
            monitor.register(FakeIntegration("Lamps"))
            flags = {status["name"]: status["sensitive"] for status in monitor.get_all_statuses()}
        finally:
            Mailbox.reset_instance()
        assert flags == {"Mailbox": True, "Lamps": False}

    def test_the_status_asks_a_built_integration_for_its_login_hint_of_the_moment(self):
        """A built integration may know how the sign-in is done right now (iCloud: whether a
        code reader is live); a placeholder was never built, so the class's hint stands."""
        class Watchful(FakeIntegration):
            login_hint = "enter the code in the tab"

            def current_login_hint(self):
                return "the server submits the code itself"

        monitor = IntegrationMonitor.get_instance()
        built = Watchful("Watchful")
        monitor.register(built)
        assert monitor._status(built)["login_hint"] == "the server submits the code itself"
        assert monitor._status(DisabledIntegration(Watchful))["login_hint"] == "enter the code in the tab"

    def test_the_status_carries_what_an_integration_says_of_its_accounts(self):
        """An account-holding integration reports its accounts one by one; the rest have none."""
        class Mailboxes(FakeIntegration):
            def accounts_status(self):
                return [{"name": "Personal", "state": "ok", "last_error": None},
                        {"name": "Work", "state": "error", "last_error": "auth failed"}]

        monitor = IntegrationMonitor.get_instance()
        monitor.register(Mailboxes("Mailboxes"))
        monitor.register(FakeIntegration("Lamps"))
        accounts = {status["name"]: status["accounts"] for status in monitor.get_all_statuses()}
        assert [a["name"] for a in accounts["Mailboxes"]] == ["Personal", "Work"]
        assert accounts["Mailboxes"][1]["last_error"] == "auth failed"
        assert accounts["Lamps"] == []

    def test_enabling_then_restart_builds_it(self):
        monitor = IntegrationMonitor.get_instance()
        started = []
        with patch.object(Integration, "enabled", classmethod(lambda cls, config: False)):
            monitor.register_class(self.Switchable, start=started.append)
        with patch.object(Integration, "enabled", classmethod(lambda cls, config: True)):
            result = monitor.restart("Switchable Thing")
        assert result["restartable"] is True and result["status"]["state"] == "ok"
        assert self.Switchable.built == 1 and len(started) == 1
        # and disabling again tears it down to a placeholder
        with patch.object(Integration, "enabled", classmethod(lambda cls, config: False)):
            result = monitor.restart("Switchable Thing")
        assert result["status"]["state"] == "disabled"
        from assistant.integration.SingletonMixin import SingletonMixin
        assert self.Switchable not in SingletonMixin._instances

    def test_registering_an_instance_ignores_the_switch(self):
        monitor = IntegrationMonitor.get_instance()
        with patch.object(Integration, "enabled", classmethod(lambda cls, config: False)):
            monitor.register(FakeIntegration("ByInstance"))
        assert monitor.find("ByInstance").state != IntegrationState.DISABLED

    def test_a_disabled_feed_placeholder_still_reads_as_a_feed(self):
        # "set up the telegram feed" must find Telegram Feed by the word "feed" while it
        # is still disabled — known from the class, since no runner exists to ask.
        from assistant.integration.Feed import Feed, FeedBatch

        class SwitchableFeed(self.Switchable, Feed):
            feed_name = "Switchable feed"
            config_slug = "switchable"

            @property
            def name(self) -> str:
                return "Switchable Feed"

            def initial_cursor(self, horizon_ts):
                return None

            def fetch(self, cursor, backfill):
                return FeedBatch()

        monitor = IntegrationMonitor.get_instance()
        with patch.object(Integration, "enabled", classmethod(lambda cls, config: False)):
            feed = monitor.register_class(SwitchableFeed)
            plain = monitor.register_class(self.Switchable)
        assert feed.is_feed is True and plain.is_feed is False
        statuses = {s["name"]: s for s in monitor.get_all_statuses()}
        assert statuses["Switchable Feed"]["state"] == "disabled" and statuses["Switchable Feed"]["feed"] is None
        assert "feed" not in statuses["Switchable Thing"]


@pytest.mark.unit
class TestMissingConfigKeys:
    """Integration.missing_config_keys — the same verdict the monitor's register() reaches,
    as a list for a caller that wants to say which key is unset."""

    def _integration(self, keys):
        class Probe(Integration):
            name = "Probe"
            config_keys = keys
            def check_health(self):
                return True
        return Probe()

    def test_lists_absent_and_empty_keys_only(self):
        values = {"api_keys.set": "k", "api_keys.empty": ""}

        def get(key):
            if key in values:
                return values[key]
            raise KeyError(key)

        with patch("common.ConfigManager.config") as config:
            config.get.side_effect = get
            probe = self._integration(["api_keys.set", "api_keys.empty", "api_keys.absent"])
            assert probe.missing_config_keys() == ["api_keys.empty", "api_keys.absent"]

    def test_an_empty_accounts_list_and_an_empty_section_are_missing_too(self):
        values = {"imap.accounts": [], "tesla": {}, "caldav.accounts": [{"name": "Family"}]}
        with patch("common.ConfigManager.config") as config:
            config.get.side_effect = values.__getitem__
            probe = self._integration(["imap.accounts", "tesla", "caldav.accounts"])
            assert probe.missing_config_keys() == ["imap.accounts", "tesla"]


def test_description_and_details_are_the_first_two_paragraphs_of_the_class_docstring_and_are_not_inherited():
    class Described(FakeIntegration):
        """The electricity
        meter.

        Reads the meter every hour, e.g. at night. Keep can
        say what the house used.

        Nothing of this paragraph is shown anywhere."""

    class Terse(FakeIntegration):
        """The water meter."""

    class Undocumented(Described):
        pass

    assert Described.description() == "The electricity meter."
    assert Described.details() == "Reads the meter every hour, e.g. at night. Keep can say what the house used."
    assert (Terse.description(), Terse.details()) == ("The water meter.", "")
    assert (Undocumented.description(), Undocumented.details()) == ("", "")
