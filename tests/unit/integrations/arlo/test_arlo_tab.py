"""ArloTab: Arlo locks an account out after too many logins, so the tab must stop trying
while the integration reports a cooldown and count it down instead of an outcome."""
import time
from unittest.mock import Mock, patch

import pytest

from integrations.arlo.ArloCameraIntegration import ArloAuthenticationError
from integrations.arlo.ArloTab import ArloTab


def make_tab(api=None):
    tab = ArloTab()
    tab._api = api or Mock()
    tab._run = lambda fn: fn()
    tab._check_auth_status = lambda: None
    return tab


@pytest.mark.unit
class TestRateLimit:
    def test_the_countdown_replaces_the_outcome_and_clears_itself_on_expiry(self):
        tab = make_tab()
        tab._status_message = "Connecting..."
        with patch("integrations.arlo.ArloTab.time") as clock:   # pinned: int(remaining) is exact
            clock.time.return_value = 1_000.0
            tab._rate_limit_until = 1_125.0
            assert tab._outcome() == "Rate limited — next attempt allowed in 02:05"
            tab._rate_limit_until = 999.0
            assert tab._outcome() == ""
        assert tab._rate_limit_until == 0 and tab._status_message == ""

    def test_a_refresh_under_cooldown_arms_the_limit_and_touches_no_endpoint(self):
        api = Mock(auth_cooldown_remaining=300)
        started = []
        tab = make_tab(api)
        tab._run = started.append
        with patch("integrations.arlo.ArloTab.time") as clock:
            clock.time.return_value = 1_000.0
            assert tab.handle_input('\x12')                     # Ctrl-R
            assert started == []
            assert tab._rate_limit_until == 1_300.0
            api.refresh_connection.assert_not_called()
            assert tab._outcome().startswith("Rate limited")

    def test_an_auth_error_with_a_cooldown_becomes_the_countdown_not_an_error_line(self):
        api = Mock(auth_cooldown_remaining=0)
        api.refresh_connection.side_effect = ArloAuthenticationError("locked out")
        tab = make_tab(api)
        tab._refresh()
        assert tab._status_message == "Auth error: locked out"
        assert tab._rate_limit_until == 0
        api.auth_cooldown_remaining = 60
        with patch("integrations.arlo.ArloTab.time") as clock:
            clock.time.return_value = 1_000.0
            tab._refresh()
            assert tab._rate_limit_until == 1_060.0
            assert tab._outcome().startswith("Rate limited")

    def test_a_successful_refresh_clears_an_expired_limit(self):
        api = Mock(auth_cooldown_remaining=0)
        tab = make_tab(api)
        tab._rate_limit_until = time.time() - 5
        tab._refresh()
        api.refresh_connection.assert_called_once()
        assert tab._rate_limit_until == 0
        assert tab._outcome() == "Connection refreshed"
