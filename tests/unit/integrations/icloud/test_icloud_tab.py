from unittest.mock import Mock

import pytest

from integrations.icloud.iCloudTab import iCloudTab


def make_tab(api):
    tab = iCloudTab()
    tab._api = api
    return tab


class TestRequestSmsAndEnterCode:
    def test_pending_request_arms_manual_entry_without_new_sms(self):
        # A second SMS request would invalidate the code already in flight
        # (the stale-code cascade) — Ctrl-E must arm manual entry instead.
        api = Mock()
        api.sms_request_pending.return_value = True
        tab = make_tab(api)

        tab._request_sms_and_enter_code()

        api.request_sms_2fa.assert_not_called()
        assert tab._input_mode is True
        assert tab._input_text == ""
        assert "already requested" in tab._status_message

    def test_no_pending_request_sends_sms_and_arms_entry(self):
        api = Mock()
        api.sms_request_pending.return_value = False
        api.request_sms_2fa.return_value = "•••05"
        tab = make_tab(api)

        tab._request_sms_and_enter_code()

        api.request_sms_2fa.assert_called_once()
        assert tab._input_mode is True
        assert "•••05" in tab._status_message

    def test_sms_request_failure_reports_error(self):
        api = Mock()
        api.sms_request_pending.return_value = False
        api.request_sms_2fa.side_effect = Exception("boom")
        tab = make_tab(api)

        tab._request_sms_and_enter_code()

        assert tab._input_mode is False
        assert "SMS request failed" in tab._status_message


class TestCheckAuthStatus:
    def test_partial_auth_still_fetches_devices(self):
        # partial_auth serves Find My fine — the tab must show the device list,
        # not "(authenticate to see devices)".
        api = Mock()
        api.get_auth_status.return_value = {"status": "partial_auth", "message": ""}
        api.get_icloud_devices.return_value = [{"id": "x", "name": "iPhone"}]
        api.get_devices_list.return_value = "iPhone"
        tab = make_tab(api)

        tab._check_auth_status()

        assert tab._icloud_devices == [{"id": "x", "name": "iPhone"}]
        assert tab._devices_list == "iPhone"

    def test_needs_2fa_does_not_fetch_devices(self):
        api = Mock()
        api.get_auth_status.return_value = {"status": "needs_2fa", "message": ""}
        tab = make_tab(api)

        tab._check_auth_status()

        api.get_icloud_devices.assert_not_called()

    def test_device_fetch_failure_leaves_empty_lists(self):
        api = Mock()
        api.get_auth_status.return_value = {"status": "authenticated", "message": ""}
        api.get_icloud_devices.side_effect = Exception("network")
        api.get_devices_list.side_effect = Exception("network")
        tab = make_tab(api)

        tab._check_auth_status()

        assert tab._icloud_devices == []
        assert tab._devices_list == ""


class TestCtrlE:
    @pytest.mark.parametrize("status, opens", [
        ("needs_2fa", True), ("partial_auth", True), ("authenticated", False), ("error", False),
        ("not_initialized", False),
    ])
    def test_ctrl_e_only_starts_the_code_entry_when_a_second_factor_is_wanted(self, status, opens):
        # Requesting an SMS on an already verified session would invalidate a code in flight.
        api = Mock()
        api.sms_request_pending.return_value = False
        api.request_sms_2fa.return_value = "•••05"
        tab = make_tab(api)
        tab._auth_status = {"status": status, "message": ""}
        assert tab.handle_input('\x05') is True
        assert tab._input_mode is opens
        assert api.request_sms_2fa.called is opens


class TestCtrlR:
    def test_ctrl_r_is_the_users_refresh(self):
        """It lifts the park a refused sign-in left and signs in for Find My — the automatic
        refresh does neither, so the tab must say which one it is."""
        api = Mock()
        api.get_auth_status.return_value = {"status": "authenticated", "message": ""}
        api.get_icloud_devices.return_value = []
        api.get_devices_list.return_value = ""
        tab = make_tab(api)
        assert tab.handle_input('\x12') is True
        api.refresh_connection.assert_called_once_with(by_user=True)
        assert "refreshed" in tab._status_message.lower()

    def test_a_parked_tab_hides_the_device_table_and_points_at_ctrl_r(self):
        tab = make_tab(Mock())
        tab._auth_status = {"status": "parked", "message": "Sign-ins parked after Apple refused one"}
        assert tab._table().rows == []
        hint, _ = tab._entry_hint()
        assert "Ctrl-R" in hint
        assert tab._status_label()[0] != "Not initialized"

    def test_idle_find_my_reads_as_connected_not_as_a_problem(self):
        """Find My's cookies go with every restart and the tab never signs in on its own, so
        the verdict is the tab's resting state: the session is connected, Ctrl-R fills the
        table — no yellow, nothing that reads as a failure."""
        from interfaces.console import Theme
        tab = make_tab(Mock())
        tab._auth_status = {"status": "needs_reauth", "message": "Signed in — Find My is idle"}
        label, color = tab._status_label()
        assert label == tab.STATUS_LABELS["authenticated"][0]
        assert color not in (Theme.COLOR_YELLOW, Theme.COLOR_RED)
        hint, hint_color = tab._entry_hint()
        assert "Ctrl-R" in hint
        assert hint_color != Theme.COLOR_YELLOW
        assert tab._table().rows == []
        assert "Ctrl-R" in tab._table().empty


class TestAutoTwoFactor:
    def test_an_auto_verified_code_cancels_a_half_typed_one_and_rechecks(self):
        api = Mock()
        api.auto_2fa_message = "2FA auto-verified from iMessage"
        api.get_auth_status.return_value = {"status": "authenticated", "message": ""}
        tab = make_tab(api)
        tab._run = lambda fn: fn()
        tab._auth_status = {"status": "needs_2fa", "message": ""}
        tab._input_mode, tab._input_text = True, "123"
        tab._before_draw()
        assert tab._status_message == "2FA auto-verified from iMessage"
        assert tab._input_mode is False and tab._input_text == ""
        api.get_auth_status.assert_called_once()
        assert tab.status == "authenticated"

    def test_no_api_and_no_message_leave_the_tab_alone(self):
        tab = iCloudTab()
        tab._connect = lambda: (_ for _ in ()).throw(RuntimeError("not configured"))
        tab._input_mode, tab._input_text = True, "12"
        tab._before_draw()
        assert tab._input_mode is True and tab._input_text == "12"
        api = Mock(auto_2fa_message=None)
        tab = make_tab(api)
        tab._input_mode, tab._input_text = True, "12"
        tab._before_draw()
        assert tab._input_mode is True and tab._input_text == "12"
        api.get_auth_status.assert_not_called()


class TestFindMyVerdictInTheSameTick:
    """The listing is the one request that learns Find My's verdict, and the status check runs
    before it: an empty table must not sit under "Connected" until the next tick."""

    def _tab(self, statuses, devices):
        api = Mock()
        api.get_auth_status.side_effect = statuses
        api.get_icloud_devices.return_value = devices
        api.get_devices_list.return_value = ""
        return api, make_tab(api)

    def test_an_empty_listing_re_reads_the_status_it_just_taught(self):
        api, tab = self._tab([{"status": "authenticated", "message": ""},
                              {"status": "needs_reauth", "message": "Find My is idle"}], [])
        tab._check_auth_status()
        assert tab.status == "needs_reauth"
        assert api.get_auth_status.call_count == 2
        assert "Ctrl-R" in tab._table().empty

    def test_a_filled_listing_keeps_the_status_it_was_read_with(self):
        api, tab = self._tab([{"status": "authenticated", "message": ""}], [{"id": "fmip-dad", "name": "Dad's iPhone"}])
        tab._check_auth_status()
        assert tab.status == "authenticated"
        assert api.get_auth_status.call_count == 1
