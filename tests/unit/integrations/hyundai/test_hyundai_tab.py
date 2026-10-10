"""The Hyundai tab: the one-time code sent by Ctrl-A / Ctrl-S and typed in after Ctrl-E, the
refresh that turns a code demand into the code-needed state, and the cars tile that follows
the sign-in. The integration is a Mock; the worker thread runs inline."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from integrations.hyundai.HyundaiIntegration import HyundaiLoginError
from integrations.hyundai.HyundaiTab import HyundaiTab
from interfaces.console.tabs.AuthTab import PENDING


def _car(name="IONIQ 5", **state):
    fields = {"name": name, "model": "Ioniq 5", "VIN": "KMHK381GFPU000001",
              "ev_battery_percentage": None, "fuel_level": None, "last_updated_at": None}
    fields.update(state)
    return SimpleNamespace(**fields)


def make_tab(status="authenticated", cars=(_car(),), token=object()):
    api = Mock()
    api.get_auth_status.return_value = {"status": status, "message": ""}
    api._manager_instance = SimpleNamespace(token=token, vehicles={c.VIN: c for c in cars})
    tab = HyundaiTab()
    tab._api = api
    tab._run = lambda fn: fn()
    tab._check_health_now = Mock()
    tab._check_auth_status()
    return tab, api


def type_code(tab, code):
    tab.handle_input('\x05')
    for digit in code:
        tab.handle_input(digit)


@pytest.mark.unit
class TestCars:
    def test_the_tile_counts_the_account_cars_once_a_token_is_held(self):
        tab, _ = make_tab(cars=(_car("IONIQ 5"), _car("Sportage", VIN="KNDJ23AU5P7000002")))
        assert tab._tiles()[0][1] == "2" and len(tab._table().rows) == 2

    def test_without_a_token_no_car_is_listed(self):
        tab, _ = make_tab(status="not_initialized", token=None)
        assert tab._tiles()[0][1] == PENDING and tab._table().rows == []

    def test_the_charge_cell_joins_battery_and_fuel_and_drops_a_whole_fuel_levels_decimals(self):
        both = HyundaiTab._charge(_car(ev_battery_percentage=64, fuel_level=40.0))
        assert "64% battery" in both and "40% fuel" in both and "40.0" not in both
        assert HyundaiTab._charge(_car()) == PENDING


@pytest.mark.unit
class TestTheCode:
    def test_ctrl_a_sends_the_code_by_mail_and_ctrl_s_by_sms(self):
        tab, api = make_tab(status="needs_auth")
        api.send_otp.return_value = "Code sent by sms to +46701740605"
        tab.handle_input('\x01')
        tab.handle_input('\x13')
        assert [c.args[0] for c in api.send_otp.call_args_list] == ["email", "sms"]
        assert "+46701740605" in tab._status_message

    def test_a_send_the_backend_refuses_is_the_outcome_not_a_crash(self):
        tab, api = make_tab(status="needs_auth")
        api.send_otp.side_effect = HyundaiLoginError("sending the code by sms failed: no phone on the account")
        tab.handle_input('\x13')
        assert tab._status_message.startswith("Error:") and "no phone" in tab._status_message
        assert tab.status == "needs_auth"

    def test_six_digits_submit_the_code_and_a_verified_sign_in_re_checks_the_health(self):
        tab, api = make_tab(status="needs_auth")
        api.verify_otp.return_value = "Signed in — the one-time code was verified."
        api.get_auth_status.return_value = {"status": "authenticated", "message": ""}

        type_code(tab, "123456")

        api.verify_otp.assert_called_once_with("123456")
        assert not tab._input_mode and tab.status == "authenticated"
        tab._check_health_now.assert_called_once_with()
        assert tab._entry_hint() is None

    def test_a_refused_code_leaves_the_tab_waiting_for_another_without_a_health_check(self):
        tab, api = make_tab(status="needs_auth")
        api.verify_otp.side_effect = HyundaiLoginError("the code was refused: wrong code")

        type_code(tab, "000000")

        assert "wrong code" in tab._status_message and tab.status == "needs_auth"
        tab._check_health_now.assert_not_called()
        assert "Ctrl-A" in tab._entry_hint()[0]


@pytest.mark.unit
class TestRefresh:
    def test_a_sign_in_that_demands_a_code_turns_the_tab_to_code_needed(self):
        tab, api = make_tab(status="not_initialized", token=None)
        api.refresh_now.side_effect = HyundaiLoginError("Hyundai is not signed in: a one-time code is needed")
        api.get_auth_status.return_value = {"status": "needs_auth", "message": "One-time code needed"}

        tab.handle_input('\x12')

        assert tab.status == "needs_auth" and "one-time code" in tab._status_message
        tab._check_health_now.assert_not_called()
        assert "Ctrl-A" in tab._entry_hint()[0]

    def test_a_sign_in_that_lands_lists_the_cars_and_re_checks_the_health(self):
        tab, api = make_tab(status="not_initialized", token=None)
        assert tab._tiles()[0][1] == PENDING

        def sign_in():
            api._manager_instance.token = object()
            api.get_auth_status.return_value = {"status": "authenticated", "message": ""}
        api.refresh_now.side_effect = sign_in

        tab.handle_input('\x12')

        assert tab.status == "authenticated" and tab._tiles()[0][1] == "1"
        tab._check_health_now.assert_called_once_with()
        assert tab._entry_hint() is None
