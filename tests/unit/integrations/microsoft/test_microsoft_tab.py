"""The Microsoft tab: the accounts table, the selection, and the sign-in and sign-out of the
selected account through the integration."""
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest

from integrations.microsoft.MicrosoftAuth import DeviceFlow
from integrations.microsoft.MicrosoftTab import MicrosoftTab


def _account(name, address, mail=True, calendar=True, signed_in=True, expires_in=None):
    account = Mock()
    account.name, account.address = name, address
    account.mail = Mock() if mail else None
    account.calendar = Mock() if calendar else None
    account.signed_in.return_value = signed_in
    account.auth.expires_at.return_value = datetime.now(timezone.utc) + timedelta(minutes=expires_in) if expires_in else None
    return account


def make_tab(*accounts, finish=""):
    api = Mock()
    api.accounts.return_value = list(accounts)
    api.get_auth_status.return_value = {"status": "needs_auth", "message": ""}
    api.login_begin.return_value = DeviceFlow("ABCD-1234", "https://microsoft.com/devicelogin", time.time() + 900, {})
    api.login_finish.return_value = finish
    tab = MicrosoftTab()
    tab._api = api
    tab._run = lambda fn: fn()
    tab._check_health_now = lambda: None
    tab._check_auth_status()
    return tab, api


def cells(tab):
    return [[cell[0] for cell in row] for _, row in tab._table().rows]


@pytest.mark.unit
class TestTable:
    def test_one_row_per_account_with_what_it_gives_and_whether_it_is_signed_in(self):
        tab, _ = make_tab(_account("Work", "marcus@company.example", expires_in=42.5),
                          _account("Family", "marcus@example.com", calendar=False, signed_in=False),
                          _account("Club", "", mail=False, expires_in=None))

        rows = cells(tab)

        assert [r[0].strip("▸ ") for r in rows] == ["Work", "Family", "Club"]
        assert [r[2] for r in rows] == ["mail + calendar", "mail", "calendar"]
        assert [r[3] for r in rows] == ["signed in", "not signed in", "signed in"]
        assert rows[0][4] == "token 42 min" and rows[1][4] == "" and rows[2][1] == "–"
        assert tab._tiles()[0][1] == "2/3"

    def test_up_and_down_move_the_selection_and_wrap(self):
        tab, _ = make_tab(_account("Work", "a@example.com"), _account("Family", "b@example.com"))
        assert cells(tab)[0][0].startswith("▸")
        tab.handle_input("down")
        assert cells(tab)[1][0].startswith("▸") and not cells(tab)[0][0].startswith("▸")
        tab.handle_input("down")
        assert cells(tab)[0][0].startswith("▸")

    def test_without_accounts_the_table_is_empty_and_the_keys_do_nothing(self):
        tab, api = make_tab()
        assert tab._table().rows == [] and tab._tiles()[0][1] == "–"
        tab.handle_input("\x0c")
        api.login_begin.assert_not_called()


@pytest.mark.unit
class TestSignIn:
    def test_ctrl_l_signs_the_selected_account_in_and_says_so(self):
        tab, api = make_tab(_account("Work", "a@example.com"), _account("Family", "b@example.com", signed_in=False))
        tab.handle_input("down")

        shown = []
        api.login_finish.side_effect = lambda name, flow: shown.append((tab._fields()[1].value, tab._status_message)) or ""
        tab.handle_input("\x0c")

        api.login_begin.assert_called_once_with("Family")
        assert shown == [("ABCD-1234", "Waiting — go to https://microsoft.com/devicelogin and enter the code ABCD-1234")]
        assert tab._flow is None and "Family signed in" in tab._status_message
        assert tab._fields()[1].value != "ABCD-1234"

    def test_a_failed_sign_in_is_the_outcome(self):
        tab, api = make_tab(_account("Work", "a@example.com", signed_in=False), finish="the code expired before it was entered")
        tab.handle_input("\x0c")
        assert "expired" in tab._status_message and tab._flow is None

    def test_ctrl_o_signs_the_selected_account_out(self):
        tab, api = make_tab(_account("Work", "a@example.com"))
        tab.handle_input("\x0f")
        api.sign_out.assert_called_once_with("Work")
        assert "signed out" in tab._status_message

    def test_ctrl_l_while_a_code_is_pending_starts_no_second_flow(self):
        tab, api = make_tab(_account("Work", "a@example.com", signed_in=False))
        tab._flow = DeviceFlow("WXYZ-9876", "https://microsoft.com/devicelogin", time.time() + 900, {})
        tab._flow_account = "Work"

        tab.handle_input("\x0c")

        api.login_begin.assert_not_called()
        assert "Work" in tab._status_message and "wait" in tab._status_message.lower()
        assert tab._fields()[1].value == "WXYZ-9876"
