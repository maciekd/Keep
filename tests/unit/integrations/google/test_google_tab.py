"""The Google tab: the accounts table, the selection, the link shown on Ctrl-L, the pasted
address taken on Ctrl-E and handed to the integration, and the sign-out."""
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest

from assistant.integration.OAuthPaste import SignIn
from integrations.google.GoogleAuth import REDIRECT_URI
from integrations.google.GoogleTab import GoogleTab

URL = "https://accounts.google.com/o/oauth2/v2/auth?client_id=cid.apps.googleusercontent.com&redirect_uri=http%3A%2F%2Flocalhost%3A18722%2F&response_type=code&scope=https%3A%2F%2Fmail.google.com%2F&access_type=offline&prompt=consent&state=st-1&login_hint=marcus%40example.com"
PASTED = f"{REDIRECT_URI}?state=st-1&code=4%2F0Acode"


def _account(name, address, mail=True, calendar=True, signed_in=True, expires_in=None, again=False):
    account = Mock()
    account.name, account.address = name, address
    account.mail = Mock() if mail else None
    account.calendar = Mock() if calendar else None
    account.signed_in.return_value = signed_in
    account.auth.needs_sign_in_again.return_value = again
    account.auth.expires_at.return_value = datetime.now(timezone.utc) + timedelta(minutes=expires_in) if expires_in else None
    return account


def make_tab(*accounts, finish=""):
    api = Mock()
    api.accounts.return_value = list(accounts)
    api.get_auth_status.return_value = {"status": "needs_auth", "message": ""}
    api.login_begin.return_value = SignIn(URL, "st-1", REDIRECT_URI, "marcus@example.com")
    api.login_finish.return_value = finish
    tab = GoogleTab()
    tab._api = api
    tab._run = lambda fn: fn()
    tab._check_health_now = lambda: None
    tab._check_auth_status()
    return tab, api


def cells(tab):
    return [[cell[0] for cell in row] for _, row in tab._table().rows]


def type_text(tab, text):
    for ch in text:
        tab.handle_input(ch)


@pytest.mark.unit
class TestTable:
    def test_one_row_per_account_with_what_it_gives_and_its_sign_in_state(self):
        tab, _ = make_tab(_account("Personal", "marcus@example.com", expires_in=42.5),
                          _account("Family", "family@example.com", calendar=False, signed_in=False),
                          _account("Club", "", mail=False, expires_in=None, again=True))

        rows = cells(tab)

        assert [r[0].strip("▸ ") for r in rows] == ["Personal", "Family", "Club"]
        assert [r[2] for r in rows] == ["mail + calendar", "mail", "calendar"]
        assert [r[3] for r in rows] == ["signed in", "not signed in", "sign in again"]
        assert rows[0][4] == "token 42 min" and rows[1][4] == "" and rows[2][1] == "–"
        assert tab._tiles()[0][1] == "1/3"

    def test_up_and_down_move_the_selection_and_wrap(self):
        tab, _ = make_tab(_account("Personal", "a@example.com"), _account("Family", "b@example.com"))
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
    def test_ctrl_l_shows_the_whole_link_of_the_selected_account_wrapped_over_rows(self):
        tab, api = make_tab(_account("Personal", "a@example.com"), _account("Family", "family@example.com", signed_in=False))
        tab.handle_input("down")

        tab.handle_input("\x0c")

        api.login_begin.assert_called_once_with("Family")
        fields = tab._fields()
        assert fields[0].value == "Family"
        assert "".join(f.value for f in fields if f.label in ("Open", "")) == URL
        assert any("paste" in f.value for f in fields) and "Waiting" in tab._status_message

    def test_ctrl_e_takes_the_pasted_address_however_long_and_enter_hands_it_to_the_integration(self):
        tab, api = make_tab(_account("Personal", "a@example.com", signed_in=False))
        tab.handle_input("\x0c")

        tab.handle_input("\x05")
        type_text(tab, PASTED + "x")
        tab.handle_input("\x7f")
        assert tab._fields()[1].waiting and tab._fields()[1].value.endswith(PASTED[-20:])
        tab.handle_input("\r")

        api.login_finish.assert_called_once_with(PASTED, "Personal")
        assert "Personal signed in" in tab._status_message and tab._sign_in is None and not tab._input_mode

    def test_enter_on_an_empty_entry_submits_nothing_and_escape_cancels(self):
        tab, api = make_tab(_account("Personal", "a@example.com", signed_in=False))
        tab.handle_input("\x05")
        tab.handle_input("\r")
        api.login_finish.assert_not_called()
        type_text(tab, "4/0A")
        tab.handle_input("escape")
        assert not tab._input_mode and tab._input_text == ""

    def test_a_failed_sign_in_is_the_outcome_and_the_link_stays(self):
        tab, api = make_tab(_account("Personal", "a@example.com", signed_in=False),
                            finish="the pasted address belongs to another sign-in")
        tab.handle_input("\x0c")
        tab.handle_input("\x05")
        type_text(tab, "4/0Awrong")
        tab.handle_input("\r")
        assert "another sign-in" in tab._status_message and tab._sign_in is not None

    def test_ctrl_o_signs_the_selected_account_out(self):
        tab, api = make_tab(_account("Personal", "a@example.com"))
        tab.handle_input("\x0f")
        api.sign_out.assert_called_once_with("Personal")
        assert "signed out" in tab._status_message
