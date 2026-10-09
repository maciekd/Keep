"""WhatsAppTab: the console's link board. The phone number is typed on the tab, then the
pairing code the integration got is shown until the phone accepts it — login, wait_paired and
finish_login on a worker thread, which the tests run inline."""
from unittest.mock import Mock, patch

import pytest

from integrations.whatsapp.WhatsAppTab import WhatsAppTab, INPUT_PHONE, PHONE_MAX, PHONE_MIN

PHONE = "+46701740605"
CODE = "ABCD-EFGH"


def make_tab(api=None) -> WhatsAppTab:
    tab = WhatsAppTab()
    tab._api = api if api is not None else Mock()
    tab._run = lambda fn: fn()
    tab._auth_status = {"status": "needs_auth", "message": ""}
    return tab


def type_keys(tab: WhatsAppTab, keys: str | list):
    for key in keys:
        assert tab.handle_input(key) is True


def start_entry(tab: WhatsAppTab, digits: str):
    tab.handle_input('\x10')
    type_keys(tab, digits)


@pytest.mark.unit
class TestPhoneEntry:
    def test_ctrl_p_opens_the_entry_with_the_plus_typed_and_drops_a_stale_code(self):
        tab = make_tab()
        tab._code = CODE
        assert tab.handle_input('\x10') is True
        assert tab._input_mode == INPUT_PHONE
        assert tab._input_text == "+"
        assert tab._code is None

    def test_only_digits_extend_the_phone_and_backspace_keeps_the_plus(self):
        tab = make_tab()
        start_entry(tab, ["4", "6", "a", "-", " ", "7"])
        assert tab._input_text == "+467"
        type_keys(tab, ["backspace"] * 5)
        assert tab._input_text == "+"
        assert tab._input_mode == INPUT_PHONE

    def test_the_phone_stops_growing_at_its_maximum(self):
        tab = make_tab()
        start_entry(tab, "4" * (PHONE_MAX + 5))
        assert len(tab._input_text) == PHONE_MAX

    def test_enter_on_a_too_short_phone_asks_for_nothing(self):
        api = Mock()
        tab = make_tab(api)
        start_entry(tab, PHONE[1:PHONE_MIN - 1])
        type_keys(tab, ["enter"])
        api.login.assert_not_called()
        assert tab._input_mode == INPUT_PHONE

    def test_escape_leaves_the_entry(self):
        tab = make_tab()
        start_entry(tab, "4670")
        type_keys(tab, ["escape"])
        assert not tab._input_mode and tab._input_text == ""


@pytest.mark.unit
class TestLinking:
    def test_the_code_is_shown_while_the_phone_is_waited_on_and_the_link_is_finished(self):
        api = Mock()
        api.login.return_value = (CODE, None)
        shown = {}
        api.wait_paired.side_effect = lambda: shown.update(code=tab._code, message=tab._status_message)
        api.get_auth_status.return_value = {"status": "authenticated", "message": ""}
        tab = make_tab(api)
        start_entry(tab, PHONE[1:])
        with patch("assistant.integration.FeedRunner.Feeds.get_instance") as feeds:
            feeds.return_value.status.return_value = {"state": "live", "stored": 3, "items": "messages"}
            type_keys(tab, ["enter"])
        api.login.assert_called_once_with(PHONE)
        assert shown["code"] == CODE
        assert tab._phone_number == PHONE and not tab._input_mode
        api.finish_login.assert_called_once()
        assert tab._code is None
        assert tab.status == "authenticated"
        assert tab._feed["state"] == "live"

    def test_a_refused_code_request_ends_the_attempt_before_any_wait(self):
        api = Mock()
        api.login.return_value = (None, "that number has no WhatsApp account")
        tab = make_tab(api)
        start_entry(tab, PHONE[1:])
        type_keys(tab, ["enter"])
        api.wait_paired.assert_not_called()
        api.finish_login.assert_not_called()
        assert "no WhatsApp account" in tab._status_message
        assert tab._code is None

    def test_a_pairing_the_phone_never_accepts_is_not_finished(self):
        api = Mock()
        api.login.return_value = (CODE, None)
        api.wait_paired.return_value = "the phone did not accept the code in time"
        tab = make_tab(api)
        start_entry(tab, PHONE[1:])
        type_keys(tab, ["enter"])
        api.finish_login.assert_not_called()
        api.get_auth_status.assert_not_called()
        assert tab._code is None
        assert tab.status == "needs_auth"
        assert "did not accept" in tab._status_message


@pytest.mark.unit
class TestFeedTile:
    def _value_and_segments(self, tab):
        (_, value, _, segments), = tab._tiles()
        return value, [text for text, _ in segments]

    def test_a_paused_feed_reads_paused_whatever_its_state(self):
        tab = make_tab()
        tab._feed = {"state": "catching_up", "stored": 1200, "items": "messages", "paused": True}
        value, segments = self._value_and_segments(tab)
        assert value == "paused"
        assert any("1,200" in s for s in segments)

    def test_a_feed_error_replaces_the_count(self):
        tab = make_tab()
        tab._feed = {"state": "failed", "stored": 1200, "items": "messages", "last_error": "socket closed"}
        value, segments = self._value_and_segments(tab)
        assert value == "failed"
        assert segments == ["socket closed"]

    def test_no_feed_status_is_not_running(self):
        tab = make_tab()
        tab._feed = None
        _, segments = self._value_and_segments(tab)
        assert segments == ["not running"]
