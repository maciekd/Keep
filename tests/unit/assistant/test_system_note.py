"""SystemNote: an integration's event lands in the dialog history (for the LLM and the
Home tab) and is pushed to the chats that do not render the history."""
from unittest.mock import Mock

import pytest

from assistant import SystemNote
from assistant.Notifier import set_notifier
from assistant.SystemNote import system_note


@pytest.fixture(autouse=True)
def detach():
    yield
    SystemNote.set_history(None)
    set_notifier(None)


@pytest.fixture
def chat():
    n = Mock()
    n.send_message.return_value = True
    set_notifier(n)
    return n


class TestSystemNote:
    def test_writes_the_row_and_pushes_it_as_already_in_history(self, chat):
        history = Mock()
        SystemNote.set_history(history)
        assert system_note("Ring", "Ring 2FA code submitted. ✅", timestamp=1234.5) is True
        history.add_context_message.assert_called_once_with(
            "system", "Ring 2FA code submitted. ✅", 1234.5, source="Ring")
        chat.send_message.assert_called_once_with("Ring 2FA code submitted. ✅", disable_notification=False, in_history=True)

    def test_a_history_only_note_is_stored_and_not_pushed(self, chat):
        history = Mock()
        SystemNote.set_history(history)
        assert system_note("voice", "Voice is ready.", timestamp=7.0, push=False) is True
        history.add_context_message.assert_called_once_with("system", "Voice is ready.", 7.0, source="voice")
        chat.send_message.assert_not_called()

    def test_a_history_only_note_without_a_history_reaches_nothing(self, chat):
        assert system_note("voice", "Voice is ready.", push=False) is False
        chat.send_message.assert_not_called()

    def test_without_a_history_the_note_is_only_pushed(self, chat):
        assert system_note("Arlo", "Arlo 2FA required.") is True
        chat.send_message.assert_called_once_with("Arlo 2FA required.", disable_notification=False, in_history=False)

    def test_without_history_or_chat_reports_nothing_reached(self):
        assert system_note("Arlo", "Arlo 2FA required.") is False

    def test_stored_counts_even_when_no_chat_took_it(self, chat):
        chat.send_message.return_value = False
        SystemNote.set_history(Mock())
        assert system_note("iCloud", "iCloud session expired.") is True

    def test_empty_text_is_skipped(self, chat):
        history = Mock()
        SystemNote.set_history(history)
        assert system_note("Ring", "  \n") is False
        history.add_context_message.assert_not_called()
        chat.send_message.assert_not_called()

    def test_text_is_stripped(self, chat):
        history = Mock()
        SystemNote.set_history(history)
        system_note("Ring", "  note \n", timestamp=1.0)
        history.add_context_message.assert_called_once_with("system", "note", 1.0, source="Ring")
