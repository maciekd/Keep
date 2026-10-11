"""The reminder commands: how the arguments reach the store and how its answers come back."""
from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from integrations.reminders.RemindersCommands import RemindersCommands
from integrations.reminders.RemindersStore import Reminder, RemindersAccessError


@pytest.fixture
def store():
    fake = Mock()
    fake.list_name = "Michelle"
    with patch.object(RemindersCommands, "_store", return_value=fake):
        yield fake


def made(**kw):
    base = dict(id="1A2B3C4D", title="Call Alice", due=None, timed=False, completed_at=None, notes="", place="")
    base.update(kw)
    return Reminder(**base)


class TestCreateReminder:
    def test_missing_trailing_arguments_are_empty_and_a_timed_due_is_said_to_ring(self, store):
        store.create.return_value = made(due=datetime(2026, 10, 13, 9, 0), timed=True)
        reply = RemindersCommands._run_create_reminder("Call Alice", "2026-10-13 09:00")
        store.create.assert_called_once_with("Call Alice", "2026-10-13 09:00", "", "")
        assert "1A2B3C4D" in reply and "2026-10-13 09:00" in reply and "alarm on the phone" in reply

    def test_a_place_reaches_the_store_as_the_fourth_argument(self, store):
        store.create.return_value = made(place="Blomsterlandet (arrive)")
        reply = RemindersCommands._run_create_reminder("Buy flowers", "", "", "Blomsterlandet; 57.70; 11.97; 150; arrive")
        assert store.create.call_args.args[3].startswith("Blomsterlandet;")
        assert "Blomsterlandet (arrive)" in reply

    def test_no_title_is_refused_before_the_store_is_touched(self, store):
        assert "failed" in RemindersCommands._run_create_reminder("")
        store.create.assert_not_called()

    def test_a_bad_due_or_place_comes_back_as_the_failure_reason(self, store):
        store.create.side_effect = ValueError("due must be 'YYYY-MM-DD HH:MM'")
        reply = RemindersCommands._run_create_reminder("x", "tomorrow")
        assert reply.startswith("create_reminder failed") and "YYYY-MM-DD" in reply

    def test_no_permission_comes_back_as_the_failure_reason(self, store):
        store.create.side_effect = RemindersAccessError("the Reminders permission was denied")
        assert "permission was denied" in RemindersCommands._run_create_reminder("x")


class TestListCompleteDelete:
    def test_list_renders_the_store_answer_under_the_list_name(self, store):
        store.list.return_value = [made(), made(id="FFFF0000", title="Done thing", completed_at=datetime(2026, 10, 10, 8, 0))]
        reply = RemindersCommands._run_list_reminders()
        assert "'Michelle'" in reply and "1A2B3C4D Call Alice" in reply and "[done 2026-10-10 08:00] FFFF0000" in reply

    def test_complete_and_delete_hand_the_id_through_and_name_the_reminder(self, store):
        store.complete.return_value = made()
        store.delete.return_value = made(title="Old one")
        assert "Call Alice" in RemindersCommands._run_complete_reminder("1a2b3c4d")
        store.complete.assert_called_once_with("1a2b3c4d")
        assert "Old one" in RemindersCommands._run_delete_reminder("1A2B3C4D")

    def test_an_unknown_id_is_a_failure_the_llm_can_read(self, store):
        store.complete.side_effect = RemindersAccessError("no reminder ZZZ in the 'Michelle' list")
        reply = RemindersCommands._run_complete_reminder("ZZZ")
        assert reply.startswith("complete_reminder failed") and "ZZZ" in reply


class TestDispatch:
    def test_the_commands_queue_background_work_with_the_first_argument_as_the_id(self):
        host = Mock()
        host.first_argument = lambda p: p.split(",", 1)[0].strip().strip("\"'")
        commands = RemindersCommands(host)
        commands.cmd_create_reminder("Call Alice", "2026-10-13 09:00")
        commands.cmd_complete_reminder("'1A2B3C4D', extra")
        commands.cmd_list_reminders("")
        queued = [c.args[1] for c in host.background_command.call_args_list]
        assert queued == [["Call Alice", "2026-10-13 09:00"], ["1A2B3C4D"], []]

    def test_create_reminder_takes_several_parameters_and_the_shopping_list_item_still_one(self):
        assert "create_reminder" in RemindersCommands.MULTI_PARAM_COMMANDS
        assert "add_shopping_list_item" not in RemindersCommands.MULTI_PARAM_COMMANDS
