from datetime import date, datetime
from unittest.mock import Mock, patch

import pytest

from integrations.reminders import RemindersIntegration as module
from integrations.reminders.RemindersCommands import RemindersCommands
from integrations.reminders.RemindersIntegration import (RemindersIntegration, add_reminder_item, complete_reminder,
                                                         list_open_reminders)


def a_list(title, shared=False):
    cal = Mock()
    cal.title.return_value = title
    cal.sharingStatus.return_value = 1 if shared else 0
    return cal


def a_reminder(title, calendar, due=None, notes=None):
    """due: None, a date (no time of day: EventKit leaves the hour undefined) or a datetime."""
    reminder = Mock()
    reminder.title.return_value = title
    reminder.calendar.return_value = calendar
    reminder.notes.return_value = notes
    if due is None:
        reminder.dueDateComponents.return_value = None
    else:
        components = Mock()
        components.year.return_value, components.month.return_value, components.day.return_value = due.year, due.month, due.day
        has_time = isinstance(due, datetime)
        components.hour.return_value = due.hour if has_time else module._UNDEFINED
        components.minute.return_value = due.minute if has_time else module._UNDEFINED
        reminder.dueDateComponents.return_value = components
    return reminder


def a_store(lists, save_ok=True, save_error=None, reminders=(), answers=True):
    """A store whose fetch hands back the `reminders` that belong to the calendars asked for;
    one that never answers when `answers` is False."""
    store = Mock()
    store.calendarsForEntityType_.return_value = lists
    store.saveReminder_commit_error_.return_value = (save_ok, save_error)
    store.predicateForIncompleteRemindersWithDueDateStarting_ending_calendars_.side_effect = \
        lambda start, end, calendars: ("predicate", list(calendars))

    def fetch(predicate, done):
        if answers:
            done([r for r in reminders if r.calendar() in predicate[1]])
    store.fetchRemindersMatchingPredicate_completion_.side_effect = fetch
    return store


@pytest.fixture
def integration():
    RemindersIntegration.reset_instance()
    yield RemindersIntegration.get_instance()
    RemindersIntegration.reset_instance()


@pytest.fixture
def eventkit():
    with patch.object(module, "EventKit") as ek:
        yield ek


class TestAddReminderItem:
    def test_the_item_is_saved_to_the_named_list(self, integration, eventkit):
        groceries = a_list("Groceries")
        store = a_store([a_list("Todo"), groceries])
        with patch.object(integration, "_event_store", return_value=store):
            assert add_reminder_item("Milk", list_name="Groceries") == ""
        reminder = eventkit.EKReminder.reminderWithEventStore_.return_value
        reminder.setTitle_.assert_called_once_with("Milk")
        reminder.setCalendar_.assert_called_once_with(groceries)
        store.saveReminder_commit_error_.assert_called_once_with(reminder, True, None)

    def test_the_list_name_is_matched_regardless_of_case(self, integration, eventkit):
        groceries = a_list("Groceries")
        store = a_store([a_list("Todo"), groceries])
        with patch.object(integration, "_event_store", return_value=store):
            assert add_reminder_item("Milk", list_name="groceries") == ""
        eventkit.EKReminder.reminderWithEventStore_.return_value.setCalendar_.assert_called_once_with(groceries)

    def test_the_list_comes_from_config_when_none_is_named(self, integration, eventkit):
        groceries = a_list("Groceries")
        store = a_store([groceries])
        with patch.object(integration, "_event_store", return_value=store), \
             patch.object(module, "config") as config:
            config.get.return_value = "Groceries"
            assert add_reminder_item("Eggs") == ""
        config.get.assert_called_with("reminders.shopping_list_name")
        eventkit.EKReminder.reminderWithEventStore_.return_value.setCalendar_.assert_called_once_with(groceries)

    def test_a_shared_list_wins_over_an_unshared_namesake(self, integration, eventkit):
        shared = a_list("Groceries", shared=True)
        store = a_store([a_list("Groceries"), shared])
        with patch.object(integration, "_event_store", return_value=store):
            add_reminder_item("Bread", list_name="Groceries")
        eventkit.EKReminder.reminderWithEventStore_.return_value.setCalendar_.assert_called_once_with(shared)

    def test_a_missing_list_fails_naming_the_lists_there_are(self, integration, eventkit):
        store = a_store([a_list("Todo"), a_list("Private")])
        with patch.object(integration, "_event_store", return_value=store):
            result = add_reminder_item("Bread", list_name="Groceries")
        assert result.startswith("Failed to add reminder")
        assert "Todo" in result and "Private" in result
        store.saveReminder_commit_error_.assert_not_called()

    def test_a_refused_save_reports_eventkits_reason(self, integration, eventkit):
        refusal = Mock()
        refusal.localizedDescription.return_value = "The store is read only"
        store = a_store([a_list("Groceries")], save_ok=False, save_error=refusal)
        with patch.object(integration, "_event_store", return_value=store):
            result = add_reminder_item("Bread", list_name="Groceries")
        assert result.startswith("Failed to add reminder")
        assert "read only" in result

    def test_a_denied_store_fails_the_add_instead_of_raising(self, integration, eventkit):
        with patch.object(integration, "_event_store", side_effect=RuntimeError("access denied")):
            result = add_reminder_item("Bread", list_name="Groceries")
        assert result.startswith("Failed to add reminder")
        assert "access denied" in result


class TestCheckHealth:
    def test_passes_when_the_configured_list_exists(self, integration, eventkit):
        store = a_store([a_list("Groceries")])
        with patch.object(integration, "_event_store", return_value=store), \
             patch.object(module, "config") as config:
            config.get.return_value = "Groceries"
            assert integration.check_health() is True

    def test_fails_when_the_configured_list_is_gone(self, integration, eventkit):
        store = a_store([a_list("Todo")])
        with patch.object(integration, "_event_store", return_value=store), \
             patch.object(module, "config") as config:
            config.get.return_value = "Groceries"
            with pytest.raises(LookupError, match="Todo"):
                integration.check_health()


class TestEventStore:
    def test_the_store_is_built_once_after_access_is_granted(self, integration, eventkit):
        store = eventkit.EKEventStore.alloc.return_value.init.return_value
        store.requestFullAccessToRemindersWithCompletion_.side_effect = lambda done: done(True, None)
        assert integration._event_store() is store
        assert integration._event_store() is store
        eventkit.EKEventStore.alloc.assert_called_once()
        store.requestFullAccessToRemindersWithCompletion_.assert_called_once()

    def test_denied_access_says_where_to_grant_it(self, integration, eventkit):
        store = eventkit.EKEventStore.alloc.return_value.init.return_value
        store.requestFullAccessToRemindersWithCompletion_.side_effect = lambda done: done(False, None)
        with pytest.raises(RuntimeError, match="Privacy & Security"):
            integration._event_store()
        assert integration._store is None

    def test_an_unanswered_prompt_times_out_instead_of_hanging(self, integration, eventkit):
        store = eventkit.EKEventStore.alloc.return_value.init.return_value
        store.requestFullAccessToRemindersWithCompletion_.side_effect = lambda done: None
        with patch.object(RemindersIntegration, "ACCESS_TIMEOUT", 0.01):
            with pytest.raises(RuntimeError, match="not answered"):
                integration._event_store()

    def test_the_pre_macos_14_access_call_is_used_when_the_new_one_is_missing(self, integration, eventkit):
        store = Mock(spec=["requestAccessToEntityType_completion_"])
        store.requestAccessToEntityType_completion_.side_effect = lambda kind, done: done(True, None)
        eventkit.EKEventStore.alloc.return_value.init.return_value = store
        assert integration._event_store() is store
        store.requestAccessToEntityType_completion_.assert_called_once()

    def test_without_eventkit_the_store_cannot_be_built(self, integration):
        with patch.object(module, "EventKit", None):
            with pytest.raises(RuntimeError, match="EventKit"):
                integration._event_store()

    def test_reset_drops_the_store(self, integration, eventkit):
        integration._store = Mock()
        RemindersIntegration.reset_instance()
        assert integration._store is None


@pytest.fixture
def shopping_list_config():
    with patch.object(module, "config") as config:
        config.get.return_value = "Groceries"
        yield config


class TestListOpenReminders:
    def test_every_list_is_shown_with_its_open_items_the_shopping_list_first(self, integration, eventkit,
                                                                              shopping_list_config):
        groceries, chores = a_list("Groceries", shared=True), a_list("Chores")
        reminders = [a_reminder("Mow the lawn", chores, due=date(2026, 10, 14)),
                     a_reminder("Milk", groceries), a_reminder("Eggs", groceries, notes="free range\nsecond line"),
                     a_reminder("Call the plumber", chores, due=datetime(2026, 10, 13, 9, 30))]
        store = a_store([chores, groceries, a_list("Archive")], reminders=reminders)
        with patch.object(integration, "_event_store", return_value=store):
            text = list_open_reminders()
        blocks = text.split("\n\n")
        assert [b.splitlines()[0] for b in blocks] == ["## Groceries (2 open, shared)", "## Archive (empty)",
                                                        "## Chores (2 open)"]
        assert "- Milk" in blocks[0] and "- Eggs — free range" in blocks[0] and "second line" not in text
        assert "- Mow the lawn (due 2026-10-14)" in blocks[2]
        assert "- Call the plumber (due 2026-10-13 09:30)" in blocks[2]

    def test_one_list_can_be_read_alone(self, integration, eventkit, shopping_list_config):
        groceries, chores = a_list("Groceries"), a_list("Chores")
        store = a_store([groceries, chores], reminders=[a_reminder("Milk", groceries), a_reminder("Mow", chores)])
        with patch.object(integration, "_event_store", return_value=store):
            text = list_open_reminders("chores")
        assert "Mow" in text and "Milk" not in text and "Groceries" not in text
        _, _, asked = store.predicateForIncompleteRemindersWithDueDateStarting_ending_calendars_.call_args[0]
        assert list(asked) == [chores]

    def test_nothing_open_anywhere_is_said_in_one_sentence(self, integration, eventkit, shopping_list_config):
        store = a_store([a_list("Groceries"), a_list("Chores")])
        with patch.object(integration, "_event_store", return_value=store):
            text = list_open_reminders()
        assert text.startswith("No open reminders")
        assert "Groceries" in text and "Chores" in text and "##" not in text

    def test_a_store_that_does_not_answer_times_out_instead_of_hanging(self, integration, eventkit,
                                                                        shopping_list_config):
        store = a_store([a_list("Groceries")], answers=False)
        with patch.object(integration, "_event_store", return_value=store), \
             patch.object(RemindersIntegration, "FETCH_TIMEOUT", 0.01):
            text = list_open_reminders()
        assert text.startswith("Failed to list reminders")
        assert "did not answer" in text

    def test_an_unknown_list_fails_naming_the_lists_there_are(self, integration, eventkit, shopping_list_config):
        store = a_store([a_list("Groceries"), a_list("Chores")])
        with patch.object(integration, "_event_store", return_value=store):
            text = list_open_reminders("Holiday")
        assert text.startswith("Failed to list reminders") and "Chores" in text


class TestCompleteReminder:
    def test_an_exact_title_is_ticked_off_and_saved(self, integration, eventkit):
        chores = a_list("Chores")
        mow, mow_more = a_reminder("Mow the lawn", chores), a_reminder("Mow the lawn again", chores)
        store = a_store([chores], reminders=[mow, mow_more])
        with patch.object(integration, "_event_store", return_value=store):
            assert complete_reminder("Mow the lawn", "Chores") == ""
        mow.setCompleted_.assert_called_once_with(True)
        mow_more.setCompleted_.assert_not_called()
        store.saveReminder_commit_error_.assert_called_once_with(mow, True, None)

    def test_a_title_matches_regardless_of_case_and_as_a_unique_fragment(self, integration, eventkit):
        chores = a_list("Chores")
        plumber, lawn = a_reminder("Call the plumber", chores), a_reminder("Mow the lawn", chores)
        store = a_store([chores], reminders=[plumber, lawn])
        with patch.object(integration, "_event_store", return_value=store):
            assert complete_reminder("call the PLUMBER", "Chores") == ""
            assert complete_reminder("lawn", "Chores") == ""
        plumber.setCompleted_.assert_called_once_with(True)
        lawn.setCompleted_.assert_called_once_with(True)

    def test_a_title_equal_but_for_its_case_wins_over_the_titles_that_merely_contain_it(self, integration, eventkit):
        """'milk' is a part of both 'Milk' and 'Milk powder'; as a fragment it would be ambiguous,
        but it is 'Milk' whatever the case, so that one is ticked off."""
        shopping = a_list("Shopping")
        milk, powder = a_reminder("Milk", shopping), a_reminder("Milk powder", shopping)
        store = a_store([shopping], reminders=[milk, powder])
        with patch.object(integration, "_event_store", return_value=store):
            assert complete_reminder("milk", "Shopping") == ""
        milk.setCompleted_.assert_called_once_with(True)
        powder.setCompleted_.assert_not_called()

    def test_an_ambiguous_fragment_is_refused_naming_the_candidates(self, integration, eventkit):
        chores = a_list("Chores")
        reminders = [a_reminder("Call the plumber", chores), a_reminder("Pay the plumber", chores)]
        store = a_store([chores], reminders=reminders)
        with patch.object(integration, "_event_store", return_value=store):
            result = complete_reminder("plumber", "Chores")
        assert result.startswith("Failed to complete reminder")
        assert "Call the plumber" in result and "Pay the plumber" in result
        store.saveReminder_commit_error_.assert_not_called()

    def test_a_missing_title_is_refused_naming_the_open_items(self, integration, eventkit):
        chores = a_list("Chores")
        store = a_store([chores], reminders=[a_reminder("Mow the lawn", chores)])
        with patch.object(integration, "_event_store", return_value=store):
            result = complete_reminder("Paint the fence", "Chores")
        assert result.startswith("Failed to complete reminder") and "Mow the lawn" in result
        store.saveReminder_commit_error_.assert_not_called()

    def test_a_refused_save_reports_eventkits_reason(self, integration, eventkit):
        chores = a_list("Chores")
        refusal = Mock()
        refusal.localizedDescription.return_value = "The store is read only"
        store = a_store([chores], save_ok=False, save_error=refusal, reminders=[a_reminder("Mow", chores)])
        with patch.object(integration, "_event_store", return_value=store):
            result = complete_reminder("Mow", "Chores")
        assert result.startswith("Failed to complete reminder") and "read only" in result


class TestCommands:
    @pytest.fixture
    def commands(self):
        return RemindersCommands(Mock())

    def test_add_rejoins_an_item_the_llm_split_on_a_comma(self, commands):
        with patch.object(module.RemindersIntegration, "get_instance") as instance:
            instance.return_value.add_reminder_item.return_value = ""
            assert commands.cmd_add_reminder("Chores", "Call the plumber", " then the electrician") == ""
        instance.return_value.add_reminder_item.assert_called_once_with("Call the plumber, then the electrician",
                                                                        "Chores")

    def test_complete_passes_the_item_and_its_list(self, commands):
        with patch.object(module.RemindersIntegration, "get_instance") as instance:
            instance.return_value.complete_reminder.return_value = ""
            assert commands.cmd_complete_reminder("Chores", "Mow the lawn") == ""
        instance.return_value.complete_reminder.assert_called_once_with("Mow the lawn", "Chores")

    @pytest.mark.parametrize("args", [(), ("Chores",), ("Chores", ""), ("", "Milk")])
    def test_a_write_without_a_list_and_an_item_is_refused_before_touching_the_store(self, commands, args):
        with patch.object(module.RemindersIntegration, "get_instance") as instance:
            assert commands.cmd_add_reminder(*args).startswith("add_reminder failed")
            assert commands.cmd_complete_reminder(*args).startswith("complete_reminder failed")
        instance.assert_not_called()

    def test_listing_runs_in_the_background_with_the_list_asked_for(self, commands):
        commands.cmd_list_reminders("")
        commands.cmd_list_reminders(" Chores ")
        calls = [c.args[1] for c in commands.commands.background_command.call_args_list]
        assert calls == [None, "Chores"]
