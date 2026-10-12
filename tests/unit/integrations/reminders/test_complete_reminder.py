"""`complete_reminder(text)` ticks off the one open reminder whose title matches case-insensitively,
refuses several matches naming them with their lists, and reports no match."""
from unittest.mock import Mock, patch

import pytest

import subprocess

from integrations.reminders.RemindersIntegration import (
    OpenReminder, RemindersIntegration, complete_reminder, parse_lists, parse_open_reminders,
)

MODULE = "integrations.reminders.RemindersIntegration"


def _open(*pairs):
    """Reminders in lists whose id is derived from the name, enough for a test that does not
    care about two lists sharing a name."""
    return [OpenReminder(f"id-{list_name}", list_name, name) for list_name, name in pairs]


@pytest.mark.unit
class TestCompleteReminder:
    def test_the_single_case_insensitive_match_is_completed_in_its_own_list(self):
        integration = RemindersIntegration()
        with patch.object(integration, "open_reminders", return_value=_open(("Shoppinglista", "Milk"), ("Todo", "Call Alice"))), \
                patch(f"{MODULE}.subprocess") as sp:
            sp.run.return_value = Mock(returncode=0, stderr="")
            assert integration.complete_reminder("  mILK ") == ""

        script = sp.run.call_args[0][0][2]
        assert 'tell list id "id-Shoppinglista"' in script
        assert 'tell list "' not in script
        assert 'name is "Milk"' in script
        assert "completed is false" in script
        assert "to true" in script

    def test_no_match_completes_nothing_and_says_so(self):
        integration = RemindersIntegration()
        with patch.object(integration, "open_reminders", return_value=_open(("Todo", "Call Alice"))), \
                patch(f"{MODULE}.subprocess") as sp:
            result = integration.complete_reminder("Milk")

        assert "Milk" in result and "No open reminder" in result
        sp.run.assert_not_called()

    def test_several_matches_are_refused_and_named_with_their_lists(self):
        integration = RemindersIntegration()
        with patch.object(integration, "open_reminders", return_value=_open(("Shoppinglista", "Milk"), ("Todo", "milk"), ("Todo", "Bread"))), \
                patch(f"{MODULE}.subprocess") as sp:
            result = integration.complete_reminder("milk")

        assert result
        assert "Shoppinglista" in result and "Todo" in result
        assert "Bread" not in result
        sp.run.assert_not_called()

    def test_every_list_is_asked_for_its_open_reminders_by_id_in_its_own_query(self):
        integration = RemindersIntegration()
        answers = {"L1": "Milk\nBread\n", "L2": "Call Alice\n", "L3": ""}

        def run(args, **_):
            script = args[2]
            if "every list" in script:
                return Mock(returncode=0, stderr="", stdout="L1\tShoppinglista\nL2\tTodo\nL3\tDone\n")
            asked = next(list_id for list_id in answers if f'list id "{list_id}"' in script)
            assert "completed is false" in script
            return Mock(returncode=0, stderr="", stdout=answers[asked])

        with patch(f"{MODULE}.subprocess") as sp:
            sp.run.side_effect = run
            found = integration.open_reminders()

        assert found == [OpenReminder("L1", "Shoppinglista", "Milk"), OpenReminder("L1", "Shoppinglista", "Bread"),
                         OpenReminder("L2", "Todo", "Call Alice")]
        assert sp.run.call_count == 4

    def test_two_lists_sharing_a_name_are_read_and_completed_as_two_lists(self):
        # A name resolves to the first list of that name, so the same reminder would be
        # read twice and the completion could land in the wrong list.
        integration = RemindersIntegration()
        answers = {"L1": "Milk\n", "L2": "Eggs\n"}

        def run(args, **_):
            script = args[2]
            if "every list" in script:
                return Mock(returncode=0, stderr="", stdout="L1\tShoppinglista\nL2\tShoppinglista\n")
            if "set completed" in script:
                return Mock(returncode=0, stderr="", stdout="")
            asked = next(list_id for list_id in answers if f'list id "{list_id}"' in script)
            return Mock(returncode=0, stderr="", stdout=answers[asked])

        with patch(f"{MODULE}.subprocess") as sp:
            sp.run.side_effect = run
            assert integration.complete_reminder("eggs") == ""
        assert 'tell list id "L2"' in sp.run.call_args[0][0][2]

    def test_a_list_that_fails_to_read_is_named_in_the_error(self):
        integration = RemindersIntegration()
        with patch(f"{MODULE}.subprocess") as sp:
            sp.run.side_effect = [Mock(returncode=0, stderr="", stdout="L1\tShoppinglista\nL2\tTodo\n"),
                                  Mock(returncode=0, stderr="", stdout="Milk\n"),
                                  Mock(returncode=1, stderr="execution error: boom (-1700)", stdout="")]
            result = integration.complete_reminder("Milk")
        assert result.startswith("Failed to read the reminders")
        assert "boom" in result and "Todo" in result

    def test_a_list_that_hangs_is_reported_with_its_name(self):
        integration = RemindersIntegration()
        with patch(f"{MODULE}.subprocess") as sp:
            sp.TimeoutExpired = subprocess.TimeoutExpired
            sp.run.side_effect = [Mock(returncode=0, stderr="", stdout="L1\tTodo\n"),
                                  subprocess.TimeoutExpired("osascript", 120)]
            result = integration.complete_reminder("Milk")
        assert result.startswith("Failed to read the reminders")
        assert "Todo" in result

    def test_empty_text_is_refused_without_asking_reminders(self):
        integration = RemindersIntegration()
        with patch(f"{MODULE}.subprocess") as sp:
            assert integration.complete_reminder("   ").startswith("complete_reminder failed")
        sp.run.assert_not_called()

    def test_a_failing_listing_is_reported_not_raised(self):
        integration = RemindersIntegration()
        with patch(f"{MODULE}.subprocess") as sp:
            sp.run.return_value = Mock(returncode=1, stderr="Reminders got an error", stdout="")
            result = integration.complete_reminder("Milk")
        assert "Reminders got an error" in result

    def test_a_failing_completion_is_reported(self):
        integration = RemindersIntegration()
        with patch.object(integration, "open_reminders", return_value=_open(("Todo", "Milk"))), \
                patch(f"{MODULE}.subprocess") as sp:
            sp.run.return_value = Mock(returncode=1, stderr="execution error")
            result = integration.complete_reminder("Milk")
        assert "Failed to complete reminder" in result and "execution error" in result

    def test_quotes_in_a_title_are_escaped_in_the_script(self):
        integration = RemindersIntegration()
        with patch.object(integration, "open_reminders", return_value=_open(("Todo", 'Buy "the" book'))), \
                patch(f"{MODULE}.subprocess") as sp:
            sp.run.return_value = Mock(returncode=0, stderr="")
            assert integration.complete_reminder('buy "the" book') == ""
        assert r'name is "Buy \"the\" book"' in sp.run.call_args[0][0][2]


@pytest.mark.unit
class TestScriptOutputParsing:
    def test_each_lists_line_becomes_an_id_and_a_name(self):
        output = "A1B2\tShoppinglista\nC3D4\tTodo list, home\n"
        assert parse_lists(output) == [("A1B2", "Shoppinglista"), ("C3D4", "Todo list, home")]

    def test_a_lists_line_without_the_separator_is_skipped(self):
        assert parse_lists("A1B2\tTodo\nstray\n\n") == [("A1B2", "Todo")]

    def test_each_name_line_becomes_a_reminder_of_the_list(self):
        output = "Milk\nOat milk, 2 l\nCall \"Alice\"\n"
        assert parse_open_reminders(output, "A1B2", "Shoppinglista") == [
            OpenReminder("A1B2", "Shoppinglista", "Milk"),
            OpenReminder("A1B2", "Shoppinglista", "Oat milk, 2 l"),
            OpenReminder("A1B2", "Shoppinglista", 'Call "Alice"'),
        ]

    def test_blank_lines_are_no_reminders(self):
        assert parse_open_reminders("\n  \nBread\n", "A1B2", "Todo") == [OpenReminder("A1B2", "Todo", "Bread")]
        assert parse_open_reminders("", "A1B2", "Todo") == []


@pytest.mark.unit
class TestCommand:
    def test_the_command_runs_in_the_background_with_its_text(self, commands_instance):
        owner = commands_instance._owner("complete_reminder")
        with patch.object(owner, "background_command") as bg:
            owner.cmd_complete_reminder("Milk")
        bg.assert_called_once_with(complete_reminder, "Milk")

    def test_the_reminders_toolkit_documents_the_command(self):
        from assistant.commands import Toolkits
        toolkit = Toolkits.owner_of("complete_reminder")
        assert toolkit is not None and "add_shopping_list_item" in toolkit.commands
        assert "complete_reminder(" in toolkit.text()
