"""`complete_reminder(text)` ticks off the one open reminder whose title matches case-insensitively,
refuses several matches naming them with their lists, and reports no match."""
from unittest.mock import Mock, patch

import pytest

from integrations.reminders.RemindersIntegration import OpenReminder, RemindersIntegration, complete_reminder

MODULE = "integrations.reminders.RemindersIntegration"


def _open(*pairs):
    return [OpenReminder(list_name, name) for list_name, name in pairs]


@pytest.mark.unit
class TestCompleteReminder:
    def test_the_single_case_insensitive_match_is_completed_in_its_own_list(self):
        integration = RemindersIntegration()
        with patch.object(integration, "open_reminders", return_value=_open(("Shoppinglista", "Milk"), ("Todo", "Call Alice"))), \
                patch(f"{MODULE}.subprocess") as sp:
            sp.run.return_value = Mock(returncode=0, stderr="")
            assert integration.complete_reminder("  mILK ") == ""

        script = sp.run.call_args[0][0][2]
        assert 'tell list "Shoppinglista"' in script
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

    def test_only_open_reminders_are_asked_for(self):
        integration = RemindersIntegration()
        with patch(f"{MODULE}.subprocess") as sp:
            sp.run.return_value = Mock(returncode=0, stderr="", stdout="Shoppinglista\tMilk\nTodo\tCall Alice\nbroken line\n")
            found = integration.open_reminders()

        assert found == _open(("Shoppinglista", "Milk"), ("Todo", "Call Alice"))
        assert "completed is false" in sp.run.call_args[0][0][2]

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
class TestCommand:
    def test_the_command_hands_its_text_to_the_integration(self, commands_instance):
        owner = commands_instance._owner("complete_reminder")
        with patch(f"{MODULE}.RemindersIntegration.get_instance") as get:
            get.return_value.complete_reminder.return_value = ""
            assert owner.cmd_complete_reminder("Milk") == ""
        get.return_value.complete_reminder.assert_called_once_with("Milk")

    def test_the_reminders_toolkit_documents_the_command(self):
        from assistant.commands import Toolkits
        toolkit = Toolkits.owner_of("complete_reminder")
        assert toolkit is not None and "add_shopping_list_item" in toolkit.commands
        assert "complete_reminder(" in toolkit.text()
