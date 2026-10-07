"""`microsoft_login(account)`: the code and URL as the command's result and a system note,
the wait for the user on a thread of its own, and a second note at the end."""
import time
from unittest.mock import Mock, patch

import pytest

from assistant.commands.Commands import Commands
from integrations.microsoft.MicrosoftAuth import DeviceFlow, NotSignedIn

MODULE = "integrations.microsoft.MicrosoftCommands"


class InlineThread:
    """`threading.Thread` whose start runs the target at once: the wait is tested in line."""

    def __init__(self, target=None, args=(), name=None, daemon=None):
        self._target, self._args = target, args

    def start(self):
        self._target(*self._args)


def _integration(finish=""):
    integration = Mock()
    integration.login_begin.return_value = DeviceFlow("ABCD-1234", "https://microsoft.com/devicelogin", time.time() + 900, {"device_code": "dc"})
    integration.login_finish.return_value = finish
    account = Mock()
    account.name = "Work"
    account.auth.signed_in_as.return_value = "marcus@company.example"
    integration.account.return_value = account
    return integration


def _monitor(integration):
    monitor = Mock()
    monitor.find.return_value = integration
    return monitor


@pytest.mark.unit
class TestCmdMicrosoftLogin:
    def test_runs_in_the_background_with_the_account_name(self, commands_instance):
        with patch.object(commands_instance, "background_command") as bg:
            assert commands_instance.run("microsoft_login", ['"Work"']) == ""
        bg.assert_called_once_with(commands_instance._owner("microsoft_login")._run_microsoft_login, ("Work",), lane=Commands.CONFIG_LANE)

    def test_without_a_name_it_refuses(self, commands_instance):
        with patch.object(commands_instance, "background_command") as bg:
            assert commands_instance.run("microsoft_login", [""]).startswith("microsoft_login failed")
        bg.assert_not_called()

    def test_the_code_comes_back_as_the_result_and_a_note_then_a_note_says_signed_in(self, commands_instance):
        integration = _integration()
        monitor = _monitor(integration)
        with patch.object(Commands, "_integration_monitor", return_value=monitor), \
             patch(f"{MODULE}.system_note") as note, patch(f"{MODULE}.threading.Thread", InlineThread):
            reply = commands_instance._owner("microsoft_login")._run_microsoft_login("Work")

        assert "ABCD-1234" in reply and "https://microsoft.com/devicelogin" in reply and "Work" in reply
        first, second = [c.args[1] for c in note.call_args_list]
        assert "ABCD-1234" in first
        assert "signed in" in second and "marcus@company.example" in second
        integration.login_finish.assert_called_once_with("Work", integration.login_begin.return_value)
        monitor.check_now.assert_called_once_with("Microsoft 365")

    def test_an_expired_code_ends_in_a_note_that_says_to_run_it_again(self, commands_instance):
        integration = _integration(finish="the code expired before it was entered")
        with patch.object(Commands, "_integration_monitor", return_value=_monitor(integration)), \
             patch(f"{MODULE}.system_note") as note, patch(f"{MODULE}.threading.Thread", InlineThread):
            commands_instance._owner("microsoft_login")._run_microsoft_login("Work")

        last = note.call_args_list[-1].args[1]
        assert "expired" in last and 'microsoft_login("Work")' in last

    def test_an_unknown_account_and_a_disabled_integration_are_errors(self, commands_instance):
        integration = _integration()
        integration.login_begin.side_effect = ValueError("no Microsoft account named 'Nobody' (configured: Work)")
        with patch.object(Commands, "_integration_monitor", return_value=_monitor(integration)), patch(f"{MODULE}.system_note") as note:
            text = commands_instance._owner("microsoft_login")._run_microsoft_login("Nobody")
        assert text.startswith("microsoft_login failed") and "Work" in text
        note.assert_not_called()

        disabled = Mock(spec=["state"])
        disabled.state = Mock(value="disabled")
        with patch.object(Commands, "_integration_monitor", return_value=_monitor(disabled)):
            assert 'enable_integration("Microsoft 365")' in commands_instance._owner("microsoft_login")._run_microsoft_login("Work")

    def test_a_missing_client_id_is_the_commands_answer_not_an_escaped_exception(self, commands_instance):
        """An enabled but not configured integration is still the instance the monitor holds,
        so login_begin reaches MSAL without a client id and raises NotSignedIn — the LLM must
        get that reason back, not a generic command failure."""
        integration = _integration()
        integration.login_begin.side_effect = NotSignedIn("microsoft.app.client_id is not set")
        with patch.object(Commands, "_integration_monitor", return_value=_monitor(integration)), patch(f"{MODULE}.system_note") as note:
            text = commands_instance._owner("microsoft_login")._run_microsoft_login("Work")
        assert text.startswith("microsoft_login failed") and "client_id" in text
        note.assert_not_called()

    def test_a_wait_that_raises_ends_in_a_note_naming_the_error_and_no_health_check(self, commands_instance):
        integration = _integration()
        integration.login_finish.side_effect = ConnectionError("login.microsoftonline.com unreachable")
        monitor = _monitor(integration)
        with patch.object(Commands, "_integration_monitor", return_value=monitor), \
             patch(f"{MODULE}.system_note") as note, patch(f"{MODULE}.threading.Thread", InlineThread):
            reply = commands_instance._owner("microsoft_login")._run_microsoft_login("Work")

        assert "ABCD-1234" in reply
        last = note.call_args_list[-1].args[1]
        assert "ConnectionError" in last and "unreachable" in last and 'microsoft_login("Work")' in last
        assert "signed in" not in last
        monitor.check_now.assert_not_called()
