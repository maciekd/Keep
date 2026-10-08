"""`google_login(account)` posts the link and what to do with it as the command's result and a
system note; `google_login_code(pasted)` finishes the flow, posts a note and re-checks the
integration."""
from unittest.mock import Mock, patch

import pytest

from assistant.commands.Commands import Commands
from assistant.integration.OAuthPaste import SignIn
from integrations.google.GoogleAuth import REDIRECT_URI

MODULE = "integrations.google.GoogleCommands"
URL = "https://accounts.google.com/o/oauth2/v2/auth?client_id=cid&state=st-1"
PASTED = f"{REDIRECT_URI}?state=st-1&code=4%2F0Acode"


def _integration(finish=""):
    integration = Mock()
    integration.login_begin.return_value = SignIn(URL, "st-1", REDIRECT_URI, "marcus@example.com")
    integration.login_finish.return_value = finish
    account = Mock()
    account.name = "Personal"
    account.auth.signed_in_as.return_value = "marcus@example.com"
    integration.account.return_value = account
    integration.login_account_for.return_value = account
    return integration


def _monitor(integration):
    monitor = Mock()
    monitor.find.return_value = integration
    return monitor


@pytest.mark.unit
class TestCmdGoogleLogin:
    def test_runs_in_the_background_with_the_account_name(self, commands_instance):
        with patch.object(commands_instance, "background_command") as bg:
            assert commands_instance.run("google_login", ['"Personal"']) == ""
        bg.assert_called_once_with(commands_instance._owner("google_login")._run_google_login, ("Personal",), lane=Commands.CONFIG_LANE)

    def test_without_a_name_it_refuses(self, commands_instance):
        with patch.object(commands_instance, "background_command") as bg:
            assert commands_instance.run("google_login", [""]).startswith("google_login failed")
        bg.assert_not_called()

    def test_the_link_and_the_paste_instruction_come_back_as_the_result_and_a_note(self, commands_instance):
        integration = _integration()
        with patch.object(Commands, "_integration_monitor", return_value=_monitor(integration)), patch(f"{MODULE}.system_note") as note:
            reply = commands_instance._owner("google_login")._run_google_login("Personal")

        assert URL in reply and "Personal" in reply and "marcus@example.com" in reply
        assert "google_login_code" in reply and "localhost:18722" in reply and "will not load" in reply
        assert note.call_args.args[1] == reply
        integration.login_begin.assert_called_once_with("Personal")

    def test_an_unknown_account_and_a_disabled_integration_are_errors(self, commands_instance):
        integration = _integration()
        integration.login_begin.side_effect = ValueError("no Google account named 'Nobody' (configured: Personal)")
        with patch.object(Commands, "_integration_monitor", return_value=_monitor(integration)), patch(f"{MODULE}.system_note") as note:
            text = commands_instance._owner("google_login")._run_google_login("Nobody")
        assert text.startswith("google_login failed") and "Personal" in text
        note.assert_not_called()

        disabled = Mock(spec=["state"])
        disabled.state = Mock(value="disabled")
        with patch.object(Commands, "_integration_monitor", return_value=_monitor(disabled)):
            assert 'enable_integration("Google")' in commands_instance._owner("google_login")._run_google_login("Personal")


@pytest.mark.unit
class TestCmdGoogleLoginCode:
    def test_the_pasted_address_goes_to_the_background_unquoted_and_whole(self, commands_instance):
        with patch.object(commands_instance, "background_command") as bg:
            assert commands_instance.run("google_login_code", [f'"{PASTED}"']) == ""
        bg.assert_called_once_with(commands_instance._owner("google_login_code")._run_google_login_code, (PASTED,), lane=Commands.CONFIG_LANE)

    def test_without_anything_pasted_it_refuses(self, commands_instance):
        with patch.object(commands_instance, "background_command") as bg:
            assert commands_instance.run("google_login_code", [""]).startswith("google_login_code failed")
        bg.assert_not_called()

    def test_a_finished_sign_in_is_a_note_and_a_health_check(self, commands_instance):
        integration = _integration()
        monitor = _monitor(integration)
        with patch.object(Commands, "_integration_monitor", return_value=monitor), patch(f"{MODULE}.system_note") as note:
            reply = commands_instance._owner("google_login_code")._run_google_login_code(PASTED)

        integration.login_finish.assert_called_once_with(PASTED)
        assert "signed in" in reply and "marcus@example.com" in reply and note.call_args.args[1] == reply
        monitor.check_now.assert_called_once_with("Google")

    def test_a_failed_exchange_is_the_reason_and_no_note(self, commands_instance):
        integration = _integration(finish='the code expired or was already used — run google_login("Personal") for a new link')
        monitor = _monitor(integration)
        with patch.object(Commands, "_integration_monitor", return_value=monitor), patch(f"{MODULE}.system_note") as note:
            reply = commands_instance._owner("google_login_code")._run_google_login_code("4/0Aold")

        assert "expired" in reply and 'google_login("Personal")' in reply and "Personal" in reply
        note.assert_not_called()
        monitor.check_now.assert_not_called()

    def test_an_exchange_that_raises_is_the_reason_not_an_escaped_exception(self, commands_instance):
        integration = _integration()
        integration.login_finish.side_effect = RuntimeError("token endpoint timed out")
        monitor = _monitor(integration)
        with patch.object(Commands, "_integration_monitor", return_value=monitor), patch(f"{MODULE}.system_note") as note:
            reply = commands_instance._owner("google_login_code")._run_google_login_code(PASTED)

        assert "token endpoint timed out" in reply and "RuntimeError" in reply and "Personal" in reply
        assert "signed in" not in reply
        note.assert_not_called()
        monitor.check_now.assert_not_called()
