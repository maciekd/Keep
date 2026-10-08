"""The google_login and google_login_code commands: the browser sign-in of a Google account from any chat."""
from assistant.commands.CommandSet import CommandSet
from assistant.SystemNote import system_note
from common.logging import info

INTEGRATION = "Google"
NOTE_SOURCE = "Google"


class GoogleCommands(CommandSet):
    """Signs a Google account in from the chat: the URL out, the address the browser lands on back in."""

    def cmd_google_login(self, parameter_string: str):
        """Starts the flow for the named account: the sign-in URL and what to do with it come
        back as the command's result and as a system note in every chat."""
        name = self.commands.first_argument(parameter_string)
        if not name:
            return 'google_login failed: name the account, e.g. google_login("Personal")'
        self.background_command(self._run_google_login, (name,), lane=self.commands.CONFIG_LANE)

    def cmd_google_login_code(self, parameter_string: str):
        """Finishes the flow with the whole address the browser landed on."""
        pasted = parameter_string.strip().strip("\"'`").strip()
        if not pasted:
            return 'google_login_code failed: paste the address the browser landed on, e.g. google_login_code("http://localhost:18722/?state=...&code=...")'
        self.background_command(self._run_google_login_code, (pasted,), lane=self.commands.CONFIG_LANE)

    def _integration(self, command: str):
        """(integration, "") when Google can sign in, else (None, the reason)."""
        monitor = self.commands.integration_monitor()
        integration = monitor.find(INTEGRATION)
        if integration is None or not hasattr(integration, "login_begin"):
            state = getattr(integration, "state", None)
            label = self.commands.state_label(getattr(state, "value", state), "not available")
            info(f"{command}: {INTEGRATION} is {label}, no sign-in possible")
            if label == "disabled":
                return None, f'{INTEGRATION} is disabled — enable_integration("{INTEGRATION}") first.'
            return None, f"{INTEGRATION} is {label} — set google.oauth.client_id, google.oauth.client_secret and google.accounts and enable it first."
        return integration, ""

    def _run_google_login(self, name: str) -> str:
        integration, problem = self._integration("google_login")
        if integration is None:
            return problem
        try:
            sign_in = integration.login_begin(name)
        except Exception as e:   # an unknown account, no client id, an account with neither flag on

            info(f"google_login: {e}")
            return f"google_login failed: {e}"
        account = integration.account(name)
        text = (f"Google sign-in for {account.name}: {sign_in.instructions} into the chat as "
                f'google_login_code("<the pasted address>") (quote it). The link is valid for about ten minutes.')
        system_note(NOTE_SOURCE, text)
        return text

    def _run_google_login_code(self, pasted: str) -> str:
        integration, problem = self._integration("google_login_code")
        if integration is None:
            return problem
        account = integration.login_account_for(pasted)
        try:
            failure = integration.login_finish(pasted)
        except Exception as e:
            failure = f"{type(e).__name__}: {e}"
        name = account.name if account is not None else "the account"
        if failure:
            info(f"google_login_code: {name} not signed in — {failure}")
            return f"Google sign-in for {name} did not complete: {failure}"
        signed_in_as = account.auth.signed_in_as()
        text = (f"Google: {name} is signed in" + (f" as {signed_in_as}" if signed_in_as else "")
                + ". Its mail and calendars are available from the next request.")
        system_note(NOTE_SOURCE, text)
        try:
            self.commands.integration_monitor().check_now(INTEGRATION)
        except Exception as e:
            info(f"google_login_code: health check after the sign-in failed: {e}")
        return text

    def describe_google_login(self, p):
        name = self.commands.first_argument(p)
        return f"Signing the Google account {name} in" if name else "Signing a Google account in"

    def describe_google_login_code(self, p):
        return "Finishing the Google sign-in"
