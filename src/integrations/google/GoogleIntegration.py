from typing import Callable, List, Optional

from assistant.integration.Accounts import AccountHealth, STATE_ERROR, id_prefixes, named_entries
from assistant.integration.Integration import Integration
from assistant.integration.calendar.CalendarAccount import CalendarAccount, split_names
from assistant.integration.calendar.CalendarSource import CalendarSource
from assistant.integration.mail.ImapMailAccount import ImapMailAccount
from assistant.integration.mail.MailAccount import MailAccount
from assistant.integration.mail.MailSource import MailSource
from common.ConfigManager import ConfigManager
from common.logging import info, trace
from assistant.integration.OAuthPaste import SignIn, parse_pasted
from integrations.google.GoogleAuth import (SCOPE_CALENDAR, SCOPE_MAIL, GoogleAuth, NotSignedIn, delete_all_token_files, not_signed_in_hint, sign_in_again_hint)
from integrations.google.GoogleCalendarAccount import GoogleCalendarAccount

SECTION = "google"
ACCOUNTS_KEY = f"{SECTION}.accounts"
CLIENT_ID_KEY = f"{SECTION}.oauth.client_id"
CLIENT_SECRET_KEY = f"{SECTION}.oauth.client_secret"
DEFAULT_CALENDAR = "primary"
MAIL_PRESET = "gmail"


class GoogleAccount:
    """One signed-in (or not yet) Google account: its auth, and the mail (the core IMAP account
    with XOAUTH2) and calendar accounts it gives when the entry asks for them. `check()` is the
    per-account health: a token, then a calendarList read and an IMAP login."""

    def __init__(self, name: str, address: Callable[[], str], auth: GoogleAuth,
                 mail: Optional[ImapMailAccount], calendar: Optional[GoogleCalendarAccount]):
        self.name = name
        self._address = address
        self.auth = auth
        self.mail = mail
        self.calendar = calendar

    @property
    def address(self) -> str:
        return self._address() or ""

    def signed_in(self) -> bool:
        return self.auth.signed_in()

    def check(self) -> bool:
        if not self.auth.signed_in():
            raise NotSignedIn(not_signed_in_hint(self.name))
        self.auth.token()
        if self.calendar is not None:
            self.calendar.check()
            trace(f"{self.name}: Google Calendar answers")
        if self.mail is not None:
            self.mail.check()
            trace(f"{self.name}: Gmail IMAP accepts the token")
        return True


class GoogleIntegration(AccountHealth, Integration, MailSource, CalendarSource):
    """Gmail and Google Calendar, one Google sign-in.

    Reads, searches and files the Gmail of every Google account listed under google.accounts,
    sends from the one the user means, and shows and adds to their Google calendars; an
    account signs in once in a browser and pastes the address it lands on back into the chat.

    A MailSource and CalendarSource building one GoogleAccount per `google.accounts[]` entry
    (`name`, `address`, `mail`, `calendar`, `default_calendar`, `shown_calendars`) over one
    OAuth Desktop-app client (`google.oauth.client_id` / `client_secret`). Mail is the core
    `ImapMailAccount` with the gmail preset and an XOAUTH2 login fed by `GoogleAuth.token`;
    the calendar is `GoogleCalendarAccount` over Calendar v3. The accounts given to the
    composites are those with the flag AND a refresh token in `cache/oauth/google-<slug>.json`;
    one without is left out and reported in `accounts_status` and the health error with the
    google_login hint. Config is read at call time; a changed list of accounts needs
    restart_integration. Losing the config deletes the token files."""
    sensitive = True

    def __init__(self):
        Integration.__init__(self)
        self._accounts = self._build_accounts()

    @property
    def name(self) -> str:
        return "Google"

    @property
    def config_keys(self) -> List[str]:
        return [CLIENT_ID_KEY, ACCOUNTS_KEY]

    # --- what the composites, the commands and the tab read ----------------------------------

    def accounts(self) -> list[GoogleAccount]:
        return list(self._accounts)

    def account(self, name: str) -> Optional[GoogleAccount]:
        wanted = name.strip().lower()
        return next((a for a in self._accounts if a.name.lower() == wanted), None)

    def mail_accounts(self) -> list[MailAccount]:
        return [a.mail for a in self._accounts if a.mail is not None and self._usable(a)]

    def calendar_accounts(self) -> list[CalendarAccount]:
        return [a.calendar for a in self._accounts if a.calendar is not None and self._usable(a)]

    @staticmethod
    def _usable(account: GoogleAccount) -> bool:
        if account.signed_in():
            return True
        info(f"Google: account '{account.name}' is {not_signed_in_hint(account.name)}, left out")
        return False

    def login_begin(self, name: str) -> SignIn:
        account = self.account(name)
        if account is None:
            raise ValueError(f"no Google account named '{name}' (configured: {', '.join(a.name for a in self._accounts) or 'none'})")
        return account.auth.begin()

    def login_account_for(self, pasted: str, name: str = "") -> Optional[GoogleAccount]:
        """The account a pasted answer belongs to: the named one, else the one whose pending
        state the address carries, else the only one with a sign-in in progress."""
        if name:
            return self.account(name)
        _, state, _ = parse_pasted(pasted)
        by_state = [a for a in self._accounts if a.auth.matches(state)]
        if by_state:
            return by_state[0]
        pending = [a for a in self._accounts if a.auth.pending()]
        if len(pending) == 1:
            return pending[0]
        trace(f"Google: {len(pending)} sign-ins in progress, the pasted answer names none of them")
        return None

    def login_finish(self, pasted: str, name: str = "") -> str:
        """"" when the pasted answer signed an account in, else why not."""
        account = self.login_account_for(pasted, name)
        if account is None:
            if name:
                return f"no Google account named '{name}'"
            return 'no sign-in is in progress that this address belongs to — run google_login("<account>") and paste its own address'
        return account.auth.finish(pasted)

    def sign_out(self, name: str) -> None:
        account = self.account(name)
        if account is not None:
            account.auth.sign_out()

    def get_auth_status(self) -> dict:
        """The console tab's summary of the accounts' sign-in state."""
        if not self._accounts:
            return {"status": "error", "message": "no account under google.accounts"}
        client_id, client_secret = self._client_settings()
        if not client_id or not client_secret:
            return {"status": "error", "message": "google.oauth.client_id and client_secret are needed"}
        missing = [a.name for a in self._accounts if not a.signed_in() or a.auth.needs_sign_in_again()]
        if missing:
            return {"status": "needs_auth", "message": f"Sign-in needed: {', '.join(missing)} — Ctrl-L starts it, Ctrl-E takes the pasted address"}
        return {"status": "authenticated", "message": f"{len(self._accounts)} account(s) signed in"}

    # --- health ------------------------------------------------------------------------------

    def check_health(self) -> bool:
        if not self._accounts:
            trace("Google: no account to check")
            return False
        return self.check_accounts(self._accounts)

    def _health_accounts(self) -> list:
        return self._accounts

    def accounts_status(self) -> list[dict]:
        status = super().accounts_status()
        by_name = {a.name: a for a in self._accounts}
        for entry in status:
            account = by_name[entry["name"]]
            if not account.signed_in():
                entry["state"], entry["last_error"] = STATE_ERROR, not_signed_in_hint(entry["name"])
            elif account.auth.needs_sign_in_again():
                entry["state"], entry["last_error"] = STATE_ERROR, sign_in_again_hint(entry["name"])
        return status

    def on_not_configured(self):
        delete_all_token_files()

    # --- building ----------------------------------------------------------------------------

    @staticmethod
    def _client_settings() -> tuple[str, str]:
        config = ConfigManager.get_instance()
        return str(config.get(CLIENT_ID_KEY, "") or ""), str(config.get(CLIENT_SECRET_KEY, "") or "")

    @staticmethod
    def _entry(index: int) -> dict:
        """The `google.accounts[index]` section as config.yaml holds it now, {} when gone."""
        entries = ConfigManager.get_instance().get(ACCOUNTS_KEY, [])
        if isinstance(entries, list) and index < len(entries) and isinstance(entries[index], dict):
            return entries[index]
        return {}

    @staticmethod
    def _flag(entry: dict, key: str) -> bool:
        value = entry.get(key, True)
        return str(value).strip().lower() not in ("false", "no", "0", "off") if value is not None else True

    @classmethod
    def _build_accounts(cls) -> list[GoogleAccount]:
        config = ConfigManager.get_instance()
        entries = named_entries(config.get(ACCOUNTS_KEY, []), SECTION)
        prefixes = id_prefixes([entry["name"] for _, entry in entries])
        accounts = []
        for (index, entry), prefix in zip(entries, prefixes):
            name = str(entry["name"]).strip()
            mail, calendar = cls._flag(entry, "mail"), cls._flag(entry, "calendar")
            if not mail and not calendar:
                info(f"Google: account '{name}' has neither mail nor calendar on, skipped")
                continue
            address = lambda i=index: str(cls._entry(i).get("address") or "")
            scopes = lambda m=mail, c=calendar: [s for s, on in ((SCOPE_MAIL, m), (SCOPE_CALENDAR, c)) if on]
            auth = GoogleAuth(name, cls._client_settings, scopes, address)
            mail_account = ImapMailAccount(
                name, prefix, lambda a=address: (a(), ""), preset=MAIL_PRESET,
                auth_help=sign_in_again_hint(name), oauth_token=auth.token) if mail else None
            calendar_account = GoogleCalendarAccount(
                name, auth,
                shown_calendars=lambda i=index: split_names(cls._entry(i).get("shown_calendars", [])),
                default_calendar=lambda i=index: str(cls._entry(i).get("default_calendar") or DEFAULT_CALENDAR),
                timezone=lambda: ConfigManager.get_instance().get_home_timezone(),
                address=address) if calendar else None
            account = GoogleAccount(name, address, auth, mail_account, calendar_account)
            trace(f"Google: account '{name}' ({prefix}) mail={mail} calendar={calendar} "
                  f"{'signed in' if account.signed_in() else 'not signed in'}")
            accounts.append(account)
        info(f"Google: {len(accounts)} account(s): {', '.join(a.name for a in accounts) or 'none'}")
        return accounts
