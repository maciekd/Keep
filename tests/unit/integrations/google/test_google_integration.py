"""The Google folder: one account per `google.accounts` entry, the mail (the core IMAP account
with the sign-in's token) and calendar accounts given to the composites only when the entry
asks for them AND a refresh token exists, and the account without one named in the status and
the health error."""
import json
from unittest.mock import MagicMock, patch

import pytest

from assistant.integration.calendar.CalendarSource import CalendarSource
from assistant.integration.mail.ImapMailAccount import ImapMailAccount
from assistant.integration.mail.MailSource import MailSource
from integrations.google import GoogleAuth as auth_module
from integrations.google.GoogleAuth import REDIRECT_URI, token_path
from integrations.google.GoogleIntegration import GoogleIntegration

from .google_fakes import FakeGoogle, calendar, calendar_list, token_file

CONFIG = 'integrations.google.GoogleIntegration.ConfigManager.get_instance'
REQUEST = "integrations.google.GoogleCalendarAccount.requests.request"
MAILBOX = "assistant.integration.mail.ImapMailAccount.MailBox"
ENTRIES = [{"name": "Personal", "address": "marcus@example.com", "mail": True, "calendar": True},
           {"name": "Family", "address": "family@example.com", "mail": True, "calendar": False},
           {"name": "Club", "address": "club@example.com", "mail": "false", "calendar": True,
            "shown_calendars": ["Matches"], "default_calendar": "Matches"}]


def _config(accounts, client_id="cid", client_secret="csecret"):
    values = {"google.accounts": accounts, "google.oauth.client_id": client_id, "google.oauth.client_secret": client_secret}
    config = MagicMock()
    config.get.side_effect = lambda key, *default: values.get(key, default[0] if default else None)
    config.get_home_timezone.return_value = "Europe/Stockholm"
    return patch(CONFIG, return_value=config)


def _signed_in(token_dir, name, address, **more):
    token_dir.mkdir(parents=True, exist_ok=True)
    token_path(name).write_text(json.dumps(token_file(address, **more)))


@pytest.fixture(autouse=True)
def fresh(tmp_path):
    GoogleIntegration.reset_instance()
    with patch.object(auth_module, "TOKEN_DIR", tmp_path / "oauth"):
        yield tmp_path / "oauth"
    GoogleIntegration.reset_instance()


@pytest.mark.unit
class TestAccounts:
    def test_only_a_flagged_account_with_a_token_reaches_the_composites_a_contract_with_google_accounts(self, fresh):
        _signed_in(fresh, "Personal", "marcus@example.com")
        _signed_in(fresh, "Club", "club@example.com")
        with _config(ENTRIES):
            integration = GoogleIntegration()
            assert isinstance(integration, MailSource) and isinstance(integration, CalendarSource)
            assert integration.config_keys == ["google.oauth.client_id", "google.accounts"]
            assert [a.name for a in integration.accounts()] == ["Personal", "Family", "Club"]
            assert [a.name for a in integration.mail_accounts()] == ["Personal"]        # Family has no token, Club no mail
            assert [a.name for a in integration.calendar_accounts()] == ["Personal", "Club"]
            personal, = integration.mail_accounts()
            assert isinstance(personal, ImapMailAccount) and personal.preset == "gmail"
            assert (personal.id_prefix, personal.address, personal.imap_host) == ("PE", "marcus@example.com", "imap.gmail.com")
            assert 'google_login("Personal")' in personal.AUTH_HELP
            own, club = integration.calendar_accounts()
            assert (own.shown_calendars, own.default_calendar) == ([], "primary")
            assert (club.shown_calendars, club.default_calendar) == (["Matches"], "Matches")

    def test_an_account_with_neither_mail_nor_calendar_is_kept_for_its_drive_and_a_repeated_name_skipped(self, fresh):
        entries = [{"name": "Personal", "mail": False, "calendar": False}, {"name": "Family"}, {"name": "family", "mail": True}]
        with _config(entries):
            personal, family = GoogleIntegration().accounts()
            assert (personal.name, family.name) == ("Personal", "Family")
            assert personal.mail is None and personal.calendar is None and personal.drive is not None

    def test_the_drives_the_read_commands_get_are_the_signed_in_accounts_in_config_order(self, fresh):
        """A file id is tried account by account, so an account without a token would meet the
        sign-in verdict on every read: files() holds the Drives of those signed in, in the order
        of google.accounts — not the order they signed in."""
        _signed_in(fresh, "Club", "club@example.com")
        _signed_in(fresh, "Personal", "marcus@example.com")
        with _config(ENTRIES):
            assert GoogleIntegration().files().accounts() == ["Personal", "Club"]

    def test_the_mail_account_logs_in_with_the_sign_ins_token_and_the_check_reads_one_calendar_entry(self, fresh):
        _signed_in(fresh, "Personal", "marcus@example.com", access_token="ya29.live", expires_in=3000)
        google = FakeGoogle()
        google.route("GET", "/users/me/calendarList", calendar_list(calendar("marcus@example.com", "marcus@example.com", primary=True)))
        with _config(ENTRIES[:1]), patch(REQUEST, google), patch(MAILBOX) as mailbox:
            personal, = GoogleIntegration().accounts()
            assert personal.check() is True
        mailbox.return_value.xoauth2.assert_called_once_with("marcus@example.com", "ya29.live", initial_folder="[Gmail]/All Mail")
        mailbox.return_value.login.assert_not_called()
        assert google.sent("GET", "/users/me/calendarList")[0].params == {"maxResults": 1}
        assert google.calls[0].headers["Authorization"] == "Bearer ya29.live"

    def test_an_account_is_found_by_name_whatever_the_case(self, fresh):
        with _config(ENTRIES):
            integration = GoogleIntegration()
            assert integration.account("personal").name == "Personal" and integration.account("Nobody") is None
            with pytest.raises(ValueError) as raised:
                integration.login_begin("Nobody")
            assert "Personal" in str(raised.value)


@pytest.mark.unit
class TestSignIn:
    def test_the_pasted_address_finds_its_account_by_the_state_or_the_only_pending_flow(self, fresh):
        with _config(ENTRIES):
            integration = GoogleIntegration()
            assert integration.login_account_for("4/0Acode") is None                       # nothing pending
            personal = integration.login_begin("Personal")
            assert integration.login_account_for("4/0Acode").name == "Personal"             # the only pending one
            family = integration.login_begin("Family")
            assert integration.login_account_for("4/0Acode") is None                        # two pending: ambiguous
            assert integration.login_account_for(f"{REDIRECT_URI}?state={family.state}&code=x").name == "Family"
            assert integration.login_account_for(f"{REDIRECT_URI}?state={personal.state}&code=x").name == "Personal"
            assert integration.login_account_for("x", "club").name == "Club"
            assert "google_login" in integration.login_finish("4/0Acode")
            assert "Nobody" in integration.login_finish("4/0Acode", "Nobody")

    def test_sign_out_forgets_the_account(self, fresh):
        _signed_in(fresh, "Personal", "marcus@example.com")
        with _config(ENTRIES[:1]):
            integration = GoogleIntegration()
            assert integration.mail_accounts()
            integration.sign_out("Personal")
            assert integration.mail_accounts() == [] and not token_path("Personal").exists()


@pytest.mark.unit
class TestStatus:
    def test_the_account_without_a_token_is_reported_with_the_sign_in_hint_before_any_check(self, fresh):
        _signed_in(fresh, "Personal", "marcus@example.com")
        with _config(ENTRIES):
            integration = GoogleIntegration()
            status = {a["name"]: a for a in integration.accounts_status()}
            assert status["Personal"]["state"] == "unknown"
            assert status["Family"]["state"] == "error" and status["Family"]["last_error"] == 'not signed in — google_login("Family")'
            assert integration.get_auth_status()["status"] == "needs_auth" and "Family" in integration.get_auth_status()["message"]

    def test_a_refused_refresh_is_reported_as_sign_in_again(self, fresh):
        _signed_in(fresh, "Personal", "marcus@example.com")
        with _config(ENTRIES[:1]):
            integration = GoogleIntegration()
            personal, = integration.accounts()
            personal.auth.refresh_error = "the refresh token is no longer valid"
            status, = integration.accounts_status()
            assert status["state"] == "error" and status["last_error"] == 'sign in again — google_login("Personal")'
            assert integration.get_auth_status()["status"] == "needs_auth"
            assert [a.name for a in integration.mail_accounts()] == ["Personal"]      # still listed: each call says why

    def test_the_health_check_names_the_account_without_a_token_and_passes_the_signed_in_one(self, fresh):
        _signed_in(fresh, "Personal", "marcus@example.com")
        with _config(ENTRIES[:2]):
            integration = GoogleIntegration()
        personal, family = integration.accounts()
        with patch.object(personal.calendar, "check", return_value=True), patch.object(personal.mail, "check", return_value=True):
            with pytest.raises(RuntimeError) as raised:
                integration.check_health()
        assert "Family" in str(raised.value) and "google_login" in str(raised.value) and "Personal:" not in str(raised.value)
        assert {a["name"]: a["state"] for a in integration.accounts_status()} == {"Personal": "ok", "Family": "error"}

    def test_everyone_signed_in_is_healthy_and_authenticated(self, fresh):
        _signed_in(fresh, "Personal", "marcus@example.com")
        with _config(ENTRIES[:1]):
            integration = GoogleIntegration()
            personal, = integration.accounts()
            with patch.object(personal.calendar, "check", return_value=True), patch.object(personal.mail, "check", return_value=True):
                assert integration.check_health() is True
            assert integration.get_auth_status()["status"] == "authenticated"

    def test_without_an_account_or_the_client_nothing_is_healthy(self, fresh):
        with _config([]):
            integration = GoogleIntegration()
            assert integration.check_health() is False and integration.get_auth_status()["status"] == "error"
        GoogleIntegration.reset_instance()
        with _config(ENTRIES[:1], client_secret=""):
            assert "client_secret" in GoogleIntegration().get_auth_status()["message"]

    def test_losing_the_config_deletes_every_token_file(self, fresh):
        _signed_in(fresh, "Personal", "marcus@example.com")
        _signed_in(fresh, "Family", "family@example.com")
        with _config(ENTRIES):
            GoogleIntegration().on_not_configured()
        assert list(fresh.iterdir()) == []
