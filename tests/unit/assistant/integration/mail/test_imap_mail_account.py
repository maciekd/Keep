"""ImapMailAccount: what reaches the IMAP and SMTP servers — the folders selected, the fetch
parameters, the Sent copy, and what the presets decide (hosts, Gmail's folded search)."""
import smtplib
from datetime import date, datetime, timedelta, timezone
from unittest.mock import patch, Mock, MagicMock

import pytest

from assistant.integration.mail.ImapMailAccount import ImapMailAccount, gmail_search_criteria, SMTP_TIMEOUT

MAILBOX = 'assistant.integration.mail.ImapMailAccount.MailBox'
SMTP = 'assistant.integration.mail.ImapMailAccount.smtplib.SMTP'


def _message(uid="123", from_="sender@example.com", to=("m@mlab.test",), subject="Test Subject",
             date=None, text="Hello body", html="", flags=("\\Seen",)):
    msg = Mock()
    msg.uid = uid
    msg.from_ = from_
    msg.to = to
    msg.subject = subject
    msg.date = date or datetime(2024, 3, 15, 10, 30, tzinfo=timezone.utc)
    msg.text = text
    msg.html = html
    msg.attachments = []
    msg.flags = flags
    return msg


@pytest.fixture
def gmail():
    return ImapMailAccount("Gmail", "GM", lambda: ("marcus@example.com", "app_password_123"), preset="gmail")


@pytest.fixture
def icloud():
    return ImapMailAccount("iCloud Email", "IC", lambda: ("m@mlab.test", "icloud_app_pw"), preset="icloud")


@pytest.fixture
def mock_mailbox():
    with patch(MAILBOX) as MockMB:
        mailbox_instance = MagicMock()
        MockMB.return_value.login.return_value.__enter__ = Mock(return_value=mailbox_instance)
        MockMB.return_value.login.return_value.__exit__ = Mock(return_value=False)
        yield MockMB, mailbox_instance


@pytest.fixture
def mock_smtp():
    with patch(SMTP) as MockSMTP:
        server = MagicMock()
        MockSMTP.return_value.__enter__.return_value = server
        MockSMTP.return_value.__exit__.return_value = False
        yield MockSMTP, server


@pytest.mark.unit
class TestPresets:
    def test_a_preset_fills_hosts_folders_and_the_sent_copy_rule(self, gmail, icloud):
        assert (gmail.imap_host, gmail.default_folder, gmail.sent_folder) == ("imap.gmail.com", "[Gmail]/All Mail", None)
        assert (icloud.imap_host, icloud.default_folder, icloud.sent_folder) == ("imap.mail.me.com", "INBOX", "Sent Messages")
        assert gmail.preset == "gmail" and icloud.preset == "icloud"

    def test_an_explicit_host_or_port_overrides_the_preset(self):
        account = ImapMailAccount("Gmail", "GM", lambda: ("a", "b"), preset="gmail",
                                  smtp_host="smtp.mail.example", smtp_port="2525")
        assert (account.smtp_host, account.smtp_port, account.imap_host) == ("smtp.mail.example", 2525, "imap.gmail.com")
        assert isinstance(account.smtp_port, int)   # config.yaml may hold the port as a string; smtplib wants a number

    def test_without_a_preset_both_hosts_are_required(self):
        account = ImapMailAccount("Work", "WO", lambda: ("a", "b"), imap_host="imap.company.example",
                                  smtp_host="smtp.company.example")
        assert (account.default_folder, account.sent_folder, account.smtp_port) == ("INBOX", None, 587)
        with pytest.raises(ValueError):
            ImapMailAccount("Work", "WO", lambda: ("a", "b"), imap_host="imap.company.example")

    def test_the_address_is_the_credentials_first_half_or_empty_when_unset(self):
        assert ImapMailAccount("A", "AA", lambda: ("m@mlab.test", "pw"), preset="icloud").address == "m@mlab.test"

        def missing():
            raise KeyError("email.address")
        assert ImapMailAccount("A", "AA", missing, preset="icloud").address == ""


@pytest.mark.unit
class TestGmailSearchCriteria:
    """Gmail rejects raw 8-bit bytes on the SEARCH command line, so the criteria must always
    be pure ASCII — folded, not dropped, since Gmail's index matches folded terms."""

    def test_diacritics_folded_to_ascii(self):
        key = str(gmail_search_criteria("café résumé"))
        assert 'cafe resume' in key and key.isascii()

    def test_non_decomposing_letters_folded(self):
        key = str(gmail_search_criteria("Ørsted Straße Ægir"))
        assert 'Orsted Strasse AEgir' in key and key.isascii()

    def test_quotes_and_backslashes_are_escaped(self):
        assert r'\"hello\"' in str(gmail_search_criteria('say "hello"'))
        assert r'path\\to\\file' in str(gmail_search_criteria(r'path\to\file'))

    def test_unfoldable_chars_dropped_when_the_query_survives(self):
        key = str(gmail_search_criteria("meeting 🎉"))
        assert 'meeting' in key and '🎉' not in key and key.isascii()

    def test_a_fully_unfoldable_query_is_kept_verbatim(self):
        # Dropping everything would yield an empty X-GM-RAW that matches the whole mailbox;
        # better to send it as-is and let Gmail's BAD reply surface to the LLM.
        assert '日本語' in str(gmail_search_criteria("日本語"))

    def test_no_implicit_date_filter(self):
        # Search must cover the full mailbox history — a silent SINCE window made old
        # emails (e.g. year-old order confirmations) unfindable.
        key = str(gmail_search_criteria("test"))
        assert 'X-GM-RAW' in key and 'SINCE' not in key

    def test_the_gmail_preset_searches_with_it_and_icloud_with_plain_text(self, gmail, icloud, mock_mailbox):
        _, mailbox = mock_mailbox
        mailbox.fetch.return_value = []
        gmail.search("Björkängen")
        icloud.search("Björkängen")
        gmail_criteria, icloud_criteria = (call.kwargs["criteria"] for call in mailbox.fetch.call_args_list)
        assert "X-GM-RAW" in str(gmail_criteria) and "Bjorkangen" in str(gmail_criteria)
        assert "X-GM-RAW" not in str(icloud_criteria) and "Björkängen" in str(icloud_criteria)


@pytest.mark.unit
class TestFetchLatest:
    def test_lists_the_default_folder_with_the_listing_parameters(self, gmail, mock_mailbox):
        MockMB, mailbox = mock_mailbox
        mailbox.fetch.return_value = [_message(uid="1"), _message(uid="2")]

        result = gmail.fetch_latest(None)

        assert "[GM-AA]" in result and "[GM-AB]" in result
        MockMB.assert_called_once_with("imap.gmail.com", timeout=gmail.IMAP_TIMEOUT)
        MockMB.return_value.login.assert_called_once_with("marcus@example.com", "app_password_123",
                                                          initial_folder="[Gmail]/All Mail")
        kwargs = mailbox.fetch.call_args.kwargs
        assert (kwargs["limit"], kwargs["reverse"], kwargs["mark_seen"], kwargs["headers_only"]) == (
            gmail.MAX_RESULTS, True, False, True)

    def test_lists_a_named_folder_and_remembers_it(self, gmail, mock_mailbox):
        MockMB, mailbox = mock_mailbox
        mailbox.fetch.return_value = []

        gmail.fetch_latest("@SaneLater")

        assert MockMB.return_value.login.call_args.kwargs["initial_folder"] == "@SaneLater"
        assert gmail._cache_folder == "@SaneLater"

    def test_unread_only_adds_the_unseen_criteria_and_nothing_else_does(self, gmail, mock_mailbox):
        _, mailbox = mock_mailbox
        mailbox.fetch.return_value = []

        gmail.fetch_latest("@SaneLater", unread_only=True)
        gmail.fetch_latest("@SaneLater")

        unread, _, plain, _ = mailbox.fetch.call_args_list
        assert "UNSEEN" in str(unread.kwargs["criteria"])
        assert "UNSEEN" not in str(plain.kwargs["criteria"])

    def test_the_listing_is_bounded_to_the_last_days_and_a_full_window_is_not_listed_again(self, gmail, mock_mailbox):
        """Unbounded, `reverse=True` searches the whole folder: on "[Gmail]/All Mail" two listings
        hung 5.7 minutes (2026-11-04). The newest MAX_RESULTS of an active folder are within the window."""
        _, mailbox = mock_mailbox
        mailbox.fetch.return_value = [_message(uid=str(n)) for n in range(gmail.MAX_RESULTS)]

        gmail.fetch_latest(None)

        call, = mailbox.fetch.call_args_list
        criteria = str(call.kwargs["criteria"])
        assert "SINCE" in criteria
        since = (date.today() - timedelta(days=gmail.LISTING_DAYS)).strftime("%-d-%b-%Y")
        assert since in criteria

    def test_a_folder_short_of_the_window_is_listed_whole(self, gmail, mock_mailbox):
        _, mailbox = mock_mailbox
        mailbox.fetch.side_effect = [[_message(uid="1")], [_message(uid="1"), _message(uid="2")]]

        result = gmail.fetch_latest(None, unread_only=True)

        bounded, whole = mailbox.fetch.call_args_list
        assert "SINCE" in str(bounded.kwargs["criteria"])
        assert "SINCE" not in str(whole.kwargs["criteria"]) and "UNSEEN" in str(whole.kwargs["criteria"])
        assert "[GM-AA]" in result and "[GM-AB]" in result

    def test_a_missing_folder_lists_as_no_emails(self, gmail, mock_mailbox):
        from imap_tools.errors import MailboxFolderSelectError
        MockMB, _ = mock_mailbox
        MockMB.return_value.login.side_effect = MailboxFolderSelectError("cmd", "no such folder")

        assert gmail.fetch_latest("@Nonexistent") == "No emails found."

    def test_a_connection_failure_is_an_error_text(self, gmail, mock_mailbox):
        MockMB, _ = mock_mailbox
        MockMB.return_value.login.side_effect = Exception("Connection refused")

        result = gmail.fetch_latest(None)

        assert "Error" in result and "Connection refused" in result


@pytest.mark.unit
class TestReadAndMove:
    def test_reads_the_uid_behind_the_short_id_from_the_listed_folder(self, gmail, mock_mailbox):
        MockMB, mailbox = mock_mailbox
        gmail._format_email_list([_message(uid="100"), _message(uid="200")], "@SaneLater")
        mailbox.fetch.return_value = []

        gmail.read("GM-AB")

        assert "200" in str(mailbox.fetch.call_args.kwargs["criteria"])
        assert MockMB.return_value.login.call_args.kwargs["initial_folder"] == "@SaneLater"

    def test_without_a_listing_the_default_folder_is_read(self, gmail, mock_mailbox):
        MockMB, mailbox = mock_mailbox
        mailbox.fetch.return_value = []

        gmail.read("99")

        assert "99" in str(mailbox.fetch.call_args.kwargs["criteria"])
        assert MockMB.return_value.login.call_args.kwargs["initial_folder"] == "[Gmail]/All Mail"

    def test_moves_from_the_listed_folder(self, gmail, mock_mailbox):
        MockMB, mailbox = mock_mailbox
        gmail._format_email_list([_message(uid="500")], "@SaneLater")

        result = gmail.move("AA", "@SaneBlackHole")

        assert MockMB.return_value.login.call_args.kwargs["initial_folder"] == "@SaneLater"
        mailbox.move.assert_called_once_with("500", "@SaneBlackHole")
        assert "@SaneBlackHole" in result

    def test_a_folder_the_server_lacks_is_a_folder_error(self, gmail, mock_mailbox):
        from imap_tools.errors import MailboxFolderSelectError
        _, mailbox = mock_mailbox
        mailbox.move.side_effect = MailboxFolderSelectError("cmd", "no such folder")

        assert gmail.move("99", "Nowhere").startswith("Error moving email: folder not found")

    def test_lists_every_folder_by_name(self, gmail, mock_mailbox):
        _, mailbox = mock_mailbox
        mailbox.folder.list.return_value = [Mock(name="f1"), Mock(name="f2")]
        mailbox.folder.list.return_value[0].name = "@SaneLater"
        mailbox.folder.list.return_value[1].name = "Archive"

        assert gmail.list_folders() == "@SaneLater\nArchive"


@pytest.mark.unit
class TestSend:
    def test_sends_over_starttls_with_the_credentials(self, icloud, mock_smtp, mock_mailbox):
        import email as email_pkg
        MockSMTP, server = mock_smtp

        assert icloud.send("alice@example.com", "Hi", "Hello there") == ""

        MockSMTP.assert_called_once_with("smtp.mail.me.com", 587, timeout=SMTP_TIMEOUT)
        server.starttls.assert_called_once()
        server.login.assert_called_once_with("m@mlab.test", "icloud_app_pw")
        from_addr, to_addrs, payload = server.sendmail.call_args[0]
        assert (from_addr, to_addrs) == ("m@mlab.test", ["alice@example.com"])
        parsed = email_pkg.message_from_string(payload)
        assert parsed["From"] == "m@mlab.test" and parsed["Subject"] == "Hi"

    def test_the_configured_smtp_server_is_a_contract_with_email_smtp_host_and_email_smtp_port(self, mock_smtp):
        account = ImapMailAccount("Gmail", "GM", lambda: ("marcus@example.com", "pw"), preset="gmail",
                                  smtp_host="smtp.mail.example", smtp_port="2525")

        assert account.send("alice@example.com", "Hi", "Hello") == ""

        assert mock_smtp[0].call_args.args[:2] == ("smtp.mail.example", 2525)

    def test_auth_failure_names_the_presets_help(self, icloud, mock_smtp):
        _, server = mock_smtp
        server.login.side_effect = smtplib.SMTPAuthenticationError(535, b"bad password")

        result = icloud.send("alice@example.com", "Hi", "body")

        assert "auth failed" in result.lower() and "appleid.apple.com" in result
        server.sendmail.assert_not_called()

    def test_a_refused_recipient_and_a_network_error_are_reported(self, icloud, mock_smtp):
        MockSMTP, server = mock_smtp
        server.sendmail.side_effect = smtplib.SMTPRecipientsRefused({"alice@example.com": (550, b"no such user")})
        refused = icloud.send("alice@example.com", "Hi", "body")
        MockSMTP.side_effect = TimeoutError("connect timed out")
        network = icloud.send("alice@example.com", "Hi", "body")

        assert "recipient refused" in refused and "alice@example.com" in refused
        assert "network error" in network.lower()

    def test_missing_credentials_never_reach_smtp(self, mock_smtp):
        account = ImapMailAccount("Gmail", "GM", lambda: ("marcus@example.com", ""), preset="gmail")
        assert "credentials not configured" in account.send("alice@example.com", "Hi", "body")
        mock_smtp[0].assert_not_called()


@pytest.mark.unit
class TestSentCopy:
    """iCloud's SMTP server does not file what it sends, so the sent mail is appended to the
    Sent folder over IMAP after the SMTP handoff; Gmail files its own, so no copy is made."""

    def test_icloud_files_the_copy_in_its_sent_folder(self, icloud, mock_smtp, mock_mailbox):
        import email as email_pkg
        MockMailBox, mailbox = mock_mailbox

        assert icloud.send("alice@example.com", "Hi", "Hello there") == ""

        MockMailBox.assert_called_once_with("imap.mail.me.com", timeout=icloud.IMAP_TIMEOUT)
        MockMailBox.return_value.login.assert_called_once_with("m@mlab.test", "icloud_app_pw")
        raw, folder = mailbox.append.call_args[0]
        assert folder == "Sent Messages"
        assert "SEEN" in str(mailbox.append.call_args.kwargs["flag_set"]).upper()
        filed = email_pkg.message_from_bytes(raw)
        sent = email_pkg.message_from_string(mock_smtp[1].sendmail.call_args[0][2])
        assert filed["Message-ID"] == sent["Message-ID"] and filed["Date"] == sent["Date"]

    def test_the_copy_is_filed_only_after_a_successful_send(self, icloud, mock_smtp, mock_mailbox):
        mock_smtp[1].sendmail.side_effect = smtplib.SMTPRecipientsRefused({"alice@example.com": (550, b"no such user")})
        icloud.send("alice@example.com", "Hi", "body")
        mock_mailbox[1].append.assert_not_called()

    def test_a_failed_copy_does_not_fail_the_send(self, icloud, mock_smtp, mock_mailbox):
        mock_mailbox[1].append.side_effect = OSError("imap down")
        assert icloud.send("alice@example.com", "Hi", "body") == ""

    def test_gmail_leaves_filing_to_the_server(self, gmail, mock_smtp, mock_mailbox):
        assert gmail.send("alice@example.com", "Hi", "body") == ""
        mock_smtp[0].assert_called_once_with("smtp.gmail.com", 587, timeout=SMTP_TIMEOUT)
        mock_mailbox[0].assert_not_called()


@pytest.mark.unit
class TestCheck:
    def test_the_health_check_signs_in_to_the_default_folder(self, icloud, mock_mailbox):
        MockMB, _ = mock_mailbox
        assert icloud.check() is True
        assert MockMB.return_value.login.call_args.kwargs["initial_folder"] == "INBOX"

    def test_a_refused_login_raises_for_the_monitor(self, icloud, mock_mailbox):
        mock_mailbox[0].return_value.login.side_effect = Exception("AUTHENTICATIONFAILED")
        with pytest.raises(Exception, match="AUTHENTICATIONFAILED"):
            icloud.check()


@pytest.mark.unit
class TestXoauth2:
    """An account given an `oauth_token` supplier signs in with XOAUTH2 instead of a password:
    IMAP through imap_tools' own mechanism, SMTP through `AUTH XOAUTH2` with the SASL string
    Gmail documents — `user=<address>^Aauth=Bearer <token>^A^A`, base64 on the wire."""

    def _account(self, token=lambda: "ya29.token"):
        return ImapMailAccount("Personal", "PE", lambda: ("marcus@example.com", ""), preset="gmail",
                               auth_help='sign in again with google_login("Personal")', oauth_token=token)

    def test_imap_signs_in_with_the_address_and_the_token_a_contract_with_gmail_imap_xoauth2(self, mock_mailbox):
        MockMB, _ = mock_mailbox
        account = self._account()

        assert account.check() is True

        MockMB.return_value.xoauth2.assert_called_once_with("marcus@example.com", "ya29.token", initial_folder="[Gmail]/All Mail")
        MockMB.return_value.login.assert_not_called()

    def test_smtp_sends_the_sasl_string_gmail_expects_a_contract_with_gmail_smtp_xoauth2(self, mock_smtp, mock_mailbox):
        import base64
        _, server = mock_smtp

        assert self._account().send("alice@example.com", "Hi", "Hello") == ""

        server.starttls.assert_called_once()
        server.login.assert_not_called()
        mechanism, authobject = server.auth.call_args.args
        assert mechanism == "XOAUTH2"
        assert authobject() == "user=marcus@example.com\x01auth=Bearer ya29.token\x01\x01"
        assert base64.b64encode(authobject().encode()) == b"dXNlcj1tYXJjdXNAZXhhbXBsZS5jb20BYXV0aD1CZWFyZXIgeWEyOS50b2tlbgEB"
        assert server.ehlo_or_helo_if_needed.called and server.sendmail.call_args.args[0] == "marcus@example.com"

    def test_a_supplier_without_a_token_fails_the_send_before_smtp_and_a_refused_token_names_the_help(self, mock_smtp):
        def no_token():
            raise RuntimeError('not signed in — google_login("Personal")')

        outcome = self._account(no_token).send("alice@example.com", "Hi", "Hello")

        assert 'google_login("Personal")' in outcome and not mock_smtp[0].called
        mock_smtp[1].auth.side_effect = smtplib.SMTPAuthenticationError(535, b"Username and Password not accepted")
        refused = self._account().send("alice@example.com", "Hi", "Hello")
        assert "auth failed" in refused.lower() and 'google_login("Personal")' in refused

    def test_a_supplier_without_a_token_makes_a_listing_an_error_text(self, mock_mailbox):
        def no_token():
            raise RuntimeError('not signed in — google_login("Personal")')

        assert 'google_login("Personal")' in self._account(no_token).fetch_latest()
        mock_mailbox[0].return_value.xoauth2.assert_not_called()
