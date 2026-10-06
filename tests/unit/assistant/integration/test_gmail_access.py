from datetime import datetime, timezone
from unittest.mock import patch, Mock, MagicMock

import pytest

from assistant.integration.integrations.email.GmailIntegration import GmailAccess


def _make_message(uid="123", from_="sender@test.com", to=("me@test.com",),
                  subject="Test Subject", date=None, text="Hello body", html="",
                  flags=("\\Seen",)):
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
    return GmailAccess()


@pytest.fixture
def mock_config():
    with patch('assistant.integration.integrations.email.GmailIntegration.ConfigManager') as MockCM:
        instance = MockCM.return_value
        instance.get.side_effect = lambda key: {
            "email.address": "user@gmail.com",
            "email.password": "app_password_123"
        }[key]
        yield instance


@pytest.fixture
def mock_mailbox():
    with patch('assistant.integration.integrations.email.EmailProvider.MailBox') as MockMB:
        mailbox_instance = MagicMock()
        MockMB.return_value.login.return_value.__enter__ = Mock(return_value=mailbox_instance)
        MockMB.return_value.login.return_value.__exit__ = Mock(return_value=False)
        yield MockMB, mailbox_instance


@pytest.mark.unit
class TestGetCredentials:

    def test_reads_email_config(self, gmail, mock_config):
        address, password = gmail._get_credentials()
        assert address == "user@gmail.com"
        assert password == "app_password_123"
        mock_config.get.assert_any_call("email.address")
        mock_config.get.assert_any_call("email.password")


@pytest.mark.unit
class TestFormatEmailList:

    def test_formats_single_email(self, gmail):
        msg = _make_message(uid="42", from_="alice@test.com", subject="Hello",
                            date=datetime(2024, 3, 15, 10, 30, tzinfo=timezone.utc))
        result = gmail._format_email_list([msg])
        assert "[GM-AA]" in result
        assert "alice@test.com" in result
        assert "Hello" in result
        assert "Mar 15" in result

    def test_date_converted_to_local_time(self, gmail):
        utc_date = datetime(2024, 3, 15, 10, 30, tzinfo=timezone.utc)
        msg = _make_message(date=utc_date)
        result = gmail._format_email_list([msg])
        expected = utc_date.astimezone().strftime("%b %d %H:%M")
        assert expected in result

    def test_empty_list(self, gmail):
        result = gmail._format_email_list([])
        assert result == "No emails found."

    def test_no_subject(self, gmail):
        msg = _make_message(subject=None)
        result = gmail._format_email_list([msg])
        assert "(no subject)" in result

    def test_short_ids_increment(self, gmail):
        messages = [_make_message(uid=str(i)) for i in range(3)]
        result = gmail._format_email_list(messages)
        assert "[GM-AA]" in result
        assert "[GM-AB]" in result
        assert "[GM-AC]" in result

    def test_marks_unread_email(self, gmail):
        msg = _make_message(subject="Newsletter", flags=())
        result = gmail._format_email_list([msg])
        assert "[GM-AA] [UNREAD]" in result

    def test_no_marker_for_read_email(self, gmail):
        msg = _make_message(subject="Newsletter", flags=("\\Seen",))
        result = gmail._format_email_list([msg])
        assert "[UNREAD]" not in result

    def test_handles_missing_flags(self, gmail):
        msg = _make_message(subject="Newsletter", flags=None)
        result = gmail._format_email_list([msg])
        assert "[UNREAD]" in result

    def test_ids_reset_on_each_listing(self, gmail):
        first_msgs = [_make_message(uid="100"), _make_message(uid="200")]
        gmail._format_email_list(first_msgs)
        assert gmail._email_id_cache["AA"] == "100"

        second_msgs = [_make_message(uid="300"), _make_message(uid="400")]
        result = gmail._format_email_list(second_msgs)
        assert "[GM-AA]" in result
        assert "[GM-AB]" in result
        assert gmail._email_id_cache["AA"] == "300"
        assert gmail._email_id_cache["AB"] == "400"
        assert "100" not in gmail._email_id_cache.values()


@pytest.mark.unit
class TestFetchLatest:

    def test_returns_formatted_list(self, gmail, mock_config, mock_mailbox):
        MockMB, mailbox_instance = mock_mailbox
        messages = [_make_message(uid="1"), _make_message(uid="2")]
        mailbox_instance.fetch.return_value = messages

        result = gmail.fetch_latest(None)

        assert "[GM-AA]" in result
        assert "[GM-AB]" in result

    def test_passes_correct_fetch_params(self, gmail, mock_config, mock_mailbox):
        MockMB, mailbox_instance = mock_mailbox
        mailbox_instance.fetch.return_value = []

        gmail.fetch_latest(None)

        mailbox_instance.fetch.assert_called_once_with(
            limit=GmailAccess.MAX_RESULTS,
            reverse=True,
            mark_seen=False,
            headers_only=True
        )

    def test_connects_with_credentials(self, gmail, mock_config, mock_mailbox):
        MockMB, mailbox_instance = mock_mailbox
        mailbox_instance.fetch.return_value = []

        gmail.fetch_latest(None)

        MockMB.assert_called_once_with(GmailAccess.IMAP_HOST, timeout=GmailAccess.IMAP_TIMEOUT)
        MockMB.return_value.login.assert_called_once_with(
            "user@gmail.com", "app_password_123",
            initial_folder=GmailAccess.IMAP_FOLDER
        )

    def test_returns_error_on_exception(self, gmail, mock_config, mock_mailbox):
        MockMB, mailbox_instance = mock_mailbox
        MockMB.return_value.login.side_effect = Exception("Connection refused")

        result = gmail.fetch_latest(None)

        assert "Error" in result
        assert "Connection refused" in result


@pytest.mark.unit
class TestFetchLatestFolder:

    def test_lists_custom_folder(self, gmail, mock_config, mock_mailbox):
        MockMB, mailbox_instance = mock_mailbox
        mailbox_instance.fetch.return_value = []

        gmail.fetch_latest("@SaneLater")

        MockMB.return_value.login.assert_called_once_with(
            "user@gmail.com", "app_password_123",
            initial_folder="@SaneLater"
        )

    def test_remembers_folder_for_move(self, gmail, mock_config, mock_mailbox):
        MockMB, mailbox_instance = mock_mailbox
        mailbox_instance.fetch.return_value = []

        gmail.fetch_latest("@SaneLater")

        assert gmail._cache_folder == "@SaneLater"

    def test_missing_folder_returns_no_emails(self, gmail, mock_config, mock_mailbox):
        from imap_tools.errors import MailboxFolderSelectError
        MockMB, mailbox_instance = mock_mailbox
        MockMB.return_value.login.side_effect = MailboxFolderSelectError("cmd", "no such folder")

        result = gmail.fetch_latest("@Nonexistent")

        assert result == "No emails found."

    def test_unread_only_adds_unseen_criteria(self, gmail, mock_config, mock_mailbox):
        MockMB, mailbox_instance = mock_mailbox
        mailbox_instance.fetch.return_value = []

        gmail.fetch_latest("@SaneLater", unread_only=True)

        criteria = mailbox_instance.fetch.call_args[1]["criteria"]
        assert "UNSEEN" in str(criteria)

    def test_default_has_no_criteria(self, gmail, mock_config, mock_mailbox):
        MockMB, mailbox_instance = mock_mailbox
        mailbox_instance.fetch.return_value = []

        gmail.fetch_latest("@SaneLater")

        assert "criteria" not in mailbox_instance.fetch.call_args[1]


@pytest.mark.unit
class TestMove:

    def test_moves_resolved_short_id_from_cache_folder(self, gmail, mock_config, mock_mailbox):
        MockMB, mailbox_instance = mock_mailbox
        mailbox_instance.fetch.return_value = []
        gmail._format_email_list([_make_message(uid="500")])
        gmail._cache_folder = "@SaneLater"

        result = gmail.move("AA", "@SaneBlackHole")

        MockMB.return_value.login.assert_called_once_with(
            "user@gmail.com", "app_password_123",
            initial_folder="@SaneLater"
        )
        mailbox_instance.move.assert_called_once_with("500", "@SaneBlackHole")
        assert "@SaneBlackHole" in result

    def test_strips_quotes_from_dest(self, gmail, mock_config, mock_mailbox):
        MockMB, mailbox_instance = mock_mailbox
        gmail._cache_folder = "@SaneLater"

        gmail.move("99", '"@SaneBlackHole"')

        mailbox_instance.move.assert_called_once_with("99", "@SaneBlackHole")

    def test_falls_back_to_default_folder_when_no_cache(self, gmail, mock_config, mock_mailbox):
        MockMB, mailbox_instance = mock_mailbox

        gmail.move("99", "Archive")

        MockMB.return_value.login.assert_called_once_with(
            "user@gmail.com", "app_password_123",
            initial_folder=GmailAccess.IMAP_FOLDER
        )

    def test_returns_error_on_exception(self, gmail, mock_config, mock_mailbox):
        MockMB, mailbox_instance = mock_mailbox
        MockMB.return_value.login.side_effect = Exception("Connection refused")

        result = gmail.move("99", "Archive")

        assert "Error" in result

    def test_echoes_cached_subject_in_result(self, gmail, mock_config, mock_mailbox):
        # The subject is the model's only human-meaningful handle on which message it
        # actually moved — without it a misattribution can't be caught on the next pass.
        gmail._format_email_list([_make_message(uid="500", subject="Your booking is confirmed")])
        gmail._cache_folder = "@SaneLater"

        result = gmail.move("AA", "@SaneBlackHole")

        assert result == 'Moved email AA ("Your booking is confirmed") to @SaneBlackHole.'

    def test_truncates_long_subject_to_50_chars_with_ellipsis(self, gmail, mock_config, mock_mailbox):
        long_subject = "A" * 80
        gmail._format_email_list([_make_message(uid="500", subject=long_subject)])
        gmail._cache_folder = "@SaneLater"

        result = gmail.move("AA", "@SaneBlackHole")

        assert f'("{"A" * 50}…")' in result
        assert "A" * 51 not in result  # never the full untruncated subject

    def test_falls_back_to_id_only_on_cache_miss(self, gmail, mock_config, mock_mailbox):
        # Nothing listed → subject cache empty → no subject to echo, but the move still reports.
        result = gmail.move("AA", "@SaneBlackHole")

        assert result == "Moved email AA to @SaneBlackHole."

    def test_falls_back_to_id_only_when_subject_is_placeholder(self, gmail, mock_config, mock_mailbox):
        # A subjectless email is cached as "(no subject)" — echoing that adds no signal.
        gmail._format_email_list([_make_message(uid="500", subject=None)])
        gmail._cache_folder = "@SaneLater"

        result = gmail.move("AA", "@SaneBlackHole")

        assert result == "Moved email AA to @SaneBlackHole."


@pytest.mark.unit
class TestListFolders:

    def _folder(self, name):
        f = Mock()
        f.name = name
        return f

    def test_returns_folder_names(self, gmail, mock_config, mock_mailbox):
        MockMB, mailbox_instance = mock_mailbox
        mailbox_instance.folder.list.return_value = [
            self._folder("@SaneLater"), self._folder("Archive")]

        result = gmail.list_folders()

        assert "@SaneLater" in result
        assert "Archive" in result

    def test_empty(self, gmail, mock_config, mock_mailbox):
        MockMB, mailbox_instance = mock_mailbox
        mailbox_instance.folder.list.return_value = []

        assert gmail.list_folders() == "No folders found."

    def test_returns_error_on_exception(self, gmail, mock_config, mock_mailbox):
        MockMB, mailbox_instance = mock_mailbox
        MockMB.return_value.login.side_effect = Exception("Connection refused")

        result = gmail.list_folders()

        assert "Error" in result


@pytest.mark.unit
class TestSearch:

    def test_returns_formatted_results(self, gmail, mock_config, mock_mailbox):
        MockMB, mailbox_instance = mock_mailbox
        mailbox_instance.fetch.return_value = [_make_message(uid="99", subject="Invoice")]

        result = gmail.search("invoice")

        assert "[GM-" in result
        assert "Invoice" in result

    def test_passes_search_criteria(self, gmail, mock_config, mock_mailbox):
        MockMB, mailbox_instance = mock_mailbox
        mailbox_instance.fetch.return_value = []

        gmail.search("receipt")

        call_args = mailbox_instance.fetch.call_args
        assert call_args[1]['mark_seen'] is False
        assert call_args[1]['reverse'] is True
        assert call_args[1]['limit'] == GmailAccess.MAX_RESULTS

    def test_returns_error_on_exception(self, gmail, mock_config, mock_mailbox):
        MockMB, mailbox_instance = mock_mailbox
        MockMB.return_value.login.side_effect = Exception("Auth failed")

        result = gmail.search("test")

        assert "Error" in result


@pytest.mark.unit
class TestRead:

    def test_returns_full_email(self, gmail, mock_config, mock_mailbox):
        MockMB, mailbox_instance = mock_mailbox
        msg = _make_message(uid="55", from_="bob@test.com", to=("me@test.com",),
                            subject="Important", text="Hello there!")
        mailbox_instance.fetch.return_value = [msg]

        result = gmail.read("55")

        assert "From: bob@test.com" in result
        assert "To: me@test.com" in result
        assert "Subject: Important" in result
        assert "Hello there!" in result

    def test_resolves_short_id_to_correct_imap_uid(self, gmail, mock_config, mock_mailbox):
        MockMB, mailbox_instance = mock_mailbox
        messages = [_make_message(uid="100"), _make_message(uid="200"), _make_message(uid="300")]
        gmail._format_email_list(messages)

        mailbox_instance.fetch.return_value = []
        gmail.read("AB")

        criteria = mailbox_instance.fetch.call_args[1]['criteria']
        assert "200" in str(criteria)

    def test_resolves_prefixed_short_id(self, gmail, mock_config, mock_mailbox):
        MockMB, mailbox_instance = mock_mailbox
        messages = [_make_message(uid="100"), _make_message(uid="200")]
        gmail._format_email_list(messages)

        mailbox_instance.fetch.return_value = []
        gmail.read("GM-AB")

        criteria = mailbox_instance.fetch.call_args[1]['criteria']
        assert "200" in str(criteria)

    def test_unknown_id_is_passed_through_to_imap(self, gmail, mock_config, mock_mailbox):
        MockMB, mailbox_instance = mock_mailbox
        mailbox_instance.fetch.return_value = []

        gmail.read("99999")

        criteria = mailbox_instance.fetch.call_args[1]['criteria']
        assert "99999" in str(criteria)

    def test_email_not_found(self, gmail, mock_config, mock_mailbox):
        MockMB, mailbox_instance = mock_mailbox
        mailbox_instance.fetch.return_value = []

        result = gmail.read("999")

        assert "not found" in result

    def test_reads_from_cache_folder(self, gmail, mock_config, mock_mailbox):
        """UIDs are scoped to the folder they were listed from — reading after
        listing @SaneLater must select @SaneLater, not the default folder."""
        MockMB, mailbox_instance = mock_mailbox
        mailbox_instance.fetch.return_value = []
        gmail._format_email_list([_make_message(uid="500")])
        gmail._cache_folder = "@SaneLater"

        gmail.read("AA")

        MockMB.return_value.login.assert_called_once_with(
            "user@gmail.com", "app_password_123",
            initial_folder="@SaneLater"
        )

    def test_falls_back_to_default_folder_when_no_cache(self, gmail, mock_config, mock_mailbox):
        MockMB, mailbox_instance = mock_mailbox
        mailbox_instance.fetch.return_value = []

        gmail.read("99")

        MockMB.return_value.login.assert_called_once_with(
            "user@gmail.com", "app_password_123",
            initial_folder=GmailAccess.IMAP_FOLDER
        )

    def test_uses_html_when_no_text(self, gmail, mock_config, mock_mailbox):
        MockMB, mailbox_instance = mock_mailbox
        msg = _make_message(text="", html="<p>HTML content</p>")
        mailbox_instance.fetch.return_value = [msg]

        result = gmail.read("123")

        assert "HTML content" in result

    def test_no_body_fallback(self, gmail, mock_config, mock_mailbox):
        MockMB, mailbox_instance = mock_mailbox
        msg = _make_message(text="", html="")
        mailbox_instance.fetch.return_value = [msg]

        result = gmail.read("123")

        assert "(no body)" in result

    def test_truncates_long_body(self, gmail, mock_config, mock_mailbox):
        MockMB, mailbox_instance = mock_mailbox
        long_body = "x" * 10000
        msg = _make_message(text=long_body)
        mailbox_instance.fetch.return_value = [msg]

        result = gmail.read("123")

        assert "... (truncated)" in result
        body_start = result.index("\n\n") + 2
        body = result[body_start:]
        assert len(body) < 10000

    def test_returns_error_on_exception(self, gmail, mock_config, mock_mailbox):
        MockMB, mailbox_instance = mock_mailbox
        MockMB.return_value.login.side_effect = Exception("Timeout")

        result = gmail.read("123")

        assert "Error" in result


@pytest.fixture
def mock_smtp():
    with patch('assistant.integration.integrations.email.EmailProvider.smtplib.SMTP') as MockSMTP:
        server = MagicMock()
        MockSMTP.return_value.__enter__.return_value = server
        MockSMTP.return_value.__exit__.return_value = False
        yield MockSMTP, server


@pytest.mark.unit
class TestGmailSend:
    """Gmail sends through EmailProvider like iCloud, but its SMTP server comes from config: the
    nightly report used to carry its own smtplib block honouring these keys, and still must."""

    @staticmethod
    def _config(values):
        patcher = patch('assistant.integration.integrations.email.GmailIntegration.ConfigManager')
        MockCM = patcher.start()
        MockCM.return_value.get.side_effect = lambda key, default=None: values.get(key, default)
        return patcher

    def test_the_smtp_server_is_a_contract_with_email_smtp_host_and_email_smtp_port(self, gmail, mock_smtp):
        patcher = self._config({"email.address": "marcus@example.com", "email.password": "app-pw",
                                "email.smtp_host": "smtp.mail.example", "email.smtp_port": "2525"})
        try:
            assert gmail.send("alice@example.com", "Hi", "Hello") == ""
        finally:
            patcher.stop()
        host, port = mock_smtp[0].call_args.args[:2]
        assert (host, port) == ("smtp.mail.example", 2525)
        assert isinstance(port, int)   # config.yaml may hold the port as a string; smtplib wants a number

    def test_a_prepared_message_keeps_its_own_headers_and_gets_the_missing_ones(self, gmail, mock_smtp):
        import email as email_pkg
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText
        report = MIMEMultipart("alternative")
        report["Subject"] = "Nightly report"
        report["To"] = "marcus@example.com"
        report["Message-ID"] = "<nightly-2026-10-06@mlab.test>"
        report.attach(MIMEText("<p>All green</p>", "html"))
        patcher = self._config({"email.address": "marcus@example.com", "email.password": "app-pw"})
        try:
            assert gmail.send_message(report, ["marcus@example.com"]) == ""
        finally:
            patcher.stop()
        sender, recipients, payload = mock_smtp[1].sendmail.call_args.args
        sent = email_pkg.message_from_string(payload)
        assert (sender, recipients) == ("marcus@example.com", ["marcus@example.com"])
        assert sent["Message-ID"] == "<nightly-2026-10-06@mlab.test>"   # the caller's id is not replaced
        assert sent["From"] == "marcus@example.com"                       # the account fills in the sender
        assert sent["Date"]                                                # and a Date the mail client can sort by
        assert "<p>All green</p>" in payload
