"""MailAccount, the protocol-free part of a mailbox: the short-id cache, the listing and body
rendering, send validation and its rate limit — exercised through an in-memory backend."""
import smtplib
from datetime import datetime, timezone
from typing import List
from unittest.mock import Mock

import pytest

from assistant.integration.mail.MailAccount import MailAccount, FolderNotFound, SEND_RATE_LIMIT_MAX


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


class _MemoryAccount(MailAccount):
    """A backend over dicts: folders of messages, a log of what was sent and moved."""
    default_folder = "INBOX"

    def __init__(self, folders=None, address="m@mlab.test"):
        super().__init__()
        self.folders = folders if folders is not None else {"INBOX": []}
        self._address = address
        self.sent: List[tuple] = []
        self.moved: List[tuple] = []
        self.send_error = ""

    @property
    def name(self) -> str:
        return "fake"

    @property
    def id_prefix(self) -> str:
        return "FK"

    @property
    def address(self) -> str:
        return self._address

    def _list(self, folder, unread_only):
        if folder not in self.folders:
            raise FolderNotFound(f"no folder {folder}")
        messages = self.folders[folder]
        return [m for m in messages if "\\Seen" not in (m.flags or ())] if unread_only else list(messages)

    def _search(self, search_string):
        return [m for m in self.folders[self.default_folder] if search_string.lower() in (m.subject or "").lower()]

    def _fetch(self, uid, folder):
        return next((m for m in self.folders.get(folder, []) if m.uid == uid), None)

    def _folders(self):
        return list(self.folders)

    def _move(self, uid, source, dest_folder):
        if dest_folder not in self.folders:
            raise FolderNotFound(f"no folder {dest_folder}")
        self.moved.append((uid, source, dest_folder))

    def _recent_from(self, sender_contains, limit):
        return [m for m in self.folders[self.default_folder] if sender_contains in m.from_][:limit]

    def _send(self, msg, recipients):
        if self.send_error:
            return self.send_error
        self.sent.append((msg, recipients))
        return ""

    def check(self):
        return True


@pytest.mark.unit
class TestListing:
    def test_a_listing_line_carries_prefix_id_local_time_sender_and_subject(self):
        utc_date = datetime(2024, 3, 15, 10, 30, tzinfo=timezone.utc)
        account = _MemoryAccount({"INBOX": [_message(uid="42", from_="alice@example.com", subject="Hello", date=utc_date)]})

        result = account.fetch_latest()

        assert result.startswith("[FK-AA] ")
        assert "From: alice@example.com" in result and "Subject: Hello" in result
        assert utc_date.astimezone().strftime("%b %d %H:%M") in result

    def test_ids_count_on_across_listings_so_parallel_searches_never_share_one(self):
        account = _MemoryAccount()
        account._format_email_list([_message(uid="100"), _message(uid="200"), _message(uid="300")])
        assert account._email_id_cache == {"AA": "100", "AB": "200", "AC": "300"}

        account._format_email_list([_message(uid="400")])

        assert account._email_id_cache == {"AA": "100", "AB": "200", "AC": "300", "AD": "400"}

    def test_a_relisted_uid_keeps_its_id(self):
        account = _MemoryAccount()
        account._format_email_list([_message(uid="100"), _message(uid="200")])
        second = account._format_email_list([_message(uid="200"), _message(uid="300")])

        assert "[FK-AB]" in second and "[FK-AC]" in second and "[FK-AA]" not in second
        assert account._email_id_cache == {"AA": "100", "AB": "200", "AC": "300"}

    def test_the_same_uid_in_another_folder_is_another_mail(self):
        account = _MemoryAccount()
        account._format_email_list([_message(uid="7")], "INBOX")
        account._format_email_list([_message(uid="7")], "Archive")

        assert account._email_id_cache == {"AA": "7", "AB": "7"}
        assert account._folder_of("FK-AA") == "INBOX" and account._folder_of("AB") == "Archive"

    def test_the_cache_is_dropped_only_when_the_ids_would_wrap(self):
        account = _MemoryAccount()
        account._format_email_list([_message(uid="1")])
        account._next_cache_id = MailAccount.SHORT_ID_SPACE - 2
        account._format_email_list([_message(uid="2"), _message(uid="3")])
        assert set(account._email_id_cache.values()) == {"1", "2", "3"}

        listing = account._format_email_list([_message(uid="4")])

        assert listing.startswith("[FK-AA] ") and account._email_id_cache == {"AA": "4"}
        assert account.subject_for("ZZ") is None

    def test_parallel_listings_hand_out_distinct_ids(self):
        import threading
        account = _MemoryAccount()
        listings = [[_message(uid=str(n * 10 + k), subject=f"mail {n}-{k}") for k in range(5)] for n in range(4)]
        threads = [threading.Thread(target=account._format_email_list, args=(batch,)) for batch in listings]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(account._email_id_cache) == 20
        assert sorted(account._email_id_cache.values()) == sorted(str(n * 10 + k) for n in range(4) for k in range(5))

    def test_unread_marker_only_without_the_seen_flag(self):
        account = _MemoryAccount()
        assert "[UNREAD]" in account._format_email_list([_message(flags=())])
        assert "[UNREAD]" in account._format_email_list([_message(flags=None)])
        assert "[UNREAD]" not in account._format_email_list([_message(flags=("\\Seen",))])

    def test_no_subject_and_no_messages_have_placeholders(self):
        account = _MemoryAccount()
        assert "(no subject)" in account._format_email_list([_message(subject=None)])
        assert account._format_email_list([]) == "No emails found."

    def test_listing_a_folder_remembers_it_for_read_and_move(self):
        account = _MemoryAccount({"INBOX": [], "@SaneLater": [_message(uid="500")]})

        account.fetch_latest("@SaneLater")
        account.move("AA", "INBOX")

        assert account.moved == [("500", "@SaneLater", "INBOX")]

    def test_a_missing_folder_lists_as_no_emails(self):
        assert _MemoryAccount().fetch_latest("@Nonexistent") == "No emails found."

    def test_unread_only_filters_the_seen_ones(self):
        account = _MemoryAccount({"INBOX": [_message(uid="1", subject="Old", flags=("\\Seen",)),
                                            _message(uid="2", subject="New", flags=())]})

        result = account.fetch_latest("INBOX", unread_only=True)

        assert "New" in result and "Old" not in result

    def test_a_backend_failure_is_an_error_text_not_an_exception(self):
        account = _MemoryAccount()
        account._list = Mock(side_effect=Exception("Connection refused"))

        result = account.fetch_latest()

        assert "Error" in result and "Connection refused" in result

    def test_search_lists_from_the_default_folder(self):
        account = _MemoryAccount({"INBOX": [_message(uid="9", subject="Invoice 4711")], "@SaneLater": []})
        account._cache_folder = "@SaneLater"

        result = account.search("invoice")

        assert "[FK-AA]" in result and "Invoice 4711" in result
        assert account._cache_folder == "INBOX"


@pytest.mark.unit
class TestSubjectCache:
    def test_subject_cached_after_listing_with_or_without_the_prefix(self):
        account = _MemoryAccount()
        account._format_email_list([_message(uid="100", subject="Hello there")])

        assert account.subject_for("AA") == "Hello there"
        assert account.subject_for("FK-AA") == "Hello there"

    def test_subjects_follow_the_id_across_listings(self):
        account = _MemoryAccount()
        account._format_email_list([_message(uid="100", subject="First batch")])
        account._format_email_list([_message(uid="200", subject="Second batch")])

        assert account.subject_for("AA") == "First batch"
        assert account.subject_for("AB") == "Second batch"

    def test_a_short_id_resolves_to_the_uid_and_an_unknown_id_passes_through(self):
        account = _MemoryAccount()
        account._email_id_cache["AA"] = "real-uid-123"

        assert account._resolve_email_id("FK-AA") == "real-uid-123"
        assert account._resolve_email_id("99999") == "99999"


@pytest.mark.unit
class TestRead:
    def test_reads_the_listed_message_whole(self):
        account = _MemoryAccount({"INBOX": [_message(uid="55", from_="bob@example.com", subject="Important", text="Hello there!")]})
        account.fetch_latest()

        result = account.read("FK-AA")

        assert "From: bob@example.com" in result and "To: m@mlab.test" in result
        assert "Subject: Important" in result and "Hello there!" in result

    def test_reads_from_the_listed_folder(self):
        account = _MemoryAccount({"INBOX": [], "@SaneLater": [_message(uid="500", text="later")]})
        account.fetch_latest("@SaneLater")

        assert "later" in account.read("AA")

    def test_reads_from_the_folder_the_id_was_listed_in_after_a_later_listing_elsewhere(self):
        account = _MemoryAccount({"INBOX": [_message(uid="1", text="inbox mail")],
                                  "@SaneLater": [_message(uid="500", text="later")]})
        account.fetch_latest("@SaneLater")
        account.fetch_latest()

        assert "later" in account.read("AA") and "inbox mail" in account.read("AB")

    def test_a_raw_uid_the_cache_never_saw_is_read_from_the_last_listings_folder(self):
        # the short ids remember their folder; a bare UID has none, so the last listing's is the guess
        account = _MemoryAccount({"INBOX": [_message(uid="7", text="inbox mail")],
                                  "@SaneLater": [_message(uid="7", text="later")]})
        account.fetch_latest("@SaneLater")

        assert "later" in account.read("7")

    def test_an_unknown_uid_is_not_found(self):
        assert "not found" in _MemoryAccount().read("999")

    def test_html_is_the_body_when_there_is_no_text(self):
        account = _MemoryAccount({"INBOX": [_message(uid="1", text="", html="<p>HTML content</p>")]})
        assert "HTML content" in account.read("1")

    def test_no_body_at_all_says_so(self):
        account = _MemoryAccount({"INBOX": [_message(uid="1", text="", html="")]})
        assert "(no body)" in account.read("1")

    def test_a_long_body_is_truncated(self):
        account = _MemoryAccount({"INBOX": [_message(uid="1", text="x" * 10000)]})

        result = account.read("1")

        assert "... (truncated)" in result
        assert len(result[result.index("\n\n") + 2:]) < 10000

    def test_html_links_missing_from_the_text_are_appended_once(self):
        url = "https://book.example.com/cancel/10203040"
        account = _MemoryAccount()
        with_link = account._format_email_body(_message(text="Your booking is confirmed.", html=f'<a href="{url}">Cancel</a>'))
        already = account._format_email_body(_message(text=f"Cancel here: {url}", html=f'<a href="{url}">Cancel</a>'))

        assert url in with_link
        assert already.count(url) == 1


@pytest.mark.unit
class TestStripHtml:
    def test_script_and_style_are_dropped(self):
        html = "<html><head><style>body{}</style></head><body><script>alert(1)</script><p>Hello</p></body></html>"
        result = MailAccount.strip_html(html)
        assert "alert" not in result and "body{}" not in result and "Hello" in result

    def test_absolute_and_mailto_anchors_keep_their_href(self):
        html = ('<p>Manage: <a href="https://book.example.com/cancel/10203040">Cancel booking</a> '
                '<a href="mailto:support@example.com">Email support</a></p>')
        result = MailAccount.strip_html(html)
        assert '<a href="https://book.example.com/cancel/10203040">Cancel booking</a>' in result
        assert '<a href="mailto:support@example.com">Email support</a>' in result

    def test_an_anchor_without_text_falls_back_to_its_href(self):
        assert "https://book.example.com/manage" in MailAccount.strip_html('<a href="https://book.example.com/manage"><img src="cid:btn"></a>')

    def test_relative_and_fragment_links_are_flattened_to_text(self):
        result = MailAccount.strip_html('<a href="#top">Top</a> <a href="/foo">Foo</a>')
        assert "href" not in result and "Top" in result and "Foo" in result

    def test_plain_variant_drops_every_url(self):
        html = '<p><a href="https://arlo.example/track/482910">View</a> Code: 654321</p>'
        result = MailAccount.strip_html_plain(html)
        assert "482910" not in result and "Code: 654321" in result


@pytest.mark.unit
class TestMove:
    def test_the_result_echoes_the_cached_subject_truncated(self):
        account = _MemoryAccount({"INBOX": [_message(uid="500", subject="A" * 80)], "Archive": []})
        account.fetch_latest()

        result = account.move("AA", '"Archive"')

        assert account.moved == [("500", "INBOX", "Archive")]
        assert f'("{"A" * 50}…")' in result and "A" * 51 not in result

    def test_a_placeholder_subject_or_a_cache_miss_leaves_the_id_alone(self):
        account = _MemoryAccount({"INBOX": [_message(uid="500", subject=None)], "Archive": []})
        account.fetch_latest()

        assert account.move("AA", "Archive") == "Moved email AA to Archive."
        assert account.move("ZZ", "Archive") == "Moved email ZZ to Archive."

    def test_moves_from_the_folder_the_id_was_listed_in_after_a_later_listing_elsewhere(self):
        # IMAP UIDs are per folder: the id listed from @SaneLater must be moved out of it even
        # though the INBOX listing came later — the source is the id's folder, not the last one
        account = _MemoryAccount({"INBOX": [_message(uid="1")], "@SaneLater": [_message(uid="500")], "Archive": []})
        account.fetch_latest("@SaneLater")
        account.fetch_latest()

        account.move("AA", "Archive")
        account.move("AB", "Archive")

        assert account.moved == [("500", "@SaneLater", "Archive"), ("1", "INBOX", "Archive")]

    def test_a_missing_destination_is_a_folder_error(self):
        result = _MemoryAccount().move("1", "Nowhere")
        assert result.startswith("Error moving email: folder not found")


@pytest.mark.unit
class TestSendValidation:
    def test_a_valid_send_fills_the_headers_and_reaches_the_backend(self):
        account = _MemoryAccount()

        assert account.send("alice@example.com", "Hi", "Hello there") == ""

        (msg, recipients), = account.sent
        assert recipients == ["alice@example.com"]
        assert msg["From"] == "m@mlab.test" and msg["To"] == "alice@example.com" and msg["Subject"] == "Hi"
        assert msg["Date"] and msg["Message-ID"].endswith("@mlab.test>")
        assert msg.get_payload(decode=True).decode("utf-8") == "Hello there"

    @pytest.mark.parametrize("to, subject, body, expected", [
        ("", "subject", "body", "empty recipient"),
        ("a@b.com", "  ", "body", "empty subject"),
        ("a@b.com", "subject", "   ", "empty body"),
        ("not-an-email", "subject", "body", "invalid recipient"),
    ])
    def test_an_incomplete_or_invalid_send_never_reaches_the_backend(self, to, subject, body, expected):
        account = _MemoryAccount()
        assert expected in account.send(to, subject, body)
        assert account.sent == []

    def test_a_prepared_message_keeps_its_own_headers(self):
        from email.mime.text import MIMEText
        report = MIMEText("<p>All green</p>", "html")
        report["Subject"] = "Nightly report"
        report["Message-ID"] = "<nightly-2026-10-06@mlab.test>"
        account = _MemoryAccount()

        assert account.send_message(report, ["m@mlab.test"]) == ""

        (msg, _), = account.sent
        assert msg["Message-ID"] == "<nightly-2026-10-06@mlab.test>" and msg["From"] == "m@mlab.test"

    def test_without_an_address_nothing_is_sent(self):
        account = _MemoryAccount(address="")
        assert "credentials not configured" in account.send("alice@example.com", "Hi", "body")
        assert account.sent == []

    def test_a_backend_error_is_the_result_and_does_not_count_toward_the_limit(self):
        account = _MemoryAccount()
        account.send_error = "SMTP error: boom"
        for _ in range(SEND_RATE_LIMIT_MAX + 1):
            assert account.send("alice@example.com", "Hi", "body") == "SMTP error: boom"

    def test_the_rate_limit_blocks_after_the_allowed_sends(self):
        account = _MemoryAccount()
        for _ in range(SEND_RATE_LIMIT_MAX):
            assert account.send("alice@example.com", "Hi", "body") == ""

        assert "rate limit" in account.send("alice@example.com", "Hi", "body")
        assert len(account.sent) == SEND_RATE_LIMIT_MAX


@pytest.mark.unit
class TestRecentRaw:
    def test_only_mail_within_the_window_comes_back(self):
        now = datetime.now(timezone.utc)
        fresh = _message(uid="1", from_="noreply@arlo.example", date=now)
        stale = _message(uid="2", from_="noreply@arlo.example", date=now.replace(year=now.year - 1))
        account = _MemoryAccount({"INBOX": [fresh, stale]})

        assert account.fetch_recent_raw("arlo", since_minutes=10) == [fresh]

    def test_a_backend_failure_is_an_empty_list(self):
        account = _MemoryAccount()
        account._recent_from = Mock(side_effect=smtplib.SMTPException("down"))
        assert account.fetch_recent_raw("arlo") == []
