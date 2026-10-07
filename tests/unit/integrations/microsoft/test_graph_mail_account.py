"""GraphMailAccount against a fake of Graph v1.0: what the listing, search, read, move, send
and 2FA polling ask of the API and what the base renders from the answers."""
import base64
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from unittest.mock import patch

import pytest

from assistant.integration.mail.MailAccount import FolderNotFound
from integrations.microsoft.GraphClient import MAX_RETRY_AFTER, GraphClient
from integrations.microsoft.GraphMailAccount import GraphMailAccount, message_body, parse_graph_time

from .graph_fakes import FakeGraph, FakeResponse, StubAuth, message, recipient

REQUEST = "integrations.microsoft.GraphClient.requests.request"
INBOX = [
    message("AAMk-1", "Invoice 2026-09", ("alice@example.com", "Alice"), "2026-10-06T08:15:00Z", False),
    message("AAMk-2", "Lunch?", ("bob@example.com",), "2026-10-05T17:40:00.1234567Z", True),
]
FOLDERS = {"value": [{"id": "f-inbox", "displayName": "Inbox", "childFolderCount": 0},
                     {"id": "f-rec", "displayName": "Receipts", "childFolderCount": 1}]}
CHILDREN = {"value": [{"id": "f-2026", "displayName": "2026"}]}


@pytest.fixture
def graph():
    fake = FakeGraph()
    with patch(REQUEST, fake):
        yield fake


def account(auth=None) -> GraphMailAccount:
    auth = auth or StubAuth()
    return GraphMailAccount("Work", "WO", lambda: "marcus@example.com", GraphClient(auth))


@pytest.mark.unit
class TestListing:
    def test_the_inbox_listing_carries_short_ids_the_unread_mark_sender_subject_and_a_local_time(self, graph):
        graph.route("GET", "/me/mailFolders/inbox/messages", {"value": INBOX})

        lines = account().fetch_latest().split("\n")

        assert lines[0].startswith("[WO-AA] [UNREAD] ") and "Alice <alice@example.com>" in lines[0] and "Invoice 2026-09" in lines[0]
        assert lines[1].startswith("[WO-AB] ") and "[UNREAD]" not in lines[1] and "bob@example.com" in lines[1]
        call, = graph.sent("GET", "/me/mailFolders/inbox/messages")
        assert call.params["$orderby"] == "receivedDateTime desc" and call.params["$top"] == 35 and "$filter" not in call.params
        assert call.headers["Authorization"] == "Bearer tok-1"

    def test_unread_only_filters_on_the_server(self, graph):
        graph.route("GET", "/me/mailFolders/inbox/messages", {"value": INBOX[:1]})

        account().fetch_latest(unread_only=True)

        assert graph.sent("GET", "/me/mailFolders/inbox/messages")[0].params["$filter"] == "isRead eq false"

    def test_a_named_folder_is_resolved_through_the_folder_list_children_as_parent_slash_child(self, graph):
        graph.route("GET", "/me/mailFolders", FOLDERS)
        graph.route("GET", "/me/mailFolders/f-rec/childFolders", CHILDREN)
        graph.route("GET", "/me/mailFolders/f-2026/messages", {"value": INBOX[:1]})
        acc = account()

        assert acc.list_folders().split("\n") == ["Inbox", "Receipts", "Receipts/2026"]
        assert "Invoice 2026-09" in acc.fetch_latest("receipts/2026")
        assert acc.fetch_latest("Nowhere") == "No emails found."

    def test_a_search_uses_search_without_an_orderby(self, graph):
        graph.route("GET", "/me/messages", {"value": INBOX[:1]})

        text = account().search('invoice "Alice"')

        call, = graph.sent("GET", "/me/messages")
        assert call.params["$search"] == '"invoice \\"Alice\\""' and "$orderby" not in call.params
        assert text.startswith("[WO-AA]")


@pytest.mark.unit
class TestRead:
    def test_an_html_body_and_a_pdf_attachment_are_rendered_through_the_base(self, graph):
        graph.route("GET", "/me/mailFolders/inbox/messages", {"value": INBOX})
        full = message("AAMk-1", "Invoice 2026-09", ("alice@example.com", "Alice"), "2026-10-06T08:15:00Z", False,
                       body=("html", "<html><body><p>Hi Marcus,</p><p>see the <a href='https://pay.example/1'>invoice</a></p></body></html>"),
                       toRecipients=[recipient("marcus@example.com", "Marcus")], hasAttachments=True)
        graph.route("GET", "/me/messages/AAMk-1", full)
        graph.route("GET", "/me/messages/AAMk-1/attachments", {"value": [
            {"@odata.type": "#microsoft.graph.fileAttachment", "id": "att-1", "name": "invoice.pdf",
             "contentType": "application/pdf", "size": 20480, "isInline": False,
             "contentBytes": base64.b64encode(b"%PDF-fake").decode()},
            {"@odata.type": "#microsoft.graph.itemAttachment", "id": "att-2", "name": "forwarded"}]})
        acc = account()
        acc.fetch_latest()

        with patch("assistant.integration.mail.MailAccount.PdfExtractor.extract_text", return_value="Amount due: 120 SEK") as extract:
            text = acc.read("WO-AA")

        assert "From: Alice <alice@example.com>" in text and "To: Marcus <marcus@example.com>" in text
        assert "Hi Marcus," in text and '<a href="https://pay.example/1">invoice</a>' in text and "<p>" not in text
        assert "invoice.pdf (PDF, 20 KB)" in text and "Amount due: 120 SEK" in text
        assert extract.call_args.args[0] == b"%PDF-fake"
        assert graph.sent("GET", "/me/messages/AAMk-1")[0].params["$select"].split(",") >= ["body"]

    def test_a_plain_text_body_is_shown_as_is(self, graph):
        graph.route("GET", "/me/mailFolders/inbox/messages", {"value": INBOX})
        graph.route("GET", "/me/messages/AAMk-2", message("AAMk-2", "Lunch?", ("bob@example.com",), "2026-10-05T17:40:00Z", True,
                                                           body=("text", "Pizza at 12?"), toRecipients=[recipient("marcus@example.com")]))
        acc = account()
        acc.fetch_latest()

        assert "Pizza at 12?" in acc.read("WO-AB")

    def test_a_message_graph_no_longer_has_is_not_found(self, graph):
        graph.route("GET", "/me/mailFolders/inbox/messages", {"value": INBOX})
        acc = account()
        acc.fetch_latest()

        assert acc.read("WO-AA") == "Email with ID AAMk-1 not found."


@pytest.mark.unit
class TestMove:
    def test_a_move_posts_the_destination_folder_id(self, graph):
        graph.route("GET", "/me/mailFolders/inbox/messages", {"value": INBOX})
        graph.route("GET", "/me/mailFolders", FOLDERS)
        graph.route("GET", "/me/mailFolders/f-rec/childFolders", CHILDREN)
        graph.route("POST", "/me/messages/AAMk-1/move", {"id": "AAMk-9"}, status=201)
        acc = account()
        acc.fetch_latest()

        result = acc.move("WO-AA", "Receipts")

        assert result == 'Moved email WO-AA ("Invoice 2026-09") to Receipts.'
        assert graph.sent("POST", "/me/messages/AAMk-1/move")[0].json == {"destinationId": "f-rec"}

    def test_a_well_known_folder_needs_no_lookup_and_an_unknown_one_is_a_folder_error(self, graph):
        graph.route("GET", "/me/mailFolders/inbox/messages", {"value": INBOX})
        graph.route("GET", "/me/mailFolders", FOLDERS)
        graph.route("GET", "/me/mailFolders/f-rec/childFolders", CHILDREN)
        graph.route("POST", "/me/messages/AAMk-1/move", {"id": "AAMk-9"}, status=201)
        acc = account()
        acc.fetch_latest()

        acc.move("WO-AA", "Deleted Items")
        assert graph.sent("POST", "/me/messages/AAMk-1/move")[0].json == {"destinationId": "deleteditems"}
        assert not graph.sent("GET", "/me/mailFolders")
        assert acc.move("WO-AA", "Attic").startswith("Error moving email: folder not found")
        with pytest.raises(FolderNotFound):
            acc._folder_id("Attic")


@pytest.mark.unit
class TestSend:
    def test_a_plain_text_send_posts_the_message_json_and_lets_graph_file_the_sent_copy(self, graph):
        graph.route("POST", "/me/sendMail", None, status=202)

        assert account().send("alice@example.com", "Hi", "See you at 12") == ""

        call, = graph.sent("POST", "/me/sendMail")
        assert call.json["saveToSentItems"] is True
        assert call.json["message"]["subject"] == "Hi"
        assert call.json["message"]["body"] == {"contentType": "Text", "content": "See you at 12"}
        assert call.json["message"]["toRecipients"] == [{"emailAddress": {"address": "alice@example.com"}}]

    def test_an_html_alternative_is_sent_as_html(self, graph):
        graph.route("POST", "/me/sendMail", None, status=202)
        msg = MIMEMultipart("alternative")
        msg["Subject"] = "Nightly report"
        msg["To"] = "marcus@example.com"
        msg.attach(MIMEText("plain fallback", "plain"))
        msg.attach(MIMEText("<html><body><b>3 tests fixed</b></body></html>", "html"))

        assert account().send_message(msg, ["marcus@example.com"]) == ""

        body = graph.sent("POST", "/me/sendMail")[0].json["message"]["body"]
        assert body["contentType"] == "HTML" and "<b>3 tests fixed</b>" in body["content"]

    def test_message_body_of_a_plain_message_keeps_its_charset(self):
        msg = MIMEText("Hej Charlotte, fika?", "plain", "utf-8")
        assert message_body(msg) == ("Text", "Hej Charlotte, fika?")


@pytest.mark.unit
class TestErrors:
    def test_a_401_tells_the_llm_to_sign_in_again(self, graph):
        graph.route("POST", "/me/sendMail", {"error": {"code": "InvalidAuthenticationToken", "message": "Lifetime validation failed"}}, status=401)
        graph.route("GET", "/me/mailFolders/inbox/messages", {"error": {"code": "InvalidAuthenticationToken", "message": "expired"}}, status=401)
        acc = account()

        assert 'microsoft_login("Work")' in acc.send("alice@example.com", "Hi", "x")
        assert 'microsoft_login("Work")' in acc.fetch_latest()

    def test_without_a_token_no_request_leaves_the_machine(self, graph):
        acc = account(StubAuth(signed_in=False))

        assert 'microsoft_login("Work")' in acc.fetch_latest()
        assert graph.calls == []

    def test_a_429_is_retried_once_after_the_retry_after_then_given_up(self, graph):
        answers = [FakeResponse(429, None, {"Retry-After": "3"}), FakeResponse(200, {"value": INBOX[:1]})]
        graph.route("GET", "/me/mailFolders/inbox/messages", lambda p, h, j: answers.pop(0))
        with patch("integrations.microsoft.GraphClient.time.sleep") as sleep:
            assert "Invoice 2026-09" in account().fetch_latest()
        sleep.assert_called_once_with(3)

        graph.route("GET", "/me/mailFolders/inbox/messages", lambda p, h, j: FakeResponse(429, None, {"Retry-After": "1"}))
        with patch("integrations.microsoft.GraphClient.time.sleep"):
            assert "throttling" in account().fetch_latest()

    def test_the_retry_after_wait_is_clamped_to_a_minute_at_least_a_second_and_five_without_the_header(self, graph):
        for headers, wait in (({"Retry-After": "600"}, MAX_RETRY_AFTER), ({}, 5), ({"Retry-After": "0"}, 1)):
            answers = [FakeResponse(429, None, headers), FakeResponse(200, {"value": INBOX[:1]})]
            graph.route("GET", "/me/mailFolders/inbox/messages", lambda p, h, j, a=answers: a.pop(0))
            with patch("integrations.microsoft.GraphClient.time.sleep") as sleep:
                assert "Invoice 2026-09" in account().fetch_latest()
            sleep.assert_called_once_with(wait)

    def test_graphs_own_message_reaches_the_llm_on_any_other_error(self, graph):
        graph.route("POST", "/me/sendMail", {"error": {"code": "ErrorSendAsDenied", "message": "Not allowed to send as this user"}}, status=403)

        assert "Not allowed to send as this user" in account().send("alice@example.com", "Hi", "x")


@pytest.mark.unit
class TestRecentFrom:
    def test_the_2fa_polling_asks_for_the_last_day_and_keeps_the_senders_mail_with_its_body(self, graph):
        graph.route("GET", "/me/messages", {"value": [
            message("AAMk-5", "Your Arlo code", ("no-reply@arlo.example", "Arlo"), "2026-10-06T08:15:00Z", False, body=("text", "Code 123456")),
            message("AAMk-6", "Hi", ("carol@example.com",), "2026-10-06T08:10:00Z", False, body=("html", "<p>hey</p>"))]})

        found = account()._recent_from("arlo", 5)

        call, = graph.sent("GET", "/me/messages")
        assert call.params["$filter"].startswith("receivedDateTime ge 20") and call.params["$orderby"] == "receivedDateTime desc"
        assert [m.subject for m in found] == ["Your Arlo code"]
        assert found[0].text == "Code 123456" and found[0].html == "" and found[0].date.tzinfo is not None
        assert found[0].from_ == "Arlo <no-reply@arlo.example>"

    def test_a_health_check_asks_who_the_token_is(self, graph):
        graph.route("GET", "/me", {"userPrincipalName": "marcus@example.com", "mail": "marcus@example.com"})
        assert account().check() is True


@pytest.mark.unit
def test_graph_times_with_seven_fraction_digits_parse_to_utc():
    parsed = parse_graph_time("2026-10-05T17:40:00.1234567Z")
    assert (parsed.hour, parsed.minute, parsed.utcoffset().total_seconds()) == (17, 40, 0)
    assert parse_graph_time("") is None and parse_graph_time("yesterday") is None
