"""Tests for GoogleDrive, the Drive API v3 calls behind the docs feed: how Google's answers
are read — a 401 and a 403 for the missing scope are sign-in verdicts, a rate limit gets one
retry and no more, any other failure carries Google's own message — and the revision list and
the export the feed is built from: every page, oldest first, and a revision without an export
link exported at the head. The requests are answered by the fakes of the calendar tests,
addressed by the Drive base instead."""
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from integrations.google.GoogleAuth import NotSignedIn
from integrations.google.GoogleDrive import (BASE_URL, SHEETS_URL, SLIDES_URL, ApiDisabled, DriveFile, GoogleDrive,
                                             GoogleDriveError, Revision, file_id_of, kind_of, rfc3339_ts)

from .google_fakes import Call, FakeGoogle, FakeResponse, StubAuth

REQUEST = "integrations.google.GoogleDrive.requests.request"
SLEEP = "integrations.google.GoogleDrive.time.sleep"
FILE = "1AbCdEfGhIjKlMnOpQrStUvWxYz"
EXPORT_LINK = "https://docs.google.com/feeds/download/documents/export/Export?id=1AbC&revision=7&exportFormat=txt"
HEAD = {"name": "Family plan 2027", "mimeType": "application/vnd.google-apps.document", "modifiedTime": "2026-10-14T12:02:33.123Z"}


def _error(status: int, message: str, reason: str = "", details_reason: str = "") -> FakeResponse:
    body = {"code": status, "message": message}
    if reason:
        body["errors"] = [{"reason": reason}]
    if details_reason:
        body["details"] = [{"reason": details_reason}]
    return FakeResponse(status, {"error": body})


def _answers(*responses):
    """A route that hands out `responses` in order — a rate limit, then the real answer."""
    queue = list(responses)
    return lambda params, json: queue.pop(0)


class FakeDriveApi(FakeGoogle):
    """Answers a Drive request by its path under the Drive base, an export link by its whole URL.
    A route is a ready FakeResponse or a callable of (params, json) returning one."""

    def route(self, method: str, path: str, body=None, status: int = 200):
        self.routes[(method, path)] = body if callable(body) or isinstance(body, FakeResponse) else FakeResponse(status, body)

    def __call__(self, method, url, params=None, headers=None, json=None, timeout=None):
        path = url[len(BASE_URL):] if url.startswith(BASE_URL) else url
        self.calls.append(Call(method, path, dict(params or {}), dict(headers or {}), json))
        handler = self.routes.get((method, path))
        if handler is None:
            return _error(404, f"nothing at {method} {path}", "notFound")
        return handler(params, json) if callable(handler) else handler


@pytest.fixture
def api():
    fake = FakeDriveApi()
    with patch(REQUEST, fake), patch(SLEEP) as sleep:
        fake.sleep = sleep
        yield fake


@pytest.fixture
def drive():
    return GoogleDrive("Personal", StubAuth("Personal"))


@pytest.mark.unit
class TestVerdicts:
    def test_a_401_means_the_account_has_to_sign_in_again(self, api, drive):
        api.route("GET", f"/files/{FILE}", _error(401, "Invalid Credentials", "authError"))
        with pytest.raises(NotSignedIn) as raised:
            drive.head(FILE)
        assert 'google_login("Personal")' in str(raised.value)

    @pytest.mark.parametrize("response", [
        _error(403, "Request had insufficient authentication scopes.", reason="insufficientPermissions"),
        _error(403, "Request had insufficient authentication scopes.", details_reason="ACCESS_TOKEN_SCOPE_INSUFFICIENT"),
    ], ids=["errors[].reason", "details[].reason"])
    def test_a_403_for_the_missing_scope_is_a_sign_in_verdict_whichever_shape_google_gives_the_reason(self, api, drive, response):
        """A token granted before the docs were listed lacks the Drive scope: that is not a
        Drive failure to log and retry but the sign-in-again hint, found under either of the
        two keys Google puts the reason in."""
        api.route("GET", f"/files/{FILE}", response)
        with pytest.raises(NotSignedIn) as raised:
            drive.head(FILE)
        assert "scope" in str(raised.value) and 'google_login("Personal")' in str(raised.value)

    def test_any_other_failure_is_a_drive_error_carrying_the_status_and_googles_message(self, api, drive):
        api.route("GET", f"/files/{FILE}", _error(404, f"File not found: {FILE}.", "notFound"))
        with pytest.raises(GoogleDriveError) as raised:
            drive.head(FILE)
        assert raised.value.status == 404
        assert FILE in str(raised.value)
        assert len(api.calls) == 1, "a missing file is not retried"


@pytest.mark.unit
class TestRateLimit:
    @pytest.mark.parametrize("limited", [
        FakeResponse(429, {"error": {"code": 429, "message": "Too Many Requests"}}),
        _error(403, "User Rate Limit Exceeded", "userRateLimitExceeded"),
    ], ids=["429", "403 userRateLimitExceeded"])
    def test_a_rate_limit_is_retried_once_after_a_pause_and_the_second_answer_is_the_result(self, api, drive, limited):
        api.route("GET", f"/files/{FILE}", _answers(limited, FakeResponse(200, HEAD)))
        head = drive.head(FILE)
        assert head.name == "Family plan 2027" and head.is_document
        assert len(api.sent("GET", f"/files/{FILE}")) == 2
        assert api.sleep.call_count == 1

    def test_a_rate_limit_that_persists_ends_after_the_one_retry_not_in_a_loop(self, api, drive):
        api.route("GET", f"/files/{FILE}", _answers(
            FakeResponse(429, {}), FakeResponse(429, {}), FakeResponse(200, HEAD)))
        with pytest.raises(GoogleDriveError) as raised:
            drive.head(FILE)
        assert raised.value.status == 429
        assert len(api.calls) == 2


@pytest.mark.unit
class TestRevisions:
    def test_every_page_is_read_and_the_revisions_come_back_oldest_first(self, api, drive):
        """Google hands the list a page at a time and in its own order; the feed walks the
        revisions chronologically, so they are sorted by their time whatever page they were on."""
        pages = {
            None: {"revisions": [{"id": "3", "modifiedTime": "2026-10-14T12:00:00.000Z",
                                  "lastModifyingUser": {"displayName": "Charlotte"}, "exportLinks": {"text/plain": EXPORT_LINK}},
                                 {"id": "1", "modifiedTime": "2026-10-12T08:00:00.000Z",
                                  "lastModifyingUser": {"displayName": "Marcus"}, "exportLinks": {"text/plain": EXPORT_LINK}}],
                   "nextPageToken": "page-two"},
            "page-two": {"revisions": [{"id": "2", "modifiedTime": "2026-10-13T09:30:00.000Z"}]},
        }
        api.route("GET", f"/files/{FILE}/revisions", lambda params, json: FakeResponse(200, pages[params.get("pageToken")]))
        revisions = drive.revisions(FILE)
        assert [r.id for r in revisions] == ["1", "2", "3"]
        assert [r.author for r in revisions] == ["Marcus", "someone", "Charlotte"], "a revision without a user is someone's"
        assert revisions[1].export_url is None, "an old revision Google keeps no export links for"
        assert revisions[2].modified_ts == rfc3339_ts("2026-10-14T12:00:00.000Z")
        first, second = api.sent("GET", f"/files/{FILE}/revisions")
        assert "pageToken" not in first.params and second.params["pageToken"] == "page-two"
        assert second.params["fields"] == first.params["fields"], "the second page asks for the same fields"


@pytest.mark.unit
class TestExport:
    def test_a_revision_with_an_export_link_is_fetched_from_that_link_and_the_bom_stripped(self, api, drive):
        response = FakeResponse(200)
        response.content = "﻿First paragraph\nSecond".encode()
        api.route("GET", EXPORT_LINK, response)
        text = drive.export(FILE, Revision("7", 1.0, "Charlotte", EXPORT_LINK))
        assert text == "First paragraph\nSecond"
        assert [c.path for c in api.calls] == [EXPORT_LINK]

    def test_a_revision_without_an_export_link_is_exported_at_the_head_as_plain_text(self, api, drive):
        response = FakeResponse(200)
        response.content = b"the document now"
        api.route("GET", f"/files/{FILE}/export", response)
        text = drive.export(FILE, Revision("1", 1.0, "Marcus", None))
        assert text == "the document now"
        call, = api.calls
        assert call.path == f"/files/{FILE}/export" and call.params["mimeType"] == "text/plain"


@pytest.mark.unit
class TestSearch:
    def test_the_words_go_into_a_full_text_query_quotes_escaped_and_the_files_come_back_with_their_owner(self, api, drive):
        api.route("GET", "/files", {"files": [
            {"id": "1AbCdEfGhIjKlMnOpQrStUvWxYz", "name": "Family plan", "mimeType": "application/vnd.google-apps.document",
             "modifiedTime": "2026-10-02T10:00:00.000Z", "owners": [{"displayName": "Marcus"}]},
            {"id": "1ZyXwVuTsRqPoNmLkJiHgFeDcBa", "name": "Lease.pdf", "mimeType": "application/pdf",
             "modifiedTime": "2026-09-02T10:00:00.000Z"}], "nextPageToken": "more"})
        files, more = drive.search("marcus's budget")
        assert [(f.id, f.name, f.kind, f.owner, f.readable) for f in files] == [
            ("1AbCdEfGhIjKlMnOpQrStUvWxYz", "Family plan", "Google Doc", "Marcus", True),
            ("1ZyXwVuTsRqPoNmLkJiHgFeDcBa", "Lease.pdf", "PDF", "", False)]
        assert more
        call, = api.sent("GET", "/files")
        assert call.params["q"] == "fullText contains 'marcus\\'s' and fullText contains 'budget' and trashed = false"
        assert "owners" in call.params["fields"] and call.params["pageSize"] == 20

    def test_every_word_is_its_own_predicate_so_the_words_may_stand_apart_in_a_document(self, api, drive):
        """Drive reads `fullText contains 'a b'` as the phrase: measured 2026-11-05, the phrase form
        found 4 files where the per-word form found 6. Surrounding blanks add no empty predicate."""
        api.route("GET", "/files", {"files": []})
        drive.search("  summer   house budget ")
        call, = api.sent("GET", "/files")
        assert call.params["q"] == ("fullText contains 'summer' and fullText contains 'house' "
                                    "and fullText contains 'budget' and trashed = false")
        assert call.params["orderBy"] == "modifiedTime desc"

    def test_no_match_is_an_empty_list_without_more(self, api, drive):
        api.route("GET", "/files", {"files": []})
        assert drive.search("nothing") == ([], False)


@pytest.mark.unit
class TestFileId:
    @pytest.mark.parametrize("ref", [
        "https://docs.google.com/document/d/1AbCdEfGhIjKlMnOpQrStUvWxYz/edit#heading=h.2",
        "https://docs.google.com/presentation/d/1AbCdEfGhIjKlMnOpQrStUvWxYz/edit?usp=sharing",
        "https://drive.google.com/file/d/1AbCdEfGhIjKlMnOpQrStUvWxYz/view",
        "https://drive.google.com/open?id=1AbCdEfGhIjKlMnOpQrStUvWxYz",
        " <1AbCdEfGhIjKlMnOpQrStUvWxYz> ",
    ])
    def test_every_link_shape_and_a_bare_id_give_the_id(self, ref):
        assert file_id_of(ref) == "1AbCdEfGhIjKlMnOpQrStUvWxYz"

    def test_a_name_or_a_short_token_is_no_id(self):
        assert file_id_of("Family plan") == "" and file_id_of("https://example.com/d/short/edit") == ""

    def test_a_kind_is_known_by_its_mime_type_and_an_unknown_one_by_its_tail(self):
        assert kind_of("application/vnd.google-apps.presentation") == "Google Slides"
        assert kind_of("image/png") == "png" and kind_of("") == "file"


@pytest.mark.unit
class TestSlides:
    DECK = {"slides": [
        {"pageElements": [{"shape": {"text": {"textElements": [{"textRun": {"content": "Drones as\x0bfirst responders\n"}}]}}},
                          {"elementGroup": {"children": [{"shape": {"text": {"textElements": [{"textRun": {"content": "Grouped\n"}}]}}}]}},
                          {"table": {"tableRows": [{"tableCells": [{"text": {"textElements": [{"textRun": {"content": "Cell A"}}]}}]}]}}],
         "slideProperties": {"notesPage": {"pageElements": [{"shape": {"shapeType": "TEXT_BOX", "text": {"textElements": [
             {"textRun": {"content": "Say hello first.\n"}}]}}}]}}},
        {"pageElements": [{"shape": {"text": {"textElements": [{"textRun": {"content": "The end\n"}}]}}}]},
    ]}

    def test_each_slide_stands_under_its_marker_with_its_shapes_groups_tables_and_notes(self, api, drive):
        api.route("GET", f"{SLIDES_URL}/presentations/{FILE}", self.DECK)
        text = drive.slides_text(FILE)
        assert text.splitlines() == ["[slide 1]", "Drones as", "first responders", "Grouped", "Cell A", "Notes: Say hello first.",
                                     "", "[slide 2]", "The end"]
        call, = api.calls
        assert call.params["fields"].startswith("slides(")

    def test_a_project_without_the_slides_api_is_api_disabled_in_either_error_shape(self, api, drive):
        api.route("GET", f"{SLIDES_URL}/presentations/{FILE}",
                  _error(403, "Google Slides API has not been used in project 1 before or it is disabled", reason="accessNotConfigured"))
        with pytest.raises(ApiDisabled):
            drive.slides_text(FILE)
        api.route("GET", f"{SLIDES_URL}/presentations/{FILE}",
                  FakeResponse(403, {"error": {"code": 403, "message": "disabled", "details": [
                      {"@type": "type.googleapis.com/google.rpc.Help"}, {"@type": "ErrorInfo", "reason": "SERVICE_DISABLED"}]}}))
        with pytest.raises(ApiDisabled):
            drive.slides_text(FILE)


@pytest.mark.unit
class TestSheets:
    BOOK = {"sheets": [
        {"properties": {"title": "Budget 2026"}, "data": [{"rowData": [
            {"values": [{"formattedValue": "Item"}, {"formattedValue": "Cost"}]},
            {"values": [{}, {}]},
            {"values": [{"formattedValue": "Rent"}, {"formattedValue": "12 000 kr"}, {}]},
            {}]}]},
        {"properties": {"title": "Notes"}, "data": [{"rowData": [{"values": [{"formattedValue": "Ask Charlotte"}]}]}]},
    ]}

    def test_every_tab_stands_under_its_title_with_its_rows_as_shown_and_empty_rows_left_out(self, api, drive):
        api.route("GET", f"{SHEETS_URL}/spreadsheets/{FILE}", self.BOOK)
        text = drive.sheets_text(FILE)
        assert text.splitlines() == ["[sheet Budget 2026]", "Item\tCost", "Rent\t12 000 kr", "", "[sheet Notes]", "Ask Charlotte"]
        call, = api.calls
        assert call.params["includeGridData"] == "true" and "formattedValue" in call.params["fields"]

    def test_a_project_without_the_sheets_api_is_api_disabled_and_the_csv_export_asks_for_csv(self, api, drive):
        api.route("GET", f"{SHEETS_URL}/spreadsheets/{FILE}", _error(403, "Google Sheets API has not been used", reason="accessNotConfigured"))
        with pytest.raises(ApiDisabled):
            drive.sheets_text(FILE)
        response = FakeResponse(200)
        response.content = b"Item,Cost\nRent,12000"
        api.route("GET", f"/files/{FILE}/export", response)
        assert drive.export(FILE, mime_type="text/csv") == "Item,Cost\nRent,12000"
        assert api.sent("GET", f"/files/{FILE}/export")[0].params["mimeType"] == "text/csv"


def test_rfc3339_ts_reads_googles_utc_stamps_with_milliseconds():
    assert rfc3339_ts("2026-10-14T12:02:33.123Z") == datetime(2026, 10, 14, 12, 2, 33, 123000, tzinfo=timezone.utc).timestamp()
