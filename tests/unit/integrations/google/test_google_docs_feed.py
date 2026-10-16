"""The Google Docs feed: the diff of a revision against the cached text, the per-doc cursor
map, the baseline on first sight, the head check that spares the revision list, and the
sign-in problems reported instead of retried."""
import json
import time
from unittest.mock import MagicMock, patch

import pytest

from integrations.google import GoogleAuth as auth_module
from integrations.google import GoogleDocsIntegration as docs_module
from integrations.google.DocDiff import describe_change, tracking_note
from integrations.google.GoogleAuth import SCOPE_CALENDAR, SCOPE_DRIVE_READONLY, SCOPE_MAIL, token_path
from integrations.google.GoogleDocsIntegration import GoogleDocsIntegration
from integrations.google.GoogleDrive import BASE_URL, DOC_MIME_TYPE
from integrations.google.GoogleIntegration import GoogleIntegration, account_scopes
from assistant.prompt.MessageHistory import MessageHistory

from .google_fakes import Call, FakeGoogle, FakeResponse, token_file

CONFIG = 'integrations.google.GoogleIntegration.ConfigManager.get_instance'
REQUEST = "integrations.google.GoogleDrive.requests.request"
DOC_ID = "1AbCdEfGhIjKlMnOpQrStUvWxYz"
EXPORT = "https://docs.google.com/feeds/download/documents/export/Export?id=1AbC&revision=%s&exportFormat=txt"
HORIZON = 1_700_000_000.0
DAY = 86400.0
ALL_SCOPES = f"{SCOPE_MAIL} {SCOPE_CALENDAR} {SCOPE_DRIVE_READONLY}"


def _config(accounts, **more):
    values = {"google.accounts": accounts, "google.oauth.client_id": "cid", "google.oauth.client_secret": "csecret", **more}
    config = MagicMock()
    config.get.side_effect = lambda key, *default: values.get(key, default[0] if default else None)
    return patch(CONFIG, return_value=config)


def _entries(docs=({"name": "Family plan", "id": DOC_ID},)):
    return [{"name": "Personal", "address": "marcus@example.com", "mail": True, "calendar": True, "docs": list(docs)}]


def _signed_in(token_dir, name="Personal", scope=ALL_SCOPES):
    token_dir.mkdir(parents=True, exist_ok=True)
    body = token_file("marcus@example.com", access_token="ya29.live")
    body["scope"] = scope
    token_path(name).write_text(json.dumps(body))


def _rfc3339(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime(ts))


def _revision(number: int, ts: float, author: str = "Charlotte") -> dict:
    return {"id": str(number), "modifiedTime": _rfc3339(ts), "lastModifyingUser": {"displayName": author},
            "exportLinks": {"text/plain": EXPORT % number, "application/pdf": "https://example.com/pdf"}}


class FakeDrive(FakeGoogle):
    """The Drive routes: the head, the revisions and the text of each revision by its export link."""

    def __init__(self, modified_ts: float, revisions: list[dict], texts: dict[str, str], mime=DOC_MIME_TYPE):
        super().__init__()
        self.route("GET", f"/files/{DOC_ID}", {"name": "Family plan 2027", "mimeType": mime, "modifiedTime": _rfc3339(modified_ts)})
        self.route("GET", f"/files/{DOC_ID}/revisions", {"revisions": revisions})
        self.texts = texts

    def __call__(self, method, url, params=None, headers=None, json=None, timeout=None):
        if url.startswith(BASE_URL):
            return super().__call__(method, url[len(BASE_URL):], params, headers, json, timeout)
        self.calls.append(Call(method, url, dict(params or {}), dict(headers or {}), None))
        number = url.rsplit("revision=", 1)[1].split("&")[0]
        text = self.texts[number]
        response = FakeResponse(200)
        response.content = ("﻿" + text).encode()
        return response


@pytest.fixture(autouse=True)
def fresh(tmp_path):
    GoogleDocsIntegration.reset_instance()
    with patch.object(auth_module, "TOKEN_DIR", tmp_path / "oauth"), \
            patch.object(docs_module, "TEXT_CACHE_DIR", tmp_path / "google_docs"), \
            patch("assistant.integration.FeedRunner.Feeds.horizon_for", return_value=HORIZON):
        yield tmp_path / "oauth"
    GoogleDocsIntegration.reset_instance()


@pytest.mark.unit
class TestDocDiff:
    def test_a_changed_word_is_marked_inline_a_new_paragraph_added_with_its_context(self):
        old = "Intro\nThe trip is in May this year.\nPack the tent.\nEnd"
        new = "Intro\nThe trip is in June this year.\nPack the tent.\nBring the dog.\nEnd"
        content = describe_change("Plan", "Charlotte", old, new)
        lines = content.splitlines()
        assert "Charlotte" in lines[0] and "1 edited" in lines[0] and "1 added" in lines[0]
        assert "~ The trip is in [-May-] {+June+} this year." in lines
        assert "+ Bring the dog." in lines
        assert lines[lines.index("+ Bring the dog.") - 1] == "  Pack the tent."
        assert not any(line.startswith("- ") for line in lines)       # the edited paragraph is not printed twice

    def test_a_rewritten_paragraph_shows_as_removed_and_added_and_same_paragraphs_are_no_change(self):
        old = "Alpha\nThe old plan was to drive.\nOmega"
        new = "Alpha\nFlights are booked for the 3rd at 09:10, hotel by the station.\nOmega"
        lines = describe_change("Plan", "Bob", old, new).splitlines()
        assert "- The old plan was to drive." in lines and "+ Flights are booked for the 3rd at 09:10, hotel by the station." in lines
        assert "1 added, 1 removed" in lines[0]
        assert describe_change("Plan", "Bob", old, old + "\n\n") is None     # blank lines are not paragraphs

    def test_the_capped_row_renders_whole_in_the_prompt_a_contract_with_the_history_render_limit(self):
        old = "\n".join(f"Paragraph {i} stays" for i in range(400))
        new = "\n".join(f"Paragraph {i} stays\nAdded after {i}" for i in range(400))
        content = describe_change("Plan", "Carol", old, new)
        assert len(content) <= MessageHistory.MAX_RENDER_CHARS      # a system row is cut there, tail included
        head, *_, tail = content.splitlines()
        assert "400 added" in head
        assert tail.startswith("… and ") and "more changed paragraph" in tail
        shown = sum(1 for line in content.splitlines() if line.startswith("+ "))
        assert shown + int(tail.split()[2]) == 400

    def test_an_item_inserted_in_a_numbered_list_is_one_addition_not_a_renumbering_of_the_rest(self):
        old = "Warm-up\n1. Run 300m\n2. 15 squats\n3. 15 push-ups\nCool-down\n* stretch"
        new = "Warm-up\n1. Row 500m\n2. Run 300m\n3. 15 squats\n4. 15 push-ups\nCool-down\n- stretch"
        content = describe_change("Plan", "Dave", old, new)
        lines = content.splitlines()
        assert lines[0].endswith(": 1 added paragraph(s)")
        assert lines[1:] == ["  Warm-up", "+ 1. Row 500m"]

    def test_the_tracking_note_counts_paragraphs_and_words(self):
        note = tracking_note("Plan", "One two three\n\nFour five\n")
        assert "2 paragraphs" in note and "5 words" in note and "Plan" in note


@pytest.mark.unit
class TestScopes:
    def test_the_drive_scope_is_asked_for_only_when_docs_are_listed_a_contract_with_google_oauth(self):
        assert SCOPE_DRIVE_READONLY not in account_scopes({"name": "Personal"})
        assert account_scopes({"name": "Personal", "mail": False, "calendar": "off", "docs": [{"id": DOC_ID}]}) == [SCOPE_DRIVE_READONLY]
        assert account_scopes({"name": "Personal", "docs": [{"name": "no id"}]}) == [SCOPE_MAIL, SCOPE_CALENDAR]

    def test_an_account_with_docs_alone_is_kept_by_the_google_integration_for_its_sign_in(self, fresh):
        GoogleIntegration.reset_instance()
        with _config([{"name": "Personal", "mail": False, "calendar": False, "docs": [{"id": DOC_ID}]}]):
            personal, = GoogleIntegration().accounts()
            assert personal.mail is None and personal.calendar is None
            assert personal.auth._scopes() == [SCOPE_DRIVE_READONLY]
        GoogleIntegration.reset_instance()


@pytest.mark.unit
class TestOpen:
    def test_without_docs_the_feed_does_not_open(self, fresh):
        with _config(_entries(docs=())):
            feed = GoogleDocsIntegration()
            assert feed.open_feed() is False and "docs" in feed.open_error
            assert feed.accounts_status() == []

    def test_a_token_without_the_drive_scope_is_reported_before_any_call(self, fresh):
        _signed_in(fresh, scope=f"{SCOPE_MAIL} {SCOPE_CALENDAR}")
        drive = FakeDrive(HORIZON + DAY, [], {})
        with _config(_entries()), patch(REQUEST, drive):
            feed = GoogleDocsIntegration()
            assert feed.open_feed() is False
            assert "Drive scope" in feed.open_error and 'google_login("Personal")' in feed.open_error
            assert drive.calls == []
            status, = feed.accounts_status()
            assert status["state"] == "error" and "Drive scope" in status["last_error"]
            with pytest.raises(RuntimeError):
                feed.check_health()

    def test_a_file_that_is_not_a_doc_is_left_out_and_the_title_fills_a_missing_name(self, fresh):
        _signed_in(fresh)
        drive = FakeDrive(HORIZON + DAY, [], {}, mime="application/vnd.google-apps.spreadsheet")
        with _config(_entries(docs=({"id": DOC_ID},))), patch(REQUEST, drive):
            feed = GoogleDocsIntegration()
            assert feed.open_feed() is False and "not a Google Doc" in feed.open_error
        drive = FakeDrive(HORIZON + DAY, [], {})
        GoogleDocsIntegration.reset_instance()
        with _config(_entries(docs=({"id": DOC_ID},))), patch(REQUEST, drive):
            feed = GoogleDocsIntegration()
            assert feed.open_feed() is True and feed.check_health() is True
            doc, = feed.docs()
            assert doc.name == "Family plan 2027"
            assert drive.sent("GET", f"/files/{DOC_ID}")[0].params == {"fields": "name,mimeType,modifiedTime"}


@pytest.mark.unit
class TestFetch:
    def _feed(self, drive):
        feed = GoogleDocsIntegration()
        assert feed.open_feed() is True
        return feed

    def test_two_days_of_revisions_are_two_steps_each_fetch_does_one_and_counts_the_rest(self, fresh):
        _signed_in(fresh)
        texts = {"1": "A\nThe plan is set\nC", "2": "A\nThe plan is not set\nC", "3": "A\nThe plan is not set\nC\nD"}
        revisions = [_revision(1, HORIZON - DAY, "Marcus"), _revision(2, HORIZON + DAY, "Charlotte"), _revision(3, HORIZON + 2 * DAY, "Frances")]
        drive = FakeDrive(HORIZON + 2 * DAY, revisions, texts)
        with _config(_entries()), patch(REQUEST, drive):
            feed = self._feed(drive)
            cursor = feed.initial_cursor(HORIZON)
            assert feed.remaining(cursor) == 2
            first = feed.fetch(cursor, backfill=True)
            entry, = first.entries
            assert (entry.timestamp, entry.location) == (HORIZON + DAY, "Family plan")
            assert "Charlotte" in entry.content and "~ The plan is {+not+} set" in entry.content.splitlines()
            assert not first.exhausted and first.remaining == 1
            assert json.loads(first.cursor) == {DOC_ID: HORIZON + DAY}
            # the revision before the horizon was the baseline: exported once, never a row
            assert [c.path.rsplit("revision=", 1)[1][:1] for c in drive.calls if "revision=" in c.path] == ["1", "2"]
            second = feed.fetch(first.cursor, backfill=True)
            entry, = second.entries
            assert "Frances" in entry.content and "+ D" in entry.content.splitlines() and entry.timestamp == HORIZON + 2 * DAY
            assert second.exhausted and second.remaining == 0 and json.loads(second.cursor) == {DOC_ID: HORIZON + 2 * DAY}
            assert feed.docs()[0].cached_text() == texts["3"]
            assert len(drive.sent("GET", f"/files/{DOC_ID}/revisions")) == 1      # listed once for the head seen

    def test_the_revisions_of_one_day_fold_into_one_row_naming_every_author_with_one_export(self, fresh):
        _signed_in(fresh)
        noon = HORIZON + DAY + 12 * 3600
        texts = {"1": "A\nB", "2": "A\nB\nC", "3": "A\nB\nC\nD", "4": "A\nB\nC\nD\nE"}
        revisions = [_revision(1, HORIZON - DAY, "Marcus"), _revision(2, noon, "Charlotte"),
                     _revision(3, noon + 600, "Marcus"), _revision(4, noon + 1200, "Charlotte")]
        drive = FakeDrive(noon + 1200, revisions, texts)
        with _config(_entries()), patch(REQUEST, drive):
            feed = self._feed(drive)
            batch = feed.fetch(None, backfill=True)
            entry, = batch.entries
            head = entry.content.splitlines()[0]
            assert "Charlotte and Marcus" in head and "(3 edits)" in head and "3 added" in head
            assert entry.timestamp == noon + 1200 and batch.exhausted
            assert [c.path.rsplit("revision=", 1)[1][:1] for c in drive.calls if "revision=" in c.path] == ["1", "4"]

    def test_a_doc_edited_within_the_last_poll_interval_is_held_until_the_session_is_over(self, fresh):
        _signed_in(fresh)
        now = time.time()
        drive = FakeDrive(now - 60, [_revision(1, now - 7200), _revision(2, now - 60)], {"1": "A", "2": "A\nB"})
        with _config(_entries()), patch(REQUEST, drive):
            feed = self._feed(drive)
            feed.docs()[0].cache("A")
            interval = feed.poll_interval   # the patched config's, which fetch() reads too — not config.yaml's
            cursor = json.dumps({DOC_ID: now - 7200})
            assert feed.remaining(cursor) == 0
            batch = feed.fetch(cursor, backfill=False)
            assert batch.entries == [] and batch.exhausted and json.loads(batch.cursor) == json.loads(cursor)
            assert not [c for c in drive.calls if "revision=" in c.path]
        drive = FakeDrive(now - 2 * interval, [_revision(1, now - 7200), _revision(2, now - 2 * interval)], {"1": "A", "2": "A\nB"})
        with _config(_entries()), patch(REQUEST, drive):
            batch = feed.fetch(cursor, backfill=False)
            assert len(batch.entries) == 1

    def test_an_unchanged_head_makes_no_revisions_call_and_keeps_the_cursor(self, fresh):
        _signed_in(fresh)
        drive = FakeDrive(HORIZON + DAY, [_revision(1, HORIZON + DAY)], {"1": "A"})
        cursor = json.dumps({DOC_ID: HORIZON + DAY})
        with _config(_entries()), patch(REQUEST, drive):
            feed = self._feed(drive)
            batch = feed.fetch(cursor, backfill=False)
            assert batch.entries == [] and json.loads(batch.cursor) == json.loads(cursor)
            assert drive.sent("GET", f"/files/{DOC_ID}/revisions") == []

    def test_a_doc_first_seen_without_an_earlier_revision_gets_a_tracking_note_not_a_whole_document_diff(self, fresh):
        _signed_in(fresh)
        texts = {"1": "One\nTwo\nThree", "2": "One\nTwo\nThree\nFour"}
        drive = FakeDrive(HORIZON + 2 * DAY, [_revision(1, HORIZON + DAY), _revision(2, HORIZON + 2 * DAY)], texts)
        with _config(_entries()), patch(REQUEST, drive):
            feed = self._feed(drive)
            batch = feed.fetch(None, backfill=True)
            note, = batch.entries
            assert "now tracked" in note.content and "3 paragraphs" in note.content and note.timestamp == HORIZON + DAY
            assert not batch.exhausted and batch.remaining == 1 and json.loads(batch.cursor) == {DOC_ID: HORIZON + DAY}
            change, = feed.fetch(batch.cursor, backfill=True).entries
            assert "+ Four" in change.content.splitlines() and "- " not in change.content

    def test_a_revision_that_changed_no_paragraph_writes_no_row_but_moves_the_cursor(self, fresh):
        _signed_in(fresh)
        drive = FakeDrive(HORIZON + DAY, [_revision(1, HORIZON + DAY)], {"1": "Same\ntext"})
        with _config(_entries()), patch(REQUEST, drive):
            feed = self._feed(drive)
            feed.docs()[0].cache("Same\ntext")
            batch = feed.fetch(None, backfill=False)
            assert batch.entries == [] and json.loads(batch.cursor) == {DOC_ID: HORIZON + DAY}

    def test_the_poll_interval_comes_from_feeds_google_docs_poll_seconds_a_contract_with_config_yaml(self, fresh):
        with _config(_entries(), **{"feeds.google_docs.poll_seconds": 30}):
            feed = GoogleDocsIntegration()
            assert feed.poll_interval == 30
        with _config(_entries(), **{"feeds.google_docs.poll_seconds": "soon"}):
            assert feed.poll_interval == 600

    def test_the_cursor_map_is_reached_only_when_every_doc_is_at_or_past_the_target(self, fresh):
        with _config(_entries()):
            feed = GoogleDocsIntegration()
            target = json.dumps({"a": 10.0, "b": 20.0})
            assert feed.cursor_reached(json.dumps({"a": 10.0, "b": 25.0}), target)
            assert not feed.cursor_reached(json.dumps({"a": 10.0}), target)
            assert not feed.cursor_reached(None, target) and feed.cursor_reached(None, None)
            assert feed.cursor_reached("garbage", "{}")


@pytest.mark.unit
class TestDocsOf:
    def test_the_docs_an_account_lists_are_kept_by_their_id_and_the_rest_logged_and_left_out(self):
        """The config is hand-written: an id with stray spaces is still the id, a doc without
        one or an entry that is no mapping is skipped on its own, and a `docs` that is not a
        list at all yields no documents instead of a crash on the whole account."""
        from integrations.google.GoogleIntegration import docs_of
        entry = {"name": "Personal", "docs": [{"name": "Family plan", "id": f" {DOC_ID} "}, {"name": "no id yet"},
                                               "just-a-string", {"id": "2ZyXwVuTsRqPoNmLkJiHgFeDcBa"}]}
        assert docs_of(entry) == [{"id": DOC_ID, "name": "Family plan"}, {"id": "2ZyXwVuTsRqPoNmLkJiHgFeDcBa", "name": ""}]
        assert docs_of({"name": "Personal", "docs": "Family plan"}) == []
        assert docs_of({"name": "Personal"}) == []
