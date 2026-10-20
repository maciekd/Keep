"""DropboxClient: the calls as Dropbox expects them (JSON on the api host, the argument header
on the content host), the shapes parsed out of the answers, and the errors — 401 as a sign-in
problem, 409 as a sentence, 429 and 5xx retried once."""
from unittest.mock import patch

import pytest
import requests

from integrations.dropbox import DropboxClient as module
from integrations.dropbox.DropboxAuth import NotSignedIn
from integrations.dropbox.DropboxClient import DropboxClient, DropboxError

from .dropbox_fakes import FakeDropbox, FakeResponse, StubAuth, account, file_metadata, folder_metadata, search_match


@pytest.fixture(autouse=True)
def no_sleep():
    with patch.object(module.time, "sleep"):
        yield


def client(fake: FakeDropbox, auth=None) -> DropboxClient:
    return DropboxClient(auth or StubAuth(), session=fake)


@pytest.mark.unit
class TestCalls:
    def test_an_rpc_call_posts_json_with_the_bearer_token(self):
        fake = FakeDropbox()
        fake.route("/users/get_current_account", account("pro"))

        got = client(fake).current_account()

        call = fake.calls[0]
        assert call.method == "POST" and call.headers["Authorization"] == "Bearer at.stub"
        assert call.headers["Content-Type"] == "application/json" and call.body is None, "no-argument endpoints get a JSON null"
        assert got.account_type == "pro" and got.email == "marcus@example.com" and got.display_name == "Marcus"

    def test_a_search_scopes_to_the_path_asks_for_active_files_and_highlights(self):
        fake = FakeDropbox()
        fake.route("/files/search_v2", {"matches": [
            search_match(file_metadata("/Household/Lease 2024.pdf", size=1_258_291), "content", "deposit"),
            search_match(folder_metadata("/Household/Invoices")),
        ], "has_more": True})

        matches, has_more = client(fake).search("deposit", "/Household")

        body = fake.calls[0].body
        assert body["query"] == "deposit" and body["options"]["path"] == "/Household"
        assert body["options"]["file_status"] == "active" and body["options"]["filename_only"] is False
        assert body["match_field_options"]["include_highlights"] is True
        assert has_more
        lease, invoices = matches
        assert lease.entry.path == "/Household/Lease 2024.pdf" and lease.entry.size == 1_258_291
        assert lease.entry.modified_date == "2024-03-02" and lease.match_type == "content"
        assert lease.highlight == "the deposit of two months", "the whole passage around the match, not the word alone"
        assert invoices.entry.is_folder and invoices.highlight == ""

    def test_a_search_of_the_whole_dropbox_sends_no_path(self):
        fake = FakeDropbox()
        fake.route("/files/search_v2", {"matches": [], "has_more": False})
        assert client(fake).search("x") == ([], False)
        assert "path" not in fake.calls[0].body["options"]

    def test_a_listing_names_the_root_as_dropbox_wants_it_and_puts_folders_first(self):
        fake = FakeDropbox()
        fake.route("/files/list_folder", {"entries": [file_metadata("/b.txt"), folder_metadata("/Zed"), file_metadata("/A.md")],
                                          "has_more": False})

        entries, has_more = client(fake).list_folder("/")

        assert fake.calls[0].body == {"path": "", "limit": module.LIST_LIMIT}
        assert [e.name for e in entries] == ["Zed", "A.md", "b.txt"] and not has_more

    def test_a_download_carries_the_path_in_the_argument_header_and_returns_the_bytes(self):
        fake = FakeDropbox()
        fake.route("/files/download", raw=b"%PDF-1.4 ...")

        assert client(fake).download("/Household/Lease 2024.pdf") == b"%PDF-1.4 ..."

        call = fake.calls[0]
        assert call.body == {"path": "/Household/Lease 2024.pdf"}
        assert "Content-Type" not in call.headers, "a content call has no JSON body"

    def test_an_export_names_the_format(self):
        fake = FakeDropbox()
        fake.route("/files/export", raw=b"# Notes\n")
        assert client(fake).export("/Notes.paper") == b"# Notes\n"
        assert fake.calls[0].body == {"path": "/Notes.paper", "export_format": "markdown"}

    def test_metadata_gives_the_hash_the_cache_is_keyed_by(self):
        fake = FakeDropbox()
        fake.route("/files/get_metadata", file_metadata("/a.txt", size=12, content_hash="f" * 64))
        entry = client(fake).metadata("/a.txt")
        assert entry.content_hash == "f" * 64 and entry.size == 12 and not entry.is_folder


@pytest.mark.unit
class TestErrors:
    def test_401_is_a_sign_in_problem(self):
        fake = FakeDropbox()
        fake.route("/files/get_metadata", {"error_summary": "invalid_access_token/", "error": {".tag": "invalid_access_token"}}, status=401)
        with pytest.raises(NotSignedIn, match="dropbox_login"):
            client(fake).metadata("/a.txt")

    def test_a_not_signed_in_auth_never_reaches_the_network(self):
        fake = FakeDropbox()
        with pytest.raises(NotSignedIn):
            client(fake, StubAuth(signed_in=False)).metadata("/a.txt")
        assert not fake.calls

    def test_409_becomes_a_sentence_about_the_path(self):
        fake = FakeDropbox()
        fake.route("/files/get_metadata", {"error_summary": "path/not_found/...", "error": {".tag": "path"}}, status=409)
        with pytest.raises(DropboxError) as caught:
            client(fake).metadata("/gone.txt")
        assert caught.value.status == 409 and "nothing at that path" in str(caught.value)

    def test_a_missing_scope_400_names_the_scope_and_carries_dropboxs_whole_explanation(self):
        fake = FakeDropbox()
        explanation = ('Error in call to API function "files/search_v2": Your app (ID: 1234567) is not permitted to access '
                       "this endpoint because it does not have the required scope 'files.metadata.read'. The owner of the "
                       "app can enable the scope for the app using the Permissions tab on the App Console.")
        fake.route("/files/search_v2", raw=explanation.encode(), status=400)
        with pytest.raises(DropboxError) as caught:
            client(fake).search("chicken")
        message = str(caught.value)
        assert "files.metadata.read" in message and "Permissions" in message and "dropbox_login()" in message
        assert message.endswith(explanation), "the cause shows without the server log"

    def test_an_unknown_error_is_given_whole_not_cut_at_a_slash(self):
        fake = FakeDropbox()
        fake.route("/files/search_v2", raw=b'Error in call to API function "files/search_v2": request body: unknown option', status=400)
        with pytest.raises(DropboxError, match="unknown option"):
            client(fake).search("chicken")

    def test_a_folder_downloaded_is_told_as_such(self):
        fake = FakeDropbox()
        fake.route("/files/download", {"error_summary": "path/not_file/..", "error": {".tag": "path"}}, status=409)
        with pytest.raises(DropboxError, match="folder, not a file"):
            client(fake).download("/Household")

    def test_a_rate_limit_is_retried_once_after_the_wait_dropbox_names(self):
        fake = FakeDropbox()
        fake.routes["/users/get_current_account"] = [
            FakeResponse(429, {"error_summary": "too_many_requests/..", "error": {"retry_after": 3}}, headers={"Retry-After": "3"}),
            FakeResponse(200, account()),
        ]

        got = client(fake).current_account()

        assert got.account_type == "basic" and len(fake.calls) == 2
        module.time.sleep.assert_called_once_with(3)

    def test_a_second_rate_limit_is_an_error(self):
        fake = FakeDropbox()
        fake.route_answers("/users/get_current_account", (429, {"error_summary": "too_many_requests/.."}))
        with pytest.raises(DropboxError, match="rate-limiting"):
            client(fake).current_account()
        assert len(fake.calls) == 2

    def test_a_server_error_is_retried_once_then_reported(self):
        fake = FakeDropbox()
        fake.route_answers("/users/get_current_account", (503, None), (200, account("business")))
        assert client(fake).current_account().account_type == "business"

        fake = FakeDropbox()
        fake.route_answers("/users/get_current_account", (503, None))
        with pytest.raises(DropboxError, match="503"):
            client(fake).current_account()

    def test_an_unreachable_host_is_a_dropbox_error_not_a_bare_exception(self):
        fake = FakeDropbox()
        fake.transport_errors.append(requests.ConnectionError("down"))
        with pytest.raises(DropboxError, match="could not be reached"):
            client(fake).current_account()
