"""DropboxIntegration: the folders a read may touch, a document fetched once and served from
the cache by its content hash, what is refused before any download, the health check and the
sign-out."""
import time
from unittest.mock import MagicMock, patch

import pytest

from assistant.integration.Integration import Integration
from integrations.dropbox import DropboxAuth as auth_module
from integrations.dropbox import DropboxIntegration as module
from integrations.dropbox.DropboxAuth import NotSignedIn, token_path
from integrations.dropbox.DropboxIntegration import DOWNLOAD_MAX_BYTES, DropboxIntegration, human_size, normalise

from .dropbox_fakes import (FakeDropbox, StubAuth, account, file_metadata, folder_metadata, make_pdf, make_rtf, search_match,
                            token_file)

CONFIG = "integrations.dropbox.DropboxIntegration.ConfigManager.get_instance"


def _config(app_key="app-key", folders=None):
    values = {"dropbox.app_key": app_key, "dropbox.folders": folders or []}
    config = MagicMock()
    config.get.side_effect = lambda key, *default: values.get(key, default[0] if default else None)
    return patch(CONFIG, return_value=config)


@pytest.fixture(autouse=True)
def fresh(tmp_path):
    DropboxIntegration.reset_instance()
    with patch.object(auth_module, "TOKEN_DIR", tmp_path / "oauth"), patch.object(module, "TEXT_CACHE_DIR", tmp_path / "dropbox"):
        yield tmp_path
    DropboxIntegration.reset_instance()


def build(fake: FakeDropbox, auth=None) -> DropboxIntegration:
    integration = DropboxIntegration(session=fake)
    integration.auth = auth or StubAuth()
    integration.client.auth = integration.auth
    return integration


@pytest.mark.unit
class TestContract:
    def test_it_is_a_sensitive_integration_configured_by_the_app_key_alone(self):
        assert issubclass(DropboxIntegration, Integration) and DropboxIntegration.sensitive
        with _config():
            assert build(FakeDropbox()).config_keys == ["dropbox.app_key"], "a contract with config.yaml: the one key the sign-in needs"
        assert "dropbox_login" in DropboxIntegration.login_hint

    def test_the_wipe_targets_are_the_token_and_the_text_cache(self):
        paths = [path for path, _ in DropboxIntegration.wipe_targets()]
        assert token_path() in paths and module.TEXT_CACHE_DIR in paths


@pytest.mark.unit
class TestFolders:
    def test_paths_are_normalised_to_one_leading_slash(self):
        assert normalise("Household/") == "/Household" and normalise(' "/Household/Lease.pdf" ') == "/Household/Lease.pdf"
        assert normalise("/") == "" and normalise("") == ""

    def test_no_folders_means_everything_is_allowed(self):
        with _config():
            assert DropboxIntegration.refused("/Anything/at all.txt") == ""

    def test_a_path_under_a_listed_folder_is_allowed_case_aside_and_one_beside_it_refused(self):
        with _config(folders=["/Household", "Documents/Shared"]):
            assert DropboxIntegration.refused("/household/Lease.pdf") == ""
            assert DropboxIntegration.refused("/Documents/Shared") == ""
            refusal = DropboxIntegration.refused("/Householder/x.txt")
            assert "outside" in refusal and "/Household" in refusal and "/Documents/Shared" in refusal
            assert DropboxIntegration.refused("/") != ""

    def test_the_root_in_the_list_opens_everything(self):
        with _config(folders=["/"]):
            assert DropboxIntegration.folders() == [] and DropboxIntegration.refused("/x") == ""

    def test_a_search_without_a_folder_fans_out_over_the_roots(self):
        fake = FakeDropbox()
        fake.route("/files/search_v2", {"matches": [search_match(file_metadata("/Household/a.txt"))], "has_more": False})
        with _config(folders=["/Household", "/Work"]):
            matches, _ = build(fake).search("a")
        assert [c.body["options"]["path"] for c in fake.sent("/files/search_v2")] == ["/Household", "/Work"]
        assert len(matches) == 2

    def test_a_search_in_a_folder_outside_the_roots_is_refused_before_any_call(self):
        fake = FakeDropbox()
        with _config(folders=["/Household"]):
            with pytest.raises(ValueError, match="outside"):
                build(fake).search("a", "/Work")
        assert not fake.calls


@pytest.mark.unit
class TestDocument:
    PATH = "/Household/Lease 2024.pdf"

    def _lease(self, fake: FakeDropbox, content_hash="a" * 64):
        fake.route("/files/get_metadata", file_metadata(self.PATH, size=1_258_291, content_hash=content_hash))
        fake.route("/files/download", raw=make_pdf(["The deposit is two months rent", "Signed in Stockholm"]))

    def test_a_pdf_is_downloaded_extracted_and_described(self):
        fake = FakeDropbox()
        self._lease(fake)
        with _config():
            document = build(fake).document(self.PATH)
        assert document.text.startswith("[p. 1]\nThe deposit") and document.kind == "PDF" and not document.from_cache
        assert document.entry.size == 1_258_291 and document.entry.modified_date == "2024-03-02"

    def test_the_second_read_of_the_same_content_hash_downloads_nothing(self):
        fake = FakeDropbox()
        self._lease(fake)
        with _config():
            integration = build(fake)
            first = integration.document(self.PATH)
            second = integration.document(self.PATH)
        assert second.from_cache and second.text == first.text
        assert len(fake.sent("/files/download")) == 1 and len(fake.sent("/files/get_metadata")) == 2

    def test_an_edited_file_has_a_new_hash_and_is_fetched_again(self):
        fake = FakeDropbox()
        self._lease(fake)
        with _config():
            integration = build(fake)
            integration.document(self.PATH)
            self._lease(fake, content_hash="b" * 64)
            again = integration.document(self.PATH)
        assert not again.from_cache and len(fake.sent("/files/download")) == 2

    def test_a_folder_a_huge_file_and_an_unreadable_kind_are_refused_with_the_reason(self):
        fake = FakeDropbox()
        with _config():
            integration = build(fake)
            fake.route("/files/get_metadata", folder_metadata("/Household"))
            with pytest.raises(ValueError, match="folder"):
                integration.document("/Household")

            fake.route("/files/get_metadata", file_metadata("/big.pdf", size=DOWNLOAD_MAX_BYTES + 1))
            with pytest.raises(ValueError, match=human_size(DOWNLOAD_MAX_BYTES)):
                integration.document("/big.pdf")

            fake.route("/files/get_metadata", file_metadata("/photo.jpg", size=4000))
            fake.route("/files/download", raw=b"\xff\xd8\xff")
            with pytest.raises(ValueError, match=r"\.jpg"):
                integration.document("/photo.jpg")
        assert len(fake.sent("/files/download")) == 1, "the folder and the huge file never got downloaded"

    def test_a_paper_doc_is_exported_as_markdown(self):
        fake = FakeDropbox()
        fake.route("/files/get_metadata", {**file_metadata("/Notes.paper", size=0, content_hash=""), "is_downloadable": False})
        fake.route("/files/export", raw="# Notes\n\nFirst point\n".encode())
        with _config():
            document = build(fake).document("/Notes.paper")
        assert document.text.startswith("# Notes") and document.kind == "PAPER"
        assert not fake.sent("/files/download")

    def test_an_rtfd_bundle_is_read_through_the_txt_rtf_inside_it(self):
        fake = FakeDropbox()
        fake.route_answers("/files/get_metadata", (200, folder_metadata("/Recept/Kyckling.rtfd")),
                           (200, file_metadata("/Recept/Kyckling.rtfd/TXT.rtf", size=1758)))
        fake.route("/files/download", raw=make_rtf(["Friterad kyckling", "Smör och ägg"]))
        with _config():
            document = build(fake).document("/Recept/Kyckling.rtfd")
        assert document.text == "Friterad kyckling\nSmör och ägg" and document.kind == "RTF"
        assert [c.body["path"] for c in fake.sent("/files/get_metadata")] == ["/Recept/Kyckling.rtfd", "/Recept/Kyckling.rtfd/TXT.rtf"]

    def test_a_path_outside_the_folders_never_reaches_dropbox(self):
        fake = FakeDropbox()
        with _config(folders=["/Household"]):
            with pytest.raises(ValueError, match="outside"):
                build(fake).document("/Work/x.txt")
        assert not fake.calls

    def test_the_cache_is_trimmed_to_its_limit_oldest_first(self, tmp_path):
        fake = FakeDropbox()
        with _config(), patch.object(module, "CACHE_MAX_BYTES", 30):
            integration = build(fake)
            for number, name in enumerate(["old", "mid", "new"]):
                fake.route("/files/get_metadata", file_metadata(f"/{name}.txt", size=12, content_hash=name * 32))
                fake.route("/files/download", raw=b"x" * 12)
                integration.document(f"/{name}.txt")
                time.sleep(0.01)
        kept = sorted(p.name for p in (tmp_path / "dropbox").glob("*.txt"))
        assert kept == [f"{'mid' * 32}.txt", f"{'new' * 32}.txt"]


@pytest.mark.unit
class TestHealth:
    def test_not_signed_in_is_the_health_error_and_the_2fa_state(self):
        fake = FakeDropbox()
        with _config():
            integration = build(fake, StubAuth(signed_in=False))
            assert integration.needs_2fa()
            with pytest.raises(NotSignedIn, match="dropbox_login"):
                integration.check_health()
        assert not fake.calls

    def test_signed_in_the_account_answers_and_its_plan_is_kept(self):
        fake = FakeDropbox()
        fake.route("/users/get_current_account", account("basic"))
        with _config():
            integration = build(fake)
            assert integration.check_health() is True
            assert integration.account_type == "basic" and not integration.needs_2fa()

    def test_a_sign_in_short_of_a_scope_is_the_2fa_state_with_the_scope_in_the_hint(self):
        fake = FakeDropbox()
        fake.route("/users/get_current_account", account())
        with _config():
            integration = build(fake, StubAuth(missing=["files.metadata.read"]))
            assert integration.needs_2fa()
            with pytest.raises(NotSignedIn, match="files.metadata.read"):
                integration.check_health()
        assert "files.metadata.read" in integration.current_login_hint() and "Submit" in integration.current_login_hint()
        assert not fake.sent("/users/get_current_account"), "the account is not asked when the file scopes are known to be short"

    def test_a_refused_refresh_is_the_2fa_state_too(self):
        with _config():
            assert build(FakeDropbox(), StubAuth(again=True)).needs_2fa()


@pytest.mark.unit
class TestLogOut:
    def test_log_out_revokes_at_dropbox_and_deletes_the_file(self, fresh):
        auth_module.DropboxAuth(lambda: "k")._save(token_file())
        fake = FakeDropbox()
        fake.route("/auth/token/revoke", raw=b"")
        real_client = module.DropboxClient
        with _config(), patch.object(module, "DropboxClient", lambda auth, session=None: real_client(auth, fake)):
            outcome = DropboxIntegration.log_out()
        assert outcome and "revoked" in outcome and "deleted" in outcome
        assert not token_path().exists() and fake.sent("/auth/token/revoke")

    def test_log_out_with_nothing_signed_in_does_nothing(self):
        with _config():
            assert DropboxIntegration.log_out() is None

    def test_losing_the_config_deletes_the_token(self):
        auth_module.DropboxAuth(lambda: "k")._save(token_file())
        with _config():
            build(FakeDropbox()).on_not_configured()
        assert not token_path().exists()
