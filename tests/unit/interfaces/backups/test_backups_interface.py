"""Tests for the backup web page: it makes, lists, downloads and restores the backups of
assistant/Backups.py, serves only the backup directory's own files, takes an uploaded backup
file, and restores nothing before the owner approved it in the chat. A real server on a
loopback port over a tmp_path install."""
import threading
import urllib.error
import urllib.parse
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from assistant import Backups, Notifier
from common import Backup, ConfigStore
from common.ConfigManager import ConfigManager
from interfaces.backups import BackupsInterface as page_module

PASSPHRASE = "correct horse battery"


@pytest.fixture
def install(tmp_path, monkeypatch):
    root = tmp_path / "Keep"
    (root / "memory").mkdir(parents=True)
    (root / "config.yaml").write_text("owner_name: Marcus\n")
    (root / "memory" / "bank.md").write_text("Marcus likes tea.\n")
    values = {Backups.PASSPHRASE_KEY: PASSPHRASE, Backups.DIRECTORY_KEY: str(tmp_path / "KeepBackups"),
              Backups.KEEP_KEY: 5}
    config = ConfigManager.get_instance()
    real_get = config.get
    monkeypatch.setattr(config, "get", lambda key, *a, **kw: values[key] if key in values else real_get(key, *a, **kw))
    monkeypatch.setattr(Backups, "AISERVER_DIR", root)
    monkeypatch.setattr(Backups, "targets", lambda: [root / "config.yaml", root / "memory"])
    return root


class Approver:
    """The chat: answers every approval with *decision*, once *release* is set."""

    def __init__(self, decision="approve"):
        self.decision = decision
        self.release = threading.Event()
        self.release.set()
        self.asked = []

    def request_approval(self, title, details="", **kw):
        self.asked.append(details)
        self.release.wait(5)
        return self.decision


@pytest.fixture
def chat():
    approver = Approver()
    Notifier.set_notifier(approver)
    yield approver
    Notifier.set_notifier(None)


@pytest.fixture
def server(install, chat, monkeypatch):
    restarts = []
    monkeypatch.setattr(page_module, "restart_after_restore", restarts.append)
    monkeypatch.setattr(page_module, "_restore", None)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), page_module._Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", restarts
    httpd.shutdown()
    httpd.server_close()


def _get(url):
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def _post(url, **fields):
    data = urllib.parse.urlencode(fields).encode()
    try:
        with urllib.request.urlopen(url, data=data, timeout=5) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def test_back_up_now_writes_a_backup_the_page_then_lists(server):
    url, _ = server
    status, _ = _post(f"{url}/backup")
    [made] = Backups.backups()
    assert status == 200
    assert made.name in _get(url + "/")[1].decode()


def test_a_download_is_the_backup_file_itself(server):
    url, _ = server
    Backups.create()
    [made] = Backups.backups()
    status, body = _get(f"{url}/download/{made.name}")
    assert status == 200 and body == made.read_bytes()


@pytest.mark.parametrize("name", ["..%2FKeep%2Fconfig.yaml", "%2Fetc%2Fpasswd", "nothing.keepbackup"])
def test_a_download_of_anything_but_a_listed_backup_is_refused(server, name):
    url, _ = server
    Backups.create()
    status, body = _get(f"{url}/download/{name}")
    assert status == 404 and b"Marcus" not in body


def _upload(url, data: bytes, filename="keep-20261118-120000.keepbackup"):
    boundary = "----keeptestboundary"
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"backup\"; filename=\"{filename}\"\r\n"
            f"Content-Type: application/octet-stream\r\n\r\n").encode() + data + f"\r\n--{boundary}--\r\n".encode()
    request = urllib.request.Request(f"{url}/upload", data=body,
                                     headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    try:
        with urllib.request.urlopen(request, timeout=5) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def _settled():
    for _ in range(300):
        job = page_module.current()
        if job is None or job.state not in (page_module.WAITING,):
            return job
        threading.Event().wait(0.01)
    raise AssertionError("the restore never settled")


def _made_elsewhere(tmp_path, passphrase):
    other = tmp_path / "Other"
    other.mkdir()
    (other / "config.yaml").write_text("owner_name: Alice\n")
    return Backup.write(tmp_path / "other.keepbackup", [other / "config.yaml"], passphrase, other).path


def test_the_restore_section_and_its_upload_are_there_without_any_backup(server):
    url, _ = server
    body = _get(url + "/")[1].decode()
    assert 'action="/upload"' in body and 'type="file"' in body


def test_every_listed_backup_has_a_restore_button(server):
    url, _ = server
    Backups.create()
    [made] = Backups.backups()
    body = _get(url + "/")[1].decode()
    assert 'action="/restore"' in body and f'value="{made.name}"' in body


def test_an_approved_restore_of_a_listed_backup_stages_it_and_restarts(server, install, chat):
    url, restarts = server
    Backups.create()
    [made] = Backups.backups()
    status, _ = _post(f"{url}/restore", name=made.name)
    job = _settled()
    assert status == 200 and job.state == page_module.DONE
    assert (install / Backup.STAGING).is_dir() and restarts == [made.name]
    assert made.name in chat.asked[0]


@pytest.mark.parametrize("decision", ["deny", "timeout", "unavailable"])
def test_a_restore_not_approved_stages_nothing_and_restarts_nothing(server, install, chat, decision):
    url, restarts = server
    chat.decision = decision
    Backups.create()
    [made] = Backups.backups()
    _post(f"{url}/restore", name=made.name)
    job = _settled()
    assert job.state == page_module.REFUSED and restarts == []
    assert not (install / Backup.STAGING).exists()


def test_nothing_is_staged_while_the_approval_is_pending_and_the_page_says_so(server, install, chat):
    url, restarts = server
    chat.release.clear()
    Backups.create()
    [made] = Backups.backups()
    _post(f"{url}/restore", name=made.name)
    try:
        for _ in range(300):
            if chat.asked:
                break
            threading.Event().wait(0.01)
        body = _get(url + "/")[1].decode()
        assert chat.asked and 'http-equiv="refresh"' in body
        assert not (install / Backup.STAGING).exists() and restarts == []
        status, _ = _post(f"{url}/restore", name=made.name)
        assert status == 409 and len(chat.asked) == 1
    finally:
        chat.release.set()
    assert _settled().state == page_module.DONE


def test_an_uploaded_backup_is_restored_once_approved_and_the_upload_removed(server, install, chat, tmp_path):
    url, restarts = server
    Backups.create()
    [made] = Backups.backups()
    status, _ = _upload(url, made.read_bytes(), filename="from-laptop.keepbackup")
    job = _settled()
    assert status == 200 and job.state == page_module.DONE and restarts == ["from-laptop.keepbackup"]
    assert (install / Backup.STAGING).is_dir()
    assert not any(page_module.upload_dir().iterdir())


def test_an_upload_made_with_another_passphrase_asks_for_it_and_opens_with_it(server, install, chat, tmp_path):
    url, restarts = server
    other = _made_elsewhere(tmp_path, "another long passphrase")
    _upload(url, other.read_bytes())
    assert page_module.current().state == page_module.NEEDS_PASSPHRASE and chat.asked == []
    assert 'action="/passphrase"' in _get(url + "/")[1].decode()
    _post(f"{url}/passphrase", passphrase="a wrong guess")
    assert page_module.current().state == page_module.NEEDS_PASSPHRASE and chat.asked == []
    _post(f"{url}/passphrase", passphrase="another long passphrase")
    assert _settled().state == page_module.DONE and len(chat.asked) == 1
    staged = install / Backup.STAGING / "files" / "config.yaml"
    assert "Alice" in staged.read_text()


def test_an_upload_that_is_no_backup_fails_without_asking_the_chat(server, install, chat):
    url, restarts = server
    _upload(url, b"this is not a backup at all")
    job = page_module.current()
    assert job.state == page_module.FAILED and chat.asked == [] and restarts == []
    assert not (install / Backup.STAGING).exists()


def test_a_cancelled_upload_is_deleted(server, chat, tmp_path):
    url, _ = server
    _upload(url, _made_elsewhere(tmp_path, "another long passphrase").read_bytes())
    uploaded = page_module.current().path
    _post(f"{url}/cancel")
    assert page_module.current() is None and not uploaded.exists()


def test_the_assistant_can_neither_open_the_page_nor_move_it():
    assert ConfigStore.is_denied("interfaces.backups.host")
    assert ConfigStore.is_denied("interfaces.backups.enabled")
