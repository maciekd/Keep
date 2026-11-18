"""Tests for the backup web page: it makes, lists, downloads and restores the backups of
assistant/Backups.py, serves only the backup directory's own files and restores only with the
backup passphrase. A real server on a loopback port over a tmp_path install."""
import threading
import urllib.error
import urllib.parse
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from assistant import Backups
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


@pytest.fixture
def server(install, monkeypatch):
    restarts = []
    monkeypatch.setattr(page_module, "restart_after_restore", restarts.append)
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


def test_a_restore_with_the_wrong_passphrase_stages_nothing_and_restarts_nothing(server, install):
    url, restarts = server
    Backups.create()
    [made] = Backups.backups()
    status, _ = _post(f"{url}/restore", name=made.name, passphrase="guess")
    assert status == 403 and restarts == []
    assert not (install / Backup.STAGING).exists()


def test_a_restore_with_the_passphrase_stages_the_backup_and_restarts(server, install):
    url, restarts = server
    Backups.create()
    [made] = Backups.backups()
    status, _ = _post(f"{url}/restore", name=made.name, passphrase=PASSPHRASE)
    assert status == 200
    assert (install / Backup.STAGING).is_dir()
    for _ in range(100):
        if restarts:
            break
        threading.Event().wait(0.01)
    assert restarts == [made]


def test_the_assistant_can_neither_open_the_page_nor_move_it():
    assert ConfigStore.is_denied("interfaces.backups.host")
    assert ConfigStore.is_denied("interfaces.backups.enabled")
