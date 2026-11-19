"""Tests for assistant/Backups.py: what a backup holds, where the backups go and how a restore is prepared.

The install and the backup folder are tmp_path trees."""
import os
import zipfile
from datetime import datetime, timedelta
from itertools import count

import pytest

from assistant import Backups
from common import Backup
from common.Paths import AISERVER_DIR, CONFIG_PATH, MEMORY_DIR, VAULT_PATH


@pytest.fixture
def install(tmp_path, monkeypatch):
    root = tmp_path / "Keep"
    (root / "memory").mkdir(parents=True)
    (root / "config.yaml").write_text("owner_name: Marcus\n")
    (root / "memory" / "bank.md").write_text("Marcus likes tea.\n")
    state = [root / "config.yaml", root / "memory"]
    seconds = count()

    class Clock(datetime):
        """Every backup is made a second after the one before: its name is its time."""
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 11, 1, 12, 0, 0) + timedelta(seconds=next(seconds))

    monkeypatch.setattr(Backups, "datetime", Clock)
    monkeypatch.setattr(Backups, "DIRECTORY", tmp_path / "KeepBackups")
    monkeypatch.setattr(Backups, "state", lambda: state)
    monkeypatch.setattr(Backups.Backup, "write", lambda path, paths, real=Backup.write: real(path, paths, root))
    monkeypatch.setattr(Backups.Backup, "stage", lambda path, paths, real=Backup.stage: real(path, paths, root))
    return root


def aged(path, seconds_ago):
    stamp = 1_700_000_000 - seconds_ago
    os.utime(path, (stamp, stamp))


def test_a_backup_lands_in_the_backup_folder_and_holds_the_state(install):
    answer = Backups.create()
    [made] = Backups.backups()
    assert made.parent == Backups.DIRECTORY and str(made) in answer
    with zipfile.ZipFile(made) as zf:
        assert sorted(zf.namelist()) == ["config.yaml", "memory/bank.md"]


def test_only_the_newest_backups_are_kept(install, monkeypatch):
    monkeypatch.setattr(Backups, "KEPT", 2)
    Backups.DIRECTORY.mkdir()
    for name, age in (("keep-backup-recent.zip", 100), ("keep-backup-old.zip", 200)):
        (Backups.DIRECTORY / name).write_bytes(b"")
        aged(Backups.DIRECTORY / name, age)
    (Backups.DIRECTORY / "holiday.zip").write_bytes(b"")
    aged(Backups.DIRECTORY / "holiday.zip", 300)
    Backups.create()
    names = [p.name for p in Backups.backups()]
    assert len(names) == 2 and names[1] == "keep-backup-recent.zip"
    assert (Backups.DIRECTORY / "holiday.zip").exists()     # not a backup of ours, never deleted


def test_a_failed_backup_is_an_answer_not_an_exception(install, monkeypatch):
    def disk_full(path, paths):
        raise OSError("disk full")

    monkeypatch.setattr(Backups.Backup, "write", disk_full)
    assert "disk full" in Backups.create()


def test_the_listing_names_every_backup_and_says_when_there_is_none(install):
    assert str(Backups.DIRECTORY) in Backups.listing()
    Backups.create()
    assert Backups.backups()[0].name in Backups.listing()


def test_a_backup_is_named_by_its_name_in_the_folder_or_by_a_path(install, tmp_path):
    Backups.create()
    [made] = Backups.backups()
    elsewhere = tmp_path / "Downloads" / "from-the-old-mac.zip"
    elsewhere.parent.mkdir()
    elsewhere.write_bytes(made.read_bytes())
    assert Backups.resolve(made.name) == made
    assert Backups.resolve(str(elsewhere)) == elsewhere
    assert Backups.resolve("keep-backup-19990101-000000.zip") is None
    assert Backups.resolve("") is None


def test_a_restore_backs_the_current_state_up_first_and_stages_the_zip(install):
    Backups.create()
    [made] = Backups.backups()
    (install / "config.yaml").write_text("owner_name: Frances\n")
    assert Backups.restore(made) == ""
    undo = next(p for p in Backups.backups() if p != made)
    with zipfile.ZipFile(undo) as zf:
        assert zf.read("config.yaml") == b"owner_name: Frances\n"
    assert made.name in Backup.apply_pending(install)
    assert (install / "config.yaml").read_text() == "owner_name: Marcus\n"


def test_restoring_the_undo_backup_brings_back_what_the_restore_replaced(install):
    Backups.create()
    [made] = Backups.backups()
    (install / "config.yaml").write_text("owner_name: Frances\n")
    Backups.restore(made)
    Backup.apply_pending(install)
    undo = next(p for p in Backups.backups() if p != made)
    assert Backups.restore(undo) == ""
    Backup.apply_pending(install)
    assert (install / "config.yaml").read_text() == "owner_name: Frances\n"


def test_a_file_that_is_no_backup_stages_nothing_and_leaves_no_undo_backup(install, tmp_path):
    other = tmp_path / "holiday.zip"
    other.write_text("hello")
    assert "holiday.zip" in Backups.restore(other)
    assert Backups.backups() == []
    assert Backup.apply_pending(install) is None


def test_no_restore_without_the_undo_backup(install, monkeypatch):
    Backups.create()
    [made] = Backups.backups()

    def disk_full(path, paths):
        raise OSError("disk full")

    monkeypatch.setattr(Backups.Backup, "write", disk_full)
    assert "disk full" in Backups.restore(made)
    assert Backup.apply_pending(install) is None


def test_the_state_is_inside_the_install_and_holds_what_makes_the_assistant():
    """The real list: config, vault, memory and history are what a restored install is the same
    assistant by. The restore's own marker is no part of it, or applying would delete it mid-way."""
    state = Backups.state()
    assert {CONFIG_PATH, VAULT_PATH, MEMORY_DIR, AISERVER_DIR / "floxy.db"} <= set(state)
    marker = AISERVER_DIR / Backup.PENDING
    for path in state:
        assert path.is_relative_to(AISERVER_DIR)
        assert not marker.is_relative_to(path)
        assert path.relative_to(AISERVER_DIR).parts[0] not in ("src", "scripts", "res", "tests", ".git", ".venv")
