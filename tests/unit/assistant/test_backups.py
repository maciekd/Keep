"""Tests for assistant/Backups.py: what a backup holds, where it goes and the refusals the
assistant passes on. The install is a tmp_path tree; the config values are patched."""
import os
from pathlib import Path

import pytest

from assistant import Backups
from common import Backup
from common.ConfigManager import ConfigManager

PASSPHRASE = "correct horse battery"


@pytest.fixture
def install(tmp_path, monkeypatch):
    root = tmp_path / "Keep"
    (root / "memory").mkdir(parents=True)
    (root / "config.yaml").write_text("owner_name: Marcus\n")
    (root / "memory" / "bank.md").write_text("Marcus likes tea.\n")
    values = {Backups.PASSPHRASE_KEY: PASSPHRASE, Backups.DIRECTORY_KEY: str(tmp_path / "KeepBackups"),
              Backups.KEEP_KEY: 2}
    config = ConfigManager.get_instance()
    real_get = config.get
    monkeypatch.setattr(config, "get", lambda key, *a, **kw: values[key] if key in values else real_get(key, *a, **kw))
    monkeypatch.setattr(Backups, "AISERVER_DIR", root)
    monkeypatch.setattr(Backups, "targets", lambda: [root / "config.yaml", root / "memory"])
    return root, values


def test_a_backup_is_written_to_the_backup_folder_and_opens_with_the_passphrase(install):
    root, values = install
    answer = Backups.create()
    [made] = Backups.backups()
    assert made.parent == Path(values[Backups.DIRECTORY_KEY]) and made.name in answer
    assert Backup.manifest(made, PASSPHRASE)["targets"] == ["config.yaml", "memory"]


def test_without_a_passphrase_the_user_is_asked_for_one_and_nothing_is_written(install):
    _, values = install
    values[Backups.PASSPHRASE_KEY] = ""
    answer = Backups.create()
    assert f'config_request("{Backups.PASSPHRASE_KEY}")' in answer
    assert Backups.backups() == []


def test_a_short_passphrase_is_refused(install):
    _, values = install
    values[Backups.PASSPHRASE_KEY] = "abc"
    assert Backups.passphrase_problem()
    Backups.create()
    assert Backups.backups() == []


def test_a_backup_folder_inside_the_install_is_refused(install):
    """A reinstall into a fresh checkout would take the backups with it."""
    root, values = install
    values[Backups.DIRECTORY_KEY] = str(root / "backups")
    answer = Backups.create()
    assert "inside the Keep folder" in answer
    assert not (root / "backups").exists()


def test_only_the_newest_backups_are_kept(install):
    _, values = install
    folder = Path(values[Backups.DIRECTORY_KEY])
    folder.mkdir()
    for n, name in enumerate(("keep-old.keepbackup", "keep-older.keepbackup")):
        old = folder / name
        old.write_bytes(b"")
        os.utime(old, (1_700_000_000 - n, 1_700_000_000 - n))
    assert "Deleted 1 older" in Backups.create()
    assert {p.name for p in Backups.backups()} - {"keep-old.keepbackup"} == {Backups.backups()[0].name}
    assert not (folder / "keep-older.keepbackup").exists()


def test_resolve_finds_the_newest_a_name_and_a_name_without_its_suffix(install):
    Backups.create()
    [made] = Backups.backups()
    assert Backups.resolve("") == made
    assert Backups.resolve("latest") == made
    assert Backups.resolve(made.name) == made
    assert Backups.resolve(made.name[:-len(Backup.SUFFIX)]) == made
    assert Backups.resolve("keep-19990101-000000") is None


def test_stage_reports_a_wrong_passphrase_with_the_way_out(install):
    _, values = install
    Backups.create()
    [made] = Backups.backups()
    values[Backups.PASSPHRASE_KEY] = "another passphrase"
    problem = Backups.stage(made)
    assert "config_request" in problem
    assert Backup.staged(install[0]) is None


def test_stage_puts_the_backup_where_the_next_start_finds_it(install):
    root, _ = install
    Backups.create()
    assert Backups.stage(Backups.backups()[0]) == ""
    assert Backup.staged(root)["targets"] == ["config.yaml", "memory"]


def test_the_targets_are_inside_the_install_and_hold_the_state_that_makes_the_assistant():
    """The real list: config, vault, memory and history are what a restored install is the
    same assistant by; an integration's sign-in outside the tree belongs to the old machine."""
    from common.Paths import AISERVER_DIR, CONFIG_PATH, MEMORY_DIR, VAULT_PATH
    targets = Backups.targets()
    assert {CONFIG_PATH, VAULT_PATH, MEMORY_DIR, AISERVER_DIR / "floxy.db"} <= set(targets)
    for path in targets:
        rel = Backup.relative_target(path, AISERVER_DIR)
        assert rel is not None and Backup.check_target(rel) == rel
