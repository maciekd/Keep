"""config.yaml holds the API keys and the vault passphrase: no writer may widen who reads it."""
import os
import stat

import pytest

from common import ConfigMigrations, ConfigStore, PrivateFile
from common.ConfigManager import ConfigManager


def mode(path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


@pytest.fixture
def loose_umask():
    """Ubuntu's default for a user: group-writable, world-readable files."""
    old = os.umask(0o002)
    yield
    os.umask(old)


@pytest.mark.unit
class TestPrivateFile:
    def test_a_rewrite_keeps_the_mode_the_file_had(self, tmp_path, loose_umask):
        path = tmp_path / "config.yaml"
        path.write_text("a: 1\n")
        for kept in (0o600, 0o640):
            path.chmod(kept)
            PrivateFile.write(path, "a: 2\n")
            assert mode(path) == kept and path.read_text() == "a: 2\n"

    def test_a_new_file_is_the_owners_alone(self, tmp_path, loose_umask):
        path = tmp_path / "config.yaml"
        PrivateFile.write(path, "a: 1\n")
        assert mode(path) == 0o600

    def test_a_temp_file_left_by_a_crash_does_not_lend_its_mode(self, tmp_path, loose_umask):
        path = tmp_path / "config.yaml"
        (tmp_path / "config.yaml.tmp").write_text("half")
        (tmp_path / "config.yaml.tmp").chmod(0o666)
        PrivateFile.write(path, "a: 1\n")
        assert mode(path) == 0o600 and not (tmp_path / "config.yaml.tmp").exists()

    def test_a_copy_is_the_owners_alone_whatever_the_source_was(self, tmp_path, loose_umask):
        source = tmp_path / "config.yaml"
        source.write_text("a: 1\n")
        source.chmod(0o664)
        PrivateFile.copy(source, tmp_path / "backup.yaml")
        assert mode(tmp_path / "backup.yaml") == 0o600 and mode(source) == 0o664


@pytest.mark.unit
class TestTheWritersOfConfigYaml:
    def test_config_set_leaves_the_installers_mode_and_backs_up_privately(self, tmp_path, monkeypatch, loose_umask):
        path = tmp_path / "config.yaml"
        path.write_text("owner_name: Marcus\n")
        path.chmod(0o600)
        monkeypatch.setenv("FLOXY_CONFIG", str(path))
        monkeypatch.setattr(ConfigStore, "BACKUP_DIR", tmp_path / "backups")
        monkeypatch.setattr(ConfigStore, "_reload", lambda: None)
        ConfigStore._rewrite_locked(lambda data: data.__setitem__("owner_name", "Charlotte"))
        assert "Charlotte" in path.read_text() and mode(path) == 0o600
        (backup,) = (tmp_path / "backups").iterdir()
        assert mode(backup) == 0o600

    def test_a_backup_of_a_config_somebody_opened_up_is_still_private(self, tmp_path, monkeypatch, loose_umask):
        path = tmp_path / "config.yaml"
        path.write_text("owner_name: Marcus\n")
        path.chmod(0o644)
        monkeypatch.setattr(ConfigStore, "BACKUP_DIR", tmp_path / "backups")
        ConfigStore._backup(path)
        (backup,) = (tmp_path / "backups").iterdir()
        assert mode(backup) == 0o600

    def test_a_migration_leaves_the_mode_and_backs_up_privately(self, tmp_path, monkeypatch, loose_umask):
        path = tmp_path / "config.yaml"
        path.write_text("enabled_integrations: [Gmail]\ngmail:\n  username: m@mlab.test\n  password: app-password-1\n")
        path.chmod(0o600)
        monkeypatch.setattr(ConfigMigrations, "BACKUP_DIR", tmp_path / "backups")
        assert ConfigMigrations.run(str(path)), "the fixture no longer needs a migration"
        ConfigMigrations.take_note()
        assert mode(path) == 0o600
        assert [mode(backup) for backup in (tmp_path / "backups").iterdir()] == [0o600]

    def test_the_skeleton_of_a_first_start_is_the_owners_alone(self, tmp_path, monkeypatch, loose_umask):
        path = tmp_path / "config.yaml"
        monkeypatch.setenv("FLOXY_CONFIG", str(path))
        ConfigManager.reset_instance()
        try:
            assert ConfigManager().created_skeleton and mode(path) == 0o600
        finally:
            ConfigManager.reset_instance()
