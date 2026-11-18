"""Tests for common/Backup.py: the encrypted backup file, its staging and its application.

Every install here is a tmp_path tree laid out like the repository root; nothing touches the
real checkout."""
import io
import json
import sqlite3
import stat
import subprocess
import sys
import tarfile
from contextlib import closing
from pathlib import Path

import pytest

from common import Backup
from common.Backup import BackupError, WrongPassphrase

PASSPHRASE = "correct horse battery"
RESTORE_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "restore_backup.py"


def make_install(root: Path, owner: str = "Marcus") -> list[Path]:
    """A small install: a config, a memory folder and a database with a row."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.yaml").write_text(f"owner_name: {owner}\napi_keys:\n  anthropic: sk-ant-test-47L\n")
    (root / "memory" / "procedures").mkdir(parents=True)
    (root / "memory" / "bank.md").write_text(f"{owner} likes the sauna at 80 degrees.\n")
    (root / "memory" / "procedures" / "sauna.md").write_text("Turn on the sauna.\n")
    with closing(sqlite3.connect(root / "floxy.db")) as db:
        db.execute("CREATE TABLE history (text TEXT)")
        db.execute("INSERT INTO history VALUES (?)", (f"hello {owner}",))
        db.commit()
    return [root / "config.yaml", root / "memory", root / "floxy.db", root / "vault.bin"]


def history_rows(db: Path) -> list[str]:
    with closing(sqlite3.connect(db)) as conn:
        return [r[0] for r in conn.execute("SELECT text FROM history")]


@pytest.fixture
def made(tmp_path):
    """A backup of one install, and a second, different install to restore it into."""
    source = tmp_path / "old"
    targets = make_install(source)
    dest = tmp_path / "backups" / "keep.keepbackup"
    written = Backup.write(dest, targets, PASSPHRASE, source, "v2026.11.1")
    target = tmp_path / "new"
    make_install(target, owner="Frances")
    return written, target


class TestRoundTrip:
    def test_a_restored_install_holds_what_the_old_one_held(self, made):
        written, target = made
        Backup.restore(written.path, PASSPHRASE, target)
        assert "Marcus" in (target / "config.yaml").read_text()
        assert (target / "memory" / "procedures" / "sauna.md").read_text() == "Turn on the sauna.\n"
        assert history_rows(target / "floxy.db") == ["hello Marcus"]

    def test_the_replaced_state_is_kept_aside(self, made):
        written, target = made
        Backup.restore(written.path, PASSPHRASE, target)
        aside = next((target / "cache").glob(f"{Backup.PREVIOUS_PREFIX}*"))
        assert "Frances" in (aside / "config.yaml").read_text()
        assert history_rows(aside / "floxy.db") == ["hello Frances"]

    def test_a_target_the_backup_lacks_is_moved_aside_not_kept(self, made):
        """The old install had no vault: a vault of the new one left beside its config would
        belong to another assistant."""
        written, target = made
        (target / "vault.bin").write_bytes(b"FLOXYVAULT1\n")
        Backup.restore(written.path, PASSPHRASE, target)
        assert not (target / "vault.bin").exists()

    def test_a_file_outside_the_backup_is_left_alone(self, made):
        written, target = made
        (target / "notes.txt").write_text("mine")
        Backup.restore(written.path, PASSPHRASE, target)
        assert (target / "notes.txt").read_text() == "mine"

    def test_the_backup_and_the_restored_files_are_the_owners_alone(self, made):
        written, target = made
        assert stat.S_IMODE(written.path.stat().st_mode) == 0o600
        Backup.restore(written.path, PASSPHRASE, target)
        assert stat.S_IMODE((target / "config.yaml").stat().st_mode) == 0o600

    def test_nothing_of_the_contents_is_readable_in_the_file(self, made):
        written, _ = made
        raw = written.path.read_bytes()
        assert b"sk-ant-test-47L" not in raw and b"Marcus" not in raw

    def test_the_manifest_says_what_and_when(self, made):
        written, _ = made
        manifest = Backup.manifest(written.path, PASSPHRASE)
        assert manifest["keep_version"] == "v2026.11.1"
        assert set(manifest["targets"]) == {"config.yaml", "memory", "floxy.db", "vault.bin"}


class TestDatabase:
    def test_a_database_in_wal_mode_comes_back_whole_without_its_sidecars(self, tmp_path):
        source = tmp_path / "old"
        source.mkdir()
        db = source / "floxy.db"
        writer = sqlite3.connect(db)
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("CREATE TABLE history (text TEXT)")
        writer.execute("INSERT INTO history VALUES ('in the wal')")
        writer.commit()     # still open: the row lives in floxy.db-wal
        try:
            assert (source / "floxy.db-wal").exists()
            dest = tmp_path / "b.keepbackup"
            Backup.write(dest, [db], PASSPHRASE, source)
        finally:
            writer.close()
        target = tmp_path / "new"
        target.mkdir()
        (target / "floxy.db-wal").write_bytes(b"stale")
        Backup.restore(dest, PASSPHRASE, target)
        assert history_rows(target / "floxy.db") == ["in the wal"]
        assert not (target / "floxy.db-wal").exists()


class TestRefusals:
    def test_a_wrong_passphrase_opens_nothing_and_stages_nothing(self, made):
        written, target = made
        with pytest.raises(WrongPassphrase):
            Backup.stage(written.path, "not it", target)
        assert Backup.staged(target) is None
        assert "Frances" in (target / "config.yaml").read_text()

    def test_a_cut_short_file_is_refused(self, made, tmp_path):
        written, target = made
        cut = tmp_path / "cut.keepbackup"
        cut.write_bytes(written.path.read_bytes()[:-10])
        with pytest.raises(BackupError):
            Backup.stage(cut, PASSPHRASE, target)
        assert Backup.staged(target) is None

    def test_a_changed_byte_is_refused(self, made, tmp_path):
        written, target = made
        data = bytearray(written.path.read_bytes())
        data[-20] ^= 0x01
        bad = tmp_path / "bad.keepbackup"
        bad.write_bytes(bytes(data))
        with pytest.raises(BackupError):
            Backup.stage(bad, PASSPHRASE, target)

    def test_a_file_that_is_no_backup_is_refused(self, tmp_path):
        other = tmp_path / "x.keepbackup"
        other.write_text("hello")
        with pytest.raises(BackupError, match="not a Keep backup"):
            Backup.stage(other, PASSPHRASE, tmp_path / "root")

    @pytest.mark.parametrize("target", ["src/common/logging.py", "start.sh", ".git/config",
                                        "cache", "cache/restore-pending/x"])
    def test_code_and_the_staging_area_are_never_targets(self, tmp_path, target):
        with pytest.raises(BackupError):
            Backup.write(tmp_path / "b.keepbackup", [tmp_path / "root" / target], PASSPHRASE, tmp_path / "root")

    def test_a_path_outside_the_install_is_left_out(self, tmp_path):
        root = tmp_path / "root"
        targets = make_install(root)
        outside = tmp_path / "token.cache"
        outside.write_text("t")
        Backup.write(tmp_path / "b.keepbackup", targets + [outside], PASSPHRASE, root)
        assert "token.cache" not in json.dumps(Backup.manifest(tmp_path / "b.keepbackup", PASSPHRASE))

    def test_a_crafted_backup_writing_into_the_code_is_refused(self, tmp_path):
        """Someone who knows the passphrase builds a file whose manifest claims only config.yaml
        but whose files reach into src/: nothing is staged."""
        dest = tmp_path / "evil.keepbackup"
        Backup.write(dest, [], PASSPHRASE, tmp_path)
        salt, header = self._seal(dest, {"format": Backup.FORMAT, "targets": ["config.yaml"]},
                                  {"files/src/common/logging.py": b"evil"})
        with pytest.raises(BackupError, match="none of its targets"):
            Backup.stage(dest, PASSPHRASE, tmp_path / "root")
        assert Backup.staged(tmp_path / "root") is None

    @staticmethod
    def _seal(dest: Path, manifest: dict, files: dict):
        """Rewrites *dest* with the given members, sealed the way Backup.write seals."""
        import base64
        raw = dest.read_bytes()
        line = raw[len(Backup.MAGIC):raw.index(b"\n", len(Backup.MAGIC)) + 1]
        header = json.loads(line)
        key = Backup._derive(PASSPHRASE, base64.b64decode(header["salt"]), header["n"])
        with open(dest, "wb") as out:
            out.write(Backup.MAGIC + line)
            sealer = Backup._Sealer(out, key, base64.b64decode(header["nonce"]), Backup.MAGIC + line)
            with tarfile.open(fileobj=sealer, mode="w|gz") as tar:
                for name, data in [("manifest.json", json.dumps(manifest).encode()), *files.items()]:
                    member = tarfile.TarInfo(name)
                    member.size = len(data)
                    tar.addfile(member, io.BytesIO(data))
            sealer.finish()
        return header["salt"], header


class TestStaging:
    def test_staging_touches_nothing_until_it_is_applied(self, made):
        written, target = made
        manifest = Backup.stage(written.path, PASSPHRASE, target)
        assert manifest["files"] == 4
        assert "Frances" in (target / "config.yaml").read_text()
        assert Backup.staged(target) is not None
        outcome = Backup.apply_pending(target)
        assert "4 file(s)" in outcome
        assert "Marcus" in (target / "config.yaml").read_text()
        assert Backup.staged(target) is None

    def test_nothing_staged_is_nothing_done(self, tmp_path):
        assert Backup.apply_pending(tmp_path) is None

    def test_an_unfinished_staging_area_is_discarded_not_applied(self, made):
        """No manifest is written until the whole backup has been unpacked."""
        _, target = made
        half = target / Backup.STAGING / "files"
        half.mkdir(parents=True)
        (half / "config.yaml").write_text("owner_name: Dave\n")
        assert Backup.apply_pending(target) is None
        assert "Frances" in (target / "config.yaml").read_text()
        assert not (target / Backup.STAGING).exists()

    def test_large_content_spans_chunks(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Backup, "CHUNK_SIZE", 1024)
        root = tmp_path / "root"
        (root / "memory").mkdir(parents=True)
        payload = bytes(range(256)) * 200
        (root / "memory" / "blob.bin").write_bytes(payload)
        dest = tmp_path / "b.keepbackup"
        Backup.write(dest, [root / "memory"], PASSPHRASE, root)
        target = tmp_path / "new"
        Backup.restore(dest, PASSPHRASE, target)
        assert (target / "memory" / "blob.bin").read_bytes() == payload


class TestRestoreScript:
    """scripts/restore_backup.py, what install.sh runs: its exit status is the installer's contract."""

    def run(self, root: Path, backup: Path, passphrase: str) -> subprocess.CompletedProcess:
        script = root / "scripts" / "restore_backup.py"
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text(RESTORE_SCRIPT.read_text())
        (root / "src").symlink_to(RESTORE_SCRIPT.parents[1] / "src")
        return subprocess.run([sys.executable, str(script), str(backup)], capture_output=True, text=True,
                              env={"KEEP_BACKUP_PASSPHRASE": passphrase, "PATH": "/usr/bin:/bin"}, timeout=60)

    def test_restores_into_the_install_it_stands_in(self, made):
        written, target = made
        result = self.run(target, written.path, PASSPHRASE)
        assert result.returncode == 0, result.stderr
        assert "Marcus" in (target / "config.yaml").read_text()

    def test_a_wrong_passphrase_exits_3(self, made):
        """Contract with install.sh's offer_restore, which asks again on 3."""
        written, target = made
        assert self.run(target, written.path, "nope").returncode == 3
        assert "Frances" in (target / "config.yaml").read_text()


class TestCommandLine:
    """scripts/restore_backup.py by hand over ssh (`./keep.sh backups` / `restore`): the backups of the
    configured folder by name, opened with the configured passphrase, the server stopped first."""

    @pytest.fixture
    def cli(self, made, tmp_path, monkeypatch):
        import importlib.util
        spec = importlib.util.spec_from_file_location("restore_backup_cli", RESTORE_SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        written, target = made
        config = tmp_path / "cli-config.yaml"
        config.write_text(f"backup:\n  passphrase: {PASSPHRASE}\n  directory: {written.path.parent}\n")
        monkeypatch.setenv("FLOXY_CONFIG", str(config))
        monkeypatch.delenv("KEEP_BACKUP_PASSPHRASE", raising=False)
        monkeypatch.setattr(module, "ROOT", target)
        monkeypatch.setattr(module.sys.stdin, "isatty", lambda: False, raising=False)
        stops = []
        monkeypatch.setattr(module.subprocess, "run", lambda command, **kw: stops.append(command))
        module.stops = stops
        return module, written, target

    def test_list_names_the_backups_of_the_configured_folder(self, cli, capsys):
        module, written, _ = cli
        assert module.main(["restore_backup.py", "--list"]) == 0
        assert written.path.name in capsys.readouterr().out

    def test_latest_is_restored_with_the_configured_passphrase_after_stopping_the_server(self, cli):
        module, _, target = cli
        assert module.main(["restore_backup.py", "--stop-server", "latest"]) == 0
        assert "Marcus" in (target / "config.yaml").read_text()
        assert [Path(c[0]).name for c in module.stops] == ["stop.sh"]

    def test_a_name_without_its_suffix_is_one_of_the_folder(self, cli):
        module, written, target = cli
        assert module.main(["restore_backup.py", written.path.stem]) == 0
        assert "Marcus" in (target / "config.yaml").read_text() and module.stops == []

    def test_a_backup_no_passphrase_opens_stops_nothing(self, cli, monkeypatch):
        module, _, target = cli
        monkeypatch.setenv("KEEP_BACKUP_PASSPHRASE", "not the one")
        assert module.main(["restore_backup.py", "--stop-server", "latest"]) == module.EXIT_WRONG_PASSPHRASE
        assert module.stops == [] and "Frances" in (target / "config.yaml").read_text()

    def test_an_unknown_name_stops_nothing(self, cli):
        module, _, target = cli
        assert module.main(["restore_backup.py", "--stop-server", "nothing-such"]) == 1
        assert module.stops == [] and "Frances" in (target / "config.yaml").read_text()
