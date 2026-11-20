"""Tests for common/Backup.py: the zip of an install's state, its staging and the restore a start runs.

Every install here is a tmp_path tree laid out like the repository root; nothing touches the
real checkout."""
import json
import os
import sqlite3
import stat
import zipfile
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import pytest

from common import Backup
from common.Backup import BackupError


def make_install(root: Path, owner: str = "Marcus") -> list[Path]:
    """A small install — a config, a memory folder and a database with a row — and its state."""
    (root / "memory" / "procedures").mkdir(parents=True)
    (root / "config.yaml").write_text(f"owner_name: {owner}\n")
    (root / "memory" / "bank.md").write_text(f"{owner} likes the sauna at 80 degrees.\n")
    (root / "memory" / "procedures" / "sauna.md").write_text("Turn on the sauna.\n")
    with closing(sqlite3.connect(root / "floxy.db")) as db:
        db.execute("CREATE TABLE history (text TEXT)")
        db.execute("INSERT INTO history VALUES (?)", (f"hello {owner}",))
        db.commit()
    return state_of(root)


def state_of(root: Path) -> list[Path]:
    return [root / "config.yaml", root / "memory", root / "floxy.db", root / "vault.bin"]


def history_rows(db: Path) -> list[str]:
    with closing(sqlite3.connect(db)) as conn:
        return [r[0] for r in conn.execute("SELECT text FROM history")]


def restore(archive: Path, root: Path):
    Backup.stage(archive, state_of(root), root)
    return Backup.apply_pending(root)


@pytest.fixture
def made(tmp_path):
    """The backup of one install, and a second, different install to restore it into."""
    old = tmp_path / "old"
    archive = Backup.write(tmp_path / "backups" / "keep.zip", make_install(old), old)
    new = tmp_path / "new"
    make_install(new, owner="Frances")
    return archive, new


class TestRoundTrip:
    def test_a_restored_install_holds_what_the_old_one_held(self, made):
        archive, new = made
        restore(archive, new)
        assert (new / "config.yaml").read_text() == "owner_name: Marcus\n"
        assert (new / "memory" / "procedures" / "sauna.md").read_text() == "Turn on the sauna.\n"
        assert history_rows(new / "floxy.db") == ["hello Marcus"]

    def test_state_the_backup_lacks_does_not_survive_the_restore(self, made):
        """The old install had no vault and no such procedure: left beside the restored config
        they would belong to another assistant."""
        archive, new = made
        (new / "vault.bin").write_bytes(b"FLOXYVAULT1\n")
        (new / "memory" / "procedures" / "boat.md").write_text("Launch the boat.\n")
        restore(archive, new)
        assert not (new / "vault.bin").exists()
        assert not (new / "memory" / "procedures" / "boat.md").exists()

    def test_a_file_outside_the_state_is_left_alone(self, made):
        archive, new = made
        (new / "notes.txt").write_text("mine")
        restore(archive, new)
        assert (new / "notes.txt").read_text() == "mine"

    def test_the_zip_and_the_restored_files_are_the_owners_alone_whatever_the_umask(self, tmp_path):
        """The zip holds config.yaml, and a zip carries no modes back."""
        old, new = tmp_path / "old", tmp_path / "new"
        new.mkdir()
        previous = os.umask(0o002)
        try:
            archive = Backup.write(tmp_path / "keep.zip", make_install(old), old)
            restore(archive, new)
        finally:
            os.umask(previous)
        assert stat.S_IMODE(archive.stat().st_mode) == 0o600
        assert stat.S_IMODE((new / "config.yaml").stat().st_mode) == 0o600

    def test_a_write_that_fails_leaves_no_zip_behind(self, tmp_path, monkeypatch):
        old = tmp_path / "old"
        targets = make_install(old)

        def disk_full(*args):
            raise OSError("disk full")

        monkeypatch.setattr(Backup, "_write_database", disk_full)
        with pytest.raises(OSError):
            Backup.write(tmp_path / "keep.zip", targets, old)
        assert not (tmp_path / "keep.zip").exists()


class TestEachFileOnce:
    def test_a_file_named_on_its_own_and_inside_a_named_directory_is_in_the_zip_once(self, tmp_path):
        """A sign-in token listed by its integration and again under the oauth directory a
        backup holds whole."""
        root = tmp_path / "Keep"
        oauth = root / "cache" / "oauth"
        oauth.mkdir(parents=True)
        (oauth / "music.json").write_text('{"refresh_token": "r-1"}')
        (oauth / "car.json").write_text('{"refresh_token": "r-2"}')
        archive = Backup.write(tmp_path / "keep.zip", [oauth / "music.json", oauth, oauth / "car.json"], root)
        with zipfile.ZipFile(archive) as zf:
            names = zf.namelist()
        assert sorted(names) == ["cache/oauth/car.json", "cache/oauth/music.json"]

    def test_a_database_named_twice_is_copied_once(self, tmp_path):
        root = tmp_path / "Keep"
        state = make_install(root)
        archive = Backup.write(tmp_path / "keep.zip", state + [root / "floxy.db", root], root)
        with zipfile.ZipFile(archive) as zf:
            names = zf.namelist()
        assert len(names) == len(set(names)) and names.count("floxy.db") == 1


class TestDatabase:
    def test_a_database_in_wal_mode_comes_back_whole_and_without_sidecars(self, tmp_path):
        old = tmp_path / "old"
        old.mkdir()
        writer = sqlite3.connect(old / "floxy.db")
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("CREATE TABLE history (text TEXT)")
        writer.execute("INSERT INTO history VALUES ('in the wal')")
        writer.commit()     # still open: the row lives in floxy.db-wal
        try:
            assert (old / "floxy.db-wal").exists()
            archive = Backup.write(tmp_path / "keep.zip", [old / "floxy.db", old / "floxy.db-wal"], old)
        finally:
            writer.close()
        with zipfile.ZipFile(archive) as zf:
            assert zf.namelist() == ["floxy.db"]
        new = tmp_path / "new"
        new.mkdir()
        (new / "floxy.db-wal").write_bytes(b"stale")
        restore(archive, new)
        assert history_rows(new / "floxy.db") == ["in the wal"]
        assert not (new / "floxy.db-wal").exists()


class TestStaging:
    def test_staging_touches_nothing_until_a_start_applies_it(self, made):
        archive, new = made
        assert Backup.stage(archive, state_of(new), new) == 4
        assert (new / "config.yaml").read_text() == "owner_name: Frances\n"
        assert archive.name in Backup.apply_pending(new)
        assert (new / "config.yaml").read_text() == "owner_name: Marcus\n"

    def test_a_restore_is_applied_once(self, made):
        archive, new = made
        restore(archive, new)
        (new / "config.yaml").write_text("owner_name: Charlotte\n")
        assert Backup.apply_pending(new) is None
        assert (new / "config.yaml").read_text() == "owner_name: Charlotte\n"

    def test_nothing_staged_is_nothing_done(self, tmp_path):
        assert Backup.apply_pending(tmp_path) is None

    def test_a_file_that_is_no_zip_is_refused_and_nothing_is_staged(self, made, tmp_path):
        _, new = made
        other = tmp_path / "holiday.zip"
        other.write_text("hello")
        with pytest.raises(BackupError):
            Backup.stage(other, state_of(new), new)
        assert Backup.apply_pending(new) is None

    def test_a_zip_with_a_damaged_member_is_refused(self, made, tmp_path):
        archive, new = made
        data = bytearray(archive.read_bytes())
        with zipfile.ZipFile(archive) as zf:
            member = zf.getinfo("memory/bank.md")
        start = member.header_offset + 30 + len(member.filename)
        data[start + 2] ^= 0xFF
        damaged = tmp_path / "damaged.zip"
        damaged.write_bytes(bytes(data))
        with pytest.raises(BackupError):
            Backup.stage(damaged, state_of(new), new)
        assert Backup.apply_pending(new) is None

    def test_a_zip_holding_nothing_of_the_state_is_refused(self, made, tmp_path):
        _, new = made
        other = tmp_path / "photos.zip"
        with zipfile.ZipFile(other, "w") as zf:
            zf.writestr("summer/boat.txt", "a boat")
        with pytest.raises(BackupError):
            Backup.stage(other, state_of(new), new)

    def test_a_zip_gone_before_the_start_fails_with_the_state_untouched(self, made):
        archive, new = made
        Backup.stage(archive, state_of(new), new)
        archive.unlink()
        assert "failed" in Backup.apply_pending(new)
        assert (new / "config.yaml").read_text() == "owner_name: Frances\n"
        assert Backup.apply_pending(new) is None    # and no start tries it again

    def test_a_marker_that_does_not_read_fails_once_with_the_state_untouched(self, made):
        """A start cut off while the marker was written: the next one says so and goes on,
        instead of failing at every start or removing the state for a zip it cannot name."""
        _, new = made
        marker = new / Backup.PENDING
        marker.parent.mkdir(parents=True)
        marker.write_text('{"archive": "/tmp/keep')
        assert "failed" in Backup.apply_pending(new)
        assert (new / "config.yaml").read_text() == "owner_name: Frances\n"
        assert history_rows(new / "floxy.db") == ["hello Frances"]
        assert Backup.apply_pending(new) is None

    def test_state_the_zip_lacks_is_counted_out_of_the_files_to_restore(self, made, tmp_path):
        """The number a staging answers with is what the restore unpacks: the members under
        the state, a folder's own entry and a file beside the state left out."""
        _, new = made
        mixed = tmp_path / "mixed.zip"
        with zipfile.ZipFile(mixed, "w") as zf:
            zf.writestr("memory/", "")
            zf.writestr("memory/bank.md", "Marcus\n")
            zf.writestr("memory/watchlist/boat.md", "The boat\n")
            zf.writestr("memory.md.bak", "beside the state, named like it")
        assert Backup.stage(mixed, state_of(new), new) == 2
        Backup.apply_pending(new)
        assert sorted(p.name for p in (new / "memory").rglob("*") if p.is_file()) == ["bank.md", "boat.md"]
        assert not (new / "memory.md.bak").exists() and not (new / "config.yaml").exists()


class TestAMarkerIsForItsOwnRelaunch:
    def test_a_marker_whose_relaunch_never_came_is_not_applied_by_a_later_start(self, made):
        """The relaunch failed and the server lived on: a start days later must not roll the
        install back to the zip."""
        archive, new = made
        Backup.stage(archive, state_of(new), new)
        staged = Backup.time.time()
        with patch.object(Backup.time, "time", return_value=staged + Backup.STAGED_FOR_SECONDS + 60):
            assert archive.name in Backup.apply_pending(new)
            assert (new / "config.yaml").read_text() == "owner_name: Frances\n"
            assert history_rows(new / "floxy.db") == ["hello Frances"]
            assert Backup.apply_pending(new) is None

    def test_a_marker_that_says_no_time_is_not_applied(self, made):
        archive, new = made
        Backup.stage(archive, state_of(new), new)
        marker = new / Backup.PENDING
        plan = json.loads(marker.read_text())
        del plan["staged"]
        marker.write_text(json.dumps(plan))
        Backup.apply_pending(new)
        assert (new / "config.yaml").read_text() == "owner_name: Frances\n"

    def test_a_restore_taken_back_is_not_applied(self, made):
        archive, new = made
        Backup.stage(archive, state_of(new), new)
        assert Backup.unstage(new) is True
        assert Backup.apply_pending(new) is None
        assert (new / "config.yaml").read_text() == "owner_name: Frances\n"
        assert Backup.unstage(new) is False


def tree(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


class TestAFailedRestoreChangesNothing:
    def test_an_extraction_that_fails_midway_leaves_the_install_as_it_was(self, made):
        """The disk fills up after the first files: the config, the memory and the database
        are the install's own still, and no half-unpacked folder is left behind."""
        archive, new = made
        before = tree(new)
        real = zipfile.ZipFile.extractall

        def half(zf, path=None, members=None, pwd=None):
            real(zf, path, members[:2], pwd)
            raise OSError(28, "No space left on device")

        Backup.stage(archive, state_of(new), new)
        with patch.object(zipfile.ZipFile, "extractall", half):
            assert "No space left" in Backup.apply_pending(new)
        assert tree(new) == before
        assert history_rows(new / "floxy.db") == ["hello Frances"]

    @pytest.mark.parametrize("failing_move", [1, 2, 3, 4, 5, 6])
    def test_a_swap_that_fails_at_any_move_puts_the_state_back(self, made, failing_move):
        archive, new = made
        (new / "floxy.db-wal").write_text("the old install's wal")
        before = tree(new)
        real, calls = Backup._move, []

        def move(source, target):
            calls.append(source)
            if len(calls) == failing_move:
                raise OSError("the volume went away")
            real(source, target)

        Backup.stage(archive, state_of(new), new)
        with patch.object(Backup, "_move", move):
            assert "the volume went away" in Backup.apply_pending(new)
        assert tree(new) == before

    def test_a_swap_that_cannot_be_undone_keeps_the_replaced_state_and_says_where(self, made):
        """The volume fails at a move and at the first move back: what was moved aside is in
        no other place, so the work folder stays and the answer names it."""
        archive, new = made
        before = tree(new)
        real, calls = Backup._move, []

        def move(source, target):
            calls.append(source)
            if len(calls) in (3, 4):
                raise OSError("the volume went away")
            real(source, target)

        Backup.stage(archive, state_of(new), new)
        with patch.object(Backup, "_move", move):
            said = Backup.apply_pending(new)
        [work] = new.glob(Backup.WORK_PREFIX + "*")
        assert str(work / "old") in said
        kept = tree(work / "old")
        assert kept and all(before[name] == content for name, content in kept.items())
        assert Backup.apply_pending(new) is None

    def test_a_folder_whose_copy_fails_midway_leaves_the_install_as_it_was(self, unpacked, tmp_path):
        _, _, folder = unpacked
        new = tmp_path / "new"
        make_install(new, owner="Frances")
        before = tree(new)
        real, copied = Backup.shutil.copyfile, []

        def copy(source, target):
            if copied:
                raise OSError(28, "No space left on device")
            copied.append(source)
            return real(source, target)

        Backup.stage(folder, state_of(new), new)
        with patch.object(Backup.shutil, "copyfile", copy):
            assert "No space left" in Backup.apply_pending(new)
        assert tree(new) == before

    def test_a_restore_leaves_no_work_folder(self, made):
        archive, new = made
        restore(archive, new)
        assert not list(new.glob(Backup.WORK_PREFIX + "*"))

    def test_a_file_listed_beside_its_folder_is_restored_with_it(self, tmp_path):
        """An install's state names `cache/oauth` and a token inside it both."""
        old, new = tmp_path / "old", tmp_path / "new"
        for root, token in ((old, "Marcus's token"), (new, "Frances's token")):
            (root / "cache" / "oauth").mkdir(parents=True)
            (root / "cache" / "oauth" / "spotify.json").write_text(token)
            (root / "cache" / "oauth" / "volvo.json").write_text(token)
        state = lambda root: [root / "cache" / "oauth" / "spotify.json", root / "cache" / "oauth"]
        archive = Backup.write(tmp_path / "keep.zip", state(old), old)
        Backup.stage(archive, state(new), new)
        assert archive.name in Backup.apply_pending(new)
        assert (new / "cache" / "oauth" / "spotify.json").read_text() == "Marcus's token"
        assert (new / "cache" / "oauth" / "volvo.json").read_text() == "Marcus's token"


class TestOnlyTheStateIsWritten:
    def test_members_outside_the_state_are_not_unpacked(self, made, tmp_path):
        """A zip somebody else built reaches into the code and climbs out of a state folder:
        neither file appears, the state it does hold is restored."""
        _, new = made
        crafted = tmp_path / "crafted.zip"
        with zipfile.ZipFile(crafted, "w") as zf:
            zf.writestr("config.yaml", "owner_name: Dave\n")
            zf.writestr("src/common/logging.py", "evil")
            zf.writestr("start.sh", "evil")
            zf.writestr("memory/../start.sh", "evil")
            zf.writestr("external/tmp/ring/token.cache", "a wipe backup's outside file")
        assert Backup.stage(crafted, state_of(new), new) == 1
        Backup.apply_pending(new)
        assert (new / "config.yaml").read_text() == "owner_name: Dave\n"
        assert not (new / "src").exists() and not (new / "start.sh").exists() and not (new / "external").exists()
        assert not (new / "memory" / "start.sh").exists()

    def test_a_state_path_outside_the_install_is_never_replaced(self, made, tmp_path):
        archive, new = made
        outside = tmp_path / "token.cache"
        outside.write_text("ring")
        Backup.stage(archive, state_of(new) + [outside], new)
        Backup.apply_pending(new)
        assert outside.read_text() == "ring"


def files_of(folder: Path) -> dict[str, bytes]:
    return {p.relative_to(folder).as_posix(): p.read_bytes() for p in folder.rglob("*") if p.is_file()}


@pytest.fixture
def unpacked(tmp_path):
    """An install, the zip of it and that zip as a folder."""
    old = tmp_path / "old"
    state = make_install(old)
    archive = Backup.write(tmp_path / "backups" / "keep.zip", state, old)
    return old, archive, Backup.unpack(archive, tmp_path / "backups" / "current")


class TestUnpack:
    def test_the_folder_holds_the_files_of_the_zip_and_the_database_as_a_whole_copy(self, unpacked):
        old, archive, folder = unpacked
        with zipfile.ZipFile(archive) as zf:
            assert sorted(files_of(folder)) == sorted(zf.namelist())
        assert (folder / "config.yaml").read_bytes() == (old / "config.yaml").read_bytes()
        assert history_rows(folder / "floxy.db") == ["hello Marcus"]

    def test_the_folder_is_the_owners_alone_whatever_the_umask(self, unpacked):
        _, _, folder = unpacked
        for path in [folder, *folder.rglob("*")]:
            assert stat.S_IMODE(path.stat().st_mode) & 0o077 == 0, path

    def test_a_new_backup_replaces_the_folder_whole(self, unpacked, tmp_path):
        """A procedure deleted since the last backup is not in the new folder: it is another
        tree, not the old one written over."""
        old, _, folder = unpacked
        (old / "memory" / "procedures" / "sauna.md").unlink()
        (old / "config.yaml").write_text("owner_name: Frances\n")
        Backup.unpack(Backup.write(tmp_path / "second.zip", state_of(old), old), folder)
        assert (folder / "config.yaml").read_text() == "owner_name: Frances\n"
        assert not (folder / "memory" / "procedures" / "sauna.md").exists()
        assert sorted(p.name for p in folder.parent.iterdir()) == ["current", "keep.zip"]

    @pytest.mark.parametrize("left", [".tmp", ".old"])
    def test_what_a_run_that_died_left_beside_the_folder_does_not_stop_the_next(self, unpacked, left):
        _, archive, folder = unpacked
        leftover = folder.with_name(folder.name + left)
        leftover.mkdir()
        (leftover / "config.yaml").write_text("half")
        Backup.unpack(archive, folder)
        assert not leftover.exists()
        assert (folder / "config.yaml").read_text() == "owner_name: Marcus\n"

    def test_a_zip_that_does_not_unpack_leaves_the_previous_folder_as_it_was(self, unpacked, tmp_path):
        _, _, folder = unpacked
        before = files_of(folder)
        broken = tmp_path / "broken.zip"
        broken.write_bytes(b"not a zip")
        with pytest.raises(zipfile.BadZipFile):
            Backup.unpack(broken, folder)
        assert files_of(folder) == before
        assert not folder.with_name("current.tmp").exists() and not folder.with_name("current.old").exists()


class TestFolderRestore:
    def test_a_folder_restores_like_a_zip(self, unpacked, tmp_path):
        _, _, folder = unpacked
        new = tmp_path / "new"
        make_install(new, owner="Frances")
        (new / "memory" / "procedures" / "boat.md").write_text("Launch the boat.\n")
        assert Backup.stage(folder, state_of(new), new) == 4
        assert "current" in Backup.apply_pending(new)
        assert (new / "config.yaml").read_text() == "owner_name: Marcus\n"
        assert history_rows(new / "floxy.db") == ["hello Marcus"]
        assert not (new / "memory" / "procedures" / "boat.md").exists()
        assert stat.S_IMODE((new / "config.yaml").stat().st_mode) == 0o600

    def test_a_file_of_the_folder_outside_the_state_is_not_put_in_the_install(self, unpacked, tmp_path):
        _, _, folder = unpacked
        (folder / "start.sh").write_text("echo not ours")
        new = tmp_path / "new"
        make_install(new, owner="Frances")
        restore(folder, new)
        assert not (new / "start.sh").exists()

    def test_a_link_in_the_folder_brings_nothing_in_from_outside_it(self, unpacked, tmp_path):
        _, _, folder = unpacked
        outside = tmp_path / "id_ed25519"
        outside.write_text("a key of the machine")
        (folder / "memory" / "stolen.md").symlink_to(outside)
        new = tmp_path / "new"
        make_install(new, owner="Frances")
        restore(folder, new)
        assert not (new / "memory" / "stolen.md").exists()

    def test_a_folder_with_nothing_of_an_install_is_refused_and_nothing_is_staged(self, tmp_path):
        photos = tmp_path / "photos"
        photos.mkdir()
        (photos / "boat.jpg").write_bytes(b"jpeg")
        new = tmp_path / "new"
        make_install(new, owner="Frances")
        with pytest.raises(BackupError):
            Backup.stage(photos, state_of(new), new)
        assert Backup.apply_pending(new) is None

    def test_a_folder_with_a_file_that_cannot_be_read_is_refused_while_the_install_is_whole(self, unpacked, tmp_path, monkeypatch):
        """Found at the start it would be found after the install's own state was removed."""
        _, _, folder = unpacked
        monkeypatch.setattr(Backup.os, "access", lambda path, mode: Path(path).name != "bank.md")
        new = tmp_path / "new"
        make_install(new, owner="Frances")
        with pytest.raises(BackupError) as refused:
            Backup.stage(folder, state_of(new), new)
        assert "bank.md" in str(refused.value)
        assert Backup.apply_pending(new) is None

    def test_a_folder_gone_since_it_was_staged_leaves_the_install_as_it_is(self, unpacked, tmp_path):
        """The copy a backup tool brought back to a temporary directory, cleaned up before the restart."""
        import shutil
        _, _, folder = unpacked
        new = tmp_path / "new"
        make_install(new, owner="Frances")
        Backup.stage(folder, state_of(new), new)
        shutil.rmtree(folder)
        assert "failed" in Backup.apply_pending(new)
        assert (new / "config.yaml").read_text() == "owner_name: Frances\n"
        assert history_rows(new / "floxy.db") == ["hello Frances"]
        assert Backup.apply_pending(new) is None
