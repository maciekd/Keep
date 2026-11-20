"""A backup of an install's state is one zip of paths relative to the install's root, written while the server runs and restored by the next start.

The zip is the one `//wipe` has always written and `scripts/migrate.sh` packs and unpacks: no
format of its own, so either reads the other's. An SQLite database goes in as a consistent copy
made by SQLite's backup API (other threads keep writing meanwhile), its sidecars left out. The
zip is not encrypted: it holds config.yaml, exactly as the install's own folder does, and is
the owner's alone (0600).

A restore cannot happen under a running server, which holds the database open and writes its
state back: `stage` checks the zip and leaves a marker, and `apply_pending` — the first thing
`floxy_ai_server.py` runs, ahead of the config — removes the state the install has and unpacks
the zip's in its place. Only members under a path of the state are unpacked, so a zip never
writes anywhere else in the tree.

The module reads no config and imports nothing of the assistant: it runs before either exists.
"""
import json
import os
import shutil
import sqlite3
import zipfile
from contextlib import closing
from pathlib import Path
from typing import Optional

from common.Paths import AISERVER_DIR
from common.logging import info, error, trace

SQLITE_MAGIC = b"SQLite format 3\x00"
SIDECARS = ("-wal", "-shm", "-journal")
PENDING = Path("cache") / "restore-pending.json"


class BackupError(Exception):
    """The zip cannot be restored; the message is fit to show the user."""


def write(zip_path: Path, paths: list[Path], root: Path = AISERVER_DIR) -> Path:
    """One zip of every file of *paths* (files and directories; a missing one is skipped),
    readable by its owner alone. A path outside *root* goes under `external/`."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    info(f"Backup: writing {len(paths)} item(s) to {zip_path}")
    fd = os.open(zip_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "wb") as raw, zipfile.ZipFile(raw, "w", zipfile.ZIP_DEFLATED) as zf:
            for file, arcname in files(paths, root):
                if _is_sqlite(file):
                    _write_database(zf, file, arcname, zip_path.with_suffix(".snapshot"))
                else:
                    zf.write(file, arcname)
    except BaseException:
        zip_path.unlink(missing_ok=True)    # half a zip must never be listed as a backup
        raise
    info(f"Backup: {zip_path} written, {zip_path.stat().st_size / 1e6:.1f} MB")
    return zip_path


def files(paths: list[Path], root: Path = AISERVER_DIR) -> list[tuple[Path, str]]:
    """Every file a backup of *paths* holds, with its name in the backup, each once: a file
    named on its own and again inside a directory of *paths* (a sign-in token under
    `cache/oauth/`) would otherwise go in twice. Sidecars of a database are left out."""
    found = {}
    for path in paths:
        for file in ([path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())):
            arcname = _arcname(file, root)
            if arcname in found:
                trace(f"Backup: {arcname} is already in the backup, not added twice")
            elif not _is_sidecar(file):
                found[arcname] = file
    return [(file, arcname) for arcname, file in found.items()]


def _write_database(zf: zipfile.ZipFile, path: Path, arcname: str, snapshot: Path):
    """*snapshot* is this zip's own scratch file, named after it: two backups never share one."""
    snapshot.unlink(missing_ok=True)
    try:
        with closing(sqlite3.connect(path)) as src, closing(sqlite3.connect(snapshot)) as dst:
            src.backup(dst)
        zf.write(snapshot, arcname)
    finally:
        snapshot.unlink(missing_ok=True)


def _arcname(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return "external/" + str(path).lstrip("/")


def _is_sqlite(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(len(SQLITE_MAGIC)) == SQLITE_MAGIC
    except OSError:
        return False


def _is_sidecar(path: Path) -> bool:
    return any(path.name.endswith(s) and _is_sqlite(path.with_name(path.name[:-len(s)])) for s in SIDECARS)


def _members(zf: zipfile.ZipFile, state: list[str]) -> list[str]:
    """The files of the zip a restore unpacks: the ones under a path of *state*."""
    return [name for name in zf.namelist()
            if not name.endswith("/") and ".." not in name.split("/")
            and any(name == rel or name.startswith(rel + "/") for rel in state)]


def stage(archive: Path, state: list[Path], root: Path = AISERVER_DIR) -> int:
    """Checks that *archive* is a whole zip holding something of *state* (the paths a restore
    replaces) and leaves the marker for the next start; the number of files it will unpack.
    Nothing of the install is touched."""
    rels = [p.relative_to(root).as_posix() for p in state if p.is_relative_to(root)]
    try:
        with zipfile.ZipFile(archive) as zf:
            damaged = zf.testzip()
            members = _members(zf, rels)
    except Exception as e:     # zipfile, zlib, the disk: whatever it is, the zip does not read
        raise BackupError(f"{archive.name} is not a readable zip ({e})")
    if damaged:
        raise BackupError(f"{archive.name} is damaged ({damaged} does not read back)")
    if not members:
        raise BackupError(f"{archive.name} holds nothing of a Keep install")
    marker = root / PENDING
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"archive": str(archive.resolve()), "state": rels}))
    info(f"Backup: {archive} staged, {len(members)} file(s) to restore at the next start")
    return len(members)


def apply_pending(root: Path = AISERVER_DIR) -> Optional[str]:
    """Puts a staged restore in place and says what happened; None when nothing is staged.
    The marker goes first, so a restore that fails is not tried again by every start, and the
    zip is opened before anything is removed."""
    marker = root / PENDING
    if not marker.is_file():
        return None
    try:
        plan = json.loads(marker.read_text())
        marker.unlink()
        archive = Path(plan["archive"])
        with zipfile.ZipFile(archive) as zf:
            members = _members(zf, plan["state"])
            for rel in plan["state"]:
                _remove(root / rel)
            zf.extractall(root, members)
        for name in members:
            os.chmod(root / name, 0o600)    # a zip carries no modes, and config.yaml is among them
    except Exception as e:
        marker.unlink(missing_ok=True)
        error(f"Backup: the staged restore failed: {type(e).__name__}: {e}")
        return f"The restore failed: {e}"
    info(f"Backup: {archive} restored, {len(members)} file(s) in place")
    return f"Restored the backup {archive.name}: {len(members)} files."


def _remove(path: Path):
    """A path of the state and the sidecars of a database: what stays would belong to the
    install the restore replaces."""
    for item in [path] + [path.with_name(path.name + s) for s in SIDECARS]:
        if item.is_dir() and not item.is_symlink():
            shutil.rmtree(item)
        elif item.exists() or item.is_symlink():
            item.unlink()
