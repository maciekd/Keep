"""A backup of an install's state is one zip of paths relative to the install's root, written while the server runs and restored by the next start.

The zip is the one `//wipe` has always written and `scripts/migrate.sh` packs and unpacks: no
format of its own, so either reads the other's. An SQLite database goes in as a consistent copy
made by SQLite's backup API (other threads keep writing meanwhile), its sidecars left out. The
zip is not encrypted: it holds config.yaml, exactly as the install's own folder does, and is
the owner's alone (0600). `write_folder` writes the same files unpacked, with a manifest of
their sha256, for a file-level backup tool to pick up; a restore takes either.

A restore cannot happen under a running server, which holds the database open and writes its
state back: `stage` checks the zip and leaves a marker, and `apply_pending` — the first thing
`floxy_ai_server.py` runs, ahead of the config — removes the state the install has and unpacks
the zip's in its place. Only members under a path of the state are unpacked, so a zip never
writes anywhere else in the tree.

The module reads no config and imports nothing of the assistant: it runs before either exists.
"""
import hashlib
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
MANIFEST = "manifest.json"
MANIFEST_FORMAT = 1
CHUNK = 1 << 20


class BackupError(Exception):
    """The backup cannot be restored; the message is fit to show the user."""


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


def _members(names: list[str], state: list[str]) -> list[str]:
    """The files of a backup a restore puts in place: the ones under a path of *state*."""
    return [name for name in names
            if not name.endswith("/") and ".." not in name.split("/")
            and any(name == rel or name.startswith(rel + "/") for rel in state)]


def write_folder(folder: Path, paths: list[Path], root: Path = AISERVER_DIR,
                 keep_version: str = "unknown", created: str = "") -> Path:
    """The same files as `write`, unpacked into *folder*, with `manifest.json` naming each
    one's size and sha256: a tree a file-level backup (restic, Time Machine) dedupes file by
    file, where a zip is new bytes every time. A database is a backup-API copy, as in the zip.

    The tree is written whole to `<folder>.tmp`, fsynced, and only then swapped in: *folder*
    moves aside to `<folder>.old`, the new one takes its name, the old one is deleted. A crash
    leaves either tree whole, never a half-written *folder*; a `.old` left without a *folder*
    (a crash between the two renames) is put back by the next write and by the error path."""
    tmp, old = _beside(folder, ".tmp"), _beside(folder, ".old")
    folder.parent.mkdir(parents=True, exist_ok=True)
    _put_back(folder, old)
    _rmtree(tmp)
    info(f"Backup: writing {len(paths)} item(s) unpacked to {tmp}")
    try:
        tmp.mkdir(mode=0o700)
        entries = []
        for file, arcname in files(paths, root):
            target = tmp / arcname
            _private_dirs(target.parent, tmp)
            if _is_sqlite(file):
                _copy_database(file, target)
            else:
                _copy_file(file, target)
            entries.append({"path": arcname, "size": target.stat().st_size, "sha256": _sha256(target)})
        manifest = {"format": MANIFEST_FORMAT, "keep_version": keep_version, "created": created, "files": entries}
        _write_private(tmp / MANIFEST, json.dumps(manifest, indent=1).encode())
        _fsync_dirs(tmp)
        if folder.exists():
            folder.rename(old)
        tmp.rename(folder)
        _fsync(folder.parent)
        _rmtree(old)
    except BaseException:
        _rmtree(tmp)
        _put_back(folder, old)
        raise
    info(f"Backup: {folder} written, {len(entries)} file(s), {sum(e['size'] for e in entries) / 1e6:.1f} MB")
    return folder


def verify_folder(folder: Path) -> list[str]:
    """The files *folder*'s manifest lists, once every one of them is there with its size and
    sha256; a BackupError naming what is missing or different otherwise. Reads, never writes."""
    try:
        manifest = json.loads((folder / MANIFEST).read_text())
        entries = [(e["path"], int(e["size"]), e["sha256"]) for e in manifest["files"]]
    except FileNotFoundError:
        raise BackupError(f"{folder} has no {MANIFEST}, so it is not a Keep backup folder")
    except Exception as e:
        raise BackupError(f"the {MANIFEST} of {folder} does not read ({type(e).__name__}: {e})")
    problems = []
    for rel, size, digest in entries:
        path = folder / rel
        if rel.startswith("/") or ".." in rel.split("/") or rel == MANIFEST:
            problems.append(f"{rel} (not a path inside the folder)")
        elif not path.is_file() or path.is_symlink():
            problems.append(f"{rel} (missing)")
        elif path.stat().st_size != size or _sha256(path) != digest:
            problems.append(f"{rel} (not what the manifest says: changed or damaged)")
    if problems:
        shown = ", ".join(problems[:5]) + (f" and {len(problems) - 5} more" if len(problems) > 5 else "")
        raise BackupError(f"{folder} does not match its {MANIFEST}: {shown}. Nothing was restored")
    trace(f"Backup: {folder} matches its {MANIFEST}, {len(entries)} file(s)")
    return [rel for rel, _, _ in entries]


def _beside(folder: Path, suffix: str) -> Path:
    return folder.with_name(folder.name + suffix)


def _put_back(folder: Path, old: Path):
    if old.is_dir() and not folder.exists():
        info(f"Backup: {folder} is missing and {old} is whole, putting it back")
        old.rename(folder)


def _rmtree(path: Path):
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)


def _private_dirs(directory: Path, top: Path):
    for d in reversed([directory, *directory.parents]):
        if d.is_relative_to(top) and not d.exists():
            d.mkdir(mode=0o700)


def _write_private(path: Path, data: bytes):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def _copy_file(source: Path, target: Path):
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with open(source, "rb") as src, os.fdopen(fd, "wb") as dst:
        shutil.copyfileobj(src, dst, CHUNK)
        dst.flush()
        os.fsync(dst.fileno())


def _copy_database(source: Path, target: Path):
    """A consistent copy through SQLite's backup API, however busy the source is: it holds
    what was committed when the copy finished, never a page of a write in flight."""
    os.close(os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600))
    with closing(sqlite3.connect(source)) as src, closing(sqlite3.connect(target)) as dst:
        src.backup(dst)
    _fsync(target)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync(path: Path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_dirs(top: Path):
    """The directory entries too: an fsynced file in a directory whose entry is not is lost
    by a crash all the same."""
    for directory in sorted((d for d in top.rglob("*") if d.is_dir()), reverse=True):
        _fsync(directory)
    _fsync(top)


def stage(archive: Path, state: list[Path], root: Path = AISERVER_DIR) -> int:
    """Checks that *archive* — a zip, or a folder `write_folder` wrote — is whole and holds
    something of *state* (the paths a restore replaces) and leaves the marker for the next
    start; the number of files it will put in place. A folder is whole when every file of its
    manifest has its sha256. Nothing of the install is touched."""
    rels = [p.relative_to(root).as_posix() for p in state if p.is_relative_to(root)]
    damaged = None
    if archive.is_dir():
        members = _members(verify_folder(archive), rels)
    else:
        try:
            with zipfile.ZipFile(archive) as zf:
                damaged = zf.testzip()
                members = _members(zf.namelist(), rels)
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
        if archive.is_dir():
            # Checked again: the folder (a restic restore in a temp dir) may have changed since staging.
            members = _members(verify_folder(archive), plan["state"])
            for rel in plan["state"]:
                _remove(root / rel)
            for name in members:
                (root / name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(archive / name, root / name)
        else:
            with zipfile.ZipFile(archive) as zf:
                members = _members(zf.namelist(), plan["state"])
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
