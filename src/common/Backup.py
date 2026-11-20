"""A backup of an install's state is one zip of paths relative to the install's root, written while the server runs and restored by the next start.

The zip is the one `//wipe` has always written and `scripts/migrate.sh` packs and unpacks: no
format of its own, so either reads the other's. An SQLite database goes in as a consistent copy
made by SQLite's backup API (other threads keep writing meanwhile), its sidecars left out. The
zip is not encrypted: it holds config.yaml, exactly as the install's own folder does, and is
the owner's alone (0600). `unpack` turns a zip into a folder of the same files, for a
file-level backup tool (restic, Time Machine) that dedupes a tree file by file where every zip
is new bytes; a restore takes either.

A restore cannot happen under a running server, which holds the database open and writes its
state back: `stage` checks the backup and leaves a marker, and `apply_pending` — the first thing
`floxy_ai_server.py` runs, ahead of the config — puts the backup's files beside the state the
install has and then swaps the two, so a restore that fails leaves the install as it was. Only
files under a path of the state are put there, so a backup never writes anywhere else in the
tree. The marker is good for the relaunch that follows it and no later start
(`STAGED_FOR_SECONDS`): a relaunch that never happened must not have a start days later roll the
install back.

The module reads no config and imports nothing of the assistant: it runs before either exists.
"""
import json
import os
import shutil
import sqlite3
import tempfile
import time
import zipfile
from contextlib import closing
from pathlib import Path
from typing import Optional

from common.Paths import AISERVER_DIR
from common.logging import info, error, trace

SQLITE_MAGIC = b"SQLite format 3\x00"
SIDECARS = ("-wal", "-shm", "-journal")
PENDING = Path("cache") / "restore-pending.json"
# How long a staged restore waits for its relaunch: start.sh stops the old server and starts the
# new one within a minute or two.
STAGED_FOR_SECONDS = 15 * 60
WORK_PREFIX = ".restore-"


class BackupError(Exception):
    """The backup cannot be restored; the message is fit to show the user."""


class RestoreHalfDone(BackupError):
    """A restore failed and the state it had moved aside could not all be put back."""


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


def unpack(zip_path: Path, folder: Path) -> Path:
    """The files of *zip_path* as the tree *folder*, the owner's alone, in place of the one
    there. It is written beside it and renamed, so whoever reads *folder* meanwhile (a backup
    tool on its own schedule) never finds half of one; what a run that died left beside it is
    removed first."""
    tmp, old = (folder.with_name(folder.name + suffix) for suffix in (".tmp", ".old"))
    for leftover in (tmp, old):
        if leftover.is_dir():
            info(f"Backup: removing {leftover}, left by a run that did not finish")
            shutil.rmtree(leftover)
    try:
        tmp.mkdir(mode=0o700, parents=True)    # closed before the first file is in it
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)
        for directory, _, names in os.walk(tmp):
            os.chmod(directory, 0o700)
            for name in names:
                os.chmod(Path(directory) / name, 0o600)    # a zip carries no modes
        if folder.exists():
            folder.rename(old)
        tmp.rename(folder)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        if old.is_dir() and not folder.exists():
            old.rename(folder)
        raise
    shutil.rmtree(old, ignore_errors=True)
    info(f"Backup: {zip_path.name} unpacked to {folder}")
    return folder


def _folder_files(folder: Path) -> list[str]:
    """The files of a backup folder by their names in a backup. A link is no part of one: it
    would bring in a file from outside the folder."""
    return sorted((Path(directory) / name).relative_to(folder).as_posix()
                  for directory, _, names in os.walk(folder) for name in names
                  if not (Path(directory) / name).is_symlink())


def _members(names: list[str], state: list[str]) -> list[str]:
    """The files of a backup a restore puts in place: the ones under a path of *state*."""
    return [name for name in names
            if not name.endswith("/") and ".." not in name.split("/")
            and any(name == rel or name.startswith(rel + "/") for rel in state)]


def stage(archive: Path, state: list[Path], root: Path = AISERVER_DIR) -> int:
    """Checks that *archive* — a zip, or a folder of the same files — is whole and holds
    something of *state* (the paths a restore replaces) and leaves the marker for the next
    start; the number of files it will put in place. Nothing of the install is touched."""
    rels = [p.relative_to(root).as_posix() for p in state if p.is_relative_to(root)]
    if archive.is_dir():
        members = _members(_folder_files(archive), rels)
        damaged = next((name for name in members if not os.access(archive / name, os.R_OK)), None)
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
    marker.write_text(json.dumps({"archive": str(archive.resolve()), "state": rels, "staged": time.time()}))
    info(f"Backup: {archive} staged, {len(members)} file(s) to restore at the next start")
    return len(members)


def unstage(root: Path = AISERVER_DIR) -> bool:
    """Takes a staged restore back — the relaunch it waited for did not happen; whether one was staged."""
    marker = root / PENDING
    staged = marker.is_file()
    marker.unlink(missing_ok=True)
    info(f"Backup: {'the staged restore is taken back' if staged else 'no staged restore to take back'}")
    return staged


def apply_pending(root: Path = AISERVER_DIR) -> Optional[str]:
    """Puts a staged restore in place and says what happened; None when nothing is staged.
    The marker goes first, so a restore that fails is not tried again by every start. A marker
    older than `STAGED_FOR_SECONDS` is not applied: its relaunch failed, and the install has
    lived on since. A restore that fails at any point leaves the state as it was — a backup
    that is gone since it was staged (a folder brought back to a temporary directory) as well."""
    marker = root / PENDING
    if not marker.is_file():
        return None
    try:
        plan = json.loads(marker.read_text())
        marker.unlink()
        archive = Path(plan["archive"])
        age = time.time() - float(plan.get("staged") or 0)
        if age > STAGED_FOR_SECONDS:
            error(f"Backup: the restore of {archive} was staged {age / 60:.0f} min ago and its restart never "
                  f"happened — not applied")
            return (f"The restore of {archive.name} was not applied: the restart it was staged for never happened. "
                    f"Ask for the restore again if it is still wanted.")
        count = _replace_state(root, archive, plan["state"])
    except RestoreHalfDone as e:
        marker.unlink(missing_ok=True)
        error(f"Backup: the staged restore failed and the state is not whole: {e}")
        return f"The restore failed midway: {e}"
    except Exception as e:
        marker.unlink(missing_ok=True)
        error(f"Backup: the staged restore failed, the state is as it was: {type(e).__name__}: {e}")
        return f"The restore failed, nothing was changed: {e}"
    info(f"Backup: {archive} restored, {count} file(s) in place")
    return f"Restored the backup {archive.name}: {count} files."


def _replace_state(root: Path, archive: Path, state: list[str]) -> int:
    """Puts the backup's part of *state* into a work directory of the install (the owner's
    alone, on the state's own volume), then swaps it with the state the install has; the
    number of files. The work directory goes whether it worked or not, with two exceptions
    that leave it with the replaced state inside: a process killed in the middle, and a
    failed swap that could not be undone (`RestoreHalfDone`) — there it is the only copy."""
    work = Path(tempfile.mkdtemp(prefix=WORK_PREFIX, dir=root))
    try:
        if archive.is_dir():
            members = _members(_folder_files(archive), state)
            for name in members:
                (work / "new" / name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(archive / name, work / "new" / name)
        else:
            with zipfile.ZipFile(archive) as zf:
                members = _members(zf.namelist(), state)
                zf.extractall(work / "new", members)
        if not members:
            raise BackupError(f"{archive} no longer holds anything of a Keep install")
        for name in members:
            os.chmod(work / "new" / name, 0o600)    # a zip carries no modes, and config.yaml is among them
        _swap(root, work, _outermost(state))
    except RestoreHalfDone:
        raise
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise
    shutil.rmtree(work, ignore_errors=True)
    return len(members)


def _outermost(state: list[str]) -> list[str]:
    """*state* without the paths under another of its paths: those move with their directory."""
    return [rel for rel in dict.fromkeys(state)
            if not any(rel != other and rel.startswith(other + "/") for other in state)]


def _swap(root: Path, work: Path, state: list[str]):
    """Moves every path of *state* (and the sidecars of a database: they would belong to the
    install the restore replaces) out to `work/old` and the unpacked one in from `work/new`;
    a failure moves back what was moved."""
    aside, placed = [], []
    try:
        for rel in state:
            for name in [rel] + [rel + s for s in SIDECARS]:
                if (root / name).exists() or (root / name).is_symlink():
                    _move(root / name, work / "old" / name)
                    aside.append(name)
        for rel in state:
            if (work / "new" / rel).exists():
                placed.append(rel)
                _move(work / "new" / rel, root / rel)
    except BaseException as e:
        error(f"Backup: the swap failed ({type(e).__name__}: {e}) — putting the {len(aside)} moved path(s) back")
        try:
            for rel in placed:
                _remove(root / rel)
            for name in aside:
                _move(work / "old" / name, root / name)
        except Exception as undo:
            error(f"Backup: the moved paths could not be put back ({type(undo).__name__}: {undo}) — "
                  f"{work / 'old'} is kept, it holds what is missing")
            raise RestoreHalfDone(f"{e}, and the state could not be put back ({undo}): what was replaced "
                                  f"is in {work / 'old'}") from e
        raise


def _move(source: Path, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    source.rename(target)


def _remove(path: Path):
    """A path of the state and the sidecars of a database: what stays would belong to the
    install the restore replaces."""
    for item in [path] + [path.with_name(path.name + s) for s in SIDECARS]:
        if item.is_dir() and not item.is_symlink():
            shutil.rmtree(item)
        elif item.exists() or item.is_symlink():
            item.unlink()
