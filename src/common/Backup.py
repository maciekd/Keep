"""A Keep backup: one encrypted file holding everything that makes an install the assistant it is.

The file (`<name>.keepbackup`) is a `KEEPBACKUP1` line, a JSON header line (the scrypt salt and
work factor, the nonce prefix, a key check), then the sealed stream: chunks of at most
CHUNK_SIZE plaintext bytes, each a 4-byte length and its AES-256-GCM ciphertext. A chunk's nonce
is the prefix and its index, its associated data the header and whether it is the last chunk,
so a reordered, dropped or truncated chunk fails to open. The plaintext is a gzipped tar:
`manifest.json` first (the format, when, which Keep, the targets — the paths relative to the
repository root, present or not), then `files/<path>` for every file of the targets. An SQLite
database goes in as a consistent copy made by SQLite's backup API, its sidecars left out.

Restoring is two halves. `stage` opens the file and unpacks it under cache/restore-pending — the
running server keeps its files; `apply_pending` moves every target the install has to
cache/pre-restore-<time>/ and the staged files in their place, and runs at the very start of a
process (floxy_ai_server.py, ahead of the config), since the running server holds the database
open. `restore` is both at once, for a process that is no server (the installer).

The module imports nothing of the assistant: the installer's `scripts/restore_backup.py` uses
it before any config exists.
"""
import base64
import hashlib
import io
import json
import os
import shutil
import sqlite3
import tarfile
import tempfile
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Optional

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

from common.Paths import AISERVER_DIR
from common.logging import info, error

MAGIC = b"KEEPBACKUP1\n"
SUFFIX = ".keepbackup"
FORMAT = 1
CHUNK_SIZE = 1 << 20
# ~32 MB and a fraction of a second per key; read from the header on restore, so a backup
# opens whatever this is when it is read. The unit conftest lowers it.
SCRYPT_N = 2 ** 15
MAX_SCRYPT_N = 2 ** 20      # a header asking for more is refused, not computed
MAX_HEADER = 4096
SQLITE_MAGIC = b"SQLite format 3\x00"
SQLITE_SIDECARS = ("-wal", "-shm", "-journal")
# Relative to the root of the install. The staging area must never be inside a target.
STAGING = Path("cache") / "restore-pending"
PREVIOUS_PREFIX = "pre-restore-"
# What a backup may never write: the code and the tools of the install. A target under one
# of these (or a root file the scripts run) is refused, whoever made the file.
PROTECTED = (".git", ".venv", "src", "scripts", "tests", "tools", "res", "docs")
PROTECTED_SUFFIXES = (".py", ".sh", ".app")
# The config keys of the backups, read here too by scripts/restore_backup.py, which runs without a server.
PASSPHRASE_KEY = "backup.passphrase"
DIRECTORY_KEY = "backup.directory"
DEFAULT_DIRECTORY = "~/KeepBackups"


class BackupError(Exception):
    """The backup cannot be made or read; the message is fit to show the user."""


class WrongPassphrase(BackupError):
    def __init__(self):
        super().__init__("the passphrase does not open this backup")


@dataclass
class Written:
    path: Path
    files: int
    size: int


# --- the targets ------------------------------------------------------------------------

def relative_target(path: Path, root: Path) -> Optional[str]:
    """*path* as a target of a backup — its POSIX path relative to *root* — or None for one
    outside the root (an absolute path belongs to that machine alone)."""
    try:
        return Path(os.path.abspath(path)).relative_to(os.path.abspath(root)).as_posix()
    except ValueError:
        return None


def check_target(rel: str) -> str:
    """*rel* unchanged, or BackupError when a backup may not hold it."""
    parts = PurePosixPath(rel).parts
    if not parts or rel.startswith("/") or "\\" in rel or any(p in ("", ".", "..") for p in parts):
        raise BackupError(f"'{rel}' is not a path inside the install")
    if parts[0] in PROTECTED or (len(parts) == 1 and rel.endswith(PROTECTED_SUFFIXES)):
        raise BackupError(f"'{rel}' is part of Keep's code, which a backup never holds")
    staging = STAGING.parts
    if tuple(parts[:len(staging)]) == staging[:len(parts)]:
        raise BackupError(f"'{rel}' holds the restore's staging area")
    return rel


def _sidecars(path: Path) -> list[Path]:
    return [path.with_name(path.name + s) for s in SQLITE_SIDECARS]


def _is_sqlite(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(len(SQLITE_MAGIC)) == SQLITE_MAGIC
    except OSError:
        return False


def _is_sidecar(path: Path) -> bool:
    return any(path.name.endswith(s) and path.with_name(path.name[:-len(s)]).is_file() for s in SQLITE_SIDECARS)


def _files_of(path: Path) -> list[Path]:
    if path.is_symlink():
        return []
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.is_file() and not p.is_symlink())
    return []


# --- the sealed stream ------------------------------------------------------------------

def _derive(passphrase: str, salt: bytes, n: int) -> bytes:
    return Scrypt(salt=salt, length=32, n=n, r=8, p=1).derive(passphrase.encode())


def _key_check(key: bytes) -> str:
    return hashlib.sha256(b"keep-backup-key-check" + key).hexdigest()[:32]


class _Sealer(io.RawIOBase):
    """Writes what it is given to *out* as sealed chunks; `finish` seals the last one."""

    def __init__(self, out, key: bytes, prefix: bytes, aad: bytes):
        super().__init__()
        self._out, self._aead, self._prefix, self._aad = out, AESGCM(key), prefix, aad
        self._buffer = bytearray()
        self._index = 0

    def writable(self):
        return True

    def write(self, data) -> int:
        self._buffer += data
        while len(self._buffer) > CHUNK_SIZE:
            self._seal(bytes(self._buffer[:CHUNK_SIZE]), final=False)
            del self._buffer[:CHUNK_SIZE]
        return len(data)

    def finish(self):
        self._seal(bytes(self._buffer), final=True)
        self._buffer.clear()

    def _seal(self, data: bytes, final: bool):
        nonce = self._prefix + self._index.to_bytes(4, "big")
        sealed = self._aead.encrypt(nonce, data, self._aad + bytes([final]))
        self._out.write(len(sealed).to_bytes(4, "big") + sealed)
        self._index += 1


class _Opener(io.RawIOBase):
    """Reads the plaintext of a sealed stream; a chunk that does not open, or an end before
    the last chunk, is a BackupError."""

    def __init__(self, source, key: bytes, prefix: bytes, aad: bytes):
        super().__init__()
        self._source, self._aead, self._prefix, self._aad = source, AESGCM(key), prefix, aad
        self._chunk = memoryview(b"")
        self._offset = 0
        self._index = 0
        self._done = False

    def readable(self):
        return True

    def readinto(self, buffer) -> int:
        while self._offset >= len(self._chunk) and not self._done:
            self._next_chunk()
        n = min(len(buffer), len(self._chunk) - self._offset)
        buffer[:n] = self._chunk[self._offset:self._offset + n]
        self._offset += n
        return n

    def drain(self):
        """Reads to the end, so that a file cut after what the caller needed is noticed."""
        while self.read(CHUNK_SIZE):
            pass

    def _next_chunk(self):
        head = self._source.read(4)
        if len(head) < 4:
            raise BackupError("the backup file is cut short")
        size = int.from_bytes(head, "big")
        sealed = self._source.read(size)
        if size > CHUNK_SIZE + 16 or len(sealed) < size:
            raise BackupError("the backup file is damaged or cut short")
        nonce = self._prefix + self._index.to_bytes(4, "big")
        for final in (False, True):
            try:
                plain = self._aead.decrypt(nonce, sealed, self._aad + bytes([final]))
                break
            except InvalidTag:
                continue
        else:
            raise BackupError(f"the backup file is damaged (part {self._index + 1} does not open)")
        self._index += 1
        self._chunk, self._offset = memoryview(plain), 0
        if final:
            self._done = True
            if self._source.read(1):
                raise BackupError("the backup file has data after its end")


def _open_stream(source, passphrase: str) -> tuple[_Opener, dict]:
    if source.read(len(MAGIC)) != MAGIC:
        raise BackupError("this is not a Keep backup file")
    line = source.readline(MAX_HEADER)
    try:
        header = json.loads(line)
        salt = base64.b64decode(header["salt"])
        prefix = base64.b64decode(header["nonce"])
        n = int(header["n"])
    except (ValueError, KeyError, TypeError):
        raise BackupError("the backup file's header is damaged")
    if header.get("format") != FORMAT:
        raise BackupError(f"backup format {header.get('format')} is newer than this Keep reads; update Keep first")
    if len(prefix) != 8 or n < 2 or n > MAX_SCRYPT_N or n & (n - 1):
        raise BackupError("the backup file's header is damaged")
    key = _derive(passphrase, salt, n)
    if header.get("check") != _key_check(key):
        raise WrongPassphrase()
    return _Opener(source, key, prefix, MAGIC + line), header


# --- writing ----------------------------------------------------------------------------

def write(dest: Path, targets: list[Path], passphrase: str, root: Path = AISERVER_DIR,
          keep_version: str = "") -> Written:
    """Writes the backup of *targets* (absolute paths; one outside *root* is left out) to
    *dest*, readable by its owner alone. A file is written next to *dest* and renamed into
    place, so *dest* is never a partial backup."""
    if not passphrase:
        raise BackupError("no passphrase to encrypt the backup with")
    rels = []
    for target in targets:
        rel = relative_target(target, root)
        if rel is None:
            info(f"Backup: {target} is outside the install, left out")
            continue
        rels.append(check_target(rel))
    rels = list(dict.fromkeys(rels))
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".partial")
    salt, prefix = os.urandom(16), os.urandom(8)
    key = _derive(passphrase, salt, SCRYPT_N)
    header = json.dumps({"format": FORMAT, "kdf": "scrypt", "n": SCRYPT_N,
                         "salt": base64.b64encode(salt).decode(), "nonce": base64.b64encode(prefix).decode(),
                         "check": _key_check(key)}).encode() + b"\n"
    manifest = json.dumps({"format": FORMAT, "created": datetime.now().isoformat(timespec="seconds"),
                           "keep_version": keep_version, "targets": rels}, indent=1).encode()
    files = 0
    info(f"Backup: writing {len(rels)} target(s) to {dest}")
    try:
        fd = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as out:
            out.write(MAGIC + header)
            sealer = _Sealer(out, key, prefix, MAGIC + header)
            with tarfile.open(fileobj=sealer, mode="w|gz") as tar:
                _add_bytes(tar, "manifest.json", manifest)
                for rel in rels:
                    for path in _files_of(root / rel):
                        files += _add_file(tar, path, root)
            sealer.finish()
        os.replace(partial, dest)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    size = dest.stat().st_size
    info(f"Backup: {dest} written, {files} file(s), {size / 1e6:.1f} MB")
    return Written(dest, files, size)


def _add_bytes(tar: tarfile.TarFile, name: str, data: bytes):
    member = tarfile.TarInfo(name)
    member.size, member.mode, member.mtime = len(data), 0o600, int(datetime.now().timestamp())
    tar.addfile(member, io.BytesIO(data))


def _add_file(tar: tarfile.TarFile, path: Path, root: Path) -> int:
    if _is_sidecar(path):
        return 0
    arcname = "files/" + Path(os.path.abspath(path)).relative_to(os.path.abspath(root)).as_posix()
    try:
        if _is_sqlite(path):
            _add_database(tar, path, arcname, root)
        else:
            tar.add(path, arcname, recursive=False, filter=_owned_by_nobody)
    except FileNotFoundError:
        info(f"Backup: {path} went away while the backup was written, left out")
        return 0
    return 1


def _add_database(tar: tarfile.TarFile, path: Path, arcname: str, root: Path):
    """A consistent copy: other threads keep writing to the database while it is read."""
    scratch_dir = root / "cache"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    fd, scratch = tempfile.mkstemp(dir=scratch_dir, prefix="backup-", suffix=".snapshot")
    os.close(fd)
    try:
        with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as src, \
                closing(sqlite3.connect(scratch)) as dst:
            src.backup(dst)
        tar.add(scratch, arcname, recursive=False, filter=_owned_by_nobody)
    finally:
        Path(scratch).unlink(missing_ok=True)


def _owned_by_nobody(member: tarfile.TarInfo) -> tarfile.TarInfo:
    member.uid = member.gid = 0
    member.uname = member.gname = ""
    member.mode = 0o600
    return member


# --- reading ----------------------------------------------------------------------------

def found_in(folder: Path) -> list[Path]:
    """The backups of *folder*, newest first."""
    if not folder.is_dir():
        return []
    return sorted(folder.glob(f"*{SUFFIX}"), key=lambda p: p.stat().st_mtime, reverse=True)


def manifest(path: Path, passphrase: str) -> dict:
    """The manifest of the backup at *path*: what it holds and when it was made."""
    with open(path, "rb") as source:
        opener, _ = _open_stream(source, passphrase)
        with tarfile.open(fileobj=opener, mode="r|gz") as tar:
            first = tar.next()
            return _read_manifest(tar, first)


def _read_manifest(tar: tarfile.TarFile, member) -> dict:
    if member is None or member.name != "manifest.json" or not member.isfile():
        raise BackupError("the backup holds no manifest")
    try:
        data = json.load(tar.extractfile(member))
        targets = [check_target(str(t)) for t in data["targets"]]
    except (ValueError, KeyError, TypeError):
        raise BackupError("the backup's manifest is damaged")
    if data.get("format") != FORMAT:
        raise BackupError(f"backup format {data.get('format')} is newer than this Keep reads; update Keep first")
    return {**data, "targets": targets}


def stage(path: Path, passphrase: str, root: Path = AISERVER_DIR) -> dict:
    """Opens the backup at *path* and unpacks it into the staging area of *root*, for
    `apply_pending` to put in place; the manifest, with `files` the count unpacked. Nothing of
    the install is touched, and a backup that does not open whole leaves nothing staged."""
    staging = root / STAGING
    discard_staged(root)
    (staging / "files").mkdir(parents=True, mode=0o700)
    try:
        with open(path, "rb") as source:
            opener, _ = _open_stream(source, passphrase)
            with tarfile.open(fileobj=opener, mode="r|gz") as tar:
                data, files = None, 0
                for member in tar:
                    if data is None:
                        data = _read_manifest(tar, member)
                        continue
                    _check_member(member, data["targets"])
                    if member.isfile():
                        tar.extract(member, staging, filter="data")
                        files += 1
            opener.drain()
        if data is None:
            raise BackupError("the backup holds no manifest")
    except (tarfile.TarError, EOFError, OSError) as e:
        discard_staged(root)
        raise BackupError(f"the backup file is damaged ({e})")
    except BaseException:
        discard_staged(root)
        raise
    data["files"] = files
    (staging / "manifest.json").write_text(json.dumps(data, indent=1))
    info(f"Backup: {path} staged, {files} file(s) of {len(data['targets'])} target(s), made {data.get('created')}")
    return data


def _check_member(member: tarfile.TarInfo, targets: list[str]):
    name = PurePosixPath(member.name)
    if name.parts[:1] != ("files",) or len(name.parts) < 2:
        raise BackupError(f"the backup holds '{member.name}', which is no file of the install")
    rel = PurePosixPath(*name.parts[1:]).as_posix()
    if not any(rel == t or rel.startswith(t + "/") for t in targets):
        raise BackupError(f"the backup holds '{rel}', which none of its targets covers")
    check_target(rel)
    if not (member.isfile() or member.isdir()):
        raise BackupError(f"the backup holds '{rel}' as a link or a device, never a file of Keep")


def staged(root: Path = AISERVER_DIR) -> Optional[dict]:
    """The manifest of a staged restore waiting for the next start, or None."""
    marker = root / STAGING / "manifest.json"
    if not marker.is_file():
        return None
    try:
        return json.loads(marker.read_text())
    except (OSError, ValueError):
        return None


def discard_staged(root: Path = AISERVER_DIR):
    shutil.rmtree(root / STAGING, ignore_errors=True)


def apply_pending(root: Path = AISERVER_DIR) -> Optional[str]:
    """Puts a staged restore in place: every target the install has goes to
    cache/pre-restore-<time>/, then the staged files take their places. Returns what happened,
    None when nothing is staged. Run before anything opens the files — the start of a process."""
    data = staged(root)
    if data is None:
        if (root / STAGING).exists():
            info("Backup: an incomplete staging area found, discarded")
            discard_staged(root)
        return None
    staging = root / STAGING
    aside = root / "cache" / f"{PREVIOUS_PREFIX}{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    try:
        targets = [check_target(t) for t in data["targets"]]
        moved = 0
        for rel in targets:
            current = root / rel
            for path in [current] + _sidecars(current):
                if path.exists() or path.is_symlink():
                    dest = aside / path.relative_to(root)
                    dest.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                    shutil.move(str(path), str(dest))
                    moved += 1
        restored = 0
        for staged_file in _files_of(staging / "files"):
            dest = root / staged_file.relative_to(staging / "files")
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(staged_file), str(dest))
            os.chmod(dest, 0o600)
            restored += 1
    except Exception as e:
        error(f"Backup: the staged restore could not be put in place: {e}; what was replaced is in {aside}")
        failed = staging.with_name(staging.name + "-failed")
        shutil.rmtree(failed, ignore_errors=True)
        staging.rename(failed)
        return f"The restore failed ({e}); the files it replaced are in {aside}"
    discard_staged(root)
    if aside.exists():
        os.chmod(aside, 0o700)
    outcome = (f"Restored the backup made {data.get('created', '?')}: {restored} file(s) in place"
               + (f", the {moved} item(s) they replaced kept in {aside.relative_to(root)}" if moved else ""))
    info(f"Backup: {outcome}")
    return outcome


def restore(path: Path, passphrase: str, root: Path = AISERVER_DIR) -> str:
    """Stages and applies at once — for a process that is not the server (the installer)."""
    stage(path, passphrase, root)
    return apply_pending(root) or "Nothing was restored"
