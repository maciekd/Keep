"""What a Keep backup holds, and the backup and restore the assistant runs on request.

The format is `common/Backup.py`'s; this module decides the targets — the core's state
(config.yaml with every key, the vault, the database, the long-term memory, the first-run
state, the voice ids, the OAuth and iCloud sign-ins) and every integration's `wipe_targets`
inside the install — and where backups go: `backup.directory` (default ~/KeepBackups, outside
the checkout), the newest `backup.keep` kept. The passphrase is `backup.passphrase`, typed by
the owner. A restore is staged here and put in place by the next start (floxy_ai_server.py).
What a backup leaves out: the logs, the browser profile (too big; a login brings it back), the
caches, and anything outside the install (extra_integration_dirs is code of the install's own).
"""
import hmac
from datetime import datetime
from pathlib import Path
from typing import Optional

from assistant import FirstRun
from assistant.integration.music.SpotifyAuth import TOKEN_PATH as SPOTIFY_TOKEN_PATH
from common import Backup, Version
from common.Backup import BackupError
from common.ConfigManager import ConfigManager
from common.logging import info, error
from common.Paths import (AISERVER_DIR, CACHE_DIR, CONFIG_PATH, ICLOUD_SESSION_DIR, MEMORY_DIR, VAULT_PATH,
                          VOICE_ID_DIR)

PASSPHRASE_KEY = "backup.passphrase"
DIRECTORY_KEY = "backup.directory"
KEEP_KEY = "backup.keep"
DEFAULT_DIRECTORY = "~/KeepBackups"
DEFAULT_KEEP = 10
MIN_PASSPHRASE_LENGTH = 8
LISTED = 10

CORE_TARGETS = (
    CONFIG_PATH,
    VAULT_PATH,
    AISERVER_DIR / "floxy.db",
    MEMORY_DIR,
    AISERVER_DIR / "memory.md",
    FirstRun.STATE_FILE,
    VOICE_ID_DIR,
    CACHE_DIR / "oauth",
    ICLOUD_SESSION_DIR,
    SPOTIFY_TOKEN_PATH,
)


def targets() -> list[Path]:
    """The core's targets, then each integration's sign-in files that live in the install."""
    from assistant.integration.bootstrap import integration_classes
    paths = list(CORE_TARGETS)
    for cls in integration_classes():
        try:
            paths += [path for path, _ in cls.wipe_targets()]
        except Exception as e:
            error(f"Backups: {cls.__name__}.wipe_targets raised {type(e).__name__}: {e}, left out")
    inside = [p for p in paths if Backup.relative_target(p, AISERVER_DIR) is not None]
    return list(dict.fromkeys(inside))


def directory() -> Path:
    return Path(str(ConfigManager.get_instance().get(DIRECTORY_KEY, DEFAULT_DIRECTORY) or DEFAULT_DIRECTORY)).expanduser()


def _passphrase() -> str:
    return str(ConfigManager.get_instance().get(PASSPHRASE_KEY, "") or "")


def _keep() -> int:
    try:
        return max(1, int(ConfigManager.get_instance().get(KEEP_KEY, DEFAULT_KEEP)))
    except (TypeError, ValueError):
        return DEFAULT_KEEP


def passphrase_matches(typed: str) -> bool:
    """Whether *typed* is the configured backup passphrase (never true while none is set)."""
    configured = _passphrase()
    return bool(configured) and hmac.compare_digest(typed.encode(), configured.encode())


def passphrase_problem() -> str:
    """Why no backup can be made or opened with the configured passphrase; "" when one can."""
    passphrase = _passphrase()
    if not passphrase:
        return (f"no backup passphrase is set. Ask the user for one with config_request(\"{PASSPHRASE_KEY}\") — "
                f"at least {MIN_PASSPHRASE_LENGTH} characters, kept in their password manager: no backup opens "
                f"without it")
    if len(passphrase) < MIN_PASSPHRASE_LENGTH:
        return (f"the backup passphrase is shorter than {MIN_PASSPHRASE_LENGTH} characters; ask the user for a "
                f"longer one with config_request(\"{PASSPHRASE_KEY}\")")
    return ""


def backups() -> list[Path]:
    """The backups of the backup directory, newest first."""
    folder = directory()
    if not folder.is_dir():
        return []
    return sorted(folder.glob(f"*{Backup.SUFFIX}"), key=lambda p: p.stat().st_mtime, reverse=True)


def create() -> str:
    """Makes a backup now; the answer for the LLM."""
    if problem := passphrase_problem():
        info(f"Backups: no backup made — {problem}")
        return f"No backup made: {problem}."
    folder = directory()
    if Backup.relative_target(folder, AISERVER_DIR) is not None:
        info(f"Backups: {folder} is inside the install, refused")
        return (f"No backup made: {DIRECTORY_KEY} ({folder}) is inside the Keep folder, where a reinstall "
                f"would lose it. Ask the user for a folder outside it with config_request(\"{DIRECTORY_KEY}\").")
    dest = folder / f"keep-{datetime.now().strftime('%Y%m%d-%H%M%S')}{Backup.SUFFIX}"
    try:
        written = Backup.write(dest, targets(), _passphrase(), AISERVER_DIR, Version.version())
    except (BackupError, OSError) as e:
        error(f"Backups: the backup failed: {e}")
        return f"The backup failed: {e}"
    pruned = _prune()
    return (f"Backup written: {written.path} ({written.size / 1e6:.1f} MB, {written.files} files), encrypted "
            f"with the backup passphrase."
            + (f" Deleted {pruned} older backup(s) beyond the {_keep()} kept." if pruned else "")
            + " To restore it on a new machine: the installer asks for the file and the passphrase.")


def _prune() -> int:
    pruned = 0
    for old in backups()[_keep():]:
        try:
            old.unlink()
            pruned += 1
            info(f"Backups: deleted {old}, beyond the {_keep()} kept")
        except OSError as e:
            error(f"Backups: could not delete {old}: {e}")
    return pruned


def listing() -> str:
    found = backups()
    if not found:
        info(f"Backups: none in {directory()}")
        return f"No backups in {directory()}."
    lines = [f"{len(found)} backup(s) in {directory()}, newest first:"]
    for path in found[:LISTED]:
        stat = path.stat()
        made = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
        lines.append(f"  {path.name} — {made}, {stat.st_size / 1e6:.1f} MB")
    if len(found) > LISTED:
        lines.append(f"  … and {len(found) - LISTED} older")
    return "\n".join(lines)


def resolve(name: str) -> Optional[Path]:
    """The backup *name* means: a file of the backup directory by its name, or a path; the
    newest backup when *name* is empty or "latest"."""
    name = name.strip().strip("\"'")
    if name in ("", "latest"):
        found = backups()
        return found[0] if found else None
    path = Path(name).expanduser()
    if not path.is_absolute():
        path = directory() / path
    if not path.is_file() and path.with_name(path.name + Backup.SUFFIX).is_file():
        path = path.with_name(path.name + Backup.SUFFIX)
    return path if path.is_file() else None


def stage(path: Path) -> str:
    """Unpacks *path* for the next start; the problem as text, "" when it is staged."""
    if problem := passphrase_problem():
        return problem
    try:
        data = Backup.stage(path, _passphrase(), AISERVER_DIR)
    except Backup.WrongPassphrase:
        info(f"Backups: {path} does not open with the configured passphrase")
        return (f"the backup passphrase does not open {path.name}. If it was made with another passphrase, ask the "
                f"user to type that one with config_request(\"{PASSPHRASE_KEY}\") and try again")
    except (BackupError, OSError) as e:
        error(f"Backups: {path} could not be staged: {e}")
        return str(e)
    info(f"Backups: {path.name} staged ({data['files']} files, made {data.get('created')}), applied at the next start")
    return ""
