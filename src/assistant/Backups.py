"""What makes an install this assistant, and the backups of it the assistant makes and restores on request.

`STATE` is everything a returning user has that a fresh install has not: what `//wipe` deletes
and what a backup holds, beside the sessions the integration folders keep in the tree
(`Integration.wipe_targets`). A backup is `common/Backup.py`'s zip of it in `~/KeepBackups`,
the newest `KEPT` kept; `backup`, `list_backups` and `restore_backup` (the `backup` toolkit,
`res/toolkits/backup.md`) are the commands. A restore backs the current state up first, so it
is undone by restoring that zip, and is put in place by the next start. Moving to a new machine
is the same thing: install Keep there, copy the zip over and ask for it by its path. Left out:
the browser profile (too big; a login brings it back), the logs and whatever lives outside the
install.

With `backup.format: folder` a backup is the folder `~/KeepBackups/current` instead — the
zip's files unpacked, replaced by each backup — for a file-level backup tool (restic, Time
Machine), which keeps the older versions itself. A restore takes that folder, or a copy of it
such a tool brought back, as it takes a zip.
"""
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from assistant import FirstRun
from assistant.integration.music.SpotifyAuth import TOKEN_PATH as SPOTIFY_TOKEN_PATH
from common import Backup
from common.ConfigManager import config
from common.logging import info, error
from common.Paths import AISERVER_DIR, CACHE_DIR, CONFIG_PATH, ICLOUD_SESSION_DIR, MEMORY_DIR, VAULT_PATH, VOICE_ID_DIR

DIRECTORY = Path.home() / "KeepBackups"
KEPT = 10
FOLDER_NAME = "current"
FORMAT_KEY = "backup.format"
FORMATS = (ZIP, FOLDER) = ("zip", "folder")
# One zip at a time: a reply that asks for two backups would have both write the same file.
_writing = threading.Lock()

# Files and directories alike; a missing one is skipped.
STATE = [
    (AISERVER_DIR / "floxy.db", "database: dialog history, usage stats, photo index, feeds"),
    (CONFIG_PATH, "config.yaml (the next start writes a skeleton and asks for the required keys)"),
    (VAULT_PATH, "vault"),
    (MEMORY_DIR, "long-term memory: bank, procedures, memos, watch list"),
    (AISERVER_DIR / "memory.md", "long-term memory bank of an install from before memory/"),
    (FirstRun.STATE_FILE, "first-run pipeline state"),
    (ICLOUD_SESSION_DIR, "iCloud session, trust token and cookies"),
    (SPOTIFY_TOKEN_PATH, "Spotify sign-in (refresh token) — the account every speaker system shares"),
    (CACHE_DIR / "config-backups", "config backups (hold the old credentials)"),
    (CACHE_DIR / "last_startup.json", "startup tracker"),
    (CACHE_DIR / "tui_state.json", "TUI state"),
    (CACHE_DIR / "audio_restarts.json", "audio recovery state"),
    (CACHE_DIR / "last_autonomous_check.json", "when the last periodic check ran"),
    # The names are interfaces/app's (Devices.FILE_NAME, Transcript.FILE_NAME and MEDIA_FOLDER),
    # which the assistant does not import; tests/unit/interfaces/app/test_app_wipe.py holds them together.
    (CACHE_DIR / "app_devices.json", "the phones the app's notifications go to (push tokens and keys)"),
    (CACHE_DIR / "app_chat.json", "the app's chat transcript"),
    (CACHE_DIR / "app_chat_media", "the pictures, videos and command results of the app's chat"),
]
# Held by a backup beside STATE; `//wipe` leaves them alone.
BACKUP_ONLY = [VOICE_ID_DIR, CACHE_DIR / "oauth"]


# Whoever holds a part of STATE in memory and would write it back (at the stop, at its next
# change) registers how to let go of it; `//wipe` calls them once the files are gone.
_wipe_listeners: list[Callable[[], None]] = []


def on_wipe(let_go: Callable[[], None]) -> None:
    _wipe_listeners.append(let_go)


def off_wipe(let_go: Callable[[], None]) -> None:
    if let_go in _wipe_listeners:
        _wipe_listeners.remove(let_go)


def wiped() -> None:
    """The state's files have just been deleted: nothing in memory may bring one back."""
    info(f"Backups: telling {len(_wipe_listeners)} holder(s) of state in memory that it was wiped")
    for let_go in list(_wipe_listeners):
        try:
            let_go()
        except Exception as e:
            error(f"Backups: a holder of wiped state failed to let go of it: {type(e).__name__}: {e}")


def integration_targets() -> list[tuple[Path, str]]:
    """The signed-in state the integration folders keep in the tree, each folder's own
    `Integration.wipe_targets`, in discovery order."""
    from assistant.integration.bootstrap import integration_classes
    targets = []
    for cls in integration_classes():
        targets += cls.wipe_targets()
    return targets


def state() -> list[Path]:
    """Every path a backup holds and a restore replaces: the ones inside the install, each once —
    a file inside a directory of the list (the Spotify and Volvo tokens under `cache/oauth/`) is
    held by the directory."""
    paths = dict.fromkeys([path for path, _ in STATE + integration_targets()] + BACKUP_ONLY)
    inside = [path for path in paths if path.is_relative_to(AISERVER_DIR)]
    return [path for path in inside if not any(path != other and path.is_relative_to(other) for other in inside)]


def backups() -> list[Path]:
    """The backups of the folder, newest first."""
    return sorted(DIRECTORY.glob("keep-*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)


def folder() -> Path:
    """The one unpacked backup, which each backup in the folder format replaces."""
    return DIRECTORY / FOLDER_NAME


def _format() -> str:
    """`backup.format`, read at each backup."""
    value = str(config.get(FORMAT_KEY, ZIP)).strip().lower()
    if value not in FORMATS:
        info(f"Backups: {FORMAT_KEY} '{value}' is neither of {FORMATS}, writing a zip")
        return ZIP
    return value


def _write(kind: str, unpacked: bool = False) -> Path:
    """The state as a new zip, or with *unpacked* as the folder: the zip is the way there and
    is gone again, since a backup tool that takes the whole directory would store every one."""
    with _writing:
        path = Backup.write(DIRECTORY / f"keep-{kind}-{datetime.now():%Y%m%d-%H%M%S}.zip", state())
        if not unpacked:
            return path
        try:
            return Backup.unpack(path, folder())
        finally:
            path.unlink(missing_ok=True)


def _megabytes(path: Path) -> float:
    return sum(f.stat().st_size for f in ([path] if path.is_file() else path.rglob("*")) if f.is_file()) / 1e6


def create() -> str:
    """Makes a backup now and deletes the ones beyond the newest KEPT; the answer for the LLM."""
    try:
        path = _write("backup", unpacked=_format() == FOLDER)
    except Exception as e:
        error(f"Backups: the backup failed: {type(e).__name__}: {e}")
        return f"The backup failed: {e}"
    for old in backups()[KEPT:]:
        info(f"Backups: deleting {old}, beyond the {KEPT} kept")
        old.unlink(missing_ok=True)
    return f"Backup written: {path} ({_megabytes(path):.0f} MB)."


def listing() -> str:
    found = backups() + ([folder()] if folder().is_dir() else [])
    if not found:
        info(f"Backups: none in {DIRECTORY}")
        return f"No backups in {DIRECTORY}."
    return "\n".join([f"Backups in {DIRECTORY}, newest first, and the unpacked folder when there is one:"] + [
        f"  {p.name}: {datetime.fromtimestamp(p.stat().st_mtime):%Y-%m-%d %H:%M}, {_megabytes(p):.0f} MB"
        for p in found])


def resolve(name: str) -> Optional[Path]:
    """The backup *name* means: one of the backup directory by its name, or a path — a zip, or
    a folder of a backup's files."""
    path = Path(name).expanduser()
    path = path if path.is_absolute() else DIRECTORY / path
    return path if name.strip() and (path.is_file() or path.is_dir()) else None


def restore(path: Path) -> str:
    """Backs the current state up and stages *path* for the next start; the problem as text,
    "" once it is staged."""
    try:
        undo = _write("before-restore")
    except Exception as e:
        error(f"Backups: no restore of {path.name}, the current state could not be backed up first: {e}")
        return f"the current state could not be backed up first ({e})"
    try:
        Backup.stage(path, state())
    except Backup.BackupError as e:
        info(f"Backups: no restore — {e}")
        undo.unlink(missing_ok=True)
        return str(e)
    return ""
