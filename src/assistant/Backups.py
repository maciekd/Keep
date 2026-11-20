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
"""
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from assistant import FirstRun
from assistant.integration.music.SpotifyAuth import TOKEN_PATH as SPOTIFY_TOKEN_PATH
from common import Backup
from common.logging import info, error
from common.Paths import AISERVER_DIR, CACHE_DIR, CONFIG_PATH, ICLOUD_SESSION_DIR, MEMORY_DIR, VAULT_PATH, VOICE_ID_DIR

DIRECTORY = Path.home() / "KeepBackups"
KEPT = 10
# One zip at a time: a reply that asks for two backups would have both write the same file.
_writing = threading.Lock()

# Files and directories alike; a missing one is skipped.
STATE = [
    (AISERVER_DIR / "floxy.db", "database: dialog history, usage stats, photo index, feeds"),
    (CONFIG_PATH, "config.yaml (the next start writes a skeleton and asks for the required keys)"),
    (VAULT_PATH, "vault"),
    (MEMORY_DIR, "long-term memory: bank, procedures, watch list"),
    (AISERVER_DIR / "memory.md", "long-term memory bank of an install from before memory/"),
    (FirstRun.STATE_FILE, "first-run pipeline state"),
    (ICLOUD_SESSION_DIR, "iCloud session, trust token and cookies"),
    (SPOTIFY_TOKEN_PATH, "Spotify sign-in (refresh token) — the account every speaker system shares"),
    (CACHE_DIR / "config-backups", "config backups (hold the old credentials)"),
    (CACHE_DIR / "last_startup.json", "startup tracker"),
    (CACHE_DIR / "tui_state.json", "TUI state"),
    (CACHE_DIR / "audio_restarts.json", "audio recovery state"),
    (CACHE_DIR / "last_autonomous_check.json", "when the last periodic check ran"),
]
# Held by a backup beside STATE; `//wipe` leaves them alone.
BACKUP_ONLY = [VOICE_ID_DIR, CACHE_DIR / "oauth"]


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


def _write(kind: str) -> Path:
    with _writing:
        return Backup.write(DIRECTORY / f"keep-{kind}-{datetime.now():%Y%m%d-%H%M%S}.zip", state())


def create() -> str:
    """Makes a backup now and deletes the ones beyond the newest KEPT; the answer for the LLM."""
    try:
        path = _write("backup")
    except Exception as e:
        error(f"Backups: the backup failed: {type(e).__name__}: {e}")
        return f"The backup failed: {e}"
    for old in backups()[KEPT:]:
        info(f"Backups: deleting {old}, beyond the {KEPT} kept")
        old.unlink(missing_ok=True)
    return f"Backup written: {path} ({path.stat().st_size / 1e6:.0f} MB)."


def listing() -> str:
    found = backups()
    if not found:
        info(f"Backups: none in {DIRECTORY}")
        return f"No backups in {DIRECTORY}."
    return "\n".join([f"Backups in {DIRECTORY}, newest first:"] + [
        f"  {p.name}: {datetime.fromtimestamp(p.stat().st_mtime):%Y-%m-%d %H:%M}, {p.stat().st_size / 1e6:.0f} MB"
        for p in found])


def resolve(name: str) -> Optional[Path]:
    """The zip *name* means: a backup of the folder by its name, or a path."""
    path = Path(name).expanduser()
    path = path if path.is_absolute() else DIRECTORY / path
    return path if name and path.is_file() else None


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
