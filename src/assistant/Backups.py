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

`NightlyFolder` (`backup.nightly_folder`, off by default) also writes the same state unpacked
to `~/KeepBackups/current` every night at `backup.nightly_time`, for restic or Time Machine to
pick up: one folder, replaced whole each night and no part of the KEPT zips. A restore takes
that folder, or a copy of it a restic restore made, as well as a zip.
"""
import json
import threading
from datetime import datetime, time as clock_time
from pathlib import Path
from typing import Optional

from assistant import FirstRun
from assistant.integration.music.SpotifyAuth import TOKEN_PATH as SPOTIFY_TOKEN_PATH
from common import Backup, Version
from common.ConfigManager import ConfigManager
from common.logging import info, error
from common.Paths import AISERVER_DIR, CACHE_DIR, CONFIG_PATH, ICLOUD_SESSION_DIR, MEMORY_DIR, VAULT_PATH, VOICE_ID_DIR

DIRECTORY = Path.home() / "KeepBackups"
KEPT = 10
FOLDER_NAME = "current"
NIGHTLY_FOLDER_KEY = "backup.nightly_folder"
NIGHTLY_TIME_KEY = "backup.nightly_time"
DEFAULT_NIGHTLY_TIME = "03:45"
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


def folder() -> Path:
    """The unpacked backup the nightly run writes."""
    return DIRECTORY / FOLDER_NAME


def write_folder() -> Path:
    """The unpacked backup, written now over the previous one (common/Backup.write_folder)."""
    with _writing:
        return Backup.write_folder(folder(), state(), keep_version=Version.version(),
                                   created=datetime.now().isoformat(timespec="seconds"))


def _folder_written() -> Optional[datetime]:
    """When the folder was written, by its manifest; None without a readable one."""
    try:
        return datetime.fromisoformat(json.loads((folder() / Backup.MANIFEST).read_text())["created"])
    except Exception:
        return None


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
    lines = [f"  {p.name}: {datetime.fromtimestamp(p.stat().st_mtime):%Y-%m-%d %H:%M}, {p.stat().st_size / 1e6:.0f} MB"
             for p in backups()]
    if (written := _folder_written()) is not None:
        lines.append(f"  {FOLDER_NAME}/: unpacked, replaced every night, written {written:%Y-%m-%d %H:%M}")
    if NightlyFolder.failing:
        lines.append(f"The last nightly backup folder failed: {NightlyFolder.failing}")
    if not lines:
        info(f"Backups: none in {DIRECTORY}")
        return f"No backups in {DIRECTORY}."
    return "\n".join([f"Backups in {DIRECTORY}, newest first:"] + lines)


def resolve(name: str) -> Optional[Path]:
    """The backup *name* means: a zip of the folder by its name, the unpacked folder, or the
    path of either (a folder restic or Time Machine brought back)."""
    path = Path(name).expanduser()
    path = path if path.is_absolute() else DIRECTORY / path
    return path if name and (path.is_file() or path.is_dir()) else None


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


class NightlyFolder:
    """Writes the unpacked backup folder once a night, at `backup.nightly_time`, while
    `backup.nightly_folder` is on (both read at every check, so neither needs a restart).

    It runs once per date at or after that time: a server that was down then writes it when it
    comes up later that day, and a restart after the night's folder is written does not write
    it again (the manifest's time says). A failure that lasts is said the way the integration
    monitor and the calendar say one: an error once, when it starts or its reason changes, info
    on every night after, and info when a write succeeds again. `failing` is what
    `list_backups` shows meanwhile."""
    CHECK_SECONDS = 60
    failing: str = ""     # the process's: the reason last reported, "" once a write worked

    def __init__(self):
        self._stop = threading.Event()
        self._thread = None
        self._tried_on = None

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="NightlyBackupFolder", daemon=True)
        self._thread.start()
        info("Starting NightlyBackupFolder...")

    def stop(self):
        self._stop.set()

    def _run(self):
        while not self._stop.is_set():
            try:
                self.tick(datetime.now())
            except Exception as e:
                error(f"NightlyBackupFolder error: {e}")
            self._stop.wait(self.CHECK_SECONDS)

    def tick(self, now: datetime) -> bool:
        """Writes the folder when it is due at *now*; whether it tried."""
        config = ConfigManager()
        if not config.get(NIGHTLY_FOLDER_KEY, False):
            return False
        at = _nightly_time(config.get(NIGHTLY_TIME_KEY, DEFAULT_NIGHTLY_TIME))
        if now.time() < at or self._tried_on == now.date():
            return False
        self._tried_on = now.date()
        due_since = datetime.combine(now.date(), at)
        written = _folder_written()
        if written is not None and written >= due_since:
            info(f"Backups: the nightly folder was written at {written:%H:%M} today, nothing to do")
            return False
        info(f"Backups: the nightly folder is due since {at:%H:%M}, writing {folder()}")
        try:
            write_folder()
        except Exception as e:
            self._report_failure(f"{type(e).__name__}: {e}")
            return True
        if NightlyFolder.failing:
            info(f"Backups: the nightly folder {folder()} is written again")
        NightlyFolder.failing = ""
        return True

    @staticmethod
    def _report_failure(reason: str):
        text = (f"Backups: the nightly folder {folder()} could not be written ({reason}); "
                f"the previous one is left as it was")
        if reason == NightlyFolder.failing:
            info(f"{text} — as already reported")
            return
        NightlyFolder.failing = reason
        error(text, outside=True)


def _nightly_time(value) -> clock_time:
    try:
        return clock_time.fromisoformat(str(value))
    except ValueError:
        info(f"Backups: {NIGHTLY_TIME_KEY} '{value}' is not a time like 03:45, using {DEFAULT_NIGHTLY_TIME}")
        return clock_time.fromisoformat(DEFAULT_NIGHTLY_TIME)
