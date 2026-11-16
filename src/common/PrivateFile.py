"""Writes the files that hold secrets — config.yaml and its backups — readable by their owner only.

config.yaml carries the API keys and the vault passphrase, and the installer writes it 0600. A
rewrite through a temp file would hand it the process's umask instead (0644, 0664 on Ubuntu), so
every writer of the file goes through here."""
import os
import shutil
from pathlib import Path

from common.logging import trace

OWNER_ONLY = 0o600


def write(path: Path, text: str) -> None:
    """*text* as the new content of *path*, replaced in one step, with the mode the file had —
    owner-only when there was none."""
    path = Path(path)
    mode = (path.stat().st_mode & 0o777) if path.exists() else OWNER_ONLY
    tmp = path.with_suffix(path.suffix + ".tmp")
    # The temp file may be left over from a crash, with any mode: the open alone does not set it.
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, OWNER_ONLY)
    with os.fdopen(fd, "w") as f:
        f.write(text)
    os.chmod(tmp, mode)
    os.replace(tmp, path)
    trace(f"PrivateFile: wrote {path.name} with mode {mode:o}")


def copy(source: Path, target: Path) -> None:
    """A copy of *source* only the owner reads, whatever the source's mode is."""
    shutil.copy2(source, target)
    os.chmod(target, OWNER_ONLY)
    trace(f"PrivateFile: copied {Path(source).name} to {Path(target).name}, owner-only")
