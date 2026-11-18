"""Restore a Keep backup (.keepbackup) into this install, with the server stopped.

    .venv/bin/python scripts/restore_backup.py <file>

install.sh runs it on a fresh install when the user names a backup. The passphrase comes from
KEEP_BACKUP_PASSPHRASE, else it is asked for on the terminal. What the backup replaces is kept in
cache/pre-restore-<time>/. Exit status: 0 restored, 3 wrong passphrase, 1 any other failure.
A running server restores with its restore_backup command instead: this script would replace
the files under it.
"""
import getpass
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common import Backup  # noqa: E402

EXIT_WRONG_PASSPHRASE = 3


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0]} <file{Backup.SUFFIX}>", file=sys.stderr)
        return 1
    path = Path(argv[1]).expanduser()
    if not path.is_file():
        print(f"No backup file at {path}", file=sys.stderr)
        return 1
    passphrase = os.environ.get("KEEP_BACKUP_PASSPHRASE") or getpass.getpass("Backup passphrase: ")
    try:
        print(Backup.restore(path, passphrase, ROOT))
    except Backup.WrongPassphrase as e:
        print(f"Not restored: {e}", file=sys.stderr)
        return EXIT_WRONG_PASSPHRASE
    except (Backup.BackupError, OSError) as e:
        print(f"Not restored: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
