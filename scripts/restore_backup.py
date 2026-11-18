"""List the Keep backups and restore one into this install, with no server running.

    .venv/bin/python scripts/restore_backup.py --list
    .venv/bin/python scripts/restore_backup.py [--stop-server] [--yes] <name>|<path>|latest

The way in when the server is what broke — it will not start, so neither the chat nor the backup
page is there: `./keep.sh backups` and `./keep.sh restore <name>` run this over ssh as the user
the install belongs to, and that shell is the authorization (no chat approval). install.sh runs
it on a fresh install with the path of a file.

A name is a file of `backup.directory` (as config.yaml has it, default ~/KeepBackups); `latest`
is the newest there. The passphrase is KEEP_BACKUP_PASSPHRASE, else `backup.passphrase` of
config.yaml, else — or when that one does not open the file — asked for on the terminal; a
KEEP_BACKUP_PASSPHRASE that does not open it is never followed by a question. On a
terminal the restore is confirmed first unless --yes. --stop-server stops the server (./stop.sh)
once the file is known to open, since a restore must not replace the files under a running
server. What the backup replaces is kept in cache/pre-restore-<time>/.
Exit status: 0 restored (or listed), 3 wrong passphrase, 1 any other failure.
"""
import getpass
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common import Backup  # noqa: E402

EXIT_WRONG_PASSPHRASE = 3
USAGE = f"usage: restore_backup.py --list | [--stop-server] [--yes] <name>|<path{Backup.SUFFIX}>|latest"


def config_value(key: str) -> str:
    """A dotted key of the install's config.yaml ($FLOXY_CONFIG when set); "" when absent or unreadable."""
    path = Path(os.environ.get("FLOXY_CONFIG") or ROOT / "config.yaml")
    try:
        value = yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError):
        return ""
    for part in key.split("."):
        value = value.get(part) if isinstance(value, dict) else None
    return "" if value is None else str(value)


def directory() -> Path:
    return Path(config_value(Backup.DIRECTORY_KEY) or Backup.DEFAULT_DIRECTORY).expanduser()


def listing() -> str:
    found = Backup.found_in(directory())
    if not found:
        return f"No backups in {directory()}."
    lines = [f"{len(found)} backup(s) in {directory()}, newest first:"]
    for path in found:
        stat = path.stat()
        made = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
        lines.append(f"  {path.name}  {made}  {stat.st_size / 1e6:.1f} MB")
    return "\n".join(lines)


def resolve(name: str) -> Path | None:
    """A path to a file, a file of the backup directory by name (with or without the suffix), or the newest one."""
    if name == "latest":
        found = Backup.found_in(directory())
        return found[0] if found else None
    path = Path(name).expanduser()
    for candidate in (path, directory() / name, directory() / f"{name}{Backup.SUFFIX}"):
        if candidate.is_file():
            return candidate
    return None


def open_manifest(path: Path) -> tuple[dict, str] | None:
    """The manifest and the passphrase that opened it; None when none did (told on stderr)."""
    if given := os.environ.get("KEEP_BACKUP_PASSPHRASE"):
        tried = [given]
    else:
        tried = [p for p in (config_value(Backup.PASSPHRASE_KEY),) if p]
        if not tried or sys.stdin.isatty():
            tried.append(None)
    for passphrase in dict.fromkeys(tried):
        if passphrase is None:
            passphrase = getpass.getpass(f"Passphrase of {path.name}: ")
        try:
            return Backup.manifest(path, passphrase), passphrase
        except Backup.WrongPassphrase:
            continue
    print(f"Not restored: no passphrase given opens {path.name}", file=sys.stderr)
    return None


def main(argv: list[str]) -> int:
    args = argv[1:]
    if args == ["--list"]:
        print(listing())
        return 0
    stop_server, confirmed = "--stop-server" in args, "--yes" in args
    names = [a for a in args if a not in ("--stop-server", "--yes")]
    if len(names) != 1 or names[0].startswith("--"):
        print(USAGE, file=sys.stderr)
        return 1
    path = resolve(names[0])
    if path is None:
        print(f"No backup {names[0]} (neither a file nor one of {directory()}).\n{listing()}", file=sys.stderr)
        return 1
    try:
        opened = open_manifest(path)
    except (Backup.BackupError, OSError) as e:
        print(f"Not restored: {path} is no backup that opens ({e})", file=sys.stderr)
        return 1
    if opened is None:
        return EXIT_WRONG_PASSPHRASE
    manifest, passphrase = opened
    if sys.stdin.isatty() and not confirmed:
        answer = input(f"Restore {path.name}, made {manifest.get('created', 'at an unknown time')}"
                       f"{' by Keep ' + manifest['keep_version'] if manifest.get('keep_version') else ''}? It replaces "
                       f"this install's config, vault, memory, history and sign-ins [y/N] ")
        if answer.strip().lower() not in ("y", "yes"):
            print("Nothing restored.")
            return 1
    if stop_server:
        print("Stopping the server first …")
        subprocess.run([str(ROOT / "stop.sh")], cwd=ROOT, check=False)
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
