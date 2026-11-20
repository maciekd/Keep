A backup is one zip of everything that makes you this assistant: config.yaml with every key, the vault, the long-term
memory, the dialog history, the voice ids and the integrations' sign-ins. The browser's web logins are not in it. The
zip is not encrypted: never send it by mail or chat.

`backup()` — makes a backup now in ~/KeepBackups; the result names the file. The newest 10 are kept. With the config
key `backup.format` set to `folder` (for a user whose backup tool — restic, Time Machine — stores a zip anew every time)
the backup is the folder ~/KeepBackups/current instead: the same files unpacked, replaced by each backup, the older
versions being the backup tool's to keep. A backup at regular times is a watch list item of yours.

`list_backups()` — the backups there, newest first, and that folder when there is one.

`restore_backup(file: str)` — replaces this install's state with a backup's and restarts the server; the user
approves first. `file` is a name from `list_backups`, or the path of a zip or of a backup folder — a copy of `current`
the user's backup tool brought back. Only on the user's explicit request:
everything said and learned since the backup is gone afterwards. The current state is backed up first, as a
keep-before-restore zip, so restoring that one undoes it. On a new machine the user installs Keep, copies a backup
over and asks for it by its path.
