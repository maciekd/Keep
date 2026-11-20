A backup is one zip of everything that makes you this assistant: config.yaml with every key, the vault, the long-term
memory, the dialog history, the voice ids and the integrations' sign-ins. The browser's web logins are not in it. The
zip is not encrypted: never send it by mail or chat.

`backup()` — makes a backup now in ~/KeepBackups; the result names the file. The newest 10 are kept.

`list_backups()` — the backups there, newest first, and the unpacked `current` folder when the nightly backup folder is
on (`backup.nightly_folder`: written every night for restic or Time Machine to pick up).

`restore_backup(file: str)` — replaces this install's state with a backup's and restarts the server; the user
approves first. `file` is a name from `list_backups` or the path of a zip, or of an unpacked backup folder — `current`,
or a copy of it the user brought back with restic or Time Machine. A folder is restored only when every file matches its
manifest.json; otherwise nothing is touched and the result names what is missing or changed. Only on the user's explicit request:
everything said and learned since the backup is gone afterwards. The current state is backed up first, as a
keep-before-restore zip, so restoring that one undoes it. On a new machine the user installs Keep, copies a backup
over and asks for it by its path.
