A backup is one encrypted file holding everything that makes you this assistant: config.yaml with every API key and
token, the vault, the long-term memory, the dialog history, the voice ids and the integrations' sign-ins. Logs, caches
and the browser profile are left out. On a new machine the installer asks for the file and restores it.

`backup()` — makes a backup now. It takes a while for a large history; the result names the file. The newest
`backup.keep` (default 10) are kept in `backup.directory` (default ~/KeepBackups, outside the Keep folder); older ones
are deleted.

`list_backups()` — the backups in that folder, newest first, with when each was made and its size.

`restore_backup(file: str = "latest")` — replaces this install's state with a backup's (a name from `list_backups`, a
path, or the newest) and restarts the server; the user approves it in the chat. What it replaces is kept in
cache/pre-restore-<time>. Only on the user's explicit request: everything said and learned since the backup is gone
from you afterwards.

Every backup is encrypted with `backup.passphrase`. Without it neither command works: ask the user for one with
`config_request("backup.passphrase")` — never in the chat — and tell them to keep it in their password manager, since
no backup opens without it, and the installer on a new machine asks for it. A backup made under an earlier passphrase
opens only once the user types that one the same way. `backup.directory` is set the same way, typed by the user: a
synced folder (iCloud Drive, Dropbox) takes the backups off this machine. Never suggest sending a backup by mail or
chat: it holds every key.
