"""A small web page for the backups: make one, list them, download one, restore one.

`interfaces.backups` (off by default) serves it on its own socket — `host` (default 0.0.0.0, the
whole network) and `port` (default 5050) — beside the assistant's HTTP server, through the
standard library's threading HTTP server. Everything it does is `assistant/Backups.py`'s: the
backup the `backup` command makes, the files `list_backups` lists, the staging and relaunch of
`restore_backup`.

There is no login: whoever reaches the page can make a backup and download every backup. A
download is the encrypted `.keepbackup` file, which opens only with `backup.passphrase` — but
it holds every key of config.yaml, so set `host` to 127.0.0.1 on a network you do not trust. A
restore asks for the backup passphrase on the page, so neither a stranger on the network nor
another site's form in a browser of the household can replace the install. A download or a
restore names a file of the backup directory by its name, never a path.
"""
import html
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, unquote, urlparse

from assistant import Backups
from common.logging import info, error, note
from interfaces.Interface import Interface

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 5050
MAX_FORM = 4096
CHUNK = 1 << 20


def find(name: str) -> Optional[Path]:
    """The backup of the backup directory called *name*; None for any other name or a path."""
    return next((p for p in Backups.backups() if p.name == name), None)


def restart_after_restore(path: Path) -> None:
    from assistant.agents.CodeQualityMaintainer import CodeQualityMaintainer
    CodeQualityMaintainer.restart(f"restoring the backup {path.name} (the backup page)")


def page(message: str = "", failed: bool = False) -> str:
    """The whole page: what the backups need, the button, the table of backups."""
    rows = []
    for path in Backups.backups():
        stat = path.stat()
        made = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
        name = html.escape(path.name)
        rows.append(
            f"<tr><td>{name}</td><td>{made}</td><td>{stat.st_size / 1e6:.1f} MB</td>"
            f"<td><a href=\"/download/{name}\">Download</a></td>"
            f"<td><form method=\"post\" action=\"/restore\" onsubmit=\"return confirm('Replace this install with "
            f"{name}? The server restarts.')\"><input type=\"hidden\" name=\"name\" value=\"{name}\">"
            f"<input type=\"password\" name=\"passphrase\" placeholder=\"backup passphrase\" required> "
            f"<button>Restore</button></form></td></tr>")
    problem = Backups.passphrase_problem()
    status = (f"<p class=\"bad\">No backup can be made: {html.escape(problem)}.</p>" if problem else "")
    banner = f"<p class=\"{'bad' if failed else 'ok'}\">{html.escape(message)}</p>" if message else ""
    table = ("<table><tr><th>Backup</th><th>Made</th><th>Size</th><th></th><th></th></tr>"
             + "".join(rows) + "</table>") if rows else "<p>No backups yet.</p>"
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Keep backups</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>body{{font-family:system-ui,sans-serif;max-width:60em;margin:2em auto;padding:0 1em}}
td,th{{padding:.3em .8em;text-align:left}} .ok{{color:#176917}} .bad{{color:#a11}}</style></head><body>
<h1>Keep backups</h1>{banner}{status}
<p>Folder: <code>{html.escape(str(Backups.directory()))}</code>. A backup holds Keep's config with every key,
the vault, the memory, the dialog history and the sign-ins, encrypted with the backup passphrase.</p>
<form method="post" action="/backup"><button>Back up now</button></form>
<h2>Backups</h2>{table}
<p>A restore replaces this install's state with the backup's and restarts the server; what it replaces is
kept in <code>cache/pre-restore-&lt;time&gt;</code>.</p></body></html>"""


class _Handler(BaseHTTPRequestHandler):
    server_version = "KeepBackups"
    busy = threading.Lock()

    def log_message(self, fmt, *args):
        info(f"Backup page: {self.client_address[0]} {fmt % args}")

    def _send(self, body: str, status: int = 200):
        data = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _form(self) -> dict[str, str]:
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_FORM:
            return {}
        fields = parse_qs(self.rfile.read(length).decode(errors="replace"))
        return {k: v[0] for k, v in fields.items()}

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            return self._send(page())
        if path.startswith("/download/"):
            return self._download(unquote(path[len("/download/"):]))
        self._send(page("No such page.", failed=True), 404)

    def do_POST(self):
        path = urlparse(self.path).path
        form = self._form()
        if path == "/backup":
            return self._backup()
        if path == "/restore":
            return self._restore(form.get("name", ""), form.get("passphrase", ""))
        self._send(page("No such page.", failed=True), 404)

    def _backup(self):
        if not self.busy.acquire(blocking=False):
            info("Backup page: a backup or restore is already running, refused")
            return self._send(page("A backup or restore is already running.", failed=True), 409)
        try:
            info(f"Backup page: backup requested from {self.client_address[0]}")
            answer = Backups.create()
        finally:
            self.busy.release()
        self._send(page(answer, failed=not answer.startswith("Backup written")))

    def _download(self, name: str):
        backup = find(name)
        if backup is None:
            info(f"Backup page: no backup '{name}' to download")
            return self._send(page(f"No backup called {name}.", failed=True), 404)
        note(f"Backup page: {backup.name} downloaded by {self.client_address[0]}")
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Disposition", f"attachment; filename=\"{backup.name}\"")
        self.send_header("Content-Length", str(backup.stat().st_size))
        self.end_headers()
        with open(backup, "rb") as f:
            while chunk := f.read(CHUNK):
                self.wfile.write(chunk)

    def _restore(self, name: str, typed: str):
        backup = find(name)
        if backup is None:
            info(f"Backup page: no backup '{name}' to restore")
            return self._send(page(f"No backup called {name}.", failed=True), 404)
        if not Backups.passphrase_matches(typed):
            note(f"Backup page: restore of {backup.name} from {self.client_address[0]} refused, wrong passphrase")
            return self._send(page("That is not the backup passphrase; nothing was restored.", failed=True), 403)
        if not self.busy.acquire(blocking=False):
            return self._send(page("A backup or restore is already running.", failed=True), 409)
        try:
            note(f"Backup page: restoring {backup.name}, requested from {self.client_address[0]}")
            problem = Backups.stage(backup)
        finally:
            self.busy.release()
        if problem:
            return self._send(page(f"The restore did not start: {problem}.", failed=True), 500)
        self._send(page(f"{backup.name} is unpacked; the server restarts to put it in place. "
                        f"Reload this page in a minute."))
        threading.Thread(target=restart_after_restore, args=(backup,), name="backup-page-restore",
                         daemon=True).start()


class BackupsInterface(Interface):
    """The backup web page on `interfaces.backups.host`:`port`, off unless enabled."""

    name = "backups"
    enabled_by_default = False

    def __init__(self):
        self._server: Optional[ThreadingHTTPServer] = None

    def start(self, assistant, app) -> None:
        from common.ConfigManager import config
        host = str(config.get("interfaces.backups.host", DEFAULT_HOST) or DEFAULT_HOST)
        port = int(config.get("interfaces.backups.port", DEFAULT_PORT) or DEFAULT_PORT)
        try:
            self._server = ThreadingHTTPServer((host, port), _Handler)
        except OSError as e:
            error(f"Backup page: cannot listen on {host}:{port}: {e}")
            return
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, name="backup-page", daemon=True).start()
        note(f"Backup page on http://{host}:{port}/ — no login: whoever reaches it can download the backups"
             + (" (the whole network)" if host in ("0.0.0.0", "::", "") else ""))

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
