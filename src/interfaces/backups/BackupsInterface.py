"""A small web page for the backups: make one, list them, download one, restore one.

`interfaces.backups` (off by default) serves it on its own socket — `host` (default 0.0.0.0, the
whole network) and `port` (default 5050) — beside the assistant's HTTP server, through the
standard library's threading HTTP server. Everything it does is `assistant/Backups.py`'s: the
backup the `backup` command makes, the files `list_backups` lists, the staging and relaunch of
`restore_backup`.

There is no login: whoever reaches the page can make a backup and download every backup. A
download is the encrypted `.keepbackup` file, which opens only with `backup.passphrase` — but
it holds every key of config.yaml, so set `host` to 127.0.0.1 on a network you do not trust.

A restore — of a listed backup or of a `.keepbackup` file uploaded from the browser (one
downloaded earlier, or made on another machine) — is one `_Restore` at a time and waits for
the owner's approve/deny in the chat, like `restart_server`: a backup somebody on the network
made themselves names its own owner, so nothing is staged before the owner said yes. A file
that does not open with `backup.passphrase` (or an install with none set) has the page ask
for the passphrase it was made with; that passphrase opens this file alone and is never kept.
An upload waits under cache/backup-upload until the restore ends. A download or a restore
names a file of the backup directory by its name, never a path.
"""
import html
import re
import shutil
import threading
from dataclasses import dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, unquote, urlparse

from assistant import Backups, Notifier
from common import Backup
from common.Backup import BackupError
from common.logging import info, error, note
from interfaces.Interface import Interface

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 5050
MAX_FORM = 4096
MAX_UPLOAD = 8 << 30
MAX_PART_HEADER = 8192
CHUNK = 1 << 20
UPLOADS = Path("cache") / "backup-upload"
REFRESH_SECONDS = 3

# The states of a restore; the first two are the ones still under way.
NEEDS_PASSPHRASE, WAITING, DONE, REFUSED, FAILED = "passphrase", "waiting", "done", "refused", "failed"

OUTCOMES = {
    "deny": "Denied in the chat; nothing was restored.",
    "timeout": "Nobody answered in the chat in time; nothing was restored.",
    "unavailable": "No chat is reachable to approve it (Telegram, the console); nothing was restored.",
    "cancelled": "The approval was withdrawn; nothing was restored.",
}


def find(name: str) -> Optional[Path]:
    """The backup of the backup directory called *name*; None for any other name or a path."""
    return next((p for p in Backups.backups() if p.name == name), None)


def upload_dir() -> Path:
    return Backups.AISERVER_DIR / UPLOADS


def restart_after_restore(label: str) -> None:
    from assistant.agents.CodeQualityMaintainer import CodeQualityMaintainer
    CodeQualityMaintainer.restart(f"restoring the backup {label} (the backup page)")


@dataclass
class _Restore:
    """The page's one restore: which file, from whom, and how far it got."""
    path: Path
    label: str
    uploaded: bool
    client: str
    state: str = NEEDS_PASSPHRASE
    message: str = ""
    passphrase: str = ""

    def discard_upload(self):
        if self.uploaded:
            self.path.unlink(missing_ok=True)


_lock = threading.Lock()
_restore: Optional[_Restore] = None
_busy = threading.Lock()


def current() -> Optional[_Restore]:
    with _lock:
        return _restore


def begin(path: Path, label: str, uploaded: bool, client: str) -> tuple[bool, str]:
    """Takes *path* as the page's restore; (False, why) while another one is waiting for approval."""
    global _restore
    with _lock:
        if _restore is not None and _restore.state == WAITING:
            info(f"Backup page: restore of {label} refused, {_restore.label} is waiting for approval")
            if uploaded:
                path.unlink(missing_ok=True)
            return False, f"A restore of {_restore.label} is waiting for approval in the chat."
        if _restore is not None and _restore.path != path:
            _restore.discard_upload()
        _restore = job = _Restore(path, label, uploaded, client)
    note(f"Backup page: restore of {label} requested from {client}")
    _open(job, "")
    return True, ""


def give_passphrase(typed: str) -> None:
    """The passphrase the user typed for the restore that asks for one."""
    job = current()
    if job is None or job.state != NEEDS_PASSPHRASE or not typed:
        info("Backup page: a passphrase came for no restore that asks for one, ignored")
        return
    _open(job, typed)


def cancel() -> None:
    global _restore
    with _lock:
        job = _restore
        if job is None or job.state == WAITING:
            info("Backup page: nothing to cancel" if job is None else "Backup page: a restore waiting for approval "
                 "is not cancelled from the page")
            return
        _restore = None
    job.discard_upload()
    info(f"Backup page: restore of {job.label} dropped")


def _open(job: _Restore, passphrase: str) -> None:
    try:
        manifest = Backups.describe(job.path, passphrase)
    except Backup.WrongPassphrase:
        info(f"Backup page: {job.label} does not open with the {'typed' if passphrase else 'configured'} passphrase")
        job.state = NEEDS_PASSPHRASE
        job.message = ("That passphrase does not open it." if passphrase else "")
        return
    except BackupError as e:
        info(f"Backup page: {job.label} is no backup that opens: {e}")
        job.state, job.message = FAILED, f"{job.label} cannot be restored: {e}."
        job.discard_upload()
        return
    with _lock:
        if _restore is not job:
            info(f"Backup page: {job.label} was replaced by another restore before it was asked about")
            job.discard_upload()
            return
        job.passphrase, job.state, job.message = passphrase, WAITING, ""
    threading.Thread(target=_ask_and_restore, args=(job, manifest), name="backup-page-approval", daemon=True).start()


def _approval_details(job: _Restore, manifest: dict) -> str:
    origin = "a file uploaded to" if job.uploaded else "the backup folder, from"
    version = f" by Keep {manifest['keep_version']}" if manifest.get("keep_version") else ""
    return (f"{job.label}: {origin} the backup page by {job.client}. Made {manifest.get('created', 'at an unknown time')}"
            f"{version}. Replaces this install's config, vault, memory, dialog history and sign-ins with the backup's "
            f"and relaunches the server; what it replaces is kept in cache/pre-restore-<time>. The page has no login: "
            f"deny unless you started this restore yourself — a backup made elsewhere brings its own owner.")


def _ask_and_restore(job: _Restore, manifest: dict) -> None:
    info(f"Backup page: asking the owner to approve the restore of {job.label}")
    try:
        decision = Notifier.notifier().request_approval("Restore a backup from the backup page?",
                                                        _approval_details(job, manifest))
    except Exception as e:
        error(f"Backup page: the approval request failed: {type(e).__name__}: {e}")
        decision = "unavailable"
    if decision != "approve":
        note(f"Backup page: restore of {job.label} not approved ({decision})")
        _finish(job, REFUSED, OUTCOMES.get(decision, f"Not approved ({decision}); nothing was restored."))
        return
    note(f"Backup page: restore of {job.label} approved, staging it")
    with _busy:
        problem = Backups.stage(job.path, job.passphrase)
    if problem:
        _finish(job, FAILED, f"Approved, but the restore did not start: {problem}.")
        return
    _finish(job, DONE, f"Approved: {job.label} is unpacked and the server restarts to put it in place. "
                       f"Reload this page in a minute.")
    restart_after_restore(job.label)


def _finish(job: _Restore, state: str, message: str) -> None:
    job.state, job.message, job.passphrase = state, message, ""
    job.discard_upload()


def _restore_section() -> str:
    job = current()
    status = ""
    if job is not None:
        label = html.escape(job.label)
        if job.state == NEEDS_PASSPHRASE:
            why = (f"<span class=\"bad\">{html.escape(job.message)}</span> " if job.message else
                   f"{label} does not open with this install's backup passphrase" +
                   (" (none is set)" if Backups.passphrase_problem() else "") + ". ")
            status = (f"<p>{why}Type the passphrase {label} was made with:</p>"
                      f"<form method=\"post\" action=\"/passphrase\"><input type=\"password\" name=\"passphrase\" "
                      f"placeholder=\"passphrase of this backup\" autocomplete=\"off\" required> <button>Open</button>"
                      f"</form><form method=\"post\" action=\"/cancel\"><button>Cancel</button></form>")
        elif job.state == WAITING:
            status = (f"<p class=\"wait\">Waiting for approval in the chat: approve or deny the restore of {label} "
                      f"there. This page refreshes by itself.</p>")
        else:
            status = f"<p class=\"{'ok' if job.state == DONE else 'bad'}\">{html.escape(job.message)}</p>"
    return f"""<h2>Restore</h2>{status}
<p>Restore a backup file from this computer — one downloaded earlier or made on another machine:</p>
<form method="post" action="/upload" enctype="multipart/form-data"><input type="file" name="backup"
accept="{Backup.SUFFIX}" required> <button>Restore from file</button></form>
<p>Every restore waits for the owner's approval in the chat; nothing is restored before it. A restore replaces this
install's state with the backup's and restarts the server; what it replaces is kept in
<code>cache/pre-restore-&lt;time&gt;</code>.</p>"""


def page(message: str = "", failed: bool = False) -> str:
    """The whole page: what the backups need, the button, the table of backups, the restore."""
    rows = []
    for path in Backups.backups():
        stat = path.stat()
        made = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
        name = html.escape(path.name)
        rows.append(
            f"<tr><td>{name}</td><td>{made}</td><td>{stat.st_size / 1e6:.1f} MB</td>"
            f"<td><a href=\"/download/{name}\">Download</a></td>"
            f"<td><form method=\"post\" action=\"/restore\"><input type=\"hidden\" name=\"name\" value=\"{name}\">"
            f"<button>Restore</button></form></td></tr>")
    problem = Backups.passphrase_problem()
    status = (f"<p class=\"bad\">No backup can be made: {html.escape(problem)}.</p>" if problem else "")
    banner = f"<p class=\"{'bad' if failed else 'ok'}\">{html.escape(message)}</p>" if message else ""
    table = ("<table><tr><th>Backup</th><th>Made</th><th>Size</th><th></th><th></th></tr>"
             + "".join(rows) + "</table>") if rows else "<p>No backups yet.</p>"
    job = current()
    refresh = (f"<meta http-equiv=\"refresh\" content=\"{REFRESH_SECONDS};url=/\">"
               if job is not None and job.state == WAITING else "")
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Keep backups</title>{refresh}
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>body{{font-family:system-ui,sans-serif;max-width:60em;margin:2em auto;padding:0 1em}}
td,th{{padding:.3em .8em;text-align:left}} form{{display:inline-block;margin:.2em 0}} .ok{{color:#176917}}
.bad{{color:#a11}} .wait{{color:#8a5a00}}</style></head><body>
<h1>Keep backups</h1>{banner}{status}
<p>Folder: <code>{html.escape(str(Backups.directory()))}</code>. A backup holds Keep's config with every key,
the vault, the memory, the dialog history and the sign-ins, encrypted with the backup passphrase.</p>
<form method="post" action="/backup"><button>Back up now</button></form>
<h2>Backups</h2>{table}
{_restore_section()}</body></html>"""


def _upload_label(filename: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._ -]", "_", Path(filename.replace("\\", "/")).name).strip()[:80]
    return name or f"uploaded{Backup.SUFFIX}"


class _Handler(BaseHTTPRequestHandler):
    server_version = "KeepBackups"

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
        if path == "/upload":
            return self._upload()
        form = self._form()
        if path == "/backup":
            return self._backup()
        if path == "/restore":
            return self._restore(form.get("name", ""))
        if path == "/passphrase":
            give_passphrase(form.get("passphrase", ""))
            return self._send(page())
        if path == "/cancel":
            cancel()
            return self._send(page())
        self._send(page("No such page.", failed=True), 404)

    def _backup(self):
        if not _busy.acquire(blocking=False):
            info("Backup page: a backup or restore is already running, refused")
            return self._send(page("A backup or restore is already running.", failed=True), 409)
        try:
            info(f"Backup page: backup requested from {self.client_address[0]}")
            answer = Backups.create()
        finally:
            _busy.release()
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

    def _restore(self, name: str):
        backup = find(name)
        if backup is None:
            info(f"Backup page: no backup '{name}' to restore")
            return self._send(page(f"No backup called {name}.", failed=True), 404)
        started, why = begin(backup, backup.name, False, self.client_address[0])
        self._send(page(why, failed=True) if not started else page(), 200 if started else 409)

    def _upload(self):
        """A multipart form whose one (or last) part is the file: the body is streamed to disk
        and the part's closing boundary cut off, so a backup of any size never sits in memory."""
        boundary = self.headers.get_param("boundary") if self.headers.get_content_type() == "multipart/form-data" else None
        length = int(self.headers.get("Content-Length") or 0)
        if not boundary or length <= 0 or length > MAX_UPLOAD:
            info(f"Backup page: upload refused (boundary {bool(boundary)}, {length} bytes)")
            self.close_connection = True
            return self._send(page("That upload is not a backup file this page takes.", failed=True), 400)
        current_job = current()
        if current_job is not None and current_job.state == WAITING:
            self.close_connection = True
            return self._send(page(f"A restore of {current_job.label} is waiting for approval in the chat.",
                                   failed=True), 409)
        delimiter = b"--" + boundary.encode()
        consumed, filename = 0, ""
        first = self.rfile.readline(MAX_PART_HEADER)
        consumed += len(first)
        if first.rstrip(b"\r\n") != delimiter:
            self.close_connection = True
            return self._send(page("That upload is not a backup file this page takes.", failed=True), 400)
        while True:
            line = self.rfile.readline(MAX_PART_HEADER)
            consumed += len(line)
            if not line or line in (b"\r\n", b"\n") or consumed > MAX_PART_HEADER:
                break
            if match := re.search(rb'filename="([^"]*)"', line):
                filename = match.group(1).decode(errors="replace")
        folder = upload_dir()
        folder.mkdir(parents=True, exist_ok=True)
        dest = folder / f"upload-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}{Backup.SUFFIX}"
        remaining = length - consumed
        with open(dest, "wb") as out:
            while remaining > 0:
                chunk = self.rfile.read(min(CHUNK, remaining))
                if not chunk:
                    break
                out.write(chunk)
                remaining -= len(chunk)
        size = dest.stat().st_size
        tail_start = max(0, size - len(delimiter) - 64)
        with open(dest, "r+b") as f:
            f.seek(tail_start)
            end = f.read().rfind(b"\r\n" + delimiter)
            if remaining > 0 or end < 0:
                dest.unlink(missing_ok=True)
                info(f"Backup page: upload from {self.client_address[0]} ended early or has no closing boundary")
                return self._send(page("The upload did not arrive whole; nothing was restored.", failed=True), 400)
            f.truncate(tail_start + end)
        label = _upload_label(filename)
        info(f"Backup page: {label} uploaded by {self.client_address[0]} ({(tail_start + end) / 1e6:.1f} MB) to {dest}")
        started, why = begin(dest, label, True, self.client_address[0])
        self._send(page(why, failed=True) if not started else page(), 200 if started else 409)


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
        if upload_dir().exists():
            info(f"Backup page: removing the uploads a restart left in {upload_dir()}")
            shutil.rmtree(upload_dir(), ignore_errors=True)
        try:
            self._server = ThreadingHTTPServer((host, port), _Handler)
        except OSError as e:
            error(f"Backup page: cannot listen on {host}:{port}: {e}")
            return
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, name="backup-page", daemon=True).start()
        note(f"Backup page on http://{host}:{port}/ — no login: whoever reaches it can download the backups; "
             f"a restore waits for approval in the chat"
             + (" (the whole network)" if host in ("0.0.0.0", "::", "") else ""))

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
