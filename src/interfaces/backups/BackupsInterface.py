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
downloaded earlier, or made on another machine) — is one `_Restore` at a time and is staged only
once approved: a backup somebody on the network made themselves names its own owner. It is
approved either in the chat — every owner chat is asked, the console among them (ssh to the
server, `./keep.sh`), like `restart_server` — or on the page with this install's current
`backup.passphrase`, for when the chat itself is what broke. The passphrase an uploaded file was
made with never approves anything (whoever uploads their own backup knows it), and wrong guesses
lock the page's approval for LOCKOUT_SECONDS after MAX_WRONG within FAILURE_WINDOW. A deny in the
chat ends the restore; a chat that cannot show the question or is not answered in time leaves
the passphrase as the way, for PASSPHRASE_WAIT_SECONDS more. A file
that does not open with `backup.passphrase` (or an install with none set) has the page ask
for the passphrase it was made with; that passphrase opens this file alone and is never kept.
An upload waits under cache/backup-upload until the restore ends. A download or a restore
names a file of the backup directory by its name, never a path.
"""
import hmac
import html
import re
import shutil
import threading
import time
from dataclasses import dataclass, field
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
PASSPHRASE_WAIT_SECONDS = 600
MAX_WRONG = 5
FAILURE_WINDOW = 15 * 60
LOCKOUT_SECONDS = 15 * 60
APPROVE, DENY, BY_PASSPHRASE = "approve", "deny", "passphrase"

# The states of a restore; the first two are the ones still under way.
NEEDS_PASSPHRASE, WAITING, DONE, REFUSED, FAILED = "passphrase", "waiting", "done", "refused", "failed"

OUTCOMES = {
    "deny": "Denied in the chat; nothing was restored.",
    "timeout": "Nobody approved it in time, in the chat or with the passphrase; nothing was restored.",
    "unavailable": "No chat could show the approval and nobody typed the passphrase in time; nothing was restored.",
    "cancelled": "The approval was withdrawn; nothing was restored.",
}
# What the page says while it still waits for the passphrase after the chat gave no answer.
CHAT_NOTES = {
    "timeout": "Nobody answered in the chat in time.",
    "unavailable": "No chat could show the approval (Telegram, the console).",
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
    verdict: str = ""
    chat_note: str = ""
    decided: threading.Event = field(default_factory=threading.Event)

    def discard_upload(self):
        if self.uploaded:
            self.path.unlink(missing_ok=True)


_lock = threading.Lock()
_restore: Optional[_Restore] = None
_busy = threading.Lock()
_attempts_lock = threading.Lock()
_wrong: list[float] = []
_locked_until = 0.0
_clock = time.monotonic


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


def approve_with_passphrase(typed: str, client: str) -> tuple[bool, str]:
    """Approves the restore waiting for approval when *typed* is this install's current
    `backup.passphrase`; (False, why) otherwise. Wrong guesses lock this way out for a while."""
    global _locked_until
    job = current()
    if job is None or job.state != WAITING or job.decided.is_set():
        info(f"Backup page: a passphrase approval from {client} came for no restore waiting for one")
        return False, "No restore is waiting for approval."
    if Backups.passphrase_problem():
        info("Backup page: no usable backup passphrase is configured, a restore cannot be approved with one")
        return False, "This install has no backup passphrase to approve with; approve it in the chat."
    with _attempts_lock:
        now = _clock()
        if now < _locked_until:
            minutes = int((_locked_until - now) // 60) + 1
            info(f"Backup page: passphrase approval from {client} refused, locked for {minutes} more minute(s)")
            return False, (f"Too many wrong passphrases: approving with it is locked for {minutes} more minute(s). "
                           f"Approve in the chat instead.")
        if not hmac.compare_digest(typed.encode(), Backups.configured_passphrase().encode()):
            _wrong[:] = [t for t in _wrong if now - t < FAILURE_WINDOW] + [now]
            note(f"Backup page: wrong backup passphrase from {client} for the restore of {job.label} "
                 f"({len(_wrong)} of {MAX_WRONG})")
            if len(_wrong) >= MAX_WRONG:
                _locked_until = now + LOCKOUT_SECONDS
                _wrong.clear()
                _tell_owner(f"The backup page refused {MAX_WRONG} wrong backup passphrases (the last from {client}) "
                            f"for the restore of {job.label}; approving with the passphrase is locked for "
                            f"{LOCKOUT_SECONDS // 60} minutes.")
                return False, (f"That is not this install's backup passphrase. Approving with it is now locked for "
                               f"{LOCKOUT_SECONDS // 60} minutes.")
            return False, "That is not this install's backup passphrase."
        _wrong.clear()
    if _settle(job, BY_PASSPHRASE) != BY_PASSPHRASE:
        return False, "The restore was answered in the chat first."
    note(f"Backup page: restore of {job.label} approved with this install's backup passphrase from {client}")
    return True, ""


def _settle(job: _Restore, verdict: str) -> str:
    """The first answer for *job* wins: the chat's or the page's passphrase. Returns the one that did."""
    with _lock:
        if not job.verdict:
            job.verdict = verdict
        job.decided.set()
        return job.verdict


def _tell_owner(text: str) -> None:
    try:
        Notifier.notifier().send_message(text)
    except Exception as e:
        error(f"Backup page: could not tell the chat: {type(e).__name__}: {e}")


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
                                                        _approval_details(job, manifest), cancelled=job.decided.is_set)
    except Exception as e:
        error(f"Backup page: the approval request failed: {type(e).__name__}: {e}")
        decision = "unavailable"
    if decision not in (APPROVE, DENY) and not job.decided.is_set():
        job.chat_note = CHAT_NOTES.get(decision, f"The chat gave no answer ({decision}).")
        info(f"Backup page: the chat gave no answer for {job.label} ({decision}), waiting "
             f"{PASSPHRASE_WAIT_SECONDS}s for the passphrase on the page")
        job.decided.wait(PASSPHRASE_WAIT_SECONDS)
    verdict = _settle(job, decision)
    if verdict not in (APPROVE, BY_PASSPHRASE):
        note(f"Backup page: restore of {job.label} not approved ({verdict})")
        _finish(job, REFUSED, OUTCOMES.get(verdict, f"Not approved ({verdict}); nothing was restored."))
        return
    how = "in the chat" if verdict == APPROVE else "with the backup passphrase on the page"
    note(f"Backup page: restore of {job.label} approved {how}, staging it")
    if verdict == BY_PASSPHRASE:
        _tell_owner(f"The restore of {job.label} was approved on the backup page with this install's backup "
                    f"passphrase (from {job.client}); the server restarts with it.")
    with _busy:
        problem = Backups.stage(job.path, job.passphrase)
    if problem:
        _finish(job, FAILED, f"Approved, but the restore did not start: {problem}.")
        return
    _finish(job, DONE, f"Approved {how}: {job.label} is unpacked and the server restarts to put it in place. "
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
            chat = (f"<span class=\"bad\">{html.escape(job.chat_note)}</span> " if job.chat_note else
                    f"Approve or deny the restore of {label} in the chat — Telegram, or the console "
                    f"(<code>ssh</code> to the server and run <code>{html.escape(str(Backups.AISERVER_DIR))}/keep.sh"
                    f"</code>). ")
            wrong = f"<p class=\"bad\">{html.escape(job.message)}</p>" if job.message else ""
            status = (f"<p class=\"wait\">Waiting for approval. {chat}Or approve it here with this install's "
                      f"backup passphrase (not the passphrase of an uploaded file):</p>{wrong}"
                      f"<form method=\"post\" action=\"/approve\"><input type=\"password\" name=\"passphrase\" "
                      f"placeholder=\"this install's backup passphrase\" autocomplete=\"off\" required> "
                      f"<button>Approve restore</button></form>")
        else:
            status = f"<p class=\"{'ok' if job.state == DONE else 'bad'}\">{html.escape(job.message)}</p>"
    return f"""<h2>Restore</h2>{status}
<p>Restore a backup file from this computer — one downloaded earlier or made on another machine:</p>
<form method="post" action="/upload" enctype="multipart/form-data"><input type="file" name="backup"
accept="{Backup.SUFFIX}" required> <button>Restore from file</button></form>
<p>Every restore waits for approval — in the chat, or with this install's backup passphrase typed here; nothing is
restored before it. Without a running server: <code>./keep.sh backups</code> and <code>./keep.sh restore</code> over
ssh. A restore replaces this
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
    # Reloads once the restore is no longer waiting — not before, so a passphrase being typed stays.
    refresh = (f"<script>setInterval(() => fetch('/state').then(r => r.text()).then(s => {{ if (s !== '{WAITING}') "
               f"location.href = '/'; }}), {REFRESH_SECONDS * 1000});</script>"
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
        if path == "/state":
            job = current()
            return self._send(job.state if job is not None else "none")
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
        if path == "/approve":
            approved, why = approve_with_passphrase(form.get("passphrase", ""), self.client_address[0])
            job = current()
            if approved or job is None or job.state != WAITING:
                return self._send(page(why, failed=True) if why else page())
            job.message = why
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
             f"a restore waits for approval in the chat or with the backup passphrase"
             + (" (the whole network)" if host in ("0.0.0.0", "::", "") else ""))

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
