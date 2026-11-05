"""The Drive API v3 calls: a document's head, its revisions, the plain text of one revision — what the
docs feed needs — and, for the read commands, a search and the text of a Slides deck slide by slide.

The shape of `GoogleCalendarAccount._call`: a bearer token from `GoogleAuth.token()`, one retry
on a rate limit, 401 → `NotSignedIn`, 403 for a missing scope → `NotSignedIn` with the
sign-in-again hint (the token on file was granted before the Drive scope was asked for).
A revision of a native Google Doc carries `exportLinks`, one per format; the `text/plain` one
is what the diff is made of. Google sends that export with a BOM, which `export` strips. A
Slides deck exports as `text/plain` too, but with the slides run together, so `slides_text`
reads it through the Slides API with a `[slide N]` line before each slide; a spreadsheet's
export is the first tab alone as CSV, so `sheets_text` reads every tab through the Sheets API
under a `[sheet <title>]` line — each when that API is enabled in the Cloud project;
`ApiDisabled` is the 403 Google answers when it is not, and the caller falls back to the export."""
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import requests

from common.logging import error, info, trace
from integrations.google.GoogleAuth import GoogleAuth, NotSignedIn, sign_in_again_hint

BASE_URL = "https://www.googleapis.com/drive/v3"
SLIDES_URL = "https://slides.googleapis.com/v1"
SHEETS_URL = "https://sheets.googleapis.com/v4"
DOC_MIME_TYPE = "application/vnd.google-apps.document"
SLIDES_MIME_TYPE = "application/vnd.google-apps.presentation"
SHEET_MIME_TYPE = "application/vnd.google-apps.spreadsheet"
EXPORT_MIME_TYPE = "text/plain"
CSV_MIME_TYPE = "text/csv"
KINDS = {DOC_MIME_TYPE: "Google Doc", SLIDES_MIME_TYPE: "Google Slides",
         SHEET_MIME_TYPE: "Google Sheet", "application/vnd.google-apps.folder": "folder",
         "application/vnd.google-apps.form": "Google Form", "application/vnd.google-apps.drawing": "Google Drawing",
         "application/pdf": "PDF", "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "Word file",
         "application/vnd.openxmlformats-officedocument.presentationml.presentation": "PowerPoint file",
         "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "Excel file"}
TIMEOUT = 60
RETRY_AFTER = 1
RATE_LIMIT_REASONS = ("rateLimitExceeded", "userRateLimitExceeded")
SCOPE_REASONS = ("insufficientPermissions", "insufficientFilePermissions", "ACCESS_TOKEN_SCOPE_INSUFFICIENT")
DISABLED_REASONS = ("accessNotConfigured", "SERVICE_DISABLED")
REVISION_FIELDS = "nextPageToken,revisions(id,modifiedTime,lastModifyingUser/displayName,exportLinks)"
HEAD_FIELDS = "name,mimeType,modifiedTime"
SEARCH_FIELDS = "nextPageToken,files(id,name,mimeType,modifiedTime,owners/displayName)"
SLIDES_FIELDS = "slides(pageElements,slideProperties/notesPage/pageElements)"
SHEETS_FIELDS = "sheets(properties/title,data/rowData/values/formattedValue)"
PAGE_SIZE = 1000
SEARCH_LIMIT = 20
# docs.google.com/document/d/<id>/edit, /presentation/d/<id>/…, drive.google.com/file/d/<id>/view, …?id=<id>
FILE_ID_RE = re.compile(r"/d/([A-Za-z0-9_-]{20,})|[?&]id=([A-Za-z0-9_-]{20,})")
BARE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{20,}$")


class GoogleDriveError(Exception):
    """Drive answered with an error status; `status` carries it."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


class ApiDisabled(GoogleDriveError):
    """The API is not enabled in the install's Cloud project (403 accessNotConfigured)."""


def kind_of(mime_type: str) -> str:
    """"Google Doc", "PDF", … — a file's kind in the words of a reply; the type's tail for one not known."""
    return KINDS.get(mime_type) or (mime_type.rsplit("/", 1)[-1] if mime_type else "file")


@dataclass(frozen=True)
class DocHead:
    name: str
    mime_type: str
    modified_ts: float

    @property
    def is_document(self) -> bool:
        return self.mime_type == DOC_MIME_TYPE

    @property
    def is_presentation(self) -> bool:
        return self.mime_type == SLIDES_MIME_TYPE

    @property
    def is_spreadsheet(self) -> bool:
        return self.mime_type == SHEET_MIME_TYPE

    @property
    def readable(self) -> bool:
        """Whether the read commands can give its text: a native Doc, Sheet or Slides deck."""
        return self.is_document or self.is_presentation or self.is_spreadsheet

    @property
    def kind(self) -> str:
        return kind_of(self.mime_type)

    @property
    def modified_date(self) -> str:
        return time.strftime("%Y-%m-%d", time.localtime(self.modified_ts))


@dataclass(frozen=True)
class DriveFile(DocHead):
    """A search result: a head with its id and owner."""
    id: str
    owner: str


@dataclass(frozen=True)
class Revision:
    id: str
    modified_ts: float
    author: str
    export_url: Optional[str]     # the text/plain export of this revision; None for an older one without links


def rfc3339_ts(value: str) -> float:
    """`2026-10-14T12:02:33.123Z` → epoch seconds."""
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def file_id_of(ref: str) -> str:
    """The file id in a Docs, Slides or Drive link, or the bare id given; "" for neither."""
    ref = ref.strip().strip("<>")
    if BARE_ID_RE.match(ref):
        return ref
    match = FILE_ID_RE.search(ref)
    return (match.group(1) or match.group(2)) if match else ""


def _escape(words: str) -> str:
    return words.replace("\\", "\\\\").replace("'", "\\'")


class GoogleDrive:
    """One account's Drive, read-only."""

    def __init__(self, account: str, auth: GoogleAuth):
        self.account = account
        self.auth = auth

    def head(self, file_id: str) -> DocHead:
        body = self._call("GET", f"/files/{file_id}", {"fields": HEAD_FIELDS})
        return DocHead(str(body.get("name") or ""), str(body.get("mimeType") or ""), rfc3339_ts(body["modifiedTime"]))

    def search(self, words: str, limit: int = SEARCH_LIMIT) -> tuple[list[DriveFile], bool]:
        """The files whose name or content carries the words, newest first, and whether more exist."""
        # One predicate per word: Drive reads `fullText contains 'a b'` as the phrase, which misses
        # the documents that carry the words apart (measured 2026-11-05: 4 files against 6).
        query = " and ".join(f"fullText contains '{_escape(word)}'" for word in words.split())
        params = {"q": f"{query} and trashed = false", "fields": SEARCH_FIELDS,
                  "pageSize": limit, "orderBy": "modifiedTime desc"}
        page = self._call("GET", "/files", params)
        found = [DriveFile(str(f.get("name") or ""), str(f.get("mimeType") or ""), rfc3339_ts(f["modifiedTime"]),
                           str(f["id"]), str(((f.get("owners") or [{}])[0]).get("displayName") or ""))
                 for f in page.get("files", [])]
        trace(f"{self.account}: {len(found)} file(s) match '{words}'" + (", more exist" if page.get("nextPageToken") else ""))
        return found, bool(page.get("nextPageToken"))

    def revisions(self, file_id: str) -> list[Revision]:
        """Every revision of the file, oldest first."""
        found: list[Revision] = []
        params: dict = {"fields": REVISION_FIELDS, "pageSize": PAGE_SIZE}
        pages = 0
        while True:
            page = self._call("GET", f"/files/{file_id}/revisions", params)
            pages += 1
            for item in page.get("revisions", []):
                links = item.get("exportLinks") or {}
                found.append(Revision(str(item["id"]), rfc3339_ts(item["modifiedTime"]),
                                      str((item.get("lastModifyingUser") or {}).get("displayName") or "someone"),
                                      links.get(EXPORT_MIME_TYPE)))
            if not page.get("nextPageToken"):
                break
            params = {**params, "pageToken": page["nextPageToken"]}
        found.sort(key=lambda r: r.modified_ts)
        trace(f"{self.account}: {len(found)} revisions of {file_id} over {pages} page(s)")
        return found

    def export(self, file_id: str, revision: Optional[Revision] = None, mime_type: str = EXPORT_MIME_TYPE) -> str:
        """The plain text of `revision`, or of the document as it is now (`mime_type` picks the
        export format — text/csv is a spreadsheet's first tab)."""
        if revision is not None and revision.export_url:
            response = self._request("GET", revision.export_url, None)
        else:
            if revision is not None:
                info(f"{self.account}: revision {revision.id} of {file_id} has no export link, exporting the head instead")
            response = self._request("GET", f"{BASE_URL}/files/{file_id}/export", {"mimeType": mime_type})
        text = response.content.decode("utf-8-sig", errors="replace")
        trace(f"{self.account}: exported {len(text)} chars of {file_id}" + (f" at revision {revision.id}" if revision else ""))
        return text

    def slides_text(self, file_id: str) -> str:
        """The text of a Slides deck, a `[slide N]` line before each slide and its speaker notes
        after it as `Notes: …`. Raises ApiDisabled when the project has no Slides API."""
        body = self._call("GET", f"{SLIDES_URL}/presentations/{file_id}", {"fields": SLIDES_FIELDS})
        out: list[str] = []
        for number, slide in enumerate(body.get("slides") or [], 1):
            out.append(f"[slide {number}]")
            out += _runs(slide.get("pageElements"))
            notes = " ".join(_runs(((slide.get("slideProperties") or {}).get("notesPage") or {}).get("pageElements")))
            if notes:
                out.append(f"Notes: {notes}")
            out.append("")
        text = "\n".join(out).strip()
        trace(f"{self.account}: {len(text)} chars over {len(body.get('slides') or [])} slide(s) of {file_id}")
        return text

    def sheets_text(self, file_id: str) -> str:
        """Every tab of a spreadsheet, a `[sheet <title>]` line then its rows as tab-separated
        cells as they are shown, empty rows left out. Raises ApiDisabled without the Sheets API."""
        body = self._call("GET", f"{SHEETS_URL}/spreadsheets/{file_id}", {"includeGridData": "true", "fields": SHEETS_FIELDS})
        out: list[str] = []
        for sheet in body.get("sheets") or []:
            out.append(f"[sheet {(sheet.get('properties') or {}).get('title') or len(out) + 1}]")
            for grid in sheet.get("data") or []:
                for row in grid.get("rowData") or []:
                    cells = [str(cell.get("formattedValue") or "") for cell in row.get("values") or []]
                    line = "\t".join(cells).rstrip("\t")
                    if line.strip():
                        out.append(line)
            out.append("")
        text = "\n".join(out).strip()
        trace(f"{self.account}: {len(text)} chars over {len(body.get('sheets') or [])} tab(s) of {file_id}")
        return text

    # --- the API -----------------------------------------------------------------------------

    def _call(self, method: str, path: str, params: Optional[dict] = None) -> dict:
        response = self._request(method, path if path.startswith("https://") else BASE_URL + path, params)
        if response.status_code == 204 or not response.content:
            return {}
        return response.json()

    def _request(self, method: str, url: str, params: Optional[dict], retried: bool = False):
        token = self.auth.token()
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
        started = time.time()
        response = requests.request(method, url, params=params, headers=headers, timeout=TIMEOUT)
        shown = _shown(url)
        trace(f"{self.account}: {method} {shown} -> {response.status_code} in {time.time() - started:.1f}s")
        if response.status_code == 401:
            info(f"{self.account}: Drive answered 401 — the token is not accepted")
            raise NotSignedIn(f"token not accepted, {sign_in_again_hint(self.account)}")
        reason = self._reason(response) if response.status_code in (403, 429) else ""
        if response.status_code == 403 and reason in SCOPE_REASONS:
            info(f"{self.account}: Drive answered 403 {reason} — the sign-in did not grant the Drive scope")
            raise NotSignedIn(f"signed in without the Drive scope, {sign_in_again_hint(self.account)}")
        if response.status_code == 403 and reason in DISABLED_REASONS:
            message = self._message(response)
            info(f"{self.account}: {method} {shown} answered 403 {reason} — the API is not enabled in the Cloud project")
            raise ApiDisabled(403, message)
        if response.status_code == 429 or reason in RATE_LIMIT_REASONS:
            if retried:
                raise GoogleDriveError(response.status_code, "Google is rate-limiting the account, try again in a minute")
            info(f"{self.account}: Drive answered {response.status_code} ({reason or 'too many requests'}), retrying once in {RETRY_AFTER}s")
            time.sleep(RETRY_AFTER)
            return self._request(method, url, params, retried=True)
        if response.status_code >= 400:
            message = self._message(response)
            error(f"{self.account}: Drive {method} {shown} failed: {response.status_code} {message}")
            raise GoogleDriveError(response.status_code, message)
        return response

    @staticmethod
    def _reason(response) -> str:
        try:
            err = response.json().get("error") or {}
            errors = err.get("errors") or []
            if errors and errors[0].get("reason"):
                return str(errors[0]["reason"])
            return next((str(d["reason"]) for d in err.get("details") or [] if d.get("reason")), "")
        except (ValueError, AttributeError, TypeError):
            return ""

    @staticmethod
    def _message(response) -> str:
        try:
            err = response.json().get("error") or {}
            return str(err.get("message") or err.get("status") or response.text)[:300]
        except (ValueError, AttributeError):
            return (response.text or f"HTTP {response.status_code}")[:300]


def _shown(url: str) -> str:
    for base, label in ((BASE_URL, ""), (SLIDES_URL, "slides "), (SHEETS_URL, "sheets ")):
        if url.startswith(base):
            return label + url[len(base):]
    return "export link"


def _runs(node) -> list[str]:
    """The lines of text under a Slides page element tree, in order: every `textRun.content`
    of every shape, table cell and group, a vertical tab (a line break in a box) a new line."""
    found: list[str] = []
    _collect(node, found)
    return [line.strip() for line in "".join(found).replace("\x0b", "\n").split("\n") if line.strip()]


def _collect(node, found: list[str]) -> None:
    if isinstance(node, dict):
        if "textRun" in node and isinstance(node["textRun"], dict):
            found.append(str(node["textRun"].get("content") or ""))
            return
        for value in node.values():
            _collect(value, found)
    elif isinstance(node, list):
        for item in node:
            _collect(item, found)
