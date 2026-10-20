"""The Dropbox API v2 calls the commands need: a search, a folder listing, a file's metadata, its bytes, and the account.

The shape of `GoogleDrive`: a bearer token from `DropboxAuth.token()`, one retry on a rate
limit (Dropbox says how long to wait) or a 5xx, 401 → `NotSignedIn` with the sign-in hint,
409 → the `error_summary` Dropbox sends (`path/not_found/..`) as a `DropboxError` the
commands turn into a sentence. A 400 comes as plain text — the app lacking a scope reads
`Error in call to API function "files/search_v2": Your app (ID: …) is not permitted … required
scope 'files.metadata.read' …` — and is carried whole: the sentence names the scope and
Dropbox's own explanation follows, so the cause shows without the server log. RPC endpoints
take a JSON body on api.dropboxapi.com; the content endpoints (download, export) on
content.dropboxapi.com take their argument in the `Dropbox-API-Arg` header and answer with
the bytes."""
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import requests

from common.logging import error, info, trace
from integrations.dropbox.DropboxAuth import DropboxAuth, NotSignedIn, missing_scopes_hint, sign_in_again_hint

API_URL = "https://api.dropboxapi.com/2"
CONTENT_URL = "https://content.dropboxapi.com/2"
TIMEOUT = 60
DOWNLOAD_TIMEOUT = 300
RETRY_AFTER = 2
SEARCH_LIMIT = 20
LIST_LIMIT = 200
SUMMARY_CHARS = 600
MISSING_SCOPE_RE = re.compile(r"required scope '([^']+)'")


class DropboxError(Exception):
    """Dropbox answered with an error status; `status` carries it, `summary` the error_summary."""

    def __init__(self, status: int, summary: str, message: str = ""):
        super().__init__(message or summary)
        self.status = status
        self.summary = summary


@dataclass(frozen=True)
class Entry:
    """A file or folder as Dropbox lists it."""
    path: str                   # path_display, the case the user gave it
    name: str
    is_folder: bool
    size: int = 0
    modified_ts: float = 0.0
    content_hash: str = ""

    @property
    def modified_date(self) -> str:
        return datetime.fromtimestamp(self.modified_ts).strftime("%Y-%m-%d") if self.modified_ts else ""


@dataclass(frozen=True)
class Match:
    entry: Entry
    match_type: str             # filename, content, both
    highlight: str              # the matching passage, when Dropbox sends one


@dataclass(frozen=True)
class Account:
    display_name: str
    email: str
    account_type: str           # basic, pro, business


def iso_ts(value: str) -> float:
    """`2024-03-02T10:00:00Z` → epoch seconds; 0 for nothing."""
    if not value:
        return 0.0
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def entry_of(metadata: dict) -> Entry:
    return Entry(path=str(metadata.get("path_display") or metadata.get("path_lower") or ""),
                 name=str(metadata.get("name") or ""),
                 is_folder=metadata.get(".tag") == "folder",
                 size=int(metadata.get("size") or 0),
                 modified_ts=iso_ts(str(metadata.get("server_modified") or "")),
                 content_hash=str(metadata.get("content_hash") or ""))


class DropboxClient:
    """The one account's Dropbox, read-only."""

    def __init__(self, auth: DropboxAuth, session: Optional[requests.Session] = None):
        self.auth = auth
        self._session = session or requests.Session()

    def current_account(self) -> Account:
        body = self._rpc("/users/get_current_account", None)
        return Account(display_name=str((body.get("name") or {}).get("display_name") or ""),
                       email=str(body.get("email") or ""),
                       account_type=str((body.get("account_type") or {}).get(".tag") or ""))

    def search(self, query: str, path: str = "", limit: int = SEARCH_LIMIT) -> tuple[list[Match], bool]:
        """The first page of matches for `query` under `path` ("" is the whole Dropbox) and
        whether Dropbox has more. Content matches come only on a plan with content search;
        the rest match by name, and Dropbox does not say which it did."""
        options: dict = {"max_results": limit, "file_status": "active", "filename_only": False}
        if path:
            options["path"] = path
        body = self._rpc("/files/search_v2", {"query": query, "options": options,
                                               "match_field_options": {"include_highlights": True}})
        matches = []
        for item in body.get("matches") or []:
            metadata = (item.get("metadata") or {}).get("metadata") or {}
            spans = item.get("highlight_spans") or []
            highlight = "".join(str(s.get("highlight_str") or "") for s in spans).strip() if any(s.get("is_highlighted") for s in spans) else ""
            matches.append(Match(entry_of(metadata), str((item.get("match_type") or {}).get(".tag") or ""), highlight))
        has_more = bool(body.get("has_more"))
        trace(f"Dropbox: search '{query}' under '{path or '/'}' -> {len(matches)} match(es){', more' if has_more else ''}")
        return matches, has_more

    def list_folder(self, path: str, limit: int = LIST_LIMIT) -> tuple[list[Entry], bool]:
        """The entries of a folder, folders first, and whether there are more than `limit`."""
        body = self._rpc("/files/list_folder", {"path": "" if path in ("", "/") else path, "limit": limit})
        entries = [entry_of(item) for item in body.get("entries") or []]
        entries.sort(key=lambda e: (not e.is_folder, e.name.lower()))
        has_more = bool(body.get("has_more"))
        trace(f"Dropbox: list '{path or '/'}' -> {len(entries)} entries{', more' if has_more else ''}")
        return entries, has_more

    def metadata(self, path: str) -> Entry:
        return entry_of(self._rpc("/files/get_metadata", {"path": path}))

    def download(self, path: str) -> bytes:
        response = self._request("POST", f"{CONTENT_URL}/files/download", arg={"path": path}, timeout=DOWNLOAD_TIMEOUT)
        trace(f"Dropbox: downloaded {len(response.content)} bytes of {path}")
        return response.content

    def export(self, path: str, export_format: str = "markdown") -> bytes:
        """A Paper doc (or another Dropbox-native file) in the given format."""
        response = self._request("POST", f"{CONTENT_URL}/files/export", arg={"path": path, "export_format": export_format},
                                 timeout=DOWNLOAD_TIMEOUT)
        trace(f"Dropbox: exported {len(response.content)} bytes of {path} as {export_format}")
        return response.content

    def revoke(self) -> None:
        """Ends the access token's life at Dropbox (the refresh token's with it)."""
        self._rpc("/auth/token/revoke", None)

    # --- the API -----------------------------------------------------------------------------

    def _rpc(self, path: str, body: Optional[dict]) -> dict:
        response = self._request("POST", API_URL + path, json_body=body)
        if not response.content:
            return {}
        try:
            parsed = response.json()
        except ValueError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _request(self, method: str, url: str, json_body: Optional[dict] = None, arg: Optional[dict] = None,
                 timeout: int = TIMEOUT, retried: bool = False):
        token = self.auth.token()
        headers = {"Authorization": f"Bearer {token}"}
        data = None
        if arg is not None:
            headers["Dropbox-API-Arg"] = json.dumps(arg)
        else:
            headers["Content-Type"] = "application/json"
            data = json.dumps(json_body)        # "null" for the endpoints without an argument
        shown = url.removeprefix(API_URL).removeprefix(CONTENT_URL)
        started = time.time()
        try:
            response = self._session.request(method, url, headers=headers, data=data, timeout=timeout)
        except requests.RequestException as e:
            error(f"Dropbox: {method} {shown} could not be sent: {e}")
            raise DropboxError(0, "unreachable", f"Dropbox could not be reached: {e}") from e
        trace(f"Dropbox: {method} {shown} -> {response.status_code} in {time.time() - started:.1f}s")
        if response.status_code == 401:
            info(f"Dropbox: {shown} answered 401 — the token is not accepted")
            raise NotSignedIn(f"token not accepted, {sign_in_again_hint()}")
        if response.status_code == 429 or response.status_code >= 500:
            if retried:
                raise DropboxError(response.status_code, self._summary(response),
                                   "Dropbox is rate-limiting the account, try again in a minute" if response.status_code == 429
                                   else f"Dropbox answered {response.status_code}, try again later")
            wait = self._retry_after(response)
            info(f"Dropbox: {shown} answered {response.status_code}, retrying once in {wait}s")
            time.sleep(wait)
            return self._request(method, url, json_body, arg, timeout, retried=True)
        if response.status_code >= 400:
            summary = self._summary(response)
            error(f"Dropbox: {method} {shown} failed: {response.status_code} {summary}")
            raise DropboxError(response.status_code, summary, self._message(response.status_code, summary))
        return response

    @staticmethod
    def _retry_after(response) -> int:
        try:
            return max(1, min(int(response.headers.get("Retry-After") or RETRY_AFTER), 30))
        except (TypeError, ValueError):
            return RETRY_AFTER

    @staticmethod
    def _summary(response) -> str:
        try:
            body = response.json()
            return str(body.get("error_summary") or body.get("error") or response.text).strip()[:SUMMARY_CHARS]
        except (ValueError, AttributeError):
            return (response.text or f"HTTP {response.status_code}").strip()[:SUMMARY_CHARS]

    @staticmethod
    def _message(status: int, summary: str) -> str:
        """The error_summary as a sentence: `path/not_found/..` → "not found"; an unknown one
        is given whole, Dropbox's explanation being the only one there is."""
        if scope := MISSING_SCOPE_RE.search(summary):
            return f"{missing_scopes_hint([scope.group(1)], '')}. Dropbox said: {summary}"
        if "not_found" in summary:
            return "nothing at that path"
        if "not_file" in summary:
            return "that path is a folder, not a file"
        if "not_folder" in summary:
            return "that path is a file, not a folder"
        if "malformed_path" in summary:
            return "that is not a valid Dropbox path (it starts with /)"
        if "restricted_content" in summary:
            return "Dropbox will not hand that file out (restricted content)"
        if "insufficient_permissions" in summary or "missing_scope" in summary:
            return f"the sign-in lacks the permission for that — {sign_in_again_hint()}"
        if "unsupported_file" in summary or "non_exportable" in summary:
            return "Dropbox cannot export that file as text"
        return f"Dropbox answered {status}: {summary.rstrip('/.') or 'no explanation'}"
