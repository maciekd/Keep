"""The household's Dropbox, searched and read by the assistant."""
import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import requests

from assistant.integration.Integration import Integration
from assistant.integration.OAuthPaste import SignIn
from common.ConfigManager import ConfigManager
from common.Paths import CACHE_DIR
from common.logging import error, info, trace
from integrations.dropbox.DocumentText import Unreadable, extension, extract
from integrations.dropbox.DropboxAuth import (DropboxAuth, NotSignedIn, delete_token_file, missing_scopes_hint,
                                              not_signed_in_hint, token_path)
from integrations.dropbox.DropboxClient import DropboxClient, Entry, Match

SECTION = "dropbox"
APP_KEY_KEY = f"{SECTION}.app_key"
FOLDERS_KEY = f"{SECTION}.folders"
TEXT_CACHE_DIR = CACHE_DIR / "dropbox"
DOWNLOAD_MAX_BYTES = 50 * 1024 * 1024
CACHE_MAX_BYTES = 200 * 1024 * 1024
PAPER_EXTENSION = "paper"
BUNDLE_EXTENSION = "rtfd"       # a TextEdit bundle: a folder Dropbox lists as such, the text in TXT.rtf inside
BUNDLE_TEXT_FILE = "TXT.rtf"


@dataclass(frozen=True)
class Document:
    """A file's text, with the metadata the page header shows."""
    entry: Entry
    text: str
    kind: str                   # PDF, DOCX, text, …: the extension upper-cased
    from_cache: bool


class DropboxIntegration(Integration):
    """The files in Dropbox, searched and read, read-only.

    Finds a file in the household's Dropbox, lists a folder, and reads a document page by
    page — a PDF, a Word or PowerPoint file, a Paper doc, an RTF (a TextEdit .rtfd too), any
    text file — with a find inside the file that says which page a phrase is on; it never
    writes, moves or shares anything.

    One account, signed in once with the app's key through the PKCE code flow by paste
    (`DropboxAuth`); `DropboxClient` makes the calls. A read fetches the file once: its text
    is cached in `cache/dropbox/<content_hash>.txt` — Dropbox's content hash is a hash of the
    bytes, so a hit is exact and an edited file gets a fresh extraction. `dropbox.folders`
    confines every search, listing and read to the listed roots (empty: the whole Dropbox, or
    the app folder of an App-folder app); a path outside them is refused before any call.
    The health check is `users/get_current_account`, which also tells the plan the account
    is on. `needs_2fa` is "the user has to sign in": the monitor shows NEEDS_2FA with the
    login hint until the paste flow completes — also for a sign-in on file that was issued
    without a file scope (from before the check at the exchange), where the hint names the
    scope and the Permissions tab. Config is read at call time."""
    sensitive = True
    login_hint = "dropbox_login() from the chat"

    def __init__(self, session: Optional[requests.Session] = None):
        Integration.__init__(self)
        self.auth = DropboxAuth(self._app_key)
        self.client = DropboxClient(self.auth, session)
        self.account_type = ""

    @property
    def name(self) -> str:
        return "Dropbox"

    @property
    def config_keys(self) -> List[str]:
        return [APP_KEY_KEY]

    # --- config --------------------------------------------------------------------------------

    @staticmethod
    def _config(key: str, default=None):
        return ConfigManager.get_instance().get(key, default)

    @classmethod
    def _app_key(cls) -> str:
        return str(cls._config(APP_KEY_KEY, "") or "").strip()

    @classmethod
    def folders(cls) -> list[str]:
        """The roots the assistant may touch, normalised (`/Documents`); empty is everything."""
        raw = cls._config(FOLDERS_KEY, []) or []
        if isinstance(raw, str):
            raw = [raw]
        roots = []
        for item in raw:
            folder = normalise(str(item))
            if folder and folder != "/":
                roots.append(folder)
            else:
                trace(f"Dropbox: folders lists '{item}', which is the root — the whole Dropbox is allowed")
                return []
        return roots

    @classmethod
    def refused(cls, path: str) -> str:
        """"" when `path` is under an allowed root, else why not."""
        roots = cls.folders()
        if not roots:
            return ""
        wanted = normalise(path).lower()
        for root in roots:
            lowered = root.lower()
            if wanted == lowered or wanted.startswith(lowered + "/"):
                return ""
        info(f"Dropbox: '{path}' is outside the allowed folders ({', '.join(roots)}), refused")
        return f"{path or '/'} is outside the folders the assistant may read: {', '.join(roots)}"

    # --- the sign-in ---------------------------------------------------------------------------

    def login_begin(self) -> SignIn:
        return self.auth.begin()

    def login_finish(self, pasted: str) -> str:
        """"" when the pasted answer signed the account in, else why not."""
        return self.auth.finish(pasted)

    def sign_out(self) -> None:
        self.auth.sign_out()

    def needs_2fa(self) -> bool:
        """The user has to sign in (never did, the refresh was refused, or the sign-in on file
        lacks a file scope): local state only."""
        return not self.auth.signed_in() or self.auth.needs_sign_in_again() or bool(self.auth.missing_scopes())

    def current_login_hint(self) -> str:
        if self.auth.signed_in() and (missing := self.auth.missing_scopes()):
            return missing_scopes_hint(missing, "")
        return self.login_hint

    @classmethod
    def wipe_targets(cls) -> list[tuple[Path, str]]:
        return [(token_path(), "the Dropbox sign-in (its OAuth tokens)"),
                (TEXT_CACHE_DIR, "the Dropbox text cache (the text of every document read)")]

    @classmethod
    def log_out(cls) -> Optional[str]:
        auth = DropboxAuth(cls._app_key)
        if not auth.signed_in():
            return None
        try:
            DropboxClient(auth).revoke()
            info("Dropbox: the token was revoked at Dropbox")
            revoked = "revoked at Dropbox and "
        except Exception as e:
            info(f"Dropbox: the token could not be revoked ({e}) — the file goes anyway")
            revoked = ""
        delete_token_file()
        return f"Dropbox: token {revoked}deleted"

    def on_not_configured(self):
        delete_token_file()

    # --- health --------------------------------------------------------------------------------

    def check_health(self) -> bool:
        if not self.auth.signed_in():
            raise NotSignedIn(not_signed_in_hint())
        if missing := self.auth.missing_scopes():
            info(f"Dropbox: the sign-in on file lacks {missing} — every file call would fail")
            raise NotSignedIn(missing_scopes_hint(missing, ""))
        account = self.client.current_account()
        if account.account_type != self.account_type:
            info(f"Dropbox: signed in as {account.display_name or account.email} on a {account.account_type or 'unknown'} "
                 f"plan{' — content search is a paid-plan feature, a basic plan searches names only' if account.account_type == 'basic' else ''}")
            self.account_type = account.account_type
        else:
            trace(f"Dropbox: account {account.email} answers ({account.account_type})")
        return True

    # --- what the commands use -----------------------------------------------------------------

    def search(self, query: str, folder: str = "") -> tuple[list[Match], bool]:
        """The matches under `folder`, or under every allowed root when none is given."""
        if folder:
            if problem := self.refused(folder):
                raise ValueError(problem)
            return self.client.search(query, normalise(folder))
        roots = self.folders() or [""]
        matches: list[Match] = []
        has_more = False
        for root in roots:
            found, more = self.client.search(query, root)
            matches.extend(found)
            has_more = has_more or more
        info(f"Dropbox: search '{query}' over {len(roots)} root(s): {len(matches)} match(es)")
        return matches, has_more

    def list_folder(self, path: str) -> tuple[list[Entry], bool]:
        if problem := self.refused(path):
            raise ValueError(problem)
        return self.client.list_folder(normalise(path))

    def document(self, path: str) -> Document:
        """The text of the file at `path`, from the cache when its content hash is known.
        Raises ValueError with the reason when the file cannot be read."""
        if problem := self.refused(path):
            raise ValueError(problem)
        entry = self.client.metadata(normalise(path))
        if entry.is_folder and extension(entry.name) == BUNDLE_EXTENSION:
            trace(f"Dropbox: {entry.path} is an .rtfd bundle — reading its {BUNDLE_TEXT_FILE}")
            entry = self.client.metadata(f"{entry.path}/{BUNDLE_TEXT_FILE}")
        if entry.is_folder:
            raise ValueError(f"{entry.path} is a folder — list_dropbox_folder shows what is in it")
        if entry.size > DOWNLOAD_MAX_BYTES:
            raise ValueError(f"{entry.name} is {human_size(entry.size)}, more than the {human_size(DOWNLOAD_MAX_BYTES)} a read may fetch")
        kind = (extension(entry.name) or "file").upper()
        key = entry.content_hash or hashlib.sha256(f"{entry.path.lower()}|{entry.modified_ts}".encode()).hexdigest()
        cached = self._cached(key)
        if cached is not None:
            trace(f"Dropbox: {entry.path} served from the text cache ({len(cached)} chars)")
            return Document(entry, cached, kind, True)
        started = time.time()
        if extension(entry.name) == PAPER_EXTENSION:
            text = self.client.export(entry.path).decode("utf-8", errors="replace")
        else:
            try:
                text = extract(entry.name, self.client.download(entry.path))
            except Unreadable as e:
                info(f"Dropbox: {entry.path} ({human_size(entry.size)}) not readable: {e}")
                raise ValueError(f"{entry.name} ({human_size(entry.size)}) cannot be read: {e}") from e
        info(f"Dropbox: {entry.path} ({human_size(entry.size)} {kind}) extracted to {len(text)} chars in {time.time() - started:.1f}s")
        self._cache(key, text)
        return Document(entry, text, kind, False)

    # --- the text cache ------------------------------------------------------------------------

    @staticmethod
    def _cache_path(key: str) -> Path:
        return TEXT_CACHE_DIR / f"{key}.txt"

    def _cached(self, key: str) -> Optional[str]:
        path = self._cache_path(key)
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError as e:
            error(f"Dropbox: cache file {path.name} unreadable, re-extracting: {e}")
            return None
        os.utime(path, None)      # the newest-touched survive the trim
        return text

    def _cache(self, key: str, text: str) -> None:
        try:
            TEXT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            path = self._cache_path(key)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, path)
            self._trim_cache()
        except OSError as e:
            error(f"Dropbox: could not cache the text of {key}: {e}")

    @staticmethod
    def _trim_cache() -> None:
        limit = CACHE_MAX_BYTES
        files = [(p.stat().st_mtime, p.stat().st_size, p) for p in TEXT_CACHE_DIR.glob("*.txt")]
        total = sum(size for _, size, _ in files)
        if total <= limit:
            return
        removed = 0
        for _, size, path in sorted(files):
            path.unlink(missing_ok=True)
            total -= size
            removed += 1
            if total <= limit:
                break
        info(f"Dropbox: text cache over {human_size(limit)}, the {removed} oldest file(s) deleted")


def normalise(path: str) -> str:
    """A Dropbox path: one leading slash, no trailing one; "" for the root."""
    path = path.strip().strip("\"'`").strip()
    if path in ("", "/"):
        return ""
    return "/" + path.strip("/")


def human_size(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.0f} KB"
    return f"{size / (1024 * 1024):.1f} MB"
