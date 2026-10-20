"""The sign-in of the household's Dropbox: the OAuth 2.0 code flow with PKCE by paste, and the token file it leaves behind.

Dropbox supports PKCE, so the install needs only the app key (`dropbox.app_key`) — no secret
in config.yaml, and a refresh goes to the token endpoint with the refresh token and the
client id alone. The flow is loopback-by-paste like Google's and Volvo's: `begin()` builds the
authorization URL (an S256 challenge, `token_access_type=offline` so a refresh token comes
back, a random `state`), the user signs in with it in any browser, lands on the loopback
`REDIRECT_URI` — a page nothing serves, registered as the app's redirect URI in the App
Console — and pastes that whole address back into `finish()`, which checks the state and
exchanges the code with the verifier (`assistant.integration.OAuthPaste` does the paste
handling). Scopes: the account, file metadata and file content, all read-only. A token is
issued with the scopes the app had ticked *and submitted* on its Permissions tab at the time
— Dropbox grants the subset silently, and a token keeps its scopes for life — so `finish()`
reads the `scope` the token endpoint answers with and refuses a sign-in short of one, naming
it and the tab to fix; `missing_scopes()` reads the same off the file for the health check.

The tokens live in `cache/oauth/dropbox.json` (mode 600). Access tokens last four hours, so
`token()` refreshes within a minute of the expiry; refresh tokens do not expire on their own.
A refresh Dropbox refuses (the app's access was removed at dropbox.com/account/connected_apps,
or the app was deleted) deletes nothing: the sign-in is marked as needed again and every call
says so until `finish()` succeeds. The app key is read at call time so one set through
config_request works without a rebuild.

A flow in progress is module state, not instance state: `restart_integration` rebuilds the
integration — a key set through config_request needs it — and the link the user was given a
minute earlier must still be the one whose answer is pasted back. The flow remembers the app
key the link was built with: a code issued for one app cannot be exchanged by another."""
import base64
import hashlib
import json
import os
import secrets
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlencode

import requests

from assistant.integration.OAuthPaste import SignIn, pasted_code
from common.logging import error, info, trace
from common.Paths import CACHE_DIR

TOKEN_DIR = CACHE_DIR / "oauth"
TOKEN_FILE = "dropbox.json"
AUTHORIZE_URL = "https://www.dropbox.com/oauth2/authorize"
TOKEN_URL = "https://api.dropboxapi.com/oauth2/token"
# A loopback address nothing serves — registered as the app's redirect URI in the App Console;
# the browser lands on it and the user copies it back. Google's sign-in uses 18722, Volvo's 18723.
REDIRECT_PORT = 18724
REDIRECT_URI = f"http://localhost:{REDIRECT_PORT}/"
SCOPES = ["account_info.read", "files.metadata.read", "files.content.read"]
REFRESH_MARGIN = 60
DEFAULT_EXPIRES_IN = 14400
TIMEOUT = 30
LOGIN_COMMAND = "dropbox_login()"


class NotSignedIn(Exception):
    """No usable token: never signed in, signed out, or the refresh was refused."""


@dataclass(frozen=True)
class PendingSignIn:
    """A sign-in link handed out and not yet answered: what `finish()` needs to accept the answer."""
    state: str
    verifier: str
    app_key: str
    started_at: float


# The flow in progress outlives the DropboxAuth instance (see the module docstring); one account.
_pending: Optional[PendingSignIn] = None
_pending_lock = threading.RLock()


def pending_sign_in() -> Optional[PendingSignIn]:
    with _pending_lock:
        return _pending


def _set_pending(flow: Optional[PendingSignIn]) -> None:
    global _pending
    with _pending_lock:
        _pending = flow


def token_path() -> Path:
    return TOKEN_DIR / TOKEN_FILE


def not_signed_in_hint() -> str:
    return f"Dropbox is not signed in — {LOGIN_COMMAND}"


def sign_in_again_hint() -> str:
    return f"sign in to Dropbox again — {LOGIN_COMMAND}"


def missing_scopes(granted: str) -> list[str]:
    """The scopes of SCOPES a token's `scope` string lacks; nothing when the string is empty
    (an answer without one says nothing about them)."""
    have = set(granted.split())
    return [scope for scope in SCOPES if scope not in have] if have else []


def missing_scopes_hint(missing: list[str], granted: str) -> str:
    return (f"the app lacks the permission{'s' if len(missing) > 1 else ''} {', '.join(missing)}"
            f"{' (granted: ' + ', '.join(granted.split()) + ')' if granted else ''}: on the app's Permissions tab at "
            f"https://www.dropbox.com/developers/apps tick {', '.join(SCOPES)} and press Submit, then {sign_in_again_hint()} "
            f"— a sign-in keeps the permissions it was made with")


def delete_token_file() -> bool:
    """The token file, gone (a sign-out, or the integration lost its config): True when there was one."""
    path = token_path()
    if not path.exists():
        trace("Dropbox: no token file to delete")
        return False
    try:
        path.unlink()
        info(f"Dropbox: token file {path.name} deleted")
        return True
    except OSError as e:
        error(f"Dropbox: could not delete {path}: {e}")
        return False


class DropboxAuth:
    """The one account's tokens. `app_key` returns the current `dropbox.app_key`."""

    def __init__(self, app_key: Callable[[], str]):
        self._app_key = app_key
        self._tokens: Optional[dict] = None
        self.refresh_error = ""
        self._lock = threading.RLock()
        flow = pending_sign_in()
        if flow is not None:
            info(f"Dropbox: a sign-in started {int(time.time() - flow.started_at)} s ago is still in progress — its link stays valid")

    @property
    def path(self) -> Path:
        return token_path()

    # --- the token file ----------------------------------------------------------------------

    def _load(self) -> dict:
        if self._tokens is None:
            self._tokens = {}
            if self.path.exists():
                try:
                    self._tokens = json.loads(self.path.read_text(encoding="utf-8")) or {}
                    trace(f"Dropbox: tokens loaded from {self.path.name}")
                except (OSError, ValueError) as e:
                    error(f"Dropbox: token file {self.path} unreadable, starting empty: {e}")
            else:
                trace("Dropbox: no token file yet")
        return self._tokens

    def _save(self, tokens: dict) -> None:
        TOKEN_DIR.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(tokens, handle)
        os.chmod(self.path, 0o600)
        self._tokens = tokens
        trace(f"Dropbox: tokens written to {self.path.name}")

    # --- what the integration reads ----------------------------------------------------------

    def signed_in(self) -> bool:
        """True when a refresh token is on file: a token can be asked for without the user."""
        with self._lock:
            return bool(self._load().get("refresh_token"))

    def needs_sign_in_again(self) -> bool:
        """True when the last refresh was refused: the file is kept, the user must sign in anew."""
        return bool(self.refresh_error)

    def missing_scopes(self) -> list[str]:
        """The scopes the sign-in on file was issued without (a sign-in from before the check)."""
        with self._lock:
            return missing_scopes(str(self._load().get("scope") or ""))

    def expires_at(self) -> Optional[datetime]:
        """When the access token on file runs out (the next call refreshes it); None without one."""
        expiry = self._load().get("expiry")
        return datetime.fromtimestamp(float(expiry), tz=timezone.utc) if expiry else None

    def token(self) -> str:
        """A current access token, refreshed when within a minute of its expiry; raises NotSignedIn."""
        with self._lock:
            tokens = self._load()
            if not tokens.get("refresh_token"):
                raise NotSignedIn(not_signed_in_hint())
            if self.refresh_error:
                raise NotSignedIn(f"{self.refresh_error}, {sign_in_again_hint()}")
            if tokens.get("access_token") and float(tokens.get("expiry") or 0) - time.time() > REFRESH_MARGIN:
                return tokens["access_token"]
            return self._refresh(tokens)

    def _refresh(self, tokens: dict) -> str:
        trace("Dropbox: refreshing the access token")
        answer, status = self._token_request({"grant_type": "refresh_token", "refresh_token": tokens["refresh_token"],
                                              "client_id": self._app_key()})
        if "access_token" not in answer:
            reason = str(answer.get("error") or "no token")
            description = str(answer.get("error_description") or reason)
            info(f"Dropbox: token refresh refused ({status}): {reason} — {description}")
            if status in (400, 401, 403):
                # invalid_grant: the app's access was removed, or the app no longer exists — both need the user.
                self.refresh_error = f"the Dropbox sign-in is no longer valid ({description})"
            raise NotSignedIn(f"token refresh refused ({description}), {sign_in_again_hint()}")
        updated = {**tokens, "access_token": answer["access_token"],
                   "expiry": time.time() + float(answer.get("expires_in") or DEFAULT_EXPIRES_IN)}
        self._save(updated)
        return updated["access_token"]

    @staticmethod
    def _token_request(data: dict) -> tuple[dict, int]:
        """(the token endpoint's answer, the HTTP status)."""
        try:
            response = requests.post(TOKEN_URL, data=data, timeout=TIMEOUT)
        except requests.RequestException as e:
            raise NotSignedIn(f"Dropbox's token endpoint could not be reached: {e}") from e
        try:
            body = response.json()
        except ValueError:
            body = {}
        if response.status_code >= 400 and "error" not in body:
            body = {"error": f"HTTP {response.status_code}", "error_description": (response.text or "")[:200]}
        return (body if isinstance(body, dict) else {}), response.status_code

    # --- the code flow -----------------------------------------------------------------------

    def begin(self) -> SignIn:
        """Starts the flow: the URL for the user. Raises RuntimeError without an app key."""
        app_key = self._app_key()
        if not app_key:
            raise RuntimeError("dropbox.app_key is not set")
        with self._lock:
            flow = PendingSignIn(state=secrets.token_urlsafe(16), verifier=secrets.token_urlsafe(64),
                                 app_key=app_key, started_at=time.time())
            challenge = base64.urlsafe_b64encode(hashlib.sha256(flow.verifier.encode("ascii")).digest()).decode("ascii").rstrip("=")
            params = {"response_type": "code", "client_id": app_key, "redirect_uri": REDIRECT_URI,
                      "token_access_type": "offline", "scope": " ".join(SCOPES), "state": flow.state,
                      "code_challenge": challenge, "code_challenge_method": "S256"}
            url = f"{AUTHORIZE_URL}?{urlencode(params)}"
            _set_pending(flow)
        info("Dropbox: sign-in started")
        return SignIn(url, flow.state, REDIRECT_URI)

    def pending(self) -> bool:
        """True while a link is out and unanswered — across a rebuild of the integration too."""
        return pending_sign_in() is not None

    def finish(self, pasted: str) -> str:
        """Exchanges the code in the pasted address for the tokens and stores them; "" when
        signed in, else why not."""
        with self._lock:
            flow = pending_sign_in()
            code, problem = pasted_code(pasted, flow.state if flow else None, "Dropbox", LOGIN_COMMAND)
            if problem:
                return problem
            if flow.app_key != self._app_key():
                info("Dropbox: dropbox.app_key changed since the sign-in link was made — the link is dropped")
                _set_pending(None)
                return f"the app key changed since the link was made — run {LOGIN_COMMAND} for a new link"
            answer, status = self._token_request({"grant_type": "authorization_code", "code": code,
                                                  "client_id": flow.app_key, "redirect_uri": REDIRECT_URI,
                                                  "code_verifier": flow.verifier})
            if "access_token" not in answer:
                reason = str(answer.get("error") or "no token")
                description = str(answer.get("error_description") or reason)
                info(f"Dropbox: code exchange failed ({status}): {reason} — {description}")
                if reason == "invalid_grant":
                    return f"the code expired or was already used — run {LOGIN_COMMAND} for a new link"
                if reason == "invalid_client":
                    return f"Dropbox does not know the app key ({description}) — check dropbox.app_key in config.yaml"
                return f"Dropbox answered {reason}: {description}"
            if not answer.get("refresh_token"):
                info("Dropbox: the code exchange returned no refresh token")
                return f"Dropbox sent no refresh token — run {LOGIN_COMMAND} again"
            granted = str(answer.get("scope") or "")
            if missing := missing_scopes(granted):
                # The code is spent; the token would fail every file call, so it is not kept.
                info(f"Dropbox: the sign-in was granted '{granted}', short of {missing} — not kept")
                _set_pending(None)
                return f"Dropbox signed in without the file permissions — {missing_scopes_hint(missing, granted)}"
            if not granted:
                info("Dropbox: the token answer names no scopes — taken as granted")
            self._save({"refresh_token": answer["refresh_token"], "access_token": answer["access_token"],
                        "expiry": time.time() + float(answer.get("expires_in") or DEFAULT_EXPIRES_IN),
                        "scope": granted, "account_id": str(answer.get("account_id") or ""),
                        "signed_in_at": time.time()})
            _set_pending(None)
            self.refresh_error = ""
        info(f"Dropbox: signed in with {granted or 'unnamed scopes'}")
        return ""

    def sign_out(self) -> None:
        """Forgets the sign-in: the token file is deleted."""
        with self._lock:
            self._tokens = None
            _set_pending(None)
            self.refresh_error = ""
            delete_token_file()
