"""The sign-in of one Google account: the OAuth 2.0 code flow by paste, and the token file it leaves behind.

A Google "Desktop app" OAuth client may only redirect to localhost — the machine the browser
runs on, which is not the server. So the flow is loopback-by-paste: `begin()` builds the
authorization URL (offline access, consent forced so a refresh token comes back, a random
`state` kept per account), the user signs in with it in any browser and lands on a
`http://localhost:18722/?state=…&code=…` page that never loads, and pastes that whole address
back into `finish()`, which checks the state and exchanges the code at the token endpoint (the
paste handling is `assistant.integration.OAuthPaste`, shared with the other sign-ins by paste). The refresh token, the access token and its expiry live in
`cache/oauth/google-<slug>.json` (mode 600); `token()` refreshes within a minute of the expiry.

A refresh Google answers with `invalid_grant` (consent revoked, or a refresh token that
expired because the consent screen is still in Testing) deletes nothing: the account is
marked as needing a new sign-in and every call says so until `finish()` succeeds again. The
client (`google.oauth.client_id` / `client_secret`) is read at call time so a key set through
config_request works without a rebuild."""
import json
import os
import secrets
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlencode

import requests

from assistant.integration.Accounts import slug
from assistant.integration.OAuthPaste import SignIn, pasted_code
from common.logging import error, info, trace
from common.Paths import CACHE_DIR

TOKEN_DIR = CACHE_DIR / "oauth"
TOKEN_FILE_PREFIX = "google-"
AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
# A loopback address nothing serves: the browser lands on it and the user copies it back.
REDIRECT_PORT = 18722
REDIRECT_URI = f"http://localhost:{REDIRECT_PORT}/"
SCOPE_MAIL = "https://mail.google.com/"          # IMAP XOAUTH2 accepts nothing narrower
SCOPE_CALENDAR = "https://www.googleapis.com/auth/calendar"
REFRESH_MARGIN = 60
TIMEOUT = 30


class NotSignedIn(Exception):
    """The account has no usable token: never signed in, signed out, or the refresh was refused."""


def token_path(name: str) -> Path:
    return TOKEN_DIR / f"{TOKEN_FILE_PREFIX}{slug(name)}.json"


def delete_all_token_files() -> int:
    """Every Google token file, gone (the integration lost its config): the count removed."""
    removed = 0
    for path in TOKEN_DIR.glob(f"{TOKEN_FILE_PREFIX}*.json"):
        try:
            path.unlink()
            removed += 1
            info(f"Google: token file {path.name} deleted")
        except OSError as e:
            error(f"Google: could not delete {path}: {e}")
    if not removed:
        trace("Google: no token files to delete")
    return removed


def not_signed_in_hint(name: str) -> str:
    return f'not signed in — google_login("{name}")'


def sign_in_again_hint(name: str) -> str:
    return f'sign in again — google_login("{name}")'


class GoogleAuth:
    """The tokens of one account. `client` returns the current (client_id, client_secret),
    `scopes` the scopes the account's flags ask for, `address` its login hint."""

    def __init__(self, name: str, client: Callable[[], tuple[str, str]], scopes: Callable[[], list[str]],
                 address: Callable[[], str]):
        self.name = name
        self._client = client
        self._scopes = scopes
        self._address = address
        self._tokens: Optional[dict] = None
        self._state: Optional[str] = None
        self.refresh_error = ""
        self._lock = threading.RLock()

    @property
    def path(self) -> Path:
        return token_path(self.name)

    # --- the token file ----------------------------------------------------------------------

    def _load(self) -> dict:
        if self._tokens is None:
            self._tokens = {}
            if self.path.exists():
                try:
                    self._tokens = json.loads(self.path.read_text(encoding="utf-8")) or {}
                    trace(f"Google: tokens of '{self.name}' loaded from {self.path.name}")
                except (OSError, ValueError) as e:
                    error(f"Google: token file {self.path} unreadable, starting empty: {e}")
            else:
                trace(f"Google: '{self.name}' has no token file yet")
        return self._tokens

    def _save(self, tokens: dict) -> None:
        TOKEN_DIR.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(tokens, handle)
        os.chmod(self.path, 0o600)
        self._tokens = tokens
        trace(f"Google: tokens of '{self.name}' written to {self.path.name}")

    # --- what the integration reads ----------------------------------------------------------

    def signed_in(self) -> bool:
        """True when a refresh token is on file: a token can be asked for without the user."""
        with self._lock:
            return bool(self._load().get("refresh_token"))

    def needs_sign_in_again(self) -> bool:
        """True when the last refresh was refused: the file is kept, the user must sign in anew."""
        return bool(self.refresh_error)

    def signed_in_as(self) -> str:
        with self._lock:
            return str(self._load().get("address") or "")

    def expires_at(self) -> Optional[datetime]:
        """When the access token on file runs out (the next call refreshes it); None without one."""
        expiry = self._load().get("expiry")
        return datetime.fromtimestamp(float(expiry), tz=timezone.utc) if expiry else None

    def token(self) -> str:
        """A current access token, refreshed when within a minute of its expiry; raises NotSignedIn."""
        with self._lock:
            tokens = self._load()
            if not tokens.get("refresh_token"):
                raise NotSignedIn(not_signed_in_hint(self.name))
            if self.refresh_error:
                raise NotSignedIn(f"{self.refresh_error}, {sign_in_again_hint(self.name)}")
            if tokens.get("access_token") and float(tokens.get("expiry") or 0) - time.time() > REFRESH_MARGIN:
                return tokens["access_token"]
            return self._refresh(tokens)

    def _refresh(self, tokens: dict) -> str:
        client_id, client_secret = self._client()
        trace(f"Google: refreshing the access token of '{self.name}'")
        answer = self._token_request({"client_id": client_id, "client_secret": client_secret,
                                      "refresh_token": tokens["refresh_token"], "grant_type": "refresh_token"})
        if "access_token" not in answer:
            reason = str(answer.get("error") or "no token")
            description = str(answer.get("error_description") or reason)
            info(f"Google: token refresh of '{self.name}' refused: {reason} — {description}")
            if reason == "invalid_grant":
                self.refresh_error = "the refresh token is no longer valid (consent revoked or expired)"
            raise NotSignedIn(f"token refresh refused ({description}), {sign_in_again_hint(self.name)}")
        updated = {**tokens, "access_token": answer["access_token"],
                   "expiry": time.time() + float(answer.get("expires_in") or 3600)}
        if answer.get("scope"):
            updated["scope"] = answer["scope"]
        self._save(updated)
        return updated["access_token"]

    @staticmethod
    def _token_request(data: dict) -> dict:
        try:
            response = requests.post(TOKEN_URL, data=data, timeout=TIMEOUT)
        except requests.RequestException as e:
            raise NotSignedIn(f"Google's token endpoint could not be reached: {e}") from e
        try:
            body = response.json()
        except ValueError:
            body = {}
        if response.status_code >= 400 and "error" not in body:
            body = {"error": f"HTTP {response.status_code}", "error_description": (response.text or "")[:200]}
        return body if isinstance(body, dict) else {}

    # --- the code flow -----------------------------------------------------------------------

    def begin(self) -> SignIn:
        """Starts the flow: the URL for the user. Raises RuntimeError without a client id."""
        client_id, _ = self._client()
        if not client_id:
            raise RuntimeError("google.oauth.client_id is not set")
        scopes = self._scopes()
        if not scopes:
            raise RuntimeError(f"account '{self.name}' asks for neither mail nor calendar")
        address = self._address()
        with self._lock:
            self._state = secrets.token_urlsafe(16)
            params = {"client_id": client_id, "redirect_uri": REDIRECT_URI, "response_type": "code",
                      "scope": " ".join(scopes), "access_type": "offline", "prompt": "consent", "state": self._state}
            if address:
                params["login_hint"] = address
            url = f"{AUTH_URL}?{urlencode(params)}"
        info(f"Google: sign-in of '{self.name}' started for {', '.join(scopes)}")
        return SignIn(url, self._state, REDIRECT_URI, address)

    def pending(self) -> bool:
        return self._state is not None

    def matches(self, state: str) -> bool:
        return bool(state) and self._state == state

    def finish(self, pasted: str) -> str:
        """Exchanges the code in the pasted address for the tokens and stores them; "" when
        signed in, else why not."""
        with self._lock:
            code, problem = pasted_code(pasted, self._state, "Google", f'google_login("{self.name}")')
            if problem:
                return problem
            client_id, client_secret = self._client()
            answer = self._token_request({"code": code, "client_id": client_id, "client_secret": client_secret,
                                          "redirect_uri": REDIRECT_URI, "grant_type": "authorization_code"})
            if "access_token" not in answer:
                reason = str(answer.get("error") or "no token")
                description = str(answer.get("error_description") or reason)
                info(f"Google: code exchange of '{self.name}' failed: {reason} — {description}")
                if reason == "invalid_grant":
                    return f'the code expired or was already used — run google_login("{self.name}") for a new link'
                return f"Google answered {reason}: {description}"
            if not answer.get("refresh_token"):
                info(f"Google: '{self.name}' got no refresh token back — the consent was not renewed")
                return ("Google sent no refresh token: remove Keep's access at myaccount.google.com/permissions "
                        f'and run google_login("{self.name}") again')
            self._save({"refresh_token": answer["refresh_token"], "access_token": answer["access_token"],
                        "expiry": time.time() + float(answer.get("expires_in") or 3600),
                        "scope": str(answer.get("scope") or ""), "address": self._address()})
            self._state = None
            self.refresh_error = ""
        info(f"Google: '{self.name}' signed in" + (f" as {self.signed_in_as()}" if self.signed_in_as() else ""))
        return ""

    def sign_out(self) -> None:
        """Forgets the account: its token file is deleted."""
        with self._lock:
            self._tokens = None
            self._state = None
            self.refresh_error = ""
            if self.path.exists():
                self.path.unlink()
                info(f"Google: '{self.name}' signed out, {self.path.name} deleted")
            else:
                trace(f"Google: '{self.name}' signed out, no token file to delete")
