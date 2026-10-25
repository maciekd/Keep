"""The sign-in of one Microsoft 365 account: MSAL's device-code flow and the token cache it leaves behind.

The account signs in once, through the device-code flow — Keep shows a code and a URL, the user
types the code at microsoft.com/devicelogin on any device, nothing comes back to paste — and
MSAL's `SerializableTokenCache` keeps the refresh token in `cache/oauth/microsoft-<slug>.json`
(mode 600). From then on `token()` is `acquire_token_silent`, which refreshes behind the scenes;
only a revoked consent or a refresh token past its life sends the user through the flow again.

The flow is two steps so the command and the console tab can share it: `begin()` returns the
code and the URL at once, `finish(flow)` blocks until the user has typed the code or the flow
has expired (up to fifteen minutes), so a caller runs it on a thread. The app registration
(`microsoft.app.client_id`, `tenant`) is read at call time so a key set through config_request
works without a rebuild; the MSAL application is built on first use since its constructor
discovers the authority over the network.

MSAL posts the refresh over a long-lived session whose adapter does not retry a POST, so a
keep-alive connection the identity platform has closed meanwhile surfaces as a ConnectionError
(RemoteDisconnected) from `acquire_token_silent`. `token()` asks once more: no answer came back,
so MSAL still holds the old refresh token and the second POST runs on a new connection."""
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import msal
import requests

from assistant.integration.Accounts import slug
from common.logging import error, info, trace
from common.Paths import CACHE_DIR

TOKEN_DIR = CACHE_DIR / "oauth"
TOKEN_FILE_PREFIX = "microsoft-"
# What Keep asks for; msal adds offline_access (the refresh token) and openid/profile itself.
SCOPES = ["Mail.ReadWrite", "Mail.Send", "Calendars.ReadWrite", "User.Read"]
AUTHORITY = "https://login.microsoftonline.com/{tenant}"
DEFAULT_TENANT = "common"
# Seconds MSAL waits on login.microsoftonline.com; without it a stalled refresh blocks the fetch indefinitely.
TIMEOUT = 30


class NotSignedIn(Exception):
    """The account has no token: it was never signed in, signed out, or the refresh failed."""


@dataclass(frozen=True)
class DeviceFlow:
    """What the user needs — the code and where to type it — and the flow dict `finish` takes."""
    user_code: str
    verification_uri: str
    expires_at: float
    flow: dict

    @property
    def instructions(self) -> str:
        return f"go to {self.verification_uri} and enter the code {self.user_code}"


def token_path(name: str) -> Path:
    return TOKEN_DIR / f"{TOKEN_FILE_PREFIX}{slug(name)}.json"


def delete_all_token_files() -> int:
    """Every Microsoft token file, gone (the integration lost its config): the count removed."""
    removed = 0
    for path in TOKEN_DIR.glob(f"{TOKEN_FILE_PREFIX}*.json"):
        try:
            path.unlink()
            removed += 1
            info(f"Microsoft 365: token file {path.name} deleted")
        except OSError as e:
            error(f"Microsoft 365: could not delete {path}: {e}")
    if not removed:
        trace("Microsoft 365: no token files to delete")
    return removed


def not_signed_in_hint(name: str) -> str:
    return f'not signed in — microsoft_login("{name}")'


class MicrosoftAuth:
    """The tokens of one account. `app` returns the current (client_id, tenant)."""

    def __init__(self, name: str, app: Callable[[], tuple[str, str]]):
        self.name = name
        self._app_settings = app
        self._cache: Optional[msal.SerializableTokenCache] = None
        self._application = None
        self._application_key: Optional[tuple[str, str]] = None
        self._lock = threading.RLock()

    @property
    def path(self) -> Path:
        return token_path(self.name)

    # --- the cache file ----------------------------------------------------------------------

    def _load_cache(self) -> msal.SerializableTokenCache:
        if self._cache is None:
            self._cache = msal.SerializableTokenCache()
            if self.path.exists():
                try:
                    self._cache.deserialize(self.path.read_text(encoding="utf-8"))
                    trace(f"Microsoft 365: token cache of '{self.name}' loaded from {self.path.name}")
                except (OSError, ValueError) as e:
                    error(f"Microsoft 365: token cache {self.path} unreadable, starting empty: {e}")
            else:
                trace(f"Microsoft 365: '{self.name}' has no token cache yet")
        return self._cache

    def _save_cache(self) -> None:
        cache = self._cache
        if cache is None or not cache.has_state_changed:
            return
        TOKEN_DIR.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(cache.serialize())
        os.chmod(self.path, 0o600)
        cache.has_state_changed = False
        trace(f"Microsoft 365: token cache of '{self.name}' written to {self.path.name}")

    # --- the MSAL application ----------------------------------------------------------------

    def _app(self):
        client_id, tenant = self._app_settings()
        if not client_id:
            raise NotSignedIn("microsoft.app.client_id is not set")
        key = (client_id, tenant or DEFAULT_TENANT)
        if self._application is None or self._application_key != key:
            trace(f"Microsoft 365: building the MSAL application for '{self.name}' (tenant {key[1]})")
            self._application = msal.PublicClientApplication(
                client_id, authority=AUTHORITY.format(tenant=key[1]), token_cache=self._load_cache(),
                timeout=TIMEOUT)
            self._application_key = key
        return self._application

    def _msal_account(self) -> Optional[dict]:
        accounts = self._app().get_accounts()
        return accounts[0] if accounts else None

    # --- what the integration reads ----------------------------------------------------------

    def _cached_account(self) -> Optional[dict]:
        """The account the cache file holds, read without building the MSAL application
        (whose constructor goes to the network): what `signed_in` rests on."""
        accounts = list(self._load_cache().search(msal.TokenCache.CredentialType.ACCOUNT))
        return accounts[0] if accounts else None

    def signed_in(self) -> bool:
        """True when the cache holds an account: a token can be asked for without the user."""
        with self._lock:
            return self._cached_account() is not None

    def signed_in_as(self) -> str:
        """The username of the cached account, "" when none."""
        with self._lock:
            return str((self._cached_account() or {}).get("username") or "")

    def expires_at(self) -> Optional[datetime]:
        """When the newest cached access token runs out (a refresh then fetches another); None without one."""
        cache = self._load_cache()
        tokens = list(cache.search(msal.TokenCache.CredentialType.ACCESS_TOKEN))
        stamps = [int(t["expires_on"]) for t in tokens if str(t.get("expires_on", "")).isdigit()]
        return datetime.fromtimestamp(max(stamps), tz=timezone.utc) if stamps else None

    def token(self) -> str:
        """A current access token, refreshed silently when needed; raises NotSignedIn."""
        with self._lock:
            account = self._msal_account()
            if account is None:
                raise NotSignedIn(not_signed_in_hint(self.name))
            try:
                result = self._app().acquire_token_silent(SCOPES, account=account)
            except requests.ConnectionError as e:
                info(f"Microsoft 365: token refresh of '{self.name}' lost its connection ({e}), retrying once")
                result = self._app().acquire_token_silent(SCOPES, account=account)
            self._save_cache()
        if not result or "access_token" not in result:
            description = (result or {}).get("error_description") or (result or {}).get("error") or "no token"
            info(f"Microsoft 365: silent token refresh of '{self.name}' failed: {description}")
            raise NotSignedIn(f"token expired, {not_signed_in_hint(self.name)}")
        return result["access_token"]

    # --- the device-code flow ----------------------------------------------------------------

    def begin(self) -> DeviceFlow:
        """Starts the flow: the code and URL for the user. Raises on an app registration Entra refuses."""
        with self._lock:
            flow = self._app().initiate_device_flow(scopes=SCOPES)
        if "user_code" not in flow:
            description = flow.get("error_description") or flow.get("error") or str(flow)
            error(f"Microsoft 365: device flow of '{self.name}' could not start: {description}")
            raise RuntimeError(f"sign-in could not start: {description}")
        expires_at = float(flow.get("expires_at") or time.time() + 900)
        info(f"Microsoft 365: device flow of '{self.name}' started, code {flow['user_code']} "
             f"valid {int(expires_at - time.time())}s")
        return DeviceFlow(flow["user_code"], flow["verification_uri"], expires_at, flow)

    def finish(self, flow: DeviceFlow) -> str:
        """Waits for the user to type the code (blocking, up to the flow's life) and stores the
        tokens; "" when signed in, else why not."""
        result = self._app().acquire_token_by_device_flow(flow.flow)
        with self._lock:
            self._save_cache()
        if result and "access_token" in result:
            info(f"Microsoft 365: '{self.name}' signed in as {self.signed_in_as() or 'an account'}")
            return ""
        description = (result or {}).get("error_description") or (result or {}).get("error") or "no token came back"
        info(f"Microsoft 365: device flow of '{self.name}' ended without a token: {description}")
        if "expired" in description.lower() or (result or {}).get("error") == "expired_token":
            return "the code expired before it was entered"
        return description

    def sign_out(self) -> None:
        """Forgets the account: the cache is emptied and its file deleted."""
        with self._lock:
            try:
                for account in self._app().get_accounts():
                    self._app().remove_account(account)
            except NotSignedIn:
                pass
            self._cache = None
            self._application = None
            if self.path.exists():
                self.path.unlink()
                info(f"Microsoft 365: '{self.name}' signed out, {self.path.name} deleted")
            else:
                trace(f"Microsoft 365: '{self.name}' signed out, no token file to delete")
