"""The sign-in of the user's own Spotify account: the authorization code flow by paste, and the token file it leaves behind.

A Spotify app redirects only to an address registered with it — here the loopback
`http://127.0.0.1:8888/callback`, on the machine the browser runs on, which is not the server.
So the flow is loopback-by-paste: `begin()` builds the authorization URL (a random `state` is
kept for the flow), the user signs in with it in any browser and lands on a
`http://127.0.0.1:8888/callback?code=…&state=…` page that never loads, and pastes that whole
address back into `finish()`, which checks the state and exchanges the code. The whole address
is required: a bare code would skip the state check, and a paste without `code=` (the bare
redirect, the link itself, a sentence) is refused as such instead of being sent to Spotify.
spotipy keeps the refresh token, the access token and its expiry in `cache/oauth/spotify.json`
(mode 600) and refreshes the access token itself when `token()` finds it expired. The failures
of `token()` are two kinds: `NotSignedIn` when only the user can help (never signed in, the
refresh refused because the consent was revoked, the scopes grown) — the token file is kept,
the account marked as needing a new sign-in and every call says so until `finish()` succeeds
again — and `TokenUnavailable` for a transient one (the token endpoint unreachable or failing),
which a caller rides out with what it has.

The client (`api_keys.spotify.client_id` / `client_secret`) is read at call time, so a key set
through config_request works without a rebuild. spotipy's own fallback for a missing token —
opening a browser or reading the code from stdin — is cut off: a server has neither."""
import os
import secrets
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import parse_qs, urlsplit

import requests
from spotipy.cache_handler import CacheFileHandler
from spotipy.oauth2 import SpotifyOAuth, SpotifyOauthError

from common.logging import error, info, trace
from common.Paths import CACHE_DIR

TOKEN_PATH = CACHE_DIR / "oauth" / "spotify.json"
# The loopback address registered with the Spotify app: nothing serves it, the browser lands
# on it and the user copies the address back.
REDIRECT_URI = "http://127.0.0.1:8888/callback"
SCOPES = ("playlist-read-private", "playlist-read-collaborative")
TIMEOUT = 30
NOT_SIGNED_IN_HINT = "not signed in — spotify_login()"
SIGN_IN_AGAIN_HINT = "sign in again — spotify_login()"


class NotSignedIn(Exception):
    """No usable user token and only the user can help: never signed in, signed out, the refresh refused."""


class TokenUnavailable(Exception):
    """No token this time: the token endpoint could not be reached or failed. The sign-in stands."""


@dataclass(frozen=True)
class SignIn:
    """What the user needs — the URL to open — and the state the pasted answer must carry."""
    url: str
    state: str

    @property
    def instructions(self) -> str:
        return (f"open {self.url} in any browser and sign in to the household's Spotify account; the browser then "
                f"lands on a {REDIRECT_URI}… page that will not load — that is expected — copy that page's whole "
                f"address from the address bar and paste it back")


def parse_pasted(pasted: str) -> tuple[str, str, str]:
    """(code, state, error) out of what the user pasted: the whole redirected URL or its query
    string. Anything without those parameters — a bare code, the bare redirect, the sign-in
    link, a sentence — parses to nothing, so `finish()` asks for the whole address."""
    text = pasted.strip().strip("\"'`<>").strip()
    if "code=" not in text and "state=" not in text and "error=" not in text:
        return "", "", ""
    query = urlsplit(text).query if "?" in text else text.lstrip("?")
    values = parse_qs(query, keep_blank_values=True)
    first = lambda key: (values.get(key) or [""])[0].strip()
    return first("code"), first("state"), first("error")


class _PasteOnlyOAuth(SpotifyOAuth):
    """spotipy's flow without its interactive fallback: with no token on file it would open a
    browser or read the code from stdin, neither of which a server has."""

    def get_auth_response(self, open_browser=None):
        raise NotSignedIn(NOT_SIGNED_IN_HINT)


class SpotifyAuth:
    """The user token of the one Spotify account. `client` returns the current (client_id, client_secret)."""

    def __init__(self, client: Callable[[], tuple[str, str]], path: Path = TOKEN_PATH):
        self._client = client
        self.path = path
        self._state: Optional[str] = None
        self.refresh_error = ""
        self._lock = threading.RLock()

    def _oauth(self) -> SpotifyOAuth:
        """A flow on the current client; raises RuntimeError without one."""
        client_id, client_secret = self._client()
        if not client_id or not client_secret:
            raise RuntimeError("api_keys.spotify.client_id and api_keys.spotify.client_secret are not set")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        return _PasteOnlyOAuth(client_id=client_id, client_secret=client_secret, redirect_uri=REDIRECT_URI,
                               scope=" ".join(SCOPES), cache_handler=CacheFileHandler(cache_path=str(self.path)),
                               open_browser=False, requests_timeout=TIMEOUT)

    def _on_file(self) -> Optional[dict]:
        if not self.path.exists():
            return None
        return CacheFileHandler(cache_path=str(self.path)).get_cached_token()

    # --- what the integration reads ----------------------------------------------------------

    def signed_in(self) -> bool:
        """True when a refresh token is on file: a token can be asked for without the user."""
        with self._lock:
            return bool((self._on_file() or {}).get("refresh_token"))

    def needs_sign_in_again(self) -> bool:
        """True when the last refresh was refused: the file is kept, the user must sign in anew."""
        return bool(self.refresh_error)

    def token(self) -> str:
        """A current access token, refreshed by spotipy when it has run out; raises NotSignedIn."""
        with self._lock:
            if not self.signed_in():
                raise NotSignedIn(NOT_SIGNED_IN_HINT)
            if self.refresh_error:
                raise NotSignedIn(f"{self.refresh_error}, {SIGN_IN_AGAIN_HINT}")
            oauth = self._oauth()
            try:
                token_info = oauth.validate_token(self._on_file())
            except SpotifyOauthError as e:
                reason, description = _oauth_error(e)
                if reason != "invalid_grant":
                    # A 5xx, a malformed answer: Spotify's side, the sign-in itself stands.
                    info(f"Spotify: token refresh failed for now: {reason} — {description}")
                    raise TokenUnavailable(f"token refresh failed ({reason}: {description})") from e
                info(f"Spotify: token refresh refused: {description}")
                self.refresh_error = "the refresh token is no longer valid (consent revoked)"
                raise NotSignedIn(f"token refresh refused ({description}), {SIGN_IN_AGAIN_HINT}") from e
            except requests.RequestException as e:
                info(f"Spotify: token endpoint could not be reached: {e}")
                raise TokenUnavailable(f"Spotify's token endpoint could not be reached: {e}") from e
            if token_info is None:
                info("Spotify: the token on file lacks the scopes now needed — a new sign-in is required")
                raise NotSignedIn(f"the token on file lacks the scopes now needed, {SIGN_IN_AGAIN_HINT}")
            os.chmod(self.path, 0o600)
            return token_info["access_token"]

    # --- the code flow -----------------------------------------------------------------------

    def begin(self) -> SignIn:
        """Starts the flow: the URL for the user. Raises RuntimeError without a client."""
        oauth = self._oauth()
        with self._lock:
            self._state = secrets.token_urlsafe(16)
            url = oauth.get_authorize_url(state=self._state)
        info(f"Spotify: sign-in started for {', '.join(SCOPES)}")
        return SignIn(url, self._state)

    def pending(self) -> bool:
        return self._state is not None

    def finish(self, pasted: str) -> str:
        """Exchanges the code in the pasted address for the tokens and stores them; "" when
        signed in, else why not."""
        code, state, refusal = parse_pasted(pasted)
        if refusal:
            info(f"Spotify: sign-in refused by Spotify: {refusal}")
            return f"Spotify refused the sign-in: {refusal}"
        with self._lock:
            if self._state is None:
                return "no sign-in is in progress — run spotify_login() first"
            if not code or not state:
                info(f"Spotify: the paste carries {'no code' if not code else 'no state'}, nothing exchanged")
                return "no code and state in what was pasted — paste the whole address the browser landed on"
            if state != self._state:
                info("Spotify: sign-in pasted with a state that is not this flow's")
                return "the pasted address belongs to another sign-in — run spotify_login() again and paste that one"
            oauth = self._oauth()
            try:
                oauth.get_access_token(code, as_dict=False, check_cache=False)
            except SpotifyOauthError as e:
                reason, description = _oauth_error(e)
                info(f"Spotify: code exchange failed: {reason} — {description}")
                if reason == "invalid_grant":
                    return "the code expired or was already used — run spotify_login() for a new link"
                return f"Spotify answered {reason}: {description}"
            except requests.RequestException as e:
                return f"Spotify's token endpoint could not be reached: {e}"
            if not self.signed_in():
                # spotipy logs and swallows a failed write; without the file the sign-in is lost at once.
                error(f"Spotify: the exchange succeeded but no token was written to {self.path}")
                return f"the token could not be saved to {self.path} — check that the folder is writable"
            os.chmod(self.path, 0o600)
            self._state = None
            self.refresh_error = ""
        info(f"Spotify: signed in, token written to {self.path.name}")
        return ""

    def sign_out(self) -> None:
        """Forgets the account: its token file is deleted."""
        with self._lock:
            self._state = None
            self.refresh_error = ""
            if self.path.exists():
                try:
                    self.path.unlink()
                    info(f"Spotify: signed out, {self.path.name} deleted")
                except OSError as e:
                    error(f"Spotify: could not delete {self.path}: {e}")
            else:
                trace("Spotify: signed out, no token file to delete")


def _oauth_error(e: SpotifyOauthError) -> tuple[str, str]:
    reason = str(getattr(e, "error", "") or "error")
    description = str(getattr(e, "error_description", "") or e)
    return reason, description
