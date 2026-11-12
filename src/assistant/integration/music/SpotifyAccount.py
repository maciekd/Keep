"""The household's one Spotify account, shared by every speaker system that plays Spotify: search, the user's playlist library and the sign-in.

Search, track and artist lookups run on the app's client credentials
(`api_keys.spotify.client_id` / `client_secret`, read at call time); only the playlist library —
and, for a speaker system that starts Spotify over Connect, playback — needs the user token
`SpotifyAuth` keeps. Search hits get short ids (`AA`, `AB`, … `ZZ`, then round again) any backend
resolves with `resolve()`, so a hit found once plays in any room of any system.

The library is the first `PLAYLISTS_IN_PROMPT` playlists of `/me/playlists` (newest added first,
`S-n` in the prompt) and its size. The speaker systems drive its reload (`refresh_library()`, from
their own 30-minute refresh); a reload asked for within `MIN_REFRESH_GAP` of the last is skipped,
so two systems refreshing do not double the calls. A transient failure (`TokenUnavailable`, a
timeout, a 5xx) keeps the cached library; only a sign-in the user has to renew (`NotSignedIn`)
empties it, said once as a system note and in the health check until the next `spotify_login`.
A fetch on a connection Spotify dropped while it sat idle is tried once more on a fresh client; a
4xx answer is never retried, and a 429 holds the fetches back for the Retry-After it names.

Spotify-owned playlists (Discover Weekly, Release Radar, the editorial ones) are left out of
`/me/playlists` for an app in development mode since the Web API change of November 2024, and the
artist top-tracks endpoint and the popularity field are gone for such apps since February 2026 —
the callers treat both as absent, not as failures."""
import http.client
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import requests
import urllib3
from spotipy import Spotify
from spotipy.cache_handler import MemoryCacheHandler
from spotipy.exceptions import SpotifyException
from spotipy.oauth2 import SpotifyClientCredentials

from assistant.integration.OAuthPaste import SignIn
from assistant.integration.SingletonMixin import SingletonMixin
from assistant.integration.music.SpotifyAuth import PLAYLIST_SCOPES, NotSignedIn, SpotifyAuth, TokenUnavailable
from assistant.SystemNote import system_note
from common.ConfigManager import ConfigManager
from common.logging import error, info, trace

# What a health check raises while the user's sign-in has to be renewed: it names the cure.
SIGN_IN_REVOKED = "the Spotify user sign-in was revoked, the user's playlists are gone — spotify_login()"

# What a speaker system that starts Spotify itself (Spotify Connect) needs on the user token.
PLAYBACK_SCOPES = ("user-read-playback-state", "user-modify-playback-state")

# A keep-alive connection Spotify closed while the client sat idle: the next call on that socket
# fails before an answer, and a call on a fresh session goes through.
DROPPED_CONNECTION = (requests.exceptions.ConnectionError, urllib3.exceptions.ProtocolError,
                      http.client.RemoteDisconnected, ConnectionError)


@dataclass(frozen=True)
class SpotifyPlaylist:
    """One playlist of the signed-in user's library: its name, its Spotify URI, and who made it."""
    name: str
    uri: str
    owner: str = ""


class SpotifyAccount(SingletonMixin):
    """The one Spotify account: the app's client, the user's token, the library and the search ids."""

    PLAYLISTS_IN_PROMPT = 10
    MIN_REFRESH_GAP = 5 * 60
    MARKET = "SE"

    # Called (no arguments) after every reload of the playlist library. On the class, so a
    # listener outlives a rebuilt account.
    _library_listeners: List[Callable[[], None]] = []
    # The scopes beyond the library's that a speaker system asked for (`want_playback()`).
    _extra_scopes: set = set()

    def __init__(self):
        self.config = ConfigManager()
        self._auth = SpotifyAuth(self.client_settings, scopes=self.scopes)
        self._app_client: Optional[Spotify] = None
        self._user_client: Optional[Spotify] = None
        self._user_client_token = ""
        self._lock = threading.Lock()

        self._playlists: List[SpotifyPlaylist] = []
        self._playlists_total = 0
        self._library_error = ""   # why the last library fetch failed, "" when it did not
        self._library_fetched_at = 0.0
        self._not_before = 0.0      # a 429's Retry-After: no library fetch before then
        self._sign_in_again_noted = False

        self._search_ids: Dict[str, str] = {}   # "AA" -> "spotify:track:…", for the conversation
        self._next_search_id = 0

    # --- the client and the scopes ------------------------------------------------------------

    def client_settings(self) -> tuple[str, str]:
        """The app's (client_id, client_secret), read at call time so a key set through config_request works at once."""
        return (self.config.get("api_keys.spotify.client_id") or "",
                self.config.get("api_keys.spotify.client_secret") or "")

    @classmethod
    def want_playback(cls) -> None:
        """A speaker system starts Spotify over Connect: the user token must carry the playback
        scopes from now on (a token without them is "sign in again")."""
        if not set(PLAYBACK_SCOPES) <= cls._extra_scopes:
            cls._extra_scopes.update(PLAYBACK_SCOPES)
            info(f"Spotify: the playback scopes are asked for from now on ({', '.join(PLAYBACK_SCOPES)})")

    @classmethod
    def scopes(cls) -> tuple[str, ...]:
        return PLAYLIST_SCOPES + tuple(sorted(cls._extra_scopes - set(PLAYLIST_SCOPES)))

    def app_client(self) -> Spotify:
        """The app's own client (client credentials): search, track and artist lookups. Raises
        ValueError without a client id and secret. The app token is kept in memory: spotipy's
        default writes it to a `.cache` file in the working directory — the checkout's root, where
        nothing ignores it."""
        if self._app_client is None:
            client_id, client_secret = self.client_settings()
            if not client_id or not client_secret:
                raise ValueError("Spotify credentials not found in config.yaml. Add api_keys.spotify.client_id "
                                 "and api_keys.spotify.client_secret")
            self._app_client = Spotify(client_credentials_manager=SpotifyClientCredentials(
                client_id=client_id, client_secret=client_secret, cache_handler=MemoryCacheHandler()))
            info("Spotify client initialized")
        return self._app_client

    def user_client(self, needs: Optional[tuple[str, ...]] = None) -> Spotify:
        """A client on the signed-in user's token; rebuilt when the token was refreshed. Raises
        NotSignedIn. *needs* is `SpotifyAuth.token`'s: the scopes the caller uses, when a token
        from before the scopes grew is enough for it."""
        token = self._auth.token(needs)
        if self._user_client is None or self._user_client_token != token:
            self._user_client = Spotify(auth=token)
            self._user_client_token = token
            trace("Spotify user client built on a fresh access token")
        return self._user_client

    # --- the sign-in (spotify_login / spotify_login_code) ------------------------------------

    def signed_in(self) -> bool:
        return self._auth.signed_in()

    def needs_sign_in_again(self) -> bool:
        return self._auth.needs_sign_in_again()

    def login_begin(self) -> SignIn:
        """Starts the sign-in of the user's Spotify account; raises RuntimeError without a client id."""
        return self._auth.begin()

    def login_finish(self, pasted: str) -> str:
        """"" when the pasted answer signed the user in — the library is then reloaded at once;
        `library_error()` says whether that worked — else why not."""
        failure = self._auth.finish(pasted)
        if failure:
            return failure
        self._user_client = None
        self._sign_in_again_noted = False
        self.refresh_library(force=True)
        return ""

    def user_name(self) -> str:
        """The signed-in user's display name, or "" when it cannot be read (nothing depends on it)."""
        try:
            profile = self.user_client().current_user() or {}
            return str(profile.get('display_name') or profile.get('id') or "")
        except Exception as e:
            trace(f"Spotify: the signed-in user's profile could not be read: {e}")
            return ""

    # --- the playlist library ---------------------------------------------------------------

    def playlists(self) -> List[SpotifyPlaylist]:
        """The cached library, in S-0, S-1, … order."""
        with self._lock:
            return self._playlists.copy()

    def playlists_total(self) -> int:
        """The size of the library, of which `playlists()` is the first part."""
        with self._lock:
            return self._playlists_total

    def library_error(self) -> str:
        """Why the last fetch of the library failed, "" when it did not."""
        return self._library_error

    def playlist(self, s_id: str) -> Optional[SpotifyPlaylist]:
        """The playlist behind an S-0, S-1, … id."""
        try:
            if not s_id.startswith('S-'):
                return None
            index = int(s_id[2:])
            with self._lock:
                if 0 <= index < len(self._playlists):
                    return self._playlists[index]
            error(f"Spotify playlist index {index} out of range (have {len(self._playlists)} playlists)")
            return None
        except ValueError:
            error(f"Invalid Spotify playlist URI format: {s_id}")
            return None

    def refresh_library(self, force: bool = False) -> bool:
        """Reloads the library and tells the listeners. False when the fetch failed (the list it
        had stands and `library_error()` says why) or was skipped as too soon after the last."""
        since = time.time() - self._library_fetched_at
        if not force and since < self.MIN_REFRESH_GAP:
            trace(f"Spotify: library reloaded {since:.0f} s ago, not again yet")
            return False
        fetched = self._fetch_library()
        if fetched is None:
            return False
        playlists, total = fetched
        with self._lock:
            self._playlists = playlists
            self._playlists_total = total
        self._library_fetched_at = time.time()
        self._notify_library_listeners()
        return True

    def _fetch_library(self) -> Optional[tuple[List[SpotifyPlaylist], int]]:
        """The first PLAYLISTS_IN_PROMPT playlists and the library's size — name and URI only,
        since the speakers resolve the tracks themselves. Empty when no user is signed in or the
        sign-in has to be renewed, None when the fetch failed for now."""
        if not self._auth.signed_in():
            trace("Spotify: no user signed in, no playlist library to list (spotify_login())")
            self._library_error = ""
            return [], 0
        wait = self._not_before - time.time()
        if wait > 0:
            info(f"Spotify: playlist library not reloaded, Spotify asked to wait {wait:.0f} s more (429), the cached list stands")
            return None
        try:
            page = self._playlists_page()
            playlists: List[SpotifyPlaylist] = []
            # Spotify answers a null item for a playlist this app may not see at all (a Spotify-owned one).
            for item in page.get('items') or []:
                if item and item.get('name') and item.get('uri'):
                    owner = (item.get('owner') or {}).get('display_name') or ""
                    playlists.append(SpotifyPlaylist(item['name'], item['uri'], owner))
            total = max(int(page.get('total') or 0), len(playlists))
            info(f"Spotify: {len(playlists)} of the {total} playlists in the signed-in user's library listed")
            self._library_error = ""
            if self._auth.lacks_scopes():
                self._note_scopes_grown()
            else:
                self._sign_in_again_noted = False
            return playlists, total
        except NotSignedIn as e:
            info(f"Spotify: playlist library not listed — {e}")
            self._library_error = str(e)
            self._note_sign_in_again()
            return [], 0
        except TokenUnavailable as e:
            info(f"Spotify: playlist library not reloaded, the cached list stands — {e}")
            self._library_error = str(e)
            return None
        except SpotifyException as e:
            if e.http_status == 429:
                self._hold_back_after_429(e)
            error(f"Error fetching the user's Spotify playlists: HTTP {e.http_status} {e}")
            self._library_error = f"HTTP {e.http_status}: {e.msg}"
            return None
        except Exception as e:
            error(f"Error fetching the user's Spotify playlists: {e}")
            self._library_error = f"{type(e).__name__}: {e}"
            return None

    def _playlists_page(self) -> dict:
        """The first page of `/me/playlists`, tried once more on a fresh client when the connection
        dropped. A SpotifyException (a 4xx, a 429, a 5xx spotipy gave up on) is never retried here."""
        try:
            return self.user_client(PLAYLIST_SCOPES).current_user_playlists(limit=self.PLAYLISTS_IN_PROMPT) or {}
        except DROPPED_CONNECTION as e:
            info(f"Spotify: connection dropped fetching the playlist library ({type(e).__name__}: {e}), "
                 f"retrying once on a fresh session")
            self._user_client = None
            return self.user_client(PLAYLIST_SCOPES).current_user_playlists(limit=self.PLAYLISTS_IN_PROMPT) or {}

    def _hold_back_after_429(self, e: SpotifyException):
        """No library fetch before the Retry-After of Spotify's 429 has passed."""
        retry_after = (e.headers or {}).get("Retry-After")
        try:
            seconds = max(0, int(retry_after))
        except (TypeError, ValueError):
            info(f"Spotify: 429 without a usable Retry-After ({retry_after!r}), the next refresh tries again")
            return
        self._not_before = time.time() + seconds
        info(f"Spotify: 429 on the playlist library, no fetch for {seconds} s (Retry-After)")

    def _note_sign_in_again(self):
        """Says once, in every chat and to the LLM, that the Spotify sign-in has to be renewed."""
        if self._sign_in_again_noted or not self._auth.needs_sign_in_again():
            return
        self._sign_in_again_noted = True
        system_note("Spotify", "Spotify: the household's Spotify sign-in is no longer valid (its consent was revoked), "
                               "so the user's Spotify playlists (S- ids) are gone until spotify_login() is run again.")

    def _note_scopes_grown(self):
        """Says once that the sign-in predates the scopes asked for now: the library is still
        listed, what needs the new scopes is not there until the user signs in again."""
        if self._sign_in_again_noted:
            return
        self._sign_in_again_noted = True
        info("Spotify: the token on file lacks the scopes now asked for — the library is still listed, the user is told")
        system_note("Spotify", "Spotify: the household's Spotify sign-in is from before a speaker system that starts "
                               "Spotify itself (over Spotify Connect) was turned on. The playlists (S- ids) are still "
                               "listed, but that system cannot start them until spotify_login() is run again.")

    @classmethod
    def add_library_listener(cls, callback: Callable[[], None]):
        """Calls `callback()` after each reload of the library — how the prompt section lists a
        fresh sign-in's playlists at once instead of on its next check."""
        if callback not in cls._library_listeners:
            cls._library_listeners.append(callback)

    @classmethod
    def remove_library_listener(cls, callback: Callable[[], None]):
        if callback in cls._library_listeners:
            cls._library_listeners.remove(callback)

    def _notify_library_listeners(self):
        listeners = list(self._library_listeners)
        if not listeners:
            trace("Spotify: library reloaded, nobody listening for it")
            return
        trace(f"Spotify: library reloaded, waking {len(listeners)} listener(s)")
        for callback in listeners:
            try:
                callback()
            except Exception as e:
                error(f"Spotify library listener {callback} failed: {e}")

    # --- search and the short ids -------------------------------------------------------------

    def _new_search_id(self) -> str:
        """AA, AB, …, AZ, BA, …, ZZ (676), then round again."""
        n = self._next_search_id
        self._next_search_id = (n + 1) % (26 * 26)
        return f"{chr(65 + n // 26)}{chr(65 + n % 26)}"

    def resolve(self, short_id: str) -> Optional[str]:
        """The Spotify URI a search hit's short id stands for, None when the id is unknown."""
        with self._lock:
            return self._search_ids.get(short_id)

    @staticmethod
    def is_search_id(text: str) -> bool:
        return len(text) == 2 and text.isalpha() and text.isupper()

    @staticmethod
    def _popularity(item: dict) -> str:
        """The popularity suffix, or nothing: Spotify dropped the field for Development Mode apps in February 2026."""
        popularity = item.get('popularity')
        return f" (popularity {popularity}/100)" if popularity is not None else ""

    def search(self, query: str, limit: int = 4) -> str:
        """Tracks, artists and public playlists for the query, each hit with its short id."""
        try:
            res = f"Searching Spotify for '{query}':\n\n"
            results = self.app_client().search(q=query, type='track,artist,playlist', market=self.MARKET, limit=limit)

            with self._lock:
                res += "TRACKS:\n"
                if results['tracks']['items']:
                    for item in results['tracks']['items']:
                        short_id = self._new_search_id()
                        self._search_ids[short_id] = item['uri']
                        artists = ', '.join([artist['name'] for artist in item['artists']])
                        res += f"{item['name']} - {artists}{self._popularity(item)} ID: {short_id}\n"
                else:
                    res += "No tracks found.\n"

                res += "\nARTISTS:\n"
                if results['artists']['items']:
                    for item in results['artists']['items']:
                        short_id = self._new_search_id()
                        self._search_ids[short_id] = item['uri']
                        genres = ', '.join(item.get('genres', [])[:2]) if item.get('genres') else 'N/A'
                        res += f"{item['name']} - {genres}{self._popularity(item)} ID: {short_id}\n"
                else:
                    res += "No artists found.\n"

                res += "\nPLAYLISTS:\n"
                # Spotify answers a null item for a playlist this app may not see (an algorithmic one).
                playlists = [p for p in (results.get('playlists') or {}).get('items') or [] if p]
                if playlists:
                    for item in playlists:
                        short_id = self._new_search_id()
                        self._search_ids[short_id] = item['uri']
                        owner = (item.get('owner') or {}).get('display_name') or ''
                        by = f" - by {owner}" if owner else ""
                        res += f"{item['name']}{by} ID: {short_id}\n"
                else:
                    res += "No playlists found.\n"

            items = results['tracks']['items'] + results['artists']['items']
            if items and not any('popularity' in item for item in items):
                trace("Spotify search results carry no popularity field (Development Mode app since February 2026) — listing without it")
            return res
        except Exception as e:
            e = f"Error searching Spotify: {e}"
            error(e)
            return e

    def artist_top_tracks(self, track_id: str) -> list:
        """The seed track's artist's top tracks, or none when this app may not ask for them.

        Spotify removed the top-tracks endpoint for Development Mode apps in February 2026; an app created since
        gets a 403 or 404 here, and a radio is then the seed track alone rather than a failure."""
        spotify = self.app_client()
        track_info = spotify.track(track_id)
        artist_id = track_info['artists'][0]['id']
        artist_name = track_info['artists'][0]['name']
        try:
            similar_tracks = spotify.artist_top_tracks(artist_id, country=self.MARKET)['tracks']
        except SpotifyException as e:
            if e.http_status not in (403, 404):
                raise
            info(f"Top tracks of {artist_name} not available to this Spotify app (HTTP {e.http_status}, Development Mode) — playing the seed track alone")
            return []
        trace(f"Building radio from {len(similar_tracks)} top tracks by {artist_name}")
        return similar_tracks
