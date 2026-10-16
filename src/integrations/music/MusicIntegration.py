import functools
import re
import threading
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional

import soco
from soco.plugins.sharelink import ShareLinkPlugin
from soco.zonegroupstate import ZoneGroupState
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.exceptions import SpotifyException

from common.logging import trace, info, error
from common.ConfigManager import ConfigManager
from assistant.integration.Integration import Integration
from assistant.SystemNote import system_note
from integrations.music.SpotifyAuth import SpotifyAuth, SignIn, NotSignedIn, TokenUnavailable, TOKEN_PATH


@dataclass(frozen=True)
class SpotifyPlaylist:
    """One playlist of the signed-in user's library: its name, its Spotify URI, and who made it."""
    name: str
    uri: str
    owner: str = ""

_zone_state_lock = threading.RLock()


def _serialise_zone_state_parsing():
    """Let one thread at a time into soco's zone-group parsing.

    soco runs every zone-group payload through one module-level lxml XSLT object and takes no
    lock around it. Two threads in that transform at once corrupt libxml2's dictionary, and
    the allocator then aborts the whole process (SIGABRT under xmlDictGrow) — no exception,
    no log line. Any `zone.group`, `zone.player_name` or `soco.discover()` can land there, so
    the lock goes on soco's one entry point rather than on each of our callers."""
    original = ZoneGroupState.process_payload
    if getattr(original, "_serialised", False):
        return

    @functools.wraps(original)
    def process_payload(self, *args, **kwargs):
        with _zone_state_lock:
            return original(self, *args, **kwargs)

    process_payload._serialised = True
    ZoneGroupState.process_payload = process_payload
    trace("Sonos: soco's zone-group parsing is serialised (one thread at a time in its XSLT)")


_serialise_zone_state_parsing()


class MusicIntegration(Integration):
    """Spotify music on the Sonos speakers: search, play, skip and volume.

    Plays Spotify music, playlists and favorites in the Sonos zones, with pause, skip and
    volume. Keep can search Spotify and start what it finds in any room, and once the user's
    Spotify account is signed in, the playlists of their own library by name.

    Sonos playlists (`P-n`), Sonos favorites (`F-n`) and the user's Spotify playlists (`S-n`)
    are cached and refreshed every 30 minutes; Spotify search results get short ids the LLM
    can refer back to within the conversation. Each of the three lists is cut to its first
    `PLAYLISTS_IN_PROMPT` entries — every one of them is a prompt line on every request — in
    the order its source gives: the Sonos lists as the Sonos app shows them, the Spotify library
    newest added first. Search runs on the app's client credentials; only the user's playlist
    library needs the user token `SpotifyAuth` keeps. A playlist is
    started by its URI through the Sonos Spotify service (ShareLinkPlugin), so Sonos resolves
    the tracks with the household's Spotify account. Spotify-owned playlists (Discover Weekly,
    Release Radar, the editorial ones) are left out of `/me/playlists` for an app in development
    mode since the Web API change of November 2024 — only an app with extended quota sees them,
    so the texts promise the user's own and followed playlists, not those; a Sonos favorite is
    the way to reach Discover Weekly from a development-mode app. A transient Spotify failure
    (`TokenUnavailable`, a timeout) keeps the cached library; only a sign-in the user has to
    renew (`NotSignedIn`) empties it, and that is said once as a system note and in the health
    check until the next `spotify_login`."""

    PLAYLISTS_IN_PROMPT = 10

    CACHE_EXPIRY_SECONDS = 30 * 60
    REFRESH_INTERVAL_SECONDS = 30 * 60
    # Called (no arguments) after every reload of the user's Spotify playlist library — the
    # sign-in's and the 30-minute one. On the class, so a listener outlives restart_integration.
    _library_listeners: List = []

    def __init__(self):
        Integration.__init__(self)
        self.config = ConfigManager()
        self._spotify_client: Optional[Spotify] = None
        self._spotify_auth = SpotifyAuth(self._spotify_client_settings)
        self._user_client: Optional[Spotify] = None
        self._user_client_token = ""
        self._sonos_zones_cache: Dict[str, soco.SoCo] = {}
        self._sonos_zones_cache_time: Optional[float] = None

        # Cache for playlists and favorites with their indices
        self._playlists_cache: List[str] = []
        self._favorites_cache: List[str] = []
        self._spotify_playlists_cache: List[SpotifyPlaylist] = []
        self._spotify_playlists_total = 0
        self._spotify_library_error = ""   # why the last library fetch failed, "" when it did not
        self._sign_in_again_noted = False
        self._cache_lock = threading.Lock()

        # Cache for Spotify search results with short IDs (persists for conversation)
        self._spotify_search_cache: Dict[str, str] = {}  # Maps "AA" -> "spotify:track:xxx"
        self._next_cache_id: int = 0

        self._refresh_timer: Optional[threading.Timer] = None
        self._stopped = threading.Event()

        # Load initial cache and start periodic refresh
        self._start_periodic_refresh()

    @property
    def name(self) -> str:
        return "Sonos & Spotify"

    @property
    def room_mapping(self) -> Dict[str, str]:
        """`sonos.room_mapping`: Sonos zone name -> the room name the LLM sees; blank hides the zone."""
        return self.config.get('sonos.room_mapping', {}) or {}

    @property
    def config_keys(self) -> List[str]:
        return ["api_keys.spotify.client_id"]

    def check_health(self) -> bool:
        zones = self._discover_sonos_zones()
        if not zones:
            return False
        try:
            self._get_spotify_client()
        except Exception:
            return False
        if self._spotify_auth.needs_sign_in_again():
            # Raised, not returned: the monitor's last_error then names the cure in list_integrations and the console.
            raise RuntimeError("the Spotify user sign-in was revoked, the user's playlists are gone — spotify_login()")
        return True

    def _refresh_playlists_and_favorites(self):
        """Load playlists and favorites and cache them with their indices. Each part is guarded
        on its own, so a Spotify failure never skips the Sonos lists or the other way round."""
        try:
            self._refresh_spotify_playlists()
        except Exception as e:
            error(f"Error refreshing the user's Spotify playlists: {e}")
        try:
            # Use any available zone to fetch playlists and favorites
            zones = self._discover_sonos_zones()
            if not zones:
                info("No Sonos zones available for fetching playlists/favorites")
                return

            zone_name = next(iter(zones.keys()))

            playlists = self._first_in_prompt("Sonos playlists", self._fetch_playlists_from_sonos(zone_name))
            favorites = self._first_in_prompt("Sonos favorites", self._fetch_favorites_from_sonos(zone_name))
            with self._cache_lock:
                self._playlists_cache = playlists
                self._favorites_cache = favorites
        except Exception as e:
            error(f"Error refreshing playlists and favorites cache: {e}")

    def _first_in_prompt(self, what: str, names: List[str]) -> List[str]:
        """The part of a Sonos list that gets P-/F- ids: the first PLAYLISTS_IN_PROMPT, the rest
        is reachable by name only."""
        if len(names) > self.PLAYLISTS_IN_PROMPT:
            info(f"{what}: {len(names)} found, the first {self.PLAYLISTS_IN_PROMPT} get prompt ids")
        return names[:self.PLAYLISTS_IN_PROMPT]

    def _refresh_spotify_playlists(self) -> bool:
        """Reloads the signed-in user's playlist library and tells the listeners. False when
        the fetch failed: the list it had stands and `spotify_library_error()` says why."""
        fetched = self._fetch_spotify_playlists()
        if fetched is None:
            return False
        playlists, total = fetched
        with self._cache_lock:
            self._spotify_playlists_cache = playlists
            self._spotify_playlists_total = total
        self._notify_library_listeners()
        return True

    @classmethod
    def wipe_targets(cls) -> list[tuple[Path, str]]:
        return [(TOKEN_PATH, "Spotify sign-in (refresh token)")]

    @classmethod
    def add_library_listener(cls, callback):
        """Calls `callback()` after each reload of the Spotify playlist library — how the prompt
        section lists a fresh sign-in's playlists at once instead of on its next check."""
        if callback not in cls._library_listeners:
            cls._library_listeners.append(callback)

    @classmethod
    def remove_library_listener(cls, callback):
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

    def _fetch_spotify_playlists(self) -> Optional[tuple[List[SpotifyPlaylist], int]]:
        """The first PLAYLISTS_IN_PROMPT playlists of the user's library in Spotify's
        order (newest added first) and the library's size — name and URI only, since Sonos
        resolves the tracks itself. Empty when no user is signed in or the sign-in has to be
        renewed, None when the fetch failed for now (the cached list stands)."""
        if not self._spotify_auth.signed_in():
            trace("Spotify: no user signed in, no playlist library to list (spotify_login())")
            self._spotify_library_error = ""
            return [], 0
        try:
            spotify = self._get_user_client()
            page = spotify.current_user_playlists(limit=self.PLAYLISTS_IN_PROMPT) or {}
            playlists: List[SpotifyPlaylist] = []
            # Spotify answers a null item for a playlist this app may not see at all (a Spotify-owned one).
            for item in page.get('items') or []:
                if item and item.get('name') and item.get('uri'):
                    owner = (item.get('owner') or {}).get('display_name') or ""
                    playlists.append(SpotifyPlaylist(item['name'], item['uri'], owner))
            total = max(int(page.get('total') or 0), len(playlists))
            info(f"Spotify: {len(playlists)} of the {total} playlists in the signed-in user's library listed")
            self._spotify_library_error = ""
            self._sign_in_again_noted = False
            return playlists, total
        except NotSignedIn as e:
            info(f"Spotify: playlist library not listed — {e}")
            self._spotify_library_error = str(e)
            self._note_sign_in_again()
            return [], 0
        except TokenUnavailable as e:
            info(f"Spotify: playlist library not reloaded, the cached list stands — {e}")
            self._spotify_library_error = str(e)
            return None
        except Exception as e:
            error(f"Error fetching the user's Spotify playlists: {e}")
            self._spotify_library_error = f"{type(e).__name__}: {e}"
            return None

    def _note_sign_in_again(self):
        """Says once, in every chat and to the LLM, that the Spotify sign-in has to be renewed."""
        if self._sign_in_again_noted or not self._spotify_auth.needs_sign_in_again():
            return
        self._sign_in_again_noted = True
        system_note("Spotify", "Spotify: the household's Spotify sign-in is no longer valid (its consent was revoked), "
                               "so the user's Spotify playlists (S- ids) are gone until spotify_login() is run again.")

    def _start_periodic_refresh(self):
        """Load playlists and favorites in the background now, then every 30 minutes."""
        self._schedule_refresh(0.1)

    def _schedule_refresh(self, delay: float):
        if self._stopped.is_set():
            return
        self._refresh_timer = threading.Timer(delay, self._refresh_tick)
        self._refresh_timer.daemon = True
        self._refresh_timer.start()

    def _refresh_tick(self):
        if self._stopped.is_set():
            trace("Sonos: refresh tick after the integration was dropped, skipped")
            return
        try:
            self._refresh_playlists_and_favorites()
        finally:
            # Whatever a refresh raises, the next one is scheduled: a stopped timer would freeze every list for good.
            self._schedule_refresh(self.REFRESH_INTERVAL_SECONDS)

    def _on_reset(self):
        """A restarted or disabled integration is dropped, not destroyed: without this its
        timer would go on discovering speakers every 30 minutes for good."""
        self._stopped.set()
        if self._refresh_timer is not None:
            self._refresh_timer.cancel()
        info("Sonos & Spotify dropped: playlist and favorites refresh stopped")

    def _uri_to_playlist_name(self, uri: str) -> Optional[str]:
        """Convert playlist URI (P-0, P-1, etc.) to playlist name."""
        try:
            if not uri.startswith('P-'):
                return None
            index = int(uri[2:])
            with self._cache_lock:
                if 0 <= index < len(self._playlists_cache):
                    return self._playlists_cache[index]
            error(f"Playlist index {index} out of range (have {len(self._playlists_cache)} playlists)")
            return None
        except ValueError:
            error(f"Invalid playlist URI format: {uri}")
            return None

    def _uri_to_favorite_name(self, uri: str) -> Optional[str]:
        """Convert favorite URI (F-0, F-1, etc.) to favorite name."""
        try:
            if not uri.startswith('F-'):
                return None
            index = int(uri[2:])
            with self._cache_lock:
                if 0 <= index < len(self._favorites_cache):
                    return self._favorites_cache[index]
            error(f"Favorite index {index} out of range (have {len(self._favorites_cache)} favorites)")
            return None
        except ValueError:
            error(f"Invalid favorite URI format: {uri}")
            return None

    def _uri_to_spotify_playlist(self, uri: str) -> Optional[SpotifyPlaylist]:
        """The user's Spotify playlist behind an S-0, S-1, … id."""
        try:
            if not uri.startswith('S-'):
                return None
            index = int(uri[2:])
            with self._cache_lock:
                if 0 <= index < len(self._spotify_playlists_cache):
                    return self._spotify_playlists_cache[index]
            error(f"Spotify playlist index {index} out of range (have {len(self._spotify_playlists_cache)} playlists)")
            return None
        except ValueError:
            error(f"Invalid Spotify playlist URI format: {uri}")
            return None

    def _generate_short_id(self) -> str:
        """Generate short IDs: AA, AB, AC, ..., AZ, BA, BB, ..., ZZ (676 total)."""
        id_num = self._next_cache_id
        self._next_cache_id = (self._next_cache_id + 1) % (26 * 26)
        first = chr(65 + (id_num // 26))
        second = chr(65 + (id_num % 26))
        return f"{first}{second}"

    def _spotify_client_settings(self) -> tuple[str, str]:
        """The app's (client_id, client_secret), read at call time so a key set through config_request works at once."""
        return (self.config.get("api_keys.spotify.client_id") or "",
                self.config.get("api_keys.spotify.client_secret") or "")

    def _get_spotify_client(self) -> Spotify:
        """The app's own client (client credentials): search, track and artist lookups."""
        if self._spotify_client is None:
            client_id, client_secret = self._spotify_client_settings()

            if not client_id or not client_secret:
                raise ValueError("Spotify credentials not found in config.yaml. Add api_keys.spotify.client_id and api_keys.spotify.client_secret")

            credentials = SpotifyClientCredentials(
                client_id=client_id,
                client_secret=client_secret
            )
            self._spotify_client = Spotify(client_credentials_manager=credentials)
            info("Spotify client initialized")

        return self._spotify_client

    def _get_user_client(self) -> Spotify:
        """A client on the signed-in user's token (the playlist library); rebuilt when the token
        was refreshed. Raises NotSignedIn."""
        token = self._spotify_auth.token()
        if self._user_client is None or self._user_client_token != token:
            self._user_client = Spotify(auth=token)
            self._user_client_token = token
            trace("Spotify user client built on a fresh access token")
        return self._user_client

    # --- the user's sign-in (spotify_login / spotify_login_code) -----------------------------

    def spotify_signed_in(self) -> bool:
        return self._spotify_auth.signed_in()

    def login_begin(self) -> SignIn:
        """Starts the sign-in of the user's Spotify account; raises RuntimeError without a client id."""
        return self._spotify_auth.begin()

    def login_finish(self, pasted: str) -> str:
        """"" when the pasted answer signed the user in — the playlist library is then reloaded
        at once; `spotify_library_error()` says whether that worked — else why not."""
        failure = self._spotify_auth.finish(pasted)
        if failure:
            return failure
        self._user_client = None
        self._sign_in_again_noted = False
        self._refresh_spotify_playlists()
        return ""

    def spotify_library_error(self) -> str:
        """Why the last fetch of the user's playlist library failed, "" when it did not."""
        return self._spotify_library_error

    def spotify_playlists_total(self) -> int:
        """The size of the signed-in user's library, of which `get_spotify_playlists()` is the first part."""
        with self._cache_lock:
            return self._spotify_playlists_total

    def spotify_user_name(self) -> str:
        """The signed-in user's display name, or "" when it cannot be read (nothing depends on it)."""
        try:
            profile = self._get_user_client().current_user() or {}
            return str(profile.get('display_name') or profile.get('id') or "")
        except Exception as e:
            trace(f"Spotify: the signed-in user's profile could not be read: {e}")
            return ""

    def _discover_sonos_zones(self) -> Dict[str, soco.SoCo]:
        current_time = time.time()

        cache_expired = (
            self._sonos_zones_cache_time is None or
            (current_time - self._sonos_zones_cache_time) > MusicIntegration.CACHE_EXPIRY_SECONDS
        )

        if not self._sonos_zones_cache or cache_expired:
            found: Dict[str, soco.SoCo] = {}
            for zone in soco.discover() or []:
                # The name is an HTTP GET to the speaker: one offline zone must not empty the house
                try:
                    found[zone.player_name] = zone
                except Exception as e:
                    info(f"Sonos: zone at {getattr(zone, 'ip_address', '?')} skipped, unreachable: {e}")
            if found:
                self._sonos_zones_cache = found
                self._sonos_zones_cache_time = current_time
            else:
                trace("Sonos: discovery found no reachable zone, keeping the cache as it is")
        return self._sonos_zones_cache

    def _get_zone(self, room_name: str) -> Optional[soco.SoCo]:
        normalized_name = self.normalize_name(room_name)
        zones = self._discover_sonos_zones()

        # Try exact match first
        zone = zones.get(normalized_name)
        if zone:
            return zone

        # Fall back to case-insensitive match
        normalized_lower = normalized_name.lower()
        for zone_name, zone in zones.items():
            if zone_name.lower() == normalized_lower:
                return zone

        error(f"Zone '{normalized_name}' not found")
        return None

    def _get_coordinator(self, zone: soco.SoCo) -> soco.SoCo:
        """Get the group coordinator for a zone. If zone is in a group, returns the coordinator.
        If zone is standalone, returns the zone itself. A zone the topology no longer lists —
        a speaker that went offline — has no group: that is a connection error, not a crash."""
        group = zone.group
        if group is None:
            raise ConnectionError("the zone is not in the Sonos topology (speaker offline?)")
        return group.coordinator

    def get_all_zones_status(self) -> str:
        """One line per zone. A zone that cannot be reached — its speaker offline, its HTTP
        refused — is listed as unreachable and dropped from the zone cache, so the next
        discovery re-finds it once it is back; the other zones' status is kept."""
        zones = self._discover_sonos_zones()
        res = ""

        for zone_name, zone in list(zones.items()):
            name = self.room_mapping.get(zone_name, zone_name)
            if name == "":
                continue
            try:
                coordinator = self._get_coordinator(zone)
                transport_info = coordinator.get_current_transport_info()
                track_info = coordinator.get_current_track_info()
            except Exception as e:
                error(f"Sonos zone {zone_name} is unreachable ({e}) — listed as unreachable and dropped "
                      f"from the zone cache until the next discovery finds it")
                zones.pop(zone_name, None)
                res += f"{name} (unreachable)\n"
                continue

            if transport_info.get('current_transport_state') == 'PLAYING':
                state = "Playing"
            else:
                state = "Paused"

            title = track_info.get('title', '').strip()
            artist = track_info.get('artist', '').strip()

            if title and artist:
                now_playing = f"{title} by {artist} - {state}"
            elif title or artist:
                now_playing = f"{title or artist} - {state}"
            else:
                now_playing = state

            res += f"{name} ({now_playing})\n"
        return res.strip()

    @staticmethod
    def _popularity(item: dict) -> str:
        """The popularity suffix, or nothing: Spotify dropped the field for Development Mode apps in February 2026."""
        popularity = item.get('popularity')
        return f" (popularity {popularity}/100)" if popularity is not None else ""

    def search_spotify(self, query: str, limit: int = 4) -> str:
        try:
            res = f"Searching Spotify for '{query}':\n\n"
            spotify = self._get_spotify_client()

            results = spotify.search(q=query, type='track,artist,playlist', market='SE', limit=limit)

            with self._cache_lock:
                res += "TRACKS:\n"
                if results['tracks']['items']:
                    for item in results['tracks']['items']:
                        short_id = self._generate_short_id()
                        self._spotify_search_cache[short_id] = item['uri']
                        artists = ', '.join([artist['name'] for artist in item['artists']])
                        res += f"{item['name']} - {artists}{self._popularity(item)} ID: {short_id}\n"
                else:
                    res += "No tracks found.\n"

                res += "\nARTISTS:\n"
                if results['artists']['items']:
                    for item in results['artists']['items']:
                        short_id = self._generate_short_id()
                        self._spotify_search_cache[short_id] = item['uri']
                        genres = ', '.join(item.get('genres', [])[:2]) if item.get('genres') else 'N/A'
                        res += f"{item['name']} - {genres}{self._popularity(item)} ID: {short_id}\n"
                else:
                    res += "No artists found.\n"

                res += "\nPLAYLISTS:\n"
                # Spotify answers a null item for a playlist this app may not see (an algorithmic one).
                playlists = [p for p in (results.get('playlists') or {}).get('items') or [] if p]
                if playlists:
                    for item in playlists:
                        short_id = self._generate_short_id()
                        self._spotify_search_cache[short_id] = item['uri']
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

    def _artist_top_tracks(self, track_id: str) -> list:
        """The seed track's artist's top tracks, or none when this app may not ask for them.

        Spotify removed the top-tracks endpoint for Development Mode apps in February 2026; an app created since
        gets a 403 or 404 here, and the radio is then the seed track alone rather than a failure."""
        spotify = self._get_spotify_client()
        track_info = spotify.track(track_id)
        artist_id = track_info['artists'][0]['id']
        artist_name = track_info['artists'][0]['name']
        try:
            similar_tracks = spotify.artist_top_tracks(artist_id, country='SE')['tracks']
        except SpotifyException as e:
            if e.http_status not in (403, 404):
                raise
            info(f"Top tracks of {artist_name} not available to this Spotify app (HTTP {e.http_status}, Development Mode) — playing the seed track alone")
            return []
        trace(f"Building radio from {len(similar_tracks)} top tracks by {artist_name}")
        return similar_tracks

    def play_in_zone(self, id: str, room_name: str) -> bool:
        zone = self._get_zone(room_name)
        if not zone:
            error(f"Could not find zone {room_name}")
            return False

        coordinator = self._get_coordinator(zone)

        try:
            # Check if it's a short ID from Spotify search cache (AA, AB, ..., ZZ)
            if len(id) == 2 and id.isalpha() and id.isupper():
                with self._cache_lock:
                    full_uri = self._spotify_search_cache.get(id)
                    if full_uri:
                        trace(f"Resolved short ID {id} to {full_uri}")
                        id = full_uri
                    else:
                        error(f"Short ID {id} not found in cache")
                        return False

            if id.startswith('P-'):
                playlist_name = self._uri_to_playlist_name(id)
                if not playlist_name:
                    error(f"Could not resolve playlist URI: {id}")
                    return False
                return self.play_playlist(room_name, playlist_name)
            elif id.startswith('F-'):
                favorite_name = self._uri_to_favorite_name(id)
                if not favorite_name:
                    error(f"Could not resolve favorite URI: {id}")
                    return False
                return self.play_favorite(room_name, favorite_name)
            elif id.startswith('S-'):
                playlist = self._uri_to_spotify_playlist(id)
                if playlist is None:
                    error(f"Could not resolve Spotify playlist URI: {id}")
                    return False
                return self.play_spotify_playlist(room_name, playlist)
            else:
                spotify_id = id.split(":")[-1]

                if ":playlist:" in id or "/playlist/" in id:
                    shuffle_msg = self._play_spotify_share(coordinator, id, shuffle=True)
                    info(f"Playing Spotify playlist {id} in {room_name}{shuffle_msg}")
                    return True
                elif "artist" in id:
                    sonos_uri = f"x-sonosapi-radio:spotify%3aartistRadio%3a{spotify_id}?sid=9&flags=8300&sn=4"
                    trace(f"Converted {id} to artist radio: {sonos_uri}")
                    coordinator.play_uri(sonos_uri)
                    info(f"Playing Spotify artist radio based on {id} in {room_name}")
                    return True
                elif "track" in id:
                    # Play the track and then shuffle top tracks from artist
                    try:
                        coordinator.clear_queue()

                        # Add the seed track first
                        track_uri = f"x-sonos-spotify:spotify:track:{spotify_id}?sid=9&flags=8224&sn=4"
                        coordinator.add_uri_to_queue(track_uri)
                        trace(f"Added seed track {spotify_id} to queue")

                        similar_tracks = self._artist_top_tracks(spotify_id)

                        # Add tracks to queue (skip the seed track if it appears)
                        tracks_added = 0
                        for track in similar_tracks:
                            if track['id'] == spotify_id:
                                continue  # Skip seed track (already added)
                            try:
                                rec_uri = f"x-sonos-spotify:spotify:track:{track['id']}?sid=9&flags=8224&sn=4"
                                coordinator.add_uri_to_queue(rec_uri)
                                tracks_added += 1
                            except Exception as e:
                                trace(f"Failed to add track {track['id']} to queue: {e}")
                                continue

                        # Enable shuffle for variety (if supported)
                        try:
                            coordinator.play_mode = 'SHUFFLE'
                            shuffled_msg = ", shuffled"
                        except Exception as e:
                            trace(f"Shuffle mode not supported, playing in order: {e}")
                            shuffled_msg = ""

                        # Start playback from beginning of queue
                        coordinator.play_from_queue(0)
                        info(f"Playing track radio for {id} in {room_name} (seed track + {tracks_added} similar tracks{shuffled_msg})")
                        return True
                    except Exception as e:
                        error(f"Error building track radio for {id}: {e}")
                        # Fallback to playing just the track
                        try:
                            track_uri = f"x-sonos-spotify:spotify:track:{spotify_id}?sid=9&flags=8224&sn=4"
                            coordinator.play_uri(track_uri)
                            info(f"Fallback: Playing single track {id} in {room_name}")
                            return True
                        except Exception as fallback_error:
                            error(f"Fallback also failed: {fallback_error}")
                            return False
                else:
                    error(f"Unknown Spotify URI type: {id}")
                    return False
        except Exception as e:
            error(f"Error playing uri in {room_name}: {e}")
            return False

    def play(self, room_name: str) -> bool:
        zone = self._get_zone(room_name)
        if not zone:
            return False

        coordinator = self._get_coordinator(zone)

        try:
            coordinator.play()
            info(f"Playing in {room_name}")
            return True
        except Exception as e:
            error(f"Error playing in {room_name}: {e}")
            return False

    def pause(self, room_name: str) -> bool:
        zone = self._get_zone(room_name)
        if not zone:
            return False

        coordinator = self._get_coordinator(zone)

        try:
            coordinator.pause()
            info(f"Paused in {room_name}")
            return True
        except Exception as e:
            error(f"Error pausing in {room_name}: {e}")
            return False

    def next_track(self, room_name: str) -> bool:
        zone = self._get_zone(room_name)
        if not zone:
            return False

        coordinator = self._get_coordinator(zone)

        try:
            coordinator.next()
            info(f"Skipping to next track in {room_name}")
            return True
        except Exception as e:
            error(f"Error skipping track in {room_name}: {e}")
            return False

    def adjust_volume(self, room_name: str, adjustment: int) -> bool:
        zone = self._get_zone(room_name)
        if not zone:
            return False

        try:
            for member in zone.group.members:
                current_volume = member.volume
                new_volume = max(0, min(100, current_volume + adjustment))
                member.volume = new_volume
            info(f"Adjusted volume by {adjustment} in {room_name} (group of {len(zone.group.members)})")
            return True
        except Exception as e:
            error(f"Error adjusting volume in {room_name}: {e}")
            return False

    def _fetch_playlists_from_sonos(self, room_name: str) -> List[str]:
        """Internal method to fetch playlists directly from Sonos."""
        zone = self._get_zone(room_name)
        if not zone:
            return []

        try:
            playlists = zone.get_sonos_playlists()
            playlist_names = [playlist.title for playlist in playlists]
            return playlist_names
        except Exception as e:
            error(f"Error fetching playlists from Sonos: {e}")
            return []

    def _fetch_favorites_from_sonos(self, room_name: str) -> List[str]:
        """Internal method to fetch favorites directly from Sonos."""
        zone = self._get_zone(room_name)
        if not zone:
            return []

        try:
            favorites = zone.music_library.get_sonos_favorites()
            favorite_names = []
            for fav in favorites:
                title = fav.title() if callable(fav.title) else fav.title
                favorite_names.append(title)
            return favorite_names
        except Exception as e:
            error(f"Error fetching favorites from Sonos: {e}")
            return []

    def get_playlists(self, room_name: str = None) -> List[str]:
        """Get cached list of playlists. room_name parameter kept for compatibility but not used."""
        with self._cache_lock:
            return self._playlists_cache.copy()

    def get_favorites(self, room_name: str = None) -> List[str]:
        """Get cached list of favorites. room_name parameter kept for compatibility but not used."""
        with self._cache_lock:
            return self._favorites_cache.copy()

    def get_spotify_playlists(self) -> List[SpotifyPlaylist]:
        """The cached playlist library of the signed-in Spotify user, in S-0, S-1, … order."""
        with self._cache_lock:
            return self._spotify_playlists_cache.copy()

    def _play_spotify_share(self, coordinator: soco.SoCo, uri: str, shuffle: bool) -> str:
        """Clears the queue, adds a Spotify container (playlist, album) through the Sonos Spotify
        service and starts it — Sonos resolves the tracks with the household's own account.
        Returns the shuffle suffix for the log line."""
        trace(f"Using ShareLinkPlugin to play Spotify container {uri}")
        share = ShareLinkPlugin(coordinator)
        coordinator.clear_queue()
        share.add_share_link_to_queue(uri)
        shuffle_msg = ""
        if shuffle:
            try:
                coordinator.play_mode = 'SHUFFLE'
                shuffle_msg = " (shuffled)"
            except Exception as e:
                trace(f"Could not enable shuffle: {e}")
        coordinator.play_from_queue(0)
        return shuffle_msg

    def play_spotify_playlist(self, room_name: str, playlist: SpotifyPlaylist, shuffle: bool = True) -> bool:
        """Starts one of the signed-in user's playlists by its Spotify URI."""
        zone = self._get_zone(room_name)
        if not zone:
            return False

        coordinator = self._get_coordinator(zone)
        try:
            shuffle_msg = self._play_spotify_share(coordinator, playlist.uri, shuffle)
            info(f"Playing Spotify playlist '{playlist.name}' ({playlist.uri}) in {room_name}{shuffle_msg}")
            return True
        except Exception as e:
            error(f"Error playing Spotify playlist '{playlist.name}' ({playlist.uri}) in {room_name}: {e}")
            return False

    def play_playlist(self, room_name: str, playlist_name: str, shuffle: bool = True) -> bool:
        zone = self._get_zone(room_name)
        if not zone:
            return False

        coordinator = self._get_coordinator(zone)

        try:
            playlists = coordinator.get_sonos_playlists()
            target_playlist = None

            for playlist in playlists:
                if playlist.title == playlist_name:
                    target_playlist = playlist
                    break

            if not target_playlist:
                error(f"Playlist '{playlist_name}' not found")
                return False

            coordinator.clear_queue()
            coordinator.add_to_queue(target_playlist)

            shuffle_msg = ""
            if shuffle:
                try:
                    coordinator.play_mode = 'SHUFFLE'
                    shuffle_msg = " (shuffled)"
                except Exception:
                    pass  # Shuffle not supported for this content

            coordinator.play_from_queue(0)
            info(f"Playing playlist '{playlist_name}' in {room_name}{shuffle_msg}")
            return True
        except Exception as e:
            error(f"Error playing playlist '{playlist_name}': {e}")
            return False

    def play_favorite(self, room_name: str, favorite_name: str, shuffle: bool = True) -> bool:
        zone = self._get_zone(room_name)
        if not zone:
            return False

        coordinator = self._get_coordinator(zone)

        try:
            favorites = coordinator.music_library.get_sonos_favorites()
            target_favorite = None

            for fav in favorites:
                title = fav.title() if callable(fav.title) else fav.title
                if title == favorite_name:
                    target_favorite = fav
                    break

            if not target_favorite:
                error(f"Favorite '{favorite_name}' not found")
                return False

            uri = target_favorite.get_uri()
            shuffle_msg = ""

            # Check if this is a Spotify container (playlist/album) that needs ShareLinkPlugin
            if uri.startswith('x-rincon-cpcontainer:'):
                parsed_uri = urllib.parse.unquote(uri)
                # Match playlist, album, track, or artist
                spotify_match = re.search(r'spotify[:%](playlist|album|track|artist)[:%]([a-zA-Z0-9]+)', parsed_uri)

                if spotify_match:
                    content_type = spotify_match.group(1)
                    content_id = spotify_match.group(2)
                    spotify_url = f'https://open.spotify.com/{content_type}/{content_id}'
                    shuffle_msg = self._play_spotify_share(coordinator, spotify_url, shuffle)
                else:
                    # Other container type - try play_uri as fallback
                    coordinator.play_uri(uri)
                    if shuffle:
                        try:
                            coordinator.play_mode = 'SHUFFLE'
                            shuffle_msg = " (shuffled)"
                        except Exception as e:
                            trace(f"Could not enable shuffle: {e}")
            else:
                # Direct URIs (radio stations, etc.) can be played directly
                coordinator.play_uri(uri)
                if shuffle:
                    try:
                        coordinator.play_mode = 'SHUFFLE'
                        shuffle_msg = " (shuffled)"
                    except Exception as e:
                        trace(f"Could not enable shuffle for favorite (not supported for this type): {e}")

            info(f"Playing favorite '{favorite_name}' in {room_name}{shuffle_msg}")
            return True
        except Exception as e:
            error(f"Error playing favorite '{favorite_name}': {e}")
            return False

    def clear_queue(self, room_name: str) -> bool:
        zone = self._get_zone(room_name)
        if not zone:
            return False

        coordinator = self._get_coordinator(zone)

        try:
            coordinator.clear_queue()
            info(f"Cleared queue in {room_name}")
            return True
        except Exception as e:
            error(f"Error clearing queue in {room_name}: {e}")
            return False

    def normalize_name(self, room_name: str) -> str:
        room_name_lower = room_name.lower()
        for zone_name, simplified_name in self.room_mapping.items():
            if simplified_name.lower() == room_name_lower:
                if zone_name == "":
                    error(f"Room '{room_name}' maps to invalid zone (empty string)")
                    return room_name
                return zone_name

        return room_name


# Backward-compatible alias
MusicManager = MusicIntegration
