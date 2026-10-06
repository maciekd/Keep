import functools
import re
import threading
import time
import urllib.parse
from typing import List, Dict, Optional

import soco
from soco.plugins.sharelink import ShareLinkPlugin
from soco.zonegroupstate import ZoneGroupState
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials

from common.logging import trace, info, error
from common.ConfigManager import ConfigManager
from ..Integration import Integration

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
    """Controls Sonos zones and plays Spotify content, playlists and favorites in them.

    Playlists and favorites are cached and refreshed periodically; Spotify search results
    get short ids the LLM can refer back to within the conversation.
    """

    CACHE_EXPIRY_SECONDS = 30 * 60
    REFRESH_INTERVAL_SECONDS = 30 * 60

    def __init__(self):
        Integration.__init__(self)
        self.config = ConfigManager()
        self._spotify_client: Optional[Spotify] = None
        self._sonos_zones_cache: Dict[str, soco.SoCo] = {}
        self._sonos_zones_cache_time: Optional[float] = None

        # Cache for playlists and favorites with their indices
        self._playlists_cache: List[str] = []
        self._favorites_cache: List[str] = []
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
            return True
        except Exception:
            return False

    def _refresh_playlists_and_favorites(self):
        """Load playlists and favorites and cache them with their indices."""
        try:
            # Use any available zone to fetch playlists and favorites
            zones = self._discover_sonos_zones()
            if not zones:
                info("No Sonos zones available for fetching playlists/favorites")
                return

            zone_name = next(iter(zones.keys()))

            with self._cache_lock:
                self._playlists_cache = self._fetch_playlists_from_sonos(zone_name)
                self._favorites_cache = self._fetch_favorites_from_sonos(zone_name)
        except Exception as e:
            error(f"Error refreshing playlists and favorites cache: {e}")

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
        self._refresh_playlists_and_favorites()
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

    def _generate_short_id(self) -> str:
        """Generate short IDs: AA, AB, AC, ..., AZ, BA, BB, ..., ZZ (676 total)."""
        id_num = self._next_cache_id
        self._next_cache_id = (self._next_cache_id + 1) % (26 * 26)
        first = chr(65 + (id_num // 26))
        second = chr(65 + (id_num % 26))
        return f"{first}{second}"

    def _get_spotify_client(self) -> Spotify:
        if self._spotify_client is None:
            client_id = self.config.get("api_keys.spotify.client_id")
            client_secret = self.config.get("api_keys.spotify.client_secret")

            if not client_id or not client_secret:
                raise ValueError("Spotify credentials not found in config.yaml. Add api_keys.spotify.client_id and api_keys.spotify.client_secret")

            credentials = SpotifyClientCredentials(
                client_id=client_id,
                client_secret=client_secret
            )
            self._spotify_client = Spotify(client_credentials_manager=credentials)
            info("Spotify client initialized")

        return self._spotify_client

    def _discover_sonos_zones(self) -> Dict[str, soco.SoCo]:
        current_time = time.time()

        cache_expired = (
            self._sonos_zones_cache_time is None or
            (current_time - self._sonos_zones_cache_time) > MusicIntegration.CACHE_EXPIRY_SECONDS
        )

        if not self._sonos_zones_cache or cache_expired:
            self._sonos_zones_cache.clear()
            zones = soco.discover()
            if zones:
                for zone in zones:
                    self._sonos_zones_cache[zone.player_name] = zone
                self._sonos_zones_cache_time = current_time
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
        If zone is standalone, returns the zone itself."""
        coordinator = zone.group.coordinator
        return coordinator

    def get_all_zones_status(self) -> str:
        zones = self._discover_sonos_zones()
        res = ""

        for zone_name, zone in zones.items():
            try:
                name = self.room_mapping.get(zone_name, zone_name)

                if name == "":
                    continue

                coordinator = self._get_coordinator(zone)
                transport_info = coordinator.get_current_transport_info()
                track_info = coordinator.get_current_track_info()

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
            except Exception as e:
                error(f"Error getting status for zone {zone_name}: {e}")
                raise
        return res.strip()

    def search_spotify(self, query: str, limit: int = 4) -> str:
        try:
            res = f"Searching Spotify for '{query}':\n\n"
            spotify = self._get_spotify_client()

            results = spotify.search(q=query, type='track,artist', market='SE', limit=limit)

            with self._cache_lock:
                res += "TRACKS:\n"
                if results['tracks']['items']:
                    for item in results['tracks']['items']:
                        short_id = self._generate_short_id()
                        self._spotify_search_cache[short_id] = item['uri']
                        res += f"{item['name']} - {', '.join([artist['name'] for artist in item['artists']])} (popularity {item['popularity']}/100) ID: {short_id}\n"
                else:
                    res += "No tracks found.\n"

                res += "\nARTISTS:\n"
                if results['artists']['items']:
                    for item in results['artists']['items']:
                        short_id = self._generate_short_id()
                        self._spotify_search_cache[short_id] = item['uri']
                        genres = ', '.join(item.get('genres', [])[:2]) if item.get('genres') else 'N/A'
                        res += f"{item['name']} - {genres} (popularity {item['popularity']}/100) ID: {short_id}\n"
                else:
                    res += "No artists found.\n"

            return res
        except Exception as e:
            e = f"Error searching Spotify: {e}"
            error(e)
            return e

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
            else:
                spotify_id = id.split(":")[-1]

                if "artist" in id:
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

                        # Get similar tracks from Spotify (artist's top tracks)
                        spotify = self._get_spotify_client()
                        track_info = spotify.track(spotify_id)
                        artist_id = track_info['artists'][0]['id']
                        artist_name = track_info['artists'][0]['name']

                        # Get all artist's top tracks
                        top_tracks = spotify.artist_top_tracks(artist_id, country='SE')
                        similar_tracks = top_tracks['tracks']

                        trace(f"Building radio from {len(similar_tracks)} top tracks by {artist_name}")

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
                    trace(f"Using ShareLinkPlugin to play Spotify {content_type}: {spotify_url}")

                    share = ShareLinkPlugin(coordinator)
                    coordinator.clear_queue()
                    share.add_share_link_to_queue(spotify_url)

                    if shuffle:
                        try:
                            coordinator.play_mode = 'SHUFFLE'
                            shuffle_msg = " (shuffled)"
                        except Exception as e:
                            trace(f"Could not enable shuffle: {e}")

                    coordinator.play_from_queue(0)
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
