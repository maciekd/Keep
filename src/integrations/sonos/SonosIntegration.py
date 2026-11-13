import functools
import re
import threading
import time
import urllib.parse
from typing import List, Dict, Optional

import soco
from soco.plugins.sharelink import ShareLinkPlugin
from soco.zonegroupstate import ZoneGroupState

from common.logging import trace, info, error
from common.ConfigManager import ConfigManager
from assistant.integration.Integration import Integration
from assistant.integration.music.MusicSystem import LIBRARY_REFRESH_SECONDS, MusicSystem
from assistant.integration.music.SpotifyAccount import SIGN_IN_REVOKED, SpotifyAccount, SpotifyPlaylist

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


class SonosIntegration(Integration, MusicSystem):
    """Sonos speakers: playback, volume, favorites and Spotify in every zone.

    Plays the Sonos favorites and playlists and Spotify music in the Sonos zones, with pause,
    skip and volume. Keep can search Spotify and start what it finds in any room, and once the
    user's Spotify account is signed in, the playlists of their own library by name.

    Sonos playlists (`P-n`) and Sonos favorites (`F-n`) are cached and refreshed every 30
    minutes, cut to their first `PLAYLISTS_IN_PROMPT` entries in the Sonos app's order — every one
    of them is a prompt line on every request. Spotify — search and its short ids, the user's
    playlist library (`S-n`), the sign-in — is the shared `SpotifyAccount`; this integration's
    30-minute refresh drives the library's reload too. A playlist is started by its URI through
    the Sonos Spotify service (ShareLinkPlugin), so Sonos resolves the tracks with the household's
    Spotify account linked in the Sonos app. Spotify-owned playlists (Discover Weekly, Release
    Radar) are hidden from a development-mode app's library; a Sonos favorite is the way to reach
    them. The Spotify app keys are required (`config_keys`): search is what the play commands
    start from. Until 2026-11-13 this was "Sonos & Spotify" in the folder `music`."""

    former_names = ("Sonos & Spotify",)

    PLAYLISTS_IN_PROMPT = 10
    ID_PREFIXES = ("F-", "P-")

    CACHE_EXPIRY_SECONDS = 30 * 60
    REFRESH_INTERVAL_SECONDS = LIBRARY_REFRESH_SECONDS

    def __init__(self):
        Integration.__init__(self)
        self.config = ConfigManager()
        self._sonos_zones_cache: Dict[str, soco.SoCo] = {}
        self._sonos_zones_cache_time: Optional[float] = None

        # Cache for playlists and favorites with their indices
        self._playlists_cache: List[str] = []
        self._favorites_cache: List[str] = []
        self._cache_lock = threading.Lock()

        self._refresh_timer: Optional[threading.Timer] = None
        self._stopped = threading.Event()

        # Load initial cache and start periodic refresh
        self._start_periodic_refresh()

    @property
    def name(self) -> str:
        return "Sonos"

    @property
    def spotify(self) -> SpotifyAccount:
        return SpotifyAccount.get_instance()

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
            self.spotify.app_client()
        except Exception as e:
            info(f"Sonos unhealthy, the Spotify app keys do not work: {e}")
            return False
        if self.spotify.needs_sign_in_again():
            # Raised, not returned: the monitor's last_error then names the cure in list_integrations and the console.
            raise RuntimeError(SIGN_IN_REVOKED)
        return True

    def _refresh_playlists_and_favorites(self):
        """Load playlists and favorites and cache them with their indices. Each part is guarded
        on its own, so a Spotify failure never skips the Sonos lists or the other way round."""
        try:
            self.spotify.refresh_library()
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
        info("Sonos dropped: playlist and favorites refresh stopped")

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

    def _discover_sonos_zones(self) -> Dict[str, soco.SoCo]:
        current_time = time.time()

        cache_expired = (
            self._sonos_zones_cache_time is None or
            (current_time - self._sonos_zones_cache_time) > SonosIntegration.CACHE_EXPIRY_SECONDS
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

    def zone_problem(self, room_name: str) -> str:
        """"" when the room names a reachable Sonos zone, else why not — the zone commands
        return it as their error, so the LLM can tell the user instead of falling silent."""
        if self._get_zone(room_name):
            return ""
        known = sorted(self._sonos_zones_cache)
        info(f"Sonos: no zone '{room_name}', {len(known)} known")
        if not known:
            return f"no Sonos zone '{room_name}': no Sonos speaker answers on the network, so nothing can play"
        return f"no Sonos zone '{room_name}'; the zones are: {', '.join(known)}"

    def _get_coordinator(self, zone: soco.SoCo) -> soco.SoCo:
        """Get the group coordinator for a zone. If zone is in a group, returns the coordinator.
        If zone is standalone, returns the zone itself. A zone the topology no longer lists —
        a speaker that went offline — has no group: that is a connection error, not a crash."""
        group = zone.group
        if group is None:
            raise ConnectionError("the zone is not in the Sonos topology (speaker offline?)")
        return group.coordinator

    # --- MusicSystem -----------------------------------------------------------------------

    def zones(self) -> List[str]:
        names = []
        for zone_name in self._discover_sonos_zones():
            name = self.room_mapping.get(zone_name, zone_name)
            if name:
                names.append(name)
        return names

    def zone_status(self) -> List[str]:
        return [line for line in self.get_all_zones_status().splitlines() if line]

    def library_text(self) -> str:
        """The Sonos favorites (F-n) and playlists (P-n), each list in the Sonos app's order."""
        lines = [f"{name} (ID: F-{index})" for index, name in enumerate(self.get_favorites())]
        lines += [f"{name} (ID: P-{index})" for index, name in enumerate(self.get_playlists())]
        trace(f"Sonos: {len(lines)} favorites and playlists for the MUSIC section")
        return "\n".join(lines)

    def play_id(self, zone: str, music_id: str) -> str:
        if self.play_in_zone(music_id, zone):
            return ""
        return f"could not start {music_id} in {zone} (see the server log)"

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

    def play_in_zone(self, id: str, room_name: str) -> bool:
        zone = self._get_zone(room_name)
        if not zone:
            error(f"Could not find zone {room_name}")
            return False

        coordinator = self._get_coordinator(zone)

        try:
            # Check if it's a short ID from Spotify search cache (AA, AB, ..., ZZ)
            if SpotifyAccount.is_search_id(id):
                full_uri = self.spotify.resolve(id)
                if not full_uri:
                    error(f"Short ID {id} not found in cache")
                    return False
                trace(f"Resolved short ID {id} to {full_uri}")
                id = full_uri

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
                playlist = self.spotify.playlist(id)
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

                        similar_tracks = self.spotify.artist_top_tracks(spotify_id)

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

    def play(self, room_name: str) -> str:
        zone = self._get_zone(room_name)
        if not zone:
            return self.zone_problem(room_name)

        try:
            coordinator = self._get_coordinator(zone)
            coordinator.play()
            info(f"Playing in {room_name}")
            return ""
        except Exception as e:
            error(f"Error playing in {room_name}: {e}")
            return f"could not play in {room_name}: {e}"

    def pause(self, room_name: str) -> str:
        zone = self._get_zone(room_name)
        if not zone:
            return self.zone_problem(room_name)

        try:
            coordinator = self._get_coordinator(zone)
            coordinator.pause()
            info(f"Paused in {room_name}")
            return ""
        except Exception as e:
            error(f"Error pausing in {room_name}: {e}")
            return f"could not pause in {room_name}: {e}"

    def next_track(self, room_name: str) -> str:
        zone = self._get_zone(room_name)
        if not zone:
            return self.zone_problem(room_name)

        try:
            coordinator = self._get_coordinator(zone)
            coordinator.next()
            info(f"Skipping to next track in {room_name}")
            return ""
        except Exception as e:
            error(f"Error skipping track in {room_name}: {e}")
            return f"could not skip the track in {room_name}: {e}"

    def adjust_volume(self, room_name: str, adjustment: int) -> str:
        zone = self._get_zone(room_name)
        if not zone:
            return self.zone_problem(room_name)

        try:
            for member in zone.group.members:
                current_volume = member.volume
                new_volume = max(0, min(100, current_volume + adjustment))
                member.volume = new_volume
            info(f"Adjusted volume by {adjustment} in {room_name} (group of {len(zone.group.members)})")
            return ""
        except Exception as e:
            error(f"Error adjusting volume in {room_name}: {e}")
            return f"could not change the volume in {room_name}: {e}"

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

