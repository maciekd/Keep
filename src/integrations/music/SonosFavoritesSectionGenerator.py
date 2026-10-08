import time
from assistant.prompt.PromptSectionGenerator import PromptSectionGenerator
from common.logging import trace, info, error
from integrations.music.MusicIntegration import MusicManager


class SonosFavoritesSectionGenerator(PromptSectionGenerator):
    """Keeps the Sonos favorites, Sonos playlists and the user's Spotify playlists section of the prompt fresh.

    It reads the lists the integration caches (reloaded every 30 minutes, and the Spotify
    library again right after a sign-in) and never fetches itself: checking them once a minute
    costs nothing, and the base loop skips the prompt rebuild when nothing changed. It reads
    the integration the IntegrationMonitor built and never builds one; it runs only while
    Sonos & Spotify is on, so a house without it never looks for a speaker."""
    key = "sonos"
    heading = "# SONOS PLAYLISTS AND ZONES"
    intro = ("Favorites, Sonos playlists and the user's own Spotify playlists the `music` toolkit can start "
             "(ids `F-0`, `P-0`, `S-0`, …), curated by the user:")
    CHECK_INTERVAL = 60

    def __init__(self, prompt_manager):
        super().__init__(prompt_manager, "SonosFavorites")

    @property
    def playback_manager(self):
        """The running integration, or None while it is being rebuilt (restart_integration)."""
        return MusicManager.peek_instance()

    def _playlists(self) -> list:
        manager = self.playback_manager
        return manager.get_playlists() if manager is not None else []

    def _favorites(self) -> list:
        manager = self.playback_manager
        return manager.get_favorites() if manager is not None else []

    def _spotify_playlists(self) -> list:
        manager = self.playback_manager
        return manager.get_spotify_playlists() if manager is not None else []

    def _has_data(self) -> bool:
        return bool(self._playlists() or self._favorites() or self._spotify_playlists())

    def _start_monitoring(self):
        """Load once the integration's data is in, then follow the cached lists."""
        self._await_first_data()
        self._refresh()

    def _await_first_data(self):
        """Poll until playlists/favorites are loaded (with timeout)."""
        max_wait = 60  # seconds
        start_time = time.time()

        while not self._stop_event.is_set():
            if self._has_data():
                info("Sonos data loaded, updating content")
                self._update_content()
                break

            if time.time() - start_time > max_wait:
                info(f"No Sonos favorites or playlists after {max_wait}s (no speakers found, or Spotify not "
                     f"configured) — section stays empty, checked again every {self.CHECK_INTERVAL}s")
                break

            self._wait(0.1)

    def _refresh(self):
        """Follows the integration's caches: a changed list (the 30-minute reload, a Spotify
        sign-in) reaches the prompt within a minute."""
        while not self._stop_event.is_set():
            self._wait(self.CHECK_INTERVAL)
            if self._stop_event.is_set():
                break
            if self._has_data():
                self._update_content()
            else:
                trace("Sonos: still no favorites or playlists, section left empty")

    def _fetch_content(self) -> str:
        """Fetch Sonos favorites and playlists."""
        trace("Fetching Sonos data...")
        res = self._fetch_sonos_favorites()
        res = self._fetch_sonos_playlists(res)
        res = self._fetch_spotify_playlists(res)
        trace(f"Successfully fetched {len(res.splitlines())} lines of Sonos favorites and playlists.")
        return res.strip()

    def _fetch_spotify_playlists(self, res) -> str:
        """The signed-in user's Spotify playlists (owned and followed), appended with S- ids."""
        try:
            playlists = self._spotify_playlists()
            if playlists:
                res += "\nThe user's own Spotify playlists (incl. followed ones such as Discover Weekly):\n"
            for index, playlist in enumerate(playlists):
                by = f" by {playlist.owner}" if playlist.owner else ""
                res += f"{playlist.name}{by} (ID: S-{index})\n"
            trace(f"Found {len(playlists)} Spotify playlists of the signed-in user")
        except Exception as e:
            error(f"Error listing the user's Spotify playlists: {e}")
        return res

    def _fetch_sonos_playlists(self, res) -> str:
        """Fetch Sonos playlists and append to result."""
        trace("Fetching Sonos playlists...")
        try:
            playlists = self._playlists()
            index = 0
            for playlist_name in playlists:
                res += playlist_name + ' (ID: P-' + str(index) + ')\n'
                index += 1
            trace(f"Found {len(playlists)} Sonos playlists")
        except Exception as e:
            error(f"Error fetching Sonos playlists: {e}")
        return res

    def _fetch_sonos_favorites(self) -> str:
        """Fetch Sonos favorites."""
        trace("Fetching Sonos favorites...")
        res = ""
        try:
            favorites = self._favorites()
            index = 0
            for favorite_name in favorites:
                res += favorite_name + ' (ID: F-' + str(index) + ')\n'
                index += 1
            trace(f"Found {len(favorites)} Sonos favorites")
        except Exception as e:
            error(f"Error fetching Sonos favorites: {e}")
        return res
