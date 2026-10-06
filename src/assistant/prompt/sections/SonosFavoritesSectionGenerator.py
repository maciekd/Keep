import time
from ..PromptSectionGenerator import PromptSectionGenerator
from common.logging import trace, info, error
from ...integration.integrations.MusicIntegration import MusicManager


class SonosFavoritesSectionGenerator(PromptSectionGenerator):
    """Refreshes the Sonos favorites and playlists section every 30 minutes once Sonos has loaded.

    It reads the integration the IntegrationMonitor built and never builds one: while
    Sonos & Spotify is disabled the section is empty and no speaker is looked for."""

    INTEGRATION_POLL_SECONDS = 1

    def __init__(self, prompt_manager):
        super().__init__(prompt_manager, "SonosFavorites")

    @property
    def playback_manager(self):
        """The running integration, or None while it is disabled (or not built yet)."""
        return MusicManager.peek_instance()

    def _playlists(self) -> list:
        manager = self.playback_manager
        return manager.get_playlists() if manager is not None else []

    def _favorites(self) -> list:
        manager = self.playback_manager
        return manager.get_favorites() if manager is not None else []

    def _await_integration(self) -> bool:
        """Wait until Sonos & Spotify is running; False when stopped first. The integrations
        are built after the prompt sections start, so an enabled one is a moment away."""
        if self.playback_manager is not None:
            return True
        info("SonosFavorites: Sonos & Spotify is not running (disabled or still starting) — "
             "section stays empty until it is")
        while not self._stop_event.is_set() and self.playback_manager is None:
            self._wait(self.INTEGRATION_POLL_SECONDS)
        if self._stop_event.is_set():
            return False
        info("SonosFavorites: Sonos & Spotify is running now")
        return True

    def _start_monitoring(self):
        """Follow the integration: wait for it, load once its data is in, refresh every 30
        minutes, and start over when it is dropped (disable_integration)."""
        while not self._stop_event.is_set():
            if not self._await_integration():
                return
            self._await_first_data()
            self._refresh_until_dropped()

    def _await_first_data(self):
        """Poll until playlists/favorites are loaded (with timeout)."""
        max_wait = 60  # seconds
        start_time = time.time()

        while not self._stop_event.is_set():
            if self._playlists() or self._favorites():
                info("Sonos data loaded, updating content")
                self._update_content()
                break

            if time.time() - start_time > max_wait:
                info(f"No Sonos favorites or playlists after {max_wait}s (no speakers found, or Spotify not "
                     f"configured) — section stays empty, checked again every 30 minutes")
                break

            self._wait(0.1)

    def _refresh_until_dropped(self):
        """The normal 30-minute refresh cycle; returns once the integration is gone."""
        while not self._stop_event.is_set():
            self._wait(30 * 60)  # 30 minutes
            if self._stop_event.is_set():
                break
            if self.playback_manager is None:
                info("SonosFavorites: Sonos & Spotify is no longer running, emptying the section")
                self._update_content()
                return
            if self._playlists() or self._favorites():
                self._update_content()
            else:
                trace("Sonos: still no favorites or playlists, section left empty")

    def _fetch_content(self) -> str:
        """Fetch Sonos favorites and playlists."""
        info("Fetching Sonos data...")
        res = self._fetch_sonos_favorites()
        res = self._fetch_sonos_playlists(res)
        trace(f"Successfully fetched {len(res.splitlines())} lines of Sonos favorites and playlists.")
        return res.strip()

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
