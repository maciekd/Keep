from ..PromptSectionGenerator import PromptSectionGenerator
from common.logging import info
from ...integration.integrations.MusicIntegration import MusicManager


class MusicZonesSectionGenerator(PromptSectionGenerator):
    """Keeps the Sonos zones status section of the prompt fresh by polling every 3 seconds.

    Polls every 3 seconds. The base loop skips downstream work when the fetched
    status is byte-identical to the current content, and the status string carries
    no per-fetch field (no elapsed time / position), so a real change — including
    one made externally in the Sonos app — is reflected in the prompt within 3s
    while a steady state costs only a cheap UPnP round-trip.

    It reads the integration the IntegrationMonitor built and never builds one: while
    Sonos & Spotify is disabled (or still starting) the section is empty and no speaker
    is looked for.
    """

    UPDATE_INTERVAL = 3

    def __init__(self, prompt_manager):
        super().__init__(prompt_manager, "MusicZones")
        self._running = None   # whether the last poll found the integration; None before the first

    def _fetch_content(self) -> str:
        manager = MusicManager.peek_instance()
        running = manager is not None
        if running != self._running:
            self._running = running
            if running:
                info("MusicZones: Sonos & Spotify is running, polling its zones")
            else:
                info("MusicZones: Sonos & Spotify is not running (disabled or still starting) — "
                     "section empty, no speaker is looked for")
        return manager.get_all_zones_status() if running else ""
