"""The music contract: what the assistant asks of a speaker system — Sonos, HEOS — and the `Speakers` composite the music commands and prompt sections run against.

A system is always an Integration too (name, config keys, health). It names its zones as the LLM
sees them (after its own room mapping) and acts on one by that name. What plays is an id: a
Spotify one any system plays through the shared `SpotifyAccount` (a search hit's `AA`…, a library
playlist's `S-n`), or one of the system's own lists, under prefixes it declares in `ID_PREFIXES`
(Sonos: `F-` favorites, `P-` playlists; HEOS: `H-` favorites, `Q-` a room's quick selects) so that
two systems' lists never answer to the same id.

`Speakers` gathers the zones of every ready system per call and resolves a room to its system;
two systems with a zone of the same name is a configuration error the LLM hears about, never a
guess. Every action returns a line for the LLM, "" when the action is its own answer, or raises
`MusicError` with what the LLM should read."""
from abc import ABC, abstractmethod
from typing import Callable, Optional

from common.logging import error, info, trace


# How often a system reloads its own lists and the shared Spotify library.
LIBRARY_REFRESH_SECONDS = 30 * 60


class MusicError(Exception):
    """A music command that failed; the message is what the LLM should read."""


class MusicSystem(ABC):
    """What a speaker-system integration implements."""

    # The prefixes of the ids of the system's own lists ("F-" …); a Spotify id is any system's.
    ID_PREFIXES: tuple[str, ...] = ()

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def zones(self) -> list[str]:
        """The zones music can play in, by the names the LLM uses (hidden ones left out)."""

    @abstractmethod
    def zone_status(self) -> list[str]:
        """One line per zone: its name and what it plays now ("Kitchen (Song by Artist - Playing)")."""

    @abstractmethod
    def library_text(self) -> str:
        """The system's own lists for the MUSIC section, one entry per line with its id; "" when none."""

    def has_library(self) -> bool:
        return bool(self.library_text())

    def pushes_changes(self) -> bool:
        """True while the system reports every change of its zones itself (`zones_changed()`), so
        their status need not be polled. A system that cannot says False, the default."""
        return False

    def refresh_library(self) -> None:
        """Reload the system's own lists now (and the shared Spotify library); a no-op by default."""

    @abstractmethod
    def play(self, zone: str) -> str: ...

    @abstractmethod
    def pause(self, zone: str) -> str: ...

    @abstractmethod
    def next_track(self, zone: str) -> str: ...

    @abstractmethod
    def adjust_volume(self, zone: str, step: int) -> str: ...

    @abstractmethod
    def play_id(self, zone: str, music_id: str) -> str:
        """Start what the id names in the zone: a Spotify id or one of the system's own."""

    def owns_id(self, music_id: str) -> bool:
        return any(music_id.startswith(prefix) for prefix in self.ID_PREFIXES)


class Speakers:
    """The zones of every ready system as one house, and the lookup from a room back to its system."""

    def __init__(self, systems: list[MusicSystem]):
        self.systems = list(systems)
        self._zones: Optional[list[tuple[MusicSystem, str]]] = None

    def zones(self) -> list[tuple[MusicSystem, str]]:
        if self._zones is None:
            self._zones = []
            for system in self.systems:
                try:
                    self._zones.extend((system, zone) for zone in system.zones())
                except Exception as e:
                    info(f"Music: {system.name} could not list its zones: {e}")
        return self._zones

    def resolve(self, room: str) -> tuple[MusicSystem, str]:
        """(system, the zone's own name) for the room the LLM named, case aside."""
        zones = self.zones()
        wanted = (room or "").strip()
        if not zones:
            raise MusicError(f"no zone '{wanted}': no speaker answers on the network, so nothing can play")
        matches = [(s, z) for s, z in zones if z.lower() == wanted.lower()]
        if len(matches) > 1:
            owners = ", ".join(s.name for s, _ in matches)
            raise MusicError(f"'{wanted}' is a zone of more than one system ({owners}); rename one of them")
        if not matches:
            known = ", ".join(sorted(z for _, z in zones))
            raise MusicError(f"no zone '{wanted}'; the zones are: {known}")
        return matches[0]

    def owner_of_id(self, music_id: str) -> Optional[MusicSystem]:
        """The system whose own list the id is from; None for a Spotify id (anyone's)."""
        return next((s for s in self.systems if s.owns_id(music_id)), None)

    def zone_status(self) -> str:
        lines = []
        for system in self.systems:
            try:
                lines.extend(system.zone_status())
            except Exception as e:
                info(f"Music: {system.name} could not report its zones: {e}")
        return "\n".join(lines)


# Called (no arguments) whenever a system reports that its zones changed — what plays, a group.
_zone_listeners: list[Callable[[], None]] = []


def add_zone_listener(callback: Callable[[], None]):
    if callback not in _zone_listeners:
        _zone_listeners.append(callback)


def remove_zone_listener(callback: Callable[[], None]):
    if callback in _zone_listeners:
        _zone_listeners.remove(callback)


def zones_changed(source: str):
    """A system's zones changed: everyone following them (the zones section) looks again now."""
    listeners = list(_zone_listeners)
    trace(f"Music: {source} reports a zone change" + ("" if listeners else ", nobody listening"))
    for callback in listeners:
        try:
            callback()
        except Exception as e:
            error(f"Music: zone listener {callback} failed: {e}")


_last_active_report: list[str] = []


def active() -> list[MusicSystem]:
    """The systems whose config keys are set, in registration order. Resolved per call, so a
    restarted or newly configured integration is what the next command uses."""
    from assistant.integration.IntegrationMonitor import IntegrationMonitor
    global _last_active_report
    ready, skipped = [], []
    for integration in IntegrationMonitor.get_instance().of_type(MusicSystem):
        missing = integration.missing_config_keys()
        if missing:
            skipped.append(f"{integration.name} is enabled but not configured ({', '.join(missing)}), skipped")
            continue
        ready.append(integration)
    report = [*skipped, f"{len(ready)} system(s) ready" + (f" ({', '.join(s.name for s in ready)})" if ready else "")]
    if report != _last_active_report:
        _last_active_report = report
        for line in report:
            trace(f"Music: {line}")
    return ready


def speakers() -> Speakers:
    return Speakers(active())


STARTING_HINT = "The speakers are still starting up — try again in a few seconds."


def no_system_hint() -> str:
    """Why `active()` came back empty, for the LLM: the keys of an unconfigured system, the
    error of one whose constructor failed, the enable line for a disabled one, or "still
    starting" while none is registered."""
    from assistant.integration.IntegrationMonitor import IntegrationMonitor, DisabledIntegration, UnavailableIntegration
    monitor = IntegrationMonitor.get_instance()
    hints = []
    for integration in monitor.of_type(MusicSystem):
        hints.append(f"{integration.name} is enabled but not configured: set "
                     f"{', '.join(integration.missing_config_keys())} (config_help / config_request).")
    for placeholder in monitor.of_type(DisabledIntegration):
        if placeholder.can_enable and issubclass(placeholder.cls, MusicSystem):
            hints.append(f'{placeholder.name} is disabled: enable_integration("{placeholder.name}").')
    for placeholder in monitor.of_type(UnavailableIntegration):
        if issubclass(placeholder.cls, MusicSystem):
            hints.append(f"{placeholder.name} could not be started: {placeholder.last_error} (restart_integration once fixed).")
    if not hints:
        trace("Music: no system registered yet — the integrations are still starting")
        return STARTING_HINT
    hint = " ".join(hints)
    trace(f"Music: no system ready — {hint}")
    return hint
