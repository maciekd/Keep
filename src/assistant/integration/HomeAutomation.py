"""The home automation contract: what the assistant asks of whatever runs the house — Indigo,
Home Assistant, or several of them at once. The contract speaks names, the ones the LLM reads
out of the snapshot and hands back to the commands; how a name maps to a REST path or an entity
id is the backend's business. A backend is always an Integration too (name, config keys,
health), and a Feed when it has an event stream to offer. `active()` is what the commands and
the prompt section run against: the one ready backend, a CompositeHome over several, or the
NotConfiguredHome stand-in.

The bracket convention is the assistant's, not the backend's: parens end a command's argument
list in CommandParser, so the prompt shows every "(Room)" as "[Room]" (`to_display`) and the
commands turn it back before calling the backend (`from_display`)."""
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from common.ConfigManager import config
from common.logging import info, trace

# The four snapshot lists, in the order the prompt section renders them.
SECTIONS = ("lights", "sensors", "variables", "actions")
_TRUE = {"true", "on", "1", "yes"}
_FALSE = {"false", "off", "0", "no"}
_LONG_FLOAT = re.compile(r'-?\d+\.\d{4,}')


@dataclass(frozen=True)
class Entity:
    name: str                       # what the LLM sees and hands back: the backend's raw name
    value: str                      # "on", "off", "80", "22.7 °C", "True", an action's description
    area: Optional[str] = None      # rendered as " [Area]" after the name when set


@dataclass
class HomeSnapshot:
    lights: list[Entity] = field(default_factory=list)
    sensors: list[Entity] = field(default_factory=list)
    variables: list[Entity] = field(default_factory=list)
    actions: list[Entity] = field(default_factory=list)
    taken_at: float = 0.0

    def flag(self, variable: str) -> Optional[bool]:
        """A variable's truth: True for true/on/1, False for false/off/0, None when the
        variable is absent or its value is neither."""
        for entity in self.variables:
            if entity.name == variable:
                value = entity.value.strip().lower()
                if value in _TRUE:
                    return True
                if value in _FALSE:
                    return False
                return None
        return None

    def is_empty(self) -> bool:
        return not any(getattr(self, section) for section in SECTIONS)


class HomeAutomation(ABC):
    """What a home automation backend implements. Every command returns "" on success, else
    the text the LLM should read (the backend's error, "Request timed out", ...)."""

    state_poll_interval: float = 30                     # seconds between snapshots for the prompt section
    section_filters: dict[str, tuple[str, ...]] = {}    # per-section substrings the backend always drops

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def config_section(self) -> str:
        """The top-level config.yaml block ("indigo"): `<section>.prompt_filters` lives there."""
        ...

    @abstractmethod
    def snapshot(self) -> HomeSnapshot:
        """The house right now. May raise: the section generator keeps the last good one."""
        ...

    @abstractmethod
    def turn_on(self, name: str) -> str: ...

    @abstractmethod
    def turn_off(self, name: str) -> str: ...

    @abstractmethod
    def dim(self, name: str, level: int) -> str: ...

    @abstractmethod
    def execute(self, name: str) -> str: ...

    def stop(self, name: str) -> str:
        """Halts a moving cover (a blind, a zipscreen, an awning) where it is. A backend
        without covers keeps this answer."""
        return f"{self.name} cannot stop anything — nothing it runs is a cover."

    def error_logs(self) -> str:
        return f"{self.name} has no error log."

    def prune(self, snapshot: HomeSnapshot) -> HomeSnapshot:
        """The snapshot minus what the prompt should not see: the lines matching the
        configured `<config_section>.prompt_filters` on every section, the backend's own
        `section_filters` per section. The commands still reach a pruned entity by name."""
        filters = config.get(f"{self.config_section}.prompt_filters", []) or []
        return prune(snapshot, filters, self.section_filters)

    @staticmethod
    def to_display(text: str) -> str:
        return text.replace("(", "[").replace(")", "]")

    @staticmethod
    def from_display(name: str) -> str:
        return name.replace("[", "(").replace("]", ")")


class NotConfiguredHome(HomeAutomation):
    """What the house commands and the prompt section run against when no backend is enabled
    and configured: every operation answers with the setup hint, so the assistant offers to set
    one up instead of failing on a missing singleton (compare email's NotConfiguredEmail)."""

    def __init__(self, hint: str, starting: bool = False):
        self.hint = hint
        self.starting = starting
        if starting:
            self.state_poll_interval = STARTUP_POLL_INTERVAL

    @property
    def name(self) -> str:
        return "Home automation"

    @property
    def config_section(self) -> str:
        return "home"

    def snapshot(self) -> HomeSnapshot:
        return HomeSnapshot()

    def turn_on(self, name: str) -> str:
        return self.hint

    def turn_off(self, name: str) -> str:
        return self.hint

    def dim(self, name: str, level: int) -> str:
        return self.hint

    def execute(self, name: str) -> str:
        return self.hint

    def stop(self, name: str) -> str:
        return self.hint

    def error_logs(self) -> str:
        return self.hint


STARTING_HINT = "Home automation is still starting up — try again in a few seconds."
STARTUP_POLL_INTERVAL = 2       # how often the prompt section looks again while no backend is registered yet
_last_state: Optional[str] = None
_composite = None               # the CompositeHome over the current set of ready backends, kept for its name index


def _log_state(line: str, log=None) -> None:
    """Log the resolution once per change: the section resolves every 2-30 s, so an unchanged
    "no backend ready" would otherwise flood the log."""
    global _last_state
    if line != _last_state:
        _last_state = line
        (log or trace)(line)


def active() -> HomeAutomation:
    """The backend the house commands run against: the registered, enabled and configured
    HomeAutomation instance. None → a NotConfiguredHome whose hint names what is missing (the
    enable line for a disabled backend, the keys for an unconfigured one, the constructor's
    error for an unavailable one). No backend registered in any form → the launcher has not
    reached the feeds yet (the prompt sections start before them), so the hint says "still
    starting" and `starting` asks the section to look again soon. Several → one CompositeHome
    over them, the same instance as long as the set of ready backends is the same (it keeps
    the index that routes a name to its backend). Resolved per call, so a restarted
    integration is picked up."""
    from assistant.integration.IntegrationMonitor import IntegrationMonitor, DisabledIntegration, UnavailableIntegration
    monitor = IntegrationMonitor.get_instance()
    ready, hints = [], []
    for integration in monitor.of_type(HomeAutomation):
        if integration.missing_config_keys():
            hints.append(f"{integration.name} is enabled but not configured: set "
                         f"{', '.join(integration.missing_config_keys())} (config_help / config_request).")
        else:
            ready.append(integration)
    if len(ready) > 1:
        _log_state(f"Home automation: {len(ready)} backends enabled ({', '.join(i.name for i in ready)}) — one house", info)
        return _composite_over(ready)
    if ready:
        _log_state(f"Home automation: using {ready[0].name}")
        return ready[0]
    for placeholder in monitor.of_type(DisabledIntegration):
        if issubclass(placeholder.cls, HomeAutomation):
            hints.append(f'{placeholder.name} is disabled: enable_integration("{placeholder.name}").')
    for placeholder in monitor.of_type(UnavailableIntegration):
        if issubclass(placeholder.cls, HomeAutomation):
            hints.append(f"{placeholder.name} could not be started: {placeholder.last_error} (restart_integration once fixed).")
    if not hints:
        _log_state("Home automation: no backend registered yet — the integrations are still starting")
        return NotConfiguredHome(STARTING_HINT, starting=True)
    hint = " ".join(hints)
    _log_state(f"Home automation: no backend ready — {hint}")
    return NotConfiguredHome(hint)


def _composite_over(ready: list) -> HomeAutomation:
    global _composite
    from assistant.integration.CompositeHome import CompositeHome
    if _composite is None or _composite.backends != ready:
        _composite = CompositeHome(ready)
    return _composite


def line_of(entity: Entity) -> str:
    """An entity as its prompt line: `"name [Area]": value` in display form (parens as
    brackets, floats with 4+ decimals rounded to 2). The filters match against this text."""
    name = f"{entity.name} [{entity.area}]" if entity.area else entity.name
    line = HomeAutomation.to_display(f'"{name}": {entity.value}').rstrip()
    return _LONG_FLOAT.sub(lambda m: f"{float(m.group()):.2f}", line)


def prune(snapshot: HomeSnapshot, prompt_filters: list[str], section_filters: dict[str, tuple[str, ...]]) -> HomeSnapshot:
    """The snapshot minus the entities whose line matches a filter, case-insensitively — the
    `prompt_filters` on every section, the `section_filters` each on its own."""
    pruned = HomeSnapshot(taken_at=snapshot.taken_at)
    for section in SECTIONS:
        drop = [f.lower() for f in list(prompt_filters) + list(section_filters.get(section, ()))]
        kept = [e for e in getattr(snapshot, section) if not any(f in line_of(e).lower() for f in drop)]
        setattr(pruned, section, kept)
    return pruned


def render(snapshot: HomeSnapshot) -> str:
    """The snapshot as the prompt's HOME AUTOMATION DATA body: one `## SECTION` block per
    non-empty list, one `line_of` per entity. Prune first — this shows everything it is given."""
    parts = []
    for section in SECTIONS:
        entities = getattr(snapshot, section)
        if entities:
            parts.append(f"## {section.upper()}\n" + "".join(line_of(e) + "\n" for e in entities) + "\n")
    return "".join(parts)
