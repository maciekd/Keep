"""The car contract: what the assistant asks of whatever brand's cloud the family's cars talk
to — Tesla through Tessie today, Volvo, Polestar or BMW tomorrow. A provider is always an
Integration too (name, config keys, health); the `cars` toolkit's commands and the CURRENT CAR
INFO prompt section run against the `Fleet` over every enabled and configured provider and
never name a folder. The LLM speaks `vehicle_id`s — the `name` of a car's config entry — and
the fleet resolves one to its provider and the provider's own handle for the car (a VIN, an
API id), so brands merge into one list of cars in the prompt.

Every action returns a line for the LLM ("Trunk opened for Family Model Y.") or raises a
VehicleError; the command decides whether the line is worth saying. A brand without an action
leaves the default in place, which raises Unsupported — the LLM hears that the car cannot do it
instead of a stack trace."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from common.logging import info, trace


class VehicleError(Exception):
    """A car command or state read that failed; the message is what the LLM should read."""


class Unsupported(VehicleError):
    """The brand's cloud offers no such action for the car."""


@dataclass(frozen=True)
class VehicleSpec:
    """One car as configured: `id` is the vehicle_id the LLM uses (the entry's `name`),
    `handle` the provider's own identifier for it (None when the entry lacks one — the
    provider then answers that the car is not fully configured), `aliases` the other strings
    that name it (a display name, the VIN)."""
    id: str
    label: str
    handle: Optional[str] = None
    aliases: tuple[str, ...] = ()
    is_default: bool = False

    def matches(self, target: str) -> bool:
        wanted = (target or "").strip().lower()
        return bool(wanted) and wanted in {c.lower() for c in (self.id, self.label, *self.aliases) if c}


@dataclass
class VehicleState:
    """What a brand reports about a car; every field a brand cannot supply stays None."""
    connectivity: Optional[str] = None      # "online" / "asleep" / "offline" as the brand says it
    in_gear: bool = False                   # moving or in gear (stopped at a red light counts)
    speed_kmh: Optional[float] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    battery_percent: Optional[int] = None
    fuel_percent: Optional[int] = None
    charging: bool = False                  # plugged in and taking charge, when the brand can tell
    range_km: Optional[float] = None
    cabin_temp_c: Optional[float] = None
    destination: Optional[str] = None       # the active route's destination, with its ETA
    minutes_to_arrival: Optional[float] = None
    data_time: Optional[float] = None       # epoch seconds of the snapshot when the brand serves a cached one

    @property
    def is_driving(self) -> bool:
        return bool(self.speed_kmh) and self.speed_kmh > 0


class VehicleProvider(ABC):
    """What a car integration implements."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def vehicles(self) -> list[VehicleSpec]:
        """The cars configured for the provider, in config order."""
        ...

    @abstractmethod
    def vehicle_state(self, spec: VehicleSpec) -> VehicleState:
        """The car right now. May raise: the prompt section shows the error on the car's line."""
        ...

    def max_defrost(self, spec: VehicleSpec) -> str:
        raise Unsupported(f"{self.name} cannot start max defrost on {spec.label}.")

    def climate_on(self, spec: VehicleSpec) -> str:
        raise Unsupported(f"{self.name} cannot start the climate in {spec.label}.")

    def climate_off(self, spec: VehicleSpec) -> str:
        raise Unsupported(f"{self.name} cannot stop the climate in {spec.label}.")

    def open_trunk(self, spec: VehicleSpec) -> str:
        raise Unsupported(f"{self.name} cannot open the trunk of {spec.label}.")

    def navigate_to(self, spec: VehicleSpec, destination: str) -> str:
        raise Unsupported(f"{self.name} cannot send a destination to {spec.label}.")


class Fleet:
    """The cars of every ready provider as one list, and the lookup from a vehicle_id back to
    (provider, car). A vehicle_id the LLM hands in matches a car's id, label or alias, case
    aside; an empty one means the default car — the first flagged `default`, else the first
    car of the first provider. Two cars answering to the same id is a configuration error the
    LLM hears about, never a guess."""

    def __init__(self, providers: list[VehicleProvider]):
        self.providers = list(providers)
        self._cars: Optional[list[tuple[VehicleProvider, VehicleSpec]]] = None

    def cars(self) -> list[tuple[VehicleProvider, VehicleSpec]]:
        if self._cars is None:
            self._cars = []
            for provider in self.providers:
                try:
                    self._cars.extend((provider, spec) for spec in provider.vehicles())
                except Exception as e:
                    info(f"Cars: {provider.name} could not list its cars: {e}")
        return self._cars

    def resolve(self, vehicle_id: Optional[str]) -> tuple[VehicleProvider, VehicleSpec]:
        cars = self.cars()
        if not cars:
            raise VehicleError("No car is configured.")
        wanted = (vehicle_id or "").strip()
        if not wanted:
            default = next(((p, s) for p, s in cars if s.is_default), cars[0])
            trace(f"Cars: no vehicle_id given — the default car is {default[1].label} ({default[0].name})")
            return default
        matches = [(p, s) for p, s in cars if s.matches(wanted)]
        if len(matches) > 1:
            owners = ", ".join(f"{s.label} ({p.name})" for p, s in matches)
            raise VehicleError(f"'{wanted}' names more than one car: {owners}. Use the vehicle_id.")
        if not matches:
            known = ", ".join(s.id for _, s in cars)
            raise VehicleError(f"No car matches '{wanted}'. The cars: {known}.")
        return matches[0]


def active() -> list[VehicleProvider]:
    """The providers whose config keys are set, in registration order. Resolved per call, so a
    restarted or newly configured integration is what the next command uses."""
    from assistant.integration.IntegrationMonitor import IntegrationMonitor
    ready = []
    for integration in IntegrationMonitor.get_instance().of_type(VehicleProvider):
        missing = integration.missing_config_keys()
        if missing:
            trace(f"Cars: {integration.name} is enabled but not configured ({', '.join(missing)}), skipped")
            continue
        ready.append(integration)
    trace(f"Cars: {len(ready)} provider(s) ready" + (f" ({', '.join(p.name for p in ready)})" if ready else ""))
    return ready


def fleet() -> Fleet:
    return Fleet(active())


def no_provider_hint() -> str:
    """Why `active()` came back empty, for the LLM: the keys of an unconfigured provider, the
    error of one whose constructor failed, the enable line for a disabled one, or "still
    starting" while none is registered."""
    from assistant.integration.IntegrationMonitor import IntegrationMonitor, DisabledIntegration, UnavailableIntegration
    monitor = IntegrationMonitor.get_instance()
    hints = []
    for integration in monitor.of_type(VehicleProvider):
        hints.append(f"{integration.name} is enabled but not configured: set "
                     f"{', '.join(integration.missing_config_keys())} (config_help / config_request).")
    for placeholder in monitor.of_type(DisabledIntegration):
        if placeholder.can_enable and issubclass(placeholder.cls, VehicleProvider):
            hints.append(f'{placeholder.name} is disabled: enable_integration("{placeholder.name}").')
    for placeholder in monitor.of_type(UnavailableIntegration):
        if issubclass(placeholder.cls, VehicleProvider):
            hints.append(f"{placeholder.name} could not be started: {placeholder.last_error} (restart_integration once fixed).")
    if not hints:
        trace("Cars: no provider registered yet — the integrations are still starting")
        return STARTING_HINT
    hint = " ".join(hints)
    trace(f"Cars: no provider ready — {hint}")
    return hint


STARTING_HINT = "The cars are still starting up — try again in a few seconds."
