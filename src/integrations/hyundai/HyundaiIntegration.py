"""The Hyundai, Kia and Genesis cars through the Bluelink / Kia Connect app API, a VehicleProvider.

Nothing is registered with the manufacturer: the `hyundai_kia_connect_api` library signs in
the way the phone app does, with the owner's username and password and the app's own client
ids, against one of eight regional backends. The library keeps the account's `Token`
(access and refresh tokens, the device id the backend knows the "phone" by); it is persisted
under `cache/hyundai/` so a restart does not sign in again — in Canada and for Kia in the US a
fresh sign-in asks for a one-time code sent by mail or SMS, which the console's Hyundai tab
takes. Every state read is the manufacturer's *cached* state (cheap, does not wake the car);
nothing here ever force-refreshes on a timer, since that drains the 12 V battery and is
throttled per account."""
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import requests
from hyundai_kia_connect_api import ClimateRequestOptions, VehicleManager
from hyundai_kia_connect_api.ApiImpl import POICoord, POIInfo
from hyundai_kia_connect_api.const import ENGINE_TYPES, OTP_NOTIFY_TYPE
from hyundai_kia_connect_api.exceptions import (
    AuthenticationError,
    AuthenticationOTPRequired,
    HyundaiKiaException,
    RateLimitingError,
    UnsupportedControlError,
)
from hyundai_kia_connect_api.Token import Token

from assistant.integration.Accounts import slug
from assistant.integration.Integration import Integration
from assistant.integration.VehicleProvider import Unsupported, VehicleError, VehicleProvider, VehicleSpec, VehicleState
from common.ConfigManager import config
from common.Paths import CACHE_DIR
from common.logging import error, info, trace

# The words config.yaml takes, mapped to the library's integer constants (hyundai_kia_connect_api/const.py:
# REGIONS = {1: Europe, 2: Canada, 3: USA, 4: China, 5: Australia, 6: India, 7: New Zealand, 8: Brazil,
# 9: Europe (CCI/GSPA)}, BRANDS = {1: Kia, 2: Hyundai, 3: Genesis}).
REGIONS = {"europe": 1, "canada": 2, "usa": 3, "china": 4, "australia": 5, "india": 6,
           "new_zealand": 7, "brazil": 8, "europe_cci": 9}
BRANDS = {"kia": 1, "hyundai": 2, "genesis": 3}

TOKEN_DIR = CACHE_DIR / "hyundai"
STATE_TTL = 120                     # seconds a cached-state pull serves every vehicle_state() call
DEFAULT_CLIMATE_C = 21.0
DEFAULT_CLIMATE_MINUTES = 10
MILES_TO_KM = 1.609344
FAHRENHEIT_REGIONS = {"usa"}        # the backends whose climate set point is in °F (the rest take °C)
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
GEOCODE_TIMEOUT = 15
OTP_HINT = "a one-time code is needed — send and enter it in the console's Hyundai tab"


class HyundaiError(VehicleError):
    """A call the Bluelink / Kia Connect backend refused; the message is what the LLM reads."""


class HyundaiConfigurationError(HyundaiError):
    """The `hyundai` section is missing something or names a region or brand the library lacks."""


class HyundaiLoginError(HyundaiError):
    """The account is not signed in: wrong credentials, or a one-time code still to be entered."""


def region_code(word: str) -> int:
    key = (word or "").strip().lower().replace(" ", "_").replace("-", "_")
    if key not in REGIONS:
        raise HyundaiConfigurationError(f"hyundai.region '{word}' is not one of {', '.join(REGIONS)}")
    return REGIONS[key]


def brand_code(word: str) -> int:
    key = (word or "").strip().lower()
    if key not in BRANDS:
        raise HyundaiConfigurationError(f"hyundai.brand '{word}' is not one of {', '.join(BRANDS)}")
    return BRANDS[key]


def token_path(username: str) -> Path:
    return TOKEN_DIR / f"token-{slug(username)}.json"


def to_km(value: Optional[float], unit: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    return round(value * MILES_TO_KM, 1) if unit == "mi" else float(value)


def to_fahrenheit(celsius: float) -> int:
    return round(celsius * 9 / 5 + 32)


def engine_kind(vehicle) -> str:
    """The library's engine type as its word ("EV", "PHEV", "HEV", "ICE"): the backends set the
    `ENGINE_TYPES` member, whose str() is "ENGINE_TYPES.EV" — so the value, not the member."""
    engine = getattr(vehicle, "engine_type", None)
    if isinstance(engine, ENGINE_TYPES):
        return engine.value
    return str(engine or "").upper()


def parse_coordinates(text: str) -> Optional[tuple[float, float]]:
    """("59.33,18.07" → (59.33, 18.07)); None for anything that is not a latitude,longitude pair."""
    parts = [p.strip() for p in (text or "").split(",")]
    if len(parts) != 2:
        return None
    try:
        lat, lon = float(parts[0]), float(parts[1])
    except ValueError:
        return None
    return (lat, lon) if -90 <= lat <= 90 and -180 <= lon <= 180 else None


class HyundaiIntegration(Integration, VehicleProvider):
    """Hyundai, Kia and Genesis cars via the Bluelink / Kia Connect app.

    Reads where the cars are, their battery or fuel and range, and sends them commands: climate
    on at a target temperature, max defrost, climate off, and a navigation destination (Europe,
    Australia, India, China). The data is what the manufacturer's servers last heard from the
    car; nothing wakes it.

    A VehicleProvider: the `cars` toolkit and the CURRENT CAR INFO section reach it through the
    fleet beside the other brands; a car's handle is its VIN, matched to the library's own
    vehicle id after the sign-in. The `VehicleManager` is built on first use (so a mistyped
    password costs one failed login, not a crashed start) behind `_manager()`; `_ensure_login`
    runs the library's token check before every call and turns a one-time-code demand into a
    pending `OTPRequest` the tab finishes (`send_otp`, `verify_otp`); while one is pending no
    call signs in again (that would replace the request the code was sent for) — only the
    tab's Ctrl-R (`refresh_now`) starts over. The library raises
    `NotImplementedError` for a call the region's backend lacks and `UnsupportedControlError`
    for one the car refuses — both become `Unsupported`, so the LLM hears that instead of a
    stack trace. The commands come back with a tracking id; the state cache is dropped so the
    next prompt refresh picks the result up, and no force refresh is sent."""

    def __init__(
        self,
        config_override: Optional[Dict[str, Any]] = None,
        manager_factory: Optional[Callable[..., VehicleManager]] = None,
        session: Optional[requests.Session] = None,
    ):
        super().__init__()
        self._config = config_override if config_override is not None else self._load_config()
        self._manager_factory = manager_factory or VehicleManager
        self._session = session or requests.Session()
        self._lock = threading.RLock()
        self._manager_instance: Optional[VehicleManager] = None
        self._otp_request = None                 # the library's OTPRequest while a code is awaited
        self._login_error: Optional[str] = None  # the last refused sign-in, for the tab
        self._cached_at: float = 0.0             # when the manufacturer's cache was last pulled
        self._persisted_token: Optional[dict] = None
        self._region = region_code(self._config.get("region"))
        self._brand = brand_code(self._config.get("brand"))
        self._validate_config()

    @property
    def name(self) -> str:
        return "Hyundai"

    @property
    def config_keys(self) -> List[str]:
        return ["hyundai.username", "hyundai.password", "hyundai.region", "hyundai.brand"]

    # --- configuration -----------------------------------------------------------------------

    @staticmethod
    def _load_config() -> Dict[str, Any]:
        section = config.get("hyundai")
        if not isinstance(section, dict):
            raise HyundaiConfigurationError("Missing 'hyundai' section in configuration.")
        return section

    def _validate_config(self):
        for key in ("username", "password"):
            if not self._config.get(key):
                raise HyundaiConfigurationError(f"Missing Hyundai configuration value: {key}")
        entries = self._config.get("vehicles")
        if entries is not None and not isinstance(entries, list):
            raise HyundaiConfigurationError("hyundai.vehicles must be a list of cars.")
        if entries == []:
            raise HyundaiConfigurationError("Hyundai vehicle list is empty — list the cars or leave the key out.")
        if not entries:
            trace("Hyundai: no hyundai.vehicles list — every car of the account will be offered under its own name")

    @classmethod
    def wipe_targets(cls) -> list[tuple[Path, str]]:
        return [(path, "Hyundai / Kia Connect sign-in token") for path in sorted(TOKEN_DIR.glob("token-*.json"))]

    # --- the library's manager and its token -------------------------------------------------

    def _manager(self) -> VehicleManager:
        with self._lock:
            if self._manager_instance is None:
                token = self._load_token()
                self._manager_instance = self._manager_factory(
                    region=self._region, brand=self._brand,
                    username=self._config["username"], password=self._config["password"],
                    pin=str(self._config.get("pin") or ""), token=token,
                )
                info(f"Hyundai: manager built for region {self._config.get('region')}, brand {self._config.get('brand')}"
                     + (" with the saved token" if token else ", no saved token — a sign-in is due"))
            return self._manager_instance

    def _load_token(self) -> Optional[Token]:
        path = token_path(self._config["username"])
        if not path.exists():
            trace(f"Hyundai: no saved token at {path}")
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self._persisted_token = data
            return Token.from_dict(data)
        except (OSError, ValueError, TypeError) as e:
            error(f"Hyundai: the saved token {path} could not be read ({e}) — signing in afresh")
            return None

    def _persist_token(self, manager: VehicleManager):
        token = getattr(manager, "token", None)
        if token is None:
            return
        data = token.to_dict()
        if data == self._persisted_token:
            return
        path = token_path(self._config["username"])
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data), encoding="utf-8")
            os.chmod(path, 0o600)
            self._persisted_token = data
            trace(f"Hyundai: token saved to {path}")
        except OSError as e:
            error(f"Hyundai: could not save the token to {path}: {e}")

    def _ensure_login(self) -> VehicleManager:
        """The manager with a valid token: the library refreshes or re-logs in as needed. A
        one-time-code demand is kept as the pending request and raised as a login error, and
        while one is pending nothing signs in again (the request the code was sent for must
        stay the one it is verified against)."""
        manager = self._manager()
        with self._lock:
            if self._otp_request is not None:
                trace("Hyundai: a one-time code is pending — not signing in again until it is entered or Ctrl-R")
                raise HyundaiLoginError(f"Hyundai is not signed in: {OTP_HINT}.")
            try:
                refreshed = self._refresh_or_login(manager)
            except AuthenticationOTPRequired as e:
                self._otp_request = getattr(manager, "otp_request", None)
                self._login_error = None
                info(f"Hyundai: the sign-in needs a one-time code ({e})")
                raise HyundaiLoginError(f"Hyundai is not signed in: {OTP_HINT}.") from e
            except AuthenticationError as e:
                self._login_error = str(e)
                info(f"Hyundai: sign-in refused: {e}")
                raise HyundaiLoginError(f"Hyundai sign-in refused: {e}") from e
            except RateLimitingError as e:
                info(f"Hyundai: rate limited while signing in: {e}")
                raise HyundaiError(f"Hyundai's servers are rate limiting the account: {e}") from e
            except HyundaiKiaException as e:
                info(f"Hyundai: sign-in failed: {e}")
                raise HyundaiError(f"Hyundai sign-in failed: {e}") from e
            self._login_error = None
            if refreshed:
                info("Hyundai: token refreshed")
            self._persist_token(manager)
            return manager

    def _refresh_or_login(self, manager: VehicleManager) -> bool:
        """The library's token check. When the saved token cannot be refreshed without a
        one-time code the library raises without keeping the request (it does so only on a
        fresh login), so the dead token is dropped and a fresh sign-in made to capture it."""
        try:
            return manager.check_and_refresh_token()
        except AuthenticationOTPRequired:
            if getattr(manager, "otp_request", None) is not None:
                raise
            info("Hyundai: the saved token cannot be refreshed without a one-time code — signing in afresh")
            self._drop_token(manager)
            if manager.login() is True:
                return True
            raise

    def _drop_token(self, manager: VehicleManager):
        manager.token = None
        self._persisted_token = None
        path = token_path(self._config["username"])
        try:
            path.unlink(missing_ok=True)
            trace(f"Hyundai: dead token {path} removed")
        except OSError as e:
            error(f"Hyundai: could not remove the dead token {path}: {e}")

    # --- the one-time code (the console tab) --------------------------------------------------

    def needs_2fa(self) -> bool:
        """True while a sign-in attempt is waiting for the one-time code; local state only."""
        return self._otp_request is not None

    def otp_destinations(self) -> dict:
        """Where the code can go — {"email": address, "sms": number} for the ones the account has."""
        request = self._otp_request
        if request is None:
            return {}
        out = {}
        if getattr(request, "has_email", False):
            out["email"] = getattr(request, "email", None) or "the account's e-mail"
        if getattr(request, "has_sms", False):
            out["sms"] = getattr(request, "sms", None) or "the account's phone"
        return out

    def send_otp(self, channel: str) -> str:
        """Have the backend send the code by "email" or "sms"; what happened, for the tab."""
        if self._otp_request is None:
            trace("Hyundai: send_otp asked but no sign-in is waiting for a code")
            return "No sign-in is waiting for a code — press Ctrl-R to sign in first."
        notify = OTP_NOTIFY_TYPE.SMS if channel == "sms" else OTP_NOTIFY_TYPE.EMAIL
        try:
            self._manager().send_otp(notify)
        except HyundaiKiaException as e:
            info(f"Hyundai: sending the one-time code by {channel} failed: {e}")
            raise HyundaiLoginError(f"sending the code by {channel} failed: {e}") from e
        destination = self.otp_destinations().get(channel, channel)
        info(f"Hyundai: one-time code sent by {channel} to {destination}")
        return f"Code sent by {channel} to {destination} — enter it with Ctrl-E."

    def verify_otp(self, code: str) -> str:
        """Finish the sign-in with the code; the token is saved and the state cache dropped."""
        if self._otp_request is None:
            return "No sign-in is waiting for a code."
        manager = self._manager()
        try:
            manager.verify_otp_and_complete_login(code.strip())
        except HyundaiKiaException as e:
            info(f"Hyundai: the one-time code was refused: {e}")
            raise HyundaiLoginError(f"the code was refused: {e}") from e
        with self._lock:
            self._otp_request = None
            self._login_error = None
            self._cached_at = 0.0
            self._persist_token(manager)
        info(f"Hyundai: signed in with the one-time code — {len(manager.vehicles)} car(s) on the account")
        return "Signed in — the one-time code was verified."

    def get_auth_status(self) -> Dict[str, Any]:
        """The console tab's summary of the sign-in state."""
        if self._otp_request is not None:
            where = " or ".join(f"{k} ({v})" for k, v in self.otp_destinations().items()) or "e-mail"
            return {"status": "needs_auth", "message": f"One-time code needed: Ctrl-A sends it by e-mail, Ctrl-S by SMS ({where}), Ctrl-E enters it"}
        if self._login_error:
            return {"status": "error", "message": f"Sign-in refused: {self._login_error}"}
        manager = self._manager_instance
        if manager is None or getattr(manager, "token", None) is None:
            return {"status": "not_initialized", "message": "Not signed in yet — Ctrl-R signs in"}
        return {"status": "authenticated", "message": f"Signed in as {self._config['username']}, {len(manager.vehicles)} car(s) on the account"}

    def refresh_now(self):
        """Sign in (or refresh the token) and pull the cached state — the tab's Ctrl-R, the one
        call that starts a new sign-in over a one-time code still waiting to be entered."""
        with self._lock:
            if self._otp_request is not None:
                info("Hyundai: Ctrl-R while a one-time code was pending — that request is dropped, signing in anew")
                self._otp_request = None
        self._ensure_login()
        self._update_cached(force=True)

    # --- the cars ----------------------------------------------------------------------------

    def vehicles(self) -> List[VehicleSpec]:
        """The configured cars (`name` the vehicle_id, the VIN the handle); without a list, every
        car of the account under the library's name, which needs the sign-in."""
        entries = self._config.get("vehicles")
        if entries:
            specs = []
            for entry in entries:
                if not isinstance(entry, dict):
                    error(f"Invalid Hyundai vehicle configuration entry: {entry}")
                    continue
                name, vin, display_name = entry.get("name"), entry.get("vin"), entry.get("display_name")
                specs.append(VehicleSpec(
                    id=name or display_name or vin or "car",
                    label=display_name or name or vin or "car",
                    handle=vin,
                    aliases=tuple(a for a in (display_name, vin) if a),
                    is_default=bool(entry.get("default", False)),
                ))
            if not specs:
                raise HyundaiConfigurationError("No valid Hyundai vehicle configuration entries were found.")
            return specs
        manager = self._ensure_login()
        specs = [VehicleSpec(id=v.name or v.VIN, label=v.name or v.model or v.VIN, handle=v.VIN,
                             aliases=tuple(a for a in (v.model, v.VIN) if a))
                 for v in manager.vehicles.values()]
        if not specs:
            info("Hyundai: the account has no car")
        return specs

    def _library_vehicle(self, manager: VehicleManager, spec: VehicleSpec):
        if not spec.handle:
            raise HyundaiConfigurationError(f"Vehicle '{spec.label}' is missing its VIN in configuration.")
        for vehicle in manager.vehicles.values():
            if (vehicle.VIN or "").upper() == spec.handle.upper():
                return vehicle
        known = ", ".join(f"{v.name} ({v.VIN})" for v in manager.vehicles.values()) or "none"
        raise HyundaiError(f"The account has no car with VIN {spec.handle} ({spec.label}); its cars: {known}.")

    def _update_cached(self, force: bool = False):
        """One pull of the manufacturer's cached state for every car, at most every STATE_TTL
        seconds unless forced. Never the library's force refresh: that wakes the car."""
        manager = self._ensure_login()
        with self._lock:
            age = time.time() - self._cached_at
            if not force and age < STATE_TTL:
                trace(f"Hyundai: cached state is {age:.0f}s old, reused")
                return manager
            try:
                manager.update_all_vehicles_with_cached_state()
            except RateLimitingError as e:
                info(f"Hyundai: rate limited while reading the cached state: {e}")
                raise HyundaiError(f"Hyundai's servers are rate limiting the account: {e}") from e
            except HyundaiKiaException as e:
                info(f"Hyundai: reading the cached state failed: {e}")
                raise HyundaiError(f"Reading the cars' state failed: {e}") from e
            self._cached_at = time.time()
            self._persist_token(manager)
            trace(f"Hyundai: cached state pulled for {len(manager.vehicles)} car(s)")
            return manager

    def vehicle_state(self, spec: VehicleSpec) -> VehicleState:
        manager = self._update_cached()
        return self._to_state(self._library_vehicle(manager, spec))

    @staticmethod
    def _to_state(vehicle) -> VehicleState:
        """The library's Vehicle folded into the contract. The EV range and battery for an
        electric car, the fuel level and range for a combustion one, both for a plug-in hybrid;
        the library has no measured cabin temperature (its `air_temperature` is the climate's
        set point), so that stays None."""
        engine = engine_kind(vehicle)
        electric = engine in ("EV", "PHEV")
        burns_fuel = engine in ("ICE", "HEV", "PHEV")
        battery = getattr(vehicle, "ev_battery_percentage", None) if electric else None
        fuel = getattr(vehicle, "fuel_level", None) if burns_fuel else None
        range_km = None
        if electric:
            range_km = to_km(getattr(vehicle, "ev_driving_range", None), getattr(vehicle, "ev_driving_range_unit", None))
        if range_km is None and burns_fuel:
            range_km = to_km(getattr(vehicle, "fuel_driving_range", None), getattr(vehicle, "_fuel_driving_range_unit", None))
        if range_km is None:
            range_km = to_km(getattr(vehicle, "total_driving_range", None), getattr(vehicle, "total_driving_range_unit", None))
        updated = getattr(vehicle, "last_updated_at", None)
        return VehicleState(
            in_gear=bool(getattr(vehicle, "engine_is_running", False)),
            latitude=getattr(vehicle, "location_latitude", None),
            longitude=getattr(vehicle, "location_longitude", None),
            battery_percent=int(battery) if battery is not None else None,
            fuel_percent=int(fuel) if fuel is not None else None,
            range_km=range_km,
            climate_on=getattr(vehicle, "air_control_is_on", None),
            data_time=updated.timestamp() if updated is not None else None,
        )

    # --- the actions -------------------------------------------------------------------------

    def _command(self, spec: VehicleSpec, verb: str, call: Callable[[VehicleManager, str], Any]) -> str:
        """Run `call(manager, library_vehicle_id)`, mapping the library's refusals to the
        contract's errors; the tracking id it returns is logged, the state cache dropped."""
        manager = self._ensure_login()
        vehicle = self._library_vehicle(manager, spec)
        try:
            tracking = call(manager, vehicle.id)
        except NotImplementedError as e:
            info(f"Hyundai: {verb} is not offered in region {self._config.get('region')}: {e}")
            raise Unsupported(f"Hyundai cannot {verb} for {spec.label}: the {self._config.get('region')} backend has no such command.") from e
        except UnsupportedControlError as e:
            info(f"Hyundai: {spec.label} refused {verb}: {e}")
            raise Unsupported(f"{spec.label} does not support {verb}: {e}") from e
        except RateLimitingError as e:
            raise HyundaiError(f"Hyundai's servers are rate limiting the account: {e}") from e
        except HyundaiKiaException as e:
            info(f"Hyundai: {verb} for {spec.label} failed: {e}")
            raise HyundaiError(f"{verb} for {spec.label} failed: {e}") from e
        with self._lock:
            self._cached_at = 0.0
        info(f"Hyundai: {verb} sent to {spec.label} (tracking {tracking})")
        return str(tracking)

    def _climate_temperature_c(self) -> float:
        return float(self._config.get("climate_temperature_c") or DEFAULT_CLIMATE_C)

    def _climate_options(self, defrost: bool) -> ClimateRequestOptions:
        """The request for the configured set point and duration. The set point is in the unit the
        region's backend takes — °F in the USA, °C elsewhere — and the backend may rewrite it
        (Kia USA clamps to "LOW" / "HIGH"), so the message quotes the °C value, never this."""
        celsius = self._climate_temperature_c()
        fahrenheit = str(self._config.get("region") or "").strip().lower() in FAHRENHEIT_REGIONS
        set_temp = to_fahrenheit(celsius) if fahrenheit else celsius
        minutes = int(self._config.get("climate_minutes") or DEFAULT_CLIMATE_MINUTES)
        trace(f"Hyundai: climate set point {celsius:g} °C sent as {set_temp:g} °{'F' if fahrenheit else 'C'}")
        return ClimateRequestOptions(set_temp=set_temp, duration=minutes, climate=True,
                                     defrost=defrost, heating=1 if defrost else 0)

    def max_defrost(self, spec: VehicleSpec) -> str:
        options = self._climate_options(defrost=True)
        self._command(spec, "max defrost", lambda m, vid: m.start_climate(vid, options))
        return f"Max defrost started in {spec.label} ({self._climate_temperature_c():g} °C, {options.duration} min)."

    def climate_on(self, spec: VehicleSpec) -> str:
        options = self._climate_options(defrost=False)
        self._command(spec, "climate start", lambda m, vid: m.start_climate(vid, options))
        return f"Climate started in {spec.label} at {self._climate_temperature_c():g} °C for {options.duration} min."

    def climate_off(self, spec: VehicleSpec) -> str:
        self._command(spec, "climate stop", lambda m, vid: m.stop_climate(vid))
        return f"Climate stopped in {spec.label}."

    def navigate_to(self, spec: VehicleSpec, destination: str) -> str:
        """A destination for the car's navigation: "latitude,longitude" goes as is, an
        address or place name is geocoded with Nominatim (OpenStreetMap) first."""
        destination = (destination or "").strip()
        if not destination:
            raise HyundaiError("No destination given.")
        poi = self._poi(destination)
        self._command(spec, "navigation", lambda m, vid: m.set_navigation(vid, [poi]))
        return f"Navigation to {poi.name or destination} sent to {spec.label}."

    def _poi(self, destination: str) -> POIInfo:
        coordinates = parse_coordinates(destination)
        if coordinates:
            trace(f"Hyundai: destination '{destination}' is a coordinate pair")
            return POIInfo(name=destination, addr=destination, coord=POICoord(lat=coordinates[0], lon=coordinates[1]))
        lat, lon, address = self._geocode(destination)
        return POIInfo(name=destination, addr=address, coord=POICoord(lat=lat, lon=lon))

    def _geocode(self, place: str) -> tuple[float, float, str]:
        info(f"Hyundai: geocoding '{place}' with Nominatim")
        try:
            response = self._session.get(NOMINATIM_URL, params={"q": place, "format": "json", "limit": 1},
                                         headers={"User-Agent": "Keep home assistant (car navigation)"},
                                         timeout=GEOCODE_TIMEOUT)
            results = response.json() if response.status_code == 200 else []
        except (requests.RequestException, ValueError) as e:
            raise HyundaiError(f"Could not look up '{place}': {e}") from e
        if not results:
            info(f"Hyundai: Nominatim knows no place called '{place}'")
            raise HyundaiError(f"No place called '{place}' was found — give a street address with its town, or coordinates.")
        hit = results[0]
        return float(hit["lat"]), float(hit["lon"]), hit.get("display_name", place)

    # --- health ------------------------------------------------------------------------------

    def check_health(self) -> bool:
        manager = self._update_cached()
        specs = self.vehicles()
        if not specs:
            return False
        vehicle = self._library_vehicle(manager, specs[0])
        healthy = getattr(vehicle, "last_updated_at", None) is not None
        if not healthy:
            trace(f"Hyundai: {specs[0].label} has no state yet")
        return healthy
