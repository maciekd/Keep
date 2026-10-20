from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
import threading
import time

import requests

from assistant.integration.Integration import Integration
from assistant.integration.OAuthPaste import SignIn
from assistant.integration.VehicleProvider import VehicleError, VehicleProvider, VehicleSpec, VehicleState
from common.ConfigManager import ConfigManager
from common.logging import error, info, trace
from integrations.volvo.VolvoAuth import (NotSignedIn, VolvoAuth, delete_token_file, not_signed_in_hint,
                                          sign_in_again_hint, token_path)

SECTION = "volvo"
API_KEY_KEY = f"{SECTION}.vcc_api_key"
CLIENT_ID_KEY = f"{SECTION}.oauth.client_id"
CLIENT_SECRET_KEY = f"{SECTION}.oauth.client_secret"
VEHICLES_KEY = f"{SECTION}.vehicles"

API_URL = "https://api.volvocars.com"
CONNECTED = f"{API_URL}/connected-vehicle/v2/vehicles"
ENERGY = f"{API_URL}/energy/v2/vehicles"
LOCATION = f"{API_URL}/location/v1/vehicles"
TIMEOUT = 15
RETRIES = 2
RETRY_PAUSE_S = 1.5
# Since 2026-10 Volvo answers about half the calls, at random per endpoint and car, with 401
# "invalid VCC-API-KEY" while the key, the VIN and the token are fine; the same call a moment
# later goes through. That answer is asked again once before it counts as a refusal.
FLAKY_REFUSAL = "invalid vcc-api-key"
# A read that still fails leaves the endpoint's last answer standing for so long; the engine
# status for a short while only, since a stale "RUNNING" would keep a parked car on the road.
KEEP_S = 24 * 3600
KEEP_ENGINE_S = 600
STATE_TTL = 60                 # the prompt section may ask every minute while a car drives: one set of reads per minute per car
COMMAND_OK = ("COMPLETED", "DELIVERED")

# The car's reachability as the command-accessibility endpoint says it, in the contract's words
CONNECTIVITY = {"AVAILABLE": "online", "CAR_IN_USE": "online", "NO_INTERNET": "offline",
                "POWER_SAVING_MODE": "asleep", "OTA_INSTALLATION_IN_PROGRESS": "updating"}
BATTERY_FUEL_TYPES = ("ELECTRIC", "PETROL/ELECTRIC", "NONE")
COMBUSTION_FUEL_TYPES = ("DIESEL", "PETROL", "PETROL/ELECTRIC")


class VolvoApiError(VehicleError):
    """A Volvo Cars API call that failed; the message is for the LLM."""


class VolvoRateLimited(VolvoApiError):
    """HTTP 429: the app's daily or per-minute quota is spent."""


class VolvoCommandError(VolvoApiError):
    """A command the car did not carry out: the invoke status and Volvo's reason."""


class VolvoIntegration(Integration, VehicleProvider):
    """The Volvos through the Volvo Cars API: state and climate.

    Shows where each Volvo is, whether it is in use and how much charge or range it has, and
    starts or stops its climate system; the household publishes its own app on Volvo's
    developer portal once and signs in with its Volvo ID from the chat or the console's tab.

    A VehicleProvider over the Connected Vehicle API v2 (vehicle details, engine status,
    statistics, fuel, command accessibility, the climatization commands), the Energy API v2
    (battery level, electric range) and the Location API; a car's handle is its VIN. Every
    call carries the household's VCC API key (`volvo.vcc_api_key`) and a bearer token from
    `VolvoAuth`, the sign-in of the app whose client id and secret are `volvo.oauth.*`. A
    car's composed state is cached for STATE_TTL seconds and its details (fuel type) for the
    process, since the app has 10,000 calls a day. Volvo answers a VIN that is not among the
    Volvo ID's cars with 401 "invalid VCC-API-KEY", the same words as for a bad key, so a
    refused per-car call is told apart by one read of the account's car list (cached for the
    process, VINs compared case aside) before the error names the VIN, the key or the scopes.
    Volvo also gives that answer at random to calls that are fine, so a 401 with those words is
    asked once more first, and a state read that still fails keeps the endpoint's last answer
    (KEEP_S) instead of blanking the car's battery, range and location. Volvo offers no defrost, trunk or
    navigation command, so those three stay the contract's Unsupported. `needs_2fa` is "the
    owner has to sign in": the monitor shows NEEDS_2FA with the login hint until the paste
    flow completes. Config is read at call time."""
    login_hint = "volvo_login() from the chat, or the console's Volvo tab"

    def __init__(self, session: Optional[requests.Session] = None):
        Integration.__init__(self)
        self._session = session or requests.Session()
        self.auth = VolvoAuth(self._client_settings)
        self._details: Dict[str, dict] = {}
        self._states: Dict[str, tuple[float, VehicleState]] = {}
        self._account_vins: Optional[List[str]] = None
        self._kept: Dict[str, Dict[str, tuple[float, dict]]] = {}     # vin -> read -> (when, answer)
        self._lock = threading.RLock()

    @property
    def name(self) -> str:
        return "Volvo"

    @property
    def config_keys(self) -> List[str]:
        return [API_KEY_KEY, CLIENT_ID_KEY, CLIENT_SECRET_KEY, VEHICLES_KEY]

    # --- config --------------------------------------------------------------------------------

    @staticmethod
    def _config(key: str, default=None):
        return ConfigManager.get_instance().get(key, default)

    @classmethod
    def _client_settings(cls) -> tuple[str, str]:
        return str(cls._config(CLIENT_ID_KEY, "") or ""), str(cls._config(CLIENT_SECRET_KEY, "") or "")

    def vehicles(self) -> List[VehicleSpec]:
        """The configured cars: `name` is the vehicle_id, the VIN the handle, the display
        name an alias (and the label when set)."""
        entries = self._config(VEHICLES_KEY, []) or []
        specs: List[VehicleSpec] = []
        for entry in entries:
            if not isinstance(entry, dict):
                error(f"Volvo: invalid vehicle configuration entry: {entry}")
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
            info("Volvo: volvo.vehicles lists no car")
        return specs

    @staticmethod
    def _require_vin(spec: VehicleSpec) -> str:
        if not spec.handle:
            raise VolvoApiError(f"Vehicle '{spec.label}' is missing its VIN in configuration.")
        return spec.handle

    # --- the sign-in, for the commands and the tab ---------------------------------------------

    def signed_in(self) -> bool:
        return self.auth.signed_in()

    def login_begin(self) -> SignIn:
        return self.auth.begin()

    def login_finish(self, pasted: str) -> str:
        """"" when the pasted answer signed the Volvo ID in, else why not."""
        failure = self.auth.finish(pasted)
        if not failure:
            self._forget_account()
        return failure

    def sign_out(self) -> None:
        self.auth.sign_out()
        self._forget_account()

    def _forget_account(self) -> None:
        """A new Volvo ID (or none): its car list and the states are the old one's."""
        with self._lock:
            self._states.clear()
            self._kept.clear()
            self._account_vins = None

    def needs_2fa(self) -> bool:
        """The owner has to sign in (never did, or the refresh was refused): local state only."""
        return not self.auth.signed_in() or self.auth.needs_sign_in_again()

    def get_auth_status(self) -> dict:
        """The console tab's summary of the sign-in."""
        client_id, client_secret = self._client_settings()
        if not client_id or not client_secret or not self._config(API_KEY_KEY):
            return {"status": "error", "message": "volvo.vcc_api_key, volvo.oauth.client_id and client_secret are needed"}
        if not self.auth.signed_in():
            return {"status": "needs_auth", "message": "Sign-in needed — Ctrl-L shows the link, Ctrl-E takes the pasted address"}
        if self.auth.needs_sign_in_again():
            return {"status": "needs_auth", "message": f"The sign-in lapsed: {self.auth.refresh_error} — Ctrl-L signs in again"}
        return {"status": "authenticated", "message": f"Signed in; {len(self.vehicles())} car(s) configured"}

    @classmethod
    def wipe_targets(cls) -> list:
        return [(token_path(), "the Volvo sign-in (its OAuth tokens)")]

    @classmethod
    def log_out(cls) -> Optional[str]:
        return "Volvo: token file deleted" if delete_token_file() else None

    def on_not_configured(self):
        delete_token_file()

    # --- health ----------------------------------------------------------------------------------

    def check_health(self) -> bool:
        specs = self.vehicles()
        if not specs or not specs[0].handle:
            info("Volvo: no car with a VIN to check")
            return False
        if not self.auth.signed_in():
            raise NotSignedIn(not_signed_in_hint())
        details = self._vehicle_details(specs[0].handle)
        return bool(details.get("vin"))

    # --- the API -------------------------------------------------------------------------------

    def _request(self, method: str, url: str, body: Optional[dict] = None) -> dict:
        """One call with the bearer token and the API key: the JSON body, {} for a 404 (an
        endpoint the car or region lacks), a VehicleError for everything else. A timeout or a
        dropped connection is tried RETRIES times and Volvo's random 401 "invalid VCC-API-KEY"
        once more, each after RETRY_PAUSE_S; a refused POST was not carried out, so asking
        again cannot run a command twice."""
        api_key = str(self._config(API_KEY_KEY, "") or "")
        headers = {"Authorization": f"Bearer {self.auth.token()}", "vcc-api-key": api_key, "Accept": "application/json"}
        if method == "POST":
            headers["Content-Type"] = "application/json"
        path = url[len(API_URL):]
        refusal_retried = False
        while True:
            response = self._send(method, url, path, headers, body)
            try:
                data = response.json() if response.content else {}
            except ValueError:
                data = {}
            if response.status_code != 401 or refusal_retried:
                break
            reason = self._error_text(data) or response.text[:200]
            if FLAKY_REFUSAL not in reason.lower():
                break
            refusal_retried = True
            info(f"Volvo: {method} {path} refused (401: {reason}) — Volvo's random refusal, asked again in {RETRY_PAUSE_S} s")
            time.sleep(RETRY_PAUSE_S)
        if response.status_code == 404:
            trace(f"Volvo: {method} {path} is not there for this car (404)")
            return {}
        if response.status_code == 429:
            info(f"Volvo: {method} {path} rate limited (429): {response.text[:200]}")
            raise VolvoRateLimited("Volvo's API rate limit is reached (10,000 calls a day per app, about 100 a minute) — try again later.")
        if response.status_code in (401, 403):
            reason = self._error_text(data) or response.text[:200]
            info(f"Volvo: {method} {path} refused ({response.status_code}{', after a second try' if refusal_retried else ''}): "
                 f"{reason} — body: {response.text[:200]}")
            raise VolvoApiError(self._refusal(path, response.status_code, reason))
        if response.status_code >= 400:
            reason = self._error_text(data) or response.text[:200]
            info(f"Volvo: {method} {path} failed ({response.status_code}): {reason} — body: {response.text[:200]}")
            raise VolvoApiError(f"Volvo {method} {path} failed: {response.status_code} {reason}")
        if refusal_retried:
            info(f"Volvo: {method} {path} answered {response.status_code} on the second try")
        return data if isinstance(data, dict) else {}

    def _send(self, method: str, url: str, path: str, headers: dict, body: Optional[dict]):
        """The response, after up to RETRIES attempts against timeouts and dropped connections."""
        cause = ""
        for attempt in range(1, RETRIES + 1):
            if attempt > 1:
                time.sleep(RETRY_PAUSE_S)
            try:
                return self._session.request(method, url, headers=headers, json=body, timeout=TIMEOUT)
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                cause = self._cause(e)
                info(f"Volvo: {method} {path} attempt {attempt}/{RETRIES} failed: {cause}")
        raise VolvoApiError(f"Volvo's API could not be reached: {cause}")

    @staticmethod
    def _cause(e: BaseException) -> str:
        """The exception and what it wraps (requests hides RemoteDisconnected two levels down)."""
        parts, seen = [], set()
        while e is not None and id(e) not in seen and len(parts) < 4:
            seen.add(id(e))
            parts.append(f"{type(e).__name__}: {e}")
            inner = e.__cause__ or e.__context__
            if inner is None and e.args and isinstance(e.args[0], BaseException):
                inner = e.args[0]
            e = inner
        return " ← ".join(parts)

    def _refusal(self, path: str, status: int, reason: str) -> str:
        """Why a 401/403 happened, for the LLM: a VIN the Volvo ID does not own (Volvo's answer
        is the same "invalid VCC-API-KEY" as for a bad key), the key itself, or the scopes."""
        refused = f"Volvo refused the call ({status}: {reason})"
        vin = self._vin_in(path)
        if not vin:
            return f"{refused} — the token or the app's scopes do not cover it; {sign_in_again_hint()} if it persists."
        owned = self._owned_vins()
        flaky = FLAKY_REFUSAL in reason.lower()
        if owned is None:
            return (f"{refused}, and so is the Volvo ID's car list — check volvo.vcc_api_key (the VCC API key of the "
                    f"app on developer.volvocars.com) and the app's scopes; {sign_in_again_hint()} if it persists.")
        if vin.upper() not in {v.upper() for v in owned}:
            listed = ", ".join(f"…{v[-6:]}" for v in owned) or "none"
            return (f"{refused} — Volvo answers this for a VIN that is not among the Volvo ID's cars, and {vin} is not: "
                    f"the account has {len(owned)} car(s) with VIN ending {listed}. Check the `vin` of that car in volvo.vehicles.")
        if flaky:
            return (f"{refused} twice in a row — the VIN is the Volvo ID's and Volvo refuses its calls at random with "
                    f"these words; try again in a moment, and {sign_in_again_hint()} if it persists.")
        return f"{refused} — the VIN is the Volvo ID's, so the app's scopes do not cover this call; {sign_in_again_hint()} if it persists."

    def _vin_in(self, path: str) -> str:
        """The VIN a per-car path carries, "" for the car list and the token endpoint."""
        for prefix in (CONNECTED, ENERGY, LOCATION):
            if path.startswith(prefix[len(API_URL):] + "/"):
                return path[len(prefix) - len(API_URL) + 1:].split("/", 1)[0]
        return ""

    def _owned_vins(self) -> Optional[List[str]]:
        """The VINs of the Volvo ID's cars (kept for the process); None when the list is refused, a
        404 or empty — a Volvo ID with cars answers at least one, so an empty list says nothing
        about the VIN and the key / sign-in wording applies."""
        with self._lock:
            if self._account_vins is not None:
                return self._account_vins
        try:
            body = self._request("GET", CONNECTED)
        except VolvoRateLimited:
            raise
        except VolvoApiError as e:
            info(f"Volvo: the car list is refused too — the key or the token is at fault, not a VIN: {e}")
            return None
        rows = body.get("data") if isinstance(body.get("data"), list) else []
        vins = [str(row.get("vin")) for row in rows if isinstance(row, dict) and row.get("vin")]
        if not vins:
            info("Volvo: the car list came back empty or missing — inconclusive, not cached")
            return None
        info(f"Volvo: the Volvo ID has {len(vins)} car(s): {', '.join('…' + v[-6:] for v in vins)}")
        with self._lock:
            self._account_vins = vins
        return vins

    @staticmethod
    def _error_text(data: dict) -> str:
        err = data.get("error") if isinstance(data, dict) else None
        if isinstance(err, dict):
            return " ".join(str(err.get(k)) for k in ("message", "description") if err.get(k)).strip()
        return str(err) if err else ""

    def _get_data(self, url: str) -> dict:
        body = self._request("GET", url)
        data = body.get("data", body)
        return data if isinstance(data, dict) else {}

    def _vehicle_details(self, vin: str) -> dict:
        with self._lock:
            cached = self._details.get(vin)
        if cached is not None:
            return cached
        details = self._get_data(f"{CONNECTED}/{vin}")
        if details:
            with self._lock:
                self._details[vin] = details
            trace(f"Volvo: {vin[-6:]} is a {details.get('modelYear', '')} {(details.get('descriptions') or {}).get('model', 'Volvo')}, fuel type {details.get('fuelType')}")
        else:
            info(f"Volvo: no vehicle details for {vin[-6:]} — the VIN may not be in the Volvo ID's cars")
        return details

    # --- the state -----------------------------------------------------------------------------

    @staticmethod
    def _value(fields: dict, key: str) -> Any:
        field = fields.get(key)
        return field.get("value") if isinstance(field, dict) else field

    @staticmethod
    def _number(fields: dict, key: str) -> Optional[float]:
        value = VolvoIntegration._value(fields, key)
        try:
            return float(value) if value not in (None, "") else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _timestamp(fields: dict, *keys: str) -> Optional[float]:
        """The newest `timestamp` / `updatedAt` among the named fields (all when none are named), epoch seconds."""
        newest = None
        for key in keys or fields.keys():
            field = fields.get(key)
            stamp = field.get("timestamp") or field.get("updatedAt") if isinstance(field, dict) else None
            if not stamp:
                continue
            try:
                at = datetime.fromisoformat(str(stamp).replace("Z", "+00:00")).timestamp()
            except ValueError:
                continue
            newest = at if newest is None or at > newest else newest
        return newest

    def _read(self, label: str, vin: str, fetch: Callable[[], dict], keep_s: float = KEEP_S) -> dict:
        """One endpoint of the composed state. A failed read leaves the endpoint's last answer
        standing for keep_s seconds (its fields carry Volvo's own timestamps), and only after that
        its fields unknown rather than failing the car; an auth or rate-limit error, which every
        read would repeat, fails it."""
        try:
            data = fetch()
        except (NotSignedIn, VolvoRateLimited):
            raise
        except VehicleError as e:
            with self._lock:
                kept = self._kept.get(vin, {}).get(label)
            if kept and time.time() - kept[0] < keep_s:
                info(f"Volvo: {label} of {vin[-6:]} not read, the answer of "
                     f"{datetime.fromtimestamp(kept[0]).strftime('%H:%M')} stands: {e}")
                return kept[1]
            if kept:
                info(f"Volvo: {label} of {vin[-6:]} not read, and its last answer "
                     f"({datetime.fromtimestamp(kept[0]).strftime('%d %b %H:%M')}) is too old to stand: {e}")
                with self._lock:
                    self._kept.get(vin, {}).pop(label, None)
            else:
                info(f"Volvo: {label} of {vin[-6:]} not read, nothing kept to stand in: {e}")
            return {}
        with self._lock:
            self._kept.setdefault(vin, {})[label] = (time.time(), data)
        return data

    def vehicle_state(self, spec: VehicleSpec) -> VehicleState:
        """The car now, composed from the cached answers Volvo's servers hold (there is no live
        feed, so `data_time` is the newest timestamp among them), and kept for STATE_TTL."""
        vin = self._require_vin(spec)
        with self._lock:
            cached = self._states.get(vin)
        if cached and time.time() - cached[0] < STATE_TTL:
            trace(f"Volvo: state of {spec.label} served from the {int(time.time() - cached[0])} s old cache")
            return cached[1]
        details = self._vehicle_details(vin)
        fuel_type = str(details.get("fuelType") or "")
        has_battery = fuel_type in BATTERY_FUEL_TYPES or not fuel_type
        has_combustion = fuel_type in COMBUSTION_FUEL_TYPES

        # kept for minutes like the engine: a stale CAR_IN_USE would say "in use" of a parked car all day
        access = self._read("command accessibility", vin, lambda: self._get_data(f"{CONNECTED}/{vin}/command-accessibility"), KEEP_ENGINE_S)
        engine = self._read("engine status", vin, lambda: self._get_data(f"{CONNECTED}/{vin}/engine-status"), KEEP_ENGINE_S)
        statistics = self._read("statistics", vin, lambda: self._get_data(f"{CONNECTED}/{vin}/statistics"))
        energy = self._read("energy state", vin, lambda: self._request("GET", f"{ENERGY}/{vin}/state")) if has_battery else {}
        fuel = self._read("fuel", vin, lambda: self._get_data(f"{CONNECTED}/{vin}/fuel")) if has_combustion else {}
        location = self._read("location", vin, lambda: self._get_data(f"{LOCATION}/{vin}/location"))

        availability = str(self._value(access, "availabilityStatus") or "").upper()
        engine_running = str(self._value(engine, "engineStatus") or "").upper() == "RUNNING"
        battery = self._number(energy, "batteryChargeLevel")
        if battery is None and has_battery:
            battery = self._number(fuel, "batteryChargeLevel")
        electric_range = self._number(energy, "electricRange")
        if electric_range is None:
            electric_range = self._number(statistics, "distanceToEmptyBattery")
        tank_range = self._number(statistics, "distanceToEmptyTank") if has_combustion else None
        ranges = [r for r in (electric_range, tank_range) if r is not None]

        coordinates = ((location.get("geometry") or {}).get("coordinates") or []) if isinstance(location, dict) else []
        latitude = longitude = None
        if len(coordinates) >= 2:
            longitude, latitude = coordinates[0], coordinates[1]   # GeoJSON order
        location_stamp = (location.get("properties") or {}).get("timestamp") if isinstance(location, dict) else None
        stamps = [t for t in (self._timestamp(access), self._timestamp(engine), self._timestamp(statistics),
                              self._timestamp(energy), self._timestamp(fuel),
                              self._timestamp({"location": {"timestamp": location_stamp}}) if location_stamp else None)
                  if t is not None]

        state = VehicleState(
            connectivity=CONNECTIVITY.get(availability),
            # Volvo reports no gear: an engine that runs (the drivetrain on an EV too) or a car in use is on the road
            in_gear=engine_running or availability == "CAR_IN_USE",
            speed_kmh=None,
            latitude=latitude,
            longitude=longitude,
            battery_percent=int(round(battery)) if battery is not None else None,
            fuel_percent=None,   # Volvo reports litres, not a share of the tank
            range_km=sum(ranges) if ranges else None,
            data_time=max(stamps) if stamps else None,
        )
        if not any((access, engine, statistics, energy, fuel, location)):
            raise VolvoApiError(f"Volvo answered none of the state reads for {spec.label}.")
        with self._lock:
            self._states[vin] = (time.time(), state)
        trace(f"Volvo: {spec.label} is {state.connectivity or 'of unknown reachability'}, "
              f"{'in use' if state.in_gear else 'parked'}, battery {state.battery_percent}%, range {state.range_km} km")
        return state

    # --- the commands --------------------------------------------------------------------------

    def _invoke(self, spec: VehicleSpec, command: str, body: Optional[dict] = None) -> str:
        """POSTs one of the Connected Vehicle API's commands and reads its invoke status."""
        vin = self._require_vin(spec)
        info(f"Volvo: sending {command} to {spec.label}")
        answer = self._request("POST", f"{CONNECTED}/{vin}/commands/{command}", body or {})
        data = answer.get("data") if isinstance(answer.get("data"), dict) else answer
        status = str(data.get("invokeStatus") or "").upper()
        if status not in COMMAND_OK:
            message = str(data.get("message") or "").strip()
            info(f"Volvo: {command} for {spec.label} did not complete: {status or 'no status'} {message}")
            raise VolvoCommandError(f"{spec.label} did not carry out {command}: {status or 'no answer'}" + (f" — {message}" if message else ""))
        with self._lock:
            self._states.pop(vin, None)
        trace(f"Volvo: {command} for {spec.label} {status}")
        return status

    def climate_on(self, spec: VehicleSpec) -> str:
        self._invoke(spec, "climatization-start")
        return f"Climate started in {spec.label}."

    def climate_off(self, spec: VehicleSpec) -> str:
        self._invoke(spec, "climatization-stop")
        return f"Climate stopped in {spec.label}."
