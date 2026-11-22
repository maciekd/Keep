from typing import Any, Dict, List, Optional

import requests

from common.ConfigManager import ConfigManager
from assistant.integration.Integration import Integration
from assistant.integration.VehicleProvider import VehicleError, VehicleProvider, VehicleSpec, VehicleState
from common.logging import info, error


class TeslaAPIError(VehicleError):
    """Base class for Tesla API related errors."""


class TeslaConfigurationError(TeslaAPIError):
    """Raised when required Tesla configuration is missing or invalid."""


class TeslaCommandError(TeslaAPIError):
    """Raised when a command call to the Tesla API fails."""


class TeslaIntegration(Integration, VehicleProvider):
    """The Teslas via Tessie: state, climate, trunk and navigation.

    Reads the state of the cars and sends them commands: defrost, climate on and off, opening
    the trunk, a destination for the navigation.

    A VehicleProvider: the `cars` toolkit and the CURRENT CAR INFO section reach it through
    the fleet, beside the other brands. Talks to the Tessie API; a car's handle is its VIN."""

    TESSIE_API_BASE = "https://api.tessie.com"
    DEFAULT_NAVIGATION_LOCALE = "en-US"

    def __init__(
        self,
        config_manager: Optional[ConfigManager] = None,
        session: Optional[requests.Session] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self._config_manager = config_manager or ConfigManager()
        self._session = session or requests.Session()

        self._config = self._load_config(config)
        self._validate_config()

        self._access_token = self._config.get("tessie_access_token")

    @property
    def name(self) -> str:
        return "Tesla"

    @property
    def config_keys(self) -> List[str]:
        return ["tesla"]

    def check_health(self) -> bool:
        specs = self.vehicles()
        if not specs or not specs[0].handle:
            return False
        state = self._api_get(specs[0].handle, "state", max_retries=1)
        inside_temp = state.get("climate_state", {}).get("inside_temp")
        return inside_temp is not None

    def _load_config(self, override_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if override_config is not None:
            return override_config

        try:
            return self._config_manager.get("tesla")
        except KeyError as exc:
            raise TeslaConfigurationError(
                "Missing 'tesla' section in configuration."
            ) from exc

    def _validate_config(self):
        if not self._config.get("tessie_access_token"):
            raise TeslaConfigurationError(
                "Missing Tesla configuration value: tessie_access_token"
            )

        vehicles = self._config.get("vehicles") or []
        if not vehicles:
            raise TeslaConfigurationError("Tesla vehicle list is empty.")

    def vehicles(self) -> List[VehicleSpec]:
        """The configured cars: `name` is the vehicle_id, the VIN the handle, the display
        name an alias (and the label when set)."""
        specs: List[VehicleSpec] = []
        for entry in self._config.get("vehicles", []):
            if not isinstance(entry, dict):
                error(f"Invalid Tesla vehicle configuration entry: {entry}")
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
            raise TeslaConfigurationError(
                "No valid Tesla vehicle configuration entries were found."
            )
        return specs

    @staticmethod
    def _require_vin(spec: VehicleSpec) -> str:
        if not spec.handle:
            raise TeslaConfigurationError(
                f"Vehicle '{spec.label}' is missing VIN in configuration."
            )
        return spec.handle

    def _api_request(self, method: str, vin: str, endpoint: str, max_retries: int = 3, **kwargs) -> Dict[str, Any]:
        """Make request to Tessie API with retries."""
        url = f"{self.TESSIE_API_BASE}/{vin}/{endpoint}"
        headers = {"Authorization": f"Bearer {self._access_token}"}

        last_exception = None
        for attempt in range(1, max_retries + 1):
            try:
                response = getattr(self._session, method.lower())(url, headers=headers, timeout=15, **kwargs)

                if response.status_code != 200:
                    raise TeslaAPIError(
                        f"Tessie {method} {endpoint} failed: {response.status_code} {response.text}"
                    )

                return response.json()
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                last_exception = e
                if attempt < max_retries:
                    info(f"Tessie {method} {endpoint} attempt {attempt}/{max_retries} failed: {e}. Retrying...")
                else:
                    error(f"Tessie {method} {endpoint} failed after {max_retries} attempts: {e}")

        raise TeslaAPIError(f"Tessie {method} {endpoint} failed after {max_retries} retries: {last_exception}")

    def _api_get(self, vin: str, endpoint: str, max_retries: int = 3) -> Dict[str, Any]:
        """Make GET request to Tessie API with retries."""
        return self._api_request("GET", vin, endpoint, max_retries)

    def _api_post(self, vin: str, endpoint: str, retry: bool = True, max_retries: int = 3,
                  query: Optional[Dict[str, Any]] = None, **body_params) -> Dict[str, Any]:
        """Make POST request to Tessie API with retries; `query` goes on the URL, the rest in the body."""
        payload = {"retry": retry, **body_params}
        kwargs = {"json": payload}
        if query:
            kwargs["params"] = query
        data = self._api_request("POST", vin, endpoint, max_retries, **kwargs)
        result = data.get("result")
        if not result:
            reason = data.get("reason", "Unknown error")
            raise TeslaCommandError(reason)
        return data

    def vehicle_state(self, spec: VehicleSpec) -> VehicleState:
        """Tessie's cached `/state` of the car, folded into the contract's shape. A car that
        is not online gets the snapshot's timestamp as `data_time`, so the prompt can say how
        old the data is."""
        vin = self._require_vin(spec)
        state = self._api_get(vin, "state")
        drive, charge, climate = state.get("drive_state", {}), state.get("charge_state", {}), state.get("climate_state", {})
        connectivity = state.get("state")
        timestamp_ms = drive.get("timestamp")
        return VehicleState(
            connectivity=connectivity,
            in_gear=drive.get("shift_state") in ("D", "R", "N"),
            speed_kmh=drive.get("speed"),
            latitude=drive.get("latitude"),
            longitude=drive.get("longitude"),
            battery_percent=charge.get("usable_battery_level"),
            charging=charge.get("charging_state") == "Charging",
            cabin_temp_c=climate.get("inside_temp"),
            climate_on=climate.get("is_climate_on"),
            destination=drive.get("active_route_destination"),
            minutes_to_arrival=drive.get("active_route_minutes_to_arrival"),
            data_time=timestamp_ms / 1000 if timestamp_ms and connectivity not in (None, "online") else None,
        )

    def max_defrost(self, spec: VehicleSpec) -> str:
        self._api_post(self._require_vin(spec), "command/start_max_defrost")
        return f"Max defrost enabled for {spec.label}."

    def climate_on(self, spec: VehicleSpec) -> str:
        self._api_post(self._require_vin(spec), "command/start_climate")
        return f"Air conditioning started for {spec.label}."

    def climate_off(self, spec: VehicleSpec) -> str:
        self._api_post(self._require_vin(spec), "command/stop_climate")
        return f"Air conditioning stopped for {spec.label}."

    def open_trunk(self, spec: VehicleSpec) -> str:
        self._api_post(self._require_vin(spec), "command/activate_rear_trunk")
        return f"Trunk opened for {spec.label}."

    def navigate_to(self, spec: VehicleSpec, destination: str) -> str:
        """Send a destination (an address or coordinates) to the vehicle's navigation."""
        destination = (destination or "").strip()
        if not destination:
            raise TeslaCommandError("No destination given.")
        vin = self._require_vin(spec)
        locale = self._config.get("navigation_locale") or self.DEFAULT_NAVIGATION_LOCALE

        info(f"Tesla: sending navigation destination '{destination}' to {spec.label} (locale {locale})")
        self._api_post(vin, "command/share", query={"value": destination, "locale": locale})
        return f"Navigation to {destination} sent to {spec.label}."
