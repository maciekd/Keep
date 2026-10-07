from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

from common.ConfigManager import ConfigManager
from assistant.integration.Integration import Integration
from common.logging import info, error


class TeslaAPIError(Exception):
    """Base class for Tesla API related errors."""


class TeslaConfigurationError(TeslaAPIError):
    """Raised when required Tesla configuration is missing or invalid."""


class TeslaCommandError(TeslaAPIError):
    """Raised when a command call to the Tesla API fails."""


@dataclass
class TeslaVehicleSpec:
    name: Optional[str]
    vin: Optional[str]
    display_name: Optional[str]
    is_default: bool = False

    def matches(self, target: str) -> bool:
        """Check if this spec matches a provided target identifier."""
        if not target:
            return False

        target_normalized = target.strip().lower()
        candidates = [
            (self.name or "").lower(),
            (self.display_name or "").lower(),
            (self.vin or "").lower(),
        ]
        candidates = [c for c in candidates if c]

        return any(target_normalized == c for c in candidates)

    def label(self) -> str:
        """Return the best label to use for human facing messages."""
        return self.display_name or self.name or self.vin or "car"


class TeslaIntegration(Integration):
    """The Teslas, through Tessie.

    Reads the state of the cars and sends them commands: defrost, climate on and off, opening
    the trunk, a destination for the navigation.

    Talks to the Tessie API."""

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
        self._vehicle_cache: Optional[List[Dict[str, Any]]] = None

    @property
    def name(self) -> str:
        return "Tesla"

    @property
    def config_keys(self) -> List[str]:
        return ["tesla"]

    def check_health(self) -> bool:
        if not self._vehicle_specs:
            return False
        spec = self._vehicle_specs[0]
        state = self._api_get(spec.vin, "state", max_retries=1)
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

    @property
    def _vehicle_specs(self) -> List[TeslaVehicleSpec]:
        specs: List[TeslaVehicleSpec] = []
        for entry in self._config.get("vehicles", []):
            if not isinstance(entry, dict):
                error(f"Invalid Tesla vehicle configuration entry: {entry}")
                continue
            specs.append(
                TeslaVehicleSpec(
                    name=entry.get("name"),
                    vin=entry.get("vin"),
                    display_name=entry.get("display_name"),
                    is_default=bool(entry.get("default", False)),
                )
            )
        if not specs:
            raise TeslaConfigurationError(
                "No valid Tesla vehicle configuration entries were found."
            )
        return specs

    def _select_vehicle_spec(self, vehicle_name: Optional[str]) -> TeslaVehicleSpec:
        specs = self._vehicle_specs
        if vehicle_name:
            for spec in specs:
                if spec.matches(vehicle_name):
                    return spec
            raise TeslaConfigurationError(
                f"No Tesla vehicle matches identifier '{vehicle_name}'."
            )

        for spec in specs:
            if spec.is_default:
                return spec
        return specs[0]

    def _require_vehicle_spec_with_vin(self, vehicle_name: Optional[str]) -> TeslaVehicleSpec:
        spec = self._select_vehicle_spec(vehicle_name)
        if not spec.vin:
            raise TeslaConfigurationError(
                f"Vehicle '{spec.label()}' is missing VIN in configuration."
            )
        return spec

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

    def set_max_defrost(self, vehicle_name: Optional[str]) -> str:
        """Enable max defrost on vehicle."""
        spec = self._require_vehicle_spec_with_vin(vehicle_name)

        self._api_post(spec.vin, "command/start_max_defrost")
        return f"Max defrost enabled for {spec.label()}."

    def start_air_conditioning(self, vehicle_name: Optional[str]) -> str:
        """Start air conditioning on vehicle."""
        spec = self._require_vehicle_spec_with_vin(vehicle_name)

        self._api_post(spec.vin, "command/start_climate")
        return f"Air conditioning started for {spec.label()}."

    def stop_air_conditioning(self, vehicle_name: Optional[str]) -> str:
        """Stop air conditioning on vehicle."""
        spec = self._require_vehicle_spec_with_vin(vehicle_name)

        self._api_post(spec.vin, "command/stop_climate")
        return f"Air conditioning stopped for {spec.label()}."

    def open_trunk(self, vehicle_name: Optional[str]) -> str:
        """Open rear trunk on vehicle."""
        spec = self._require_vehicle_spec_with_vin(vehicle_name)

        self._api_post(spec.vin, "command/activate_rear_trunk")
        return f"Trunk opened for {spec.label()}."

    def navigate_to(self, vehicle_name: Optional[str], destination: str) -> str:
        """Send a destination (an address or coordinates) to the vehicle's navigation."""
        destination = (destination or "").strip()
        if not destination:
            raise TeslaCommandError("No destination given.")
        spec = self._require_vehicle_spec_with_vin(vehicle_name)
        locale = self._config.get("navigation_locale") or self.DEFAULT_NAVIGATION_LOCALE

        info(f"Tesla: sending navigation destination '{destination}' to {spec.label()} (locale {locale})")
        self._api_post(spec.vin, "command/share", query={"value": destination, "locale": locale})
        return f"Navigation to {destination} sent to {spec.label()}."


TeslaAPI = TeslaIntegration
