from unittest.mock import Mock

import pytest
import requests

from assistant.integration.VehicleProvider import Fleet, VehicleSpec, VehicleState
from integrations.tesla.TeslaIntegration import (
    TeslaIntegration,
    TeslaAPIError,
    TeslaCommandError,
    TeslaConfigurationError,
)


class DummyResponse:
    def __init__(self, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json = json_data or {}
        self.text = text or ""

    def json(self):
        return self._json


class FakeSession:
    def __init__(self):
        self.headers = {}
        self._responses = {}
        self.calls = []

    def queue_response(self, method, url, response):
        self._responses.setdefault((method.upper(), url), []).append(response)

    def _next_response(self, method, url):
        key = (method.upper(), url)
        if key not in self._responses or not self._responses[key]:
            raise AssertionError(f"No queued response for {method} {url}")
        response = self._responses[key].pop(0)
        return response() if callable(response) else response

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        return self._next_response("GET", url)

    def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs))
        return self._next_response("POST", url)


def build_default_config():
    return {
        "tessie_access_token": "test_token",
        "vehicles": [
            {
                "name": "family",
                "vin": "VIN123",
                "display_name": "Floxy",
                "default": True,
            }
        ],
    }




def family(api):
    """The default car's spec, as the fleet would hand it to the provider."""
    return api.vehicles()[0]


@pytest.mark.unit
class TestVehicles:
    def test_the_entry_becomes_a_spec_with_the_name_as_id_and_the_vin_as_handle(self):
        spec = family(TeslaIntegration(config=build_default_config()))
        assert spec == VehicleSpec(id="family", label="Floxy", handle="VIN123", aliases=("Floxy", "VIN123"), is_default=True)

    def test_without_a_display_name_the_name_is_the_label(self):
        api = TeslaIntegration(config={"tessie_access_token": "tok", "vehicles": [{"name": "work", "vin": "VIN9"}]})
        spec = family(api)
        assert spec.label == "work" and spec.aliases == ("VIN9",) and not spec.is_default

    def test_an_entry_that_is_not_a_mapping_is_skipped(self):
        api = TeslaIntegration(config={"tessie_access_token": "tok", "vehicles": ["oops", {"name": "a", "vin": "V"}]})
        assert [s.id for s in api.vehicles()] == ["a"]

    def test_only_invalid_entries_is_a_configuration_error(self):
        api = TeslaIntegration(config={"tessie_access_token": "tok", "vehicles": ["oops"]})
        with pytest.raises(TeslaConfigurationError):
            api.vehicles()


@pytest.mark.unit
def test_set_max_defrost_success():
    session = FakeSession()
    session.queue_response(
        "POST",
        f"{TeslaIntegration.TESSIE_API_BASE}/VIN123/command/start_max_defrost",
        DummyResponse(200, {"result": True}),
    )

    api = TeslaIntegration(config=build_default_config(), session=session)
    result = api.max_defrost(family(api))

    assert "Max defrost enabled" in result and "Floxy" in result

    post_calls = [call for call in session.calls if call[0] == "POST"]
    assert len(post_calls) == 1
    assert post_calls[0][1].endswith("/command/start_max_defrost")


@pytest.mark.unit
def test_start_air_conditioning_failure():
    session = FakeSession()
    session.queue_response(
        "POST",
        f"{TeslaIntegration.TESSIE_API_BASE}/VIN123/command/start_climate",
        DummyResponse(200, {"result": False, "reason": "vehicle offline"}),
    )

    api = TeslaIntegration(config=build_default_config(), session=session)

    with pytest.raises(TeslaCommandError, match="vehicle offline"):
        api.climate_on(family(api))


@pytest.mark.unit
def test_missing_tessie_token():
    with pytest.raises(TeslaConfigurationError, match="tessie_access_token"):
        TeslaIntegration(config={"vehicles": [{"name": "test", "vin": "VIN123"}]})


@pytest.mark.unit
class TestVehicleState:
    def _api(self, state):
        session = FakeSession()
        session.queue_response("GET", f"{TeslaIntegration.TESSIE_API_BASE}/VIN123/state", DummyResponse(200, state))
        return TeslaIntegration(config=build_default_config(), session=session)

    def test_tessies_state_is_folded_into_the_contract(self):
        api = self._api({
            "state": "online",
            "drive_state": {"latitude": 57.5, "longitude": 12.0, "speed": 80, "shift_state": "D",
                            "active_route_minutes_to_arrival": 12.7, "active_route_destination": "IKEA",
                            "timestamp": 1_700_000_000_000},
            "charge_state": {"usable_battery_level": 60},
            "climate_state": {"inside_temp": 21.5},
        })
        state = api.vehicle_state(family(api))
        assert state == VehicleState(connectivity="online", in_gear=True, speed_kmh=80, latitude=57.5, longitude=12.0,
                                     battery_percent=60, cabin_temp_c=21.5, destination="IKEA", minutes_to_arrival=12.7)
        assert state.is_driving

    def test_an_asleep_car_carries_the_snapshots_time_and_a_stopped_one_is_not_driving(self):
        api = self._api({
            "state": "asleep",
            "drive_state": {"latitude": 57.5, "longitude": 12.0, "speed": None, "shift_state": "P",
                            "timestamp": 1_700_000_000_000},
            "charge_state": {"usable_battery_level": 59},
            "climate_state": {},
        })
        state = api.vehicle_state(family(api))
        assert state.data_time == 1_700_000_000 and state.connectivity == "asleep"
        assert not state.in_gear and not state.is_driving and state.cabin_temp_c is None

    def test_an_online_car_has_no_data_time_even_with_a_timestamp(self):
        api = self._api({"state": "online", "drive_state": {"timestamp": 1_700_000_000_000}, "charge_state": {}})
        assert api.vehicle_state(family(api)).data_time is None

    def test_missing_blocks_leave_every_field_unset(self):
        state = self._api({}).vehicle_state(family(self._api({})))
        assert state == VehicleState()
@pytest.mark.unit
def test_api_get_non_200_raises():
    session = FakeSession()
    session.queue_response("GET", f"{TeslaIntegration.TESSIE_API_BASE}/VIN123/state",
                           DummyResponse(500, text="Internal Server Error"))
    api = TeslaIntegration(config=build_default_config(), session=session)
    with pytest.raises(TeslaAPIError, match="500"):
        api._api_get("VIN123", "state")


@pytest.mark.unit
def test_api_get_timeout_then_success():
    session = FakeSession()
    url = f"{TeslaIntegration.TESSIE_API_BASE}/VIN123/state"
    session.queue_response("GET", url, lambda: (_ for _ in ()).throw(requests.exceptions.Timeout("timed out")))
    session.queue_response("GET", url, DummyResponse(200, {"drive_state": {}}))
    api = TeslaIntegration(config=build_default_config(), session=session)
    result = api._api_get("VIN123", "state")
    assert result == {"drive_state": {}}


@pytest.mark.unit
def test_api_get_all_retries_exhausted():
    session = FakeSession()
    url = f"{TeslaIntegration.TESSIE_API_BASE}/VIN123/state"
    for _ in range(3):
        session.queue_response("GET", url, lambda: (_ for _ in ()).throw(requests.exceptions.Timeout("timed out")))
    api = TeslaIntegration(config=build_default_config(), session=session)
    with pytest.raises(TeslaAPIError, match="failed after 3 retries"):
        api._api_get("VIN123", "state")


@pytest.mark.unit
def test_start_air_conditioning_success():
    session = FakeSession()
    session.queue_response("POST", f"{TeslaIntegration.TESSIE_API_BASE}/VIN123/command/start_climate",
                           DummyResponse(200, {"result": True}))
    api = TeslaIntegration(config=build_default_config(), session=session)
    result = api.climate_on(family(api))
    assert "Air conditioning started" in result


@pytest.mark.unit
def test_stop_air_conditioning_success():
    session = FakeSession()
    session.queue_response("POST", f"{TeslaIntegration.TESSIE_API_BASE}/VIN123/command/stop_climate",
                           DummyResponse(200, {"result": True}))
    api = TeslaIntegration(config=build_default_config(), session=session)
    result = api.climate_off(family(api))
    assert "Air conditioning stopped" in result


@pytest.mark.unit
def test_open_trunk_success():
    session = FakeSession()
    session.queue_response("POST", f"{TeslaIntegration.TESSIE_API_BASE}/VIN123/command/activate_rear_trunk",
                           DummyResponse(200, {"result": True}))
    api = TeslaIntegration(config=build_default_config(), session=session)
    result = api.open_trunk(family(api))
    assert "Trunk opened" in result

    post_calls = [call for call in session.calls if call[0] == "POST"]
    assert len(post_calls) == 1
    assert post_calls[0][1].endswith("/command/activate_rear_trunk")


@pytest.mark.unit
def test_api_post_merges_extra_body_params():
    session = FakeSession()
    session.queue_response("POST", f"{TeslaIntegration.TESSIE_API_BASE}/VIN123/command/test_cmd",
                           DummyResponse(200, {"result": True}))
    api = TeslaIntegration(config=build_default_config(), session=session)
    api._api_post("VIN123", "command/test_cmd", extra_param="value")

    post_calls = [call for call in session.calls if call[0] == "POST"]
    body = post_calls[0][2]["json"]
    assert body["retry"] is True
    assert body["extra_param"] == "value"


@pytest.mark.unit
def test_a_car_without_a_vin_is_listed_but_cannot_be_driven():
    config = {
        "tessie_access_token": "tok",
        "vehicles": [{"name": "novin", "display_name": "No VIN Car"}],
    }
    session = FakeSession()
    api = TeslaIntegration(config=config, session=session)
    spec = family(api)
    assert spec.handle is None and spec.label == "No VIN Car"
    with pytest.raises(TeslaConfigurationError, match="missing VIN"):
        api.open_trunk(spec)
    with pytest.raises(TeslaConfigurationError, match="missing VIN"):
        api.vehicle_state(spec)
    assert session.calls == []


@pytest.mark.unit
def test_navigate_to_shares_the_destination_on_the_query_string():
    """Tessie's share command takes `value` and `locale` as URL parameters, not in the body."""
    session = FakeSession()
    session.queue_response("POST", f"{TeslaIntegration.TESSIE_API_BASE}/VIN123/command/share",
                           DummyResponse(200, {"result": True}))
    config = build_default_config()
    config["navigation_locale"] = "sv-SE"
    api = TeslaIntegration(config=config, session=session)

    result = api.navigate_to(family(api), "Storgatan 1, Stockholm")

    assert "Storgatan 1, Stockholm" in result and "Floxy" in result
    post_calls = [call for call in session.calls if call[0] == "POST"]
    assert len(post_calls) == 1
    assert post_calls[0][2]["params"] == {"value": "Storgatan 1, Stockholm", "locale": "sv-SE"}
    assert "value" not in post_calls[0][2]["json"]


@pytest.mark.unit
def test_navigate_to_without_a_configured_locale_falls_back_to_the_default():
    session = FakeSession()
    session.queue_response("POST", f"{TeslaIntegration.TESSIE_API_BASE}/VIN123/command/share",
                           DummyResponse(200, {"result": True}))
    api = TeslaIntegration(config=build_default_config(), session=session)

    api.navigate_to(family(api), "59.3293,18.0686")

    assert session.calls[0][2]["params"]["locale"] == TeslaIntegration.DEFAULT_NAVIGATION_LOCALE


@pytest.mark.unit
def test_navigate_to_refuses_a_blank_destination_without_a_request():
    session = FakeSession()
    api = TeslaIntegration(config=build_default_config(), session=session)
    with pytest.raises(TeslaCommandError):
        api.navigate_to(family(api), "   ")
    assert session.calls == []


@pytest.mark.unit
def test_navigate_to_reports_the_cars_refusal():
    session = FakeSession()
    session.queue_response("POST", f"{TeslaIntegration.TESSIE_API_BASE}/VIN123/command/share",
                           DummyResponse(200, {"result": False, "reason": "vehicle unavailable"}))
    api = TeslaIntegration(config=build_default_config(), session=session)
    with pytest.raises(TeslaCommandError, match="vehicle unavailable"):
        api.navigate_to(family(api), "Storgatan 1")


@pytest.mark.unit
class TestHealthAndConfig:
    def _api(self, state):
        session = FakeSession()
        session.queue_response("GET", f"{TeslaIntegration.TESSIE_API_BASE}/VIN123/state", DummyResponse(200, state))
        return TeslaIntegration(config=build_default_config(), session=session)

    def test_health_is_a_cabin_temperature_in_the_state(self):
        assert self._api({"climate_state": {"inside_temp": 20}}).check_health() is True
        assert self._api({"climate_state": {}}).check_health() is False

    def test_health_of_a_car_without_a_vin_is_false_without_a_request(self):
        session = FakeSession()
        api = TeslaIntegration(config={"tessie_access_token": "tok", "vehicles": [{"name": "novin"}]}, session=session)
        assert api.check_health() is False and session.calls == []

    def test_health_gives_a_slow_api_one_attempt_not_the_commands_three(self):
        session = FakeSession()
        url = f"{TeslaIntegration.TESSIE_API_BASE}/VIN123/state"
        session.queue_response("GET", url, lambda: (_ for _ in ()).throw(requests.exceptions.Timeout("timed out")))
        session.queue_response("GET", url, DummyResponse(200, {"climate_state": {"inside_temp": 20}}))
        api = TeslaIntegration(config=build_default_config(), session=session)
        with pytest.raises(TeslaAPIError):
            api.check_health()
        assert len(session.calls) == 1

    def test_an_empty_vehicle_list_or_a_missing_section_is_refused_at_construction(self):
        with pytest.raises(TeslaConfigurationError, match="empty"):
            TeslaIntegration(config={"tessie_access_token": "tok", "vehicles": []})
        manager = Mock()
        manager.get.side_effect = KeyError("tesla")
        with pytest.raises(TeslaConfigurationError, match="tesla"):
            TeslaIntegration(config_manager=manager)

    def test_a_car_known_only_by_its_vin_is_addressed_by_it(self):
        api = TeslaIntegration(config={"tessie_access_token": "tok", "vehicles": [{"vin": "VIN9"}]})
        spec = family(api)
        assert (spec.id, spec.label) == ("VIN9", "VIN9")
        assert Fleet([api]).resolve("vin9")[1].handle == "VIN9"
