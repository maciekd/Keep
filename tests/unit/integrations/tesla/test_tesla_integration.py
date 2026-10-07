import pytest
import requests

from integrations.tesla.TeslaIntegration import (
    TeslaIntegration,
    TeslaAPIError,
    TeslaCommandError,
    TeslaConfigurationError,
    TeslaVehicleSpec,
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


@pytest.mark.unit
def test_set_max_defrost_success():
    session = FakeSession()
    session.queue_response(
        "POST",
        f"{TeslaIntegration.TESSIE_API_BASE}/VIN123/command/start_max_defrost",
        DummyResponse(200, {"result": True}),
    )

    api = TeslaIntegration(config=build_default_config(), session=session)
    result = api.set_max_defrost("family")

    assert "Max defrost enabled" in result

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
        api.start_air_conditioning("family")


@pytest.mark.unit
def test_missing_tessie_token():
    with pytest.raises(TeslaConfigurationError, match="tessie_access_token"):
        TeslaIntegration(config={"vehicles": [{"name": "test", "vin": "VIN123"}]})


@pytest.mark.unit
class TestTeslaVehicleSpec:
    def test_matches_by_name(self):
        spec = TeslaVehicleSpec(name="family", vin="VIN1", display_name="Floxy")
        assert spec.matches("family") is True

    def test_matches_by_display_name(self):
        spec = TeslaVehicleSpec(name="family", vin="VIN1", display_name="Floxy")
        assert spec.matches("Floxy") is True

    def test_matches_by_vin_case_insensitive(self):
        spec = TeslaVehicleSpec(name="family", vin="VIN123", display_name="Floxy")
        assert spec.matches("vin123") is True

    def test_matches_returns_false_for_empty(self):
        spec = TeslaVehicleSpec(name="family", vin="VIN1", display_name="Floxy")
        assert spec.matches("") is False

    def test_label_fallback_display_name(self):
        spec = TeslaVehicleSpec(name="family", vin="VIN1", display_name="Floxy")
        assert spec.label() == "Floxy"

    def test_label_fallback_name(self):
        spec = TeslaVehicleSpec(name="family", vin="VIN1", display_name=None)
        assert spec.label() == "family"

    def test_label_fallback_vin(self):
        spec = TeslaVehicleSpec(name=None, vin="VIN1", display_name=None)
        assert spec.label() == "VIN1"

    def test_label_fallback_car(self):
        spec = TeslaVehicleSpec(name=None, vin=None, display_name=None)
        assert spec.label() == "car"


@pytest.mark.unit
def test_select_vehicle_spec_no_match_raises():
    api = TeslaIntegration(config=build_default_config())
    with pytest.raises(TeslaConfigurationError, match="No Tesla vehicle matches"):
        api._select_vehicle_spec("nonexistent")


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
    result = api.start_air_conditioning("family")
    assert "Air conditioning started" in result


@pytest.mark.unit
def test_stop_air_conditioning_success():
    session = FakeSession()
    session.queue_response("POST", f"{TeslaIntegration.TESSIE_API_BASE}/VIN123/command/stop_climate",
                           DummyResponse(200, {"result": True}))
    api = TeslaIntegration(config=build_default_config(), session=session)
    result = api.stop_air_conditioning("family")
    assert "Air conditioning stopped" in result


@pytest.mark.unit
def test_open_trunk_success():
    session = FakeSession()
    session.queue_response("POST", f"{TeslaIntegration.TESSIE_API_BASE}/VIN123/command/activate_rear_trunk",
                           DummyResponse(200, {"result": True}))
    api = TeslaIntegration(config=build_default_config(), session=session)
    result = api.open_trunk("family")
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
def test_require_vehicle_spec_with_vin_raises_on_missing_vin():
    config = {
        "tessie_access_token": "tok",
        "vehicles": [{"name": "novin", "display_name": "No VIN Car"}],
    }
    api = TeslaIntegration(config=config)
    with pytest.raises(TeslaConfigurationError, match="missing VIN"):
        api._require_vehicle_spec_with_vin("novin")


@pytest.mark.unit
def test_navigate_to_shares_the_destination_on_the_query_string():
    """Tessie's share command takes `value` and `locale` as URL parameters, not in the body."""
    session = FakeSession()
    session.queue_response("POST", f"{TeslaIntegration.TESSIE_API_BASE}/VIN123/command/share",
                           DummyResponse(200, {"result": True}))
    config = build_default_config()
    config["navigation_locale"] = "sv-SE"
    api = TeslaIntegration(config=config, session=session)

    result = api.navigate_to("family", "Storgatan 1, Stockholm")

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

    api.navigate_to(None, "59.3293,18.0686")

    assert session.calls[0][2]["params"]["locale"] == TeslaIntegration.DEFAULT_NAVIGATION_LOCALE


@pytest.mark.unit
def test_navigate_to_refuses_a_blank_destination_without_a_request():
    session = FakeSession()
    api = TeslaIntegration(config=build_default_config(), session=session)
    with pytest.raises(TeslaCommandError):
        api.navigate_to("family", "   ")
    assert session.calls == []


@pytest.mark.unit
def test_navigate_to_reports_the_cars_refusal():
    session = FakeSession()
    session.queue_response("POST", f"{TeslaIntegration.TESSIE_API_BASE}/VIN123/command/share",
                           DummyResponse(200, {"result": False, "reason": "vehicle unavailable"}))
    api = TeslaIntegration(config=build_default_config(), session=session)
    with pytest.raises(TeslaCommandError, match="vehicle unavailable"):
        api.navigate_to("family", "Storgatan 1")
