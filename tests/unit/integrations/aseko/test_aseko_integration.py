"""AsekoIntegration: the calls to Aseko Cloud and the rendering of a unit for the prompt. No
request leaves the test: the session is a stub answering canned bodies by path."""
from unittest.mock import Mock, patch

import pytest

from integrations.aseko.AsekoIntegration import (AsekoError, AsekoIntegration, readings, render_unit,
                                                 render_units)

UNIT = {
    "name": "Garden pool", "serialNumber": "AQ-0001", "online": True,
    "statusValues": {
        "waterTemperature": 27.4, "waterTemperatureRequired": 28.0, "airTemperature": 19,
        "ph": 7.21, "phRequired": 7.2, "redox": 712, "redoxRequired": 700,
        "clFree": 0.62, "clFreeRequired": 0.6, "clFreeRequiredUnit": "mg/l", "clBounded": None,
        "salinity": 3.1, "electrolyzerRunning": True, "electrodePower": 24, "electrolyzerDirection": "LEFT",
        "filtrationRunning": True, "filtrationSpeed": "MEDIUM",
        "upcomingFiltrationPeriod": {"isNonstop": False, "isNext": False, "start": "06:00", "end": "12:00"},
        "filterPressure": 0.8, "filterFlowSpeed": 11, "heatingRunning": False,
        "solarRunning": False, "solarTemperature": 31, "waterFlowToProbes": True,
        "waterLevel": 12, "waterLevelState": "OK", "waterFillingRunning": False, "mode": "AUTO",
    },
    "statusMessages": [],
}


def _response(status: int, body=None, reason="") -> Mock:
    response = Mock(status_code=status, reason=reason)
    if body is None:
        response.json.side_effect = ValueError("no body")
    else:
        response.json.return_value = body
    return response


def _api(answers: dict) -> tuple[AsekoIntegration, Mock]:
    """An integration whose session answers each path from `answers` ({path suffix: (status, body)})."""
    session = Mock()

    def get(url, headers=None, params=None, timeout=None):
        for suffix, (status, body) in answers.items():
            if url.endswith(suffix):
                return _response(status, body)
        raise AssertionError(f"unexpected call {url}")

    session.get.side_effect = get
    AsekoIntegration.reset_instance()
    with patch("integrations.aseko.AsekoIntegration.config") as cfg:
        cfg.get.return_value = "key-1234"
        api = AsekoIntegration(session=session)
    return api, session


@pytest.mark.unit
class TestCalls:
    def test_every_call_carries_the_headers_the_api_contract_demands(self):
        """Contract with api.aseko.cloud: a bearer key plus the two client headers, or 400/401."""
        api, session = _api({"/paired-units": (200, {"items": [], "totalItems": 0})})
        with patch("integrations.aseko.AsekoIntegration.config") as cfg, \
                patch("integrations.aseko.AsekoIntegration.version", return_value="v2026.10.7"):
            cfg.get.return_value = "key-1234"
            api.fetch_units()
        headers = session.get.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer key-1234"
        assert headers["X-Client-Name"] and headers["X-Client-Version"] == "v2026.10.7"

    def test_fetch_units_reads_the_detail_of_every_paired_unit_and_keeps_it(self):
        detail = dict(UNIT)
        api, session = _api({
            "/paired-units": (200, {"items": [{"serialNumber": "AQ-0001", "name": "Garden pool"}], "totalItems": 1}),
            "/paired-units/AQ-0001": (200, detail),
        })
        with patch("integrations.aseko.AsekoIntegration.config"):
            units = api.fetch_units()
        assert units == [detail] and api.last_units == [detail]
        assert session.get.call_count == 2

    def test_health_is_the_unit_list_answering(self):
        api, _ = _api({"/paired-units": (200, {"items": [], "totalItems": 0})})
        with patch("integrations.aseko.AsekoIntegration.config"):
            assert api.check_health() is True

    def test_unaccepted_terms_name_the_account_page_to_fix_it_on(self):
        api, _ = _api({"/paired-units": (403, {"error": "Terms of services are not accepted.",
                                               "errorType": "TOS_NOT_ACCEPTED", "statusCode": 403})})
        with patch("integrations.aseko.AsekoIntegration.config"), pytest.raises(AsekoError) as exc:
            api.fetch_units()
        assert "account.aseko.cloud" in str(exc.value) and "terms" in str(exc.value).lower()

    def test_a_bad_key_names_the_config_key_to_change(self):
        api, _ = _api({"/paired-units": (401, {"error": "Invalid key", "errorType": "API_KEY_INVALID"})})
        with patch("integrations.aseko.AsekoIntegration.config"), pytest.raises(AsekoError) as exc:
            api.fetch_units()
        assert "aseko.api_key" in str(exc.value)

    def test_a_refusal_without_a_json_body_still_says_the_status(self):
        api, _ = _api({"/paired-units": (500, None)})
        with patch("integrations.aseko.AsekoIntegration.config"), pytest.raises(AsekoError) as exc:
            api.fetch_units()
        assert "500" in str(exc.value)

    def test_the_rate_limit_is_its_own_message(self):
        api, _ = _api({"/paired-units": (429, None)})
        with patch("integrations.aseko.AsekoIntegration.config"), pytest.raises(AsekoError) as exc:
            api.fetch_units()
        assert "rate limit" in str(exc.value)


@pytest.mark.unit
class TestReadings:
    def test_values_render_with_their_targets_and_without_a_trailing_zero(self):
        got = dict(readings(UNIT["statusValues"]))
        assert got["Water"] == "27.4°C (target 28°C)"
        assert got["pH"] == "7.21 (target 7.2)"
        assert got["Free chlorine"] == "0.62 mg/l (target 0.6 mg/l)"
        assert got["Redox"] == "712 mV (target 700 mV)"

    def test_a_null_or_absent_sensor_is_left_out(self):
        labels = [label for label, _ in readings(UNIT["statusValues"])]
        assert "Bound chlorine" not in labels      # null
        assert "Pool flow" not in labels           # absent: the unit has no flow control
        assert "Probes" not in labels              # water flows to the probes: nothing to say

    def test_the_order_is_fixed_water_first_mode_last(self):
        labels = [label for label, _ in readings(UNIT["statusValues"])]
        assert labels[0] == "Water" and labels[-1] == "Mode"
        assert labels.index("pH") < labels.index("Filtration")

    def test_running_equipment_carries_its_detail_and_stopped_equipment_does_not(self):
        got = dict(readings(UNIT["statusValues"]))
        assert got["Electrolyzer"] == "running, 24 g/h, left"
        assert got["Filtration"] == "running at medium, period 06:00–12:00"
        stopped = dict(UNIT["statusValues"], electrolyzerRunning=False, filtrationRunning=False,
                       upcomingFiltrationPeriod={"isNonstop": False, "isNext": True, "start": "06:00", "end": "12:00"})
        got = dict(readings(stopped))
        assert got["Electrolyzer"] == "off"
        assert got["Filtration"] == "off, next 06:00–12:00"

    def test_nonstop_filtration_has_no_window(self):
        got = dict(readings({"filtrationRunning": True, "upcomingFiltrationPeriod": {"isNonstop": True}}))
        assert got["Filtration"] == "running, nonstop"

    def test_a_level_alarm_and_dry_probes_are_said_so(self):
        got = dict(readings({"waterLevelState": "LOW", "waterLevel": 3, "waterFillingRunning": True,
                             "waterFlowToProbes": False}))
        assert got["Water level"] == "LOW (alarm), 3 cm, filling"
        assert "cannot be taken" in got["Probes"]


@pytest.mark.unit
class TestRenderUnit:
    def test_a_healthy_unit_lists_its_readings_and_says_there_are_no_messages(self):
        text = render_unit(UNIT)
        assert text.startswith("Garden pool: online")
        assert "  Water: 27.4°C (target 28°C)" in text
        assert "Unit messages: none" in text

    def test_messages_are_listed_errors_first_with_their_detail(self):
        unit = dict(UNIT, statusMessages=[
            {"type": "START_BY_SOLAR", "severity": "INFO", "message": "Started by solar", "detail": ""},
            {"type": "PUMP_OVERHEATED", "severity": "ERROR", "message": "Pump overheated", "detail": "Check the pump"},
        ])
        text = render_unit(unit)
        error_at, info_at = text.index("ERROR: Pump overheated — Check the pump"), text.index("INFO: Started by solar")
        assert error_at < info_at
        assert "Unit messages: none" not in text

    def test_filter_and_solar_parts_join_without_a_dangling_comma(self):
        text = render_unit(dict(UNIT, statusValues={"filterFlowSpeed": 11, "solarTemperature": 31}))
        assert "  Filter: 11 m³/h" in text and "  Solar: 31°C" in text
        both = render_unit(dict(UNIT, statusValues={"filterPressure": 0.8, "filterFlowSpeed": 11}))
        assert "  Filter: 0.8 bar, 11 m³/h" in both

    def test_a_message_without_text_or_severity_shows_its_type_as_info(self):
        text = render_unit(dict(UNIT, statusMessages=[{"type": "START_BY_SOLAR"}]))
        assert "INFO: START_BY_SOLAR" in text

    def test_an_offline_unit_flags_its_readings_as_the_last_sent(self):
        text = render_unit(dict(UNIT, online=False))
        assert "OFFLINE" in text.splitlines()[0] and "Water: 27.4°C" in text

    def test_a_unit_that_never_connected_has_no_readings(self):
        text = render_unit({"name": "New unit", "serialNumber": "AQ-9", "online": False})
        assert "never connected" in text and "Water" not in text

    def test_a_unit_without_a_name_goes_by_its_serial(self):
        assert render_unit({"serialNumber": "AQ-9", "statusValues": {}}).startswith("AQ-9:")

    def test_no_units_is_said_and_several_are_parted_by_a_blank_line(self):
        assert "No pool unit" in render_units([])
        two = render_units([UNIT, dict(UNIT, name="Spa")])
        assert "\n\nSpa: online" in two
