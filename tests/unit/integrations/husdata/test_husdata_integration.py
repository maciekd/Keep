from unittest.mock import patch

import pytest
import requests

from integrations.husdata.HusdataIntegration import (
    HOT_WATER_SETPOINT, ROOM_TARGET, HusdataError, HusdataIntegration, decode, to_wire)

# A ground-source pump's typical report: tenths for the scaled types, a wrapped negative, an
# unavailable sensor, hours as they are.
ALLDATA = {"0007": 34, "0008": 212, "0203": 210, "0009": 485, "0002": 341, "0001": 298,
           "0005": 65532, "0011": 32768, "1A01": 1, "1A04": 1, "1A02": 0, "3108": 550, "6F63": 12345, "2A20": 0}


def _gateway(settings: dict):
    """A HusdataIntegration over *settings*, whose HTTP layer is a dict of path → reply recording every call."""
    with patch("integrations.husdata.HusdataIntegration.config") as cfg:
        cfg.get.side_effect = lambda key, default=None: settings.get(key, default)
        integration = HusdataIntegration()
        integration.replies = {"/api/alldata": dict(ALLDATA), "/api/set": {"response": "Ok"},
                               "/api/status": {"status": {"hpconn": 1}}}
        integration.calls = []

        def fake_get(path, params=None):
            integration.calls.append((path, params))
            return integration.replies[path]

        integration._get = fake_get
        yield integration


@pytest.fixture
def gateway():
    yield from _gateway({"husdata.host": "gateway.local", "husdata.room_target_range": [15, 25],
                         "husdata.hot_water_range": [40, 60]})


@pytest.fixture
def bare_gateway():
    """The same gateway with no range configured: the code's own defaults apply."""
    yield from _gateway({"husdata.host": "gateway.local"})


class TestDecode:
    def test_temperature_is_tenths_of_a_degree(self):
        assert (decode("0007", 34).value, decode("0007", 34).unit) == (3.4, "°C")

    def test_a_wrapped_negative_is_signed(self):
        assert decode("0005", 65532).value == -0.4

    def test_hours_are_not_scaled(self):
        assert decode("6F63", 12345).value == 12345

    def test_the_unavailable_marker_is_no_reading(self):
        assert decode("0011", 32768) is None

    def test_wire_value_is_tenths(self):
        assert to_wire(21.5) == 215


class TestRefresh:
    def test_ids_are_upper_cased_and_a_non_number_is_skipped(self, gateway):
        gateway.replies["/api/alldata"] = {"000a": 480, "0007": "n/a", "0008": 212}
        registers = gateway.refresh()
        assert set(registers) == {"000A", "0008"}
        assert gateway.reading("000A", "Hot water mid").value == 48.0
        assert gateway.reading("0008", "Indoor").value == 21.2
        assert gateway.reading("0007", "Outdoor") is None

    def test_the_alarm_number_outranks_the_flag(self, gateway):
        gateway.replies["/api/alldata"].update({"2A20": 0, "1A20": 1})
        gateway.refresh()
        assert "active" not in gateway._alarm_line()

    def test_a_pump_without_an_alarm_register_says_so(self, gateway):
        del gateway.replies["/api/alldata"]["2A20"]
        gateway.refresh()
        assert "not reported" in gateway._alarm_line()


class TestSnapshot:
    def test_groups_and_names_what_the_pump_reports(self, gateway):
        text = gateway.snapshot()
        lines = text.split("\n")
        assert "Outdoor 3.4 °C" in lines[0] and "Brine in -0.4 °C" in lines[0] and "Room target 21.0 °C" in lines[0]
        assert "Compressor on" in lines[1] and "Additional heat step 1 off" in lines[1]
        assert "Compressor speed 55.0 %" in lines[2] and "Compressor hours heating 12345 h" in lines[2]
        assert lines[-1] == "Alarm: none"

    def test_an_unreported_or_unavailable_register_is_left_out(self, gateway):
        text = gateway.snapshot()
        assert "Pool" not in text and "Hot water setpoint" not in text and "Fan" not in text

    def test_an_alarm_code_is_called_out(self, gateway):
        gateway.replies["/api/alldata"]["2A20"] = 7
        assert "ALARM active (code 7)" in gateway.snapshot()

    def test_the_flag_style_alarm_register_counts_too(self, gateway):
        del gateway.replies["/api/alldata"]["2A20"]
        gateway.replies["/api/alldata"]["1A20"] = 1
        assert "ALARM active" in gateway.snapshot()

    def test_an_empty_report_is_an_error_not_an_empty_section(self, gateway):
        gateway.replies["/api/alldata"] = {}
        with pytest.raises(HusdataError, match="no registers"):
            gateway.snapshot()


class TestWrites:
    def test_room_target_goes_out_in_tenths(self, gateway):
        reply = gateway.set_room_target(21.5)
        assert ("/api/set", {"idx": ROOM_TARGET, "val": 215}) in gateway.calls
        assert "21.5" in reply

    def test_the_cached_reading_follows_the_write(self, gateway):
        gateway.set_room_target(22)
        assert gateway.reading(ROOM_TARGET, "Room target").value == 22.0

    def test_a_value_outside_the_configured_range_never_reaches_the_gateway(self, gateway):
        with pytest.raises(HusdataError, match="15–25"):
            gateway.set_room_target(28)
        assert not any(path == "/api/set" for path, _ in gateway.calls)

    def test_a_setting_the_pump_does_not_report_is_refused(self, gateway):
        with pytest.raises(HusdataError, match="does not expose"):
            gateway.set_hot_water(50)
        assert not any(path == "/api/set" for path, _ in gateway.calls)

    def test_hot_water_writes_its_register_once_the_pump_reports_it(self, gateway):
        gateway.replies["/api/alldata"][HOT_WATER_SETPOINT] = 500
        gateway.set_hot_water(55)
        assert ("/api/set", {"idx": HOT_WATER_SETPOINT, "val": 550}) in gateway.calls

    def test_a_refusal_names_the_api_enabled_setting(self, gateway):
        gateway.replies["/api/set"] = {"response": "Error wrong format"}
        with pytest.raises(HusdataError, match="API_ENABLED"):
            gateway.set_room_target(21)


class TestDefaultRanges:
    def test_the_default_room_range_is_15_to_25_inclusive(self, bare_gateway):
        for celsius in (15.0, 25.0):
            bare_gateway.set_room_target(celsius)
        assert [p["val"] for path, p in bare_gateway.calls if path == "/api/set"] == [150, 250]
        with pytest.raises(HusdataError, match="15"):
            bare_gateway.set_room_target(14.9)
        assert sum(1 for path, _ in bare_gateway.calls if path == "/api/set") == 2

    def test_the_default_hot_water_range_tops_at_60(self, bare_gateway):
        with pytest.raises(HusdataError, match="60"):
            bare_gateway.set_hot_water(65)
        assert not any(path == "/api/set" for path, _ in bare_gateway.calls)


class TestHealth:
    def test_a_gateway_without_its_pump_is_unhealthy(self, gateway):
        gateway.replies["/api/status"] = {"status": {"hpconn": 0}}
        with pytest.raises(HusdataError, match="no connection to the heat pump"):
            gateway.check_health()

    def test_a_connected_pump_is_healthy(self, gateway):
        assert gateway.check_health() is True


class TestTransport:
    @pytest.fixture
    def integration(self):
        with patch("integrations.husdata.HusdataIntegration.config") as cfg:
            cfg.get.return_value = "gateway.local"
            yield HusdataIntegration()

    def test_a_non_json_answer_hints_at_the_api_switch(self, integration):
        with patch("integrations.husdata.HusdataIntegration.requests.get") as get:
            get.return_value.json.side_effect = ValueError("not json")
            with pytest.raises(HusdataError, match="API_ENABLED"):
                integration._get("/api/alldata")

    def test_a_slow_gateway_is_said_to_not_answer_in_time(self, integration):
        with patch("integrations.husdata.HusdataIntegration.requests.get", side_effect=requests.ReadTimeout()):
            with pytest.raises(HusdataError, match="in time"):
                integration._get("/api/alldata")

    def test_an_unreachable_gateway_names_the_host(self, integration):
        with patch("integrations.husdata.HusdataIntegration.requests.get", side_effect=requests.ConnectionError()):
            with pytest.raises(HusdataError, match="gateway.local"):
                integration._get("/api/alldata")
