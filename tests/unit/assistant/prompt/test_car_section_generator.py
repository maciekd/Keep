"""CarSectionGenerator: one line per car of every ready brand — status, place, route or
distance from home, cabin, charge, staleness — the hint without a provider, and the refresh
cadence that follows whether a car is moving."""
import time
from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from assistant.integration.VehicleProvider import VehicleSpec, VehicleState
from assistant.prompt.sections.CarSectionGenerator import CarSectionGenerator

MODULE = "assistant.prompt.sections.CarSectionGenerator"
HOME = (57.7089, 11.9746)
AWAY = (57.5, 12.0)
CAR = VehicleSpec("family", "Family Model Y", "VIN1")


def _section():
    with patch(f"{MODULE}.config") as config:
        config.get_home_latitude.return_value = HOME[0]
        config.get_home_longitude.return_value = HOME[1]
        return CarSectionGenerator(None)


def _provider(name, cars: dict):
    """A brand whose vehicle_state answers per spec id with a VehicleState or raises an Exception."""
    provider = Mock()
    provider.name = name
    provider.vehicles.return_value = [spec for spec, _ in cars.values()]
    states = {spec.id: state for spec, state in cars.values()}

    def state_of(spec):
        state = states[spec.id]
        if isinstance(state, Exception):
            raise state
        return state
    provider.vehicle_state.side_effect = state_of
    return provider


@pytest.fixture
def section():
    return _section()


@pytest.fixture
def no_lookups():
    with patch(f"{MODULE}.reverse_geocode", return_value="E6 Highway, Mellanstad"), \
            patch.object(CarSectionGenerator, "_get_eta_to_home", return_value=None):
        yield


@pytest.mark.unit
class TestRender:
    def test_parked_at_home(self, section):
        line = section._render(CAR, VehicleState(latitude=HOME[0], longitude=HOME[1], battery_percent=80))
        assert line.startswith("Family Model Y (vehicle_id: family): Parked Home")
        assert "(80% battery)" in line
        assert not section._any_car_driving

    def test_driving_away_shows_speed_and_the_address(self, section, no_lookups):
        line = section._render(CAR, VehicleState(latitude=AWAY[0], longitude=AWAY[1], speed_kmh=80, in_gear=True, battery_percent=60))
        assert "Driving 80 km/h E6 Highway, Mellanstad" in line
        assert section._any_car_driving

    def test_a_route_in_gear_shows_the_destination_and_a_rounded_eta(self, section, no_lookups):
        line = section._render(CAR, VehicleState(latitude=AWAY[0], longitude=AWAY[1], speed_kmh=0, in_gear=True,
                                                 destination="IKEA", minutes_to_arrival=12.7))
        assert "Navigating to IKEA, ETA 13 min" in line
        assert section._any_car_driving, "stopped at a red light is still driving"

    def test_a_route_left_in_a_parked_car_is_not_shown(self, section):
        line = section._render(CAR, VehicleState(latitude=HOME[0], longitude=HOME[1], destination="Work", minutes_to_arrival=10))
        assert "Parked Home" in line and "Navigating" not in line and "Work" not in line

    def test_away_without_a_route_shows_the_minutes_from_home(self, section):
        with patch(f"{MODULE}.reverse_geocode", return_value="Avenyn"), \
                patch.object(section, "_get_eta_to_home", return_value=25):
            line = section._render(CAR, VehicleState(latitude=AWAY[0], longitude=AWAY[1]))
        assert "Avenyn ~25 min from home" in line

    def test_without_a_position_the_location_is_unknown(self, section):
        assert "Location unknown" in section._render(CAR, VehicleState(battery_percent=50))

    def test_cabin_fuel_and_range(self, section):
        line = section._render(CAR, VehicleState(latitude=HOME[0], longitude=HOME[1], cabin_temp_c=22, fuel_percent=40, range_km=312.4))
        assert "Cabin 22°C" in line and "(40% fuel)" in line and "312 km range" in line
        assert "battery" not in line

    def test_no_charge_at_all_is_said(self, section):
        assert "Battery unknown" in section._render(CAR, VehicleState(latitude=HOME[0], longitude=HOME[1]))

    def test_an_asleep_car_carries_the_time_of_its_data(self, section):
        data_time = time.time() - 3600
        line = section._render(CAR, VehicleState(connectivity="asleep", latitude=HOME[0], longitude=HOME[1], data_time=data_time))
        assert f"[car is asleep, data from {datetime.fromtimestamp(data_time).strftime('%H:%M')}]" in line

    def test_an_offline_car_without_a_timestamp_is_flagged_stale(self, section):
        line = section._render(CAR, VehicleState(connectivity="offline", latitude=HOME[0], longitude=HOME[1]))
        assert "[car is offline, data may be stale]" in line

    def test_an_online_car_has_no_staleness_note(self, section):
        line = section._render(CAR, VehicleState(connectivity="online", latitude=HOME[0], longitude=HOME[1]))
        assert "online" not in line and "stale" not in line


@pytest.mark.unit
class TestFetchContent:
    def test_two_brands_make_one_list(self, section):
        tesla = _provider("Tesla", {"a": (CAR, VehicleState(latitude=HOME[0], longitude=HOME[1], battery_percent=80))})
        volvo = _provider("Volvo", {"b": (VehicleSpec("wagon", "The Volvo", "V1"), VehicleState(fuel_percent=30))})
        with patch(f"{MODULE}.active", return_value=[tesla, volvo]):
            text = section._fetch_content()
        lines = text.splitlines()
        assert lines[0].startswith("Family Model Y (vehicle_id: family): Parked Home")
        assert lines[1].startswith("The Volvo (vehicle_id: wagon): Parked Location unknown")
        assert section._ready

    def test_a_brand_that_fails_keeps_the_other_cars_lines(self, section):
        tesla = _provider("Tesla", {"a": (CAR, RuntimeError("API timeout"))})
        volvo = _provider("Volvo", {"b": (VehicleSpec("wagon", "The Volvo", "V1"), VehicleState(latitude=HOME[0], longitude=HOME[1]))})
        with patch(f"{MODULE}.active", return_value=[tesla, volvo]):
            text = section._fetch_content()
        assert "Family Model Y (vehicle_id: family): Error fetching data - API timeout" in text
        assert "The Volvo (vehicle_id: wagon): Parked Home" in text

    def test_without_a_ready_provider_the_section_is_the_hint(self, section):
        section._any_car_driving = True
        with patch(f"{MODULE}.active", return_value=[]), \
                patch(f"{MODULE}.no_provider_hint", return_value="Tesla is enabled but not configured: set tesla."):
            assert section._fetch_content() == "(Tesla is enabled but not configured: set tesla.)"
        assert not section._ready and not section._any_car_driving

    def test_a_provider_with_no_cars_says_so(self, section):
        with patch(f"{MODULE}.active", return_value=[_provider("Tesla", {})]):
            assert section._fetch_content() == "No car information available."


@pytest.mark.unit
class TestUpdateInterval:
    """Driving status is carried over from the fetch pass — the interval check makes no calls."""

    def _fetch(self, section, state):
        provider = _provider("Tesla", {"a": (CAR, state)})
        with patch(f"{MODULE}.active", return_value=[provider]), \
                patch(f"{MODULE}.reverse_geocode", return_value="Some Road"), \
                patch.object(section, "_get_eta_to_home", return_value=None):
            section._fetch_content()
        return provider

    def test_parked_polls_slowly_and_driving_every_minute(self, section):
        self._fetch(section, VehicleState(latitude=HOME[0], longitude=HOME[1]))
        assert section._get_update_interval() == CarSectionGenerator.UPDATE_INTERVAL_PARKED
        self._fetch(section, VehicleState(latitude=AWAY[0], longitude=AWAY[1], speed_kmh=80, in_gear=True))
        assert section._get_update_interval() == CarSectionGenerator.UPDATE_INTERVAL_DRIVING
        self._fetch(section, VehicleState(latitude=HOME[0], longitude=HOME[1]))
        assert section._get_update_interval() == CarSectionGenerator.UPDATE_INTERVAL_PARKED

    def test_the_interval_check_makes_no_state_calls(self, section):
        provider = self._fetch(section, VehicleState(latitude=HOME[0], longitude=HOME[1]))
        calls = provider.vehicle_state.call_count
        section._get_update_interval()
        assert provider.vehicle_state.call_count == calls

    def test_not_ready_looks_again_soon(self, section):
        with patch(f"{MODULE}.active", return_value=[]), patch(f"{MODULE}.no_provider_hint", return_value="x"):
            section._fetch_content()
        assert section._get_update_interval() < CarSectionGenerator.UPDATE_INTERVAL_DRIVING


@pytest.mark.unit
class TestEtaToHome:
    def test_prefers_the_duration_in_traffic(self, section):
        response = Mock(status_code=200)
        response.json.return_value = {"status": "OK", "rows": [{"elements": [{
            "status": "OK", "duration_in_traffic": {"value": 600}, "duration": {"value": 1200}}]}]}
        with patch(f"{MODULE}.config") as config, patch(f"{MODULE}.requests") as requests:
            config.get.return_value = "fake-api-key"
            requests.get.return_value = response
            assert section._get_eta_to_home(*AWAY) == 10

    def test_without_an_api_key_there_is_no_request(self, section):
        with patch(f"{MODULE}.config") as config, patch(f"{MODULE}.requests") as requests:
            config.get.return_value = ""
            assert section._get_eta_to_home(*AWAY) is None
            requests.get.assert_not_called()

    def test_a_network_error_is_none(self, section):
        with patch(f"{MODULE}.config") as config, patch(f"{MODULE}.requests") as requests:
            config.get.return_value = "fake-api-key"
            requests.get.side_effect = Exception("network error")
            assert section._get_eta_to_home(*AWAY) is None


@pytest.mark.unit
def test_haversine_matches_a_known_distance():
    # Gothenburg central station to Lindholmen, about 2.5 km
    assert abs(CarSectionGenerator._haversine_distance(57.7089, 11.9746, 57.7210, 11.9420) - 2500) < 500


@pytest.mark.unit
class TestPlacingTheCar:
    def test_without_a_home_location_a_located_car_is_unknown_and_nothing_is_looked_up(self):
        with patch(f"{MODULE}.config") as config:
            config.get_home_latitude.return_value = None
            config.get_home_longitude.return_value = None
            section = CarSectionGenerator(None)
        with patch(f"{MODULE}.config") as config, patch(f"{MODULE}.reverse_geocode") as geocode, \
                patch(f"{MODULE}.requests") as requests:
            config.get.return_value = "fake-api-key"
            line = section._render(CAR, VehicleState(latitude=AWAY[0], longitude=AWAY[1]))
        assert "Location unknown" in line and "from home" not in line
        geocode.assert_not_called()
        requests.get.assert_not_called()

    def test_an_address_the_geocoder_cannot_name_is_not_home_with_the_minutes(self, section):
        with patch(f"{MODULE}.reverse_geocode", return_value=None), \
                patch.object(section, "_get_eta_to_home", return_value=25):
            line = section._render(CAR, VehicleState(latitude=AWAY[0], longitude=AWAY[1]))
        assert "Not home ~25 min from home" in line

    def test_a_plug_in_hybrid_shows_its_battery_not_its_fuel(self, section):
        line = section._render(CAR, VehicleState(latitude=HOME[0], longitude=HOME[1], battery_percent=64, fuel_percent=40))
        assert "(64% battery)" in line and "fuel" not in line


@pytest.mark.unit
class TestEtaToHomeFallbacks:
    @staticmethod
    def _answer(element):
        response = Mock(status_code=200)
        response.json.return_value = {"status": "OK", "rows": [{"elements": [element]}]}
        return response

    def test_the_plain_duration_stands_in_without_traffic_and_a_non_ok_answer_is_none(self, section):
        with patch(f"{MODULE}.config") as config, patch(f"{MODULE}.requests") as requests:
            config.get.return_value = "fake-api-key"
            requests.get.return_value = self._answer({"status": "OK", "duration": {"value": 1500}})
            assert section._get_eta_to_home(*AWAY) == 25
            requests.get.return_value = self._answer({"status": "ZERO_RESULTS"})
            assert section._get_eta_to_home(*AWAY) is None
            requests.get.return_value = Mock(status_code=403)
            assert section._get_eta_to_home(*AWAY) is None
