import time
from datetime import datetime
from unittest.mock import patch, Mock


def _make_tesla_section(tesla_api=None):
    """Create a TeslaSectionGenerator with mocked config."""
    with patch('assistant.prompt.sections.TeslaSectionGenerator.config') as mock_config:
        mock_config.get_home_latitude.return_value = 57.7089
        mock_config.get_home_longitude.return_value = 11.9746
        from assistant.prompt.sections.TeslaSectionGenerator import TeslaSectionGenerator
        section = TeslaSectionGenerator(None)
    section.tesla_api = tesla_api
    return section


# ===========================================================================
# TestHaversineDistance
# ===========================================================================

class TestHaversineDistance:
    def _haversine(self, *args):
        from assistant.prompt.sections.TeslaSectionGenerator import TeslaSectionGenerator
        return TeslaSectionGenerator._haversine_distance(*args)

    def test_known_distance(self):
        # Gothenburg central station to Lindholmen (~2.5km)
        dist = self._haversine(57.7089, 11.9746, 57.7210, 11.9420)
        assert abs(dist - 2500) < 500  # within 500m tolerance

    def test_short_distance(self):
        # Two points ~100m apart
        dist = self._haversine(57.7089, 11.9746, 57.7098, 11.9746)
        assert abs(dist - 100) < 50


# ===========================================================================
# TestReverseGeocode
# ===========================================================================

class TestReverseGeocode:
    @patch('assistant.tools.Geocoding.requests')
    def test_returns_address_parts(self, mock_requests):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "address": {
                "road": "Main Street",
                "suburb": "Downtown",
                "city": "Stockholm"
            }
        }
        mock_requests.get.return_value = mock_response
        from assistant.tools.Geocoding import reverse_geocode
        result = reverse_geocode(57.7, 11.9)
        assert "Main Street" in result
        assert "Downtown" in result
        assert "Stockholm" in result

    @patch('assistant.tools.Geocoding.requests')
    def test_returns_none_on_timeout(self, mock_requests):
        mock_requests.get.side_effect = Exception("timeout")
        from assistant.tools.Geocoding import reverse_geocode
        result = reverse_geocode(57.7, 11.9)
        assert result is None

    @patch('assistant.tools.Geocoding.requests')
    def test_returns_none_on_http_error(self, mock_requests):
        mock_response = Mock()
        mock_response.status_code = 500
        mock_requests.get.return_value = mock_response
        from assistant.tools.Geocoding import reverse_geocode
        result = reverse_geocode(57.7, 11.9)
        assert result is None

    @patch('assistant.tools.Geocoding.requests')
    def test_filters_home_city(self, mock_requests):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "address": {
                "road": "Avenyn",
                "city": "Göteborg"
            }
        }
        mock_requests.get.return_value = mock_response
        from assistant.tools.Geocoding import reverse_geocode
        result = reverse_geocode(57.7, 11.9)
        assert "Avenyn" in result
        assert "öteborg" not in result


# ===========================================================================
# TestGetEtaToHome
# ===========================================================================

class TestGetEtaToHome:
    @patch('assistant.prompt.sections.TeslaSectionGenerator.config')
    @patch('assistant.prompt.sections.TeslaSectionGenerator.requests')
    def test_returns_none_on_error(self, mock_requests, mock_config):
        mock_config.get.return_value = "fake-api-key"
        mock_requests.get.side_effect = Exception("network error")
        section = _make_tesla_section()
        result = section._get_eta_to_home(57.5, 12.0)
        assert result is None

    @patch('assistant.prompt.sections.TeslaSectionGenerator.config')
    @patch('assistant.prompt.sections.TeslaSectionGenerator.requests')
    def test_prefers_duration_in_traffic(self, mock_requests, mock_config):
        mock_config.get.return_value = "fake-api-key"
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "OK",
            "rows": [{"elements": [{"status": "OK",
                "duration_in_traffic": {"value": 600},
                "duration": {"value": 1200}
            }]}]
        }
        mock_requests.get.return_value = mock_response
        section = _make_tesla_section()
        result = section._get_eta_to_home(57.5, 12.0)
        assert result == 10  # 600/60, not 1200/60


# ===========================================================================
# TestGetVehicleInfo
# ===========================================================================

class TestGetVehicleInfo:
    def _make_section_with_api(self):
        mock_api = Mock()
        section = _make_tesla_section(tesla_api=mock_api)
        return section, mock_api

    def test_vehicle_at_home(self):
        section, mock_api = self._make_section_with_api()
        mock_api._api_get.return_value = {
            "drive_state": {"latitude": 57.7089, "longitude": 11.9746, "speed": None, "shift_state": None},
            "charge_state": {"usable_battery_level": 80},
            "climate_state": {}
        }
        result = section._get_vehicle_info("VIN123", "Model 3", "car1")
        assert "Parked" in result
        assert "Home" in result

    def test_vehicle_driving(self):
        section, mock_api = self._make_section_with_api()
        mock_api._api_get.return_value = {
            "drive_state": {"latitude": 57.5, "longitude": 12.0, "speed": 80, "shift_state": "D"},
            "charge_state": {"usable_battery_level": 60},
            "climate_state": {}
        }
        # Away from home with no route: the ETA lookup would be a real Distance Matrix request.
        with patch('assistant.prompt.sections.TeslaSectionGenerator.reverse_geocode', return_value="E6 Highway, Mellanstad"), \
             patch.object(section, '_get_eta_to_home', return_value=None):
            result = section._get_vehicle_info("VIN123", "Model 3", "car1")
        assert "Driving" in result
        assert "80 km/h" in result
        assert "E6 Highway" in result

    def test_vehicle_with_active_navigation(self):
        section, mock_api = self._make_section_with_api()
        mock_api._api_get.return_value = {
            "drive_state": {
                "latitude": 57.5, "longitude": 12.0, "speed": 80, "shift_state": "D",
                "active_route_minutes_to_arrival": 15, "active_route_destination": "IKEA"
            },
            "charge_state": {"usable_battery_level": 60},
            "climate_state": {}
        }
        with patch('assistant.prompt.sections.TeslaSectionGenerator.reverse_geocode', return_value="Some Road"):
            result = section._get_vehicle_info("VIN123", "Model 3", "car1")
        assert "Navigating to IKEA" in result
        assert "ETA 15 min" in result

    def test_parked_with_route_loaded_hides_navigation(self):
        # Tesla keeps a route loaded in the nav even when parked - must not show "Navigating"
        section, mock_api = self._make_section_with_api()
        mock_api._api_get.return_value = {
            "state": "online",
            "drive_state": {
                "latitude": 57.7089, "longitude": 11.9746, "speed": None, "shift_state": "P",
                "active_route_minutes_to_arrival": 10.13, "active_route_destination": "Work"
            },
            "charge_state": {"usable_battery_level": 59},
            "climate_state": {}
        }
        result = section._get_vehicle_info("VIN123", "Model X", "car1")
        assert "Parked" in result
        assert "Home" in result
        assert "Navigating" not in result
        assert "Work" not in result

    def test_in_gear_at_zero_speed_still_shows_navigation(self):
        # Stopped at a red light: speed 0 but shift_state D - still navigating
        section, mock_api = self._make_section_with_api()
        mock_api._api_get.return_value = {
            "state": "online",
            "drive_state": {
                "latitude": 57.5, "longitude": 12.0, "speed": 0, "shift_state": "D",
                "active_route_minutes_to_arrival": 12.7, "active_route_destination": "IKEA"
            },
            "charge_state": {"usable_battery_level": 60},
            "climate_state": {}
        }
        with patch('assistant.prompt.sections.TeslaSectionGenerator.reverse_geocode', return_value="Some Road"):
            result = section._get_vehicle_info("VIN123", "Model 3", "car1")
        assert "Navigating to IKEA" in result
        assert "ETA 13 min" in result  # float ETA is rounded

    def test_asleep_car_flagged_with_data_timestamp(self):
        section, mock_api = self._make_section_with_api()
        data_ts_ms = (time.time() - 3600) * 1000  # snapshot from an hour ago
        mock_api._api_get.return_value = {
            "state": "asleep",
            "drive_state": {"latitude": 57.7089, "longitude": 11.9746, "speed": None,
                            "shift_state": None, "timestamp": data_ts_ms},
            "charge_state": {"usable_battery_level": 59},
            "climate_state": {"inside_temp": 23.7}
        }
        result = section._get_vehicle_info("VIN123", "Model X", "car1")
        expected_time = datetime.fromtimestamp(data_ts_ms / 1000).strftime("%H:%M")
        assert f"[car is asleep, data from {expected_time}]" in result

    def test_asleep_car_without_timestamp_flagged_stale(self):
        section, mock_api = self._make_section_with_api()
        mock_api._api_get.return_value = {
            "state": "asleep",
            "drive_state": {"latitude": 57.7089, "longitude": 11.9746, "speed": None, "shift_state": None},
            "charge_state": {"usable_battery_level": 59},
            "climate_state": {}
        }
        result = section._get_vehicle_info("VIN123", "Model X", "car1")
        assert "[car is asleep, data may be stale]" in result

    def test_online_car_has_no_staleness_note(self):
        section, mock_api = self._make_section_with_api()
        mock_api._api_get.return_value = {
            "state": "online",
            "drive_state": {"latitude": 57.7089, "longitude": 11.9746, "speed": None, "shift_state": None},
            "charge_state": {"usable_battery_level": 80},
            "climate_state": {}
        }
        result = section._get_vehicle_info("VIN123", "Model 3", "car1")
        assert "asleep" not in result
        assert "stale" not in result

    def test_location_unknown(self):
        section, mock_api = self._make_section_with_api()
        mock_api._api_get.return_value = {
            "drive_state": {"speed": None, "shift_state": None},
            "charge_state": {"usable_battery_level": 50},
            "climate_state": {}
        }
        result = section._get_vehicle_info("VIN123", "Model 3", "car1")
        assert "Location unknown" in result

    def test_cabin_temperature(self):
        section, mock_api = self._make_section_with_api()
        mock_api._api_get.return_value = {
            "drive_state": {"latitude": 57.7089, "longitude": 11.9746, "speed": None, "shift_state": None},
            "charge_state": {"usable_battery_level": 80},
            "climate_state": {"inside_temp": 22}
        }
        result = section._get_vehicle_info("VIN123", "Model 3", "car1")
        assert "Cabin 22°C" in result


# ===========================================================================
# TestFetchContent
# ===========================================================================

class TestFetchContent:
    def test_multiple_vehicles(self):
        mock_api = Mock()
        spec1 = Mock()
        spec1.vin = "VIN1"
        spec1.label.return_value = "Model 3"
        spec1.name = "car1"
        spec2 = Mock()
        spec2.vin = "VIN2"
        spec2.label.return_value = "Model Y"
        spec2.name = "car2"
        mock_api._vehicle_specs = [spec1, spec2]
        section = _make_tesla_section(tesla_api=mock_api)
        with patch.object(section, '_get_vehicle_info', side_effect=["Model 3: Parked Home", "Model Y: Driving 80 km/h"]):
            result = section._fetch_content()
        assert "Model 3: Parked Home" in result
        assert "Model Y: Driving 80 km/h" in result

    def test_error_per_vehicle_handled(self):
        mock_api = Mock()
        spec1 = Mock()
        spec1.vin = "VIN1"
        spec1.label.return_value = "Model 3"
        spec1.name = "car1"
        spec2 = Mock()
        spec2.vin = "VIN2"
        spec2.label.return_value = "Model Y"
        spec2.name = "car2"
        mock_api._vehicle_specs = [spec1, spec2]
        section = _make_tesla_section(tesla_api=mock_api)
        with patch.object(section, '_get_vehicle_info', side_effect=[RuntimeError("API timeout"), "Model Y: Parked Home"]):
            result = section._fetch_content()
        assert "Error" in result
        assert "Model Y: Parked Home" in result


# ===========================================================================
# TestGetUpdateInterval
# ===========================================================================

class TestGetUpdateInterval:
    """Driving status is carried over from the fetch pass - no extra API calls."""

    def _make_section_with_one_car(self, drive_state):
        mock_api = Mock()
        spec = Mock()
        spec.vin = "VIN1"
        spec.label.return_value = "Model 3"
        spec.name = "car1"
        mock_api._vehicle_specs = [spec]
        mock_api._api_get.return_value = {
            "state": "online",
            "drive_state": drive_state,
            "charge_state": {"usable_battery_level": 80},
            "climate_state": {}
        }
        return _make_tesla_section(tesla_api=mock_api), mock_api

    def test_parked_after_fetch_returns_300(self):
        section, _ = self._make_section_with_one_car(
            {"latitude": 57.7089, "longitude": 11.9746, "speed": None, "shift_state": None})
        section._fetch_content()
        assert section._get_update_interval() == 300

    def test_driving_after_fetch_returns_60(self):
        section, mock_api = self._make_section_with_one_car(
            {"latitude": 57.5, "longitude": 12.0, "speed": 80, "shift_state": "D"})
        with patch('assistant.prompt.sections.TeslaSectionGenerator.reverse_geocode', return_value="Some Road"), \
             patch.object(section, '_get_eta_to_home', return_value=None):
            section._fetch_content()
        assert section._get_update_interval() == 60

    def test_in_gear_at_zero_speed_returns_60(self):
        section, _ = self._make_section_with_one_car(
            {"latitude": 57.7089, "longitude": 11.9746, "speed": 0, "shift_state": "D"})
        section._fetch_content()
        assert section._get_update_interval() == 60

    def test_interval_check_makes_no_api_calls(self):
        section, mock_api = self._make_section_with_one_car(
            {"latitude": 57.7089, "longitude": 11.9746, "speed": None, "shift_state": None})
        section._fetch_content()
        calls_after_fetch = mock_api._api_get.call_count
        section._get_update_interval()
        assert mock_api._api_get.call_count == calls_after_fetch

    def test_flag_resets_when_car_parks(self):
        section, mock_api = self._make_section_with_one_car(
            {"latitude": 57.7089, "longitude": 11.9746, "speed": 0, "shift_state": "D"})
        section._fetch_content()
        assert section._get_update_interval() == 60
        mock_api._api_get.return_value = {
            "state": "online",
            "drive_state": {"latitude": 57.7089, "longitude": 11.9746, "speed": None, "shift_state": None},
            "charge_state": {"usable_battery_level": 80},
            "climate_state": {}
        }
        section._fetch_content()
        assert section._get_update_interval() == 300
