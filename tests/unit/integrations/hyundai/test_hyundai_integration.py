"""The Hyundai / Kia / Genesis VehicleProvider against a fake of the library's VehicleManager:
the config words, the cars, the state mapping, the cached-state TTL, the commands and the
one-time-code sign-in. Nothing here reaches the network."""
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
import requests
from hyundai_kia_connect_api.ApiImpl import OTPRequest
from hyundai_kia_connect_api.const import OTP_NOTIFY_TYPE
from hyundai_kia_connect_api.exceptions import (
    APIError,
    AuthenticationError,
    AuthenticationOTPRequired,
    RateLimitingError,
    UnsupportedControlError,
)
from hyundai_kia_connect_api.Token import Token

import integrations.hyundai.HyundaiIntegration as module
from assistant.integration.VehicleProvider import Unsupported, VehicleError, VehicleSpec
from integrations.hyundai.HyundaiIntegration import (
    HyundaiConfigurationError,
    HyundaiError,
    HyundaiIntegration,
    HyundaiLoginError,
    brand_code,
    parse_coordinates,
    region_code,
    to_km,
)

EV_VIN, ICE_VIN, UNKNOWN_VIN = "KMHK381GFPU000001", "KNDJ23AU5P7000002", "KMHL14JA1PA000003"
UPDATED = datetime(2026, 10, 9, 7, 30, tzinfo=timezone.utc)


def ev_vehicle():
    return SimpleNamespace(id="lib-ev", name="IONIQ 5", model="Ioniq 5", VIN=EV_VIN, engine_type="EV",
                           ev_battery_percentage=64, ev_driving_range=210, ev_driving_range_unit="km",
                           fuel_level=None, fuel_driving_range=None, _fuel_driving_range_unit=None,
                           total_driving_range=210, total_driving_range_unit="km",
                           engine_is_running=False, location_latitude=59.33, location_longitude=18.07,
                           last_updated_at=UPDATED)


def ice_vehicle():
    return SimpleNamespace(id="lib-ice", name="Sportage", model="Sportage", VIN=ICE_VIN, engine_type="ICE",
                           ev_battery_percentage=None, ev_driving_range=None, ev_driving_range_unit=None,
                           fuel_level=40.0, fuel_driving_range=250, _fuel_driving_range_unit="mi",
                           total_driving_range=250, total_driving_range_unit="mi",
                           engine_is_running=True, location_latitude=57.7, location_longitude=11.97,
                           last_updated_at=None)


class FakeManager:
    """What the integration asks of hyundai_kia_connect_api.VehicleManager, recorded."""

    def __init__(self, *, region, brand, username, password, pin, token=None, otp_on_login=False,
                 login_error=None, refresh_error=None):
        self.region, self.brand, self.username, self.password, self.pin = region, brand, username, password, pin
        self.token = token
        self.vehicles = {}
        self.otp_request = None
        self.otp_on_login = otp_on_login
        self.login_error = login_error
        self.refresh_error = refresh_error
        self.calls = []
        self.cached_updates = 0
        self.force_refreshes = 0
        self.command_error = None
        self.sent_otp = []
        self.verified_otp = []

    def check_and_refresh_token(self):
        self.calls.append("check_and_refresh_token")
        if self.login_error:
            raise self.login_error
        if self.token is None:
            if self.otp_on_login:
                self.otp_request = OTPRequest("req-1", None, True, True, "m@mlab.test", "+46701740605")
                raise AuthenticationOTPRequired("OTP required to refresh token")
            self.token = Token(username=self.username, access_token="access-1", refresh_token="refresh-1",
                               valid_until=datetime(2026, 10, 10, tzinfo=timezone.utc), device_id="dev-1")
            return True
        if self.refresh_error:
            raise self.refresh_error
        return False

    def send_otp(self, notify_type):
        self.sent_otp.append(notify_type)

    def verify_otp_and_complete_login(self, code):
        self.verified_otp.append(code)
        if code != "123456":
            raise AuthenticationError("wrong code")
        self.token = Token(username=self.username, access_token="access-otp", refresh_token="refresh-otp",
                           valid_until=datetime(2026, 10, 10, tzinfo=timezone.utc), device_id="dev-1")
        self.otp_request = None

    def update_all_vehicles_with_cached_state(self):
        self.cached_updates += 1
        if self.refresh_error:
            raise self.refresh_error

    def force_refresh_all_vehicles_states(self):
        self.force_refreshes += 1

    def _command(self, name, *args):
        self.calls.append((name, *args))
        if self.command_error:
            raise self.command_error
        return f"track-{len(self.calls)}"

    def start_climate(self, vehicle_id, options):
        return self._command("start_climate", vehicle_id, options)

    def stop_climate(self, vehicle_id):
        return self._command("stop_climate", vehicle_id)

    def set_navigation(self, vehicle_id, poi_list):
        return self._command("set_navigation", vehicle_id, poi_list)


class NoNavigationManager(FakeManager):
    def set_navigation(self, vehicle_id, poi_list):
        raise NotImplementedError("set_navigation is not implemented for this region")


def base_config(**overrides):
    cfg = {
        "username": "m@mlab.test", "password": "secret", "pin": "1234", "region": "europe", "brand": "hyundai",
        "vehicles": [
            {"name": "family_car", "display_name": "Family Ioniq", "vin": EV_VIN, "default": True},
            {"name": "old_car", "display_name": "Marcus' Sportage", "vin": ICE_VIN},
        ],
    }
    cfg.update(overrides)
    return cfg


@pytest.fixture(autouse=True)
def token_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "TOKEN_DIR", tmp_path / "hyundai")
    return tmp_path / "hyundai"


@pytest.fixture
def build(monkeypatch):
    """(integration, manager) over a FakeManager built with the given kwargs; the clock is a
    patchable `now`."""
    clock = {"now": 1_000_000.0}
    monkeypatch.setattr(module.time, "time", lambda: clock["now"])

    def make(config=None, manager_cls=FakeManager, **manager_kwargs):
        made = {}

        def factory(**kwargs):
            made["manager"] = manager_cls(**kwargs, **manager_kwargs)
            made["manager"].vehicles = {"lib-ev": ev_vehicle(), "lib-ice": ice_vehicle()}
            return made["manager"]

        integration = HyundaiIntegration(config_override=config or base_config(), manager_factory=factory)
        integration.clock = clock
        return integration, lambda: made.get("manager")

    return make


# --- the config words --------------------------------------------------------------------------

@pytest.mark.parametrize("word, code", [("europe", 1), ("Canada", 2), ("usa", 3), ("new zealand", 7), ("New-Zealand", 7), ("europe_cci", 9)])
def test_region_words_map_to_the_library_constants(word, code):
    assert region_code(word) == code


@pytest.mark.parametrize("word, code", [("kia", 1), ("Hyundai", 2), ("GENESIS", 3)])
def test_brand_words_map_to_the_library_constants(word, code):
    assert brand_code(word) == code


def test_an_unknown_region_or_brand_is_a_configuration_error_naming_the_choices():
    with pytest.raises(HyundaiConfigurationError, match="europe"):
        region_code("mars")
    with pytest.raises(HyundaiConfigurationError, match="genesis"):
        brand_code("volvo")
    with pytest.raises(HyundaiConfigurationError):
        HyundaiIntegration(config_override=base_config(region="atlantis"), manager_factory=FakeManager)


def test_missing_credentials_or_an_empty_vehicle_list_fail_at_construction():
    with pytest.raises(HyundaiConfigurationError, match="password"):
        HyundaiIntegration(config_override=base_config(password=""), manager_factory=FakeManager)
    with pytest.raises(HyundaiConfigurationError, match="empty"):
        HyundaiIntegration(config_override=base_config(vehicles=[]), manager_factory=FakeManager)


def test_the_manager_is_built_from_the_mapped_codes_and_a_string_pin(build):
    integration, manager = build(base_config(region="canada", brand="kia", pin=4321))
    assert manager() is None, "the manager is built on first use, not in the constructor"
    integration.vehicle_state(integration.vehicles()[0])
    assert (manager().region, manager().brand, manager().pin) == (2, 1, "4321")


# --- the cars ---------------------------------------------------------------------------------

def test_the_configured_list_becomes_specs_with_the_vin_as_handle(build):
    integration, _ = build()
    specs = integration.vehicles()
    assert [s.id for s in specs] == ["family_car", "old_car"]
    assert specs[0].handle == EV_VIN and specs[0].is_default and not specs[1].is_default
    assert specs[1].matches("marcus' sportage") and specs[1].matches(ICE_VIN.lower())


def test_without_a_list_every_car_of_the_account_is_offered_under_its_own_name(build):
    config = base_config()
    del config["vehicles"]
    integration, manager = build(config)
    specs = integration.vehicles()
    assert [(s.id, s.handle) for s in specs] == [("IONIQ 5", EV_VIN), ("Sportage", ICE_VIN)]
    assert "check_and_refresh_token" in manager().calls, "listing the account's cars needs the sign-in"


def test_a_configured_vin_the_account_lacks_is_an_error_listing_the_account_cars(build):
    integration, _ = build(base_config(vehicles=[{"name": "ghost", "vin": UNKNOWN_VIN}]))
    with pytest.raises(HyundaiError, match="IONIQ 5"):
        integration.vehicle_state(integration.vehicles()[0])


def test_a_car_without_a_vin_is_reported_as_not_configured(build):
    integration, _ = build(base_config(vehicles=[{"name": "nameless"}]))
    with pytest.raises(HyundaiConfigurationError, match="VIN"):
        integration.vehicle_state(integration.vehicles()[0])


# --- the state ----------------------------------------------------------------------------------

def test_an_electric_car_reports_battery_range_location_and_the_data_age(build):
    integration, _ = build()
    state = integration.vehicle_state(integration.vehicles()[0])
    assert (state.battery_percent, state.fuel_percent, state.range_km) == (64, None, 210.0)
    assert (state.latitude, state.longitude) == (59.33, 18.07)
    assert state.in_gear is False and state.cabin_temp_c is None
    assert state.data_time == UPDATED.timestamp()


def test_a_combustion_car_reports_fuel_and_a_range_converted_from_miles(build):
    integration, _ = build()
    state = integration.vehicle_state(integration.vehicles()[1])
    assert (state.battery_percent, state.fuel_percent) == (None, 40)
    assert state.range_km == pytest.approx(250 * 1.609344, abs=0.1)
    assert state.in_gear is True and state.data_time is None


def test_to_km_leaves_kilometres_and_nothing_alone():
    assert to_km(None, "mi") is None and to_km(100, "km") == 100.0 and to_km(100, None) == 100.0


def test_the_cached_state_is_pulled_once_per_ttl_for_every_car(build):
    integration, manager = build()
    specs = integration.vehicles()
    integration.vehicle_state(specs[0])
    integration.vehicle_state(specs[1])
    assert manager().cached_updates == 1
    integration.clock["now"] += module.STATE_TTL + 1
    integration.vehicle_state(specs[0])
    assert manager().cached_updates == 2
    assert manager().force_refreshes == 0, "the prompt must never wake the car"


def test_a_rate_limited_read_is_a_vehicle_error_the_llm_can_read(build):
    integration, _ = build(refresh_error=RateLimitingError("too many requests"))
    with pytest.raises(VehicleError, match="rate limiting"):
        integration.vehicle_state(integration.vehicles()[0])


# --- the commands -------------------------------------------------------------------------------

def test_climate_on_asks_for_the_configured_temperature_without_defrost(build):
    integration, manager = build(base_config(climate_temperature_c=19.5, climate_minutes=15))
    line = integration.climate_on(integration.vehicles()[0])
    name, vehicle_id, options = manager().calls[-1]
    assert (name, vehicle_id) == ("start_climate", "lib-ev")
    assert (options.set_temp, options.duration, options.climate, options.defrost) == (19.5, 15, True, False)
    assert "Family Ioniq" in line and "19.5" in line


def test_max_defrost_is_a_climate_start_with_defrost_and_heating(build):
    integration, manager = build()
    integration.max_defrost(integration.vehicles()[0])
    _, _, options = manager().calls[-1]
    assert options.defrost is True and options.heating == 1 and options.set_temp == module.DEFAULT_CLIMATE_C


def test_climate_off_calls_stop_climate_and_drops_the_state_cache(build):
    integration, manager = build()
    spec = integration.vehicles()[0]
    integration.vehicle_state(spec)
    integration.climate_off(spec)
    assert ("stop_climate", "lib-ev") in manager().calls
    integration.vehicle_state(spec)
    assert manager().cached_updates == 2, "a command invalidates the cached state so the next read is fresh"


def test_the_trunk_stays_unsupported(build):
    integration, _ = build()
    with pytest.raises(Unsupported):
        integration.open_trunk(integration.vehicles()[0])


def test_a_coordinate_destination_goes_to_the_navigation_as_is(build):
    integration, manager = build()
    line = integration.navigate_to(integration.vehicles()[0], "59.3293, 18.0686")
    name, vehicle_id, pois = manager().calls[-1]
    assert (name, vehicle_id, len(pois)) == ("set_navigation", "lib-ev", 1)
    assert (pois[0].coord.lat, pois[0].coord.lon) == (59.3293, 18.0686)
    assert "Family Ioniq" in line


def test_an_address_is_geocoded_before_it_goes_to_the_navigation(build):
    class Response:
        status_code = 200

        def json(self):
            return [{"lat": "57.7089", "lon": "11.9746", "display_name": "Avenyn 1, Göteborg"}]

    class Session:
        def __init__(self):
            self.requests = []

        def get(self, url, **kwargs):
            self.requests.append((url, kwargs))
            return Response()

    session = Session()
    integration = HyundaiIntegration(config_override=base_config(), manager_factory=lambda **kw: _with_cars(FakeManager(**kw)),
                                     session=session)
    integration.navigate_to(integration.vehicles()[0], "Avenyn 1, Göteborg")
    assert session.requests[0][1]["params"]["q"] == "Avenyn 1, Göteborg"
    assert "User-Agent" in session.requests[0][1]["headers"], "Nominatim refuses anonymous clients"


def test_a_place_nominatim_does_not_know_is_an_error_not_a_blind_route(build):
    class Session:
        def get(self, url, **kwargs):
            return SimpleNamespace(status_code=200, json=lambda: [])

    integration = HyundaiIntegration(config_override=base_config(), manager_factory=lambda **kw: _with_cars(FakeManager(**kw)),
                                     session=Session())
    with pytest.raises(HyundaiError, match="No place"):
        integration.navigate_to(integration.vehicles()[0], "Nowhere in particular")


def test_navigation_in_a_region_without_it_is_unsupported(build):
    integration, _ = build(manager_cls=NoNavigationManager)
    with pytest.raises(Unsupported, match="europe"):
        integration.navigate_to(integration.vehicles()[0], "59.3,18.0")


def test_a_car_refusing_a_control_is_unsupported_and_a_backend_error_is_a_vehicle_error(build):
    integration, manager = build()
    spec = integration.vehicles()[1]
    integration.vehicle_state(spec)
    manager().command_error = UnsupportedControlError("resCode 4005")
    with pytest.raises(Unsupported, match="Sportage"):
        integration.climate_off(spec)
    manager().command_error = APIError("server said no")
    with pytest.raises(VehicleError, match="server said no"):
        integration.climate_on(spec)


@pytest.mark.parametrize("text, expected", [
    ("59.33,18.07", (59.33, 18.07)), (" 59.33 , 18.07 ", (59.33, 18.07)),
    ("Storgatan 1, Lund", None), ("91,18", None), ("59.33", None), ("", None),
])
def test_parse_coordinates(text, expected):
    assert parse_coordinates(text) == expected


# --- the sign-in and the token -----------------------------------------------------------------

def test_the_token_is_saved_after_the_sign_in_and_handed_to_the_next_manager(build, token_dir):
    integration, manager = build()
    integration.vehicle_state(integration.vehicles()[0])
    files = list(token_dir.glob("token-*.json"))
    assert len(files) == 1 and (files[0].stat().st_mode & 0o777) == 0o600
    assert json.loads(files[0].read_text())["refresh_token"] == "refresh-1"

    again, manager_again = build()
    again.vehicle_state(again.vehicles()[0])
    assert manager_again().token.refresh_token == "refresh-1", "the saved token is passed in, no second login"
    assert [p for p, _ in HyundaiIntegration.wipe_targets()] == files


def test_a_one_time_code_demand_is_a_login_error_and_the_integration_reports_needing_2fa(build):
    integration, manager = build(otp_on_login=True)
    assert integration.needs_2fa() is False
    with pytest.raises(HyundaiLoginError, match="one-time code"):
        integration.vehicle_state(integration.vehicles()[0])
    assert integration.needs_2fa() is True
    assert integration.otp_destinations() == {"email": "m@mlab.test", "sms": "+46701740605"}
    assert integration.get_auth_status()["status"] == "needs_auth"

    integration.send_otp("sms")
    assert manager().sent_otp == [OTP_NOTIFY_TYPE.SMS]
    with pytest.raises(HyundaiLoginError, match="refused"):
        integration.verify_otp("000000")
    assert integration.needs_2fa() is True

    integration.verify_otp("123456")
    assert integration.needs_2fa() is False
    assert integration.get_auth_status()["status"] == "authenticated"
    manager().otp_on_login = False
    state = integration.vehicle_state(integration.vehicles()[0])
    assert state.battery_percent == 64


def test_sending_a_code_without_a_pending_sign_in_does_nothing(build):
    integration, manager = build()
    assert "Ctrl-R" in integration.send_otp("email")
    assert manager() is None


def test_refused_credentials_are_a_login_error_shown_in_the_auth_status(build):
    integration, _ = build(login_error=AuthenticationError("invalid password"))
    with pytest.raises(HyundaiLoginError, match="invalid password"):
        integration.check_health()
    assert integration.get_auth_status()["status"] == "error"
    assert integration.needs_2fa() is False, "a wrong password is not a code to enter"


def test_check_health_is_true_once_the_first_car_has_state(build):
    integration, _ = build()
    assert integration.check_health() is True
    swapped, _ = build(base_config(vehicles=[{"name": "old_car", "vin": ICE_VIN}]))
    assert swapped.check_health() is False, "the Sportage's state has no timestamp yet"


def _with_cars(manager):
    manager.vehicles = {"lib-ev": ev_vehicle(), "lib-ice": ice_vehicle()}
    return manager


# --- the state mapping per engine type -----------------------------------------------------------

def _vehicle(**fields):
    base = dict(ev_battery_percentage=None, ev_driving_range=None, ev_driving_range_unit=None,
                fuel_level=None, fuel_driving_range=None, _fuel_driving_range_unit=None,
                total_driving_range=None, total_driving_range_unit=None, engine_is_running=False,
                location_latitude=None, location_longitude=None, last_updated_at=None)
    base.update(fields)
    return SimpleNamespace(**base)


def test_a_plug_in_hybrid_reports_battery_and_fuel_and_its_electric_range_first():
    state = HyundaiIntegration._to_state(_vehicle(engine_type="PHEV", ev_battery_percentage=80, fuel_level=55.5,
                                                  ev_driving_range=45, ev_driving_range_unit="km",
                                                  fuel_driving_range=400, _fuel_driving_range_unit="km"))
    assert (state.battery_percent, state.fuel_percent, state.range_km) == (80, 55, 45.0)


def test_a_hybrid_without_an_electric_range_falls_back_to_the_fuel_range_whatever_the_case_of_its_type():
    state = HyundaiIntegration._to_state(_vehicle(engine_type="phev", ev_battery_percentage=12,
                                                  fuel_driving_range=100, _fuel_driving_range_unit="mi"))
    assert state.battery_percent == 12 and state.range_km == pytest.approx(160.9, abs=0.05)


def test_an_unknown_engine_type_claims_neither_charge_and_uses_the_total_range():
    state = HyundaiIntegration._to_state(_vehicle(engine_type=None, ev_battery_percentage=50, fuel_level=50,
                                                  total_driving_range=300, total_driving_range_unit="km"))
    assert (state.battery_percent, state.fuel_percent, state.range_km) == (None, None, 300.0)


# --- the token file and the sign-in's failures ----------------------------------------------------

def test_a_corrupt_token_file_is_replaced_by_a_fresh_sign_in(build):
    path = module.token_path("m@mlab.test")
    path.parent.mkdir(parents=True)
    path.write_text("not json")
    integration, manager = build()
    integration.vehicle_state(integration.vehicles()[0])
    assert manager().token.access_token == "access-1", "a fresh login, not a token read from the file"
    assert json.loads(path.read_text())["refresh_token"] == "refresh-1"


def test_an_unchanged_token_is_not_rewritten_on_every_read_a_changed_one_is(build):
    integration, manager = build()
    spec = integration.vehicles()[0]
    integration.vehicle_state(spec)
    path = module.token_path("m@mlab.test")
    path.write_text("left alone")
    integration.clock["now"] += module.STATE_TTL + 1
    integration.vehicle_state(spec)
    assert path.read_text() == "left alone"

    manager().token = Token(username="m@mlab.test", access_token="access-2", refresh_token="refresh-2",
                            valid_until=datetime(2026, 10, 11, tzinfo=timezone.utc), device_id="dev-1")
    integration.clock["now"] += module.STATE_TTL + 1
    integration.vehicle_state(spec)
    assert json.loads(path.read_text())["refresh_token"] == "refresh-2"


def test_a_rate_limited_or_failed_sign_in_is_not_a_refused_one(build):
    integration, _ = build(login_error=RateLimitingError("slow down"))
    with pytest.raises(HyundaiError, match="rate limiting") as caught:
        integration.vehicle_state(integration.vehicles()[0])
    assert not isinstance(caught.value, HyundaiLoginError)
    assert integration.get_auth_status()["status"] != "error" and not integration.needs_2fa()

    broken, _ = build(login_error=APIError("backend down"))
    with pytest.raises(HyundaiError, match="backend down") as caught:
        broken.check_health()
    assert not isinstance(caught.value, HyundaiLoginError)


def test_a_verified_code_drops_the_state_cache(build):
    integration, manager = build()
    spec = integration.vehicles()[0]
    integration.vehicle_state(spec)
    manager().token = None
    manager().otp_on_login = True
    with pytest.raises(HyundaiLoginError):
        integration.vehicle_state(spec)

    integration.verify_otp("123456")

    integration.vehicle_state(spec)
    assert manager().cached_updates == 2, "pulled afresh after the sign-in, TTL or not"


def test_refresh_now_pulls_the_cached_state_inside_the_ttl_without_waking_the_car(build):
    integration, manager = build()
    integration.vehicle_state(integration.vehicles()[0])
    integration.refresh_now()
    assert (manager().cached_updates, manager().force_refreshes) == (2, 0)


def test_a_failed_read_does_not_age_the_cache_so_the_next_call_retries(build):
    integration, manager = build()
    spec = integration.vehicles()[0]
    integration.vehicle_state(spec)

    def fail():
        raise APIError("backend hiccup")
    manager().update_all_vehicles_with_cached_state = fail
    integration.clock["now"] += module.STATE_TTL + 1
    with pytest.raises(VehicleError, match="backend hiccup"):
        integration.vehicle_state(spec)

    del manager().update_all_vehicles_with_cached_state
    integration.vehicle_state(spec)
    assert manager().cached_updates == 2, "the clock has not moved: a cache stamped by the failure would serve the stale state"


# --- the cars and the commands ---------------------------------------------------------------------

def test_an_account_car_without_a_name_is_offered_by_its_vin_and_labelled_by_its_model(build):
    config = base_config()
    del config["vehicles"]
    integration, manager = build(config)
    integration.vehicles()
    manager().vehicles["lib-ev"].name = None
    spec = integration.vehicles()[0]
    assert (spec.id, spec.label, spec.handle) == (EV_VIN, "Ioniq 5", EV_VIN)
    assert integration.vehicle_state(spec).battery_percent == 64


def test_the_configured_vin_matches_the_account_car_whatever_its_case(build):
    integration, _ = build(base_config(vehicles=[{"name": "family_car", "vin": EV_VIN.lower()}]))
    assert integration.vehicle_state(integration.vehicles()[0]).battery_percent == 64


def test_a_rate_limited_command_is_a_vehicle_error_not_an_unsupported_one_and_keeps_the_cache(build):
    integration, manager = build()
    spec = integration.vehicles()[0]
    integration.vehicle_state(spec)
    manager().command_error = RateLimitingError("too many commands")
    with pytest.raises(VehicleError, match="rate limiting") as caught:
        integration.climate_off(spec)
    assert not isinstance(caught.value, Unsupported)
    integration.vehicle_state(spec)
    assert manager().cached_updates == 1, "a command that never went out leaves the cached state valid"


def test_a_blank_destination_is_refused_before_any_sign_in(build):
    integration, manager = build()
    with pytest.raises(HyundaiError, match="destination"):
        integration.navigate_to(integration.vehicles()[0], "   ")
    assert manager() is None


def test_a_geocoder_outage_or_refusal_is_an_error_naming_the_place():
    class Down:
        def get(self, url, **kwargs):
            raise requests.ConnectionError("dns")

    class Refusing:
        def get(self, url, **kwargs):
            return SimpleNamespace(status_code=503, json=lambda: {"error": "busy"})

    for session, reason in ((Down(), "Could not look up"), (Refusing(), "No place")):
        integration = HyundaiIntegration(config_override=base_config(), session=session,
                                         manager_factory=lambda **kw: _with_cars(FakeManager(**kw)))
        with pytest.raises(HyundaiError, match=f"{reason}.*Avenyn 1"):
            integration.navigate_to(integration.vehicles()[0], "Avenyn 1")
