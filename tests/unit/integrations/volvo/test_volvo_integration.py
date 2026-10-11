"""The Volvo folder as a VehicleProvider: the cars from `volvo.vehicles`, the state composed from
the Connected Vehicle, Energy and Location APIs and cached for a minute, the climatization
commands and their invoke status, the API's refusals as VehicleErrors, and the three actions
Volvo lacks answering Unsupported."""
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
import requests

from assistant.integration.VehicleProvider import Unsupported, VehicleError, VehicleProvider, VehicleSpec
from integrations.volvo import VolvoAuth as auth_module
from integrations.volvo import VolvoIntegration as module
from integrations.volvo.VolvoAuth import NotSignedIn, token_path
from integrations.volvo.VolvoIntegration import (VolvoApiError, VolvoCommandError, VolvoIntegration, VolvoRateLimited)

from .volvo_fakes import (NOW_ISO, VIN, FakeVolvo, StubAuth, command_result, details, energy_value, location, value)

CONFIG = "integrations.volvo.VolvoIntegration.ConfigManager.get_instance"
CAR_LIST = "/connected-vehicle/v2/vehicles"
CONNECTED = f"{CAR_LIST}/{VIN}"
ENERGY_STATE = f"/energy/v2/vehicles/{VIN}/state"
LOCATION = f"/location/v1/vehicles/{VIN}/location"
VEHICLES = [{"name": "family_xc40", "display_name": "Family XC40", "vin": VIN, "default": True},
            {"name": "v60", "vin": "YV1KEP47L00000002"}]


def _config(vehicles=VEHICLES, api_key="vcc-key", client_id="app-id", client_secret="app-secret"):
    values = {"volvo.vehicles": vehicles, "volvo.vcc_api_key": api_key,
              "volvo.oauth.client_id": client_id, "volvo.oauth.client_secret": client_secret}
    config = MagicMock()
    config.get.side_effect = lambda key, *default: values.get(key, default[0] if default else None)
    return patch(CONFIG, return_value=config)


@pytest.fixture(autouse=True)
def fresh(tmp_path):
    VolvoIntegration.reset_instance()
    with patch.object(auth_module, "TOKEN_DIR", tmp_path / "oauth"):
        yield tmp_path / "oauth"
    VolvoIntegration.reset_instance()


def build(fake: FakeVolvo, auth=None) -> VolvoIntegration:
    integration = VolvoIntegration(session=fake)
    integration.auth = auth or StubAuth()
    return integration


def electric_car(fake: FakeVolvo, availability="AVAILABLE", engine="STOPPED", battery="72", electric_range="310", stamp=NOW_ISO):
    fake.route("GET", CONNECTED, details("ELECTRIC"))
    fake.route("GET", f"{CONNECTED}/command-accessibility", {"data": {"availabilityStatus": value(availability, stamp)}})
    fake.route("GET", f"{CONNECTED}/engine-status", {"data": {"engineStatus": value(engine, stamp)}})
    fake.route("GET", f"{CONNECTED}/statistics", {"data": {"averageSpeed": value(47, stamp, "km/h"), "distanceToEmptyBattery": value(290, stamp, "km")}})
    fake.route("GET", ENERGY_STATE, {"batteryChargeLevel": energy_value(battery, stamp, "percentage"),
                                     "electricRange": energy_value(electric_range, stamp, "km"),
                                     "chargingStatus": energy_value("IDLE", stamp)})
    fake.route("GET", LOCATION, location(stamp=stamp))


def spec(integration: VolvoIntegration) -> VehicleSpec:
    return integration.vehicles()[0]


@pytest.mark.unit
class TestVehicles:
    def test_the_entries_become_specs_the_name_the_id_the_vin_the_handle(self):
        with _config():
            integration = build(FakeVolvo())
            assert isinstance(integration, VehicleProvider)
            cars = integration.vehicles()

        assert [c.id for c in cars] == ["family_xc40", "v60"]
        assert cars[0].label == "Family XC40" and cars[0].handle == VIN and cars[0].is_default
        assert cars[0].matches("family xc40") and cars[0].matches(VIN)
        assert cars[1].label == "v60" and not cars[1].is_default

    def test_the_config_keys_are_the_four_the_sign_in_and_the_calls_need_a_contract_with_config_yaml(self):
        with _config():
            assert build(FakeVolvo()).config_keys == ["volvo.vcc_api_key", "volvo.oauth.client_id", "volvo.oauth.client_secret", "volvo.vehicles"]

    def test_a_car_without_a_vin_cannot_be_read(self):
        with _config(vehicles=[{"name": "nameless"}]):
            integration = build(FakeVolvo())
            with pytest.raises(VehicleError, match="VIN"):
                integration.vehicle_state(spec(integration))


@pytest.mark.unit
class TestVehicleState:
    def test_an_electric_car_parked_and_reachable(self):
        fake = FakeVolvo()
        electric_car(fake)
        with _config():
            integration = build(fake)
            state = integration.vehicle_state(spec(integration))

        assert state.connectivity == "online" and not state.in_gear and not state.is_driving
        assert state.battery_percent == 72 and state.range_km == 310 and state.fuel_percent is None
        assert (state.latitude, state.longitude) == (59.3456, 17.9876)      # GeoJSON lists longitude first
        assert state.data_time == 1791533700.0                               # 2026-10-09T08:15:00Z
        assert state.speed_kmh is None and state.destination is None
        assert {c.path for c in fake.api_calls()} == {CONNECTED, f"{CONNECTED}/command-accessibility", f"{CONNECTED}/engine-status",
                                                      f"{CONNECTED}/statistics", ENERGY_STATE, LOCATION}
        assert f"{CONNECTED}/fuel" not in {c.path for c in fake.api_calls()}

    def test_every_call_carries_the_bearer_token_and_the_vcc_api_key(self):
        fake = FakeVolvo()
        electric_car(fake)
        with _config(api_key="vcc-key-77"):
            integration = build(fake, StubAuth(token="at.77"))
            integration.vehicle_state(spec(integration))

        for call in fake.api_calls():
            assert call.headers["Authorization"] == "Bearer at.77" and call.headers["vcc-api-key"] == "vcc-key-77"

    def test_a_running_engine_or_a_car_in_use_is_in_gear_and_asleep_or_offline_cars_say_so(self):
        for availability, engine, in_gear, connectivity in (("CAR_IN_USE", "STOPPED", True, "online"),
                                                             ("AVAILABLE", "RUNNING", True, "online"),
                                                             ("POWER_SAVING_MODE", "STOPPED", False, "asleep"),
                                                             ("NO_INTERNET", "STOPPED", False, "offline"),
                                                             ("UNSPECIFIED", "STOPPED", False, None)):
            fake = FakeVolvo()
            electric_car(fake, availability=availability, engine=engine)
            with _config():
                VolvoIntegration.reset_instance()
                integration = build(fake)
                state = integration.vehicle_state(spec(integration))
            assert (state.in_gear, state.connectivity) == (in_gear, connectivity), availability

    def test_a_combustion_car_reads_fuel_and_tank_range_not_the_energy_api(self):
        fake = FakeVolvo()
        fake.route("GET", CONNECTED, details("DIESEL", "V60"))
        fake.route("GET", f"{CONNECTED}/command-accessibility", {"data": {"availabilityStatus": value("AVAILABLE")}})
        fake.route("GET", f"{CONNECTED}/engine-status", {"data": {"engineStatus": value("STOPPED")}})
        fake.route("GET", f"{CONNECTED}/statistics", {"data": {"distanceToEmptyTank": value(540, unit="km")}})
        fake.route("GET", f"{CONNECTED}/fuel", {"data": {"fuelAmount": value(38, unit="l")}})
        fake.route("GET", LOCATION, location())
        with _config():
            integration = build(fake)
            state = integration.vehicle_state(spec(integration))

        assert state.battery_percent is None and state.fuel_percent is None and state.range_km == 540
        paths = {c.path for c in fake.api_calls()}
        assert f"{CONNECTED}/fuel" in paths and ENERGY_STATE not in paths

    def test_a_hybrid_adds_the_electric_and_the_tank_range_and_reads_its_battery(self):
        fake = FakeVolvo()
        fake.route("GET", CONNECTED, details("PETROL/ELECTRIC", "XC60"))
        fake.route("GET", f"{CONNECTED}/command-accessibility", {"data": {"availabilityStatus": value("AVAILABLE")}})
        fake.route("GET", f"{CONNECTED}/engine-status", {"data": {"engineStatus": value("STOPPED")}})
        fake.route("GET", f"{CONNECTED}/statistics", {"data": {"distanceToEmptyTank": value(400, unit="km"), "distanceToEmptyBattery": value(35, unit="km")}})
        fake.route("GET", ENERGY_STATE, {"batteryChargeLevel": energy_value("64.5", unit="percentage"), "electricRange": energy_value("38", unit="km")})
        fake.route("GET", f"{CONNECTED}/fuel", {"data": {"fuelAmount": value(30, unit="l")}})
        with _config():
            integration = build(fake)
            state = integration.vehicle_state(spec(integration))

        assert state.battery_percent == 64 and state.range_km == 438
        assert state.latitude is None       # no location route: the 404 of an unsupported region is not an error

    def test_a_car_whose_energy_api_fails_still_gets_a_line_from_the_other_reads(self):
        fake = FakeVolvo()
        electric_car(fake)
        fake.route("GET", ENERGY_STATE, {"error": {"message": "INTERNAL_ERROR", "description": "try later"}}, status=500)
        with _config():
            integration = build(fake)
            state = integration.vehicle_state(spec(integration))

        assert state.battery_percent is None and state.range_km == 290     # the statistics' battery range stands in
        assert state.latitude == 59.3456

    def test_a_car_none_of_whose_reads_answer_is_an_error_not_an_empty_line(self):
        fake = FakeVolvo()
        with _config():
            integration = build(fake)
            with pytest.raises(VolvoApiError, match="none of the state reads"):
                integration.vehicle_state(spec(integration))

    def test_the_state_is_cached_for_a_minute_and_the_details_for_good(self):
        fake = FakeVolvo()
        electric_car(fake)
        with _config():
            integration = build(fake)
            car = spec(integration)
            first = integration.vehicle_state(car)
            calls_after_first = len(fake.api_calls())
            assert integration.vehicle_state(car) is first
            assert len(fake.api_calls()) == calls_after_first

            with patch.object(module.time, "time", return_value=time.time() + module.STATE_TTL + 1):
                integration.vehicle_state(car)
            assert len(fake.api_calls()) == 2 * calls_after_first - 1       # everything but the vehicle details again

    def test_a_rate_limited_api_is_one_clear_error_not_five_failed_reads(self):
        fake = FakeVolvo()
        electric_car(fake)
        fake.route("GET", f"{CONNECTED}/command-accessibility", {"error": {"message": "TOO_MANY_REQUESTS"}}, status=429)
        with _config():
            integration = build(fake)
            with pytest.raises(VolvoRateLimited, match="rate limit"):
                integration.vehicle_state(spec(integration))
        assert len(fake.api_calls()) == 2       # the details, then the first refused read stopped the rest

    def test_an_unsigned_in_integration_raises_the_login_hint_before_calling_the_api(self):
        fake = FakeVolvo()
        electric_car(fake)
        with _config():
            integration = build(fake, StubAuth(signed_in=False))
            with pytest.raises(NotSignedIn, match="volvo_login"):
                integration.vehicle_state(spec(integration))
        assert fake.api_calls() == []

    def test_a_refused_call_for_a_car_the_volvo_id_owns_names_the_scopes_and_the_sign_in(self):
        fake = FakeVolvo()
        fake.route("GET", CONNECTED, {"error": {"message": "FORBIDDEN", "description": "scope missing"}}, status=403)
        fake.route("GET", CAR_LIST, {"data": [{"vin": VIN}]})
        with _config():
            integration = build(fake)
            with pytest.raises(VolvoApiError, match="scope") as refused:
                integration.vehicle_state(spec(integration))
        assert "VIN" not in str(refused.value).split("—")[1] or "is the Volvo ID's" in str(refused.value)
        assert "volvo_login" in str(refused.value)


@pytest.mark.unit
class TestWrongVin:
    """Volvo answers a VIN that is not the Volvo ID's with 401 "invalid VCC-API-KEY" — the words of
    a bad key — so the integration reads the account's car list once before it says what to fix."""
    REFUSED = {"error": {"message": "invalid VCC-API-KEY", "description": ""}}

    def test_a_vin_the_account_lacks_is_blamed_on_the_vin_with_the_cars_the_account_has(self):
        fake = FakeVolvo()
        fake.route("GET", CONNECTED, self.REFUSED, status=401)
        fake.route("GET", CAR_LIST, {"data": [{"vin": "YV1KEP47L00000007"}, {"vin": "YV1KEP47L00000008"}]})
        with _config():
            integration = build(fake)
            with pytest.raises(VolvoApiError) as refused:
                integration.vehicle_state(spec(integration))
        text = str(refused.value)
        assert "invalid VCC-API-KEY" in text and VIN in text and "volvo.vehicles" in text
        assert "…000007" in text and "…000008" in text and "2 car" in text
        assert "vcc_api_key" not in text, "the key is fine, the message must not send the user after it"

    def test_a_car_list_refused_too_blames_the_key(self):
        fake = FakeVolvo()
        fake.route("GET", CONNECTED, self.REFUSED, status=401)
        fake.route("GET", CAR_LIST, self.REFUSED, status=401)
        with _config():
            integration = build(fake)
            with pytest.raises(VolvoApiError) as refused:
                integration.vehicle_state(spec(integration))
        text = str(refused.value)
        assert "volvo.vcc_api_key" in text and "volvo_login" in text
        assert "volvo.vehicles" not in text

    def test_the_car_list_is_read_once_for_the_process(self):
        fake = FakeVolvo()
        fake.route("GET", CONNECTED, self.REFUSED, status=401)
        fake.route("GET", CAR_LIST, {"data": [{"vin": "YV1KEP47L00000007"}]})
        with _config():
            integration = build(fake)
            for _ in range(2):
                with pytest.raises(VolvoApiError, match="volvo.vehicles"):
                    integration.vehicle_state(spec(integration))
        assert len(fake.sent("GET", CAR_LIST)) == 1

    def test_the_health_check_carries_the_verdict(self):
        fake = FakeVolvo()
        fake.route("GET", CONNECTED, self.REFUSED, status=401)
        fake.route("GET", CAR_LIST, {"data": []})
        with _config():
            integration = build(fake)
            with pytest.raises(VolvoApiError, match="volvo.vehicles"):
                integration.check_health()

    def test_a_rate_limit_while_diagnosing_is_reported_as_such(self):
        fake = FakeVolvo()
        fake.route("GET", CONNECTED, self.REFUSED, status=401)
        fake.route("GET", CAR_LIST, {"error": {"message": "TOO_MANY_REQUESTS"}}, status=429)
        with _config():
            integration = build(fake)
            with pytest.raises(VolvoRateLimited):
                integration.vehicle_state(spec(integration))


@pytest.mark.unit
class TestCommands:
    def test_climate_on_posts_climatization_start_and_a_completed_status_is_the_reply(self):
        fake = FakeVolvo()
        electric_car(fake)
        fake.route("POST", f"{CONNECTED}/commands/climatization-start", command_result("COMPLETED"))
        with _config():
            integration = build(fake)
            car = spec(integration)
            integration.vehicle_state(car)
            reply = integration.climate_on(car)

        assert "Family XC40" in reply and "started" in reply
        post = fake.sent("POST", f"{CONNECTED}/commands/climatization-start")[0]
        assert post.headers["Content-Type"] == "application/json" and post.headers["vcc-api-key"] == "vcc-key"

    def test_a_command_drops_the_cached_state_so_the_next_read_is_fresh(self):
        fake = FakeVolvo()
        electric_car(fake)
        fake.route("POST", f"{CONNECTED}/commands/climatization-stop", command_result("DELIVERED"))
        with _config():
            integration = build(fake)
            car = spec(integration)
            integration.vehicle_state(car)
            reads = len(fake.api_calls())
            assert "stopped" in integration.climate_off(car)
            integration.vehicle_state(car)
        assert len(fake.api_calls()) > reads + 1

    def test_a_rejected_command_is_an_error_with_volvos_reason(self):
        fake = FakeVolvo()
        electric_car(fake)
        fake.route("POST", f"{CONNECTED}/commands/climatization-start", command_result("REJECTED", "Car in use"))
        with _config():
            integration = build(fake)
            with pytest.raises(VolvoCommandError, match="REJECTED.*Car in use"):
                integration.climate_on(spec(integration))

    def test_a_command_the_api_refuses_is_the_api_error(self):
        fake = FakeVolvo()
        fake.route("POST", f"{CONNECTED}/commands/climatization-start", {"error": {"message": "UNPROCESSABLE", "description": "not supported by this car"}}, status=422)
        with _config():
            integration = build(fake)
            with pytest.raises(VolvoApiError, match="not supported"):
                integration.climate_on(spec(integration))

    def test_defrost_trunk_and_navigation_are_unsupported_without_a_call(self):
        fake = FakeVolvo()
        with _config():
            integration = build(fake)
            car = spec(integration)
            with pytest.raises(Unsupported, match="defrost"):
                integration.max_defrost(car)
            with pytest.raises(Unsupported, match="trunk"):
                integration.open_trunk(car)
            with pytest.raises(Unsupported, match="destination"):
                integration.navigate_to(car, "Stockholm")
        assert fake.api_calls() == []


@pytest.mark.unit
class TestHealthAndSignIn:
    def test_health_is_the_first_cars_details(self):
        fake = FakeVolvo()
        fake.route("GET", CONNECTED, details())
        with _config():
            integration = build(fake)
            assert integration.check_health() is True
            assert not integration.needs_2fa()

    def test_health_without_a_sign_in_raises_the_hint_and_needs_2fa_says_the_owner_must_sign_in(self):
        with _config():
            integration = build(FakeVolvo(), StubAuth(signed_in=False))
            assert integration.needs_2fa()
            with pytest.raises(NotSignedIn):
                integration.check_health()
            assert build(FakeVolvo(), StubAuth(again=True)).needs_2fa()

    def test_the_login_hint_names_the_command(self):
        assert "volvo_login" in VolvoIntegration.login_hint

    def test_the_auth_status_for_the_tab(self):
        with _config():
            assert build(FakeVolvo(), StubAuth(signed_in=False)).get_auth_status()["status"] == "needs_auth"
            assert build(FakeVolvo(), StubAuth(again=True)).get_auth_status()["status"] == "needs_auth"
            assert build(FakeVolvo()).get_auth_status()["status"] == "authenticated"
        with _config(client_secret=""):
            assert build(FakeVolvo()).get_auth_status()["status"] == "error"

    def test_a_sign_in_or_sign_out_drops_the_cached_states(self):
        fake = FakeVolvo()
        electric_car(fake)
        with _config():
            integration = build(fake)
            car = spec(integration)
            integration.vehicle_state(car)
            reads = len(fake.api_calls())
            integration.auth.sign_out = lambda: None
            integration.sign_out()
            integration.vehicle_state(car)
        assert len(fake.api_calls()) > reads

    def test_wipe_targets_log_out_and_losing_the_config_remove_the_token_file(self, fresh):
        fresh.mkdir(parents=True)
        token_path().write_text("{}")
        assert [p for p, _ in VolvoIntegration.wipe_targets()] == [token_path()]

        assert VolvoIntegration.log_out()
        assert not token_path().exists()
        assert VolvoIntegration.log_out() is None

        token_path().write_text("{}")
        with _config():
            build(FakeVolvo()).on_not_configured()
        assert not token_path().exists()


@pytest.mark.unit
class TestTransport:
    def test_a_timeout_is_retried_and_the_second_attempt_serves(self):
        fake = FakeVolvo()
        electric_car(fake)
        fake.transport_errors = [requests.exceptions.Timeout("slow")]
        with _config():
            integration = build(fake)
            state = integration.vehicle_state(spec(integration))
        assert state.battery_percent == 72 and fake.transport_errors == []

    def test_an_api_that_stays_unreachable_is_one_error_once_the_retries_are_spent(self):
        fake = FakeVolvo()
        electric_car(fake)
        fake.transport_errors = [requests.exceptions.ConnectionError("down")] * module.RETRIES
        with _config():
            integration = build(fake)
            with pytest.raises(VolvoApiError, match="could not be reached"):
                integration.vehicle_state(spec(integration))
        assert fake.transport_errors == [] and fake.api_calls() == []

    def test_a_token_volvo_no_longer_accepts_points_at_the_sign_in(self):
        fake = FakeVolvo()
        # A dead token is refused on every call, the car list included — which is how the
        # refusal is told apart from a VIN the Volvo ID lacks (TestWrongVin).
        fake.route("GET", CONNECTED, {"error": {"message": "UNAUTHORIZED"}}, status=401)
        fake.route("GET", CAR_LIST, {"error": {"message": "UNAUTHORIZED"}}, status=401)
        with _config():
            integration = build(fake)
            with pytest.raises(VolvoApiError, match="volvo_login"):
                integration.vehicle_state(spec(integration))


@pytest.mark.unit
class TestVehicleStateEdges:
    def test_a_car_without_details_is_read_as_a_battery_car_and_asked_for_them_again_next_time(self):
        fake = FakeVolvo()
        electric_car(fake)
        del fake.routes[("GET", CONNECTED)]
        with _config():
            integration = build(fake)
            car = spec(integration)
            state = integration.vehicle_state(car)
            with patch.object(module.time, "time", return_value=time.time() + module.STATE_TTL + 1):
                integration.vehicle_state(car)
        assert state.battery_percent == 72
        paths = {c.path for c in fake.api_calls()}
        assert ENERGY_STATE in paths and f"{CONNECTED}/fuel" not in paths
        assert len(fake.sent("GET", CONNECTED)) == 2, "an empty answer is not cached the way details are"

    def test_a_value_volvo_cannot_give_is_unknown_not_a_crash(self):
        fake = FakeVolvo()
        electric_car(fake, battery="N/A")
        with _config():
            integration = build(fake)
            state = integration.vehicle_state(spec(integration))
        assert state.battery_percent is None and state.range_km == 310

    def test_the_data_time_is_the_newest_stamp_and_an_unreadable_one_is_skipped(self):
        fake = FakeVolvo()
        electric_car(fake, stamp="2026-10-09T08:15:00Z")
        fake.route("GET", LOCATION, location(stamp="2026-10-09T09:00:00Z"))
        fake.route("GET", f"{CONNECTED}/engine-status", {"data": {"engineStatus": value("STOPPED", "garbage")}})
        with _config():
            integration = build(fake)
            state = integration.vehicle_state(spec(integration))
        assert state.data_time == datetime(2026, 10, 9, 9, 0, tzinfo=timezone.utc).timestamp()

    def test_a_hybrid_whose_energy_api_is_down_reads_its_battery_from_the_fuel_endpoint(self):
        fake = FakeVolvo()
        fake.route("GET", CONNECTED, details("PETROL/ELECTRIC", "XC60"))
        fake.route("GET", f"{CONNECTED}/statistics", {"data": {"distanceToEmptyTank": value(400, unit="km")}})
        fake.route("GET", ENERGY_STATE, {"error": {"message": "INTERNAL_ERROR"}}, status=500)
        fake.route("GET", f"{CONNECTED}/fuel", {"data": {"fuelAmount": value(30, unit="l"),
                                                         "batteryChargeLevel": value(55, unit="percentage")}})
        with _config():
            integration = build(fake)
            state = integration.vehicle_state(spec(integration))
        assert state.battery_percent == 55 and state.range_km == 400


@pytest.mark.unit
class TestCommandAnswers:
    def test_an_answer_without_the_data_wrapper_is_read_flat_and_an_empty_one_is_no_answer(self):
        fake = FakeVolvo()
        fake.route("POST", f"{CONNECTED}/commands/climatization-start", {"vin": VIN, "invokeStatus": "COMPLETED"})
        fake.route("POST", f"{CONNECTED}/commands/climatization-stop", {"data": {}})
        with _config():
            integration = build(fake)
            car = spec(integration)
            assert "started" in integration.climate_on(car)
            with pytest.raises(VolvoCommandError, match="no answer"):
                integration.climate_off(car)

    def test_a_rejected_command_keeps_the_cached_state(self):
        fake = FakeVolvo()
        electric_car(fake)
        fake.route("POST", f"{CONNECTED}/commands/climatization-start", command_result("REJECTED", "Car in use"))
        with _config():
            integration = build(fake)
            car = spec(integration)
            first = integration.vehicle_state(car)
            with pytest.raises(VolvoCommandError):
                integration.climate_on(car)
            assert integration.vehicle_state(car) is first


@pytest.mark.unit
class TestHealthAndSignInEdges:
    def test_health_without_a_car_or_a_vin_is_false_without_a_call(self):
        fake = FakeVolvo()
        with _config(vehicles=[]):
            assert build(fake).check_health() is False
        with _config(vehicles=[{"name": "nameless"}]):
            assert build(fake).check_health() is False
        assert fake.api_calls() == []

    def test_a_finished_sign_in_drops_the_cached_states_a_failed_one_keeps_them(self):
        fake = FakeVolvo()
        electric_car(fake)
        pasted = "http://localhost:18723/?code=c0de&state=st-1"
        with _config():
            integration = build(fake)
            car = spec(integration)
            first = integration.vehicle_state(car)
            integration.auth.finish = lambda p: "the pasted address belongs to another sign-in"
            assert integration.login_finish(pasted)
            assert integration.vehicle_state(car) is first
            integration.auth.finish = lambda p: ""
            assert integration.login_finish(pasted) == ""
            assert integration.vehicle_state(car) is not first

    def test_a_missing_api_key_is_an_error_even_with_the_client_set(self):
        with _config(api_key=""):
            assert build(FakeVolvo()).get_auth_status()["status"] == "error"

    def test_an_entry_that_is_not_a_mapping_is_skipped(self):
        with _config(vehicles=["oops", VEHICLES[0]]):
            assert [c.id for c in build(FakeVolvo()).vehicles()] == ["family_xc40"]
