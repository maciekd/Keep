"""The integration against a fake HomeClaw app: what one refresh asks, how every write goes
out (by UUID, verified, a zone by its service, a dry run flagged), the guards that keep a
request from going out at all, and the feed of changes."""
import pytest

from integrations.homeclaw.HomeClawClient import HomeClawError, HomeClawUnavailable
from integrations.homeclaw.HomeClawIntegration import HomeClawIntegration
from integrations.homeclaw.HomeClawModel import Unresolved
from .conftest import (
    AWNING, FLOOR_LAMP, GARAGE_DOOR, HUE_LAMP, PLEJD_LAMP, SENSOR_A, THERMOSTAT, WATER_CONTROL, ZONE_PATIO, ok,
    settings_patch)


class TestHealth:
    def test_a_ready_app_with_a_home_is_healthy(self, homeclaw):
        assert homeclaw.check_health() is True

    def test_an_app_still_loading_is_not(self, app, homeclaw):
        app.replies["status"] = ok({"ready": False, "homes": 1})
        with pytest.raises(HomeClawError, match="not ready"):
            homeclaw.check_health()

    def test_a_missing_socket_is_reported_as_not_running(self):
        with settings_patch({"homeclaw.socket": "/tmp/no-such-dir-for-homeclaw/homeclaw.sock"}):
            with pytest.raises(HomeClawUnavailable, match="not running"):
                HomeClawIntegration().check_health()


class TestSnapshot:
    def test_hue_and_plejd_stay_out_and_the_rest_is_shown_by_room(self, homeclaw):
        text = homeclaw.snapshot()
        assert PLEJD_LAMP[:8] not in text and HUE_LAMP[:8] not in text
        assert FLOOR_LAMP[:8] in text and "## Game room" in text
        assert "zone Patio" in text and "zone Greenhouse" in text

    def test_the_zones_are_read_from_the_apps_cache_once_per_irrigation_control(self, app, homeclaw):
        homeclaw.snapshot()
        assert app.of("get_accessory") == [{"id": WATER_CONTROL, "refresh": False}]

    def test_an_irrigation_control_whose_zones_cannot_be_read_stays_one_valve(self, app, homeclaw):
        app.replies["get_accessory"] = {"success": False, "error": "timeout"}
        homeclaw.snapshot()
        assert homeclaw.set_valve(WATER_CONTROL[:8], True).endswith("watering.")
        assert "service_id" not in app.of("control")[-1]

    def test_the_include_list_brings_a_plejd_accessory_back(self, app):
        with settings_patch({"homeclaw.socket": app.path, "homeclaw.include": [PLEJD_LAMP]}):
            assert PLEJD_LAMP[:8] in HomeClawIntegration().snapshot()


class TestControl:
    def test_a_cover_position_goes_out_by_uuid_with_verify(self, app, homeclaw):
        reply = homeclaw.set_cover("Awning Roof", 40)
        sent = app.of("control")[-1]
        assert (sent["id"], sent["characteristic"], sent["value"], sent["verify"]) == (AWNING, "target_position", "40", "true")
        assert "dry_run" not in sent and "40" in reply

    def test_a_zone_is_addressed_by_its_valve_service(self, app, homeclaw):
        homeclaw.set_valve("Patio", True)
        sent = app.of("control")[-1]
        assert (sent["id"], sent["service_id"], sent["characteristic"], sent["value"]) == (WATER_CONTROL, ZONE_PATIO, "active", "1")

    def test_a_dry_run_is_flagged_and_says_nothing_changed(self, app, homeclaw):
        reply = homeclaw.set_garage("", True, dry_run=True)
        sent = app.of("control")[-1]
        assert (sent["id"], sent["value"], sent["dry_run"]) == (GARAGE_DOOR, "open", "true")
        assert "nothing was changed" in reply

    def test_a_dry_run_the_app_finds_invalid_says_it_would_be_refused(self, app, homeclaw):
        app.replies["control"] = ok({"valid": False, "error": "value out of range"})
        assert "refuse" in homeclaw.set_light(FLOOR_LAMP[:8], False, dry_run=True)

    def test_a_write_the_accessory_does_not_confirm_is_reported(self, app, homeclaw):
        app.replies["control"] = ok({"verified": False, "current_value": "closed"})
        reply = homeclaw.set_garage(GARAGE_DOOR[:8], True)
        assert "closed" in reply and "reports" in reply

    def test_a_thermostat_target_outside_the_range_never_reaches_the_app(self, app, homeclaw):
        with pytest.raises(HomeClawError, match="range"):
            homeclaw.set_thermostat(THERMOSTAT[:8], 35)
        assert app.of("control") == []
        homeclaw.set_thermostat("", 21.5)
        assert app.of("control")[-1]["value"] == "21.5"

    def test_an_ambiguous_name_sends_nothing(self, app, homeclaw):
        with pytest.raises(Unresolved):
            homeclaw.set_light("Climate Sensor T1", True)
        assert app.of("control") == []

    def test_an_unknown_name_reads_the_app_again_before_giving_up(self, app, homeclaw):
        homeclaw.snapshot()
        reads = len(app.of("list_accessories"))
        with pytest.raises(Unresolved):
            homeclaw.set_cover("Shed Door", 0)
        assert len(app.of("list_accessories")) == reads + 1


class TestScenes:
    def test_a_scene_runs_by_name_whatever_the_case(self, app, homeclaw):
        homeclaw.trigger_scene("movie evening")
        assert app.of("trigger_scene") == [{"id": "F1000002-0000-4000-8000-000000000002"}]

    def test_a_dry_run_triggers_nothing(self, app, homeclaw):
        assert "would run" in homeclaw.trigger_scene("Good Night", dry_run=True)
        assert app.of("trigger_scene") == []

    def test_the_hidden_uuid_scenes_cannot_be_matched_by_a_fragment(self, app, homeclaw):
        with pytest.raises(Unresolved):
            homeclaw.trigger_scene("0000000000AA")


class TestEvents:
    def _event(self, uid, when, characteristic, value, before="--"):
        return {"type": "characteristic_change", "timestamp": when, "characteristic": characteristic,
                "previous_value": before, "value": value, "accessory": {"id": uid}}

    def test_changes_come_oldest_first_minus_the_skipped_and_the_left_out(self, app, homeclaw):
        app.replies["events"] = ok({"events": [
            self._event(GARAGE_DOOR, "2026-03-01T10:05:00Z", "current_door_state", "closed", "open"),
            self._event(SENSOR_A, "2026-03-01T10:03:00Z", "current_temperature", "16.2°C"),
            self._event(PLEJD_LAMP, "2026-03-01T10:02:00Z", "power", "true"),
            self._event(AWNING, "2026-03-01T10:01:00Z", "current_position", "0", "100"),
        ]})
        entries = homeclaw.events("2h")
        assert [("Awning" in e.content, "Garage" in e.content) for e in entries] == [(True, False), (False, True)]
        assert app.of("events")[-1]["since"] == "2h"

    def test_the_skip_list_comes_from_the_config(self, app):
        app.replies["events"] = ok({"events": [self._event(SENSOR_A, "2026-03-01T10:03:00Z", "current_temperature", "16.2°C")]})
        with settings_patch({"homeclaw.socket": app.path, "homeclaw.event_skip": []}):
            assert len(HomeClawIntegration().events("1h")) == 1

    def test_the_app_not_running_is_said_not_raised_past_the_client(self):
        with settings_patch({"homeclaw.socket": "/tmp/no-such-dir-for-homeclaw/homeclaw.sock"}):
            with pytest.raises(HomeClawUnavailable, match="not running"):
                HomeClawIntegration().events("1h")


class TestHealthWithoutAHome:
    def test_a_ready_app_that_sees_no_home_is_not_healthy_and_says_what_to_check(self, app, homeclaw):
        app.replies["status"] = ok({"ready": True, "homes": 0, "accessories": 0})
        with pytest.raises(HomeClawError, match="HomeKit access"):
            homeclaw.check_health()


class TestNewAccessories:
    def test_an_accessory_added_since_the_last_read_gets_its_row(self, app, homeclaw):
        """The index is from the last refresh; an event of an accessory it does not know is
        the sign the house changed, so the app is read again rather than the change dropped."""
        from .conftest import ACCESSORIES
        shed_lamp = "B7000001-0000-4000-8000-000000000001"
        homeclaw.snapshot()
        reads = len(app.of("list_accessories"))
        app.replies["list_accessories"] = ok(ACCESSORIES + [
            {"id": shed_lamp, "name": "Shed Lamp", "room": "Shed", "category": "lightbulb", "manufacturer": "Nanoleaf",
             "reachable": True, "bridge": None, "state": {"power": "false"}}])
        app.replies["events"] = ok({"events": [
            {"type": "characteristic_change", "timestamp": "2026-03-01T10:00:00Z", "characteristic": "power",
             "previous_value": "false", "value": "true", "accessory": {"id": shed_lamp}}]})
        entries = homeclaw.events("1h")
        assert len(app.of("list_accessories")) == reads + 1
        assert len(entries) == 1 and "Shed Lamp [Shed]" in entries[0].content

    def test_events_of_known_accessories_read_the_app_no_second_time(self, app, homeclaw):
        homeclaw.snapshot()
        reads = len(app.of("list_accessories"))
        app.replies["events"] = ok({"events": [
            {"type": "characteristic_change", "timestamp": "2026-03-01T10:00:00Z", "characteristic": "current_door_state",
             "previous_value": "closed", "value": "open", "accessory": {"id": GARAGE_DOOR}}]})
        assert len(homeclaw.events("1h")) == 1
        assert len(app.of("list_accessories")) == reads
