"""HomeClawIntegration with the client faked: the health check, the snapshot, each command's
`control`, the re-read on an unknown name, the app's absence as text, and the recent events."""
from unittest.mock import patch

import pytest

from assistant.integration.HomeAutomation import render
from integrations.homeclaw.HomeClawClient import HomeClawError, HomeClawUnavailable
from integrations.homeclaw.HomeClawIntegration import EVENT_LIMIT, HomeClawIntegration

MODULE = "integrations.homeclaw.HomeClawIntegration"
VALVE = "000000D0-0000-1000-8000-0026BB765291"


def accessory(uid, name, room, category, state=None, **extra):
    return {"id": uid, "name": name, "room": room, "category": category, "state": state, "reachable": True,
            "manufacturer": "Acme", **extra}


class FakeClient:
    def __init__(self, socket_path=""):
        self.socket_path = socket_path
        self.calls: list[tuple[str, dict]] = []
        self.failure: Exception | None = None
        self.status = {"ready": True, "homes": 1, "accessories": 6}
        self.control_reply = {"verified": True, "new_value": "set"}
        self.events: list[dict] = []
        self.accessories = [
            accessory("l1", "Nanoleaf Panels", "Playroom", "lightbulb", {"brightness": "70", "power": "true"}),
            accessory("c1", "Pergola Roof", "Terrace", "window_covering", {"current_position": "40", "target_position": "40"}),
            accessory("d1", "Garage Door", "Garage", "garage_door", {"current_door_state": "closed"}),
            accessory("v1", "Dual Water Control", "Greenhouse", "irrigation", {"active": "0"}),
            accessory("t1", "Child Radiator", "Child Room", "thermostat", {"current_temperature": "19.5°C", "target_temperature": "21.0°C"}),
            accessory("a1", "Aqara Sensor", "Cellar", "sensor", {"current_temperature": "16.1°C"}),
            accessory("p1", "Kitchen Spots", "Kitchen", "lightbulb", {"power": "true"}, manufacturer="Plejd",
                      bridge={"manufacturer": "Plejd", "model": "GWY-01", "name": "Plejd Gateway"}),
        ]
        self.scenes = [{"id": "s1", "name": "Good Night", "type": "sleep"}]

    def call(self, command, **args):
        self.calls.append((command, args))
        if self.failure:
            raise self.failure
        if command == "status":
            return self.status
        if command == "list_accessories":
            return self.accessories
        if command == "list_scenes":
            return self.scenes
        if command == "get_accessory":
            return {"services": [
                {"type": VALVE, "name": "Terrace", "characteristics": [{"name": "active", "value": "0"}]},
                {"type": VALVE, "name": "Greenhouse", "characteristics": [{"name": "active", "value": "1"}]},
                {"type": "0000003E-0000-1000-8000-0026BB765291", "name": "Dual Water Control", "characteristics": []},
            ]}
        if command == "control":
            return self.control_reply
        if command == "trigger_scene":
            return {"triggered": True}
        if command == "events":
            return {"events": self.events}
        raise AssertionError(command)

    def controls(self):
        return [args for command, args in self.calls if command == "control"]


@pytest.fixture
def integration():
    HomeClawIntegration.reset_instance()
    with patch(f"{MODULE}.HomeClawClient", FakeClient), patch(f"{MODULE}.config") as config:
        config.get.side_effect = lambda key, default=None: {"homeclaw.socket": "/tmp/hc.sock"}.get(key, default)
        yield HomeClawIntegration()
    HomeClawIntegration.reset_instance()


@pytest.fixture
def fake(integration) -> FakeClient:
    return integration.client()


@pytest.mark.unit
class TestHealth:
    def test_a_ready_app_with_a_home_is_healthy(self, integration):
        assert integration.check_health() is True

    def test_an_app_still_loading_or_without_a_home_says_so(self, integration, fake):
        fake.status = {"ready": False, "homes": 1}
        with pytest.raises(RuntimeError, match="not ready"):
            integration.check_health()
        fake.status = {"ready": True, "homes": 0}
        with pytest.raises(RuntimeError, match="no HomeKit home"):
            integration.check_health()

    def test_the_app_not_running_reaches_the_monitor_as_its_own_words(self, integration, fake):
        fake.failure = HomeClawUnavailable("the HomeClaw app is not running (no socket at /tmp/hc.sock)")
        with pytest.raises(HomeClawUnavailable, match="not running"):
            integration.check_health()

    def test_the_client_takes_the_configured_socket_and_needs_no_config_keys(self, integration, fake):
        assert fake.socket_path == "/tmp/hc.sock"
        assert integration.missing_config_keys() == []


@pytest.mark.unit
class TestSnapshot:
    def test_lists_the_house_minus_the_excluded_bridge_with_the_zones_read_from_the_app(self, integration, fake):
        text = render(integration.snapshot())
        assert '"Nanoleaf Panels [Playroom]": on 70%' in text
        assert '"Dual Water Control Greenhouse [Greenhouse]": watering' in text
        assert '"Child Radiator [Child Room]": 19.5°C, target 21.0°C' in text
        assert '"Good Night": scene, sleep' in text
        assert "Kitchen Spots" not in text
        assert [c for c, _ in fake.calls] == ["list_accessories", "list_scenes", "get_accessory"]
        assert fake.calls[2][1] == {"id": "v1", "refresh": False}

    def test_the_exclusion_list_comes_from_config(self, integration, fake):
        with patch(f"{MODULE}.config") as config:
            config.get.side_effect = lambda key, default=None: {"homeclaw.exclude": ["Acme"]}.get(key, default)
            text = render(integration.snapshot())
        assert "Kitchen Spots" in text and "Nanoleaf" not in text

    def test_a_control_the_app_cannot_describe_stays_one_entity(self, integration, fake):
        original = fake.call

        def call(command, **args):
            if command == "get_accessory":
                raise HomeClawError("Accessory not found")
            return original(command, **args)

        fake.call = call
        assert '"Dual Water Control [Greenhouse]": off' in render(integration.snapshot())

    def test_an_absent_app_raises_so_the_section_keeps_its_last_content(self, integration, fake):
        fake.failure = HomeClawUnavailable("the HomeClaw app is not running")
        with pytest.raises(HomeClawUnavailable):
            integration.snapshot()


@pytest.mark.unit
class TestCommands:
    def test_every_write_is_a_verified_control_on_the_uuid(self, integration, fake):
        assert integration.turn_on("Nanoleaf Panels [Playroom]") == ""
        assert integration.turn_off("Pergola Roof [Terrace]") == ""
        assert integration.turn_on("Garage Door [Garage]") == ""
        assert fake.controls() == [
            {"id": "l1", "characteristic": "power", "value": "true", "verify": True, "service_name": None},
            {"id": "c1", "characteristic": "target_position", "value": "0", "verify": True, "service_name": None},
            {"id": "d1", "characteristic": "target_door_state", "value": "open", "verify": True, "service_name": None},
        ]

    def test_a_zone_names_its_service(self, integration, fake):
        assert integration.turn_on("Dual Water Control Greenhouse [Greenhouse]") == ""
        assert fake.controls()[-1] == {"id": "v1", "characteristic": "active", "value": "1", "verify": True,
                                       "service_name": "Greenhouse"}

    def test_dim_switches_a_light_on_before_the_brightness_and_drives_a_cover(self, integration, fake):
        assert integration.dim("Nanoleaf Panels", 130) == ""
        assert integration.dim("Pergola Roof", 55) == ""
        assert [(c["characteristic"], c["value"]) for c in fake.controls()] == [
            ("power", "true"), ("brightness", "100"), ("target_position", "55")]

    def test_dim_to_zero_is_off(self, integration, fake):
        assert integration.dim("Nanoleaf Panels", 0) == ""
        assert [(c["characteristic"], c["value"]) for c in fake.controls()] == [("power", "false")]

    def test_stop_holds_a_cover_at_its_position(self, integration, fake):
        assert integration.stop("Pergola Roof") == ""
        assert fake.controls()[-1]["characteristic"] == "target_position" and fake.controls()[-1]["value"] == "40"

    def test_execute_triggers_the_scene_by_id(self, integration, fake):
        assert integration.execute("Good Night") == ""
        assert ("trigger_scene", {"id": "s1"}) in fake.calls

    @pytest.mark.parametrize("verb, name, hint", [
        ("execute", "Garage Door", "garage door"),
        ("turn_on", "Good Night", "use execute"),
        ("turn_on", "Child Radiator", "homeclaw_set"),
        ("turn_on", "Aqara Sensor", "sensor reading"),
        ("dim", "Garage Door", "garage door"),
        ("stop", "Nanoleaf Panels", "is a light"),
    ])
    def test_the_wrong_command_says_which_to_use(self, integration, fake, verb, name, hint):
        args = (name, 50) if verb == "dim" else (name,)
        assert hint in getattr(integration, verb)(*args)
        assert fake.controls() == []

    def test_an_unknown_name_reads_the_app_again_then_answers(self, integration, fake):
        integration.snapshot()
        fake.accessories.append(accessory("l9", "New Lamp", "Hall", "lightbulb", {"power": "false"}))
        assert integration.turn_on("New Lamp") == ""
        assert fake.controls()[-1]["id"] == "l9"
        assert "No HomeClaw accessory or scene" in integration.turn_on("Nothing")

    def test_the_apps_refusal_and_its_absence_are_the_answer(self, integration, fake):
        integration.snapshot()
        fake.failure = HomeClawError("Invalid value: Cannot parse 'xyz' for power")
        assert integration.turn_on("Nanoleaf Panels") == "Invalid value: Cannot parse 'xyz' for power"
        fake.failure = TimeoutError("slow")
        assert integration.turn_on("Nanoleaf Panels") == "Request timed out"
        fake.failure = HomeClawUnavailable("the HomeClaw app is not running")
        assert "not running" in integration.turn_on("Nanoleaf Panels")

    def test_set_characteristic_reports_the_verified_value(self, integration, fake):
        fake.control_reply = {"verified": True, "new_value": "22.0°C", "current_value": "21.0°C"}
        text = integration.set_characteristic("Child Radiator (Child Room)", "target_temperature", "22")
        assert text == "Child Radiator [Child Room]: target_temperature is now 22.0°C."
        assert fake.controls()[-1] == {"id": "t1", "characteristic": "target_temperature", "value": "22",
                                       "verify": True, "service_name": None}

    def test_set_characteristic_says_when_the_accessory_reports_otherwise(self, integration, fake):
        fake.control_reply = {"verified": False, "current_value": "21.0°C"}
        text = integration.set_characteristic("Child Radiator", "target_temperature", "22")
        assert "21.0°C" in text and "but" in text

    def test_set_characteristic_on_a_scene(self, integration, fake):
        assert "use execute" in integration.set_characteristic("Good Night", "power", "true")
        assert fake.controls() == []


@pytest.mark.unit
class TestRecentEvents:
    def test_asks_the_app_for_the_window_in_its_own_time_format(self, integration, fake):
        with patch(f"{MODULE}.time") as clock:
            clock.time.return_value = 1792174800.0            # 2026-10-16T18:20:00Z
            assert integration.recent_events(2) == []
        assert ("events", {"limit": EVENT_LIMIT, "since": "2026-10-16T16:20:00Z"}) in fake.calls

    def test_rows_come_oldest_first_minus_the_ticks_and_the_other_integrations_accessories(self, integration, fake):
        fake.events = [
            {"type": "characteristic_change", "accessory": {"id": "d1"}, "characteristic": "current_door_state",
             "value": "open", "service": "Garage Door", "timestamp": "2026-10-16T17:10:30Z"},
            {"type": "characteristic_change", "accessory": {"id": "l1"}, "characteristic": "brightness",
             "value": "53", "service": "Nanoleaf Panels", "timestamp": "2026-10-16T17:10:10Z"},
            {"type": "characteristic_change", "accessory": {"id": "p1"}, "characteristic": "power",
             "value": "true", "service": "Kitchen Spots", "timestamp": "2026-10-16T17:10:05Z"},
            {"type": "characteristic_change", "accessory": {"id": "l1"}, "characteristic": "power",
             "value": "false", "service": "Nanoleaf Panels", "timestamp": "2026-10-16T17:10:00Z"},
        ]
        entries = integration.recent_events(24)
        assert [e.content for e in entries] == ["Nanoleaf Panels [Playroom] turned off", "Garage Door [Garage] is open"]
        assert entries[0].timestamp < entries[1].timestamp

    def test_the_first_read_takes_a_snapshot_to_know_the_accessories(self, integration, fake):
        fake.events = [{"type": "characteristic_change", "accessory": {"id": "l1"}, "characteristic": "power",
                        "value": "true", "timestamp": "2026-10-16T17:10:00Z"}]
        assert len(integration.recent_events(1)) == 1
        assert [c for c, _ in fake.calls][:2] == ["events", "list_accessories"]

    def test_an_absent_app_raises_for_the_command_to_report(self, integration, fake):
        fake.failure = HomeClawUnavailable("the HomeClaw app is not running")
        with pytest.raises(HomeClawUnavailable):
            integration.recent_events(1)
