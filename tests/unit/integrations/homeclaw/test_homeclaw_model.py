"""HomeClawModel: the exclusion, the numbering of twins, each category's value, the zones, the
scenes, the sort by room, the name index, and the events the feed keeps."""
import pytest

from assistant.integration.HomeAutomation import line_of, render
from integrations.homeclaw.HomeClawModel import (
    DEFAULT_EXCLUDE, NameIndex, Target, Unresolved, build, cover_value, event_time, excluded, feed_entry, iso_time,
    light_value, sensor_value, thermostat_value)

PLEJD_BRIDGE = {"manufacturer": "Plejd", "model": "GWY-01", "name": "Plejd Gateway"}
HUE_BRIDGE = {"manufacturer": "Signify Netherlands B.V.", "model": "BSB002", "name": "Philips hue - 1A2B3C"}
AQARA_HUB = {"manufacturer": "Aqara", "model": "Aqara Hub M2", "name": "Aqara Hub Cellar"}


def accessory(uid, name, room, category, state=None, **extra):
    return {"id": uid, "name": name, "room": room, "category": category, "state": state, "reachable": True,
            "manufacturer": "Acme", **extra}


ACCESSORIES = [
    accessory("a3", "Aqara Sensor", "Cellar", "sensor",
              {"battery_level": "42", "current_humidity": "82.42", "current_temperature": "16.1°C"}, bridge=AQARA_HUB),
    accessory("a1", "Aqara Sensor", "Cellar", "sensor",
              {"battery_level": "57", "current_humidity": "55.97", "current_temperature": "18.3°C"}, bridge=AQARA_HUB),
    accessory("a2", "Aqara Sensor", "Cellar", "sensor", None, bridge=AQARA_HUB, reachable=False),
    accessory("g1", "Soil Probe", "Greenhouse", "sensor", {"current_humidity": "5", "current_temperature": "14.0°C"},
              manufacturer="GARDENA"),
    accessory("m1", "Eve Motion", "Terrace", "sensor", {"battery_level": "90", "occupancy_detected": "true"}),
    accessory("c1", "Pergola Roof", "Terrace", "window_covering", {"current_position": "100", "target_position": "100"}),
    accessory("c2", "Pergola Zip", "Terrace", "window_covering", {"current_position": "40", "target_position": "0"}),
    accessory("d1", "Garage Door", "Garage", "garage_door", {"current_door_state": "closed", "target_door_state": "closed"}),
    accessory("v1", "Water Control 1", "Beds", "irrigation", {"active": "0", "battery_level": "100"}),
    accessory("v2", "Dual Water Control", "Greenhouse", "irrigation", {"active": "1", "battery_level": "100"}),
    accessory("t1", "Child Radiator", "Child Room", "thermostat",
              {"battery_level": "55", "current_heating_cooling": "heat", "current_temperature": "19.5°C",
               "target_heating_cooling": "heat", "target_temperature": "21.0°C"}),
    accessory("l1", "Nanoleaf Panels", "Playroom", "lightbulb", {"brightness": "70", "color_temperature": "50", "power": "true"}),
    accessory("l2", "Night Light", "Living Room", "lightbulb", {"brightness": "50", "power": "false"}),
    accessory("b1", "Button", "Bedroom", "programmable_switch", {"battery_level": "100", "input_event": "--"}),
    accessory("p1", "Kitchen Spots", "Kitchen", "lightbulb", {"brightness": "100", "power": "true"}, bridge=PLEJD_BRIDGE),
    accessory("h1", "Hue Go", "Office", "lightbulb", {"power": "false"}, bridge=HUE_BRIDGE, manufacturer="Signify Netherlands B.V."),
    {"id": "br1", "name": "Aqara Hub Cellar", "room": "Cellar", "category": "bridge", "is_bridge": True,
     "manufacturer": "Aqara", "state": {"power": "false"}},
]
SCENES = [
    {"id": "s1", "name": "Good Morning", "type": "wake_up", "action_count": 3},
    {"id": "s2", "name": "Office Lights", "type": "user_defined", "action_count": 2},
    {"id": "s3", "name": "A02DF36E-BFFF-5A14-827F-BE0E294B0098", "type": "user_defined", "action_count": 2},
]
ZONES = {"v2": [{"name": "Terrace", "active": "1"}, {"name": "Greenhouse", "active": "0"}],
         "v1": [{"name": "--", "active": "0"}]}


@pytest.fixture
def built():
    return build(ACCESSORIES, SCENES, ZONES)


def names(entities):
    return [e.name for e in entities]


@pytest.mark.unit
class TestExclusion:
    def test_a_bridges_maker_excludes_its_accessories(self):
        assert excluded(ACCESSORIES[-3], DEFAULT_EXCLUDE)      # Plejd gateway
        assert excluded(ACCESSORIES[-2], DEFAULT_EXCLUDE)      # Signify bridge
        assert not excluded(ACCESSORIES[0], DEFAULT_EXCLUDE)   # Aqara hub

    def test_the_match_is_a_case_insensitive_substring(self):
        assert excluded(ACCESSORIES[-3], ["plejd gate"])
        assert not excluded(ACCESSORIES[-3], ["Gateway Plejd"])

    def test_an_empty_list_keeps_everything_but_the_bridges(self):
        snapshot, _ = build(ACCESSORIES, [], {}, filters=[])
        listed = names(snapshot.lights) + names(snapshot.sensors)
        assert "Kitchen Spots" in listed and "Hue Go" in listed and "Aqara Hub Cellar" not in listed


@pytest.mark.unit
class TestBuild:
    def test_the_other_integrations_accessories_and_the_bridges_are_left_out(self, built):
        snapshot, _ = built
        listed = names(snapshot.lights) + names(snapshot.sensors)
        assert "Kitchen Spots" not in listed and "Hue Go" not in listed and "Aqara Hub Cellar" not in listed

    def test_twins_are_numbered_by_uuid_and_a_non_breaking_space_is_a_space(self, built):
        snapshot, _ = built
        cellar = [e for e in snapshot.sensors if e.area == "Cellar"]
        assert names(cellar) == ["Aqara Sensor #1", "Aqara Sensor #2", "Aqara Sensor #3"]
        assert [e.value for e in cellar] == ["18.3°C, 56% humidity, battery 57%",
                                             "unreachable",
                                             "16.1°C, 82% humidity, battery 42%"]

    def test_sensors_and_lights_are_sorted_by_room_then_name(self, built):
        snapshot, _ = built
        assert [e.area for e in snapshot.sensors] == sorted((e.area for e in snapshot.sensors), key=str.lower)
        assert names(snapshot.lights)[:2] == ["Water Control 1 Zon 1", "Garage Door"]

    def test_a_soil_sensor_reads_as_soil_moisture(self, built):
        snapshot, _ = built
        assert next(e.value for e in snapshot.sensors if e.name == "Soil Probe") == "14.0°C, soil moisture 5%"

    def test_motion_the_thermostat_and_a_button_are_sensor_lines(self, built):
        snapshot, _ = built
        by_name = {e.name: e.value for e in snapshot.sensors}
        assert by_name["Eve Motion"] == "motion, battery 90%"
        assert by_name["Child Radiator"] == "19.5°C, target 21.0°C, heating, battery 55%"
        assert by_name["Button"] == "battery 100%"

    def test_lights_covers_the_garage_door_and_the_zones_are_light_lines(self, built):
        snapshot, _ = built
        by_name = {e.name: e.value for e in snapshot.lights}
        assert by_name["Nanoleaf Panels"] == "on 70%" and by_name["Night Light"] == "off"
        assert by_name["Pergola Roof"] == "open" and by_name["Pergola Zip"] == "closing"
        assert by_name["Garage Door"] == "closed"
        assert by_name["Dual Water Control Terrace"] == "watering" and by_name["Dual Water Control Greenhouse"] == "off"
        assert "Dual Water Control" not in by_name

    def test_an_irrigation_control_without_zones_is_one_entity(self):
        snapshot, index = build(ACCESSORIES, [], {})
        assert {e.name: e.value for e in snapshot.lights}["Dual Water Control"] == "watering"
        assert index.resolve("Dual Water Control").service is None

    def test_a_zone_without_a_name_is_numbered(self, built):
        _, index = built
        target = index.resolve("Water Control 1 Zon 1 (Beds)")
        assert target.id == "v1" and target.service is None

    def test_a_named_zone_carries_its_service_name(self, built):
        _, index = built
        target = index.resolve("Dual Water Control Greenhouse")
        assert (target.id, target.service, target.kind) == ("v2", "Greenhouse", "valve")

    def test_the_uuid_named_action_sets_are_hidden_and_a_scene_type_shows(self, built):
        snapshot, _ = built
        assert [(e.name, e.value) for e in snapshot.actions] == [("Good Morning", "scene, wake up"), ("Office Lights", "scene")]

    def test_the_render_shows_rooms_as_brackets(self, built):
        snapshot, _ = built
        text = render(snapshot)
        assert '"Garage Door [Garage]": closed' in text and '"Good Morning": scene, wake up' in text


@pytest.mark.unit
class TestValues:
    @pytest.mark.parametrize("state, expected", [
        ({"power": "true", "brightness": "100"}, "on"),
        ({"power": "true"}, "on"),
        ({"power": "false", "brightness": "70"}, "off"),
        ({"power": "--"}, "unknown"),
        ({}, "unknown"),
    ])
    def test_light(self, state, expected):
        assert light_value(state) == expected

    @pytest.mark.parametrize("state, expected", [
        ({"current_position": "0", "target_position": "0"}, "closed"),
        ({"current_position": "40", "target_position": "40"}, "40% open"),
        ({"current_position": "40", "target_position": "100"}, "opening"),
        ({"current_position": "--"}, "unknown"),
    ])
    def test_cover(self, state, expected):
        assert cover_value(state) == expected

    def test_sensor_without_readings(self):
        assert sensor_value({"input_event": "--"}) == "no reading"

    def test_thermostat_without_a_mode(self):
        assert thermostat_value({"current_temperature": "19.5°C", "target_temperature": "--"}) == "19.5°C"


@pytest.mark.unit
class TestNameIndex:
    def test_resolves_the_display_form_the_bare_name_and_the_uuid(self, built):
        _, index = built
        assert index.resolve("Garage Door [Garage]").id == "d1"
        assert index.resolve("garage door").id == "d1"
        assert index.resolve("d1").id == "d1"

    def test_an_invented_room_falls_back_to_the_bare_name(self, built):
        _, index = built
        assert index.resolve("Garage Door (Carport)").id == "d1"

    def test_twins_by_their_unnumbered_name_are_offered_numbered(self, built):
        _, index = built
        with pytest.raises(Unresolved, match="Did you mean: .*#1.*#2.*#3"):
            index.resolve("Aqara Sensor")

    def test_the_same_target_under_two_keys_is_one_holder(self):
        index = NameIndex()
        target = Target("x1", "lightbulb", "Lamp [Hall]")
        index.add(target, "Lamp", "Hall")
        index.add(target, "Lamp", "Hall")
        assert index.resolve("Lamp") is target and len(index) == 1

    def test_an_unknown_name_suggests_close_ones(self, built):
        _, index = built
        with pytest.raises(Unresolved, match="Did you mean: Pergola Roof \\[Terrace\\]"):
            index.resolve("Pergola Rof")

    def test_an_empty_name(self, built):
        _, index = built
        with pytest.raises(Unresolved):
            index.resolve("  ")

    def test_targets_lists_each_once(self, built):
        _, index = built
        ids = [t.id for t in index.targets()]
        assert ids.count("v2") == 2 and ids.count("d1") == 1 and len(ids) == len(index)


def event(uid, characteristic, value, stamp="2026-10-16T17:09:58Z", service="", kind="characteristic_change"):
    return {"type": kind, "accessory": {"id": uid, "name": "x", "room": "y"}, "characteristic": characteristic,
            "value": value, "service": service, "timestamp": stamp}


@pytest.mark.unit
class TestEvents:
    @pytest.fixture
    def known(self, built):
        _, index = built
        known = {}
        for target in index.targets():
            known.setdefault(target.id, []).append(target)
        return known

    def test_time_round_trips_the_apps_zulu_stamp(self):
        assert iso_time(event_time("2026-10-16T17:09:58Z")) == "2026-10-16T17:09:58Z"
        assert event_time("yesterday") is None

    @pytest.mark.parametrize("uid, characteristic, value, text", [
        ("l1", "power", "false", "Nanoleaf Panels [Playroom] turned off"),
        ("d1", "current_door_state", "opening", "Garage Door [Garage] is opening"),
        ("c2", "target_position", "100", "Pergola Zip [Terrace] set to open"),
        ("c2", "target_position", "35", "Pergola Zip [Terrace] set to 35% open"),
        ("m1", "occupancy_detected", "true", "Eve Motion [Terrace] detected motion"),
        ("b1", "input_event", "single_press", "Button [Bedroom] was pressed (single_press)"),
        ("t1", "target_temperature", "22.0°C", "Child Radiator [Child Room] target set to 22.0°C"),
    ])
    def test_a_change_the_dialog_wants(self, known, uid, characteristic, value, text):
        entry = feed_entry(event(uid, characteristic, value), known)
        assert entry.content == text and entry.timestamp == event_time("2026-10-16T17:09:58Z")

    def test_a_zone_event_names_its_zone(self, known):
        entry = feed_entry(event("v2", "active", "1", service="Greenhouse"), known)
        assert entry.content == "Dual Water Control Greenhouse [Greenhouse] started watering"

    @pytest.mark.parametrize("ev", [
        event("l1", "brightness", "53"),
        event("a1", "current_temperature", "16.2°C"),
        event("c2", "current_position", "41"),
        event("b1", "input_event", "--"),
        event("p1", "power", "true"),                          # a Plejd light: not listed
        event("l1", "power", "true", stamp="not a time"),
        {"type": "homes_updated", "homes": 1, "timestamp": "2026-10-16T17:11:46Z"},
    ])
    def test_a_change_the_dialog_does_not_want(self, known, ev):
        assert feed_entry(ev, known) is None
