"""HueModel: CLIP v2 resources → the snapshot lists, the values, rooms through devices, the
sensors' readings and the name index."""
import json
from pathlib import Path

import pytest

from assistant.integration.HomeAutomation import render
from integrations.hue.HueModel import GROUP_NAME, WHOLE_HOUSE, NameIndex, Target, Unresolved, build, light_value, lux

DATA = Path(__file__).parents[2] / "data" / "home" / "hue"


def load_house() -> dict:
    return json.loads((DATA / "resources.json").read_text())


@pytest.fixture
def house():
    return build(load_house())


def _entity(snapshot, section, name, area=None):
    for entity in getattr(snapshot, section):
        if entity.name == name and entity.area == area:
            return entity
    raise AssertionError(f"{name} [{area}] not in {section}: {[(e.name, e.area) for e in getattr(snapshot, section)]}")


@pytest.mark.unit
class TestBuild:
    def test_renders_the_golden_text(self, house):
        snapshot, _ = house
        assert render(snapshot) == (DATA / "rendered.md").read_text()

    def test_a_light_gets_its_room_through_its_device(self, house):
        snapshot, _ = house
        assert _entity(snapshot, "lights", "Taklampa", "Kök").value == "40%"
        assert _entity(snapshot, "lights", "Golvlampa", "Hall").value == "on"      # a plug: on, no dimming

    def test_a_light_in_no_room_has_no_area(self, house):
        snapshot, _ = house
        assert _entity(snapshot, "lights", "Trädgårdslampa (vänster)").area is None

    def test_two_lights_of_one_name_in_one_room_are_numbered(self, house):
        snapshot, index = house
        assert _entity(snapshot, "lights", "Sänglampa", "Sovrum").value == "100%"
        assert _entity(snapshot, "lights", "Sänglampa 2", "Sovrum").value == "off"
        assert index.resolve("Sänglampa 2 (Sovrum)").rid == "l-sang2"

    def test_rooms_zones_and_the_whole_house_are_grouped_lights(self, house):
        snapshot, index = house
        assert _entity(snapshot, "lights", GROUP_NAME, "Kök").value == "70%"
        assert _entity(snapshot, "lights", GROUP_NAME, "Nedervåning").value == "55%"
        assert _entity(snapshot, "lights", WHOLE_HOUSE).value == "61%"
        assert index.resolve("All lights (Nedervåning)") == Target("grouped_light", "g-ned", "All lights [Nedervåning]")

    def test_scenes_are_actions_with_their_group_as_area(self, house):
        snapshot, index = house
        assert _entity(snapshot, "actions", "Relax", "Sovrum").value == "scene"
        assert _entity(snapshot, "actions", "Mysigt", "Nedervåning").value == "active scene"
        assert index.resolve("Relax (Sovrum)").rtype == "scene"

    def test_sensor_readings_carry_the_device_name_and_room(self, house):
        snapshot, _ = house
        assert _entity(snapshot, "sensors", "Hallsensor temperature", "Hall").value == "21.5 °C"
        assert _entity(snapshot, "sensors", "Hallsensor motion", "Hall").value == "clear"
        assert _entity(snapshot, "sensors", "Hallsensor light level", "Hall").value == "100 lx"

    def test_disabled_and_invalid_readings_are_left_out(self, house):
        snapshot, _ = house
        names = {e.name for e in snapshot.sensors}
        assert "Garagesensor motion" not in names           # enabled: false
        assert "Garagesensor temperature" not in names      # temperature_valid: false

    def test_a_report_block_is_read_when_the_plain_field_is_missing(self, house):
        snapshot, _ = house
        assert _entity(snapshot, "sensors", "Garagesensor light level").value == "1000 lx"

    def test_no_variables(self, house):
        snapshot, _ = house
        assert snapshot.variables == []

    def test_lists_are_sorted_by_name(self, house):
        snapshot, _ = house
        names = [e.name.lower() for e in snapshot.lights]
        assert names == sorted(names)

    def test_missing_resource_types_build_an_empty_house(self):
        snapshot, _ = build({})
        assert snapshot.is_empty()

    def test_a_grouped_light_of_an_unknown_owner_is_left_out(self):
        orphan = {"id": "g-x", "owner": {"rid": "r-gone", "rtype": "room"}, "on": {"on": True}}
        snapshot, _ = build({"grouped_light": [orphan]})
        assert snapshot.lights == []

    def test_a_light_without_metadata_goes_by_its_v1_id_then_its_id(self):
        lights = [{"id": "l-9", "id_v1": "/lights/9", "on": {"on": True}}, {"id": "l-10", "on": {"on": False}}]
        snapshot, index = build({"light": lights})
        assert [e.name for e in snapshot.lights] == ["/lights/9", "l-10"]
        assert index.resolve("/lights/9").rid == "l-9"


@pytest.mark.unit
class TestValues:
    @pytest.mark.parametrize("resource, expected", [
        ({"on": {"on": False}, "dimming": {"brightness": 80}}, "off"),
        ({"on": {"on": True}, "dimming": {"brightness": 33.4}}, "33%"),
        ({"on": {"on": True}}, "on"),
        ({}, "unknown"),
    ])
    def test_light_value(self, resource, expected):
        assert light_value(resource) == expected

    def test_lux_inverts_the_log_scale(self):
        assert lux(1) == 1
        assert lux(40001) == 10000


@pytest.mark.unit
class TestNameIndex:
    def test_the_display_form_and_the_bare_name_both_resolve(self, house):
        _, index = house
        assert index.resolve("Taklampa [Kök]").rid == "l-tak"
        assert index.resolve("  taklampa ").rid == "l-tak"

    def test_a_name_held_by_several_rooms_asks_which(self, house):
        _, index = house
        with pytest.raises(Unresolved, match=r"could be .*All lights \[Kök\]"):
            index.resolve("All lights")

    def test_an_invented_room_falls_back_to_the_bare_name(self, house):
        _, index = house
        assert index.resolve("Taklampa (Garage)").rid == "l-tak"
        with pytest.raises(Unresolved, match="could be"):
            index.resolve("All lights (Garage)")

    def test_a_name_with_its_own_parentheses_round_trips_through_the_display_form(self, house):
        _, index = house
        assert index.resolve("Trädgårdslampa [vänster]").rid == "l-garden"

    def test_an_unknown_name_suggests_close_ones(self, house):
        _, index = house
        with pytest.raises(Unresolved, match=r"Did you mean: Taklampa \[Kök\]"):
            index.resolve("Taklamp")

    def test_an_empty_name(self):
        with pytest.raises(Unresolved):
            NameIndex().resolve("  ")
