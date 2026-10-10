"""PlejdModel: the MAC from an advertisement, the dim scales, the snapshot built from fake
devices and the name index — all on a fictional site."""
import pytest

from assistant.integration.HomeAutomation import render
from integrations.plejd.PlejdModel import (
    MANUFACTURER_ID, NameIndex, Unresolved, build, cover_value_of, mac_from_advertisement, state_text, to_plejd_level,
    value_of)


class Room:
    def __init__(self, title):
        self.title = title


class Device:
    def __init__(self, name, kind="LIGHT", room=None, hidden=False, dimmable=True):
        self.name, self.outputType, self.hidden, self.dimmable = name, kind, hidden, dimmable
        self.roomData = Room(room) if room is not None else None
        self.identifier = ("dev", name, room)


class Scene:
    def __init__(self, name, hidden=False):
        self.name, self.outputType, self.hidden = name, "SCENE", hidden
        self.identifier = (name,)


def advertisement(mac="AA:BB:CC:DD:EE:FF", company=MANUFACTURER_ID, prefix=b"\x01\x02\x03\x04", tail=b"\x00\x00"):
    return {company: prefix + bytes.fromhex(mac.replace(":", ""))[::-1] + tail}


@pytest.mark.unit
class TestAdvertisement:
    def test_the_mac_is_bytes_4_to_10_reversed(self):
        assert mac_from_advertisement(advertisement()) == "AABBCCDDEEFF"
        assert mac_from_advertisement(advertisement("00:11:22:33:44:55", tail=b"")) == "001122334455"

    def test_other_manufacturers_and_short_data_are_not_plejd(self):
        assert mac_from_advertisement(advertisement(company=76)) is None
        assert mac_from_advertisement({MANUFACTURER_ID: b"\x01" * 9}) is None
        assert mac_from_advertisement({}) is None


@pytest.mark.unit
class TestScales:
    def test_a_percentage_lands_on_1_to_255(self):
        assert to_plejd_level(100) == 255
        assert to_plejd_level(50) == 128
        assert to_plejd_level(1) == 3
        assert to_plejd_level(0) == 1
        assert to_plejd_level(250) == 255

    def test_a_state_reads_the_same_whichever_way_the_dim_arrived(self):
        assert value_of({"state": True, "dim": 204}) == "on 80%"          # an event: an int
        assert value_of({"state": True, "dim": 204.0}) == "on 80%"        # a light level poll: a float
        assert value_of({"state": True, "dim": 255.99}) == "on 100%"
        assert value_of({"state": True}) == "on"
        assert value_of({"state": False, "dim": 204}) == "off"
        assert value_of({"available": True}) == "unknown"

    def test_a_cover_reads_its_position_or_its_motion(self):
        assert cover_value_of({"position": 100.0, "moving": False, "opening": True}) == "open"
        assert cover_value_of({"position": 0.0, "moving": False, "opening": False}) == "closed"
        assert cover_value_of({"position": 40.157, "moving": False, "opening": False}) == "40% open"
        assert cover_value_of({"position": 99.6, "moving": False, "opening": True}) == "open"
        assert cover_value_of({"position": 20.0, "moving": True, "opening": True}) == "opening"
        assert cover_value_of({"position": 20.0, "moving": True, "opening": False}) == "closing"
        assert cover_value_of({"position": None, "moving": False, "opening": False}) == "unknown"   # pyplejd lost track
        assert cover_value_of({"available": True}) == "unknown"

    def test_a_state_is_read_by_what_the_device_is(self):
        state = {"state": True, "dim": 255, "position": 0.0, "moving": False}
        assert state_text(Device("Lampa"), state) == "on 100%"
        assert state_text(Device("Screen", kind="COVERABLE"), state) == "closed"


@pytest.mark.unit
class TestBuild:
    def test_visible_lights_relays_and_scenes_land_in_their_lists_sorted_with_rooms(self):
        devices = [Device("Taklampa", room="Kök"), Device("Fläkt", kind="RELAY", room="Badrum"), Scene("Kväll"),
                   Device("Spots", room="Hall", hidden=True), Device("Rörelse", kind="MOTION", room="Hall"),
                   Scene("Gömd", hidden=True), Device("  ", room="Hall"),
                   Device("Rullgardin väst", kind="COVERABLE", room="Uterum", dimmable=False),
                   Device("Persienn", kind="COVERABLE", room="Kontor", hidden=True)]
        states = {("dev", "Taklampa", "Kök"): {"state": True, "dim": 128}, ("dev", "Fläkt", "Badrum"): {"state": False},
                  ("dev", "Rullgardin väst", "Uterum"): {"position": 35.0, "moving": False, "opening": False}}
        snapshot, index = build(devices, states)
        assert render(snapshot) == ('## LIGHTS\n"Fläkt [Badrum]": off\n"Rullgardin väst [Uterum]": 35% open\n'
                                    '"Taklampa [Kök]": on 50%\n\n'
                                    '## ACTIONS\n"Kväll": scene\n\n')
        assert snapshot.sensors == [] and snapshot.variables == []
        assert len(index) == 4

    def test_a_light_without_a_room_or_a_report_yet(self):
        snapshot, _ = build([Device("Lykta"), Device("Fönster", room="  ")], {})
        assert [(e.name, e.area, e.value) for e in snapshot.lights] == [("Fönster", None, "unknown"), ("Lykta", None, "unknown")]


@pytest.mark.unit
class TestNameIndex:
    @pytest.fixture
    def index(self):
        _, index = build([Device("Taklampa", room="Kök"), Device("Taklampa", room="Hall"), Device("Bordslampa", room="Vardagsrum"),
                          Scene("Kväll"), Device("Markis", kind="COVERABLE", room="Uterum")], {})
        return index

    def test_bare_name_and_display_form_both_resolve(self, index):
        device, display = index.resolve("Bordslampa")
        assert device.identifier == ("dev", "Bordslampa", "Vardagsrum") and display == "Bordslampa [Vardagsrum]"
        assert index.resolve("Bordslampa (Vardagsrum)")[0] is device       # what from_display hands the backend
        assert index.resolve("  bordslampa   (VARDAGSRUM) ")[0] is device

    def test_an_invented_room_falls_back_to_the_bare_name(self, index):
        assert index.resolve("Bordslampa (Köket)")[1] == "Bordslampa [Vardagsrum]"

    def test_a_name_in_two_rooms_needs_the_room(self, index):
        assert index.resolve("Taklampa (Hall)")[0].identifier == ("dev", "Taklampa", "Hall")
        with pytest.raises(Unresolved) as e:
            index.resolve("Taklampa")
        assert "Taklampa [Kök]" in str(e.value) and "Taklampa [Hall]" in str(e.value)

    def test_an_unknown_name_is_refused_with_the_closest_ones(self, index):
        with pytest.raises(Unresolved) as e:
            index.resolve("Bordlampa")
        assert "Bordslampa [Vardagsrum]" in str(e.value)
        with pytest.raises(Unresolved):
            index.resolve("")
        with pytest.raises(Unresolved) as e:
            index.resolve("Poolpump")
        assert "Did you mean" not in str(e.value)

    def test_a_scene_resolves_by_name(self, index):
        device, display = index.resolve("kväll")
        assert device.outputType == "SCENE" and display == "Kväll"

    def test_a_cover_resolves_like_a_light_and_is_offered_for_a_near_miss(self, index):
        device, display = index.resolve("Markis (Uterum)")
        assert device.outputType == "COVERABLE" and display == "Markis [Uterum]"
        with pytest.raises(Unresolved) as e:
            index.resolve("Markiss")
        assert "Markis [Uterum]" in str(e.value)
