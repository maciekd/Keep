"""The MusicSystem contract: the Speakers composite merging every system's zones and resolving a
room, the owner of a list id, `active()` gathering the ready systems through the monitor and the
hint for when none is."""
from unittest.mock import Mock

import pytest

from assistant.integration.Integration import Integration
from assistant.integration.IntegrationMonitor import IntegrationMonitor, DisabledIntegration, UnavailableIntegration
from assistant.integration.music.MusicSystem import (
    STARTING_HINT, MusicError, Speakers, active, no_system_hint, speakers,
)
from tests.unit.assistant.integration.music.music_fakes import FakeSpeakers, OtherSpeakers


class Weather(Integration):
    @property
    def name(self): return "Weather"
    @property
    def config_keys(self): return []
    def check_health(self): return True


@pytest.fixture
def monitor():
    IntegrationMonitor.reset_instance()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(IntegrationMonitor, "_config_manager", Mock(), raising=False)
        yield IntegrationMonitor.get_instance()
    IntegrationMonitor.reset_instance()


def _register(monitor, integration):
    with monitor._lock:
        monitor._integrations[integration.name] = integration


@pytest.mark.unit
class TestSpeakers:
    def test_the_zones_of_every_system_in_order(self):
        sonos, heos = FakeSpeakers(["Kitchen"]), OtherSpeakers(["Porch", "Hall"])
        assert Speakers([sonos, heos]).zones() == [(sonos, "Kitchen"), (heos, "Porch"), (heos, "Hall")]

    def test_a_room_resolves_to_its_system_case_aside(self):
        sonos, heos = FakeSpeakers(["Kitchen"]), OtherSpeakers(["Porch"])
        assert Speakers([sonos, heos]).resolve(" porch ") == (heos, "Porch")

    def test_an_unknown_room_names_every_zone(self):
        with pytest.raises(MusicError) as raised:
            Speakers([FakeSpeakers(["Kitchen"]), OtherSpeakers(["Porch"])]).resolve("Garage")
        assert "Garage" in str(raised.value) and "Kitchen" in str(raised.value) and "Porch" in str(raised.value)

    def test_a_zone_name_two_systems_have_is_refused_naming_both(self):
        with pytest.raises(MusicError) as raised:
            Speakers([FakeSpeakers(["Kitchen"]), OtherSpeakers(["kitchen"])]).resolve("Kitchen")
        assert "Fake Speakers" in str(raised.value) and "Other Speakers" in str(raised.value)

    def test_no_zones_at_all_is_an_error_not_a_guess(self):
        with pytest.raises(MusicError):
            Speakers([FakeSpeakers()]).resolve("Kitchen")

    def test_a_system_that_cannot_list_its_zones_costs_only_itself(self):
        broken, heos = FakeSpeakers(), OtherSpeakers(["Porch"])
        broken.zones = Mock(side_effect=ConnectionError("unreachable"))
        assert Speakers([broken, heos]).resolve("Porch") == (heos, "Porch")

    def test_the_zones_are_listed_once_per_house(self):
        sonos = FakeSpeakers(["Kitchen"])
        sonos.zones = Mock(return_value=["Kitchen"])
        house = Speakers([sonos])
        house.resolve("Kitchen")
        house.resolve("kitchen")
        assert sonos.zones.call_count == 1

    def test_a_list_id_belongs_to_the_system_with_its_prefix_and_a_spotify_id_to_none(self):
        sonos, heos = FakeSpeakers(), OtherSpeakers()
        house = Speakers([sonos, heos])
        assert house.owner_of_id("F-2") is sonos
        assert house.owner_of_id("H-0") is heos
        assert house.owner_of_id("S-0") is None and house.owner_of_id("AA") is None

    def test_the_status_joins_every_systems_lines_and_skips_a_broken_one(self):
        broken = FakeSpeakers()
        broken.zone_status = Mock(side_effect=ConnectionError("unreachable"))
        house = Speakers([FakeSpeakers(["Kitchen"], status=["Kitchen (Playing)"]), broken,
                          OtherSpeakers(["Porch"], status=["Porch (Paused)"])])
        assert house.zone_status().splitlines() == ["Kitchen (Playing)", "Porch (Paused)"]

    def test_a_system_has_a_library_when_it_lists_one(self):
        assert FakeSpeakers(library="Radio (ID: F-0)").has_library()
        assert not FakeSpeakers().has_library()


@pytest.mark.unit
class TestActive:
    def test_the_configured_systems_in_registration_order(self, monitor):
        first, second = FakeSpeakers(), OtherSpeakers()
        _register(monitor, first)
        _register(monitor, Weather())
        _register(monitor, second)
        assert active() == [first, second]
        assert speakers().systems == [first, second]

    def test_an_unconfigured_system_is_left_out(self, monitor):
        ready = FakeSpeakers()
        _register(monitor, OtherSpeakers(missing=["fake.speakers"]))
        _register(monitor, ready)
        assert active() == [ready]

    def test_nothing_registered_is_an_empty_house(self, monitor):
        assert active() == [] and speakers().zones() == []

    def test_the_readiness_is_logged_when_it_changes_not_at_every_look(self, monitor, monkeypatch):
        """The zones section resolves the systems every few seconds: the same answer is one line."""
        from assistant.integration.music import MusicSystem as module
        lines = []
        monkeypatch.setattr(module, "_last_active_report", [])
        monkeypatch.setattr(module, "trace", lines.append)
        _register(monitor, FakeSpeakers())
        for _ in range(3):
            active()
        first_look = list(lines)
        assert len(first_look) == 1 and "Fake Speakers" in first_look[0]

        _register(monitor, OtherSpeakers(missing=["fake.speakers"]))
        active()
        active()
        changed = lines[len(first_look):]
        assert len(changed) == 2, "the skipped system and the ready count, once each"
        assert any("Other Speakers" in line and "fake.speakers" in line for line in changed)

        with monitor._lock:
            del monitor._integrations["Other Speakers"]
        active()
        assert len(lines) == len(first_look) + 3, "back to the first state is a change again"


@pytest.mark.unit
class TestNoSystemHint:
    def test_nothing_registered_says_still_starting(self, monitor):
        _register(monitor, DisabledIntegration(Weather))
        assert no_system_hint() == STARTING_HINT

    def test_an_unconfigured_system_names_its_missing_keys(self, monitor):
        _register(monitor, FakeSpeakers(missing=["fake.speakers"]))
        hint = no_system_hint()
        assert "Fake Speakers" in hint and "fake.speakers" in hint and "config_help" in hint

    def test_a_disabled_system_gives_the_enable_line(self, monitor):
        _register(monitor, DisabledIntegration(FakeSpeakers))
        assert 'enable_integration("Fake Speakers")' in no_system_hint()

    def test_a_broken_system_carries_its_error(self, monitor):
        _register(monitor, UnavailableIntegration(OtherSpeakers, KeyError("heos")))
        hint = no_system_hint()
        assert "OtherSpeakers could not be started" in hint and "KeyError" in hint and "restart_integration" in hint

    def test_a_broken_non_music_integration_is_not_a_music_problem(self, monitor):
        _register(monitor, UnavailableIntegration(Weather, KeyError("weather")))
        assert no_system_hint() == STARTING_HINT


# --- the zone-change listeners ----------------------------------------------------------------

from assistant.integration.music import MusicSystem as music_module   # noqa: E402


@pytest.fixture
def listeners():
    saved = list(music_module._zone_listeners)
    music_module._zone_listeners.clear()
    yield music_module._zone_listeners
    music_module._zone_listeners[:] = saved


def test_a_reported_zone_change_reaches_every_listener(listeners):
    first, second = Mock(), Mock()
    music_module.add_zone_listener(first)
    music_module.add_zone_listener(second)
    music_module.zones_changed("Fake Speakers")
    first.assert_called_once_with()
    second.assert_called_once_with()


def test_a_raising_listener_costs_only_itself(listeners):
    broken, fine = Mock(side_effect=RuntimeError("bad listener")), Mock()
    music_module.add_zone_listener(broken)
    music_module.add_zone_listener(fine)
    music_module.zones_changed("Fake Speakers")
    fine.assert_called_once_with()


def test_a_removed_listener_is_not_called_again(listeners):
    callback = Mock()
    music_module.add_zone_listener(callback)
    music_module.remove_zone_listener(callback)
    music_module.zones_changed("Fake Speakers")
    callback.assert_not_called()


def test_a_listener_added_twice_is_called_once(listeners):
    callback = Mock()
    music_module.add_zone_listener(callback)
    music_module.add_zone_listener(callback)
    music_module.zones_changed("Fake Speakers")
    callback.assert_called_once_with()


def test_a_system_that_says_nothing_about_it_does_not_push_its_changes():
    assert FakeSpeakers(["Kitchen"]).pushes_changes() is False
