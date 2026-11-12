"""The music sections the music toolkit carries: the zones of every speaker system with what each
plays, and the library — every system's own lists, then the user's Spotify playlists with S- ids —
waiting for the first data, following the caches, woken by a Spotify library reload."""
import threading
import time as real_time
from unittest.mock import Mock, patch

import pytest

from assistant.integration.music import MusicSystem as music
from assistant.integration.music.MusicSystem import Speakers
from assistant.integration.music.SpotifyAccount import SpotifyAccount, SpotifyPlaylist
from assistant.prompt.sections.MusicLibrarySectionGenerator import MusicLibrarySectionGenerator
from assistant.prompt.sections.MusicZonesSectionGenerator import MusicZonesSectionGenerator
from tests.unit.assistant.integration.music.music_fakes import FakeSpeakers, OtherSpeakers
from tests.unit.assistant.integration.music.spotify_fakes import install, make_account


@pytest.fixture
def spotify():
    """The shared account with no playlists; a test gives it some."""
    account = install(make_account())
    account._playlists = []
    account._playlists_total = 0
    yield account
    SpotifyAccount.reset_instance()


def _systems(*systems):
    return patch.object(music, "active", return_value=list(systems))


@pytest.mark.unit
class TestZones:
    def test_without_a_system_the_section_is_empty(self):
        with patch.object(music, "speakers", return_value=Speakers([])):
            assert MusicZonesSectionGenerator(None)._fetch_content() == ""

    def test_every_systems_zones_are_listed_with_what_they_play(self):
        house = Speakers([FakeSpeakers(["Kitchen"], status=["Kitchen (Song by Alice - Playing)"]),
                          OtherSpeakers(["Porch"], status=["Porch (Paused)"])])
        with patch.object(music, "speakers", return_value=house):
            content = MusicZonesSectionGenerator(None)._fetch_content()
        assert content.splitlines() == ["Kitchen (Song by Alice - Playing)", "Porch (Paused)"]

    def test_a_system_enabled_later_is_listed_from_the_next_poll(self):
        section = MusicZonesSectionGenerator(None)
        with patch.object(music, "speakers", return_value=Speakers([])):
            assert section._fetch_content() == ""
        with patch.object(music, "speakers", return_value=Speakers([FakeSpeakers(["Kitchen"])])):
            assert "Kitchen" in section._fetch_content()


@pytest.mark.unit
class TestLibraryContent:
    def test_a_systems_lists_then_the_spotify_playlists_with_one_blank_line_between(self, spotify):
        spotify._playlists = [SpotifyPlaylist("Road Trip", "spotify:playlist:rt2", "Alice")]
        spotify._playlists_total = 1
        with _systems(FakeSpeakers(library="Morning Mix (ID: F-0)\nDinner (ID: P-0)")):
            lines = MusicLibrarySectionGenerator(None)._fetch_content().splitlines()
        assert lines[:2] == ["Morning Mix (ID: F-0)", "Dinner (ID: P-0)"]
        assert lines[2] == "", "one blank line between a system's lists and Spotify's"
        assert lines[3] != "" and "Road Trip by Alice (ID: S-0)" in lines[4:]

    def test_every_system_contributes_its_own_lists(self, spotify):
        with _systems(FakeSpeakers(library="Morning Mix (ID: F-0)"), OtherSpeakers(library="Jazz Radio (ID: H-0)")):
            content = MusicLibrarySectionGenerator(None)._fetch_content()
        assert content.index("(ID: F-0)") < content.index("(ID: H-0)")

    def test_a_system_that_cannot_list_its_lists_costs_only_itself(self, spotify):
        broken = FakeSpeakers()
        broken.library_text = Mock(side_effect=ConnectionError("unreachable"))
        with _systems(broken, OtherSpeakers(library="Jazz Radio (ID: H-0)")):
            assert "(ID: H-0)" in MusicLibrarySectionGenerator(None)._fetch_content()

    def test_the_spotify_playlists_carry_s_ids_and_their_owner_never_their_uri(self, spotify):
        spotify._playlists = [SpotifyPlaylist("Discover Weekly", "spotify:playlist:dw1", "Spotify"),
                              SpotifyPlaylist("Road Trip", "spotify:playlist:rt2", "")]
        spotify._playlists_total = 2
        with _systems():
            content = MusicLibrarySectionGenerator(None)._fetch_content()
        assert "Discover Weekly by Spotify (ID: S-0)" in content and "Road Trip (ID: S-1)" in content
        assert "spotify:playlist" not in content and "not listed" not in content

    def test_a_library_larger_than_the_listed_part_says_how_many_are_left_out(self, spotify):
        spotify._playlists = [SpotifyPlaylist(f"List {i}", f"spotify:playlist:l{i}") for i in range(10)]
        spotify._playlists_total = 57
        with _systems():
            content = MusicLibrarySectionGenerator(None)._fetch_content()
        assert "(ID: S-9)" in content and "S-10" not in content
        assert "47" in content and "search_music" in content

    def test_no_spotify_library_lists_nothing_under_spotify(self, spotify):
        with _systems(FakeSpeakers(library="Morning Mix (ID: F-0)")):
            content = MusicLibrarySectionGenerator(None)._fetch_content()
        assert "S-0" not in content and "Spotify playlists" not in content

    def test_spotify_playlists_alone_count_as_data(self, spotify):
        spotify._playlists = [SpotifyPlaylist("Road Trip", "spotify:playlist:rt2")]
        spotify._playlists_total = 1
        with _systems(FakeSpeakers()):
            section = MusicLibrarySectionGenerator(None)
            assert section._has_data() and "S-0" in section._fetch_content()

    def test_nothing_anywhere_is_no_data(self, spotify):
        with _systems(FakeSpeakers()):
            assert not MusicLibrarySectionGenerator(None)._has_data()


@pytest.mark.unit
class TestLibraryDataWithABrokenSystem:
    @pytest.mark.xfail(strict=True, reason="MusicLibrarySectionGenerator._has_data asks every system's has_library() "
                                           "unguarded: a system whose library_text raises (HEOS with no player "
                                           "answering) raises out of the monitoring loop, which then restarts every "
                                           "5 s and never loads the lists of the systems after it")
    def test_a_system_that_cannot_list_its_lists_does_not_hide_the_data_of_the_next(self, spotify):
        broken = FakeSpeakers()
        broken.library_text = Mock(side_effect=ConnectionError("unreachable"))
        with _systems(broken, OtherSpeakers(library="Jazz Radio (ID: H-0)")):
            assert MusicLibrarySectionGenerator(None)._has_data()

    def test_the_lists_of_a_system_before_the_broken_one_count_as_data(self, spotify):
        broken = OtherSpeakers()
        broken.library_text = Mock(side_effect=ConnectionError("unreachable"))
        with _systems(FakeSpeakers(library="Morning Mix (ID: F-0)"), broken):
            assert MusicLibrarySectionGenerator(None)._has_data()


def _drive(section, on_wait):
    """Runs the monitoring loop with `on_wait(duration, count)` as its wait; the loop's own
    clock is frozen, so only the waits move it."""
    waits = []

    def wait(duration):
        waits.append(duration)
        on_wait(duration, len(waits))
        return section._stop_event.is_set()

    section._wait = wait
    section._update_content = Mock()
    with patch("assistant.prompt.sections.MusicLibrarySectionGenerator.time") as clock:
        clock.time.return_value = 100.0
        section._start_monitoring()
    return waits


@pytest.mark.unit
class TestLibraryMonitoring:
    def test_data_there_at_once_is_loaded_once_and_then_followed(self, spotify):
        section = MusicLibrarySectionGenerator(None)
        with _systems(FakeSpeakers(library="Morning Mix (ID: F-0)")):
            _drive(section, lambda d, n: section._stop_event.set() if d == section.CHECK_INTERVAL else None)
        section._update_content.assert_called_once()

    def test_it_polls_until_the_first_list_is_in(self, spotify):
        sonos = FakeSpeakers()
        section = MusicLibrarySectionGenerator(None)

        def on_wait(duration, count):
            if count == 3:
                sonos._library = "Morning Mix (ID: F-0)"
            if duration == section.CHECK_INTERVAL:
                section._stop_event.set()
        with _systems(sonos):
            waits = _drive(section, on_wait)
        assert waits[:3] == [0.1, 0.1, 0.1]
        section._update_content.assert_called_once()

    def test_a_stop_before_any_data_loads_nothing(self, spotify):
        section = MusicLibrarySectionGenerator(None)
        section._stop_event.set()
        with _systems(FakeSpeakers(library="Morning Mix (ID: F-0)")):
            _drive(section, lambda d, n: None)
        section._update_content.assert_not_called()

    def test_a_system_enabled_while_it_waits_is_loaded_then(self, spotify):
        systems = []
        section = MusicLibrarySectionGenerator(None)

        def on_wait(duration, count):
            if count == 2:
                systems.append(FakeSpeakers(library="Morning Mix (ID: F-0)"))   # enable_integration built it
            if duration == section.CHECK_INTERVAL:
                section._stop_event.set()
        with patch.object(music, "active", side_effect=lambda: list(systems)):
            _drive(section, on_wait)
        section._update_content.assert_called_once()

    def test_the_section_listens_for_library_reloads_only_while_it_runs(self, spotify):
        section = MusicLibrarySectionGenerator(None)
        section._stop_event.set()
        with patch.object(SpotifyAccount, "add_library_listener") as add, \
                patch.object(SpotifyAccount, "remove_library_listener") as remove, _systems():
            _drive(section, lambda d, n: None)
        add.assert_called_once_with(section._on_library_reload)
        remove.assert_called_once_with(section._on_library_reload)


@pytest.mark.unit
class TestLibraryWake:
    def test_a_reloaded_library_ends_the_wait_early_and_only_once(self):
        section = MusicLibrarySectionGenerator(None)
        SpotifyAccount.add_library_listener(section._on_library_reload)
        try:
            threading.Timer(0.02, make_account()._notify_library_listeners).start()
            started = real_time.monotonic()
            assert section._wait(section.CHECK_INTERVAL) is False, "woken, not stopped"
            assert real_time.monotonic() - started < 2
        finally:
            SpotifyAccount.remove_library_listener(section._on_library_reload)

        started = real_time.monotonic()
        section._wait(0.02)
        assert real_time.monotonic() - started >= 0.02, "the wake is consumed by the wait it ended"

    def test_stop_ends_the_wait_as_well(self):
        section = MusicLibrarySectionGenerator(None)
        threading.Timer(0.02, section.stop).start()
        started = real_time.monotonic()
        assert section._wait(section.CHECK_INTERVAL) is True
        assert real_time.monotonic() - started < 2


class PushingSpeakers(FakeSpeakers):
    """A system that reports its own zone changes (HEOS while its events flow)."""

    def pushes_changes(self):
        return True


@pytest.mark.unit
class TestZonesFollowTheChanges:
    def test_it_polls_while_any_system_cannot_report_its_changes(self):
        with _systems(PushingSpeakers(["Kitchen"]), FakeSpeakers(["Porch"])):
            assert MusicZonesSectionGenerator(None)._interval() == MusicZonesSectionGenerator.UPDATE_INTERVAL

    def test_it_only_checks_now_and_then_while_every_system_reports_its_changes(self):
        with _systems(PushingSpeakers(["Kitchen"])):
            interval = MusicZonesSectionGenerator(None)._interval()
        assert interval == MusicZonesSectionGenerator.PUSHED_INTERVAL
        assert interval > MusicZonesSectionGenerator.UPDATE_INTERVAL

    def test_without_a_system_it_polls(self):
        with _systems():
            assert MusicZonesSectionGenerator(None)._interval() == MusicZonesSectionGenerator.UPDATE_INTERVAL

    def test_a_reported_change_ends_the_wait_at_once(self):
        section = MusicZonesSectionGenerator(None)
        music.add_zone_listener(section._on_zones_changed)
        try:
            done = []
            waiter = threading.Thread(target=lambda: done.append(section._wait(30)))
            waiter.start()
            music.zones_changed("Fake Speakers")
            waiter.join(timeout=2)
            assert not waiter.is_alive() and done == [False]
        finally:
            music.remove_zone_listener(section._on_zones_changed)

    def test_the_monitoring_listens_while_it_runs_and_not_after(self):
        section = MusicZonesSectionGenerator(None)
        listening = []

        def wait(seconds):
            listening.append(section._on_zones_changed in music._zone_listeners)
            return True   # stopped
        section._wait = wait
        with _systems(FakeSpeakers(["Kitchen"])):
            section._start_monitoring()
        assert listening == [True]
        assert section._on_zones_changed not in music._zone_listeners

    def test_each_wake_brings_a_fresh_look(self):
        section = MusicZonesSectionGenerator(None)
        waits = iter([False, False, True])
        section._wait = lambda seconds: next(waits)
        section._update_content = Mock()
        with _systems(FakeSpeakers(["Kitchen"])):
            section._start_monitoring()
        assert section._update_content.call_count == 2
