from unittest.mock import Mock, patch

import pytest

from assistant.integration.music.MusicSystem import Speakers
from assistant.integration.music.SpotifyAccount import SpotifyAccount
from integrations.heos.HeosClient import HeosError
from integrations.heos.HeosIntegration import HeosIntegration
from integrations.heos.SpotifyConnect import NotListed

# Den leads a group with Porch (the group's gid is the leader's pid); Kitchen plays alone.
PLAYERS = [{"name": "Kitchen", "pid": 1}, {"name": "Den", "pid": 2, "gid": 2}, {"name": "Porch", "pid": 3, "gid": 2}]


class FakeClient:
    def __init__(self, players=PLAYERS, favorites=None, unreachable=(), quickselects=None):
        self.players = players
        self.quickselects = quickselects or {}   # pid -> the slots, or an exception the player answers with
        self.paused = set()                      # pids paused since: Den (2) plays otherwise
        self.favorites = favorites if favorites is not None else [{"name": "Radio One"}, {"name": "Jazz FM"}]
        self.unreachable = set(unreachable)
        self.sent = []

    def command(self, cmd, **params):
        self.sent.append((cmd, params))
        if params.get("pid") in self.unreachable:
            raise HeosError("no answer")
        if cmd == "player/get_players":
            return {"payload": self.players}
        if cmd == "player/get_play_state":
            playing = params["pid"] == 2 and params["pid"] not in self.paused
            return {"heos": {"message": f"pid={params['pid']}&state={'play' if playing else 'stop'}"}}
        if cmd == "player/get_now_playing_media":
            return {"payload": {"song": "Blue Sky", "artist": "Alice"} if params["pid"] == 2 else {}}
        if cmd == "player/get_quickselects":
            slots = self.quickselects.get(params["pid"], [])
            if isinstance(slots, Exception):
                raise slots
            return {"payload": slots}
        if cmd == "browse/browse":
            if isinstance(self.favorites, Exception):
                raise self.favorites
            return {"payload": self.favorites}
        return {"heos": {"result": "success"}}

    def close(self):
        pass

    def sent_commands(self, cmd):
        return [params for c, params in self.sent if c == cmd]


@pytest.fixture(autouse=True)
def scopes():
    saved = set(SpotifyAccount._extra_scopes)
    yield
    SpotifyAccount._extra_scopes.clear()
    SpotifyAccount._extra_scopes.update(saved)


def make(client=None, room_mapping=None, spotify=None):
    with patch.object(HeosIntegration, "_schedule_refresh"):
        heos = HeosIntegration()
    heos._client = client or FakeClient()
    heos.config = Mock()
    heos.config.get.side_effect = lambda key, default=None: (room_mapping if key == "heos.room_mapping" else default)
    if spotify is not None:
        patch.object(SpotifyAccount, "get_instance", return_value=spotify).start()
    return heos


@pytest.fixture(autouse=True)
def stop_patches():
    yield
    patch.stopall()


def test_construction_asks_for_the_playback_scopes():
    SpotifyAccount._extra_scopes.clear()
    make()
    assert {"user-read-playback-state", "user-modify-playback-state"} <= set(SpotifyAccount.scopes())


def test_zones_follow_the_room_mapping_and_a_blank_name_hides_a_player():
    heos = make(room_mapping={"Den": "Living Room", "Porch": ""})
    assert heos.zones() == ["Kitchen", "Living Room"]


def test_a_mapped_room_acts_on_its_player():
    client = FakeClient()
    heos = make(client, room_mapping={"Kitchen": "Galley"})
    assert heos.pause("galley") == ""
    assert client.sent_commands("player/set_play_state") == [{"pid": 1, "state": "pause"}]


def test_a_group_member_acts_on_the_group_leader():
    client = FakeClient()
    heos = make(client)
    assert heos.play("Porch") == ""
    assert heos.next_track("Porch") == ""
    assert client.sent_commands("player/set_play_state") == [{"pid": 2, "state": "play"}]
    assert client.sent_commands("player/play_next") == [{"pid": 2}]


def test_a_grouped_room_changes_the_group_volume():
    client = FakeClient()
    make(client).adjust_volume("Porch", 4)
    assert client.sent_commands("group/volume_up") == [{"gid": 2, "step": 4}]


def test_an_ungrouped_room_changes_its_own_volume_clamped_to_the_cli_range():
    client = FakeClient()
    heos = make(client)
    heos.adjust_volume("Kitchen", -25)
    assert client.sent_commands("player/volume_down") == [{"pid": 1, "step": 10}]


def test_a_refused_command_comes_back_as_a_line_for_the_llm():
    client = FakeClient(unreachable={1})
    result = make(client).pause("Kitchen")
    assert "Kitchen" in result and "no answer" in result


def test_an_unknown_room_is_refused_without_a_command():
    client = FakeClient()
    heos = make(client)
    assert "Attic" in heos.play("Attic")
    assert not client.sent_commands("player/set_play_state")


def test_zone_status_asks_each_leader_once():
    client = FakeClient()
    lines = make(client).zone_status()
    assert len(lines) == 3
    assert sorted(p["pid"] for p in client.sent_commands("player/get_now_playing_media")) == [1, 2]
    porch = next(l for l in lines if l.startswith("Porch"))
    assert "Blue Sky" in porch and "Alice" in porch and "Playing" in porch
    assert "Paused" in next(l for l in lines if l.startswith("Kitchen"))


def test_an_unreachable_leader_is_marked_and_the_others_still_report():
    lines = make(FakeClient(unreachable={1})).zone_status()
    assert "unreachable" in next(l for l in lines if l.startswith("Kitchen"))
    assert "Blue Sky" in next(l for l in lines if l.startswith("Den"))


def test_the_favorites_are_numbered_and_capped():
    favorites = [{"name": f"Station {n}"} for n in range(HeosIntegration.FAVORITES_IN_PROMPT + 3)]
    heos = make(FakeClient(favorites=favorites), spotify=Mock())
    heos.refresh_library()
    lines = heos.library_text().splitlines()
    assert len(lines) == HeosIntegration.FAVORITES_IN_PROMPT
    assert "Station 0" in lines[0] and "H-0" in lines[0]
    assert f"H-{HeosIntegration.FAVORITES_IN_PROMPT - 1}" in lines[-1]


def test_a_failed_favorites_reload_keeps_the_old_list_and_still_reloads_spotify():
    client = FakeClient()
    spotify = Mock()
    heos = make(client, spotify=spotify)
    heos.refresh_library()
    client.favorites = HeosError("busy")
    heos.refresh_library()
    assert "Jazz FM" in heos.library_text()
    assert spotify.refresh_library.call_count == 2


def test_a_favorite_plays_its_one_based_preset_on_the_leader():
    client = FakeClient()
    heos = make(client, spotify=Mock())
    heos.refresh_library()
    assert heos.play_id("Porch", "H-1") == ""
    assert client.sent_commands("browse/play_preset") == [{"pid": 2, "preset": 2}]


def test_a_favorite_out_of_range_is_refused():
    client = FakeClient()
    heos = make(client, spotify=Mock())
    heos.refresh_library()
    assert heos.play_id("Kitchen", "H-5")
    assert not client.sent_commands("browse/play_preset")


@pytest.mark.parametrize("music_id, uri", [("AB", "spotify:track:t1"), ("S-0", "spotify:playlist:p1"),
                                           ("spotify:album:alb1", "spotify:album:alb1")])
def test_a_spotify_id_is_started_on_the_leaders_device(music_id, uri):
    spotify = Mock()
    spotify.resolve.return_value = "spotify:track:t1"
    spotify.playlist.return_value = Mock(uri="spotify:playlist:p1")
    heos = make(spotify=spotify)
    connect = Mock()
    with patch.object(heos, "connect", return_value=connect):
        assert heos.play_id("Porch", music_id) == ""
    connect.play.assert_called_once_with("Den", uri)


def test_a_player_the_account_does_not_list_says_to_pick_it_in_the_spotify_app():
    heos = make(spotify=Mock())
    connect = Mock()
    connect.play.side_effect = NotListed("Kitchen is not in the Spotify account's device list")
    with patch.object(heos, "connect", return_value=connect):
        result = heos.play_id("Kitchen", "spotify:playlist:p1")
    assert "Spotify app" in result


def test_an_unknown_id_is_refused_without_starting_anything():
    spotify = Mock()
    spotify.resolve.return_value = None
    heos = make(spotify=spotify)
    connect = Mock()
    with patch.object(heos, "connect", return_value=connect):
        assert heos.play_id("Kitchen", "ZZ")
        assert heos.play_id("Kitchen", "F-0")
    connect.play.assert_not_called()


def test_health_is_false_without_players():
    assert make(FakeClient(players=[])).check_health() is False


def test_the_speakers_composite_resolves_a_heos_room_to_it():
    heos = make()
    system, zone = Speakers([heos]).resolve("porch")
    assert system is heos and zone == "Porch"
    assert Speakers([heos]).owner_of_id("H-3") is heos


PORCH_BUTTONS = {3: [{"id": 1, "name": "Morning"}, {"id": 2, "name": "Evening"}]}


def test_a_players_quick_selects_are_listed_under_its_room_and_only_where_there_are_some():
    heos = make(FakeClient(quickselects=PORCH_BUTTONS), room_mapping={"Porch": "Veranda"}, spotify=Mock())
    heos.refresh_library()
    buttons = [line for line in heos.library_text().splitlines() if "Q-" in line]
    assert len(buttons) == 1
    assert buttons[0].startswith("Veranda") and "Morning" in buttons[0] and "Q-1" in buttons[0] and "Q-2" in buttons[0]


def test_a_quick_select_plays_on_its_own_player_even_in_a_group():
    client = FakeClient(quickselects=PORCH_BUTTONS)
    heos = make(client, spotify=Mock())
    heos.refresh_library()
    assert heos.play_id("Porch", "Q-2") == ""
    assert client.sent_commands("player/play_quickselect") == [{"pid": 3, "id": 2}]


def test_a_quick_select_the_room_does_not_have_is_refused_naming_the_ones_it_has():
    client = FakeClient(quickselects=PORCH_BUTTONS)
    heos = make(client, spotify=Mock())
    heos.refresh_library()
    assert "Q-1" in heos.play_id("Porch", "Q-5")
    assert heos.play_id("Kitchen", "Q-1")
    assert not client.sent_commands("player/play_quickselect")


def test_a_player_that_refuses_quick_selects_costs_only_itself():
    client = FakeClient(quickselects={1: HeosError("not supported"), **PORCH_BUTTONS})
    heos = make(client, spotify=Mock())
    heos.refresh_library()
    assert "Morning" in heos.library_text()


# --- the change events ------------------------------------------------------------------------

from assistant.integration.music import MusicSystem as music   # noqa: E402
from integrations.heos import HeosIntegration as heos_module   # noqa: E402


@pytest.fixture
def zone_listener():
    calls = Mock()
    music.add_zone_listener(calls)
    yield calls
    music.remove_zone_listener(calls)


def listening(heos):
    heos._events = Mock(connected=True)
    return heos


def test_with_the_events_up_each_leader_is_asked_once_and_then_served_from_the_cache():
    client = FakeClient()
    heos = listening(make(client))
    first = heos.zone_status()
    asked = {params["pid"] for params in client.sent_commands("player/get_play_state")}
    assert asked == {1, 2}
    client.sent.clear()
    assert heos.zone_status() == first
    assert not client.sent_commands("player/get_play_state")


def test_with_the_events_down_the_status_is_asked_every_time():
    client = FakeClient()
    heos = make(client)
    heos.zone_status()
    heos.zone_status()
    assert len(client.sent_commands("player/get_play_state")) == 4
    assert heos.pushes_changes() is False


def test_a_state_change_refetches_only_that_player_and_tells_the_zone_listeners(zone_listener):
    client = FakeClient()
    heos = listening(make(client))
    heos.zone_status()
    client.sent.clear()
    client.paused.add(2)
    heos._on_event("player_state_changed", {"pid": "2", "state": "pause"})
    assert [p["pid"] for p in client.sent_commands("player/get_play_state")] == [2]
    zone_listener.assert_called_once_with()
    client.sent.clear()
    heos.zone_status()
    assert not client.sent_commands("player/get_play_state")


def test_a_now_playing_change_refetches_that_player(zone_listener):
    client = FakeClient()
    heos = listening(make(client))
    heos._on_event("player_now_playing_changed", {"pid": "1"})
    assert [p["pid"] for p in client.sent_commands("player/get_now_playing_media")] == [1]
    zone_listener.assert_called_once_with()


@pytest.mark.parametrize("name", ["players_changed", "groups_changed"])
def test_a_change_of_players_or_groups_makes_the_rooms_read_again(name, zone_listener):
    client = FakeClient()
    heos = listening(make(client))
    heos.zone_status()
    client.players = [{"name": "Kitchen", "pid": 1}, {"name": "Attic", "pid": 4}]
    heos._on_event(name, {})
    assert heos._status == {}
    assert heos.zones() == ["Kitchen", "Attic"]
    zone_listener.assert_called_once_with()


def test_an_event_without_a_usable_pid_is_ignored(zone_listener):
    client = FakeClient()
    heos = listening(make(client))
    heos._on_event("player_state_changed", {"state": "play"})
    assert not client.sent
    zone_listener.assert_not_called()


def test_an_event_the_prompt_does_not_show_changes_nothing(zone_listener):
    client = FakeClient()
    heos = listening(make(client))
    heos._on_event("player_volume_changed", {"pid": "1", "level": "30"})
    assert not client.sent
    zone_listener.assert_not_called()


def test_the_stream_coming_or_going_drops_the_cached_status_and_tells_the_listeners(zone_listener):
    heos = listening(make())
    heos.zone_status()
    assert heos._status
    heos._on_events_connected(False)
    assert heos._status == {}
    zone_listener.assert_called_once_with()


def test_the_events_start_once_with_the_refresh():
    heos = make(spotify=Mock())
    with patch.object(heos_module, "HeosEvents") as events_class, \
            patch.object(heos, "refresh_library"), patch.object(heos, "_schedule_refresh"):
        heos._refresh_tick()
        heos._refresh_tick()
    events_class.assert_called_once()
    events_class.return_value.start.assert_called_once_with()


def test_a_dropped_integration_starts_no_events():
    heos = make(spotify=Mock())
    heos._on_reset()
    with patch.object(heos_module, "HeosEvents") as events_class:
        heos._start_events()
    events_class.assert_not_called()


def test_dropping_the_integration_stops_the_events():
    heos = make(spotify=Mock())
    events = heos._events = Mock()
    heos._on_reset()
    events.stop.assert_called_once_with()


def test_an_event_that_leaves_the_status_as_it_was_tells_nobody():
    """HEOS repeats most changes: the second report of the same status is no change."""
    from assistant.integration.music import MusicSystem as music
    heos = make()
    heos._events = Mock(connected=True)
    heard = []
    listener = lambda: heard.append(1)
    music.add_zone_listener(listener)
    try:
        heos._on_event("player_now_playing_changed", {"pid": "2"})
        heos._on_event("player_now_playing_changed", {"pid": "2"})
    finally:
        music.remove_zone_listener(listener)
    assert len(heard) == 1


def test_a_cached_status_past_its_age_is_asked_for_again():
    """The zones section's own look, long after the last event, must reach HEOS: a missed event heals there."""
    client = FakeClient()
    heos = listening(make(client))
    heos.zone_status()
    client.paused.add(2)   # no event tells of it
    heos._status_at = {pid: at - heos.STATUS_MAX_AGE - 1 for pid, at in heos._status_at.items()}
    lines = heos.zone_status()
    assert "Den (Blue Sky by Alice - Paused)" in lines


def test_an_unreachable_player_is_not_remembered_as_such_past_the_age():
    client = FakeClient(unreachable={1})
    heos = listening(make(client))
    assert heos.zone_status()[0] == "Kitchen (unreachable)"
    client.unreachable.clear()
    heos._status_at = {pid: at - heos.STATUS_MAX_AGE - 1 for pid, at in heos._status_at.items()}
    assert heos.zone_status()[0] == "Kitchen (Paused)"


class PlayingClient(FakeClient):
    """Every player answers the given state and media."""

    def __init__(self, state, media):
        super().__init__()
        self.state, self.media = state, media

    def command(self, cmd, **params):
        if cmd == "player/get_play_state":
            self.sent.append((cmd, params))
            return {"heos": {"message": f"pid={params['pid']}&state={self.state}"}}
        if cmd == "player/get_now_playing_media":
            self.sent.append((cmd, params))
            return {"payload": self.media}
        return super().command(cmd, **params)


@pytest.mark.parametrize("state, media, shown, hidden", [
    ("play", {"song": "Blue Sky", "artist": "Alice"}, ["Blue Sky", "Alice", "Playing"], ["Paused"]),
    ("pause", {"song": "Blue Sky", "artist": "Alice"}, ["Blue Sky", "Alice", "Paused"], ["Playing"]),
    ("play", {"song": "", "artist": "Radio One"}, ["Radio One", "Playing"], [" by "]),
    ("play", {"song": "Morning Show"}, ["Morning Show", "Playing"], [" by "]),
    ("stop", {}, ["Paused"], [" by ", " - "]),
], ids=["playing", "paused", "a station without a title", "a title without an artist", "nothing loaded"])
def test_a_zone_line_says_what_plays_with_only_the_parts_there_are(state, media, shown, hidden):
    line = make(PlayingClient(state, media)).zone_status()[0]
    assert line.startswith("Kitchen (") and line.endswith(")")
    assert all(part in line for part in shown) and not any(part in line for part in hidden)


@pytest.mark.parametrize("music_id, command", [("H-first", "browse/play_preset"), ("Q-one", "player/play_quickselect"),
                                               ("H-", "browse/play_preset")])
def test_a_list_id_without_a_number_is_refused_without_a_command(music_id, command):
    client = FakeClient(quickselects=PORCH_BUTTONS)
    heos = make(client, spotify=Mock())
    heos.refresh_library()
    assert music_id in heos.play_id("Porch", music_id)
    assert not client.sent_commands(command)


def test_a_grouped_player_whose_leader_is_not_among_the_players_acts_on_itself():
    """The leader was unplugged and the group list has not caught up: the room still answers."""
    client = FakeClient(players=[{"name": "Kitchen", "pid": 1}, {"name": "Porch", "pid": 3, "gid": 2}])
    assert make(client).pause("Porch") == ""
    assert client.sent_commands("player/set_play_state") == [{"pid": 3, "state": "pause"}]


@pytest.mark.parametrize("step, command", [(4, "group/volume_up"), (-4, "group/volume_down")])
def test_the_sign_of_the_step_is_the_direction_and_the_cli_gets_its_size(step, command):
    client = FakeClient()
    make(client).adjust_volume("Den", step)
    assert client.sent_commands(command) == [{"gid": 2, "step": 4}]


def test_a_refresh_that_fails_is_still_followed_by_the_next_one():
    heos = make(spotify=Mock())
    with patch.object(heos, "refresh_library", side_effect=RuntimeError("boom")), \
            patch.object(heos, "_schedule_refresh") as schedule:
        with pytest.raises(RuntimeError):
            heos._refresh_tick()
    schedule.assert_called_once_with(HeosIntegration.REFRESH_INTERVAL_SECONDS)


def test_a_dropped_integration_refreshes_nothing_and_schedules_nothing():
    heos = make(spotify=Mock())
    heos._on_reset()
    with patch.object(heos, "refresh_library") as refresh, patch.object(heos_module.threading, "Timer") as timer:
        heos._refresh_tick()
        heos._schedule_refresh(1)
    refresh.assert_not_called()
    timer.assert_not_called()


def test_the_spotify_library_is_reloaded_even_when_no_player_answers():
    """The favorites and the quick selects fail with the speakers; the shared Spotify library is not theirs."""
    client = FakeClient(favorites=HeosError("no answer"))
    client.players = []
    spotify = Mock()
    heos = make(client, spotify=spotify)
    with patch.object(client, "command", side_effect=HeosError("no HEOS device answered")):
        heos.refresh_library()
    spotify.refresh_library.assert_called_once_with()
