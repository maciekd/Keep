"""HeosEvents: the registration on connect, events handed on without the noise, the heart beat on a
silent connection, a broken stream told and retried, stop() ending the thread."""
import json
import socket
import threading
from unittest.mock import Mock, patch

import pytest

from integrations.heos import HeosEvents as module
from integrations.heos.HeosEvents import HeosEvents

HOST = "192.0.2.21"
REGISTERED = {"heos": {"command": "system/register_for_change_events", "result": "success", "message": "enable=on"}}


def line(answer: dict) -> bytes:
    return json.dumps(answer).encode() + b"\r\n"


def event(name: str, message: str = "") -> bytes:
    return line({"heos": {"command": f"event/{name}", "message": message}})


class FakeSocket:
    """recv answers from `script`: bytes, or socket.timeout to raise; an empty script is a closed connection."""

    def __init__(self, script):
        self.script = list(script)
        self.sent = []

    def settimeout(self, seconds):
        pass

    def sendall(self, data):
        self.sent.append(data)

    def recv(self, size):
        if not self.script:
            return b""
        item = self.script.pop(0)
        if item is socket.timeout:
            raise socket.timeout()
        return item

    def close(self):
        pass


def listen(script, on_event=None, on_connected=None):
    """Runs one `_listen` over a fake connection; returns (events, the socket, the exception that ended it)."""
    sock = FakeSocket(script)
    events = HeosEvents(lambda: HOST, on_event or Mock(), on_connected or Mock())
    with patch.object(module.socket, "create_connection", return_value=sock) as connect:
        try:
            events._listen(HOST)
            ended = None
        except (ConnectionError, OSError) as e:
            ended = e
    connect.assert_called_once()
    assert connect.call_args.args[0] == (HOST, module.PORT)
    return events, sock, ended


def test_it_registers_for_the_change_events_on_connect():
    _, sock, _ = listen([])
    assert b"register_for_change_events" in sock.sent[0] and b"enable=on" in sock.sent[0]


def test_a_successful_registration_means_connected():
    on_connected = Mock()
    events, _, _ = listen([line(REGISTERED)], on_connected=on_connected)
    on_connected.assert_called_with(True)
    assert events.connected


def test_a_refused_registration_is_not_connected():
    refused = {"heos": {"command": "system/register_for_change_events", "result": "fail", "message": "eid=2"}}
    events, _, _ = listen([line(refused)])
    assert not events.connected


def test_an_event_reaches_the_callback_by_name_with_its_fields():
    on_event = Mock()
    listen([line(REGISTERED), event("player_state_changed", "pid=7&state=pause")], on_event=on_event)
    on_event.assert_called_once_with("player_state_changed", {"pid": "7", "state": "pause"})


def test_the_playing_position_never_reaches_the_callback():
    on_event = Mock()
    listen([event("player_now_playing_progress", "pid=7&cur_pos=1000&duration=2000"), event("groups_changed")],
           on_event=on_event)
    assert [c.args[0] for c in on_event.call_args_list] == ["groups_changed"]


def test_a_silent_connection_gets_a_heart_beat_and_an_unanswered_one_ends_the_stream():
    _, sock, ended = listen([line(REGISTERED), socket.timeout, socket.timeout])
    beats = [data for data in sock.sent if b"heart_beat" in data]
    assert len(beats) == 1
    assert isinstance(ended, ConnectionError)


def test_an_answered_heart_beat_keeps_the_stream():
    beat = line({"heos": {"command": "system/heart_beat", "result": "success", "message": ""}})
    on_event = Mock()
    listen([socket.timeout, beat, socket.timeout, beat, event("players_changed")], on_event=on_event)
    on_event.assert_called_once_with("players_changed", {})


def test_a_closed_connection_ends_the_stream():
    _, _, ended = listen([line(REGISTERED)])
    assert isinstance(ended, ConnectionError)


def test_a_raising_callback_does_not_end_the_stream():
    on_event = Mock(side_effect=[RuntimeError("bad handler"), None])
    listen([event("players_changed"), event("groups_changed")], on_event=on_event)
    assert on_event.call_count == 2


def test_a_broken_stream_is_told_and_leaves_it_not_connected():
    told = []
    events = HeosEvents(lambda: HOST, Mock(), lambda up: told.append(up) or (not up and events.stop()))
    with patch.object(module, "RETRY_MIN", 0.01), \
            patch.object(module.socket, "create_connection", return_value=FakeSocket([line(REGISTERED)])):
        events._run()
    assert told == [True, False]
    assert not events.connected


def test_with_no_device_known_it_does_not_connect():
    events = None

    def host():
        events.stop()
        return ""
    events = HeosEvents(host, Mock())
    with patch.object(module, "RETRY_MIN", 0.01), patch.object(module.socket, "create_connection") as connect:
        events._run()
    connect.assert_not_called()


def test_stop_ends_the_thread():
    events = HeosEvents(lambda: "", Mock())
    with patch.object(module, "RETRY_MIN", 0.01), patch.object(module, "RETRY_MAX", 0.02):
        events.start()
        events.stop()
        events._thread.join(timeout=2)
    assert not events._thread.is_alive()


class CountedStop:
    """Stands in for the stop event: records each wait and reports the stop after `waits` of them."""

    def __init__(self, waits):
        self.left, self.waited = waits, []

    def is_set(self):
        return self.left <= 0

    def wait(self, seconds):
        self.waited.append(seconds)
        self.left -= 1
        return self.left <= 0

    def set(self):
        self.left = 0


def test_the_backoff_starts_over_after_a_stream_that_listened():
    """Two refused connections double the wait; a stream that registered and then broke is retried from the start."""
    events = HeosEvents(lambda: HOST, Mock())
    events._stop = CountedStop(4)
    connections = [OSError("refused"), OSError("refused"), FakeSocket([line(REGISTERED)]), OSError("refused")]

    def create_connection(address, timeout=None):
        made = connections.pop(0)
        if isinstance(made, Exception):
            raise made
        return made
    with patch.object(module.socket, "create_connection", side_effect=create_connection):
        events._run()
    first = module.RETRY_MIN
    assert events._stop.waited == [first, first * 2, first, first * 2]


def test_an_event_cut_in_two_by_the_network_arrives_once_and_whole():
    on_event = Mock()
    whole = event("player_state_changed", "pid=7&state=play")
    listen([whole[:20], whole[20:] + event("groups_changed")], on_event=on_event)
    assert [c.args for c in on_event.call_args_list] == [("player_state_changed", {"pid": "7", "state": "play"}),
                                                         ("groups_changed", {})]


def test_a_device_that_keeps_refusing_is_asked_ever_more_rarely_up_to_the_cap():
    events = HeosEvents(lambda: HOST, Mock())
    events._stop = CountedStop(6)
    with patch.object(module.socket, "create_connection", side_effect=OSError("refused")):
        events._run()
    waited = events._stop.waited
    assert waited == sorted(waited) and waited[0] == module.RETRY_MIN and waited[-1] == module.RETRY_MAX
    assert all(later <= 2 * earlier for earlier, later in zip(waited, waited[1:]))


def test_an_event_that_arrives_before_the_heart_beat_is_due_resets_the_silence():
    """One quiet spell, an event, another quiet spell: two heart beats, and the stream is not given up."""
    answered = line({"heos": {"command": "system/heart_beat", "result": "success", "message": ""}})
    _, sock, ended = listen([socket.timeout, event("groups_changed"), socket.timeout, answered])
    assert sock.sent.count(b"heos://system/heart_beat\r\n") == 2
    assert "closed" in str(ended), "the script ran out — the stream itself never gave up on a heart beat"
