import json

import pytest

from integrations.heos import HeosClient as module
from integrations.heos.HeosClient import HeosClient, HeosError, message_fields


def line(command, result="success", message="", **extra) -> bytes:
    return (json.dumps({"heos": {"command": command, "result": result, "message": message}, **extra}) + "\r\n").encode()


class FakeSocket:
    """A CLI connection: each sendall queues the next scripted reply (bytes), or raises it when it is an exception."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.sent = []
        self.buffer = b""
        self.closed = False

    def sendall(self, data):
        self.sent.append(data.decode())
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        self.buffer += reply

    def recv(self, n):
        chunk, self.buffer = self.buffer[:n], self.buffer[n:]
        return chunk

    def close(self):
        self.closed = True


@pytest.fixture
def sockets(monkeypatch):
    """The sockets create_connection hands out, in order, and the addresses it was asked for."""
    made, addresses = [], []

    def create_connection(address, timeout=None):
        addresses.append(address[0])
        sock = made.pop(0)
        if isinstance(sock, Exception):
            raise sock
        return sock
    monkeypatch.setattr(module.socket, "create_connection", create_connection)
    return made, addresses


def test_the_answer_after_the_under_process_placeholder_is_the_one_returned(sockets):
    made, _ = sockets
    made.append(FakeSocket([line("player/get_players", message="command under process")
                            + line("player/get_players", payload=[{"name": "Kitchen", "pid": 1}])]))
    answer = HeosClient("192.0.2.10").command("player/get_players")
    assert answer["payload"] == [{"name": "Kitchen", "pid": 1}]


def test_a_line_about_another_command_is_skipped(sockets):
    made, _ = sockets
    made.append(FakeSocket([line("event/player_state_changed", message="pid=1&state=play")
                            + line("player/get_play_state", message="pid=1&state=pause")]))
    answer = HeosClient("192.0.2.10").command("player/get_play_state", pid=1)
    assert message_fields(answer)["state"] == "pause"


def test_the_command_line_carries_its_parameters(sockets):
    made, _ = sockets
    sock = FakeSocket([line("player/volume_up")])
    made.append(sock)
    HeosClient("192.0.2.10").command("player/volume_up", pid=7, step=3)
    assert sock.sent[0].startswith("heos://player/volume_up?") and "pid=7" in sock.sent[0] and "step=3" in sock.sent[0]


def test_a_refusal_raises_with_its_reason(sockets):
    made, _ = sockets
    made.append(FakeSocket([line("browse/play_preset", result="fail", message="eid=2&text=ID Not Valid")]))
    with pytest.raises(HeosError, match="ID Not Valid"):
        HeosClient("192.0.2.10").command("browse/play_preset", pid=1, preset=9)


def test_a_broken_connection_is_reopened_once_and_the_command_sent_again(sockets):
    made, _ = sockets
    first = FakeSocket([line("player/get_players", payload=[]), BrokenPipeError("gone")])
    second = FakeSocket([line("player/get_players", payload=[{"name": "Den", "pid": 2}])])
    made.extend([first, second])
    client = HeosClient("192.0.2.10")
    client.command("player/get_players")
    answer = client.command("player/get_players")
    assert answer["payload"][0]["name"] == "Den"
    assert first.closed and len(second.sent) == 1


def test_a_second_failure_in_a_row_raises(sockets):
    made, _ = sockets
    made.extend([FakeSocket([BrokenPipeError("gone")]), FakeSocket([ConnectionResetError("reset")])])
    with pytest.raises(HeosError):
        HeosClient("192.0.2.10").command("player/get_players")


def test_no_host_and_no_ssdp_answer_raises(monkeypatch, sockets):
    monkeypatch.setattr(module, "discover", lambda timeout=3.0: [])
    with pytest.raises(HeosError, match="heos.host"):
        HeosClient("").command("player/get_players")


def test_a_discovered_host_that_cannot_be_reached_is_forgotten(monkeypatch, sockets):
    made, addresses = sockets
    found = iter([["192.0.2.20"], ["192.0.2.21"]])
    monkeypatch.setattr(module, "discover", lambda timeout=3.0: next(found))
    made.extend([OSError("refused"), FakeSocket([line("player/get_players", payload=[])])])
    client = HeosClient("")
    with pytest.raises(HeosError):
        client.command("player/get_players")
    client.command("player/get_players")
    assert addresses[-1] == "192.0.2.21"


def test_a_configured_host_is_kept_when_it_cannot_be_reached(monkeypatch, sockets):
    made, addresses = sockets
    monkeypatch.setattr(module, "discover", lambda timeout=3.0: pytest.fail("a configured host is never discovered"))
    made.extend([OSError("refused"), FakeSocket([line("player/get_players", payload=[])])])
    client = HeosClient("192.0.2.10")
    with pytest.raises(HeosError):
        client.command("player/get_players")
    client.command("player/get_players")
    assert set(addresses) == {"192.0.2.10"}


def test_message_fields_reads_the_query_of_the_message():
    fields = message_fields({"heos": {"message": "pid=5&state=play&level="}})
    assert fields == {"pid": "5", "state": "play", "level": ""}


def test_discover_answers_nothing_when_multicast_is_denied(monkeypatch):
    class DeniedSocket:
        def __init__(self, *a): pass
        def setsockopt(self, *a): pass
        def settimeout(self, *a): pass
        def sendto(self, *a): raise OSError(65, "No route to host")
        def close(self): pass
    monkeypatch.setattr(module.socket, "socket", DeniedSocket)
    assert module.discover() == []


def test_a_search_nobody_answered_is_not_repeated_at_once(monkeypatch, sockets):
    """Speakers that are away: the commands that follow fail without a search each, until the wait is over."""
    made, addresses = sockets
    searches = []
    answers = [[], ["192.0.2.20"]]
    monkeypatch.setattr(module, "discover", lambda timeout=3.0: searches.append(1) or answers.pop(0))
    client = HeosClient("")
    for _ in range(3):
        with pytest.raises(HeosError, match="heos.host"):
            client.command("player/get_players")
    assert len(searches) == 1
    client._search_again_at -= module.SEARCH_AGAIN_SECONDS
    made.append(FakeSocket([line("player/get_players", payload=[])]))
    client.command("player/get_players")
    assert len(searches) == 2 and addresses == ["192.0.2.20"]


class TricklingSocket(FakeSocket):
    """A connection that hands its bytes over a few at a time, as TCP may."""

    def recv(self, n):
        return super().recv(7)


def test_an_answer_that_arrives_in_pieces_is_put_together(sockets):
    made, _ = sockets
    made.append(TricklingSocket([line("player/get_players", payload=[{"name": "Kitchen", "pid": 1}])]))
    assert HeosClient("192.0.2.10").command("player/get_players")["payload"] == [{"name": "Kitchen", "pid": 1}]


def test_the_connection_is_kept_from_one_command_to_the_next(sockets):
    """One connection for the whole system — and a line left over from the first exchange is not the second's answer."""
    made, addresses = sockets
    sock = FakeSocket([line("player/get_players", payload=[]) + line("event/players_changed"),
                       line("player/get_volume", message="pid=1&level=20")])
    made.append(sock)
    client = HeosClient("192.0.2.10")
    client.command("player/get_players")
    assert message_fields(client.command("player/get_volume", pid=1))["level"] == "20"
    assert addresses == ["192.0.2.10"] and not sock.closed


def test_a_line_that_is_no_json_counts_as_a_broken_connection(sockets):
    made, addresses = sockets
    garbled = FakeSocket([b"<html>not the CLI</html>\r\n"])
    made.extend([garbled, FakeSocket([line("player/get_players", payload=[{"name": "Den", "pid": 2}])])])
    assert HeosClient("192.0.2.10").command("player/get_players")["payload"] == [{"name": "Den", "pid": 2}]
    assert garbled.closed and len(addresses) == 2


def test_without_a_configured_host_the_first_device_that_answered_is_the_one_asked(monkeypatch, sockets):
    made, addresses = sockets
    monkeypatch.setattr(module, "discover", lambda timeout=3.0: ["192.0.2.31", "192.0.2.30"])
    made.append(FakeSocket([line("player/get_players", payload=[])]))
    client = HeosClient("")
    client.command("player/get_players")
    assert addresses == ["192.0.2.31"] and client.host == "192.0.2.31"


def test_discover_lists_each_device_once_in_the_order_they_answered(monkeypatch):
    """A device answers an SSDP search more than once (one answer per interface and repeat)."""
    answers = ["192.0.2.31", "192.0.2.30", "192.0.2.31", "192.0.2.32"]
    clock = iter(range(100))

    class AnsweringSocket:
        def __init__(self, *a): pass
        def setsockopt(self, *a): pass
        def settimeout(self, *a): pass
        def sendto(self, *a): pass
        def close(self): pass

        def recvfrom(self, size):
            if not answers:
                raise module.socket.timeout()
            return b"HTTP/1.1 200 OK\r\n\r\n", (answers.pop(0), 1900)
    monkeypatch.setattr(module.socket, "socket", AnsweringSocket)
    monkeypatch.setattr(module.time, "time", lambda: next(clock))
    assert module.discover(timeout=10) == ["192.0.2.31", "192.0.2.30", "192.0.2.32"]
