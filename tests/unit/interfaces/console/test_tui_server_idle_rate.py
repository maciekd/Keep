"""The animation loop draws at the full rate while frames change, drops to the idle rate once
the picture has stood still for IDLE_AFTER, and comes back at once on a wake (a key, a client)
or when a frame changes again."""
import socket
import threading
import time

import pytest

from interfaces.console.framework import TUIServer as module
from interfaces.console.framework.TUIServer import TUIServer

TICK = 0.002
IDLE_TICK = 0.04


def wait_until(predicate, timeout=0.5) -> bool:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if predicate():
            return True
        time.sleep(0.001)
    return predicate()


class Picture:
    """A draw callback: `text` is what every frame shows; change it and the frame changes."""
    def __init__(self):
        self.text = "still"
        self.moving = False   # True: every frame shows something new
        self.frames = 0

    def __call__(self, frame, screen):
        self.frames += 1
        screen.write_at(1, 1, f"{self.frames}" if self.moving else self.text)


@pytest.fixture
def idle_slot():
    return IDLE_TICK


@pytest.fixture
def running(monkeypatch, idle_slot):
    monkeypatch.setattr(module, "FRAME_INTERVAL", TICK)
    monkeypatch.setattr(module, "IDLE_INTERVAL", idle_slot)
    monkeypatch.setattr(module, "IDLE_AFTER", 0.01)
    picture = Picture()
    srv = TUIServer(draw_screen_callback=picture, host="127.0.0.1", port=0)
    server_end, client_end = socket.socketpair()
    srv.clients.add(server_end)
    srv.client_dimensions[server_end] = (20, 5)
    srv.client_terminals[server_end] = ""
    srv.running = True
    thread = threading.Thread(target=srv._animation_loop, daemon=True)
    thread.start()
    drain = threading.Thread(target=lambda: _drain(client_end), daemon=True)
    drain.start()
    yield srv, picture
    srv.stop()
    thread.join(1.0)
    server_end.close()
    client_end.close()


def _drain(sock):
    try:
        while sock.recv(4096):
            pass
    except OSError:
        pass


def frames_per_second(picture: Picture, seconds: float) -> float:
    before = picture.frames
    time.sleep(seconds)
    return (picture.frames - before) / seconds


def test_a_still_picture_drops_the_loop_to_the_idle_rate(running):
    srv, picture = running
    assert wait_until(lambda: srv.idle), "the unchanged frames should have idled the loop"
    assert frames_per_second(picture, 0.03) < 1 / TICK / 4, "idle frames come far slower than the full rate"


def test_a_changing_picture_keeps_the_full_rate_and_a_change_ends_an_idle(running):
    srv, picture = running
    assert wait_until(lambda: srv.idle)
    picture.moving = True
    assert wait_until(lambda: not srv.idle), "a frame with changes brings the full rate back"
    assert frames_per_second(picture, 0.03) > 1 / IDLE_TICK, "and the frames come at the full rate again"


def idle_at_each_log_from_now(srv, monkeypatch) -> list:
    """`srv.idle` as the loop itself read it at each of its log calls from here on — it logs as it
    turns idle and as it leaves that. The picture of these tests stands still, so the loop is
    idle again IDLE_AFTER after a woken frame: `srv.idle` read from the test's thread says
    whether that thread got to run within those 10 ms, and on a busy machine it does not."""
    states = []
    monkeypatch.setattr(module, "fine", lambda _message: states.append(srv.idle))
    return states


# An idle slot of a minute: a frame drawn at all after the loop went idle did not wait the slot out.
A_MINUTE = pytest.mark.parametrize("idle_slot", [60.0], ids=["an idle slot of a minute"])


@A_MINUTE
def test_a_wake_ends_an_idle_before_the_slot_is_up(running, monkeypatch):
    srv, picture = running
    assert wait_until(lambda: srv.idle)
    states = idle_at_each_log_from_now(srv, monkeypatch)
    srv.wake()
    assert wait_until(lambda: states, timeout=5), "the wake drew no frame: the loop sleeps its idle slot out"
    assert states[0] is False, "the woken frame runs at the full rate"


@A_MINUTE
def test_a_key_wakes_the_loop(running, monkeypatch):
    srv, picture = running
    assert wait_until(lambda: srv.idle)
    states = idle_at_each_log_from_now(srv, monkeypatch)
    srv._emit_key("x", ("test", 0))
    assert wait_until(lambda: states, timeout=5), "the key drew no frame: the loop sleeps its idle slot out"
    assert states[0] is False


def test_a_wake_landing_while_a_frame_is_being_finished_is_not_lost(running, monkeypatch):
    """A wake between the loop's read of the event and the end of the frame used to be cleared
    unseen, and the change it announced waited out a whole idle slot. The widest such window
    is the log call of the frame that turns idle, so the wake is dropped in right there."""
    srv, picture = running
    idle_at_each_log = []

    def log_and_wake_once(_message):
        idle_at_each_log.append(srv.idle)
        if srv.idle and idle_at_each_log.count(True) == 1:
            srv.wake()

    monkeypatch.setattr(module, "fine", log_and_wake_once)
    # A new picture, so that the loop turns idle with the hook in place: on a busy machine it had
    # already done so, and said so, before the test got to put the hook in.
    picture.text = "another still picture"

    def the_log_call_after_turning_idle():
        return idle_at_each_log[idle_at_each_log.index(True) + 1:] if True in idle_at_each_log else []

    assert wait_until(the_log_call_after_turning_idle, timeout=5), "the wake was erased: the loop stayed idle"
    assert the_log_call_after_turning_idle()[0] is False, "the first thing after turning idle is the woken frame"
