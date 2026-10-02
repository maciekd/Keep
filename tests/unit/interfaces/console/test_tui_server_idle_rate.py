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
def running(monkeypatch):
    monkeypatch.setattr(module, "FRAME_INTERVAL", TICK)
    monkeypatch.setattr(module, "IDLE_INTERVAL", IDLE_TICK)
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


def test_a_wake_ends_an_idle_before_the_slot_is_up(running):
    srv, picture = running
    assert wait_until(lambda: srv.idle)
    before = picture.frames
    started = time.perf_counter()
    srv.wake()
    assert wait_until(lambda: picture.frames > before, timeout=0.1)
    assert time.perf_counter() - started < IDLE_TICK, "the woken frame did not wait out the idle slot"
    assert not srv.idle


def test_a_key_wakes_the_loop(running):
    srv, picture = running
    assert wait_until(lambda: srv.idle)
    srv._emit_key("x", ("test", 0))
    assert wait_until(lambda: not srv.idle, timeout=0.1)


def test_a_wake_landing_while_a_frame_is_being_finished_is_not_lost(running, monkeypatch):
    """A wake between the loop's read of the event and the end of the frame used to be cleared
    unseen, and the change it announced waited out a whole idle slot. The widest such window
    is the log call of the frame that turns idle, so the wake is dropped in right there."""
    srv, picture = running
    idle_at_each_log = []

    def log_and_wake_once(_message):
        idle_at_each_log.append(srv.idle)
        if len(idle_at_each_log) == 1:
            srv.wake()

    monkeypatch.setattr(module, "trace", log_and_wake_once)
    assert wait_until(lambda: len(idle_at_each_log) >= 2), "the wake was erased: the loop stayed idle"
    assert idle_at_each_log[:2] == [True, False], "the first thing after turning idle is the woken frame"
