"""The single-slot gate itself (BackgroundSlot), apart from the backends that share it: what
an interactive arrival does to a background request that will not unwind, and the idle clock a
background request waits on."""
import threading
import time
from types import SimpleNamespace

import pytest

from assistant.llm.AbstractLLM import LLMPreempted
from assistant.llm.BackgroundSlot import BackgroundSlot


@pytest.fixture
def slot():
    return BackgroundSlot("test-engine")


def _run(target):
    t = threading.Thread(target=target, daemon=True)
    t.start()
    return t


class _Clock:
    """The monotonic clock the slot's idle timer reads, moved by the test: a waiting background
    request is woken the way an interactive end would wake it, and re-reads the clock."""

    def __init__(self, monkeypatch):
        self.now = 1000.0
        monkeypatch.setattr("assistant.llm.BackgroundSlot.time", SimpleNamespace(monotonic=lambda: self.now))

    def advance(self, slot, seconds):
        with slot._cond:
            self.now += seconds
            slot._cond.notify_all()


@pytest.mark.unit
class TestInteractiveArrival:
    def test_a_background_request_that_never_unwinds_does_not_hold_the_user_turn(self, slot, monkeypatch):
        """The eviction wakes the background thread by shutting its socket, but the gate must not
        trust that: a request stuck before its socket exists (connecting) cannot be woken, and
        the user's turn goes to the engine after the unwind budget anyway, queued behind it."""
        monkeypatch.setattr(BackgroundSlot, "PREEMPT_UNWIND_TIMEOUT_S", 0.05)
        slot.acquire(idle_s=0)           # a background request holds the slot, no socket yet
        t0 = time.monotonic()
        slot.interactive_begin()
        assert time.monotonic() - t0 < 1
        assert slot.abort is True and slot.interactive_active == 1
        assert slot.active is True       # the gate did not pretend the slot was freed

    def test_a_connection_registered_after_the_eviction_is_closed_and_the_request_preempted(self, slot, monkeypatch):
        """The background request had not sent yet when the user's turn arrived: sending now would
        put a 60 s image prefill ahead of the turn in the engine. It stands down instead and the
        connection it was about to use is closed, not leaked."""
        monkeypatch.setattr(BackgroundSlot, "PREEMPT_UNWIND_TIMEOUT_S", 0.05)
        slot.acquire(idle_s=0)
        slot.interactive_begin()

        class Conn:
            sock, closed = object(), False

            def close(self):
                self.closed = True

        conn = Conn()
        with pytest.raises(LLMPreempted):
            slot.register_connection(conn)
        assert conn.closed is True
        assert slot.sock is None
        assert slot.preempted(chunks=0) is not None    # the failure that follows is the eviction, not a real one

    def test_interactive_requests_do_not_wait_for_each_other(self, slot):
        """The gate is between the user and the background work only: two interactive requests
        (a turn and the post-turn pre-heat) pass straight through, and the count never drops
        below zero when an end is reported twice."""
        slot.interactive_begin()
        t0 = time.monotonic()
        slot.interactive_begin()
        assert time.monotonic() - t0 < 0.05
        assert slot.interactive_active == 2
        for _ in range(3):
            slot.interactive_end()
        assert slot.interactive_active == 0


@pytest.mark.unit
class TestIdleClock:
    def test_a_background_request_waits_out_the_idle_time_after_the_last_interactive_request(self, slot, monkeypatch):
        clock = _Clock(monkeypatch)
        slot.interactive_begin()
        slot.interactive_end()
        got = threading.Event()
        _run(lambda: (slot.acquire(idle_s=20), got.set()))
        assert not got.wait(0.02)
        clock.advance(slot, 19)
        assert not got.wait(0.02)               # a second short of the idle time is not enough
        clock.advance(slot, 1)
        assert got.wait(1)
        assert slot.wait_seconds() == 0.0       # the main thread queued nothing: the wait is per thread

    def test_an_end_that_does_not_reset_the_clock_lets_the_background_request_through(self, slot):
        """The pre-heat refresh is interactive while it runs (it must not share the engine with an
        image) but is not the user: it does not push the idle clock out, or a periodic refresh
        would starve the background work for good."""
        slot.interactive_begin()
        slot.interactive_end(resets_idle=False)
        got = threading.Event()
        _run(lambda: (slot.acquire(idle_s=5), got.set()))
        assert got.wait(1)
        assert slot.active is True

    def test_a_released_slot_is_free_for_the_next_background_request_at_once(self, slot):
        slot.acquire(idle_s=0)
        got = threading.Event()
        _run(lambda: (slot.acquire(idle_s=0), got.set()))
        assert not got.wait(0.05)
        slot.release()
        assert got.wait(1)
