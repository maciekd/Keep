"""Assistant facade: pool, dispatch lock, kill semantics and the event stream, with a
fake handler standing in for RequestProcessor."""
import threading
import time
from unittest.mock import Mock, patch

import pytest
from types import SimpleNamespace

from assistant.RequestProcessor import RequestData
from assistant.Assistant import Assistant, AssistantBusy, AssistantStream
from assistant.AssistantEvents import Done, RoundEnd, Token, UserIdentified
from assistant.AssistantRequest import RequestContext


class FakeHandler:
    """Mimics the RequestProcessor surface Assistant touches."""
    built = 0

    def __init__(self, events=None, block=None):
        FakeHandler.built += 1
        self.is_killed = False
        self.request_data = None
        self.request_context = None
        self.dialog_manager = Mock(request_id=None)
        self.turn_gate = None
        self.is_performing_request = False
        self._events = events if events is not None else [Token("hi"), RoundEnd("hi"), Done("hi")]
        self._block = block
        self.processed_with = None

    def kill(self, reason: str = ""):
        self.is_killed = True

    def process(self, prompt_manager):
        self.processed_with = prompt_manager
        self.is_performing_request = True
        self.dialog_manager.request_id = (self.request_data.args or {}).get("request_id")

        def gen():
            for ev in self._events:
                if self._block is not None:
                    self._block.wait()
                if self.is_killed:
                    return
                yield ev
        return gen()


@pytest.fixture(autouse=True)
def _reset_counter():
    FakeHandler.built = 0


def make_assistant(pool_size=2, factory=None, **handler_kwargs):
    factory = factory or (lambda: FakeHandler(**handler_kwargs))
    a = Assistant(prompt_manager=Mock(name="pm"), handler_factory=factory, pool_size=pool_size)
    a.replenish_pool()
    return a


def ctx(**kw):
    defaults = dict(source="TextChat", user="Marcus", user_input="hello")
    defaults.update(kw)
    return RequestContext(**defaults)


class TestPool:
    def test_replenish_fills_to_pool_size(self):
        a = make_assistant(pool_size=3)
        assert a.pooled_handlers() == 3 == a.pool_size
        assert FakeHandler.built == 3

    def test_replenish_is_idempotent(self):
        a = make_assistant(pool_size=2)
        a.replenish_pool()
        assert FakeHandler.built == 2

    def test_drain_kills_every_pooled_handler(self):
        handlers = []

        def factory():
            h = FakeHandler()
            handlers.append(h)
            return h

        a = make_assistant(pool_size=2, factory=factory)
        a.drain_pool()
        assert a.pooled_handlers() == 0
        assert all(h.is_killed for h in handlers)

    def test_drain_survives_a_handler_that_fails_to_kill(self):
        bad = FakeHandler()
        bad.kill = Mock(side_effect=RuntimeError("boom"))
        a = make_assistant(pool_size=1, factory=lambda: bad)
        a.drain_pool()  # must not raise
        assert a.pooled_handlers() == 0

    def test_stream_takes_from_the_pool(self):
        a = make_assistant(pool_size=2)
        a.stream(ctx())
        assert a.pooled_handlers() == 1

    def test_replenishment_loop_drains_on_time_of_day_change(self):
        a = make_assistant(pool_size=1)
        a.REPLENISH_INTERVAL = 0.01
        first = a.running_handlers()  # empty; just exercising the API
        assert first == []
        with patch("assistant.Assistant.TimeOfDay.get_current", side_effect=["day", "night", "night", "night"] + ["night"] * 50), \
             patch.object(a, "drain_pool", wraps=a.drain_pool) as drain:
            t = threading.Thread(target=a._replenishment_loop, daemon=True)
            t.start()
            deadline = time.time() + 2
            while not (drain.called and a.pooled_handlers() == 1) and time.time() < deadline:
                time.sleep(0.01)
        assert drain.called
        assert a.pooled_handlers() == 1
        a._stop_event.set()
        t.join(timeout=2)
        assert not t.is_alive(), "stop must end the loop mid-wait, not after the interval"

    def test_stop_ends_the_loop_kills_the_pool_and_stops_the_prompt_manager(self):
        a = make_assistant(pool_size=2)
        a.REPLENISH_INTERVAL = 60
        a.start()
        assert a._replenish_thread.is_alive()
        pooled = list(a._pool.queue)     # a snapshot: taking them out would race the Replenish thread
        thread = a._replenish_thread
        a.stop()
        assert not thread.is_alive() and a._replenish_thread is None
        assert a.pooled_handlers() == 0
        assert pooled and all(h.is_killed for h in pooled)
        a.prompt_manager.stop.assert_called_once()

    def test_stop_ends_a_loop_sitting_in_the_retry_backoff(self):
        # A failing replenish parks the loop for REPLENISH_RETRY_DELAY (5 s) — longer than
        # stop()'s 2 s join, so a backoff on time.sleep would leave the thread alive.
        a = make_assistant(pool_size=1)
        failed = threading.Event()

        def broken():
            failed.set()
            raise RuntimeError("factory down")

        a.replenish_pool = broken
        a.start()
        thread = a._replenish_thread
        assert failed.wait(2)
        started = time.time()
        a.stop()
        assert not thread.is_alive() and a._replenish_thread is None
        assert time.time() - started < 1

    def test_start_launches_the_replenish_thread_once(self):
        a = make_assistant(pool_size=1)
        with patch("assistant.Assistant.threading.Thread") as thread_cls:
            a.start()
            a.start()
        assert thread_cls.call_count == 1
        assert thread_cls.call_args.kwargs["name"] == "Replenish"
        thread_cls.return_value.start.assert_called_once()


class TestDispatch:
    def test_stream_wires_context_and_request_data_into_the_handler(self):
        a = make_assistant()
        rd = RequestData("GET", "/stream", {"query": "hello", "request_id": "r1"}, {})
        s = a.stream(ctx(), request_data=rd)
        h = a.running_handlers()[0]
        assert h.request_data is rd
        assert h.request_context.user_input == "hello"
        assert h.processed_with is a.prompt_manager
        assert isinstance(s, AssistantStream)

    def test_query_and_images_override_the_context(self):
        a = make_assistant()
        c = ctx(user_input="old")
        a.stream(c, query="new", images=[{"data": "x"}])
        assert c.user_input == "new"
        assert c.images == [{"data": "x"}]

    def test_in_process_caller_gets_an_empty_request_data(self):
        a = make_assistant()
        a.stream(ctx())
        rd = a.running_handlers()[0].request_data
        assert rd.args == {} and rd.headers == {}

    def test_event_sequence_starts_with_user_identified(self):
        a = make_assistant()
        events = list(a.stream(ctx()))
        assert events == [UserIdentified("Marcus"), Token("hi"), RoundEnd("hi"), Done("hi")]

    def test_no_user_identified_without_a_user(self):
        a = make_assistant()
        assert list(a.stream(ctx(user=None))) == [Token("hi"), RoundEnd("hi"), Done("hi")]

    def test_new_request_kills_the_running_one(self):
        a = make_assistant(pool_size=2)
        a.stream(ctx())
        first = a.running_handlers()[0]
        a.stream(ctx())
        assert first.is_killed
        assert a.running_handlers() == [] or first not in a.running_handlers()
        assert len(a.running_handlers()) == 1

    def test_a_guest_request_does_not_supersede_a_running_one(self):
        # A group member who is not an owner must not abort the owners' turns by saying
        # her name: the guest's turn is refused as busy and the running one is left alone.
        a = make_assistant(pool_size=2)
        a.stream(ctx())
        first = a.running_handlers()[0]
        with pytest.raises(AssistantBusy, match="guest turn does not supersede"):
            a.stream(ctx(user="Bob", guest=True))
        assert not first.is_killed and a.running_handlers() == [first]
        assert a.pooled_handlers() == 1  # nothing was taken from the pool for the refused turn

    def test_a_guest_request_runs_when_the_previous_one_is_over(self):
        # A handler stays in the running set after its turn until the next dispatch: that
        # must not count as in flight, or guests would be refused nearly always.
        a = make_assistant(pool_size=2)
        a.stream(ctx())
        first = a.running_handlers()[0]
        first.is_performing_request = False  # what the generator's finally does when the turn ends
        a.stream(ctx(user="Bob", guest=True))
        assert first.is_killed and len(a.running_handlers()) == 1

    def test_a_guest_request_runs_over_one_killed_before_its_turn(self):
        # Killed before generation began, a handler never clears its flag; it is not in flight.
        a = make_assistant(pool_size=2)
        a.stream(ctx())
        first = a.running_handlers()[0]
        first.kill("test")
        a.stream(ctx(user="Bob", guest=True))
        assert len(a.running_handlers()) == 1

    def test_busy_for_a_channel_sees_only_that_channel_s_turn(self):
        # An announcement waits for a voice turn, not for the Telegram turn whose post_message
        # it is — that turn is in flight while its own command runs, so it must not count.
        a = make_assistant(pool_size=2)
        a.stream(ctx())
        handler = a.running_handlers()[0]
        handler.ctx = SimpleNamespace(source="Telegram")
        assert a.is_busy() is True
        assert a.is_busy("Telegram") is True
        assert a.is_busy("VoiceToVoice") is False
        handler.is_performing_request = False
        assert a.is_busy() is False

    def test_busy_leaves_out_the_turn_that_asks(self):
        # A phone outpost's voice turn posting to the speakers is itself a voice turn in flight:
        # asked with its own context it does not count, another voice turn's context still does.
        a = make_assistant(pool_size=2)
        a.stream(ctx())
        handler = a.running_handlers()[0]
        handler.ctx = SimpleNamespace(source="VoiceToVoice")
        assert a.is_busy("VoiceToVoice") is True
        assert a.is_busy("VoiceToVoice", except_ctx=handler.ctx) is False
        assert a.is_busy("VoiceToVoice", except_ctx=SimpleNamespace(source="VoiceToVoice")) is True

    def test_an_owner_request_still_supersedes_a_running_one(self):
        a = make_assistant(pool_size=2)
        a.stream(ctx())
        first = a.running_handlers()[0]
        a.stream(ctx(user="Charlotte"))
        assert first.is_killed

    def test_dispatch_lock_is_released_after_a_refused_guest_request(self):
        a = make_assistant(pool_size=2)
        a.stream(ctx())
        with pytest.raises(AssistantBusy):
            a.stream(ctx(user="Bob", guest=True))
        assert a._dispatch_lock.acquire(timeout=0.05)
        a._dispatch_lock.release()

    def test_pool_exhausted_raises_busy(self):
        a = make_assistant(pool_size=1)
        a.POOL_GET_TIMEOUT = 0.05
        a.stream(ctx())
        with pytest.raises(AssistantBusy, match="no available handlers"):
            a.stream(ctx())

    def test_dispatch_lock_timeout_raises_busy(self):
        a = make_assistant()
        a.DISPATCH_LOCK_TIMEOUT = 0.05
        a._dispatch_lock.acquire()
        try:
            with pytest.raises(AssistantBusy, match="another request"):
                a.stream(ctx())
        finally:
            a._dispatch_lock.release()

    def test_dispatch_lock_is_released_after_busy(self):
        a = make_assistant(pool_size=1)
        a.POOL_GET_TIMEOUT = 0.05
        a.stream(ctx())
        with pytest.raises(AssistantBusy):
            a.stream(ctx())
        assert a._dispatch_lock.acquire(timeout=0.1)
        a._dispatch_lock.release()

class TestKill:
    def test_kill_all_kills_and_clears_running(self):
        a = make_assistant(pool_size=2)
        a.stream(ctx())
        h = a.running_handlers()[0]
        a.kill_all("test")
        assert h.is_killed
        assert a.running_handlers() == []

    def test_kill_by_request_id(self):
        a = make_assistant(pool_size=2)
        rd = RequestData("GET", "/", {"request_id": "abc"}, {})
        a.stream(ctx(), request_data=rd)
        h = a.running_handlers()[0]
        assert a.kill("abc") is True
        assert h.is_killed

    def test_kill_unknown_request_id_returns_false(self):
        a = make_assistant()
        a.stream(ctx())
        assert a.kill("nope") is False
        assert not a.running_handlers()[0].is_killed

    def test_stream_reports_killed_and_ends_without_done(self):
        release = threading.Event()
        a = make_assistant(block=release)
        s = a.stream(ctx())
        assert next(s) == UserIdentified("Marcus")
        assert s.killed is False
        s.kill()
        release.set()
        assert s.killed is True
        assert list(s) == []
