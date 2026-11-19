
import pytest

from interfaces.voice.ios.TextStreamRegistry import TextStreamRegistry
from tests.unit.waiting import FAILURE_BOUND


@pytest.mark.unit
class TestTextStreamRegistry:
    @pytest.fixture
    def registry(self):
        TextStreamRegistry.reset_instance()
        yield TextStreamRegistry.get_instance()
        TextStreamRegistry.reset_instance()

    def test_get_or_create_returns_same_queue(self, registry):
        q1 = registry.get_or_create("req-1")
        q2 = registry.get_or_create("req-1")
        assert q1 is q2

    def test_publish_then_drain_preserves_order(self, registry):
        registry.publish("req-1", {"type": "token", "text": "Hello"})
        registry.publish("req-1", {"type": "token", "text": " world"})
        q = registry.get_or_create("req-1")
        assert q.get_nowait() == {"type": "token", "text": "Hello"}
        assert q.get_nowait() == {"type": "token", "text": " world"}

    def test_close_enqueues_sentinel(self, registry):
        registry.publish("req-1", {"type": "token", "text": "hi"})
        registry.close("req-1")
        q = registry.get_or_create("req-1")
        assert q.get_nowait() == {"type": "token", "text": "hi"}
        assert q.get_nowait() is None

    def test_full_queue_drops_events(self, registry):
        registry.MAX_QUEUED_EVENTS = 2
        q = registry.get_or_create("req-1")
        assert q.maxsize == 2
        registry.publish("req-1", {"i": 1})
        registry.publish("req-1", {"i": 2})
        registry.publish("req-1", {"i": 3})  # dropped, must not raise
        assert q.qsize() == 2

    def test_discard_removes_queue(self, registry):
        registry.get_or_create("req-1")
        registry.discard("req-1")
        assert "req-1" not in registry._queues

@pytest.mark.unit
class TestOrphanTimers:
    @pytest.fixture
    def registry(self):
        TextStreamRegistry.reset_instance()
        yield TextStreamRegistry.get_instance()
        TextStreamRegistry.reset_instance()

    def test_close_arms_a_timer_that_discard_cancels(self, registry):
        registry.close("req-1")
        timer = registry._timers["req-1"]
        assert timer.is_alive()
        registry.discard("req-1")
        timer.join(timeout=FAILURE_BOUND)
        assert not timer.is_alive() and "req-1" not in registry._timers

    def test_reset_instance_cancels_every_pending_timer(self, registry):
        registry.close("a")
        registry.close("b")
        timers = list(registry._timers.values())
        TextStreamRegistry.reset_instance()
        for timer in timers:
            timer.join(timeout=FAILURE_BOUND)
        assert not any(t.is_alive() for t in timers)
