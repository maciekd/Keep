from unittest.mock import patch

import pytest

from assistant.llm import PrefillTracker as prefill_module


class FakeClock:
    """The PrefillTracker module's `time`, standing still until a test moves it."""
    def __init__(self, now: float = 1000.0):
        self.now = now

    def monotonic(self) -> float:
        return self.now


@pytest.fixture
def prefill_clock():
    """The prefill tracker's clock, frozen: the bar carries the server's progress on at the
    measured rate since the last report, so its numbers are exact only on a clock that waits."""
    clock = FakeClock()
    with patch.object(prefill_module, "time", clock):
        yield clock
