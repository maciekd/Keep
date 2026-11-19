"""An event that parks a section's loop in its wait until stop(), so a test sees stop() end the wait."""
import threading

from tests.unit.waiting import FAILURE_BOUND


class ParkedEvent(threading.Event):
    """Stands for the event a section's `_wait` waits on: a wait runs until the event is set, whatever
    its timeout, and `waiting` says one has begun. A loop that leaves such a wait was ended by
    stop() and not by its interval running out — no clock needed to tell the two apart."""

    def __init__(self):
        super().__init__()
        self.waiting = threading.Event()

    def wait(self, timeout=None):
        self.waiting.set()
        return super().wait(FAILURE_BOUND)
