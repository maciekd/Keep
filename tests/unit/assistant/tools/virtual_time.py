"""A virtual clock for the browser's waits: WebAccess's `time.time()` and every `asyncio.sleep` run on it."""
import asyncio
import time
from contextlib import contextmanager
from unittest.mock import Mock, patch


class VirtualTime:
    """The clock and sleep, driven by the sleeps themselves: a sleeper goes on once its wake-up is the
    earliest one outstanding, and the clock then reads that moment. The order of what happens is the
    one the scripted times give, however long the machine takes over a step."""

    def __init__(self):
        self.now = time.time()
        self._wakes = []
        self._yield = asyncio.sleep

    def time(self):
        return self.now

    async def sleep(self, seconds):
        wake = self.now + seconds
        self._wakes.append(wake)
        await self._yield(0)                # every other task first runs up to its own sleep
        while min(self._wakes) < wake:
            await self._yield(0)
        self._wakes.remove(wake)
        self.now = max(self.now, wake)


@contextmanager
def virtual_time():
    """WebAccess reads the clock and polls through a VirtualTime, and so does the test's own script."""
    clock = VirtualTime()
    with patch("assistant.tools.WebAccess.time", Mock(wraps=time, time=clock.time)), \
            patch("assistant.tools.WebAccess.asyncio.sleep", clock.sleep):
        yield clock
