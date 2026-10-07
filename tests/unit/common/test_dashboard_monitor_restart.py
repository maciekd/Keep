"""The four dashboard monitors are stopped and started by tab switches, on the TUI's input
thread: a stop must not wait for a loop that is mid-sample, and coming back to the tab before
that loop has ended must still leave a loop running."""
import threading
from unittest.mock import Mock

import pytest

from common.AnthropicCostMonitor import AnthropicCostMonitor
from common.GpuMonitor import GpuMonitor
from common.MemoryLogMonitor import MemoryLogMonitor
from common.TempMonitor import TempMonitor

MONITORS = [(GpuMonitor, "_update_metrics"), (AnthropicCostMonitor, "_fetch_cost_data"),
            (MemoryLogMonitor, "_scan"), (TempMonitor, "_sample")]


@pytest.fixture(params=MONITORS, ids=lambda m: m[0].__name__)
def stuck(request, monkeypatch):
    """A started monitor whose first sample hangs until `release` is set; later samples count."""
    cls, sample_method = request.param
    monitor = cls(Mock())
    entered, release, samples = threading.Event(), threading.Event(), []

    def sample():
        samples.append(threading.current_thread())
        if len(samples) == 1:
            entered.set()
            release.wait(5)

    monkeypatch.setattr(monitor, sample_method, sample)
    monitor.start()
    assert entered.wait(5)
    yield monitor, release, samples
    release.set()
    monitor.stop()


@pytest.mark.unit
def test_stop_does_not_wait_for_a_loop_that_is_mid_sample(stuck, monkeypatch):
    monitor, release, _ = stuck
    first = monitor._thread
    join, waited = first.join, Mock()
    monkeypatch.setattr(first, "join", waited)
    monitor.stop()
    assert not waited.called, "stop waited for the loop on the caller's thread"
    release.set()
    join(5)
    assert not first.is_alive(), "the stopped loop never ended"


@pytest.mark.unit
def test_a_start_while_the_stopped_loop_is_still_sampling_leaves_a_loop_running(stuck):
    monitor, release, _ = stuck
    first = monitor._thread
    monitor.stop()
    monitor.start()
    second = monitor._thread
    release.set()
    first.join(5)
    assert not first.is_alive(), "the stopped loop was revived by the restart"
    assert second is not first and second.is_alive(), "no loop is left running after the restart"


@pytest.mark.unit
def test_a_second_start_on_a_running_monitor_keeps_the_one_loop(stuck):
    monitor, _, samples = stuck
    first = monitor._thread
    monitor.start()
    assert monitor._thread is first, "a second start spawned another loop beside the running one"
    assert len(samples) == 1
