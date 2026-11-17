"""An exception nothing caught is logged as an error and handed to the error sink."""
import asyncio
import sys
import threading

import pytest

import common.logging as logging_module
from common import UncaughtExceptions


@pytest.fixture
def reported(monkeypatch):
    """What the error sink was handed, with the process's own hooks put back afterwards."""
    seen = []
    monkeypatch.setattr(logging_module, "_error_sink", lambda message, exception, handled: seen.append(
        (message, exception, handled)))
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)
    return seen


def test_an_exception_that_ends_a_thread_is_reported_unhandled(reported):
    UncaughtExceptions.install()

    def work():
        raise ValueError("the feed fell over")
    thread = threading.Thread(target=work, name="photo-feed")
    thread.start()
    thread.join()
    [(message, exception, handled)] = reported
    assert isinstance(exception, ValueError) and handled is False
    assert "photo-feed" in message and "the feed fell over" in message


def test_an_exception_on_the_main_thread_is_logged_with_its_traceback_and_reported_without(reported, monkeypatch):
    logged = []
    monkeypatch.setattr(logging_module, "log", lambda message, *_: logged.append(message))
    UncaughtExceptions.install()
    try:
        raise RuntimeError("the launcher fell over")
    except RuntimeError as e:
        sys.excepthook(type(e), e, e.__traceback__)
    [(message, exception, handled)] = reported
    assert isinstance(exception, RuntimeError) and handled is False
    assert "RuntimeError" in message and "the launcher fell over" in message
    assert "test_uncaught_exceptions.py" not in message
    assert any("test_uncaught_exceptions.py" in line for line in logged)


def test_ctrl_c_is_not_an_error(reported, monkeypatch):
    UncaughtExceptions.install()
    monkeypatch.setattr(sys, "__excepthook__", lambda *args: None)
    sys.excepthook(KeyboardInterrupt, KeyboardInterrupt(), None)
    assert reported == []


def test_a_task_nobody_awaited_is_reported(reported):
    async def scenario():
        loop = asyncio.get_running_loop()
        UncaughtExceptions.watch_loop(loop)
        loop.call_exception_handler({"message": "Task exception was never retrieved",
                                     "exception": KeyError("zones")})
        loop.call_exception_handler({"message": "the loop is slow"})
    asyncio.run(scenario())
    assert [(type(e).__name__ if e else None, handled) for _, e, handled in reported] == [
        ("KeyError", False), (None, True)]
