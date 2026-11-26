"""What the HTTP interface answers when a route fails: the catch-all that sends an exception
nothing caught to the log (and from there to the error sink) instead of Quart's own page."""
import asyncio
from unittest.mock import Mock

import pytest

import common.logging as logging_module
from interfaces.http import HttpInterface as module
from interfaces.http.HttpInterface import HttpInterface
from interfaces.Interface import on_shutdown
from tests.unit.waiting import FAILURE_BOUND


@pytest.fixture
def reported(monkeypatch):
    seen = []
    monkeypatch.setattr(logging_module, "_error_sink", lambda message, exception, handled, outside: seen.append(
        (message, exception, handled)))
    return seen


@pytest.fixture
def app():
    interface = HttpInterface(port=0)
    interface.start(Mock())

    @interface.app.route("/falls-over")
    async def falls_over():
        raise KeyError("zones")

    return interface.app


def _get(app, path):
    async def scenario():
        response = await app.test_client().get(path)
        return response.status_code, await response.get_json()
    return asyncio.run(scenario())


def test_a_route_that_raises_answers_500_and_is_reported_as_uncaught(app, reported):
    status, body = _get(app, "/falls-over")
    assert status == 500 and "zones" not in str(body)
    [(message, exception, handled)] = reported
    assert isinstance(exception, KeyError) and handled is False
    assert "KeyError" in message


def test_a_path_nobody_serves_is_a_404_and_no_error(app, reported):
    """Quart raises its HTTP answers as exceptions too: the catch-all must hand them back as
    they are, or every stray request would be a 500 and an error event."""
    status, _ = _get(app, "/no-such-route")
    assert status == 404
    assert reported == []


class TestShutdown:
    """serve() must return once the shutdown is asked for, whatever hypercorn waits for: the
    launcher's stop steps run only after it."""

    @staticmethod
    def _interface():
        interface = HttpInterface(port=0)
        interface.start(Mock())
        return interface

    def test_a_server_that_never_returns_is_not_waited_for_past_the_grace(self, monkeypatch):
        monkeypatch.setattr(module, "SHUTDOWN_GRACE_SECONDS", 0.01)
        interface = self._interface()
        cancelled = []

        async def hypercorn_with_a_connection_that_never_ends():
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.append(True)
                raise

        async def scenario():
            interface._shutdown_event = asyncio.Event()
            serving = asyncio.ensure_future(interface._serve_until_shutdown(hypercorn_with_a_connection_that_never_ends()))
            await asyncio.sleep(0)
            interface.request_shutdown()
            await asyncio.wait_for(serving, FAILURE_BOUND)

        asyncio.run(scenario())
        assert cancelled == [True]

    def test_a_server_that_returns_in_time_is_left_to_finish(self):
        interface = self._interface()
        finished = []

        async def hypercorn():
            await interface._shutdown_event.wait()
            await asyncio.sleep(0)
            finished.append(True)

        async def scenario():
            interface._shutdown_event = asyncio.Event()
            serving = asyncio.ensure_future(interface._serve_until_shutdown(hypercorn()))
            await asyncio.sleep(0)
            interface.request_shutdown()
            await asyncio.wait_for(serving, FAILURE_BOUND)

        asyncio.run(scenario())
        assert finished == [True]

    def test_a_server_that_fails_to_start_raises_out_of_serve(self):
        interface = self._interface()

        async def hypercorn():
            raise OSError("address already in use")

        async def scenario():
            interface._shutdown_event = asyncio.Event()
            await interface._serve_until_shutdown(hypercorn())

        with pytest.raises(OSError):
            asyncio.run(scenario())

    def test_the_shutdown_request_tells_each_hook_once_and_a_failing_one_stops_no_other(self):
        interface = self._interface()
        told = []

        def falls_over():
            raise RuntimeError("no loop")

        on_shutdown(interface.app, falls_over)
        on_shutdown(interface.app, lambda: told.append("streams"))
        interface.request_shutdown()
        interface.stop()
        assert told == ["streams"]
