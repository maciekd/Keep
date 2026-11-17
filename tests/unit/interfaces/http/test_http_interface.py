"""What the HTTP interface answers when a route fails: the catch-all that sends an exception
nothing caught to the log (and from there to the error sink) instead of Quart's own page."""
import asyncio
from unittest.mock import Mock

import pytest

import common.logging as logging_module
from interfaces.http.HttpInterface import HttpInterface


@pytest.fixture
def reported(monkeypatch):
    seen = []
    monkeypatch.setattr(logging_module, "_error_sink", lambda message, exception, handled: seen.append(
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
