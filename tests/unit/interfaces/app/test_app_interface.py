"""AppInterface: what the launcher's start and stop leave behind — a transport on the process's
hub, the /app routes on the HTTP app, the hub as the notifier, and the log in its file."""
import asyncio

import pytest
from quart import Quart

from assistant import Destinations, Notifier
from assistant.AssistantEvents import Done, RoundEnd, Token
from interfaces.app import AppInterface as module
from interfaces.app.AppInterface import AppInterface
from interfaces.app.Transcript import Transcript
from interfaces.chat.ChatHub import ChatHub
from tests.unit.interfaces.chat.fakes import FakeAssistant, wait_until


@pytest.fixture
def started(tmp_path, monkeypatch):
    """(the interface, the HTTP app, the assistant) after start; stopped again at the end."""
    monkeypatch.setattr(module, "CACHE_DIR", tmp_path)
    notifiers = []
    monkeypatch.setattr(Notifier, "set_notifier", notifiers.append)
    ChatHub.reset_instance()
    Destinations.clear()
    interface, app = AppInterface(), Quart(__name__)
    assistant = FakeAssistant([Token("Hello"), RoundEnd("Hello"), Done("Hello")])
    interface.start(assistant, app)
    interface.notifiers = notifiers
    yield interface, app, assistant
    interface.stop()
    ChatHub.reset_instance()
    Destinations.clear()


def post(app, path, body) -> int:
    async def run():
        return (await app.test_client().post(path, json=body)).status_code
    return asyncio.run(run())


def test_a_message_posted_to_the_http_app_reaches_the_assistant_as_an_app_turn(started):
    interface, app, assistant = started
    assert post(app, "/app/send", {"id": "bubble-1", "text": "is the door locked?"}) == 200
    assert wait_until(lambda: assistant.requests)
    request, _ = assistant.requests[0]
    assert request.source == interface.transport.source
    assert wait_until(lambda: any(item["kind"] == "assistant" and item["final"]
                                  for item in interface.transport.transcript.items()))


def test_the_hub_it_joined_is_the_notifier_and_carries_the_owner_destination(started):
    interface, _, _ = started
    assert interface.notifiers == [ChatHub.get_instance()]
    assert interface.transport in ChatHub.get_instance().transports
    assert Destinations.find("owner") is not None


def test_stop_takes_the_transport_off_the_hub_and_leaves_the_log_for_the_next_start(started, tmp_path):
    interface, app, _ = started
    interface.transport.send_text("app", "The washing machine is done")
    interface.stop()
    assert interface.transport not in ChatHub.get_instance().transports
    assert [item["html"] for item in Transcript(tmp_path, autosave=False).items()] == ["The washing machine is done"]


def test_stop_before_any_start_does_nothing():
    AppInterface().stop()
