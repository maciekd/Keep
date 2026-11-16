"""A turn the server starts (the first-run brief): ChatHub.open_conversation → ChatSession.open.
The query reaches the assistant as a system turn and the reply renders like any other."""
import pytest

from assistant.AssistantEvents import Done, RoundEnd, Token
from interfaces.chat.ChatHub import ChatHub
from interfaces.chat.ChatTypes import IncomingMessage

from .fakes import FakeAssistant, FakeTransport, wait_until

BRIEF = "The server has just started…"


@pytest.fixture(autouse=True)
def reset_singleton():
    ChatHub.reset_instance()
    yield
    ChatHub.reset_instance()


def greeting():
    return FakeAssistant([Token("Hi, I'm Floxy."), RoundEnd("Hi, I'm Floxy."), Done("Hi, I'm Floxy.")])


@pytest.mark.unit
class TestOpenConversation:
    def test_the_brief_is_a_system_turn_and_the_reply_is_rendered(self):
        transport = FakeTransport(owners=[111])
        hub = ChatHub(assistant=greeting())
        hub.register(transport)
        assert hub.open_conversation(BRIEF) is True
        assert wait_until(lambda: ("reply_finished", 111) in transport.calls)
        req, query = hub.assistant.requests[0]
        assert (req.user, req.source, req.conversation, query) == ("system", "TextChat", 111, BRIEF)
        assert transport.finalized == ["Hi, I'm Floxy."]
        hub.stop()

    def test_only_the_first_owner_conversation_is_opened(self):
        console = FakeTransport(owners=["pane"], name="Console", source="Console")
        telegram = FakeTransport(owners=[1, 2], name="Telegram", source="Telegram")
        hub = ChatHub(assistant=greeting())
        hub.register(console)
        hub.register(telegram)
        assert hub.open_conversation(BRIEF) is True
        assert wait_until(lambda: ("reply_finished", "pane") in console.calls)
        assert len(hub.assistant.requests) == 1
        assert telegram.calls == []
        hub.stop()

    def test_an_unattended_console_hands_the_turn_to_the_chat_someone_is_at(self):
        # After the Full Disk Access grant the terminal is gone: the console transport is still
        # registered first, but nobody sees its pane, and the user is on Telegram.
        console = FakeTransport(owners=["pane"], name="Console", source="Console")
        console.attended_now = False
        telegram = FakeTransport(owners=[1, 2], name="Telegram", source="Telegram")
        hub = ChatHub(assistant=greeting())
        hub.register(console)
        hub.register(telegram)
        assert hub.open_conversation(BRIEF) is True
        assert wait_until(lambda: ("reply_finished", 1) in telegram.calls)
        assert len(hub.assistant.requests) == 1
        assert console.calls == []
        hub.stop()

    def test_with_no_attended_chat_the_first_one_still_takes_the_turn(self):
        console = FakeTransport(owners=["pane"], name="Console", source="Console")
        console.attended_now = False
        hub = ChatHub(assistant=greeting())
        hub.register(console)
        assert hub.open_conversation(BRIEF) is True
        assert wait_until(lambda: ("reply_finished", "pane") in console.calls)
        hub.stop()

    def test_without_an_owner_conversation_nothing_is_opened(self):
        hub = ChatHub(assistant=greeting())
        hub.register(FakeTransport(owners=[]))
        assert hub.open_conversation(BRIEF) is False
        assert hub.assistant.requests == []
        hub.stop()

    def test_reply_started_brackets_every_reply(self):
        transport = FakeTransport(owners=[111])
        hub = ChatHub(assistant=greeting())
        hub.register(transport)
        hub.receive(transport, IncomingMessage(conversation=111, sender="Anna", text="hello"))
        assert wait_until(lambda: ("reply_finished", 111) in transport.calls)
        lifecycle = [c for c in transport.calls if c[0] in ("reply_started", "reply_finished")]
        assert lifecycle == [("reply_started", 111), ("reply_finished", 111)]
        hub.stop()
