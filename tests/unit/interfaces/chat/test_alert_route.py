"""Where a message Keep sends on its own goes (docs/plans/APP_PUSH_PLAN.md): every chat that is a
list the owner reads when they look gets it, and exactly one chat — the one `notify_channel`
names while it reaches somebody — makes the owner look."""
import pytest

from assistant import Destinations
from interfaces.chat.ChatHub import ChatHub

from .fakes import FakeTransport, set_notify_channel

MESSAGE = "The laundry is done."


@pytest.fixture(autouse=True)
def reset_singleton():
    ChatHub.reset_instance()
    Destinations.clear()
    yield
    ChatHub.reset_instance()
    Destinations.clear()


class House:
    """A hub with the three chats of an install: the console's pane and the app's transcript,
    which are lists, and Telegram, where a message is its own alert. *phone*: a phone with the
    app takes notifications. *telegram*: Telegram has somebody to notify."""

    def __init__(self, phone: bool, telegram: bool):
        self.console = FakeTransport(owners=["tui"], name="Console", source="Console")
        self.app = FakeTransport(owners=["app"], name="App", source="App", channel="app", alerts_now=phone)
        self.telegram = FakeTransport(owners=[111] if telegram else [], name="Telegram", source="Telegram",
                                      channel="telegram", message_is_alert=True, alerts_now=telegram)
        self.hub = ChatHub()
        for transport in (self.console, self.telegram, self.app):
            self.hub.register(transport)

    def got(self) -> dict:
        """Per chat, the texts it was sent."""
        return {t.name: t.sent_texts for t in (self.console, self.app, self.telegram)}

    def alerted(self) -> dict:
        return {t.name: [text for _, text in t.alerted] for t in (self.console, self.app, self.telegram)}


@pytest.mark.unit
class TestWhereAnAutonomousMessageGoes:
    """The rows of the plan's table."""

    def test_app_and_a_phone_is_registered(self, monkeypatch):
        set_notify_channel(monkeypatch, "app")
        house = House(phone=True, telegram=True)
        assert house.hub.send_message(MESSAGE)
        assert house.got() == {"Console": [MESSAGE], "App": [MESSAGE], "Telegram": []}
        assert house.alerted() == {"Console": [], "App": [MESSAGE], "Telegram": []}

    def test_app_and_no_phone_is_registered(self, monkeypatch):
        set_notify_channel(monkeypatch, "app")
        house = House(phone=False, telegram=True)
        assert house.hub.send_message(MESSAGE)
        assert house.got() == {"Console": [MESSAGE], "App": [MESSAGE], "Telegram": [MESSAGE]}
        assert house.app.alerted == []

    def test_telegram_and_it_has_someone_to_notify(self, monkeypatch):
        set_notify_channel(monkeypatch, "telegram")
        house = House(phone=True, telegram=True)
        assert house.hub.send_message(MESSAGE)
        assert house.got() == {"Console": [MESSAGE], "App": [MESSAGE], "Telegram": [MESSAGE]}
        assert house.app.alerted == []

    def test_telegram_and_it_has_nobody_to_notify(self, monkeypatch):
        set_notify_channel(monkeypatch, "telegram")
        house = House(phone=True, telegram=False)
        assert house.hub.send_message(MESSAGE)
        assert house.got() == {"Console": [MESSAGE], "App": [MESSAGE], "Telegram": []}
        assert house.alerted()["App"] == [MESSAGE]

    @pytest.mark.parametrize("channel", ["app", "telegram"])
    def test_neither_reaches_anybody_and_the_lists_alone_have_it(self, monkeypatch, channel):
        set_notify_channel(monkeypatch, channel)
        house = House(phone=False, telegram=False)
        assert house.hub.send_message(MESSAGE)
        assert house.got() == {"Console": [MESSAGE], "App": [MESSAGE], "Telegram": []}
        assert house.alerted() == {"Console": [], "App": [], "Telegram": []}

    def test_an_install_that_never_set_the_key_announces_in_the_app(self, monkeypatch):
        """`notify_channel: app` is the default config.example.yaml documents."""
        set_notify_channel(monkeypatch, None)
        house = House(phone=True, telegram=True)
        house.hub.send_message(MESSAGE)
        assert house.got()["Telegram"] == [] and house.alerted()["App"] == [MESSAGE]

    @pytest.mark.parametrize("value", ["carrier pigeon", ""], ids=["an unknown chat", "empty"])
    def test_a_value_that_names_no_chat_still_announces_in_exactly_one(self, monkeypatch, value):
        set_notify_channel(monkeypatch, value)
        house = House(phone=True, telegram=True)
        assert house.hub.send_message(MESSAGE)
        assert sorted([bool(house.app.alerted), bool(house.telegram.sent_texts)]) == [False, True]

    def test_a_value_is_read_as_a_person_types_it(self, monkeypatch):
        set_notify_channel(monkeypatch, " Telegram ")
        house = House(phone=True, telegram=True)
        house.hub.send_message(MESSAGE)
        assert house.got()["Telegram"] == [MESSAGE] and house.app.alerted == []

    def test_the_key_is_read_at_every_message(self, monkeypatch):
        house = House(phone=True, telegram=True)
        set_notify_channel(monkeypatch, "app")
        house.hub.send_message("one")
        set_notify_channel(monkeypatch, "telegram")
        house.hub.send_message("two")
        assert house.got()["Telegram"] == ["two"] and house.alerted()["App"] == ["one"]


@pytest.mark.unit
class TestTheAlert:
    def test_the_chat_that_alerts_is_the_named_one_while_it_reaches_somebody(self, monkeypatch):
        set_notify_channel(monkeypatch, "telegram")
        assert House(phone=True, telegram=True).hub.alert_channel().name == "Telegram"
        assert House(phone=True, telegram=False).hub.alert_channel().name == "App"
        assert House(phone=False, telegram=False).hub.alert_channel() is None

    def test_a_silent_message_is_sent_and_alerts_nobody(self, monkeypatch):
        set_notify_channel(monkeypatch, "app")
        house = House(phone=True, telegram=True)
        assert house.hub.send_message(MESSAGE, disable_notification=True)
        assert house.got()["App"] == [MESSAGE] and house.app.alerted == []
        assert [call[3] for call in house.app.calls if call[0] == "text"] == [True]

    def test_a_message_no_chat_took_alerts_nobody(self, monkeypatch):
        set_notify_channel(monkeypatch, "app")
        house = House(phone=True, telegram=True)
        house.console.accept = house.app.accept = False
        assert house.hub.send_message(MESSAGE) is False
        assert house.app.alerted == []

    def test_an_empty_message_alerts_nobody(self, monkeypatch):
        set_notify_channel(monkeypatch, "app")
        house = House(phone=True, telegram=True)
        assert house.hub.send_message("   ") is False
        assert house.app.alerted == []

    def test_post_message_to_the_owner_takes_this_route(self, monkeypatch):
        set_notify_channel(monkeypatch, "app")
        house = House(phone=True, telegram=True)
        Destinations.find(ChatHub.OWNER_DESTINATION).send(MESSAGE)
        assert house.got()["Telegram"] == [] and house.alerted()["App"] == [MESSAGE]

    def test_a_chat_that_cannot_say_whether_it_alerts_is_passed_over(self, monkeypatch):
        set_notify_channel(monkeypatch, "app")
        house = House(phone=True, telegram=True)

        def broken():
            raise RuntimeError("the registry is gone")
        house.app.alerts = broken
        assert house.hub.send_message(MESSAGE)
        assert house.got() == {"Console": [MESSAGE], "App": [MESSAGE], "Telegram": [MESSAGE]}

    def test_an_alert_that_fails_does_not_undo_the_message(self, monkeypatch):
        set_notify_channel(monkeypatch, "app")
        house = House(phone=True, telegram=True)

        def broken(conversation, text):
            raise RuntimeError("the relay is gone")
        house.app.alert = broken
        assert house.hub.send_message(MESSAGE) is True

    def test_telegrams_notify_list_still_says_which_of_its_chats(self, monkeypatch):
        set_notify_channel(monkeypatch, "telegram")
        house = House(phone=True, telegram=True)
        house.telegram.owners = [111, 222]
        house.telegram.notify = [222]
        house.hub.send_message(MESSAGE)
        assert [call[1] for call in house.telegram.calls if call[0] == "text"] == [222]


@pytest.mark.unit
class TestMediaKeepSendsUnasked:
    def test_a_photo_is_announced_by_its_caption_where_the_alerts_are(self, monkeypatch):
        set_notify_channel(monkeypatch, "app")
        house = House(phone=True, telegram=True)
        assert house.hub.send_photo(b"jpeg", "Someone is at the front door")
        assert [call[0] for call in house.app.calls] == ["photo"] and house.telegram.calls == []
        assert house.alerted()["App"] == ["Someone is at the front door"]

    @pytest.mark.parametrize("send", [lambda hub: hub.send_photo(b"jpeg"),
                                      lambda hub: hub.send_video(b"mp4"),
                                      lambda hub: hub.send_photo_album([(b"one", ""), (b"two", "")])],
                             ids=["a photo", "a video", "an album"])
    def test_media_without_a_caption_is_announced_by_what_it_is(self, monkeypatch, send):
        set_notify_channel(monkeypatch, "app")
        house = House(phone=True, telegram=True)
        assert send(house.hub)
        (said,) = house.alerted()["App"]
        assert said.strip()

    def test_an_album_is_announced_by_the_first_caption_it_has(self, monkeypatch):
        set_notify_channel(monkeypatch, "app")
        house = House(phone=True, telegram=True)
        house.hub.send_photo_album([(b"one", ""), (b"two", "The driveway at 14:05"), (b"three", "and at 14:06")])
        assert house.alerted()["App"] == ["The driveway at 14:05"]

    def test_with_telegram_announcing_the_photo_is_there_and_in_the_apps_list(self, monkeypatch):
        set_notify_channel(monkeypatch, "telegram")
        house = House(phone=True, telegram=True)
        house.hub.send_photo(b"jpeg", "Someone is at the front door")
        assert [call[0] for call in house.telegram.calls] == ["photo"]
        assert [call[0] for call in house.app.calls] == ["photo"] and house.app.alerted == []

    def test_a_photo_that_answers_a_request_goes_back_to_its_chat_unannounced(self, monkeypatch):
        set_notify_channel(monkeypatch, "app")
        house = House(phone=True, telegram=True)
        house.hub.send_photo(b"jpeg", "The driveway now", source="Telegram", conversation=111)
        assert [call[0] for call in house.telegram.calls] == ["photo"]
        assert house.app.calls == [] and house.app.alerted == []

    def test_a_photo_for_a_request_from_outside_the_chats_goes_to_every_owner_chat_unannounced(self, monkeypatch):
        set_notify_channel(monkeypatch, "app")
        house = House(phone=True, telegram=True)
        house.hub.send_photo(b"jpeg", "The driveway now", source="VoiceToVoice", conversation="kitchen")
        assert [call[0] for call in house.telegram.calls] == ["photo"]
        assert house.app.alerted == []


@pytest.mark.unit
class TestOwnerChannels:
    """What `post_message` to `owner` amounts to at the moment — the reactivation telemetry reads it."""

    def test_they_follow_the_route(self, monkeypatch):
        house = House(phone=True, telegram=True)
        set_notify_channel(monkeypatch, "app")
        assert [name for name, _ in house.hub.owner_channels()] == ["Console", "App"]
        set_notify_channel(monkeypatch, "telegram")
        assert [name for name, _ in house.hub.owner_channels()] == ["Console", "Telegram", "App"]

    def test_each_says_whether_somebody_sees_it_now(self, monkeypatch):
        set_notify_channel(monkeypatch, "app")
        house = House(phone=True, telegram=True)
        house.console.attended_now = False
        assert dict(house.hub.owner_channels()) == {"Console": False, "App": True}

    def test_a_chat_with_two_notified_conversations_is_one_channel(self, monkeypatch):
        set_notify_channel(monkeypatch, "telegram")
        house = House(phone=False, telegram=True)
        house.telegram.owners = [111, 222]
        assert [name for name, _ in house.hub.owner_channels()].count("Telegram") == 1
