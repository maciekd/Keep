"""What of the app chat becomes a notification on a phone, and on which: AppTransport on a real
ChatHub over a recording sender (docs/plans/APP_PUSH_PLAN.md, "What is pushed")."""
import base64
import threading

import pytest

from assistant.AssistantEvents import Done, RoundEnd, Token
from interfaces.app.AppTransport import CONVERSATION, AppTransport
from interfaces.app.Devices import DeviceRefused, Devices
from interfaces.app.Transcript import Transcript
from interfaces.chat.ChatHub import ChatHub
from tests.unit.interfaces.chat.fakes import FakeAssistant, FakeTransport, set_notify_channel, wait_until

KEY = base64.b64encode(bytes(range(32))).decode()
IPHONE, IPAD = "marcus-iphone-0001", "charlotte-ipad-0002"
TOKENS = {IPHONE: "a1" * 32, IPAD: "c3" * 32}
APPROVAL_TOKEN, SECRET_TOKEN = "5" * 32, "7" * 32


class RecordingSender:
    """Stands in for `Push.Sender`: what would have gone to the relay."""

    def __init__(self):
        self.pushes = []        # (device id, kind, body, title)
        self.stopped = False

    def push(self, device, kind, body, title=""):
        self.pushes.append((device.id, kind, body, title))

    def stop(self):
        self.stopped = True

    def to(self, kind=None):
        """The phones notified, in order — of one kind of item when *kind* is given."""
        return [push[0] for push in self.pushes if kind is None or push[1] == kind]


@pytest.fixture(autouse=True)
def reset_singleton():
    ChatHub.reset_instance()
    yield
    ChatHub.reset_instance()


@pytest.fixture(autouse=True)
def the_app_is_where_messages_are_announced(monkeypatch):
    """Whatever the config.yaml of the run says; a test of the other route sets it again."""
    set_notify_channel(monkeypatch, "app")


@pytest.fixture
def sender():
    return RecordingSender()


@pytest.fixture
def transcript(tmp_path):
    return Transcript(tmp_path, autosave=False)


@pytest.fixture
def transport(transcript, sender):
    return AppTransport(transcript, Devices(), sender)


@pytest.fixture
def hub(transport):
    hub = ChatHub()
    hub.register(transport)
    transport.start(hub)
    yield hub
    hub.stop()


@pytest.fixture
def telegram(hub):
    """A chat where a message is its own alert and somebody is there to notify."""
    telegram = FakeTransport(owners=[111], name="Telegram", source="Telegram", channel="telegram",
                             message_is_alert=True, alerts_now=True)
    hub.register(telegram)
    return telegram


def register(transport, device=IPHONE, name="Marcus's iPhone"):
    transport.register_device(device, TOKENS[device], "sandbox", KEY, name)


def both_phones(transport):
    register(transport)
    register(transport, IPAD, "Charlotte's iPad")


def notices(transcript) -> list[str]:
    return [item["html"] for item in transcript.items() if item["kind"] == "notice"]


def idle(hub, transport) -> bool:
    return not hub.session_for(transport, CONVERSATION).busy


def answering(text: str) -> FakeAssistant:
    return FakeAssistant([Token(text), RoundEnd(text), Done(text)])


@pytest.mark.unit
class TestTheUserHasTheApp:
    def test_the_app_alerts_once_a_phone_is_registered(self, transport):
        assert transport.alerts() is False
        register(transport)
        assert transport.alerts() is True

    def test_a_phone_that_turned_its_notifications_off_is_forgotten(self, transport):
        register(transport)
        transport.register_device(IPHONE, None, "sandbox", KEY, "Marcus's iPhone")
        assert transport.alerts() is False

    def test_the_name_notify_channel_knows_the_app_by_is_a_contract_with_config_yaml(self, transport):
        assert transport.channel == "app"
        assert not transport.message_is_alert, "the app's chat is a list: it has every message, pushed or not"

    def test_stopping_the_transport_stops_the_sender(self, transport, hub, sender):
        transport.stop()
        assert sender.stopped


@pytest.mark.unit
class TestAMessageKeepSendsOnItsOwn:
    def test_it_is_an_item_and_a_notification_on_every_phone(self, hub, transport, transcript, sender):
        both_phones(transport)
        assert hub.send_message("The laundry is done.")
        assert notices(transcript) == ["The laundry is done."]
        assert sender.pushes == [(IPHONE, "notice", "The laundry is done.", ""),
                                 (IPAD, "notice", "The laundry is done.", "")]

    def test_the_notification_is_the_messages_words_without_its_markup(self, hub, transport, sender):
        register(transport)
        hub.send_message("<b>Front door</b>: open since 14:05 &amp; nobody home")
        assert sender.pushes[0][2] == "Front door: open since 14:05 & nobody home"

    def test_with_no_phone_registered_it_is_an_item_alone(self, hub, transport, transcript, sender):
        assert hub.send_message("The laundry is done.")
        assert notices(transcript) == ["The laundry is done."]
        assert sender.pushes == []

    def test_a_silent_message_is_an_item_alone(self, hub, transport, transcript, sender):
        register(transport)
        assert hub.send_message("Restarted after the nightly update.", disable_notification=True)
        assert notices(transcript) == ["Restarted after the nightly update."]
        assert sender.pushes == []

    def test_while_another_chat_alerts_it_is_an_item_alone(self, hub, transport, telegram, transcript, sender, monkeypatch):
        set_notify_channel(monkeypatch, "telegram")
        register(transport)
        hub.send_message("The laundry is done.")
        assert telegram.sent_texts == ["The laundry is done."]
        assert notices(transcript) == ["The laundry is done."]
        assert sender.pushes == []

    def test_a_photo_it_sends_unasked_is_announced_by_its_caption(self, hub, transport, transcript, sender):
        register(transport)
        hub.send_photo(b"not a real picture", "Someone is at the front door")
        assert [item["kind"] for item in transcript.items()] == ["media"]
        assert sender.pushes == [(IPHONE, "notice", "Someone is at the front door", "")]

    def test_a_photo_without_a_caption_is_still_announced(self, hub, transport, sender):
        register(transport)
        hub.send_photo(b"not a real picture")
        ((phone, _, body, _),) = sender.pushes
        assert phone == IPHONE and body.strip()

    def test_a_photo_that_answers_a_request_from_the_app_is_not_announced(self, hub, transport, transcript, sender):
        register(transport)
        hub.send_photo(b"not a real picture", "The driveway now", source="App", conversation=CONVERSATION)
        assert [item["kind"] for item in transcript.items()] == ["media"]
        assert sender.pushes == []

    def test_a_notice_of_the_transports_own_is_not_a_notification(self, transport, hub, sender):
        """Only what the hub routes as Keep's own message alerts — not every line in the list."""
        register(transport)
        transport.send_text(CONVERSATION, "Sorry, the chat is not connected to the assistant.")
        assert sender.pushes == []


@pytest.mark.unit
class TestAQuestionOnlyTheOwnerAnswers:
    def test_an_approval_rings_every_phone_with_its_title(self, hub, transport, sender):
        both_phones(transport)
        transport.send_approval_prompt(CONVERSATION, APPROVAL_TOKEN, "Send the email to Alice?", "Subject: Saturday")
        assert sender.to("approval") == [IPHONE, IPAD]
        _, _, body, title = sender.pushes[0]
        assert body == "Send the email to Alice?"
        assert title and "Saturday" not in title + body

    def test_a_secret_prompts_notification_is_its_title_and_nothing_of_its_details(self, hub, transport, sender):
        register(transport)
        transport.send_secret_prompt(CONVERSATION, SECRET_TOKEN, "The door code",
                                     "The four digits from the letter of 3 March")
        ((phone, kind, body, title),) = sender.pushes
        assert (phone, kind, body) == (IPHONE, "secret", "The door code")
        assert "letter" not in title + body

    def test_a_question_asked_through_the_hub_rings_the_phone(self, hub, transport, transcript, sender):
        register(transport)
        answers = []
        asking = threading.Thread(target=lambda: answers.append(
            hub.request_approval("Send the email to Alice?", "Subject: Saturday")))
        asking.start()
        assert wait_until(lambda: sender.to("approval") == [IPHONE])
        (card,) = [item for item in transcript.items() if item["kind"] == "approval"]
        assert transport.answer_approval(card["token"], "approve")
        asking.join()
        assert answers == ["approve"]

    @pytest.mark.parametrize("ask", ["approval", "secret"])
    def test_while_another_chat_alerts_the_question_is_an_item_alone(self, hub, transport, telegram, transcript,
                                                                    sender, monkeypatch, ask):
        set_notify_channel(monkeypatch, "telegram")
        register(transport)
        prompt = transport.send_approval_prompt if ask == "approval" else transport.send_secret_prompt
        prompt(CONVERSATION, APPROVAL_TOKEN, "Send the email to Alice?", "")
        assert [item["kind"] for item in transcript.items()] == [ask]
        assert sender.pushes == []

    @pytest.mark.parametrize("ask", ["approval", "secret"])
    def test_during_a_turn_of_the_apps_own_it_rings_though_another_chat_alerts(self, hub, transport, telegram,
                                                                              sender, monkeypatch, ask):
        """The question of an app turn is asked nowhere else: nothing but the phone can say so."""
        set_notify_channel(monkeypatch, "telegram")
        both_phones(transport)
        transport.reply_started(CONVERSATION)
        prompt = transport.send_approval_prompt if ask == "approval" else transport.send_secret_prompt
        prompt(CONVERSATION, APPROVAL_TOKEN, "Send the email to Alice?", "")
        assert sender.to(ask) == [IPHONE, IPAD]

    def test_once_the_apps_turn_is_over_the_other_chat_alerts_again(self, hub, transport, telegram, sender, monkeypatch):
        set_notify_channel(monkeypatch, "telegram")
        register(transport)
        transport.reply_started(CONVERSATION)
        transport.reply_finished(CONVERSATION)
        transport.send_approval_prompt(CONVERSATION, APPROVAL_TOKEN, "Send the email to Alice?", "")
        assert sender.pushes == []

    def test_with_no_phone_registered_a_question_is_an_item_alone(self, hub, transport, transcript, sender):
        transport.send_approval_prompt(CONVERSATION, APPROVAL_TOKEN, "Send the email to Alice?", "")
        assert [item["kind"] for item in transcript.items()] == ["approval"]
        assert sender.pushes == []


@pytest.mark.unit
class TestTheEndOfAReply:
    def test_it_goes_to_the_phone_that_asked_and_to_no_other(self, hub, transport, sender):
        both_phones(transport)
        hub.assistant = answering("Yes, the <b>front door</b> is locked.")
        assert transport.submit("bubble-1", "Is the door locked?", "Charlotte", device=IPAD)
        assert wait_until(lambda: idle(hub, transport) and sender.pushes)
        assert sender.pushes == [(IPAD, "assistant", "Yes, the front door is locked.", "")]

    def test_it_goes_back_to_the_asking_phone_whatever_chat_alerts(self, hub, transport, telegram, sender, monkeypatch):
        set_notify_channel(monkeypatch, "telegram")
        both_phones(transport)
        hub.assistant = answering("Yes, it is locked.")
        transport.submit("bubble-1", "Is the door locked?", device=IPHONE)
        assert wait_until(lambda: idle(hub, transport) and sender.pushes)
        assert sender.to("assistant") == [IPHONE]
        assert telegram.calls == []

    def test_a_turn_the_server_started_ends_on_every_phone(self, hub, transport, sender):
        both_phones(transport)
        hub.assistant = answering("Good morning — the heating is back on.")
        assert hub.open_conversation("The server has just started…")
        assert wait_until(lambda: idle(hub, transport) and sender.pushes)
        assert sender.to("assistant") == [IPHONE, IPAD]

    def test_a_message_that_named_no_phone_ends_on_none(self, hub, transport, sender):
        """The Mac, or an app older than the notifications: the reply is read where it was asked."""
        both_phones(transport)
        hub.assistant = answering("Yes, it is locked.")
        transport.submit("bubble-1", "Is the door locked?")
        assert wait_until(lambda: hub.assistant.requests and idle(hub, transport))
        assert sender.pushes == []

    def test_a_phone_that_is_not_registered_asks_and_nobody_is_rung(self, hub, transport, sender):
        register(transport, IPAD, "Charlotte's iPad")
        hub.assistant = answering("Yes, it is locked.")
        transport.submit("bubble-1", "Is the door locked?", device=IPHONE)
        assert wait_until(lambda: hub.assistant.requests and idle(hub, transport))
        assert sender.pushes == []

    def test_the_turn_after_a_phones_is_not_that_phones_any_more(self, hub, transport, sender):
        both_phones(transport)
        hub.assistant = answering("Yes, it is locked.")
        transport.submit("bubble-1", "Is the door locked?", device=IPAD)
        assert wait_until(lambda: idle(hub, transport) and sender.pushes)
        hub.open_conversation("The server has just started…")
        assert wait_until(lambda: idle(hub, transport) and len(sender.pushes) == 3)
        assert sender.to("assistant") == [IPAD, IPHONE, IPAD]

    def test_a_reply_of_several_bubbles_is_announced_once_by_its_last(self, hub, transport, sender):
        register(transport)
        transport.reply_started(CONVERSATION)
        transport.begin_draft(CONVERSATION).finalize("Let me look at the car.")
        transport.begin_draft(CONVERSATION).finalize("It is at 78 %.")
        assert sender.pushes == [], "nothing is announced before the reply is over"
        transport.reply_finished(CONVERSATION)
        assert sender.pushes == [(IPHONE, "assistant", "It is at 78 %.", "")]

    def test_a_superseded_reply_winding_down_beside_its_successor_announces_nothing_of_its_own(self, hub, transport, sender):
        register(transport)
        transport.reply_started(CONVERSATION)
        first = transport.begin_draft(CONVERSATION)
        first.update("The car is")
        transport.reply_started(CONVERSATION)
        first.finalize("The car is")                 # what was written so far is kept
        transport.reply_finished(CONVERSATION)
        assert sender.pushes == []
        transport.begin_draft(CONVERSATION).finalize("The door is locked.")
        transport.reply_finished(CONVERSATION)
        assert sender.pushes == [(IPHONE, "assistant", "The door is locked.", "")]

    def test_what_a_superseded_reply_had_written_is_not_announced_as_its_successors_end(self, hub, transport, sender):
        register(transport)
        transport.reply_started(CONVERSATION)
        first = transport.begin_draft(CONVERSATION)
        transport.reply_started(CONVERSATION)
        first.finalize("The car is")
        transport.reply_finished(CONVERSATION)
        transport.reply_finished(CONVERSATION)       # the successor ended without a bubble
        assert sender.pushes == []

    def test_a_successor_that_ends_before_the_superseded_reply_has_wound_down_is_still_announced(self, hub, transport, sender):
        register(transport)
        transport.reply_started(CONVERSATION)
        first = transport.begin_draft(CONVERSATION)
        transport.reply_started(CONVERSATION)
        transport.begin_draft(CONVERSATION).finalize("The door is locked.")
        transport.reply_finished(CONVERSATION)
        first.finalize("The car is")
        transport.reply_finished(CONVERSATION)
        assert sender.pushes == [(IPHONE, "assistant", "The door is locked.", "")]

    def test_a_debug_command_does_not_pass_for_the_next_turns_asker(self, hub, transport, sender):
        both_phones(transport)
        hub.assistant = answering("Good morning — the heating is back on.")
        transport.submit("bubble-1", "//help", device=IPAD)
        assert wait_until(lambda: len(notices(transport.transcript)) == 1)
        assert hub.open_conversation("The server has just started…")
        assert wait_until(lambda: idle(hub, transport) and sender.pushes)
        assert sender.to("assistant") == [IPHONE, IPAD]

    def test_a_discarded_draft_announces_nothing(self, hub, transport, sender):
        register(transport)
        transport.reply_started(CONVERSATION)
        draft = transport.begin_draft(CONVERSATION)
        draft.update("The car is")
        draft.discard()
        transport.reply_finished(CONVERSATION)
        assert sender.pushes == []

    def test_what_one_reply_said_is_not_announced_again_by_the_next(self, hub, transport, sender):
        register(transport)
        transport.reply_started(CONVERSATION)
        transport.begin_draft(CONVERSATION).finalize("It is at 78 %.")
        transport.reply_finished(CONVERSATION)
        transport.reply_started(CONVERSATION)
        transport.reply_finished(CONVERSATION)       # a turn answered with nothing
        assert len(sender.pushes) == 1


@pytest.mark.unit
class TestTheAlertsMoved:
    def test_the_registration_that_takes_the_alerts_from_another_chat_says_so_once(self, hub, transport, telegram,
                                                                                  transcript, sender):
        register(transport)
        (notice,) = notices(transcript)
        assert "Telegram" in notice and "notify_channel" in notice and "telegram" in notice
        register(transport)                                 # the same phone at its next launch
        register(transport, IPAD, "Charlotte's iPad")       # and a second one
        assert len(notices(transcript)) == 1
        assert sender.pushes == [], "the notice itself rings no phone"

    def test_nothing_is_said_while_the_other_chat_keeps_alerting(self, hub, transport, telegram, transcript, monkeypatch):
        set_notify_channel(monkeypatch, "telegram")
        register(transport)
        assert notices(transcript) == []

    def test_nothing_is_said_where_no_other_chat_alerted(self, hub, transport, transcript):
        register(transport)
        assert notices(transcript) == []

    def test_a_chat_that_has_nobody_to_notify_loses_nothing(self, hub, transport, telegram, transcript):
        telegram.alerts_now = False
        register(transport)
        assert notices(transcript) == []

    def test_a_refused_registration_says_nothing_and_registers_nothing(self, hub, transport, telegram, transcript):
        with pytest.raises(DeviceRefused):
            transport.register_device(IPHONE, "not a token", "sandbox", KEY, "Marcus's iPhone")
        assert notices(transcript) == [] and transport.alerts() is False
