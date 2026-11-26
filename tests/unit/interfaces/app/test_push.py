"""The box a phone opens and the sender that hands it to the relay — over a faked relay: no
unit test reaches push.keep-ai.dev (tests/unit/test_no_server_relaunch.py)."""
import base64
import json

import pytest
import requests
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

from interfaces.app import Push
from interfaces.app.Devices import Devices
from interfaces.app.Push import Sender, cut, plaintext, seal
from tests.unit.waiting import wait_until

KEY_BYTES = bytes(range(32))
KEY = base64.b64encode(KEY_BYTES).decode()
IPHONE, TOKEN = "marcus-iphone-0001", "a1" * 32


def opened(box: str, key: bytes = KEY_BYTES) -> dict:
    """The box as the phone's extension opens it: a 12-byte nonce, then the ciphertext with
    its 16-byte tag — CryptoKit's combined form."""
    data = base64.b64decode(box, validate=True)
    return json.loads(ChaCha20Poly1305(key).decrypt(data[:12], data[12:], None))


def boxed(title: str, body: str, badge: int = 1, kind: str = "notice") -> str:
    return seal(KEY_BYTES, plaintext(title, body, badge, kind))


class Relay:
    """Stands in for `Push.post`: records what it was handed and answers as scripted — a
    (status, reason) of Apple's, or an exception for a relay that was not reached."""

    def __init__(self, *answers):
        self.answers = list(answers) or [(200, "")]
        self.posted = []

    def __call__(self, body):
        self.posted.append(body)
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(answer, Exception):
            raise answer
        return answer


@pytest.fixture
def devices():
    devices = Devices()
    devices.register(IPHONE, TOKEN, "sandbox", KEY, "Marcus's iPhone")
    return devices


@pytest.fixture
def relay(monkeypatch):
    relay = Relay()
    monkeypatch.setattr(Push, "post", relay)
    return relay


@pytest.mark.unit
class TestTheBox:
    def test_opened_with_the_phones_key_it_gives_back_what_was_put_in(self):
        fields = opened(boxed("Approval needed", "Send the email to Alice?", badge=3, kind="approval"))
        assert fields == {"title": "Approval needed", "body": "Send the email to Alice?", "badge": 3, "kind": "approval"}

    def test_the_fields_of_the_box_are_a_contract_with_the_app(self):
        assert set(opened(boxed("", "The laundry is done."))) == {"title", "body", "badge", "kind"}

    def test_another_key_opens_nothing(self):
        with pytest.raises(InvalidTag):
            opened(boxed("", "The laundry is done."), key=bytes(range(32, 64)))

    def test_the_same_text_is_never_the_same_box_twice(self):
        assert boxed("", "The laundry is done.") != boxed("", "The laundry is done.")

    def test_a_text_that_fits_is_not_touched(self):
        body = "Tvätten är klar — häng den innan 18.\n\"Glöm inte\" strumporna 🙂"
        assert opened(boxed("", body))["body"] == body

    @pytest.mark.parametrize("character", ["å", "猫", "🙂"], ids=["two bytes", "three bytes", "four bytes"])
    def test_a_long_text_of_multi_byte_characters_is_cut_at_a_character_and_fits_the_relay(self, character):
        whole = character * 5000
        box = boxed("", whole)
        assert len(box) <= Push.MAX_BOX_CHARS
        body = opened(box)["body"]      # the decrypt alone says it is whole characters: it is read as UTF-8
        assert body.endswith(Push.CUT) and whole.startswith(body[:-len(Push.CUT)])
        assert len(body) > 400, "the cut left next to nothing of the text"

    @pytest.mark.parametrize("piece", ['"', "\n", "\\", "\x01", 'a"\nå'],
                             ids=["quotes", "line breaks", "backslashes", "control characters", "a mix"])
    def test_a_long_text_json_writes_longer_than_it_is_still_fits_the_relay(self, piece):
        whole = piece * 4000
        box = boxed("", whole)
        assert len(box) <= Push.MAX_BOX_CHARS
        body = opened(box)["body"]
        assert body.endswith(Push.CUT) and whole.startswith(body[:-len(Push.CUT)])

    def test_a_title_of_any_length_leaves_room_for_the_text(self):
        box = boxed("Approval needed " * 200, "Send the email to Alice?")
        assert len(box) <= Push.MAX_BOX_CHARS
        assert opened(box)["body"] == "Send the email to Alice?"

    def test_a_cut_never_ends_inside_a_character(self):
        for limit in range(len(Push.CUT.encode()), 40):
            kept = cut("blåbärssoppa på älven 🙂🙂🙂", limit)
            assert len(kept.encode("utf-8")) <= limit


@pytest.mark.unit
class TestDelivering:
    def test_a_notification_goes_to_the_relay_as_the_token_its_service_and_a_box_the_phone_opens(self, devices, relay):
        assert Sender(devices).deliver(devices.get(IPHONE), "notice", "", "The laundry is done.") == "sent"
        (posted,) = relay.posted
        assert (posted["token"], posted["environment"]) == (TOKEN, "sandbox")
        assert opened(posted["box"]) == {"title": "", "body": "The laundry is done.", "badge": 1, "kind": "notice"}

    def test_the_relay_is_handed_no_readable_text_a_contract_with_the_relay(self, devices, relay):
        """tools/push_relay/worker.js takes these three fields and nothing else."""
        Sender(devices).deliver(devices.get(IPHONE), "notice", "Keep", "The laundry is done.")
        (posted,) = relay.posted
        assert set(posted) == {"token", "environment", "box"}
        assert "laundry" not in json.dumps(posted)

    def test_the_badge_is_the_count_of_what_the_phone_has_not_seen(self, devices, relay):
        sender, phone = Sender(devices), devices.get(IPHONE)
        sender.deliver(phone, "notice", "", "one")
        sender.deliver(phone, "notice", "", "two")
        assert [opened(posted["box"])["badge"] for posted in relay.posted] == [1, 2]

    @pytest.mark.parametrize("answer", [(410, "Unregistered"), (410, ""), (400, "BadDeviceToken"),
                                        (400, "DeviceTokenNotForTopic")])
    def test_a_token_apple_no_longer_knows_takes_its_phone_out_of_the_registry(self, devices, monkeypatch, answer):
        monkeypatch.setattr(Push, "post", Relay(answer))
        assert Sender(devices).deliver(devices.get(IPHONE), "notice", "", "The laundry is done.") == "gone"
        assert devices.all() == []

    @pytest.mark.parametrize("answer", [(403, "InvalidProviderToken"), (429, "TooManyRequests"),
                                        (400, "PayloadTooLarge"), (503, "ServiceUnavailable")])
    def test_any_other_refusal_leaves_the_phone_registered(self, devices, monkeypatch, answer):
        monkeypatch.setattr(Push, "post", Relay(answer))
        assert Sender(devices).deliver(devices.get(IPHONE), "notice", "", "The laundry is done.") == "refused"
        assert [device.id for device in devices.all()] == [IPHONE]

    def test_a_phone_forgotten_while_its_notification_waited_is_sent_nothing(self, devices, relay):
        phone = devices.get(IPHONE)
        devices.forget(IPHONE)
        assert Sender(devices).deliver(phone, "notice", "", "The laundry is done.") == "gone"
        assert relay.posted == []


class Outcomes:
    """Stands in for the telemetry: the (kind, outcome) of each notification that left the queue."""

    def __init__(self):
        self.seen = []

    def get_instance(self):
        return self

    def app_push(self, kind, outcome):
        self.seen.append((kind, outcome))


@pytest.mark.unit
class TestTheQueue:
    @pytest.fixture
    def outcomes(self, monkeypatch):
        outcomes = Outcomes()
        monkeypatch.setattr(Push, "Telemetry", outcomes)
        return outcomes

    def test_a_pushed_notification_goes_out_on_the_senders_own_thread(self, devices, relay, outcomes):
        sender = Sender(devices)
        sender.push(devices.get(IPHONE), "approval", "Send the email to Alice?", "Approval needed")
        assert wait_until(lambda: outcomes.seen == [("approval", "sent")])
        assert opened(relay.posted[0]["box"])["title"] == "Approval needed"
        sender.stop()

    def test_a_relay_that_is_not_reached_fails_one_notification_and_not_the_next(self, devices, monkeypatch, outcomes):
        relay = Relay(requests.ConnectionError("no route to the relay"), (200, ""))
        monkeypatch.setattr(Push, "post", relay)
        sender, phone = Sender(devices), devices.get(IPHONE)
        sender.push(phone, "notice", "one")
        sender.push(phone, "notice", "two")
        assert wait_until(lambda: outcomes.seen == [("notice", "failed"), ("notice", "sent")])
        assert [device.id for device in devices.all()] == [IPHONE], "a relay out of reach says nothing about the phone"
        sender.stop()

    def test_a_stopped_sender_ends_its_thread_and_a_later_push_starts_another(self, devices, relay, outcomes):
        sender, phone = Sender(devices), devices.get(IPHONE)
        sender.push(phone, "notice", "one")
        assert wait_until(lambda: len(outcomes.seen) == 1)
        first = sender._thread
        sender.stop()
        assert wait_until(lambda: not first.is_alive())
        sender.push(phone, "notice", "two")
        assert wait_until(lambda: len(outcomes.seen) == 2)
        sender.stop()

    def test_a_queue_that_is_full_drops_the_notification_instead_of_waiting(self, devices, monkeypatch, outcomes):
        import threading
        held, entered = threading.Event(), threading.Event()

        def stuck(body):
            entered.set()
            held.wait()
            return 200, ""

        monkeypatch.setattr(Push, "post", stuck)
        monkeypatch.setattr(Push, "QUEUE_SIZE", 2)
        sender, phone = Sender(devices), devices.get(IPHONE)
        sender.push(phone, "notice", "on its way")
        assert wait_until(entered.is_set)
        for number in range(5):
            sender.push(phone, "notice", f"waiting {number}")     # returns at once, whatever the queue holds
        held.set()
        assert wait_until(lambda: len(outcomes.seen) == 3)
        sender.stop()


class Answer:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body

    def json(self):
        if self._body is None:
            raise ValueError("no JSON")
        return self._body


@pytest.mark.unit
class TestTheRelaysAnswer:
    """`Push.post` over a faked `requests.post`: the relay passes Apple's own answer back."""

    @pytest.fixture
    def answer_with(self, monkeypatch):
        sent = []

        def answer(status_code, body):
            def post(url, **kwargs):
                sent.append((url, kwargs))
                return Answer(status_code, body)
            monkeypatch.setattr(requests, "post", post)
            return sent
        return answer

    def test_apples_status_and_reason_are_handed_back(self, answer_with):
        answer_with(200, {"status": 410, "reason": "Unregistered"})
        assert Push.post({"token": TOKEN, "environment": "sandbox", "box": "AAAA"}) == (410, "Unregistered")

    def test_a_notification_apple_took_has_no_reason(self, answer_with):
        answer_with(200, {"status": 200, "reason": ""})
        assert Push.post({"token": TOKEN, "environment": "sandbox", "box": "AAAA"}) == (200, "")

    def test_the_request_is_the_three_fields_as_json_and_does_not_wait_for_ever(self, answer_with):
        sent = answer_with(200, {"status": 200, "reason": ""})
        body = {"token": TOKEN, "environment": "sandbox", "box": "AAAA"}
        Push.post(body)
        ((url, kwargs),) = sent
        assert url.startswith("https://") and kwargs["json"] == body
        assert kwargs["timeout"] and kwargs["timeout"] > 0

    @pytest.mark.parametrize("status_code, body", [(429, {"error": "too many pushes"}), (400, {"error": "box must be base64"}),
                                                   (502, None)], ids=["limited", "refused", "an answer that is no JSON"])
    def test_a_refusal_of_the_relays_own_is_never_read_as_apples(self, answer_with, status_code, body):
        """Apple's 410 drops a phone: the relay's own trouble must not."""
        answer_with(status_code, body)
        with pytest.raises(RuntimeError, match=str(status_code)):
            Push.post({"token": TOKEN, "environment": "sandbox", "box": "AAAA"})
