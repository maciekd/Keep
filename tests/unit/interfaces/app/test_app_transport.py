"""AppTransport on a real ChatHub: what is typed in an app becomes an assistant request, and
what the chat layer renders becomes the items the apps draw."""
import io
import threading

import pytest
from PIL import Image

from assistant.AssistantEvents import Done, RoundEnd, Thinking, ThinkingDone, Token
from interfaces.app.AppTransport import CONVERSATION, AppDraft, AppTransport
from interfaces.app.Transcript import Transcript
from interfaces.chat.ChatHub import ChatHub
from interfaces.chat.ChatTypes import PHASE_COMMANDS, PHASE_THINKING, Progress
from tests.unit.interfaces.chat.fakes import FakeAssistant, FakeTransport, wait_until
from tests.unit.waiting import FAILURE_BOUND


@pytest.fixture(autouse=True)
def reset_singleton():
    ChatHub.reset_instance()
    yield
    ChatHub.reset_instance()


@pytest.fixture
def transcript(tmp_path):
    return Transcript(tmp_path, autosave=False)


@pytest.fixture
def transport(transcript):
    return AppTransport(transcript)


@pytest.fixture
def hub(transport):
    hub = ChatHub()
    hub.register(transport)
    transport.start(hub)
    yield hub
    hub.stop()


def png(width: int, height: int) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (width, height), (40, 200, 40)).save(out, format="PNG")
    return out.getvalue()


def kinds(transcript) -> list[str]:
    return [item["kind"] for item in transcript.items()]


def idle(hub, transport) -> bool:
    return not hub.session_for(transport, CONVERSATION).busy


class TestATurn:
    def test_a_typed_message_becomes_an_app_request_and_a_streamed_reply(self, hub, transport, transcript):
        hub.assistant = FakeAssistant([Token("Hel"), Token("lo"), RoundEnd("Hello"), Done("Hello")])
        assert transport.submit("bubble-1", "hi there", "Marcus")
        assert wait_until(lambda: idle(hub, transport))
        req, query = hub.assistant.requests[0]
        assert (req.source, req.user, query) == ("App", "Marcus", "hi there")
        user, reply = transcript.items()
        assert (user["id"], user["kind"], user["text"]) == ("bubble-1", "user", "hi there")
        assert (reply["kind"], reply["html"], reply["final"]) == ("assistant", "Hello", True)

    def test_a_turn_with_thinking_two_commands_and_a_reply(self, hub, transport, transcript):
        def run_commands():
            first = hub.send_command_notification("Loading the cars toolkit", source="App", conversation=CONVERSATION)
            second = hub.send_command_notification("Checking the car", source="App", conversation=CONVERSATION)
            hub.complete_command_notification(first, "done after 0.2s")
            hub.complete_command_notification(second, "failed after 4s")
            return True

        def thinking_folded():
            # The commands run on the stream's thread: they wait for the session to have drawn the round.
            return any(item["kind"] == "thinking" and item["done"] for item in transcript.items())

        hub.assistant = FakeAssistant([Thinking("let me ", 10), Thinking("see", 20), ThinkingDone(0.2),
                                       RoundEnd(""), thinking_folded, run_commands,
                                       Token("78 %."), RoundEnd("78 %."), Done("78 %.")])
        transport.submit("bubble-1", "is the car charged?")
        assert wait_until(lambda: idle(hub, transport))
        assert kinds(transcript) == ["user", "thinking", "command", "command", "assistant"]
        _, thinking, first, second, reply = transcript.items()
        assert (thinking["text"], thinking["done"]) == ("let me see", True)
        assert (first["text"], first["state"], first["detail"]) == ("Loading the cars toolkit", "done", "done after 0.2s")
        assert second["state"] == "failed"
        assert (reply["html"], reply["final"]) == ("78 %.", True)

    def test_an_empty_message_is_no_turn(self, hub, transport, transcript):
        hub.assistant = FakeAssistant()
        assert not transport.submit("bubble-1", "   ")
        assert transcript.items() == [] and hub.assistant.requests == []

    def test_an_empty_reply_leaves_no_bubble_and_no_working_line(self, hub, transport, transcript):
        hub.assistant = FakeAssistant([RoundEnd(""), Done("")])
        transport.submit("bubble-1", "silence")
        assert wait_until(lambda: idle(hub, transport))
        assert kinds(transcript) == ["user"]
        lines, _ = transcript.subscribe(0, "", lambda: None)
        assert not [line for line in lines if "working" in line]

    def test_there_is_a_working_line_for_as_long_as_the_turn_is_on(self, hub, transport, transcript):
        gate = threading.Event()
        hub.assistant = FakeAssistant([Token("Well"), gate, RoundEnd("Well"), Done("Well")])
        _, subscription = transcript.subscribe(0, "", lambda: None)
        transport.submit("bubble-1", "go on")
        assert wait_until(lambda: "assistant" in kinds(transcript))
        assert [line for line in subscription.drain() if "working" in line][-1]["working"] is not None
        gate.set()
        assert wait_until(lambda: idle(hub, transport))
        assert [line for line in subscription.drain() if "working" in line][-1]["working"] is None

    def test_stop_ends_the_reply_and_takes_its_bubble_away(self, hub, transport, transcript):
        gate = threading.Event()
        hub.assistant = FakeAssistant([Token("Well"), gate, Token(" then"), RoundEnd("Well then"), Done("Well then")])
        transport.submit("bubble-1", "go on")
        assert wait_until(lambda: "assistant" in kinds(transcript))
        transport.interrupt()
        assert wait_until(lambda: idle(hub, transport))
        assert kinds(transcript) == ["user"]
        assert hub.assistant.stream_obj.killed


class TestDraft:
    def test_the_working_line_carries_the_progress_facts(self, transcript):
        _, subscription = transcript.subscribe(0, "", lambda: None)
        AppDraft(transcript).show_progress(Progress(phase=PHASE_COMMANDS, elapsed=12.34, commands=2))
        (line,) = subscription.drain()
        assert (line["working"]["phase"], line["working"]["commands"], line["working"]["elapsed"]) == ("commands", 2, 12.3)

    def test_thinking_is_live_until_the_reply_starts_and_then_folds(self, transcript):
        draft = AppDraft(transcript)
        draft.show_progress(Progress(phase=PHASE_THINKING, thinking_text="hm"))
        (thinking,) = transcript.items()
        assert (thinking["kind"], thinking["done"]) == ("thinking", False)
        draft.update("So")
        thinking, reply = transcript.items()
        assert thinking["done"] and not reply["final"]

    def test_thinking_stays_when_its_round_said_nothing(self, transcript):
        draft = AppDraft(transcript)
        draft.show_progress(Progress(phase=PHASE_THINKING, thinking_text="hm"))
        draft.discard()
        (thinking,) = transcript.items()
        assert thinking["done"]

    def test_a_repeated_progress_tick_changes_no_item(self, transcript):
        draft = AppDraft(transcript)
        draft.show_progress(Progress(phase=PHASE_THINKING, thinking_text="hm", tick=1))
        _, subscription = transcript.subscribe(0, "", lambda: None)
        draft.show_progress(Progress(phase=PHASE_THINKING, thinking_text="hm", tick=2))
        assert not [line for line in subscription.drain() if "item" in line]

    def test_nothing_is_shown_after_the_bubble_is_closed(self, transcript):
        draft = AppDraft(transcript)
        draft.finalize("Done.")
        assert not draft.update("more")
        draft.show_progress(Progress(phase=PHASE_THINKING, thinking_text="late"))
        assert kinds(transcript) == ["assistant"]


class TestWhatTheServerStarts:
    def test_a_message_to_the_owner_is_a_notice(self, hub, transcript):
        hub.send_message("The laundry is done.")
        (notice,) = transcript.items()
        assert (notice["kind"], notice["html"]) == ("notice", "The laundry is done.")

    def test_a_command_of_another_chat_is_not_echoed_into_the_app(self, hub, transcript):
        other = FakeTransport(name="Other", source="Telegram")
        hub.register(other)
        hub.send_command_notification("Checking", source="Telegram", conversation=1)
        assert transcript.items() == []

    def test_a_photo_is_kept_and_named_by_its_address_and_size(self, hub, transport, transcript):
        hub.send_photo(png(40, 30), "The driveway")
        (photo,) = transcript.items()
        assert (photo["media"], photo["caption"], photo["width"], photo["height"]) == ("photo", "The driveway", 40, 30)
        data, content_type = transcript.media(photo["url"].rsplit("/", 1)[1])
        assert data == png(40, 30) and content_type == "image/png"

    def test_the_photos_of_an_album_share_its_name(self, transport, transcript):
        transport.send_photo_album(CONVERSATION, [(png(4, 4), "one"), (png(4, 4), "two")])
        transport.send_photo(CONVERSATION, png(4, 4))
        first, second, alone = transcript.items()
        assert first["album"] and first["album"] == second["album"]
        assert alone["album"] == ""

    def test_a_video_has_its_poster(self, transport, transcript):
        transport.send_video(CONVERSATION, b"mp4-bytes", "The cat", thumbnail=png(16, 9))
        (video,) = transcript.items()
        assert (video["media"], video["width"], video["height"]) == ("video", 16, 9)
        assert transcript.media(video["poster"].rsplit("/", 1)[1])[0] == png(16, 9)
        assert transcript.media(video["url"].rsplit("/", 1)[1]) == (b"mp4-bytes", "video/mp4")

    def test_the_app_is_attended_while_a_phone_listens(self, transport, transcript):
        assert not transport.attended()
        _, subscription = transcript.subscribe(0, "", lambda: None)
        assert transport.attended()
        transcript.unsubscribe(subscription)
        assert not transport.attended()


class TestApprovals:
    def ask(self, hub, result):
        thread = threading.Thread(target=lambda: result.update(
            d=hub.request_approval("Send the email?", "To Alice", timeout=FAILURE_BOUND)), daemon=True)
        thread.start()
        return thread

    def test_an_approval_answered_in_the_app_resolves_and_folds(self, hub, transport, transcript):
        result = {}
        thread = self.ask(hub, result)
        assert wait_until(lambda: kinds(transcript) == ["approval"])
        (card,) = transcript.items()
        assert (card["title"], card["details"], card["outcome"]) == ("Send the email?", "To Alice", "")
        assert transport.answer_approval(card["token"], "approve")
        thread.join(timeout=FAILURE_BOUND)
        assert result["d"] == "approve"
        assert transcript.items()[0]["outcome"]

    def test_an_answer_to_a_question_no_longer_open_closes_the_card(self, hub, transport, transcript):
        ref = transport.send_approval_prompt(CONVERSATION, "stale-token", "Old", "")
        assert not transport.answer_approval("stale-token", "approve")
        assert transcript.item(ref)["outcome"]


class TestSecrets:
    def test_the_typed_value_reaches_the_requester_and_no_item(self, hub, transport, transcript):
        seen, result = [], {}

        def accept(value):
            seen.append(value)
            return "Stored"

        thread = threading.Thread(target=lambda: result.update(
            d=hub.request_secret("Card number", "", accept, timeout=FAILURE_BOUND)), daemon=True)
        thread.start()
        assert wait_until(lambda: kinds(transcript) == ["secret"])
        token = transcript.items()[0]["token"]
        assert transport.answer_secret(token, "4111 1111 1111 1111")
        thread.join(timeout=FAILURE_BOUND)
        assert result["d"] == "accepted" and seen == ["4111 1111 1111 1111"]
        assert kinds(transcript) == ["secret"]
        assert transcript.items()[0]["outcome"]
        transcript.flush()
        assert "4111" not in "".join(path.read_text() for path in transcript._path.parent.glob("*.json"))
        assert hub.transcript == []

    def test_a_value_typed_into_an_expired_prompt_is_not_handed_to_the_one_open_now(self, hub, transport, transcript):
        seen, result = [], {}
        stale = transport.send_secret_prompt(CONVERSATION, "expired-token", "Router password", "")
        thread = threading.Thread(target=lambda: result.update(
            d=hub.request_secret("Card number", "", lambda v: seen.append(v) or "Stored", timeout=FAILURE_BOUND)), daemon=True)
        thread.start()
        assert wait_until(lambda: kinds(transcript) == ["secret", "secret"])
        assert transport.answer_secret("expired-token", "hunter2") is False
        assert seen == [] and transcript.item(stale)["outcome"]
        open_now = transcript.items()[1]
        assert not open_now["outcome"]
        assert transport.answer_secret(open_now["token"], None)
        thread.join(timeout=FAILURE_BOUND)
        assert result["d"] == "cancel"

    def test_a_dismissed_prompt_cancels_the_request(self, hub, transport, transcript):
        result = {}
        thread = threading.Thread(target=lambda: result.update(
            d=hub.request_secret("Password", "", lambda v: "ok", timeout=FAILURE_BOUND)), daemon=True)
        thread.start()
        assert wait_until(lambda: kinds(transcript) == ["secret"])
        assert transport.answer_secret(transcript.items()[0]["token"], None)
        thread.join(timeout=FAILURE_BOUND)
        assert result["d"] == "cancel"


class TestAfterARestart:
    def test_what_the_last_run_left_unfinished_is_closed(self, tmp_path):
        before = AppTransport(Transcript(tmp_path, autosave=False))
        draft = before.begin_draft(CONVERSATION)
        draft.show_progress(Progress(phase=PHASE_THINKING, thinking_text="hm"))
        draft.update("Half a sen")
        before.send_command_line(CONVERSATION, "Checking the car")
        before.send_approval_prompt(CONVERSATION, "t1", "Send?", "")
        before.send_text(CONVERSATION, "A notice")
        before.transcript.flush()

        after = AppTransport(Transcript(tmp_path, autosave=False))
        after.start(ChatHub())
        thinking, reply, command, approval, notice = after.transcript.items()
        assert thinking["done"] and reply["final"] and reply["html"] == "Half a sen"
        assert command["state"] == "failed"
        assert approval["outcome"]
        assert notice["html"] == "A notice"


class TestWithoutAHub:
    """Before the interface has started the transport, or after it stopped it."""

    def test_a_message_keeps_its_bubble_and_is_answered_that_nothing_is_connected(self, transport, transcript):
        assert transport.submit("bubble-1", "anyone there?")
        user, notice = transcript.items()
        assert (user["id"], user["kind"], user["text"]) == ("bubble-1", "user", "anyone there?")
        assert notice["kind"] == "notice" and notice["html"].strip()

    def test_an_answer_is_reported_as_not_taken_and_stop_is_harmless(self, transport, transcript):
        ref = transport.send_approval_prompt(CONVERSATION, "t1", "Send?", "")
        assert transport.answer_approval("t1", "approve") is False
        assert transport.answer_secret("t2", "hunter2") is False
        transport.interrupt()
        assert transcript.item(ref)["outcome"] == ""

    def test_stopping_ends_the_working_line_and_writes_the_log(self, hub, transport, transcript, tmp_path):
        woken = []
        _, subscription = transcript.subscribe(0, "", lambda: woken.append(1))
        transport.send_text(CONVERSATION, "The door is locked")
        transport.reply_started(CONVERSATION)
        subscription.drain()
        transport.stop()
        assert subscription.drain() == [{"working": None}]
        assert [item["html"] for item in Transcript(tmp_path, autosave=False).items()] == ["The door is locked"]
        assert transport.submit("bubble-2", "still there?")     # no hub any more: answered, not raised
        assert kinds(transcript)[-2:] == ["user", "notice"]


class TestWhatCannotBeShownAsAsked:
    def test_a_picture_that_cannot_be_read_is_sent_without_a_size(self, transport, transcript):
        assert transport.send_photo(CONVERSATION, b"not a picture at all", "From the porch")
        (item,) = transcript.items()
        assert (item["width"], item["height"], item["caption"]) == (0, 0, "From the porch")
        data, _ = transcript.media(item["url"].rsplit("/", 1)[-1])
        assert data == b"not a picture at all"

    def test_a_video_without_a_poster_frame_names_none(self, transport, transcript):
        transport.send_video(CONVERSATION, b"mp4-bytes", "Driveway")
        (item,) = transcript.items()
        assert (item["media"], item["poster"], item["width"], item["height"]) == ("video", "", 0, 0)
        assert transcript.media(item["url"].rsplit("/", 1)[-1]) == (b"mp4-bytes", "video/mp4")

    def test_text_of_spaces_alone_is_no_notice(self, transport, transcript):
        assert transport.send_text(CONVERSATION, " \n ") is False
        assert transcript.items() == []


class TestPrompts:
    def test_a_question_asked_again_under_its_token_is_one_card(self, transport, transcript):
        first = transport.send_approval_prompt(CONVERSATION, "t1", "Send the email?", "To Alice")
        again = transport.send_approval_prompt(CONVERSATION, "t1", "Send the email?", "To Alice and Bob")
        assert first == again
        (card,) = transcript.items()
        assert card["details"] == "To Alice and Bob"

    def test_an_approval_and_a_secret_under_one_token_are_two_cards(self, transport, transcript):
        transport.send_approval_prompt(CONVERSATION, "t1", "Send?", "")
        transport.send_secret_prompt(CONVERSATION, "t1", "Password", "")
        assert kinds(transcript) == ["approval", "secret"]

    def test_a_card_closed_without_a_word_still_reads_as_closed(self, transport, transcript):
        """The apps fold a card on a non-empty outcome: an empty one would leave its buttons up."""
        ref = transport.send_secret_prompt(CONVERSATION, "t1", "Password", "For the router")
        transport.close_secret_prompt(CONVERSATION, ref, "")
        card = transcript.item(ref)
        assert card["outcome"] and (card["token"], card["title"], card["details"]) == ("t1", "Password", "For the router")

    def test_closing_a_card_the_log_no_longer_holds_brings_nothing_back(self, transport, transcript):
        ref = transport.send_approval_prompt(CONVERSATION, "t1", "Send?", "")
        transcript.remove(ref)
        transport.close_approval_prompt(CONVERSATION, ref, "Approved")
        assert transcript.items() == []
