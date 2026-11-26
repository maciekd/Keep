"""AppTransport on a real ChatHub: what is typed in an app becomes an assistant request, and
what the chat layer renders becomes the items the apps draw."""
import base64
import io
import threading

import pytest
from PIL import Image

from assistant.AssistantEvents import Done, RoundEnd, Thinking, ThinkingDone, Token
from interfaces.app import AppTransport as app_transport
from interfaces.app import Transcript as transcript_module
from interfaces.app.AppTransport import CONVERSATION, MAX_PICTURES, AppDraft, AppTransport, PicturesRefused
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


def image_block(data: bytes) -> dict:
    """A picture as a command hands it to the model."""
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                        "data": base64.b64encode(data).decode()}}


def run_command(hub, detail: str, result=None, text: str = "Checking the car") -> dict:
    """A command echoed into the app and ended with *result*; its item as the apps get it."""
    handle = hub.send_command_notification(text, source="App", conversation=CONVERSATION)
    hub.complete_command_notification(handle, detail, result)
    (ref,) = [ref for _, _, ref in handle.refs]
    return hub.transports[0].transcript.item(ref)


def kept(transcript, address: str):
    """The file behind an address an item names: its bytes and content type, None when gone."""
    return transcript.media(address.rsplit("/", 1)[1])


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


class TestPicturesIn:
    def test_a_message_with_pictures_is_one_request_carrying_them_all(self, hub, transport, transcript):
        hub.assistant = FakeAssistant([Token("A lawn"), RoundEnd("A lawn"), Done("A lawn")])
        assert transport.submit("bubble-1", "what is this?", "Marcus", [png(40, 30), png(8, 8)])
        assert wait_until(lambda: hub.assistant.requests and idle(hub, transport))
        ((req, query),) = hub.assistant.requests
        assert (req.user, query) == ("Marcus", "what is this?")
        assert [image["media_type"] for image in hub.assistant.images[0]] == ["image/jpeg", "image/jpeg"]

    def test_pictures_without_a_word_are_a_message(self, hub, transport, transcript):
        hub.assistant = FakeAssistant([Token("A cat"), RoundEnd("A cat"), Done("A cat")])
        assert transport.submit("bubble-1", "  ", pictures=[png(4, 4)])
        assert wait_until(lambda: hub.assistant.requests and idle(hub, transport))
        assert len(hub.assistant.images[0]) == 1
        assert transcript.item("bubble-1")["text"] == ""

    def test_the_bubble_names_each_picture_by_address_and_size_a_contract_with_the_apps(self, transport, transcript):
        transport.submit("bubble-1", "look", pictures=[png(40, 30)])
        (image,) = transcript.item("bubble-1")["images"]
        assert set(image) == {"url", "width", "height"}
        assert (image["width"], image["height"]) == (40, 30)
        data, content_type = transcript.media(image["url"].rsplit("/", 1)[1])
        assert content_type == "image/jpeg" and Image.open(io.BytesIO(data)).size == (40, 30)

    def test_a_picture_larger_than_vision_reads_is_kept_at_the_size_it_is_sent_on_in(self, transport, transcript):
        transport.submit("bubble-1", "look", pictures=[png(4000, 1000)])
        (image,) = transcript.item("bubble-1")["images"]
        assert (image["width"], image["height"]) == (2048, 512)

    def test_a_picture_that_cannot_be_read_refuses_the_message_and_leaves_no_bubble(self, transport, transcript,
                                                                                    monkeypatch):
        monkeypatch.setattr(app_transport, "picture_as_jpeg", lambda data: None)
        with pytest.raises(PicturesRefused):
            transport.submit("bubble-1", "look", pictures=[b"not a picture"])
        assert transcript.items() == []

    def test_more_pictures_than_a_message_may_carry_are_refused(self, transport, transcript):
        with pytest.raises(PicturesRefused):
            transport.submit("bubble-1", "look", pictures=[png(2, 2)] * (MAX_PICTURES + 1))
        assert transcript.items() == []

    def test_a_message_sent_again_under_its_id_is_no_second_turn(self, hub, transport, transcript, monkeypatch):
        received = []
        monkeypatch.setattr(hub, "receive", lambda transport, message: received.append(message))
        assert transport.submit("bubble-1", "look", pictures=[png(4, 4)])
        assert transport.submit("bubble-1", "look", pictures=[png(4, 4)])
        assert len(received) == 1
        assert len(transcript.item("bubble-1")["images"]) == 1


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


class TestCommandResults:
    """What a command handed back is kept beside the log: the item names it, an app asks for it."""

    def test_the_text_a_command_handed_back_is_served_by_the_address_its_item_names(self, hub, transcript):
        command = run_command(hub, "done after 4s, received 26 chars", "Battery 78 %, plugged in.\n")
        data, content_type = kept(transcript, command["result"])
        assert data.decode("utf-8") == "Battery 78 %, plugged in."
        assert content_type.startswith("text/plain")
        assert command["images"] == [] and command["state"] == "done"

    def test_nothing_of_the_result_is_in_the_item_or_the_stream(self, hub, transcript):
        _, subscription = transcript.subscribe(0, "", lambda: None)
        run_command(hub, "done after 4s", "KEP 47L is parked at home, " + "locked " * 500)
        sent = "".join(str(line) for line in subscription.drain())
        assert "KEP 47L" not in sent and len(sent) < 1000

    def test_a_failed_command_serves_its_reason(self, hub, transcript):
        command = run_command(hub, "failed after 2s", "Command failed: the car did not answer")
        assert command["state"] == "failed"
        assert b"the car did not answer" in kept(transcript, command["result"])[0]

    def test_a_block_list_is_its_texts_and_its_pictures_with_their_sizes(self, hub, transcript):
        blocks = [{"type": "text", "text": "The driveway at 14:02."}, image_block(png(40, 30)),
                  image_block(png(8, 6)), {"type": "text", "text": "Nobody in sight."}]
        command = run_command(hub, "done after 3s", blocks, text="Looking at the driveway")
        text = kept(transcript, command["result"])[0].decode("utf-8")
        assert text.index("The driveway at 14:02.") < text.index("Nobody in sight.")
        assert [(image["width"], image["height"]) for image in command["images"]] == [(40, 30), (8, 6)]
        assert kept(transcript, command["images"][0]["url"])[0] == png(40, 30)

    def test_pictures_alone_are_a_result_without_a_text(self, hub, transcript):
        command = run_command(hub, "done after 3s", [image_block(png(40, 30))])
        assert command["result"] == "" and len(command["images"]) == 1

    def test_a_vault_value_is_its_placeholder_in_the_file(self, hub, transcript, monkeypatch):
        monkeypatch.setattr(app_transport, "redact_for_storage",
                            lambda text: text.replace("hunter2", "{{secret:gate-code}}"))
        command = run_command(hub, "done after 1s", "The gate code is hunter2.")
        text = kept(transcript, command["result"])[0].decode("utf-8")
        assert "hunter2" not in text and "{{secret:gate-code}}" in text

    def test_a_result_longer_than_an_app_is_shown_is_cut_and_says_its_whole_size(self, hub, transcript, monkeypatch):
        monkeypatch.setattr(app_transport, "RESULT_CHARS", 50)
        command = run_command(hub, "done after 1s", "å" * 12_345)
        text = kept(transcript, command["result"])[0].decode("utf-8")
        assert text.startswith("å" * 50) and text.count("å") == 50
        assert "12,345" in text

    def test_a_result_at_the_limit_is_kept_whole(self, hub, transcript, monkeypatch):
        monkeypatch.setattr(app_transport, "RESULT_CHARS", 50)
        command = run_command(hub, "done after 1s", "x" * 50)
        assert kept(transcript, command["result"])[0] == b"x" * 50

    @pytest.mark.parametrize("nothing", [None, "", "  \n", [], [{"type": "text", "text": " "}], 4711])
    def test_a_command_that_handed_nothing_back_has_no_result(self, hub, transcript, tmp_path, nothing):
        command = run_command(hub, "done", nothing)
        assert (command["result"], command["images"], command["state"]) == ("", [], "done")
        assert not list(tmp_path.glob("*/*"))       # and no file was written for it

    def test_a_result_that_cannot_be_kept_still_ends_the_row(self, hub, transcript, monkeypatch):
        def full(data, extension):
            raise OSError("no space left on device")
        monkeypatch.setattr(transcript, "add_media", full)
        command = run_command(hub, "done after 4s, received 9 chars", "Battery 78 %")
        assert (command["state"], command["detail"]) == ("done", "done after 4s, received 9 chars")
        assert (command["result"], command["images"]) == ("", [])

    def test_a_picture_without_bytes_is_left_out_and_the_text_is_kept(self, hub, transcript):
        broken = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "%%%"}}
        command = run_command(hub, "done after 4s", [{"type": "text", "text": "A look."}, broken])
        assert command["state"] == "done"
        assert command["images"] == []
        assert kept(transcript, command["result"])[0] == b"A look."

    def test_a_dropped_command_takes_its_result_and_its_pictures_along(self, hub, transport, transcript, monkeypatch):
        command = run_command(hub, "done after 3s", [{"type": "text", "text": "The driveway."}, image_block(png(40, 30))])
        text, picture = command["result"], command["images"][0]["url"]
        assert kept(transcript, text) and kept(transcript, picture)
        monkeypatch.setattr(transcript_module, "CAP_ITEMS", 1)
        transport.send_text(CONVERSATION, "A notice")
        assert transcript.item(command["id"]) is None
        assert kept(transcript, text) is None and kept(transcript, picture) is None

    def test_the_result_fields_of_a_command_item_a_contract_with_the_apps(self, hub, transcript):
        command = run_command(hub, "done after 3s", [{"type": "text", "text": "The driveway."}, image_block(png(40, 30))])
        assert set(command) == {"id", "kind", "at", "order", "text", "state", "detail", "result", "images"}
        assert command["result"].startswith("/app/media/") and command["result"].endswith(".txt")
        (image,) = command["images"]
        assert set(image) == {"url", "width", "height"} and image["url"].startswith("/app/media/")

    def test_a_running_command_names_no_result_yet(self, hub, transcript):
        hub.send_command_notification("Checking the car", source="App", conversation=CONVERSATION)
        (command,) = transcript.items()
        assert command["state"] == "running" and not command.get("result") and not command.get("images")


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

    def test_a_finished_command_keeps_its_result_while_the_unfinished_are_closed(self, tmp_path):
        before = AppTransport(Transcript(tmp_path, autosave=False))
        done = before.send_command_line(CONVERSATION, "Looking at the driveway")
        before.finish_command_line(CONVERSATION, done, "Looking at the driveway", "done after 3s",
                                   [{"type": "text", "text": "The driveway."}, image_block(png(40, 30))])
        before.send_command_line(CONVERSATION, "Checking the car")
        before.transcript.flush()
        finished = before.transcript.item(done)

        after = AppTransport(Transcript(tmp_path, autosave=False))
        after.start(ChatHub())
        command, interrupted = after.transcript.items()
        assert interrupted["state"] == "failed"
        assert (command["state"], command["result"], command["images"]) == \
            ("done", finished["result"], finished["images"])
        assert kept(after.transcript, command["result"])[0] == b"The driveway."


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


    def test_a_stopped_server_leaves_a_log_the_next_run_continues_for_a_phones_cursor(self, transport, transcript, tmp_path):
        """Only a file the transport closed is the whole log: a phone then gets what is new, not the list again."""
        transport.send_text(CONVERSATION, "The door is locked")
        seen, subscription = transcript.subscribe(0, "", lambda: None)
        transcript.unsubscribe(subscription)
        cursor = max(line["seq"] for line in seen if "seq" in line)
        transport.stop()
        again = Transcript(tmp_path, autosave=False)
        again.upsert("bubble-2", "user", text="and the garage?", images=[])
        lines, _ = again.subscribe(cursor, transcript.log, lambda: None)
        assert {"reset": True} not in lines
        assert [line["item"]["id"] for line in lines if "item" in line] == ["bubble-2"]


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
