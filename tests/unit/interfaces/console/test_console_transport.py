"""ConsoleTransport on a real ChatHub: typed text becomes an assistant request, the reply
state machine renders into the ChatPane, and the Notifier/approval paths reach the console."""
import io
import threading
from unittest.mock import MagicMock

import pytest
from PIL import Image

from assistant.AssistantEvents import Done, Notice, RoundEnd, Thinking, ThinkingDone, Token
from interfaces.chat.ChatHub import ChatHub
from interfaces.chat.ChatSession import ChatSession
from interfaces.chat.ChatTypes import PHASE_WAITING, Progress
from interfaces.console.ConsoleTransport import CONVERSATION, ConsoleDraft, ConsoleTransport
from interfaces.console.widgets.ChatPane import (KIND_APPROVAL, KIND_ASSISTANT, KIND_MEDIA, KIND_NOTICE,
                                                  KIND_THINKING, KIND_USER, ChatPane)
from tests.unit.interfaces.chat.fakes import FakeAssistant, wait_until


@pytest.fixture(autouse=True)
def reset_singleton():
    ChatHub.reset_instance()
    yield
    ChatHub.reset_instance()


@pytest.fixture
def pane():
    return ChatPane()


@pytest.fixture
def transport(pane):
    return ConsoleTransport(pane)


@pytest.fixture
def hub(transport):
    hub = ChatHub()
    hub.register(transport)
    transport.start(hub)
    yield hub
    hub.stop()


def jpeg(width: int, height: int) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (width, height), (40, 200, 40)).save(out, format="JPEG")
    return out.getvalue()


def kinds(pane: ChatPane) -> list[str]:
    return [e.kind for e in pane.entries]


class TestSubmit:
    def test_typed_text_becomes_a_console_request_and_a_streamed_reply(self, hub, pane, transport):
        hub.assistant = FakeAssistant([Token("Hel"), Token("lo"), RoundEnd("Hello"), Done("Hello")])
        pane.on_submit("hi there")
        assert wait_until(lambda: any(e.kind == KIND_ASSISTANT and e.final for e in pane.entries))
        req, query = hub.assistant.requests[0]
        assert (req.source, query, req.text_only) == ("Console", "hi there", True)
        assert kinds(pane) == [KIND_USER, KIND_ASSISTANT]
        assert pane.entries[0].body.text == "hi there"
        assert pane.entries[1].body.text == "Hello"
        assert pane.working is None

    def test_pasted_images_go_out_as_one_photo_message_with_the_text_as_caption(self, hub, pane):
        hub.assistant = FakeAssistant([Token("A cat"), RoundEnd("A cat"), Done("A cat")])
        pane.attach_image(jpeg(40, 30), "image/jpeg")
        pane.attach_image(jpeg(40, 30), "image/jpeg")
        for ch in "what":
            pane.handle_input(ch)
        pane.handle_input('enter')
        assert wait_until(lambda: hub.assistant.requests)
        _, query = hub.assistant.requests[0]
        assert query == "what"
        assert [img["media_type"] for img in hub.assistant.images[0]] == ["image/jpeg", "image/jpeg"]
        assert kinds(pane)[:2] == [KIND_USER, KIND_USER]
        assert all(e.picture is not None for e in pane.entries[:2])
        assert pane.entries[1].body.text == "what"

    def test_a_picture_without_text_asks_the_assistant_to_look(self, hub, pane):
        hub.assistant = FakeAssistant([Token("ok"), RoundEnd("ok"), Done("ok")])
        pane.attach_image(jpeg(40, 30), "image/jpeg")
        pane.handle_input('enter')
        assert wait_until(lambda: hub.assistant.requests)
        assert hub.assistant.requests[0][1] == ChatSession.IMAGES_ONLY_QUERY
        assert len(hub.assistant.images[0]) == 1

    def test_one_bubble_per_round(self, hub, pane):
        hub.assistant = FakeAssistant([Token("first"), RoundEnd("first"), Token("second"), RoundEnd("second"),
                                       Done("second")])
        pane.on_submit("go")
        assert wait_until(lambda: len([e for e in pane.entries if e.kind == KIND_ASSISTANT and e.final]) == 2)
        assert [e.body.text for e in pane.entries if e.kind == KIND_ASSISTANT] == ["first", "second"]

    def test_thinking_lands_in_its_own_block_before_the_reply(self, hub, pane):
        hub.assistant = FakeAssistant([Thinking("let me ", 10), Thinking("see", 20), ThinkingDone(0.2),
                                       Token("Sure."), RoundEnd("Sure."), Done("Sure.")])
        pane.on_submit("think")
        assert wait_until(lambda: any(e.kind == KIND_ASSISTANT and e.final for e in pane.entries))
        assert kinds(pane) == [KIND_USER, KIND_THINKING, KIND_ASSISTANT]
        thinking = pane.entries[1]
        assert thinking.body.text == "let me see" and thinking.final

    def test_notice_from_a_busy_assistant(self, hub, pane):
        hub.assistant = FakeAssistant(busy="another request")
        pane.on_submit("hi")
        assert wait_until(lambda: any(e.kind == KIND_ASSISTANT for e in pane.entries))
        assert "busy" in pane.entries[-1].body.text

    def test_notice_events_render_like_replies(self, hub, pane):
        hub.assistant = FakeAssistant([Notice("Request denied."), RoundEnd(""), Done("")])
        pane.on_submit("open the vault")
        assert wait_until(lambda: any(e.kind == KIND_ASSISTANT and e.final for e in pane.entries))
        assert pane.entries[-1].body.text == "Request denied."

    def test_empty_reply_leaves_no_bubble(self, hub, pane):
        hub.assistant = FakeAssistant([RoundEnd(""), Done("")])
        pane.on_submit("silence")
        assert wait_until(lambda: not hub.session_for(hub.transports[0], CONVERSATION).busy)
        assert kinds(pane) == [KIND_USER]
        assert pane.working is None

    def test_blank_submission_is_ignored(self, hub, pane):
        hub.assistant = FakeAssistant()
        pane.on_submit("   ")
        assert pane.entries == []
        assert hub.assistant.requests == []

    def test_without_a_hub_the_user_is_told(self, pane, transport):
        pane.on_submit("hello?")
        assert kinds(pane) == [KIND_USER, KIND_NOTICE]


class TestHistoryFeedHooks:
    def test_submit_and_reply_end_reach_the_feed(self, pane):
        feed = MagicMock()
        transport = ConsoleTransport(pane, feed)
        hub = ChatHub()
        hub.register(transport)
        transport.start(hub)
        hub.assistant = FakeAssistant([Token("Hi"), RoundEnd("Hi"), Done("Hi")])
        pane.on_submit("hello")
        feed.user_submitted.assert_called_once_with("hello")
        assert wait_until(lambda: feed.turn_finished.call_count == 1)
        feed.turn_started.assert_called_once_with()
        hub.stop()

    def test_no_turn_starts_without_a_hub(self, pane):
        feed = MagicMock()
        ConsoleTransport(pane, feed)
        pane.on_submit("hello?")
        feed.user_submitted.assert_not_called()
        feed.turn_started.assert_not_called()

    def test_a_turn_the_server_opens_counts_without_a_user_line(self, pane):
        feed = MagicMock()
        transport = ConsoleTransport(pane, feed)
        hub = ChatHub()
        hub.register(transport)
        transport.start(hub)
        hub.assistant = FakeAssistant([Token("Hi, I'm Floxy."), RoundEnd("Hi, I'm Floxy."), Done("Hi, I'm Floxy.")])
        assert hub.open_conversation("The server has just started…") is True
        assert wait_until(lambda: feed.turn_finished.call_count == 1)
        feed.turn_started.assert_called_once_with()
        feed.user_submitted.assert_not_called()
        assert kinds(pane) == [KIND_ASSISTANT]
        req, query = hub.assistant.requests[0]
        assert (req.user, query) == ("system", "The server has just started…")
        hub.stop()


class TestConsoleDraft:
    """A closed draft ignores whatever the reply thread still emits, so a late spinner or
    token cannot resurrect the working line or a removed bubble."""

    def test_finalized_draft_ignores_late_progress_and_tokens(self, pane):
        draft = ConsoleDraft(pane)
        draft.update("<b>partial</b>")
        assert draft.finalize("done") is True
        draft.show_progress(Progress(PHASE_WAITING))
        assert draft.update("late token") is False
        assert pane.working is None
        assert [e.kind for e in pane.entries] == [KIND_ASSISTANT]
        assert pane.entries[0].final and pane.entries[0].body.text == "done"

    def test_discarded_draft_stays_empty(self, pane):
        draft = ConsoleDraft(pane)
        draft.show_progress(Progress(PHASE_WAITING))
        draft.update("partial")
        draft.discard()
        draft.show_progress(Progress(PHASE_WAITING))
        assert draft.update("partial again") is False
        assert pane.working is None and pane.entries == []


class TestInterrupt:
    def test_escape_cancels_the_running_reply_and_discards_the_partial_bubble(self, hub, pane):
        hub.assistant = FakeAssistant([Token("partial "), (30, Token("never"))])
        pane.on_submit("slow one")
        assert wait_until(lambda: any(e.kind == KIND_ASSISTANT for e in pane.entries))
        pane.on_interrupt()
        assert wait_until(lambda: kinds(pane) == [KIND_USER])
        assert wait_until(lambda: hub.assistant.stream_obj.killed)

class TestNotifierPaths:
    def test_send_message_reaches_the_console(self, hub, pane):
        assert hub.send_message("Front door opened")
        assert kinds(pane) == [KIND_NOTICE]
        assert pane.entries[0].body.text == "Front door opened"

    def test_command_notification_is_annotated_when_the_command_ends(self, hub, pane):
        handle = hub.send_command_notification("Reading x", source="Console", conversation=CONVERSATION)
        assert kinds(pane) == [KIND_NOTICE] and pane.entries[0].detail is None
        hub.complete_command_notification(handle, "done after 4s, received 345 chars")
        assert pane.entries[0].detail == "done after 4s, received 345 chars"
        assert pane.entries[0].body.text == " • <i>Reading x</i>"

    def test_ephemeral_notification_is_the_status_line(self, hub, pane):
        hub.send_command_notification("Checking", ephemeral=True)
        assert pane.entries == []
        assert "Checking" in pane.status

    def test_photos_and_videos_become_captions(self, hub, pane):
        hub.send_photo(b"x" * 3000, "garden")
        hub.send_photo_album([(b"a", "one"), (b"b", "two")])
        hub.send_video(b"v" * 5000, "clip")
        assert kinds(pane) == [KIND_MEDIA] * 4
        assert pane.entries[0].body.text == "[photo, 3 kB] garden"
        assert pane.entries[-1].body.text == "[video, 5 kB] clip"

    def test_a_photo_that_decodes_reaches_the_pane_as_a_picture(self, hub, pane):
        import io
        from PIL import Image
        out = io.BytesIO()
        Image.new("RGB", (30, 20), (10, 200, 10)).save(out, format="JPEG")
        hub.send_photo(out.getvalue(), "garden")
        hub.send_video(b"v" * 5000, "clip", thumbnail=out.getvalue())
        assert [e.picture is not None for e in pane.entries] == [True, True]
        assert [e.body.text for e in pane.entries] == ["garden", "clip"]

class TestApprovals:
    def test_y_in_the_pane_approves_through_the_hub(self, hub, pane):
        result = {}

        def ask():
            result["decision"] = hub.request_approval("Unlock the door", "for Guest", timeout=5)

        t = threading.Thread(target=ask, daemon=True)
        t.start()
        assert wait_until(lambda: pane.pending_approval is not None)
        assert pane.entries[-1].kind == KIND_APPROVAL
        pane.handle_input('y')
        t.join(timeout=5)
        assert result["decision"] == "approve"
        assert pane.pending_approval is None
        assert "Approved by console" in pane.entries[-1].body.text

    def test_timeout_closes_the_prompt(self, hub, pane):
        assert hub.request_approval("Quick", "", timeout=0.05) == "timeout"
        assert pane.pending_approval is None
        assert "Timed out" in pane.entries[-1].body.text

    def test_stale_answer_clears_the_modal(self, hub, pane, transport):
        entry = pane.add_approval("unknown-token", "Old", "")
        transport.answer_approval("unknown-token", "approve")
        assert pane.pending_approval is None
        assert "no longer pending" in entry.body.text


class TestSecretPrompts:
    def test_secret_request_shows_the_masked_prompt_and_the_typed_value_reaches_accept_only(self, hub, pane):
        from interfaces.console.widgets.ChatPane import KIND_SECRET
        seen, result = [], {}

        def accept(value):
            seen.append(value)
            return "Stored as amex (•••• 1111)"

        t = threading.Thread(target=lambda: result.update(
            d=hub.request_secret("Amex (card)", "", accept, timeout=5)), daemon=True)
        t.start()
        assert wait_until(lambda: pane.pending_secret is not None)
        assert pane.entries[-1].kind == KIND_SECRET
        for ch in "4111 1111 1111 1111":
            pane.handle_input(ch)
        pane.handle_input('enter')
        t.join(timeout=5)
        assert result["d"] == "accepted" and seen == ["4111 1111 1111 1111"]
        assert pane.pending_secret is None
        assert pane.entries[-1].body.text == "Amex (card) — Stored as amex (•••• 1111)"
        assert KIND_USER not in kinds(pane)
        assert all("4111" not in e.body.text for e in pane.entries)
        assert hub.transcript == []

    def test_escape_cancels_through_the_hub(self, hub, pane):
        result = {}
        t = threading.Thread(target=lambda: result.update(
            d=hub.request_secret("Password", "", lambda v: "ok", timeout=5)), daemon=True)
        t.start()
        assert wait_until(lambda: pane.pending_secret is not None)
        pane.handle_input('escape')
        t.join(timeout=5)
        assert result["d"] == "cancel"
        assert "Cancelled by console" in pane.entries[-1].body.text

    def test_timeout_closes_the_prompt(self, hub, pane):
        assert hub.request_secret("Quick", "", lambda v: "ok", timeout=0.05) == "timeout"
        assert pane.pending_secret is None
        assert "Timed out" in pane.entries[-1].body.text

    def test_stale_answer_clears_the_modal(self, hub, pane, transport):
        entry = pane.add_secret_prompt("unknown-token", "Old", "")
        transport.answer_secret("unknown-token", "value")
        assert pane.pending_secret is None
        assert "no longer pending" in entry.body.text
        entry = pane.add_secret_prompt("unknown-token", "Old", "")
        transport.answer_secret("unknown-token", None)
        assert pane.pending_secret is None


@pytest.mark.unit
class TestAttended:
    def test_attended_follows_the_tui_clients(self, pane):
        transport = ConsoleTransport(pane)
        assert transport.attended() is True     # nothing wired: the console counts as attended
        clients = set()
        transport.has_clients = lambda: bool(clients)
        assert transport.attended() is False
        clients.add("a socket")
        assert transport.attended() is True
