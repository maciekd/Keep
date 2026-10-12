"""ChatHub: transport registry, session routing, the Notifier fan-out and approvals."""
import threading
import time

import pytest

from assistant import Destinations
from assistant.AssistantEvents import RoundEnd, Token
from assistant.Notifier import Notifier
from interfaces.chat.ChatHub import ChatHub
from interfaces.chat.ChatTypes import IncomingMessage

from .fakes import FakeAssistant, FakeTransport, wait_until


@pytest.fixture(autouse=True)
def reset_singleton():
    ChatHub.reset_instance()
    Destinations.clear()
    yield
    ChatHub.reset_instance()
    Destinations.clear()


@pytest.fixture
def transport():
    return FakeTransport(owners=[111])


@pytest.fixture
def hub(transport):
    hub = ChatHub()
    hub.register(transport)
    yield hub
    hub.stop()


@pytest.mark.unit
class TestRegistry:
    def test_register_is_idempotent(self, hub, transport):
        hub.register(transport)
        assert hub.transports == [transport]

    def test_receive_routes_to_one_session_per_conversation(self, hub, transport):
        hub.assistant = FakeAssistant([None])
        hub.receive(transport, IncomingMessage(conversation=1, sender="A", text="x"))
        hub.receive(transport, IncomingMessage(conversation=2, sender="B", text="y"))
        assert hub.session_for(transport, 1) is not hub.session_for(transport, 2)
        assert hub.session_for(transport, 1) is hub.session_for(transport, 1)
        assert wait_until(lambda: len(hub.assistant.requests) == 2)

    def test_unregister_only_drops_its_own_sessions(self, hub, transport):
        other = FakeTransport(owners=[9], name="Other")
        hub.register(other)
        mine = hub.session_for(transport, 1)
        theirs = hub.session_for(other, 1)
        assert mine is not theirs  # same conversation id on two platforms never shares a session

        hub.unregister(transport)
        assert hub.transports == [other]
        assert hub.session_for(other, 1) is theirs
        assert hub.session_for(transport, 1) is not mine

    @staticmethod
    def _slow_reply(hub, transport):
        assistant = FakeAssistant([Token("Hel"), (5.0, Token("lo")), RoundEnd(), None])
        hub.assistant = assistant
        hub.receive(transport, IncomingMessage(conversation=1, sender="A", text="x"))
        assert wait_until(lambda: transport.texts == ["Hel"])
        return assistant

    def test_stop_kills_every_in_flight_reply(self, hub, transport):
        # Shutdown must not leave a worker streaming into a torn-down transport.
        assistant = self._slow_reply(hub, transport)
        hub.stop()
        assert assistant.stream_obj.killed
        assert wait_until(lambda: transport.drafts[0].discarded)
        assert hub._sessions == {}

    def test_unregister_kills_the_transports_in_flight_replies(self, hub, transport):
        assistant = self._slow_reply(hub, transport)
        hub.unregister(transport)
        assert assistant.stream_obj.killed
        assert wait_until(lambda: transport.drafts[0].discarded)

    def test_transcript_is_bounded(self, hub, transport):
        hub.MAX_TRANSCRIPT = 5
        for i in range(7):
            hub.record(transport, 1, "user", f"m{i}")
        assert [t[4] for t in hub.transcript] == ["m2", "m3", "m4", "m5", "m6"]


@pytest.mark.unit
class TestSendMessage:
    def test_rejects_empty(self, hub, transport):
        assert hub.send_message("") is False
        assert hub.send_message("   \n\t ") is False
        assert hub.send_message(None) is False
        assert transport.calls == []

    def test_fans_out_to_every_owner_conversation_on_every_transport(self, hub, transport):
        other = FakeTransport(owners=[7, 8], name="Other")
        hub.register(other)
        transport.owners = [111, 222]
        assert hub.send_message("Hello", disable_notification=True, disable_web_page_preview=True) is True
        assert transport.calls == [("text", 111, "Hello", True, False), ("text", 222, "Hello", True, False)]
        assert other.calls == [("text", 7, "Hello", True, False), ("text", 8, "Hello", True, False)]

    def test_goes_to_the_transports_notification_route_not_every_owner(self, hub, transport):
        transport.owners = [111, 222]
        transport.notify = [111]
        assert hub.send_message("finding") is True
        assert [c for _, c, *_ in transport.calls] == [111]
        hub.send_photo(b"jpg", "cap")
        assert [c for _, c, *_ in transport.calls] == [111, 111]

    def test_false_when_nobody_accepts_or_no_targets(self, hub, transport):
        transport.accept = False
        assert hub.send_message("x") is False
        transport.owners = []
        assert hub.send_message("x") is False

    def test_a_history_row_is_not_pushed_to_a_transport_that_shows_the_history(self, hub, transport):
        # A system note is already a row the TUI's HistoryFeed draws; Telegram gets it pushed.
        console = FakeTransport(owners=[9], name="Console")
        console.shows_history = True
        hub.register(console)
        assert hub.send_message("Ring 2FA code submitted", in_history=True) is True
        assert transport.sent_texts == ["Ring 2FA code submitted"]
        assert console.calls == []

    def test_a_history_row_with_only_history_showing_transports_counts_as_delivered(self, hub, transport):
        # The console draws the row from the history, so nothing is pushed yet the user sees it.
        transport.shows_history = True
        assert hub.send_message("Ring 2FA code submitted", in_history=True) is True
        assert transport.calls == []

    def test_a_history_row_with_no_chat_at_all_is_dropped(self):
        assert ChatHub().send_message("Ring 2FA code submitted", in_history=True) is False

    def test_splits_to_the_transports_limit(self, hub, transport):
        transport.max_message_length = 10
        hub.send_message("aaaa bbbb cccc")
        assert transport.sent_texts == ["aaaa bbbb", " cccc"]

    def test_a_failing_transport_does_not_stop_the_fan_out(self, hub, transport):
        def down(*args, **kwargs):
            raise RuntimeError("down")

        broken = FakeTransport(owners=[9], name="Broken")
        broken.send_text = down
        hub.register(broken)
        assert hub.send_message("Hello") is True
        assert transport.sent_texts == ["Hello"]

        broken.set_status_line = down
        hub.send_command_notification("Working…", ephemeral=True)
        assert transport.status_calls() == [("status", 111, " • <i>Working…</i>")]

        broken.owner_conversations = down
        hub.send_command_notification("Still…", ephemeral=True)
        assert transport.status_calls()[-1] == ("status", 111, " • <i>Still…</i>")


@pytest.mark.unit
class TestOwnerDestination:
    """`post_message("owner", ...)` is the proactive route: what send_message pushes to."""

    def test_the_first_transport_brings_the_owner_destination(self, hub, transport):
        assert Destinations.names() == ["owner"]

    def test_a_post_to_owner_lands_on_the_notification_route_not_every_owner(self, hub, transport):
        transport.owners = [111, 222]
        transport.notify = [222]
        assert Destinations.send("owner", "<b>Pasta</b> is done") == ""
        assert [c[1] for c in transport.calls if c[0] == "text"] == [222]
        assert transport.calls[0][2] == "<b>Pasta</b> is done"

    def test_the_command_reaches_the_owner_through_the_registry(self, hub, transport):
        from unittest.mock import Mock
        from assistant.commands.Commands import Commands
        assert Commands(Mock()).cmd_post_message("owner", "hi there") is None
        assert [c[2] for c in transport.calls if c[0] == "text"] == ["hi there"]

    def test_the_destination_leaves_with_the_last_transport_and_on_stop(self, hub, transport):
        hub.unregister(transport)
        assert Destinations.names() == []
        hub.register(transport)
        hub.stop()
        assert Destinations.names() == []


@pytest.mark.unit
class TestCommandNotification:
    def test_ephemeral_sets_the_status_line_escaped(self, hub, transport):
        hub.send_command_notification("Reading https://x.com/?a=1&b=<2>…", ephemeral=True)
        assert transport.status_calls() == [("status", 111, " • <i>Reading https://x.com/?a=1&amp;b=&lt;2&gt;…</i>")]
        assert transport.sent_texts == []

    def test_non_ephemeral_sends_a_silent_message(self, hub, transport):
        hub.send_command_notification("Turning on kitchen…", ephemeral=False)
        assert transport.calls == [("text", 111, " • <i>Turning on kitchen…</i>", True, False)]

    def test_skips_empty(self, hub, transport):
        hub.send_command_notification("")
        hub.send_command_notification("   ", ephemeral=True)
        assert transport.calls == []

    def test_echo_goes_back_to_the_requests_own_chat_only(self, hub, transport):
        # A command typed into the TUI must not post into the owner's Telegram chats.
        console = FakeTransport(owners=["tui"], name="Console", source="Console")
        hub.register(console)
        hub.send_command_notification("Turning off kitchen", ephemeral=False, source="Console", conversation="tui")
        assert console.calls == [("text", "tui", " • <i>Turning off kitchen</i>", True, False)]
        assert transport.calls == []

    def test_ephemeral_echo_stays_in_the_requests_own_chat(self, hub, transport):
        # A Telegram request's status line must not land on the console, where nothing
        # would ever clear it; the origin need not be an owner conversation.
        console = FakeTransport(owners=["tui"], name="Console", source="Console")
        hub.register(console)
        hub.send_command_notification("Reading…", ephemeral=True, source=transport.source, conversation=222)
        assert transport.status_calls() == [("status", 222, " • <i>Reading…</i>")]
        assert console.calls == []

    def test_origin_outside_the_chat_layer_falls_back_to_every_owner_chat(self, hub, transport):
        hub.send_command_notification("Hi", ephemeral=False, source="SomeHttpClient", conversation=5)
        hub.send_command_notification("Hi", ephemeral=False, source=transport.source, conversation=None)
        assert transport.calls == [("text", 111, " • <i>Hi</i>", True, False)] * 2


class AnnotatingTransport(FakeTransport):
    """A transport that can annotate a command line after posting it."""

    def send_command_line(self, conversation, text):
        self.calls.append(("cmd", conversation, text))
        return 7

    def finish_command_line(self, conversation, ref, text, detail):
        self.calls.append(("cmd_done", conversation, ref, text, detail))


@pytest.mark.unit
class TestCommandEcho:
    @pytest.fixture
    def console(self, hub):
        console = AnnotatingTransport(owners=["tui"], name="Console", source="Console")
        hub.register(console)
        return console

    def test_completion_annotates_the_line_it_posted(self, hub, console):
        handle = hub.send_command_notification("Reading x", source="Console", conversation="tui")
        assert console.calls == [("cmd", "tui", "Reading x")]
        hub.complete_command_notification(handle, "done after 4s, received 345 chars")
        assert console.calls[-1] == ("cmd_done", "tui", 7, "Reading x", "done after 4s, received 345 chars")

    def test_a_plain_transport_gets_a_silent_message_and_no_annotation(self, hub, transport):
        handle = hub.send_command_notification("Turning on kitchen…", ephemeral=False)
        assert handle is not None
        hub.complete_command_notification(handle, "done after 0.2s")
        assert transport.calls == [("text", 111, " • <i>Turning on kitchen…</i>", True, False)]

    def test_ephemeral_and_empty_echoes_have_no_handle(self, hub, transport):
        assert hub.send_command_notification("Reading…", ephemeral=True) is None
        assert hub.send_command_notification("  ") is None
        hub.complete_command_notification(None, "done")          # nothing to do, no error
        hub.complete_command_notification("garbage", "done")

    def test_activity_counts_the_running_commands_of_one_conversation(self, hub, console, transport):
        assert hub.command_activity(console, "tui") == (0, 0.0)
        before = time.time()
        first = hub.send_command_notification("Reading x", source="Console", conversation="tui")
        second = hub.send_command_notification("Reading y", source="Console", conversation="tui")
        running, since = hub.command_activity(console, "tui")
        assert running == 2 and before <= since <= time.time()
        assert hub.command_activity(transport, 111) == (0, 0.0)   # somebody else's chat
        hub.complete_command_notification(first, "done after 1s")
        assert hub.command_activity(console, "tui")[0] == 1
        hub.complete_command_notification(second, "done after 2s")
        running, ended = hub.command_activity(console, "tui")
        assert running == 0 and since <= ended <= time.time()
        assert hub.command_activity(transport, 111) == (0, 0.0)   # the end is that chat's alone

    def test_a_command_that_never_reports_back_is_forgotten(self, hub, console):
        # Negative, not 0.0: the hub forgets an echo older than the threshold (strictly), and
        # two time.time() reads in a row can be equal — which made this flake under xdist.
        hub.STALE_COMMAND_SECONDS = -1.0
        hub.send_command_notification("Reading x", source="Console", conversation="tui")
        assert hub.command_activity(console, "tui") == (0, 0.0)


@pytest.mark.unit
class TestAlbumHold:
    def test_expect_and_drop_reach_the_conversations_session(self, hub, transport):
        session = hub.session_for(transport, 1)
        session.ALBUM_DEBOUNCE_SECONDS = 5
        hub.expect_photo(transport, 1, "Anna", "g")
        assert session._album is not None and session._album.pending == 1
        hub.photo_dropped(transport, 1)
        assert session._album.pending == 0


@pytest.mark.unit
class TestMediaSends:
    def test_photo_album_video_fan_out(self, hub, transport):
        assert hub.send_photo(b"jpg", "cap") is True
        assert hub.send_photo_album([(b"a", "1"), (b"b", "2")]) is True
        assert hub.send_video(b"mp4", "clip", thumbnail=b"thumb") is True
        assert transport.calls == [("photo", 111, b"jpg", "cap"),
                                   ("album", 111, [(b"a", "1"), (b"b", "2")]),
                                   ("video", 111, b"mp4", "clip", b"thumb")]

    def test_media_answering_a_request_goes_back_to_its_chat(self, hub, transport):
        """telegram_bot.notify_chat_ids narrows the pushes only: a photo the requester asked for
        must reach the requester, whichever chat they wrote from."""
        other = FakeTransport(owners=[7], name="Other", source="Console")
        hub.register(other)
        transport.owners = [111, 222]
        transport.notify = [111]
        assert hub.send_photo(b"jpg", "cap", source="TextChat", conversation=222) is True
        assert hub.send_photo_album([(b"a", "1")], source="TextChat", conversation=222) is True
        assert hub.send_video(b"mp4", "clip", source="TextChat", conversation=222) is True
        assert [c[:2] for c in transport.calls] == [("photo", 222), ("album", 222), ("video", 222)]
        assert other.calls == []

    def test_media_answering_a_request_without_a_chat_goes_to_every_owner(self, hub, transport):
        """A voice request has no chat transport of its own; its photo goes to every owner
        chat, not only the notification route — whoever asked may be reading either."""
        transport.owners = [111, 222]
        transport.notify = [111]
        assert hub.send_photo(b"jpg", "cap", source="VoiceToVoice", conversation=None) is True
        assert [c[1] for c in transport.calls] == [111, 222]

    def test_false_without_targets(self, hub, transport):
        transport.owners = []
        assert hub.send_photo(b"jpg") is False
        assert hub.send_photo_album([(b"a", "1")]) is False
        assert hub.send_video(b"mp4") is False

    def test_a_failing_transport_does_not_stop_the_fan_out(self, hub, transport):
        def down(*args, **kwargs):
            raise RuntimeError("down")

        broken = FakeTransport(owners=[9], name="Broken")
        broken.send_photo = broken.send_video = down
        hub.register(broken)
        assert hub.send_photo(b"jpg", "cap") is True
        assert hub.send_video(b"mp4") is True
        assert transport.calls == [("photo", 111, b"jpg", "cap"), ("video", 111, b"mp4", "", None)]

@pytest.mark.unit
class TestApprovals:
    def test_unavailable_without_targets(self, hub, transport):
        transport.owners = []
        assert hub.request_approval("test", "details", timeout=0.1) == "unavailable"
        assert hub.pending_approvals() == 0

    def test_unavailable_when_no_prompt_could_be_shown(self, hub, transport):
        transport.prompt_ref = None
        assert hub.request_approval("test", "details", timeout=0.1) == "unavailable"
        assert hub.pending_approvals() == 0

    def test_times_out_and_closes_the_prompts(self, hub, transport):
        assert hub.request_approval("test", "details", timeout=0.05) == "timeout"
        assert transport.calls[0][:2] == ("approval", 111)
        assert transport.calls[0][3:] == ("test", "details")
        assert transport.calls[1] == ("close", 111, 101, "⏱ Timed out")
        assert hub.pending_approvals() == 0

    def test_first_answer_wins_and_every_prompt_is_closed(self, hub, transport):
        other = FakeTransport(owners=[9], name="Other")
        other.prompt_ref = "ref-9"
        hub.register(other)
        result = {}

        def ask():
            result["decision"] = hub.request_approval("Send a message", "hi", timeout=2.0)

        t = threading.Thread(target=ask, daemon=True)
        t.start()
        assert wait_until(lambda: any(c[0] == "approval" for c in other.calls))
        token = next(c[2] for c in transport.calls if c[0] == "approval")

        assert hub.resolve_approval(token, "approve", who="alice") is True
        t.join(timeout=2)
        assert result["decision"] == "approve"
        assert ("close", 111, 101, "✅ Approved by alice") in transport.calls
        assert ("close", 9, "ref-9", "✅ Approved by alice") in other.calls
        assert hub.resolve_approval(token, "deny", who="bob") is False
        assert hub.pending_approvals() == 0

    def test_asked_only_in_the_chat_the_request_came_from(self, hub, transport):
        # A family member's Telegram chat must not get the keyboard for a command the owner
        # ran from the console — the person who asked is the one who answers.
        console = FakeTransport(owners=["tui"], name="Console", source="Console")
        hub.register(console)
        other_telegram_chat = FakeTransport(owners=[222], name="Other", source="TextChat")
        hub.register(other_telegram_chat)
        result = {}
        t = threading.Thread(target=lambda: result.update(
            d=hub.request_approval("x", timeout=2.0, source="Console", conversation="tui")), daemon=True)
        t.start()
        assert wait_until(lambda: any(c[0] == "approval" for c in console.calls))
        assert transport.calls == [] and other_telegram_chat.calls == []
        token = next(c[2] for c in console.calls if c[0] == "approval")
        assert hub.resolve_approval(token, "approve") is True
        t.join(timeout=2)
        assert result["d"] == "approve"
        assert transport.calls == [] and other_telegram_chat.calls == []

    def test_a_request_from_outside_the_chat_layer_asks_every_owner_chat(self, hub, transport):
        console = FakeTransport(owners=["tui"], name="Console", source="Console")
        hub.register(console)
        result = {}
        t = threading.Thread(target=lambda: result.update(
            d=hub.request_approval("x", timeout=2.0, source="Voice", conversation=None)), daemon=True)
        t.start()
        assert wait_until(lambda: any(c[0] == "approval" for c in console.calls)
                          and any(c[0] == "approval" for c in transport.calls))
        token = next(c[2] for c in transport.calls if c[0] == "approval")
        assert hub.resolve_approval(token, "approve") is True
        t.join(timeout=2)
        assert result["d"] == "approve"

    def test_deny(self, hub, transport):
        result = {}
        t = threading.Thread(target=lambda: result.update(d=hub.request_approval("x", timeout=2.0)), daemon=True)
        t.start()
        assert wait_until(lambda: any(c[0] == "approval" for c in transport.calls))
        token = next(c[2] for c in transport.calls if c[0] == "approval")
        hub.resolve_approval(token, "deny", who="<b>")
        t.join(timeout=2)
        assert result["d"] == "deny"
        assert ("close", 111, 101, "❌ Denied by &lt;b&gt;") in transport.calls

    def test_unknown_token_is_ignored(self, hub):
        assert hub.resolve_approval("nope", "approve") is False

    def test_a_transport_whose_prompt_raises_does_not_block_the_others(self, hub, transport):
        # A sensitive command must still be askable in the healthy chat; otherwise it is
        # silently answered "unavailable" because one platform is down.
        def down(*args, **kwargs):
            raise RuntimeError("down")

        broken = FakeTransport(owners=[9], name="Broken")
        broken.send_approval_prompt = down
        hub.register(broken)
        result = {}
        t = threading.Thread(target=lambda: result.update(d=hub.request_approval("x", timeout=2.0)), daemon=True)
        t.start()
        assert wait_until(lambda: any(c[0] == "approval" for c in transport.calls))
        token = next(c[2] for c in transport.calls if c[0] == "approval")
        assert hub.resolve_approval(token, "approve") is True
        t.join(timeout=2)
        assert result["d"] == "approve"
        assert ("close", 111, 101, "✅ Approved by user") in transport.calls

    def test_a_close_that_raises_still_closes_the_other_prompts(self, transport):
        def down(*args, **kwargs):
            raise RuntimeError("down")

        broken = FakeTransport(owners=[9], name="Broken")
        broken.close_approval_prompt = down
        hub = ChatHub()
        hub.register(broken)  # first, so its failing close runs before the good transport's
        hub.register(transport)
        assert hub.request_approval("x", timeout=0.05) == "timeout"
        assert ("close", 111, 101, "⏱ Timed out") in transport.calls
        assert hub.pending_approvals() == 0

    def test_simultaneous_answers_resolve_exactly_once_without_raising(self, hub, transport):
        # Console and Telegram answering in the same instant: one wins, the rest are told
        # "not pending" — never an InvalidStateError from a second set_result.
        result = {}
        asker = threading.Thread(target=lambda: result.update(d=hub.request_approval("x", timeout=2.0)), daemon=True)
        asker.start()
        assert wait_until(lambda: any(c[0] == "approval" for c in transport.calls))
        token = next(c[2] for c in transport.calls if c[0] == "approval")
        answers = ["approve", "deny"] * 8
        barrier = threading.Barrier(len(answers))
        outcomes = []

        def answer(decision):
            barrier.wait()
            try:
                outcomes.append(hub.resolve_approval(token, decision, who=decision))
            except Exception as e:
                outcomes.append(e)

        threads = [threading.Thread(target=answer, args=(d,), daemon=True) for d in answers]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=2)
        asker.join(timeout=2)
        assert outcomes.count(True) == 1
        assert outcomes.count(False) == len(answers) - 1, outcomes
        assert result["d"] in ("approve", "deny")

    def test_configured_timeout_is_used(self, transport):
        from unittest.mock import patch
        with patch('interfaces.chat.ChatHub.config') as cfg:
            cfg.get = lambda key, default=None: {'chat.approval_timeout_seconds': 0.05}.get(key, default)
            hub = ChatHub()
        hub.register(transport)
        start = time.monotonic()
        assert hub.request_approval("x") == "timeout"
        assert time.monotonic() - start < 1.0


@pytest.mark.unit
class TestSecrets:
    """A value typed into a masked prompt goes to the requester's `accept` and nowhere else."""

    @staticmethod
    def _ask(hub, accept, timeout=2.0, title="Amex (card)", details="Reply with the number", **origin):
        result = {}
        t = threading.Thread(target=lambda: result.update(
            d=hub.request_secret(title, details, accept, timeout=timeout, **origin)), daemon=True)
        t.start()
        return t, result

    @staticmethod
    def _token(transport):
        return next(c[2] for c in transport.calls if c[0] == "secret")

    @staticmethod
    def _prompts(transport):
        return [c for c in transport.calls if c[0] == "secret"]

    @staticmethod
    def _closes(transport):
        return [c for c in transport.calls if c[0] == "close_secret"]

    @pytest.fixture
    def log_lines(self, monkeypatch):
        import common.logging as logging
        lines = []
        monkeypatch.setattr(logging, "log", lambda message, *a, **k: lines.append(message))
        return lines

    def test_unavailable_without_targets(self, hub, transport):
        transport.owners = []
        assert hub.request_secret("t", "d", lambda v: "ok", timeout=0.1) == "unavailable"
        assert hub.pending_secrets() == 0

    def test_prompt_opens_only_in_the_chat_that_asked(self, hub, transport):
        # A vault request typed into the TUI must not open a masked prompt in Telegram too.
        console = FakeTransport(owners=["tui"], name="Console", source="Console")
        hub.register(console)
        t, result = self._ask(hub, lambda v: "ok", source="Console", conversation="tui")
        wait_until(lambda: self._prompts(console))
        assert transport.calls == []
        assert hub.claim_secret(console, "tui", "4111") is True
        t.join(2.0)
        assert result["d"] == "accepted"
        assert self._closes(console) and not transport.calls

    def test_origin_outside_the_chat_layer_asks_every_owner_chat(self, hub, transport):
        console = FakeTransport(owners=["tui"], name="Console", source="Console")
        hub.register(console)
        t, result = self._ask(hub, lambda v: "ok", source="SomeHttpClient", conversation=5)
        wait_until(lambda: self._prompts(console) and self._prompts(transport))
        assert hub.claim_secret(transport, 111, "4111") is True
        t.join(2.0)
        assert result["d"] == "accepted"

    def test_unavailable_when_no_prompt_could_be_shown(self, hub, transport):
        transport.prompt_ref = None
        assert hub.request_secret("t", "d", lambda v: "ok", timeout=0.1) == "unavailable"
        assert hub.pending_secrets() == 0

    def test_times_out_and_closes_the_prompts(self, hub, transport):
        assert hub.request_secret("Amex (card)", "d", lambda v: "ok", timeout=0.05) == "timeout"
        assert transport.calls[0][:2] == ("secret", 111)
        assert transport.calls[0][3:] == ("Amex (card)", "d")
        assert transport.calls[1] == ("close_secret", 111, 101, "⏱ Timed out")
        assert hub.pending_secrets() == 0

    def test_accepted_answer_closes_every_prompt_with_the_confirmation(self, hub, transport):
        other = FakeTransport(owners=[9], name="Other")
        other.prompt_ref = "ref-9"
        hub.register(other)
        seen = []

        def accept(value):
            seen.append(value)
            return "Stored as amex (•••• 1111)"

        t, result = self._ask(hub, accept)
        assert wait_until(lambda: self._prompts(other))
        assert hub.claim_secret(other, 9, "4111 1111 1111 1111", who="alice") is True
        t.join(timeout=2)
        assert result["d"] == "accepted"
        assert seen == ["4111 1111 1111 1111"]
        assert ("close_secret", 111, 101, "✅ Stored as amex (•••• 1111)") in transport.calls
        assert ("close_secret", 9, "ref-9", "✅ Stored as amex (•••• 1111)") in other.calls
        assert hub.pending_secrets() == 0
        # Over: the next text in that chat is ordinary chat again.
        assert hub.claim_secret(other, 9, "hello") is False

    def test_claim_is_false_in_a_conversation_the_prompt_was_not_shown_in(self, hub, transport):
        t, result = self._ask(hub, lambda v: "ok")
        assert wait_until(lambda: self._prompts(transport))
        assert hub.claim_secret(transport, 222, "secret") is False
        hub.cancel_secret(self._token(transport))
        t.join(timeout=2)

    def test_rejected_value_is_asked_again_with_the_reason(self, hub, transport):
        from assistant.Notifier import SecretRejected
        answers = iter([SecretRejected("card number must be 13-19 digits"), "Stored as visa"])

        def accept(value):
            a = next(answers)
            if isinstance(a, Exception):
                raise a
            return a

        t, result = self._ask(hub, accept)
        assert wait_until(lambda: self._prompts(transport))
        first_token = self._token(transport)
        assert hub.claim_secret(transport, 111, "1234") is True
        assert wait_until(lambda: len(self._prompts(transport)) == 2)
        # Re-asked with the same token, not closed and reopened: a transport updates its
        # open prompt in place, so the chat keeps one prompt per request.
        assert self._closes(transport) == []
        again = self._prompts(transport)[1]
        assert again[2] == first_token and again[3] == "Amex (card)"
        assert again[4] == "Reply with the number\n⚠ card number must be 13-19 digits — try again (2/3)"
        assert hub.claim_secret(transport, 111, "4111 1111 1111 1111") is True
        t.join(timeout=2)
        assert result["d"] == "accepted"
        assert self._closes(transport) == [("close_secret", 111, 101, "✅ Stored as visa")]

    def test_three_rejections_give_up(self, hub, transport):
        from assistant.Notifier import SecretRejected

        def accept(value):
            raise SecretRejected("too short to vault safely (min 6)")

        t, result = self._ask(hub, accept)
        assert wait_until(lambda: self._prompts(transport))
        for attempt in range(3):
            n = len(self._prompts(transport))
            assert hub.claim_secret(transport, 111, "abc") is True
            if attempt < 2:   # the third rejection gives up instead of asking again
                assert wait_until(lambda: len(self._prompts(transport)) == n + 1)
        t.join(timeout=2)
        assert result["d"] == "rejected"
        assert len(self._prompts(transport)) == 3
        assert self._closes(transport) == [("close_secret", 111, 101, "⚠ too short to vault safely (min 6) — giving up")]
        assert hub.pending_secrets() == 0

    def test_cancel(self, hub, transport):
        t, result = self._ask(hub, lambda v: "ok")
        assert wait_until(lambda: self._prompts(transport))
        token = self._token(transport)
        assert hub.cancel_secret(token, who="<b>") is True
        t.join(timeout=2)
        assert result["d"] == "cancel"
        assert ("close_secret", 111, 101, "✖ Cancelled by &lt;b&gt;") in transport.calls
        assert hub.cancel_secret(token) is False
        assert hub.cancel_secret("nope") is False

    def test_a_second_request_in_the_same_chat_is_unavailable(self, hub, transport):
        t, result = self._ask(hub, lambda v: "ok")
        assert wait_until(lambda: self._prompts(transport))
        assert hub.request_secret("Another", "d", lambda v: "ok", timeout=1.0) == "unavailable"
        assert len(self._prompts(transport)) == 1
        hub.cancel_secret(self._token(transport))
        t.join(timeout=2)

    def test_the_value_is_never_logged_or_recorded(self, hub, transport, log_lines):
        from assistant.Notifier import SecretRejected
        value = "hunter2-very-secret"
        answers = iter([SecretRejected("no good"), RuntimeError(f"boom {value}"), "Stored"])

        def accept(v):
            a = next(answers)
            if isinstance(a, Exception):
                raise a
            return a

        t, result = self._ask(hub, accept)
        assert wait_until(lambda: self._prompts(transport))
        hub.claim_secret(transport, 111, value)
        wait_until(lambda: len(self._prompts(transport)) == 2)
        hub.claim_secret(transport, 111, value)   # the RuntimeError: its text carries the value
        t.join(timeout=2)
        assert result["d"] == "rejected"
        assert not any(value in line for line in log_lines), log_lines
        assert not any(value in c[-1] for c in self._closes(transport))
        assert all(value not in row[4] for row in hub.transcript)

    def test_an_answer_arriving_while_one_is_handled_is_dropped(self, hub, transport):
        gate = threading.Event()
        seen = []

        def accept(value):
            seen.append(value)
            gate.wait(2)
            return "ok"

        t, result = self._ask(hub, accept)
        assert wait_until(lambda: self._prompts(transport))
        first = threading.Thread(target=lambda: hub.claim_secret(transport, 111, "one"), daemon=True)
        first.start()
        assert wait_until(lambda: seen == ["one"])
        assert hub.claim_secret(transport, 111, "two") is True   # consumed, not handed to accept
        gate.set()
        first.join(timeout=2)
        t.join(timeout=2)
        assert result["d"] == "accepted" and seen == ["one"]

    def test_a_transport_whose_prompt_raises_does_not_block_the_others(self, hub, transport):
        def down(*args, **kwargs):
            raise RuntimeError("down")

        broken = FakeTransport(owners=[9], name="Broken")
        broken.send_secret_prompt = down
        hub.register(broken)
        t, result = self._ask(hub, lambda v: "ok")
        assert wait_until(lambda: self._prompts(transport))
        assert hub.claim_secret(transport, 111, "value") is True
        t.join(timeout=2)
        assert result["d"] == "accepted"


@pytest.mark.unit
class TestQuestionsAreTheOwnersToAnswer:
    """A bot with an empty allowed_chat_ids lets anyone talk; the approval gate was the last
    line, so a stranger's request must not put the approval in the stranger's chat."""

    def test_an_approval_from_a_non_owner_chat_is_asked_in_the_owner_chats(self, hub, transport):
        result = {}
        t = threading.Thread(target=lambda: result.update(d=hub.request_approval(
            "send_email", timeout=2.0, source=transport.source, conversation=999)), daemon=True)
        t.start()
        assert wait_until(lambda: any(c[0] == "approval" for c in transport.calls))
        assert [c[1] for c in transport.calls if c[0] == "approval"] == [111]
        token = next(c[2] for c in transport.calls if c[0] == "approval")
        hub.resolve_approval(token, "deny")
        t.join(timeout=2)
        assert result["d"] == "deny"

    def test_a_failing_owner_list_never_puts_the_question_in_the_requesters_chat(self, hub, transport):
        other = FakeTransport(owners=[7], name="Other", source="Other")
        hub.register(other)

        def down(*args, **kwargs):
            raise RuntimeError("owners unavailable")
        transport.owner_conversations = down
        result = {}
        t = threading.Thread(target=lambda: result.update(d=hub.request_approval(
            "send_email", timeout=2.0, source=transport.source, conversation=999)), daemon=True)
        t.start()
        assert wait_until(lambda: any(c[0] == "approval" for c in other.calls))
        assert [c[1] for c in other.calls if c[0] == "approval"] == [7]
        assert not any(c[0] == "approval" for c in transport.calls), "fail closed: the stranger is never asked"
        token = next(c[2] for c in other.calls if c[0] == "approval")
        hub.resolve_approval(token, "deny")
        t.join(timeout=2)
        assert result["d"] == "deny"

    def test_an_approval_with_no_owner_chat_at_all_is_unavailable(self, hub, transport):
        transport.owners = []
        assert hub.request_approval("send_email", timeout=0.1, source=transport.source, conversation=999) == "unavailable"
        assert transport.calls == []

    def test_a_secret_prompt_from_a_non_owner_chat_opens_in_the_owner_chats(self, hub, transport):
        t = threading.Thread(target=lambda: hub.request_secret(
            "t", "d", lambda v: "ok", timeout=2.0, source=transport.source, conversation=999), daemon=True)
        t.start()
        assert wait_until(lambda: any(c[0] == "secret" for c in transport.calls))
        assert [c[1] for c in transport.calls if c[0] == "secret"] == [111]
        assert hub.claim_secret(transport, 111, "4111") is True
        t.join(timeout=2)


@pytest.mark.unit
class TestQuestionsReachEveryOwner:
    """telegram_bot.notify_chat_ids narrows the proactive pushes only: an approval or a secret
    prompt still goes to every owner conversation, since any of them may answer."""

    def test_approval_prompts_ignore_the_notification_route(self, hub, transport):
        transport.owners = [111, 222]
        transport.notify = [111]
        result = {}
        t = threading.Thread(target=lambda: result.update(d=hub.request_approval("x", timeout=2.0)), daemon=True)
        t.start()
        assert wait_until(lambda: sum(c[0] == "approval" for c in transport.calls) == 2)
        assert sorted(c[1] for c in transport.calls if c[0] == "approval") == [111, 222]
        token = next(c[2] for c in transport.calls if c[0] == "approval")
        hub.resolve_approval(token, "approve")
        t.join(timeout=2)
        assert result["d"] == "approve"

    def test_secret_prompts_ignore_the_notification_route(self, hub, transport):
        transport.owners = [111, 222]
        transport.notify = [111]
        t = threading.Thread(target=lambda: hub.request_secret("t", "d", lambda v: "ok", timeout=2.0), daemon=True)
        t.start()
        assert wait_until(lambda: sum(c[0] == "secret" for c in transport.calls) == 2)
        assert sorted(c[1] for c in transport.calls if c[0] == "secret") == [111, 222]
        hub.claim_secret(transport, 222, "value")
        t.join(timeout=2)
        assert not t.is_alive()


@pytest.mark.unit
class TestTelemetry:
    def test_a_delivered_push_is_a_proactive_assistant_turn(self, hub, transport, monkeypatch):
        from unittest.mock import Mock
        from assistant.Telemetry import Telemetry
        sink = Mock()
        monkeypatch.setattr(Telemetry, "get_instance", classmethod(lambda cls: sink))
        assert hub.send_message("The car is charged.") is True
        sink.assistant_turn.assert_called_once_with("The car is charged.", "notifier", proactive=True)

    def test_a_dropped_push_is_not_reported(self, monkeypatch):
        from unittest.mock import Mock
        from assistant.Telemetry import Telemetry
        sink = Mock()
        monkeypatch.setattr(Telemetry, "get_instance", classmethod(lambda cls: sink))
        assert ChatHub().send_message("Nobody to tell.") is False
        sink.assistant_turn.assert_not_called()


@pytest.mark.unit
class TestReceiveChatMessage:
    """A feed hands a message to the transport whose source it names (the owner's Telegram
    account saw her name in a group the bot is in)."""

    def test_the_message_becomes_a_turn_of_that_transport(self, hub, transport):
        seen = []
        hub.receive = lambda t, m: seen.append((t, m))
        assert hub.receive_chat_message("TextChat", -5001, "Erik", "Michelle, hi", guest=True,
                                        sender_is_bot=False, chat_name="Family") is True
        (t, m), = seen
        assert t is transport and m.conversation == -5001 and m.sender == "Erik" and m.text == "Michelle, hi"
        assert m.guest is True and m.chat_name == "Family"

    def test_no_transport_for_the_source_drops_it(self, hub):
        hub.receive = lambda t, m: pytest.fail("nothing should be received")
        assert hub.receive_chat_message("Carrier Pigeon", 1, "Erik", "hi") is False
