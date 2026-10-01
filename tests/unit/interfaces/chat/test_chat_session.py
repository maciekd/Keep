"""ChatSession against the FakeTransport: the reply state machine and the intake rules,
independent of any chat platform."""
import base64
import threading
import time
from unittest.mock import patch

import pytest

from assistant.AssistantEvents import (CommandRan, Done, Notice, RoundEnd, Thinking, ThinkingDone, Token,
                                       UserIdentified)
from interfaces.chat.ChatHub import ChatHub
from interfaces.chat.ChatSession import ChatSession, _Request
from interfaces.chat.ChatTypes import (KIND_PHOTO, KIND_REACTION, KIND_TEXT, KIND_VIDEO, KIND_VOICE,
                                       PHASE_COMMANDS, PHASE_PREFILL, PHASE_THINKING, PHASE_WAITING, Attachment,
                                       IncomingMessage)

from .fakes import FakeAssistant, FakeStream, FakeTransport, wait_until

CONV = 123


@pytest.fixture
def transport():
    return FakeTransport(owners=[CONV])


@pytest.fixture
def hub(transport):
    hub = ChatHub()
    hub.register(transport)
    yield hub
    hub.stop()


@pytest.fixture
def session(hub, transport):
    return hub.session_for(transport, CONV)


def run_reply(session, hub, events, suppress_no=False, query="hello", sender="Anna", images=None, busy=None,
              raise_exc=None, assistant=None):
    """Drive the reply state machine synchronously on the test thread."""
    if assistant is None:
        assistant = FakeAssistant(events, busy=busy, raise_exc=raise_exc)
    hub.assistant = assistant
    request = _Request()
    session._reply(request, query, sender, images, suppress_no)
    return assistant


def msg(kind=KIND_TEXT, text="hello", **kw):
    return IncomingMessage(conversation=CONV, sender="Anna", kind=kind, text=text, **kw)


@pytest.mark.unit
class TestReplyRendering:
    def test_tokens_stream_into_one_draft_finalized_on_round_end(self, session, hub, transport):
        run_reply(session, hub, [Token("Hello"), Token(" world"), Token("!"), RoundEnd(), None])
        assert len(transport.drafts) == 2  # the reply, and the empty draft opened after RoundEnd
        draft = transport.drafts[0]
        assert draft.texts[0].startswith("Hello")
        assert draft.finalized == "Hello world!"
        assert transport.drafts[1].discarded  # nothing arrived after the round ended

    def test_reply_finished_is_the_last_call_even_when_the_assistant_fails(self, session, hub, transport):
        run_reply(session, hub, [Token("Hi"), Done("Hi")])
        assert transport.calls[-1] == ("reply_finished", CONV)
        transport.calls.clear()
        run_reply(session, hub, [], raise_exc=RuntimeError("boom"))
        assert transport.calls[-1] == ("reply_finished", CONV)

    def test_round_end_finalizes_and_starts_a_new_bubble(self, session, hub, transport):
        run_reply(session, hub, [Token("First round"), RoundEnd(), Token("Second round"), RoundEnd(), Done("")])
        assert transport.finalized == ["First round", "Second round"]

    def test_no_text_means_nothing_delivered(self, session, hub, transport):
        run_reply(session, hub, [Done("")])
        assert transport.finalized == []
        assert transport.drafts[0].discarded

    def test_done_flushes_the_buffer(self, session, hub, transport):
        run_reply(session, hub, [Token("Final words"), Done("Final words")])
        assert transport.finalized == ["Final words"]

    def test_notice_is_delivered_as_text(self, session, hub, transport):
        run_reply(session, hub, [Notice("Request denied."), RoundEnd(), None])
        assert "Request denied." in transport.finalized

    def test_unrendered_events_are_ignored(self, session, hub, transport):
        run_reply(session, hub, [UserIdentified("Anna"), CommandRan("turn_on", "kitchen"), Token("Ok"), RoundEnd(), None])
        assert transport.texts == ["Ok", "Ok"]

    def test_partial_html_is_cut_at_a_balanced_prefix(self, session, hub, transport):
        run_reply(session, hub, [Token("Hi <b>bo"), Token("ld</b>"), RoundEnd(), None])
        draft = transport.drafts[0]
        assert draft.texts[0] == "Hi "
        assert draft.finalized == "Hi <b>bold</b>"

    def test_markdown_bold_closing_in_a_later_token_reaches_the_chat_as_html(self, session, hub, transport):
        """The repair runs on the growing buffer, not per token: a pair opened in one chunk
        and closed in the next is converted, and no bubble ever carries the closed pair raw."""
        run_reply(session, hub, [Token("Call **Char"), Token("lotte** with `ls"), Token(" -la`"), RoundEnd(), None])
        draft = transport.drafts[0]
        assert draft.finalized == "Call <b>Charlotte</b> with <code>ls -la</code>"
        assert not any("**Charlotte**" in text for text in draft.texts)

    def test_transcript_records_user_facing_reply(self, session, hub, transport):
        run_reply(session, hub, [Token("A"), RoundEnd(), Token("B"), RoundEnd(), None])
        assert hub.transcript[-1][3:] == ("assistant", "A\nB")


@pytest.mark.unit
class TestProgress:
    def test_idle_tick_reports_progress_then_reply_replaces_it(self, session, hub, transport):
        session.SPINNER_TICK_SECONDS = 0.01
        run_reply(session, hub, [(0.05, Token("Hello")), RoundEnd(), None])
        draft = transport.drafts[0]
        assert draft.calls[0][0] == "progress"
        assert draft.calls[0][1].phase == PHASE_WAITING
        assert draft.finalized == "Hello"

    def test_thinking_events_report_thinking_phase_with_token_counts(self, session, hub, transport):
        run_reply(session, hub, [Thinking("let me see", 123), Thinking(" more", 20700), Token("Hello"), RoundEnd(), None])
        progress = transport.drafts[0].progress
        assert [p.phase for p in progress] == [PHASE_THINKING, PHASE_THINKING]
        assert [p.output_tokens for p in progress] == [123, 20700]
        assert progress[-1].thinking_text == "let me see more"
        assert [p.tick for p in progress] == [1, 2]

    def test_thinking_done_returns_to_waiting_on_the_next_tick(self, session, hub, transport):
        session.SPINNER_TICK_SECONDS = 0.01
        run_reply(session, hub, [Thinking("x", 5), ThinkingDone(1.0), (0.05, Token("Hi")), RoundEnd(), None])
        phases = [p.phase for p in transport.drafts[0].progress]
        assert phases[0] == PHASE_THINKING
        assert PHASE_WAITING in phases[1:]

    def test_waiting_reads_as_prefill_while_a_local_model_prefills(self, session, hub, transport):
        from assistant.llm.OllamaLLM import OllamaLLM
        session.SPINNER_TICK_SECONDS = 0.01
        with patch.object(OllamaLLM, "prefill_progress", return_value=43.0):
            run_reply(session, hub, [(0.05, Token("Hi")), RoundEnd(), None])
        first = transport.drafts[0].progress[0]
        assert first.phase == PHASE_PREFILL and first.prefill == 43.0

    @pytest.mark.parametrize("pct", [0.0, 100.0])
    def test_prefill_yields_to_thinking_and_reads_as_waiting_at_either_end(self, session, hub, transport, pct):
        from assistant.llm.OllamaLLM import OllamaLLM
        session.SPINNER_TICK_SECONDS = 0.01
        with patch.object(OllamaLLM, "prefill_progress", return_value=pct):
            run_reply(session, hub, [(0.05, Thinking("hm", 3)), (0.05, Token("Hi")), RoundEnd(), None])
        phases = [p.phase for p in transport.drafts[0].progress]
        assert phases[0] == PHASE_WAITING and PHASE_THINKING in phases and PHASE_PREFILL not in phases

    def test_no_progress_after_reply_started(self, session, hub, transport):
        run_reply(session, hub, [Token("Hello"), Thinking("late"), Token(" world"), RoundEnd(), None])
        assert transport.drafts[0].progress == []
        assert transport.drafts[0].finalized == "Hello world"

    def test_a_running_command_shows_as_the_command_phase_until_it_completes(self, session, hub, transport):
        session.SPINNER_TICK_SECONDS = 0.01
        handle = hub.send_command_notification("Reading x", source=transport.source, conversation=CONV)
        threading.Timer(0.06, hub.complete_command_notification, args=(handle, "done after 0.1s")).start()
        run_reply(session, hub, [(0.15, Token("Hi")), RoundEnd(), None])
        progress = transport.drafts[0].progress
        phases = [p.phase for p in progress]
        assert phases[0] == PHASE_COMMANDS and progress[0].commands == 1
        assert PHASE_WAITING in phases and phases.index(PHASE_WAITING) > 0
        assert all(p == PHASE_WAITING for p in phases[phases.index(PHASE_WAITING):])
        # The wait for the LLM is counted from the command's end, not from the request's start.
        assert progress[-1].elapsed < 0.1

    def test_commands_of_another_conversation_do_not_show(self, session, hub, transport):
        session.SPINNER_TICK_SECONDS = 0.01
        handle = hub.send_command_notification("Reading x", source=transport.source, conversation=CONV + 1)
        run_reply(session, hub, [(0.03, Token("Hi")), RoundEnd(), None])
        assert {p.phase for p in transport.drafts[0].progress} == {PHASE_WAITING}
        hub.complete_command_notification(handle, "done")

    def test_progress_resets_per_round(self, session, hub, transport):
        session.SPINNER_TICK_SECONDS = 0.01
        run_reply(session, hub, [Thinking("a", 100), Token("One"), RoundEnd(),
                                 (0.05, Thinking("b", 5)), Token("Two"), RoundEnd(), None])
        assert transport.drafts[0].progress[0].tick == 1
        second = transport.drafts[1].progress
        assert second[0].tick == 1
        # The idle tick before the new round's own thinking must not replay round one's
        # (the console would open a second thinking block with the old text).
        assert second[0].phase == PHASE_WAITING and second[0].thinking_text == "" and second[0].output_tokens == 0
        assert second[-1].thinking_text == "b" and second[-1].output_tokens == 5


@pytest.mark.unit
class TestMessageSplitting:
    def test_long_reply_is_split_across_bubbles_while_streaming(self, session, hub, transport):
        transport.max_message_length = 5
        run_reply(session, hub, [Token("abcdefgh"), RoundEnd(), None])
        assert transport.finalized == ["abcde", "fgh"]

    def test_one_huge_chunk_is_cut_into_full_bubbles(self, session, hub, transport):
        transport.max_message_length = 10
        run_reply(session, hub, [Token("aaaa "), Token("bbbb cccc" + "x" * 12), Done("")])
        assert transport.finalized == ["aaaa bbbb ", "ccccxxxxxx", "xxxxxx"]

    def test_unlimited_length_never_splits(self, session, hub, transport):
        transport.max_message_length = 0
        run_reply(session, hub, [Token("x" * 10000), RoundEnd(), None])
        assert transport.finalized == ["x" * 10000]


@pytest.mark.unit
class TestSuppressNo:
    @pytest.mark.parametrize("no_chunk", ["NO", '"NO"', "no", " NO ", "'NO'", "NO.", "No!", '"No."'])
    def test_bare_no_is_suppressed(self, session, hub, transport, no_chunk):
        run_reply(session, hub, [Token(no_chunk), RoundEnd(), None], suppress_no=True)
        assert transport.finalized == []
        assert transport.texts == []
        assert transport.drafts[0].discarded

    def test_suppressed_no_after_progress_discards_the_draft(self, session, hub, transport):
        run_reply(session, hub, [Thinking("…"), Token("NO"), RoundEnd(), None], suppress_no=True)
        assert transport.drafts[0].discarded and transport.finalized == []

    def test_suppressed_no_without_round_end_discards(self, session, hub, transport):
        run_reply(session, hub, [Thinking("…"), Token("No."), None], suppress_no=True)
        assert transport.drafts[0].discarded and transport.finalized == []

    def test_no_split_across_chunks_is_suppressed(self, session, hub, transport):
        run_reply(session, hub, [Token("N"), Token("O"), RoundEnd(), None], suppress_no=True)
        assert transport.texts == []

    def test_real_reply_starting_like_no_is_sent(self, session, hub, transport):
        run_reply(session, hub, [Token("No"), Token(", good point"), RoundEnd(), None], suppress_no=True)
        assert transport.finalized == ["No, good point"]
        assert "No" not in transport.texts[:-1] or transport.texts[0] == "No, good point"

    def test_no_is_sent_when_suppress_disabled(self, session, hub, transport):
        run_reply(session, hub, [Token("NO"), RoundEnd(), None], suppress_no=False)
        assert transport.finalized == ["NO"]

    def test_helpers(self):
        assert ChatSession._is_no_reply("NO") and ChatSession._is_no_reply('"No."') and ChatSession._is_no_reply(" no ")
        assert not ChatSession._is_no_reply("Nope") and not ChatSession._is_no_reply("No, thanks")
        assert ChatSession._is_no_prefix("") and ChatSession._is_no_prefix("N") and ChatSession._is_no_prefix('"no')
        assert ChatSession._is_no_prefix("No.") and ChatSession._is_no_prefix("NO!")
        assert not ChatSession._is_no_prefix("Nope") and not ChatSession._is_no_prefix("No way")


@pytest.mark.unit
class TestStatusLine:
    def test_cleared_when_reply_starts_and_at_the_end(self, session, hub, transport):
        run_reply(session, hub, [Token("Hello there"), None])
        assert transport.status_calls(CONV) == [("status", CONV, None), ("status", CONV, None)]
        first_clear = transport.calls.index(("status", CONV, None))
        first_text = next(i for i, c in enumerate(transport.calls) if c[0] == "draft:update")
        assert first_clear < first_text

    def test_cleared_even_without_a_reply(self, session, hub, transport):
        run_reply(session, hub, [None])
        assert transport.status_calls(CONV) == [("status", CONV, None)]


@pytest.mark.unit
class TestAssistantBridge:
    def test_request_context_carries_sender_channel_and_images(self, session, hub, transport):
        transport.source = "Telegram"
        images = [{"data": "b64", "media_type": "image/jpeg"}]
        assistant = run_reply(session, hub, [Done("")], query="what is this", sender="Anna", images=images)
        req, query = assistant.requests[0]
        assert query == "what is this"
        assert req.source == "Telegram" and req.user == "Anna" and req.lang == "en"
        assert req.text_only is True and req.server_sound is False
        assert req.images == images
        assert req.conversation == CONV  # command echoes find their way back here

    def test_no_images_means_none(self, session, hub, transport):
        assistant = run_reply(session, hub, [Done("")], images=[])
        assert assistant.requests[0][0].images is None

    def test_busy_assistant_apologises(self, session, hub, transport):
        run_reply(session, hub, [], busy="Server busy processing another request")
        assert len(transport.finalized) == 1 and "busy" in transport.finalized[0]

    def test_missing_assistant_apologises(self, session, hub, transport):
        hub.assistant = None
        session._reply(_Request(), "hi", "Anna", None, False)
        assert "not available" in transport.finalized[0]

    def test_assistant_exception_apologises(self, session, hub, transport):
        run_reply(session, hub, [], raise_exc=RuntimeError("boom"))
        assert "something went wrong" in transport.finalized[0]

    def test_cancel_after_attach_kills_the_stream(self):
        request = _Request()
        stream = FakeStream([Token("a")])
        assert request.attach(stream) is True
        request.cancel()
        assert stream.killed is True

    def test_cancel_before_attach_kills_on_attach(self):
        request = _Request()
        request.cancel()
        stream = FakeStream([Token("a")])
        assert request.attach(stream) is False
        assert stream.killed is True

    def test_cancelling_mid_stream_discards_the_draft(self, session, hub, transport):
        session.SPINNER_TICK_SECONDS = 0.01
        assistant = FakeAssistant([Token("Hel"), (5.0, Token("lo")), RoundEnd(), None])
        hub.assistant = assistant
        request = _Request()
        import threading
        worker = threading.Thread(target=session._reply, args=(request, "hi", "Anna", None, False), daemon=True)
        worker.start()
        assert wait_until(lambda: transport.texts == ["Hel"])
        request.cancel()
        worker.join(timeout=3)
        assert not worker.is_alive()
        assert assistant.stream_obj.killed
        assert transport.drafts[0].discarded
        assert transport.finalized == []
        assert not any(t[3] == "assistant" for t in hub.transcript)


@pytest.mark.unit
class TestIntake:
    def _submit_and_wait(self, session, hub, transport, message, events=(Token("Ok"), RoundEnd(), None)):
        assistant = FakeAssistant(events)
        hub.assistant = assistant
        session.submit(message)
        assert wait_until(lambda: assistant.requests and not session.busy)
        return assistant.requests[0]

    def test_text_goes_to_the_assistant_as_is(self, session, hub, transport):
        req, query = self._submit_and_wait(session, hub, transport, msg(text="hello"))
        assert query == "hello" and req.user == "Anna"
        assert hub.transcript[0][3:] == ("user", "hello")

    def test_debug_command_is_answered_without_the_assistant(self, session, hub, transport):
        assistant = FakeAssistant([Token("Ok"), RoundEnd(), None])
        hub.assistant = assistant
        session.submit(msg(text="//help"))
        assert wait_until(lambda: transport.sent_texts)
        assert transport.sent_texts[0].startswith("<pre>Debug commands:") and "//WIPE" in transport.sent_texts[0]
        assert assistant.requests == [] and not session.busy
        assert hub.transcript[0][3:] == ("user", "//help")
        assert hub.transcript[1][3] == "assistant" and hub.transcript[1][4].startswith("Debug commands:")

    def test_reply_context_is_quoted(self, session, hub, transport):
        _, query = self._submit_and_wait(session, hub, transport, msg(text="my reply", reply_to_text="Previous message"))
        assert query == '[Replying to: "Previous message"]\nmy reply'

    def test_long_quote_truncated_to_80_chars(self):
        assert ChatSession._with_reply_context("x" * 120, "reply") == f'[Replying to: "{"x" * 80}..."]\nreply'
        assert ChatSession._with_reply_context("", "hello") == "hello"
        assert ChatSession._with_reply_context(None, "hello") == "hello"

    def test_reaction_with_known_message(self, session, hub, transport):
        _, query = self._submit_and_wait(session, hub, transport,
                                         msg(kind=KIND_REACTION, text="", reaction="👍", reaction_target_text="Hello world"))
        assert '👍' in query and 'Hello world' in query and 'reply with "NO"' in query

    def test_reaction_with_unknown_message_and_truncation(self):
        query = ChatSession._reaction_query(msg(kind=KIND_REACTION, reaction="👎"))
        assert '👎' in query and 'to a message' in query
        query = ChatSession._reaction_query(msg(kind=KIND_REACTION, reaction="👍", reaction_target_text="x" * 120))
        assert "x" * 40 + "..." in query and "x" * 41 not in query

    def test_reaction_no_reply_is_suppressed(self, session, hub, transport):
        self._submit_and_wait(session, hub, transport,
                              msg(kind=KIND_REACTION, text="", reaction="👍", reaction_target_text="Hello"),
                              events=[Token("NO"), RoundEnd(), None])
        assert transport.finalized == []

    def test_voice_is_transcribed(self, session, hub, transport):
        with patch('interfaces.chat.MediaIntake.transcribe_voice', return_value="Hello world") as stt:
            _, query = self._submit_and_wait(session, hub, transport,
                                             msg(kind=KIND_VOICE, text="",
                                                 attachments=[Attachment(b'\x00\x01\x02', "audio/ogg", "voice")]))
        stt.assert_called_once_with(b'\x00\x01\x02')
        assert query == "[Voice message, transcribed via STT] Hello world"

    def test_message_arriving_during_transcription_supersedes_the_voice_note(self, session, hub, transport):
        # STT is a slow network call made before dispatch; a superseded voice note must never
        # reach the assistant after the user has already typed something new.
        assistant = FakeAssistant([Token("Ok"), RoundEnd(), None])
        hub.assistant = assistant
        entered, release = threading.Event(), threading.Event()

        def slow_stt(data):
            entered.set()
            release.wait(3)
            return "hi"

        with patch('interfaces.chat.MediaIntake.transcribe_voice', side_effect=slow_stt):
            session.submit(msg(kind=KIND_VOICE, text="", attachments=[Attachment(b'\x00', "audio/ogg", "voice")]))
            assert entered.wait(3)
            session.submit(msg(text="never mind"))
            assert wait_until(lambda: assistant.requests)
            release.set()
            assert wait_until(lambda: not session.busy)
        assert [q for _, q in assistant.requests] == ["never mind"]
        assert transport.finalized == ["Ok"]
        # Both submissions must announce their end, or the console feed keeps a turn in flight forever
        assert wait_until(lambda: transport.calls.count(("reply_finished", CONV)) == 2)

    def test_voice_transcription_failure_tells_the_user(self, session, hub, transport):
        hub.assistant = FakeAssistant([])
        with patch('interfaces.chat.MediaIntake.transcribe_voice', return_value=""):
            session.submit(msg(kind=KIND_VOICE, text="", attachments=[Attachment(b'\x00', "audio/ogg", "voice")]))
            assert wait_until(lambda: transport.sent_texts)
        assert transport.sent_texts == ["Sorry, I couldn't transcribe the voice message."]
        assert hub.assistant.requests == []
        assert wait_until(lambda: ("reply_finished", CONV) in transport.calls)

    def test_video_is_described_then_sent_with_caption(self, session, hub, transport):
        with patch('interfaces.chat.MediaIntake.describe_video', return_value="Video description: [A dog] A dog runs.") as va:
            _, query = self._submit_and_wait(session, hub, transport,
                                             msg(kind=KIND_VIDEO, text="What happens here?",
                                                 attachments=[Attachment(b"mp4", "video/mp4", "video")]))
        args = va.call_args[0]
        assert args[0] == b"mp4" and args[1] == "video/mp4" and "sent via Fake by Anna" in args[2]
        assert query == "[User sent a video] Video description: [A dog] A dog runs.\nCaption: What happens here?"

    @pytest.mark.parametrize("kind,label", [("video_note", "video note"), ("animation", "GIF")])
    def test_video_note_and_gif_labels(self, session, hub, transport, kind, label):
        with patch('interfaces.chat.MediaIntake.describe_video', return_value="desc"):
            _, query = self._submit_and_wait(session, hub, transport,
                                             msg(kind=KIND_VIDEO, text="", attachments=[Attachment(b"mp4", "video/mp4", kind)]))
        assert query == f"[User sent a {label}] desc"

    def test_video_analysis_failure_tells_the_user(self, session, hub, transport):
        hub.assistant = FakeAssistant([])
        with patch('interfaces.chat.MediaIntake.describe_video', return_value=""):
            session.submit(msg(kind=KIND_VIDEO, text="", attachments=[Attachment(b"mp4", "video/mp4", "video")]))
            assert wait_until(lambda: transport.sent_texts)
        assert transport.sent_texts == ["Sorry, I couldn't analyze that video."]
        assert hub.assistant.requests == []


@pytest.mark.unit
class TestAlbums:
    def _photo(self, data=b'\xff\xd8', caption="", group_id=None):
        return msg(kind=KIND_PHOTO, text=caption, group_id=group_id,
                   attachments=[Attachment(data, "image/jpeg", "photo")])

    def test_grouped_photos_become_one_request(self, session, hub, transport):
        session.ALBUM_DEBOUNCE_SECONDS = 0.05
        assistant = FakeAssistant([Token("Two"), RoundEnd(), None])
        hub.assistant = assistant
        session.submit(self._photo(b"one", caption="How many?", group_id="g1"))
        session.submit(self._photo(b"two", group_id="g1"))
        assert wait_until(lambda: assistant.requests)
        req, query = assistant.requests[0]
        assert query == "How many?"
        assert [i["data"] for i in req.images] == [base64.b64encode(b"one").decode(), base64.b64encode(b"two").decode()]
        assert req.images[0]["media_type"] == "image/jpeg"
        assert len(assistant.requests) == 1

    def test_captions_are_concatenated_and_default_to_see_images(self, session, hub, transport):
        session.ALBUM_DEBOUNCE_SECONDS = 0.02
        assistant = FakeAssistant([None])
        hub.assistant = assistant
        session.submit(self._photo(caption="First caption", group_id="g"))
        session.submit(self._photo(caption="Second caption", group_id="g"))
        assert wait_until(lambda: assistant.requests)
        assert assistant.requests[0][1] == "First caption Second caption"

        assistant = FakeAssistant([None])
        hub.assistant = assistant
        session.submit(self._photo())
        assert wait_until(lambda: assistant.requests)
        assert assistant.requests[0][1] == "See images"

    def test_text_is_folded_into_a_pending_album(self, session, hub, transport):
        session.PHOTO_DEBOUNCE_SECONDS = 0.05
        assistant = FakeAssistant([None])
        hub.assistant = assistant
        session.submit(self._photo(caption="My caption"))
        session.submit(msg(text="Extra context"))
        assert wait_until(lambda: assistant.requests)
        assert assistant.requests[0][1] == "My caption Extra context"
        assert len(assistant.requests) == 1

    def test_voice_cancels_a_pending_album(self, session, hub, transport):
        session.ALBUM_DEBOUNCE_SECONDS = 0.2
        assistant = FakeAssistant([None])
        hub.assistant = assistant
        session.submit(self._photo(group_id="g"))
        with patch('interfaces.chat.MediaIntake.transcribe_voice', return_value="hi"):
            session.submit(msg(kind=KIND_VOICE, text="", attachments=[Attachment(b'\x00', "audio/ogg", "voice")]))
            assert wait_until(lambda: assistant.requests)
        assert session._album is None
        assert wait_until(lambda: not session.busy)
        assert [q for _, q in assistant.requests] == ["[Voice message, transcribed via STT] hi"]

    def test_superseded_album_never_dispatches_a_second_request(self, session, hub, transport):
        session.ALBUM_DEBOUNCE_SECONDS = 0.05
        assistant = FakeAssistant([None])
        hub.assistant = assistant
        session.submit(self._photo(group_id="g"))
        session.submit(msg(kind=KIND_VIDEO, text="", attachments=[]))  # any non-text kind drops the batch
        assert wait_until(lambda: transport.sent_texts)  # the video path apologised, i.e. it ran
        time.sleep(0.15)  # past the album debounce: a stale timer would fire here
        assert assistant.requests == []
        assert session._album is None

    def test_album_waits_for_announced_photos_still_downloading(self, session, hub, transport):
        session.ALBUM_DEBOUNCE_SECONDS = 0.02
        assistant = FakeAssistant([None])
        hub.assistant = assistant
        session.expect_photo("Anna", "g")
        session.expect_photo("Anna", "g")
        session.submit(self._photo(b"one", group_id="g"))
        time.sleep(0.15)  # many debounces later the batch still holds for the slow member
        assert assistant.requests == []
        session.submit(self._photo(b"two", group_id="g"))
        assert wait_until(lambda: assistant.requests)
        assert len(assistant.requests[0][0].images) == 2
        assert len(assistant.requests) == 1

    def test_text_arriving_during_the_download_becomes_the_caption(self, session, hub, transport):
        session.PHOTO_DEBOUNCE_SECONDS = 0.02
        assistant = FakeAssistant([None])
        hub.assistant = assistant
        session.expect_photo("Anna")
        session.submit(msg(text="What is this?"))
        time.sleep(0.1)
        assert assistant.requests == []  # not a text request of its own
        session.submit(self._photo(b"one"))
        assert wait_until(lambda: assistant.requests)
        req, query = assistant.requests[0]
        assert query == "What is this?" and len(req.images) == 1
        assert len(assistant.requests) == 1

    def test_dropped_download_releases_the_album(self, session, hub, transport):
        session.ALBUM_DEBOUNCE_SECONDS = 0.02
        assistant = FakeAssistant([None])
        hub.assistant = assistant
        session.expect_photo("Anna", "g")
        session.expect_photo("Anna", "g")
        session.submit(self._photo(b"one", group_id="g"))
        session.photo_dropped()
        assert wait_until(lambda: assistant.requests)
        assert len(assistant.requests[0][0].images) == 1

    def test_stuck_download_stops_holding_the_album(self, session, hub, transport):
        session.ALBUM_DEBOUNCE_SECONDS = 0.02
        session.ALBUM_DOWNLOAD_TIMEOUT_SECONDS = 0.1
        assistant = FakeAssistant([None])
        hub.assistant = assistant
        session.expect_photo("Anna", "g")
        session.expect_photo("Anna", "g")
        session.submit(self._photo(b"one", group_id="g"))
        assert wait_until(lambda: assistant.requests)
        assert len(assistant.requests[0][0].images) == 1
        assert session._album is None

    def test_new_message_supersedes_the_running_reply(self, session, hub, transport):
        session.SPINNER_TICK_SECONDS = 0.01
        first = FakeAssistant([Token("Hel"), (5.0, Token("lo")), RoundEnd(), None])
        hub.assistant = first
        session.submit(msg(text="first"))
        assert wait_until(lambda: transport.texts == ["Hel"])
        second = FakeAssistant([Token("Second"), RoundEnd(), None])
        hub.assistant = second
        session.submit(msg(text="second"))
        assert wait_until(lambda: "Second" in transport.finalized)
        assert first.stream_obj.killed
        assert wait_until(lambda: transport.drafts[0].discarded)

    def test_a_superseded_reply_finishing_late_leaves_the_current_one_in_charge(self, session, hub, transport):
        # The first reply's worker exits after the second has taken over; if it cleared the
        # slot, the third message would not cancel the second and two replies would stream
        # into the chat at once.
        session.SPINNER_TICK_SECONDS = 0.01
        first = FakeAssistant([Token("One"), (5.0, Token("…")), RoundEnd(), None])
        hub.assistant = first
        session.submit(msg(text="first"))
        assert wait_until(lambda: transport.texts == ["One"])
        second = FakeAssistant([Token("Two"), (5.0, Token("…")), RoundEnd(), None])
        hub.assistant = second
        session.submit(msg(text="second"))
        assert wait_until(lambda: transport.calls.count(("reply_finished", CONV)) == 1)  # first wound down
        assert wait_until(lambda: "Two" in transport.texts)
        assert session.busy

        third = FakeAssistant([Token("Three"), RoundEnd(), None])
        hub.assistant = third
        session.submit(msg(text="third"))
        assert wait_until(lambda: "Three" in transport.finalized)
        assert second.stream_obj.killed
