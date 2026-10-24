import queue
import threading
import time
from unittest.mock import ANY, Mock, patch

import httpx
import pytest
from anthropic import BadRequestError

from assistant.AssistantRequest import RequestContext
from assistant.RequestProcessor import RequestProcessor, RequestKilledException, RequestData
from assistant.AssistantEvents import AudioChunk, CommandRan, Done, Notice, RoundEnd, Thinking, Token
from assistant.prompt.PromptManager import PromptManager
from assistant.tools.MessageFormatter import MessageFormatter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BILLING_MESSAGE = "Your credit balance is too low to access the Anthropic API. Please go to Plans & Billing to upgrade or purchase credits."
BILLING_NOTICE = f"The Claude server returned an error: {BILLING_MESSAGE}"


def _billing_error() -> BadRequestError:
    """The 2026-09-08 prod failure: Anthropic's 400 for an empty credit balance."""
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return BadRequestError("Error code: 400 - {...}", response=httpx.Response(400, request=request),
                           body={"type": "error", "error": {"type": "invalid_request_error", "message": BILLING_MESSAGE}})


@pytest.fixture
def handler():
    h = RequestProcessor.__new__(RequestProcessor)
    h.is_killed = False
    h.events = queue.SimpleQueue()
    h._tts_events = queue.SimpleQueue()
    h.turn_gate = None
    h.text_stream = None
    h.playback = None
    h.voice_factory = None
    h.my_thread_name = "test"
    h.per_request_prefix = False
    h.request_context = RequestContext()
    h._rounds = []
    h._silent_end_followed_up = False
    h._cap_hit_followed_up = False
    h._cut_round_result = None
    h.dialog_manager = Mock()
    h.dialog_manager.reply_to_llm = "User: hello how are you"
    h.dialog_manager.reply_text = "User: hello how are you"
    h.dialog_manager.time_to_first_token = 0.1
    h.dialog_manager.kill = Mock()
    return h


@pytest.fixture
def mock_tui_console():
    """A Mock observer subscribed to the hub (the conftest fixture clears it afterwards)."""
    from assistant.AssistantEvents import observers
    observer = Mock()
    observers().subscribe(observer)
    return observer


# ---------------------------------------------------------------------------
# TestRemoveThinking
# ---------------------------------------------------------------------------

class TestRemoveThinking:
    def test_simple_block(self):
        result = RequestProcessor._remove_thinking("<thinking>x</thinking>Hello")
        assert result == "Hello"

    def test_multiple_blocks(self):
        result = RequestProcessor._remove_thinking(
            "<thinking>a</thinking>Hi<thinking>b</thinking> there"
        )
        assert result == "Hi there"

    def test_spaced_tags(self):
        result = RequestProcessor._remove_thinking("< thinking >x</thinking>Hello")
        assert result == "Hello"

    def test_multiline_content(self):
        result = RequestProcessor._remove_thinking(
            "<thinking>\nline1\nline2\n</thinking>Hello"
        )
        assert result == "Hello"




# ---------------------------------------------------------------------------
# TestKill
# ---------------------------------------------------------------------------

class TestKill:
    def test_calls_dialog_manager_kill(self, handler):
        handler.kill()
        handler.dialog_manager.kill.assert_called_once()

    def test_stops_playback_when_present(self, handler):
        handler.playback = Mock()
        handler.kill()
        handler.playback.stop.assert_called_once()

    def test_no_playback_is_fine(self, handler):
        handler.playback = None
        handler.kill()  # should not raise

    def test_idempotent(self, handler):
        handler.kill()
        handler.dialog_manager.kill.reset_mock()
        handler.kill()
        handler.dialog_manager.kill.assert_not_called()



# ---------------------------------------------------------------------------
# TestHandleBackgroundCommands
# ---------------------------------------------------------------------------

class TestHandleBackgroundCommands:

    @pytest.fixture
    def handler(self):
        h = RequestProcessor.__new__(RequestProcessor)
        h.is_killed = False
        h.events = queue.SimpleQueue()
        h._tts_events = queue.SimpleQueue()
        h.turn_gate = None
        h.text_stream = None
        h.audio_queue = queue.Queue()
        h._audio_stream_lock = threading.Lock()
        h.audio_stream = None
        h.audio_thread = None
        h._chunks_played = 0
        h.my_thread_name = "test"
        h.per_request_prefix = False
        h._rounds = []
        h._silent_end_followed_up = False
        h._cap_hit_followed_up = False
        h._cut_round_result = None
        h.request_context = RequestContext(
            user_input="hello",
            user="User",
        )
        h.first_sound_chunk = True
        h.voice_factory = None
        return h

    @staticmethod
    def _make_mock_dm(command_running=False, command_reply=None, queue_empty=True):
        dm = Mock()
        dm.command_processor = Mock()
        dm.command_processor.command_running_in_background = command_running
        dm.command_processor.command_reply = command_reply
        dm.command_processor.command_queue = Mock()
        dm.command_processor.command_queue.empty.return_value = queue_empty
        # Explicit falsy values: Mock auto-attributes are truthy and would route every
        # test into the error-framing / cache-invalidation branches.
        dm.command_processor.command_reply_contains_error = False
        dm.command_processor.executed_commands = []
        dm.command_processor.unterminated_command_report = None
        dm.command_processor.take_unterminated_command_report.return_value = None
        dm.follow_up = False
        dm.last_response_was_cached = False
        dm.ai = Mock()
        dm.messages = [{"role": "user", "content": "hello"}]
        dm.llm_response_raw = "response text"
        dm.llm_response = "response text"
        dm.reply_to_llm = "User: hello"
        dm.reply_text = "User: hello"
        dm.is_killed = False
        dm.kill = Mock()
        dm.time_to_first_token = 0.1
        return dm

    @staticmethod
    def _exhaust_generator(gen):
        """Exhaust generator, return (list_of_yielded, return_value)."""
        yielded = []
        try:
            while True:
                yielded.append(next(gen))
        except StopIteration as e:
            return yielded, e.value

    def test_no_background_commands_yields_nothing(self, handler):
        """All conditions false → empty generator, returns None"""
        dm = self._make_mock_dm(command_running=False, command_reply=None, queue_empty=True)
        handler.dialog_manager = dm

        prompt_manager = Mock()
        gen = handler._handle_background_commands(prompt_manager)
        yielded, retval = self._exhaust_generator(gen)

        assert yielded == []
        assert retval is None

    def _run_one_round(self, handler, dm):
        """Drive _handle_background_commands through one round with dm as the acting DM."""
        handler.dialog_manager = dm
        with patch.object(handler, 'stream_response', return_value=iter([])), \
             patch.object(handler, '_wait_for_initial_command'), \
             patch.object(handler, '_wait_for_background_command'), \
             patch.object(handler, '_wait_for_command_reply'), \
             patch('assistant.RequestProcessor.DialogManager') as MockDM:
            MockDM.return_value = self._make_mock_dm()
            gen = handler._handle_background_commands(Mock())
            self._exhaust_generator(gen)
        return handler.dialog_manager.reply_to_llm

    @staticmethod
    def _give_unterminated_report(dm, report):
        """Make dm's mock processor behave like the real one for an unterminated command."""
        cp = dm.command_processor
        cp.unterminated_command_report = report

        def take():
            taken, cp.unterminated_command_report = cp.unterminated_command_report, None
            return taken

        def append(text):
            cp.command_reply = text if cp.command_reply is None else f"{cp.command_reply}\n{text}"
            cp.command_reply_contains_error = True
        cp.take_unterminated_command_report.side_effect = take
        cp.append_command_reply.side_effect = append

    def test_a_follow_up_round_tells_its_fresh_llms_the_prefix_is_this_requests_own(self, handler):
        """A round builds a fresh DialogManager and so fresh LLM instances: without the
        request's prepend verdict they would write their blocks at the long tier no later
        request reads (the 1h writes the 5m-tier change removed came back on every round)."""
        for prepend in (True, False):
            handler.per_request_prefix = prepend
            self._run_one_round(handler, self._make_mock_dm(command_reply="result"))
            new_dm = handler.dialog_manager
            for ai in (new_dm.ai_fast, new_dm.ai_smart, new_dm.ai_vision):
                if prepend:
                    ai.set_per_request_prefix.assert_called_once_with(True)
                else:
                    ai.set_per_request_prefix.assert_not_called()

    def test_unterminated_command_alone_circles_back_as_an_error_report(self, handler):
        """The 2026-09-17 prod shape: nothing ran, nothing replied — only the report."""
        dm = self._make_mock_dm()
        self._give_unterminated_report(dm, "browse_eval was not run: ...")

        reply = self._run_one_round(handler, dm)

        assert "browse_eval was not run" in reply
        assert "No other commands were executed." in reply

    def test_unterminated_command_joins_the_reply_of_a_command_that_did_run(self, handler):
        dm = self._make_mock_dm(command_reply="page snapshot")
        dm.command_processor.executed_commands = ["browse_open(https://example.com)"]
        self._give_unterminated_report(dm, "browse_eval was not run: ...")

        reply = self._run_one_round(handler, dm)

        assert "page snapshot" in reply
        assert "browse_eval was not run" in reply
        assert "browse_open(https://example.com)" in reply

    def test_unterminated_command_reports_are_capped_per_request(self, handler):
        """An LLM that keeps truncating the same call must not loop forever."""
        rounds = []

        def next_dm(*args, **kwargs):
            dm = self._make_mock_dm()
            self._give_unterminated_report(dm, "browse_eval was not run: ...")
            rounds.append(dm)
            return dm

        first = self._make_mock_dm()
        self._give_unterminated_report(first, "browse_eval was not run: ...")
        handler.dialog_manager = first
        with patch.object(handler, 'stream_response', side_effect=lambda: iter([])), \
             patch.object(handler, '_wait_for_initial_command'), \
             patch.object(handler, '_wait_for_background_command'), \
             patch.object(handler, '_wait_for_command_reply'), \
             patch('assistant.RequestProcessor.DialogManager', side_effect=next_dm):
            self._exhaust_generator(handler._handle_background_commands(Mock()))

        assert len(rounds) == handler.MAX_UNTERMINATED_COMMAND_REPORTS

    def _run_repeating_rounds(self, handler, reply="The setup toolkit was loaded at 11:04: use it from there.",
                              commands=('load_toolkit("setup")',), replies=None):
        """Drive the loop with every round running `commands` and getting `reply` (or the next
        of `replies`) back, until the guard ends it; returns (round dms, yielded events)."""
        rounds = []

        def make_dm():
            text = replies[len(rounds)] if replies else reply
            dm = self._make_mock_dm(command_reply=text)
            dm.command_processor.executed_commands = list(commands)
            rounds.append(dm)
            if len(rounds) > 10:
                raise AssertionError("the loop did not end")
            return dm

        handler.dialog_manager = make_dm()
        with patch.object(handler, 'stream_response', side_effect=lambda: iter([])), \
             patch.object(handler, '_wait_for_initial_command'), \
             patch.object(handler, '_wait_for_background_command'), \
             patch.object(handler, '_wait_for_command_reply'), \
             patch('assistant.RequestProcessor.DialogManager', side_effect=lambda *a, **k: make_dm()):
            yielded, _ = self._exhaust_generator(handler._handle_background_commands(Mock()))
        return rounds, yielded

    def test_a_round_repeating_the_previous_one_is_told_so_under_the_result(self, handler):
        rounds, _ = self._run_repeating_rounds(handler)
        first, second = rounds[1].reply_to_llm, rounds[2].reply_to_llm
        assert first.startswith("result: \n") and "same" not in first, "the first time the result stands alone"
        assert "use it from there" in second and "same" in second

    def test_the_third_identical_round_ends_the_request(self, handler):
        """ds4 at effort low reloaded a toolkit the window already held four rounds running
        over the same 102-char answer, until Ctrl-C (2026-10-22)."""
        rounds, yielded = self._run_repeating_rounds(handler)
        assert len(rounds) == 1 + handler.MAX_REPEATED_ROUNDS + 1, "two repeats stream, the third round never does"
        assert handler.dialog_manager is rounds[-1]
        notices = [e for e in yielded if isinstance(e, Notice)]
        assert len(notices) == 1 and isinstance(yielded[-1], RoundEnd)

    def test_a_different_result_is_not_a_repeat(self, handler):
        """The same command with a changing answer (a page refreshed, a log read on) is work."""
        replies = ["line 1", "line 2", "line 3", "line 4", "line 5", "line 5", "line 5", "line 5"]
        rounds, _ = self._run_repeating_rounds(handler, commands=("server_logs()",), replies=replies)
        assert len(rounds) == 8, "the three identical tails end it, the changing ones did not"
        assert "same" not in rounds[5].reply_to_llm, "the first 'line 5' is a change from 'line 4'"
        assert "same" in rounds[6].reply_to_llm

    def test_an_image_reply_is_never_a_repeat(self, handler):
        rounds = []

        def make_dm():
            dm = self._make_mock_dm(command_reply=[{"type": "image", "source": {}}] if rounds else None)
            dm.command_processor.executed_commands = ["fetch_camera_images()"] if rounds else []
            rounds.append(dm)
            if len(rounds) > 4:
                dm.command_processor.command_reply = None
                dm.command_processor.executed_commands = []
            return dm

        first = self._make_mock_dm(command_reply=[{"type": "image", "source": {}}])
        first.command_processor.executed_commands = ["fetch_camera_images()"]
        rounds.append(first)
        handler.dialog_manager = first
        with patch.object(handler, 'stream_response', side_effect=lambda: iter([])), \
             patch.object(handler, '_wait_for_initial_command'), \
             patch.object(handler, '_wait_for_background_command'), \
             patch.object(handler, '_wait_for_command_reply'), \
             patch('assistant.RequestProcessor.DialogManager', side_effect=lambda *a, **k: make_dm()):
            yielded, _ = self._exhaust_generator(handler._handle_background_commands(Mock()))
        assert not [e for e in yielded if isinstance(e, Notice)]
        assert len(rounds) == 5, "every image round streamed; the loop ended on the silent round"

    def test_error_reply_gets_error_framing(self, handler):
        """A command-error round is framed as an informational report — failures, what
        already ran, and that the previous reply was already spoken — not 'result:'."""
        error_text = "Command 'turn_on(Mysbelysning (Kök))' returned an error: \"could not find device &quot;Mysbelysning (Kök)&quot;\"."
        dm = self._make_mock_dm(command_reply=error_text)
        dm.command_processor.command_reply_contains_error = True
        dm.command_processor.executed_commands = ["dim(Spots kök (Kök), 1)"]

        reply = self._run_one_round(handler, dm)

        assert error_text in reply
        assert "already spoken to the user" in reply
        assert "These commands already executed successfully: dim(Spots kök (Kök), 1)." in reply
        assert not reply.startswith("result: \n")

    def test_error_framing_with_no_succeeded_commands(self, handler):
        dm = self._make_mock_dm(command_reply="Command 'turn_on(x)' returned an error: \"boom\".")
        dm.command_processor.command_reply_contains_error = True

        reply = self._run_one_round(handler, dm)

        assert "No other commands were executed." in reply

    def test_list_reply_with_error_gets_preamble_block(self, handler):
        """List replies (e.g. camera snapshot + appended error) get one instruction text
        block prepended; the original blocks stay intact after it."""
        blocks = [{"type": "text", "text": "Command 'x()' returned an error: \"nope\"."},
                  {"type": "image", "source": {"data": "abc", "media_type": "image/png"}}]
        dm = self._make_mock_dm(command_reply=blocks)
        dm.command_processor.command_reply_contains_error = True

        reply = self._run_one_round(handler, dm)

        assert isinstance(reply, list)
        assert len(reply) == 3
        assert "already spoken to the user" in reply[0]["text"]
        assert reply[1:] == blocks

    def test_follow_up_round_switches_to_vision_after_the_reply_is_set_and_before_streaming(self, handler):
        """use_vision_for_image_reply reads reply_to_llm to pick the tier and strips the
        <thinking>-tag instruction from the carried messages; it is only worth anything between
        the reply landing and the follow-up stream. The unit tests of the switch itself call it
        directly, so this ordering was pinned by nothing."""
        blocks = [{"type": "image", "source": {"type": "base64", "data": "abc"}}]
        dm = self._make_mock_dm(command_reply=blocks)
        follow_up_dm = self._make_mock_dm()
        order = []
        follow_up_dm.use_vision_for_image_reply.side_effect = lambda: order.append(("vision", follow_up_dm.reply_to_llm))

        handler.dialog_manager = dm
        with patch.object(handler, 'stream_response', side_effect=lambda: order.append(("stream", None)) or iter([])), \
             patch.object(handler, '_wait_for_initial_command'), \
             patch.object(handler, '_wait_for_background_command'), \
             patch.object(handler, '_wait_for_command_reply'), \
             patch('assistant.RequestProcessor.DialogManager', return_value=follow_up_dm):
            self._exhaust_generator(handler._handle_background_commands(Mock()))

        assert order == [("vision", blocks), ("stream", None)]

    def test_background_reply_keeps_result_framing(self, handler):
        """Non-error background replies (google/web results) keep the plain framing so
        the model still produces a full spoken reply."""
        dm = self._make_mock_dm(command_reply="search result here")

        reply = self._run_one_round(handler, dm)

        assert reply == "result: \nsearch result here"

    def test_cached_response_with_missing_device_invalidates_cache(self, handler):
        from assistant.llm.CachedLLM import CachedLLM
        error_text = "Command 'turn_on(Mysbelysning (Kök))' returned an error: \"could not find device &quot;Mysbelysning (Kök)&quot;\"."
        dm = self._make_mock_dm(command_reply=error_text)
        dm.command_processor.command_reply_contains_error = True
        dm.last_response_was_cached = True
        dm.ai = Mock(spec=CachedLLM)

        self._run_one_round(handler, dm)

        dm.ai.invalidate_served_response.assert_called_once()

    def test_no_cache_invalidation_when_not_cached(self, handler):
        from assistant.llm.CachedLLM import CachedLLM
        dm = self._make_mock_dm(command_reply="Command 'x()' returned an error: \"could not find device &quot;y&quot;\".")
        dm.command_processor.command_reply_contains_error = True
        dm.ai = Mock(spec=CachedLLM)

        self._run_one_round(handler, dm)

        dm.ai.invalidate_served_response.assert_not_called()

    def test_no_cache_invalidation_for_other_errors(self, handler):
        from assistant.llm.CachedLLM import CachedLLM
        dm = self._make_mock_dm(command_reply="Command 'x()' returned an error: \"timeout talking to Indigo\".")
        dm.command_processor.command_reply_contains_error = True
        dm.last_response_was_cached = True
        dm.ai = Mock(spec=CachedLLM)

        self._run_one_round(handler, dm)

        dm.ai.invalidate_served_response.assert_not_called()

    def test_cache_invalidation_survives_non_cachedllm_ai(self, handler):
        """test_suite_mode skips the CachedLLM wrapper — the isinstance guard must keep
        the round alive instead of crashing."""
        dm = self._make_mock_dm(command_reply="could not find device &quot;y&quot;")
        dm.command_processor.command_reply_contains_error = True
        dm.last_response_was_cached = True
        dm.ai = Mock(spec=[])

        reply = self._run_one_round(handler, dm)

        assert "already spoken to the user" in reply

    def test_silent_command_completion_breaks(self, handler):
        """bg completes but reply=None, queue empty → breaks, stream_response NOT called"""
        dm = self._make_mock_dm(command_running=True, command_reply=None, queue_empty=True)
        # Simulate: command was running, now it's done
        dm.command_processor.command_running_in_background = False
        # reply is None and queue is empty - should break

        # But the while condition needs command_running_in_background=True initially
        call_count = [0]
        def running_side_effect():
            call_count[0] += 1
            return call_count[0] <= 1

        dm.command_processor.__class__ = type(dm.command_processor)
        type(dm.command_processor).command_running_in_background = property(
            lambda self: running_side_effect() if call_count[0] == 0 else False
        )

        # Simpler approach: just set initial state that enters the loop
        dm2 = Mock()
        dm2.command_processor = Mock()
        dm2.command_processor.command_reply = None
        dm2.command_processor.command_queue = Mock()
        dm2.command_processor.command_reply_contains_error = False
        dm2.command_processor.executed_commands = []
        dm2.last_response_was_cached = False
        dm2.ai = Mock()
        dm2.messages = [{"role": "user", "content": "hello"}]
        dm2.llm_response_raw = "response text"
        dm2.llm_response = "response text"
        dm2.reply_to_llm = "User: hello"
        dm2.is_killed = False

        # Use a list to track calls for command_running_in_background
        bg_states = iter([True, False])  # First call: True (enter loop), second call from _wait: False
        dm2.command_processor.command_running_in_background = True
        dm2.command_processor.command_queue.empty.return_value = True

        handler.dialog_manager = dm2

        with patch.object(handler, 'stream_response') as mock_stream, \
             patch.object(handler, '_wait_for_background_command') as mock_wait_bg:
            def on_wait_bg(old_dm):
                # Simulate: bg command done, no reply
                old_dm.command_processor.command_running_in_background = False
            mock_wait_bg.side_effect = on_wait_bg

            prompt_manager = Mock()
            gen = handler._handle_background_commands(prompt_manager)
            yielded, retval = self._exhaust_generator(gen)

        mock_stream.assert_not_called()
        assert retval is None

    def test_stream_error_in_follow_up_round_yields_friendly_message(self, handler):
        """An exception in a follow-up round must not escape (the response body is
        already open by then — an escaping exception severs the socket). It should
        yield the same apology as the first-round handler in generate_audio."""
        dm = self._make_mock_dm(command_running=False, command_reply="search result", queue_empty=True)
        handler.dialog_manager = dm
        handler.request_context.text_only = True

        with patch.object(handler, 'stream_response',
                          side_effect=RuntimeError("RemoteProtocolError: incomplete chunked read")), \
             patch.object(handler, '_wait_for_initial_command'), \
             patch.object(handler, '_wait_for_background_command'), \
             patch.object(handler, '_wait_for_command_reply'), \
             patch('assistant.RequestProcessor.DialogManager') as MockDM:
            MockDM.return_value = self._make_mock_dm()

            gen = handler._handle_background_commands(Mock())
            yielded, retval = self._exhaust_generator(gen)

        assert any(isinstance(chunk, Notice) and "incomplete chunked read" in chunk.text for chunk in yielded)
        assert yielded[-1] == RoundEnd()
        assert retval == "hello"

    def test_server_error_in_follow_up_round_names_the_server_and_its_message(self, handler):
        dm = self._make_mock_dm(command_running=False, command_reply="search result", queue_empty=True)
        handler.dialog_manager = dm
        handler.request_context.text_only = True
        follow_up_dm = self._make_mock_dm()
        follow_up_dm.ai.provider.return_value = "Claude"

        with patch.object(handler, 'stream_response', side_effect=_billing_error()), \
             patch.object(handler, '_wait_for_initial_command'), \
             patch.object(handler, '_wait_for_background_command'), \
             patch.object(handler, '_wait_for_command_reply'), \
             patch('assistant.RequestProcessor.DialogManager') as MockDM:
            MockDM.return_value = follow_up_dm

            gen = handler._handle_background_commands(Mock())
            yielded, retval = self._exhaust_generator(gen)

        assert Notice(BILLING_NOTICE) in yielded
        assert yielded[-1] == RoundEnd()
        assert retval == "hello"

    def test_kill_in_follow_up_round_propagates(self, handler):
        """RequestKilledException must NOT be swallowed by the follow-up round handler."""
        dm = self._make_mock_dm(command_running=False, command_reply="search result", queue_empty=True)
        handler.dialog_manager = dm

        with patch.object(handler, 'stream_response', side_effect=RequestKilledException), \
             patch.object(handler, '_wait_for_initial_command'), \
             patch.object(handler, '_wait_for_background_command'), \
             patch.object(handler, '_wait_for_command_reply'), \
             patch('assistant.RequestProcessor.DialogManager') as MockDM:
            MockDM.return_value = self._make_mock_dm()

            gen = handler._handle_background_commands(Mock())
            with pytest.raises(RequestKilledException):
                self._exhaust_generator(gen)

    def test_multiple_background_command_loops(self, handler):
        """Two bg commands in sequence → loop iterates twice, stream_response called twice"""
        dm1 = self._make_mock_dm(command_running=True, command_reply="reply1", queue_empty=True)
        handler.dialog_manager = dm1

        loop_count = [0]

        dm2 = self._make_mock_dm(command_running=True, command_reply="reply2", queue_empty=True)
        dm3 = self._make_mock_dm(command_running=False, command_reply=None, queue_empty=True)

        new_dms = iter([dm2, dm3])

        with patch.object(handler, 'stream_response', return_value=iter([])) as mock_stream, \
             patch.object(handler, '_wait_for_initial_command') as mock_wait_init, \
             patch.object(handler, '_wait_for_background_command'), \
             patch.object(handler, '_wait_for_command_reply'), \
             patch('assistant.RequestProcessor.DialogManager') as MockDM:
            def create_new_dm(*args, **kwargs):
                loop_count[0] += 1
                return next(new_dms)
            MockDM.side_effect = create_new_dm

            prompt_manager = Mock()
            gen = handler._handle_background_commands(prompt_manager)
            yielded, retval = self._exhaust_generator(gen)

        assert mock_stream.call_count == 2

    def test_user_query_recorded_in_history_only_once_across_loops(self, handler):
        """Multi-round command chains must record the user query in history only on
        the first circle-back — later rounds append just the assistant command turn,
        otherwise the same user message duplicates in every subsequent prompt."""
        dm1 = self._make_mock_dm(command_running=True, command_reply="reply1", queue_empty=True)
        handler.dialog_manager = dm1

        dm2 = self._make_mock_dm(command_running=True, command_reply="reply2", queue_empty=True)
        dm3 = self._make_mock_dm(command_running=False, command_reply=None, queue_empty=True)
        new_dms = iter([dm2, dm3])

        with patch.object(handler, 'stream_response', return_value=iter([])), \
             patch.object(handler, '_wait_for_initial_command'), \
             patch.object(handler, '_wait_for_background_command'), \
             patch.object(handler, '_wait_for_command_reply'), \
             patch('assistant.RequestProcessor.DialogManager', side_effect=lambda *a, **k: next(new_dms)):
            prompt_manager = Mock()
            gen = handler._handle_background_commands(prompt_manager)
            yielded, retval = self._exhaust_generator(gen)

        calls = prompt_manager.update_history.call_args_list
        assert len(calls) == 2
        assert calls[0].args[1] == "hello"
        assert calls[1].args[0] is None and calls[1].args[1] is None
        assert retval == "hello"

    def test_mid_round_history_updates_disable_pre_heat(self, handler):
        """Both mid-round update_history call sites (first circle-back and
        query-already-recorded) must pass pre_heat=False: the follow-up round's
        inference starts immediately after and would race a pre-heat for Ollama's
        single slot, clobbering its prefill state. Only the terminal
        _update_history after the loop — pinned in TestUpdateHistory — keeps the
        default and warms the full conversation prefix."""
        dm1 = self._make_mock_dm(command_running=True, command_reply="reply1", queue_empty=True)
        handler.dialog_manager = dm1

        dm2 = self._make_mock_dm(command_running=True, command_reply="reply2", queue_empty=True)
        dm3 = self._make_mock_dm(command_running=False, command_reply=None, queue_empty=True)
        new_dms = iter([dm2, dm3])

        with patch.object(handler, 'stream_response', return_value=iter([])), \
             patch.object(handler, '_wait_for_initial_command'), \
             patch.object(handler, '_wait_for_background_command'), \
             patch.object(handler, '_wait_for_command_reply'), \
             patch('assistant.RequestProcessor.DialogManager', side_effect=lambda *a, **k: next(new_dms)):
            prompt_manager = Mock()
            gen = handler._handle_background_commands(prompt_manager)
            self._exhaust_generator(gen)

        calls = prompt_manager.update_history.call_args_list
        assert len(calls) == 2
        for call in calls:
            assert call.kwargs.get("pre_heat") is False

    def test_a_follow_up_round_persists_the_user_row_with_the_request_stamp(self, handler):
        """A command round persists the exchange mid-request; on a byte-prefix request that
        row must carry the time and clock note the query was sent with, or the follow-up
        round re-renders a different tail than the one the runner just cached."""
        handler.turn_timestamp = 1_700_000_000.0
        handler.time_note = "(Current time: 14:40, Saturday, 2025-03-15)"
        dm1 = self._make_mock_dm(command_running=True, command_reply="reply1", queue_empty=True)
        handler.dialog_manager = dm1
        dm2 = self._make_mock_dm(command_running=False, command_reply=None, queue_empty=True)

        with patch.object(handler, 'stream_response', return_value=iter([])), \
             patch.object(handler, '_wait_for_initial_command'), \
             patch.object(handler, '_wait_for_background_command'), \
             patch.object(handler, '_wait_for_command_reply'), \
             patch('assistant.RequestProcessor.DialogManager', return_value=dm2):
            prompt_manager = Mock()
            self._exhaust_generator(handler._handle_background_commands(prompt_manager))

        first = prompt_manager.update_history.call_args_list[0]
        assert first.args[1] == handler.request_context.user_input
        assert first.kwargs["timestamp"] == handler.turn_timestamp
        assert first.kwargs["time_note"] == handler.time_note

    def test_the_chat_layers_facts_stand_in_the_note_under_the_clock(self, handler):
        # A guest's or a group's facts are the system's words: they go in the note row, never
        # inside the person's own text, and they ride along even when no clock note is sent.
        prompt_manager = Mock()
        prompt_manager.current_time_note.return_value = "(Current time: 14:40, Saturday, 2025-03-15)"
        handler.request_context.note = "The next message is from Dave, who is not one of the household's owners."
        note = handler._turn_note(prompt_manager, byte_prefix_tail=True)
        assert note.split("\n") == ["(Current time: 14:40, Saturday, 2025-03-15)",
                                    "The next message is from Dave, who is not one of the household's owners."]
        assert handler._turn_note(prompt_manager, byte_prefix_tail=False).endswith("owners.")
        handler.request_context.note = None
        assert handler._turn_note(prompt_manager, byte_prefix_tail=False) is None

    def test_round_end_carries_the_rounds_raw_text(self, handler):
        """RoundEnd(raw) closes each follow-up round (the wire renderer turns it into \\x1e)"""
        handler.request_context.text_only = True
        dm = self._make_mock_dm(command_running=False, command_reply="reply", queue_empty=True)
        handler.dialog_manager = dm

        new_dm = self._make_mock_dm(command_running=False, command_reply=None, queue_empty=True)

        with patch.object(handler, 'stream_response', return_value=iter(["audio"])), \
             patch.object(handler, '_wait_for_initial_command'), \
             patch.object(handler, '_wait_for_background_command'), \
             patch.object(handler, '_wait_for_command_reply'), \
             patch('assistant.RequestProcessor.DialogManager', return_value=new_dm):
            prompt_manager = Mock()
            gen = handler._handle_background_commands(prompt_manager)
            yielded, retval = self._exhaust_generator(gen)

        assert RoundEnd("response text") in yielded

    def test_kill_raises_exception(self, handler):
        """is_killed=True before entering → RequestKilledException raised"""
        from assistant.RequestProcessor import RequestKilledException

        dm = self._make_mock_dm(command_running=True, command_reply=None, queue_empty=True)
        handler.dialog_manager = dm
        handler.is_killed = True

        prompt_manager = Mock()
        gen = handler._handle_background_commands(prompt_manager)
        with pytest.raises(RequestKilledException):
            next(gen)

    def test_messages_copied_to_new_dm(self, handler):
        """Old DM's messages list copied to new DM, reply_from_role set to 'user'"""
        original_messages = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
        dm = self._make_mock_dm(command_running=False, command_reply="reply", queue_empty=True)
        dm.messages = original_messages
        handler.dialog_manager = dm

        new_dm = self._make_mock_dm(command_running=False, command_reply=None, queue_empty=True)

        with patch.object(handler, 'stream_response', return_value=iter([])), \
             patch.object(handler, '_wait_for_initial_command'), \
             patch.object(handler, '_wait_for_background_command'), \
             patch.object(handler, '_wait_for_command_reply'), \
             patch('assistant.RequestProcessor.DialogManager', return_value=new_dm):
            prompt_manager = Mock()
            gen = handler._handle_background_commands(prompt_manager)
            self._exhaust_generator(gen)

        assert new_dm.messages == original_messages
        assert new_dm.reply_from_role == "user"

    def test_request_id_carried_to_new_dm(self, handler):
        """The follow-up DialogManager spun up after a command reply must inherit
        the original request_id, otherwise the /stream_text endpoint stops
        receiving tokens mid-conversation (each TextStreamRegistry queue is
        keyed by request_id)."""
        dm = self._make_mock_dm(command_running=False, command_reply="reply", queue_empty=True)
        dm.request_id = "req-abc-123"
        handler.dialog_manager = dm

        new_dm = self._make_mock_dm(command_running=False, command_reply=None, queue_empty=True)
        new_dm.request_id = None  # DialogManager.__init__ leaves this None

        with patch.object(handler, 'stream_response', return_value=iter([])), \
             patch.object(handler, '_wait_for_initial_command'), \
             patch.object(handler, '_wait_for_background_command'), \
             patch.object(handler, '_wait_for_command_reply'), \
             patch('assistant.RequestProcessor.DialogManager', return_value=new_dm):
            prompt_manager = Mock()
            gen = handler._handle_background_commands(prompt_manager)
            self._exhaust_generator(gen)

        assert new_dm.request_id == "req-abc-123"

    @pytest.mark.parametrize("tail", [True, False])
    def test_byte_prefix_tail_carried_to_new_dm(self, handler, tail):
        """The command follow-up round is the same byte-prefix request: a fresh DialogManager
        starts without the flag, and folding the directives onto the result turn there would
        diverge from the history before the runner's restore point."""
        dm = self._make_mock_dm(command_running=False, command_reply="reply", queue_empty=True)
        dm.byte_prefix_tail = tail
        handler.dialog_manager = dm

        new_dm = self._make_mock_dm(command_running=False, command_reply=None, queue_empty=True)
        new_dm.byte_prefix_tail = not tail

        with patch.object(handler, 'stream_response', return_value=iter([])), \
             patch.object(handler, '_wait_for_initial_command'), \
             patch.object(handler, '_wait_for_background_command'), \
             patch.object(handler, '_wait_for_command_reply'), \
             patch('assistant.RequestProcessor.DialogManager', return_value=new_dm):
            gen = handler._handle_background_commands(Mock())
            self._exhaust_generator(gen)

        assert new_dm.byte_prefix_tail is tail

    # --- a request that would end with nothing said to the user ---

    SELF_EVIDENT = {"turn_off", "pause"}

    def _silent_dm(self, *executed):
        dm = self._make_mock_dm()
        dm.llm_response = ""
        dm.command_processor.executed_commands = list(executed)
        dm.command_processor.commands.is_self_evident.side_effect = self.SELF_EVIDENT.__contains__
        return dm

    def _end_request(self, handler, first, follow_up=None):
        """Drive the end of a request whose rounds so far are [first]; returns the rounds
        streamed after it."""
        handler.dialog_manager = first
        handler._rounds = [first]
        streamed = []
        follow_up = follow_up or self._make_mock_dm()
        with patch.object(handler, 'stream_response', side_effect=lambda: iter(streamed.append(handler.dialog_manager) or [])), \
             patch.object(handler, '_wait_for_initial_command'), \
             patch.object(handler, '_wait_for_background_command'), \
             patch.object(handler, '_wait_for_command_reply'), \
             patch.object(handler, '_new_dialog_manager', return_value=follow_up):
            self._exhaust_generator(handler._handle_background_commands(Mock()))
        return streamed

    def test_a_request_ending_silently_on_bookkeeping_gets_one_round_to_reply(self, handler):
        streamed = self._end_request(handler, self._silent_dm("open_url(https://example.com)", "checked_watch(news)"))
        assert len(streamed) == 1
        assert RequestProcessor.SILENT_END_NOTE in streamed[0].reply_to_llm

    def test_a_request_ending_silently_on_actions_whose_effect_is_the_answer_stays_silent(self, handler):
        assert self._end_request(handler, self._silent_dm("turn_off(Hall)", "pause(Kitchen)")) == []

    def test_one_bookkeeping_command_beside_the_actions_still_gets_the_round(self, handler):
        assert len(self._end_request(handler, self._silent_dm("turn_off(Hall)", "remember(x)"))) == 1

    def test_a_round_with_no_command_after_earlier_ones_gets_the_round(self, handler):
        earlier = self._silent_dm("open_url(https://example.com)")
        last = self._silent_dm()
        handler._rounds = [earlier]
        handler.dialog_manager = last
        handler._rounds.append(last)
        with patch.object(handler, 'stream_response', return_value=iter([])) as stream, \
             patch.object(handler, '_wait_for_initial_command'), \
             patch.object(handler, '_wait_for_background_command'), \
             patch.object(handler, '_wait_for_command_reply'), \
             patch.object(handler, '_new_dialog_manager', return_value=self._make_mock_dm()):
            self._exhaust_generator(handler._handle_background_commands(Mock()))
        stream.assert_called_once()

    def test_a_request_that_said_something_gets_no_extra_round(self, handler):
        dm = self._silent_dm("checked_watch(news)")
        dm.llm_response = "Nothing new on the rumour front."
        assert self._end_request(handler, dm) == []

    def test_a_silent_reply_without_any_command_is_left_silent(self, handler):
        assert self._end_request(handler, self._silent_dm()) == []

    def test_the_extra_round_is_given_once_even_when_it_is_silent_too(self, handler):
        assert len(self._end_request(handler, self._silent_dm("save_watch(a)"),
                                     follow_up=self._silent_dm("save_watch(b)"))) == 1

    def test_a_test_suite_request_gets_no_extra_round(self, handler):
        handler.request_context.test = True
        assert self._end_request(handler, self._silent_dm("checked_watch(news)")) == []

    def test_a_wake_word_misfire_is_not_answered(self, handler):
        """false_positive() is the LLM saying the input was not meant for it: the request ends
        there, without a round that would speak a reply to gibberish."""
        from assistant.commands.Commands import Commands
        with patch.object(Commands, '_mail_access', return_value=Mock()):
            resolver = Commands(Mock(), command_sets=[]).is_self_evident
        dm = self._silent_dm("false_positive()")
        dm.command_processor.commands.is_self_evident.side_effect = resolver
        assert self._end_request(handler, dm) == []

    # --- a round the LLM ended at its output cap ---

    def _cut_dm(self, said="", cap=16000):
        """A round ended at max_tokens: `said` is what reached the user before the cut (nothing
        when the thinking alone used the cap). The note goes through CommandProcessor.add_note,
        which composes it into the reply the loop reads."""
        dm = self._make_mock_dm()
        dm.llm_response = dm.llm_response_raw = said
        dm.ai.last_stop_reason = 'max_tokens'
        dm.ai.output_cap.return_value = cap
        dm.command_processor.add_note.side_effect = lambda text: setattr(dm.command_processor, 'command_reply', text)
        return dm

    def _drive_from(self, handler, first, follow_up=None):
        """The request's first round is `first` and has just been streamed; returns the rounds
        streamed after it and every event yielded while they ran."""
        handler.dialog_manager = first
        handler._rounds = [first]
        handler._cap_hit()
        streamed = []
        follow_up = follow_up or self._make_mock_dm()
        with patch.object(handler, 'stream_response', side_effect=lambda: iter(streamed.append(handler.dialog_manager) or [])), \
             patch.object(handler, '_wait_for_initial_command'), \
             patch.object(handler, '_wait_for_background_command'), \
             patch.object(handler, '_wait_for_command_reply'), \
             patch.object(handler, '_new_dialog_manager', return_value=follow_up):
            yielded, _ = self._exhaust_generator(handler._handle_background_commands(Mock()))
        return streamed, yielded

    def test_a_round_that_thought_through_the_cap_circles_back_told_so(self, handler):
        streamed, _ = self._drive_from(handler, self._cut_dm())
        assert len(streamed) == 1
        note = streamed[0].reply_to_llm
        assert "16000-token" in note and "nothing reached the user" in note

    def test_a_reply_cut_mid_text_circles_back_with_where_it_stopped(self, handler):
        streamed, _ = self._drive_from(handler, self._cut_dm("The plan:\n- six automations, one per lamp, bui"))
        assert len(streamed) == 1
        note = streamed[0].reply_to_llm
        assert "one per lamp, bui" in note and "saw the text" in note
        assert "nothing reached the user" not in note

    def test_a_backend_without_a_fixed_cap_is_named_without_a_number(self, handler):
        streamed, _ = self._drive_from(handler, self._cut_dm(cap=None))
        assert "-token" not in streamed[0].reply_to_llm

    def test_a_second_cut_ends_the_request_with_a_notice_to_the_user(self, handler):
        streamed, yielded = self._drive_from(handler, self._cut_dm(), follow_up=self._cut_dm())
        assert len(streamed) == 1
        notices = [e.text for e in yielded if isinstance(e, Notice)]
        assert notices == [RequestProcessor.CAP_HIT_NOTICE]
        assert any(isinstance(e, RoundEnd) for e in yielded)

    def test_a_second_cut_waits_for_the_rounds_commands_and_keeps_their_result(self, handler):
        """The round that overran the cap had already dispatched a command: the request ends
        only once it finished, and its result follows the round's text into the history (the
        next turn would otherwise not know it ran). No third LLM round."""
        first = self._cut_dm()
        cut_again = self._cut_dm()
        cut_again.command_processor.command_reply = "Lamp Marcus is off."
        cut_again.command_processor.executed_commands = ["home_control(lamp, off)"]
        handler.dialog_manager = first
        handler._rounds = [first]
        handler._cap_hit()
        waited, streamed = [], []
        with patch.object(handler, 'stream_response', side_effect=lambda: iter(streamed.append(handler.dialog_manager) or [])), \
             patch.object(handler, '_wait_for_initial_command'), \
             patch.object(handler, '_wait_for_background_command', side_effect=waited.append), \
             patch.object(handler, '_wait_for_command_reply'), \
             patch.object(handler, '_new_dialog_manager', return_value=cut_again):
            self._exhaust_generator(handler._handle_background_commands(Mock()))
        assert streamed == [cut_again]
        assert waited[-1] is cut_again
        handler.request_context = RequestContext(user_input="turn the lamp off", user="Marcus")
        prompt_manager = Mock()
        handler._update_history(prompt_manager, "turn the lamp off")
        names = [c[0] for c in prompt_manager.mock_calls]
        assert names.index('update_history') < names.index('add_command_result')
        assert "Lamp Marcus is off." in prompt_manager.add_command_result.call_args.args[0]

    def test_a_second_cut_round_without_a_result_adds_no_command_row(self, handler):
        handler.dialog_manager = first = self._cut_dm()
        handler._rounds = [first]
        handler._cap_hit()
        with patch.object(handler, 'stream_response', return_value=iter([])), \
             patch.object(handler, '_wait_for_initial_command'), \
             patch.object(handler, '_wait_for_background_command'), \
             patch.object(handler, '_wait_for_command_reply'), \
             patch.object(handler, '_new_dialog_manager', return_value=self._cut_dm()):
            self._exhaust_generator(handler._handle_background_commands(Mock()))
        prompt_manager = Mock()
        handler._update_history(prompt_manager, "hello")
        prompt_manager.add_command_result.assert_not_called()

    def test_a_round_that_ended_on_its_own_is_not_told_anything(self, handler):
        dm = self._silent_dm()
        dm.ai.last_stop_reason = 'end_turn'
        streamed, _ = self._drive_from(handler, dm)
        assert streamed == []
        dm.command_processor.add_note.assert_not_called()

    def test_the_note_stands_in_history_as_the_rounds_result(self, handler):
        first = self._cut_dm()
        handler.dialog_manager = first
        handler._rounds = [first]
        handler._cap_hit()
        prompt_manager = Mock()
        with patch.object(handler, 'stream_response', return_value=iter([])), \
             patch.object(handler, '_wait_for_initial_command'), \
             patch.object(handler, '_wait_for_background_command'), \
             patch.object(handler, '_wait_for_command_reply'), \
             patch.object(handler, '_new_dialog_manager', return_value=self._make_mock_dm()):
            self._exhaust_generator(handler._handle_background_commands(prompt_manager))
        stored = prompt_manager.add_command_result.call_args.args[0]
        assert "nothing reached the user" in stored


# ---------------------------------------------------------------------------
# Extended handler fixture for new tests
# ---------------------------------------------------------------------------

@pytest.fixture
def full_handler():
    """Handler with all fields needed for the new test classes."""
    h = RequestProcessor.__new__(RequestProcessor)
    h.is_killed = False
    h.events = queue.SimpleQueue()
    h._tts_events = queue.SimpleQueue()
    h.turn_gate = None
    h.text_stream = None
    h.playback = Mock()
    h.voice_factory = None
    h.playback.is_running.return_value = False
    h.my_thread_name = "test"
    h.first_sound_chunk = True
    h.is_performing_request = False
    h.request_context = RequestContext(
        source="VoiceToVoice",
        location="Kitchen",
        user="TestUser",
        user_input="hello",
    )
    h.dialog_manager = Mock()
    h.dialog_manager.reply_to_llm = "User: hello how are you"
    h.dialog_manager.reply_text = "User: hello how are you"
    h.dialog_manager.time_to_first_token = 0.1
    h.dialog_manager.kill = Mock()
    h.dialog_manager.llm_response = "Hi there!"
    h.dialog_manager.llm_response_raw = "Hi there!"
    h.dialog_manager.first_token_got_at = time.time()
    h.dialog_manager.reply_token_got_at = time.time()
    h.dialog_manager.last_response_was_cached = False
    h.dialog_manager.ai = Mock()
    h.dialog_manager.ai.name.return_value = "TestAI"
    h.dialog_manager.ai.supports_native_thinking.return_value = False
    h.dialog_manager.ai.prefers_byte_prefix_tail.return_value = False
    h.request_data = RequestData("GET", "/api", {"query": "hello"}, {"User-Agent": "test"})
    h._cut_round_result = None
    return h



# ---------------------------------------------------------------------------
# TestInitializeAudioPlayback
# ---------------------------------------------------------------------------


class TestInitializeAudioPlayback:
    def test_server_sound_false_does_not_start_playback(self, full_handler):
        full_handler.request_context.server_sound = False
        full_handler.initialize_audio_playback()
        full_handler.playback.ensure_started.assert_not_called()

    def test_server_sound_true_starts_playback(self, full_handler):
        full_handler.request_context.server_sound = True
        full_handler.initialize_audio_playback()
        full_handler.playback.ensure_started.assert_called_once()

    def test_no_playback_in_process_is_noop(self, full_handler):
        full_handler.request_context.server_sound = True
        full_handler.playback = None
        full_handler.initialize_audio_playback()  # should not raise


# ---------------------------------------------------------------------------
# TestDeliverAudioChunk
# ---------------------------------------------------------------------------

class TestDeliverAudioChunk:
    def test_server_sound_enqueues_and_returns_none(self, full_handler):
        full_handler.request_context.server_sound = True
        result = full_handler._deliver_audio_chunk(b"audio_data")
        assert result is None
        full_handler.playback.enqueue.assert_called_once_with(b"audio_data")

    def test_no_server_sound_returns_chunk(self, full_handler):
        full_handler.request_context.server_sound = False
        result = full_handler._deliver_audio_chunk(b"audio_data")
        assert result == b"audio_data"
        full_handler.playback.enqueue.assert_not_called()

    def test_server_sound_without_playback_streams_chunk(self, full_handler):
        """A process without a voice interface can't play locally — fall back to streaming."""
        full_handler.request_context.server_sound = True
        full_handler.playback = None
        assert full_handler._deliver_audio_chunk(b"audio_data") == b"audio_data"


# ---------------------------------------------------------------------------
# TestStreamResponse
# ---------------------------------------------------------------------------

class TestStreamResponse:
    def test_no_voice_synthesizer_streams_text_even_when_not_text_only(self, full_handler):
        """A process without a voice interface (voice_factory=None) leaves
        dialog_manager.voice None; a client asking for audio then gets the reply as text
        instead of a crash on voice.start()."""
        full_handler.request_context.text_only = False
        full_handler.request_context.source = "VoiceToVoice"
        full_handler.dialog_manager.voice = None
        full_handler.dialog_manager.stream_llm_response.return_value = iter(["tok"])
        full_handler.dialog_manager.llm_response = "cleaned"
        full_handler.dialog_manager.llm_response_raw = "raw"

        result = list(full_handler.stream_response())
        assert result == [Token("tok")]

    def test_text_only_streams_tokens_for_telegram(self, full_handler):
        full_handler.request_context.text_only = True
        full_handler.request_context.source = "Telegram"
        full_handler.dialog_manager.stream_llm_response.return_value = iter(["token1", "token2"])
        full_handler.dialog_manager.llm_response = "full response"
        full_handler.dialog_manager.llm_response_raw = "raw response"

        result = list(full_handler.stream_response())
        assert result == [Token("token1"), Token("token2")]

    def test_text_only_yields_tokens_for_every_source(self, full_handler):
        """Swallowing tokens for VoiceToVoice text mode is the wire renderer's job now; the
        raw text travels in RoundEnd from the caller, never in the round itself."""
        full_handler.request_context.text_only = True
        full_handler.request_context.source = "VoiceToVoice"
        full_handler.dialog_manager.stream_llm_response.return_value = iter(["tok"])
        full_handler.dialog_manager.llm_response = "cleaned"
        full_handler.dialog_manager.llm_response_raw = "raw with <thinking>x</thinking>stuff"

        result = list(full_handler.stream_response())
        assert result == [Token("tok")]

    def test_side_events_are_interleaved_with_tokens(self, full_handler):
        """Thinking / CommandRan raised off the token loop (command processor, command
        worker thread) surface in the stream at the next yield point, in order."""
        full_handler.request_context.text_only = True
        full_handler.request_context.source = "TextChat"

        def llm():
            full_handler.emit(Thinking("hmm", 3))
            yield "tok1"
            full_handler.emit(CommandRan("turn_on", "kitchen"))
            yield "tok2"

        full_handler.dialog_manager.stream_llm_response.return_value = llm()
        full_handler.dialog_manager.llm_response = "cleaned"
        full_handler.dialog_manager.llm_response_raw = "raw"

        result = list(full_handler.stream_response())
        assert result == [Token("tok1"), Thinking("hmm", 3), Token("tok2"), CommandRan("turn_on", "kitchen")]

    def test_events_yielded_by_the_llm_stream_pass_through(self, full_handler):
        full_handler.request_context.text_only = True
        full_handler.dialog_manager.stream_llm_response.return_value = iter([Thinking("a", 1), "tok"])
        full_handler.dialog_manager.llm_response_raw = "raw"
        assert list(full_handler.stream_response()) == [Thinking("a", 1), Token("tok")]

    def test_audio_mode_feeds_only_text_to_the_synthesizer(self, full_handler):
        full_handler.request_context.text_only = False
        full_handler.request_context.server_sound = False
        mock_voice = Mock()
        full_handler.dialog_manager.voice = mock_voice
        full_handler.dialog_manager.stream_llm_response.return_value = iter([Thinking("t", 1), "spoken"])
        full_handler.dialog_manager.first_token_got_at = time.time()
        seen = {}

        def start(text_stream):
            seen["text"] = list(text_stream)
            return iter([b"c1"])

        mock_voice.start.side_effect = start
        with patch.object(full_handler, '_handle_first_audio_chunk'), \
             patch.object(full_handler, '_deliver_audio_chunk', side_effect=lambda c: c):
            result = list(full_handler.stream_response())
        assert seen["text"] == ["spoken"]
        assert result == [AudioChunk(b"c1"), Thinking("t", 1)]

    def test_audio_mode_first_chunk_handler_called_once(self, full_handler):
        full_handler.request_context.text_only = False
        mock_voice = Mock()
        full_handler.dialog_manager.voice = mock_voice
        mock_voice.start.return_value = iter([b"c1", b"c2"])
        full_handler.dialog_manager.first_token_got_at = time.time()

        def handle_first(chunk_time):
            full_handler.first_sound_chunk = False

        with patch.object(full_handler, '_handle_first_audio_chunk', side_effect=handle_first) as mock_first, \
             patch.object(full_handler, '_deliver_audio_chunk', return_value=None):
            list(full_handler.stream_response())
        mock_first.assert_called_once()

    def test_audio_mode_none_chunks_skipped(self, full_handler):
        full_handler.request_context.text_only = False
        mock_voice = Mock()
        full_handler.dialog_manager.voice = mock_voice
        mock_voice.start.return_value = iter([None, b"real", None])
        full_handler.dialog_manager.first_token_got_at = time.time()

        with patch.object(full_handler, '_handle_first_audio_chunk'), \
             patch.object(full_handler, '_deliver_audio_chunk', side_effect=lambda c: c):
            result = list(full_handler.stream_response())
        assert result == [AudioChunk(b"real")]

# ---------------------------------------------------------------------------
# TestTextOnlyOutage
# ---------------------------------------------------------------------------

class TestTextOnlyOutage:
    """_text_only runs on the TTS collect_text thread, so a backend error escaping it
    would die as an unhandled thread exception and end the reply silently (the
    2026-08-27 Gemini 503). It must become a spoken outage notice instead."""

    def _outage_dm(self, full_handler):
        dm = full_handler.dialog_manager
        dm.llm_response = ""
        dm.llm_response_raw = ""
        dm.first_token_got_at = 0
        dm.reply_token_got_at = 0
        return dm

    def test_passes_text_and_routes_events(self, full_handler):
        event = Thinking("hmm")
        out = list(full_handler._text_only(iter(["Hello ", event, "world"])))

        assert out == ["Hello ", "world"]
        # Parked on the side queue, not on `events`: the LLM stream the feeder consumes
        # drains `events`, and putting the event back there spun the feeder thread forever.
        assert full_handler._tts_events.get_nowait() is event
        assert full_handler.events.empty()

    def test_an_event_in_the_stream_it_feeds_from_does_not_loop(self, full_handler):
        """The LLM stream drains `events` between tokens (DialogManager.drain_events).
        Before the side queue, _text_only put each event back there, the stream popped it
        again and the feeder thread spun at 100% until the request's own loop took it —
        never, once the request was killed. The stream must end with the event seen once."""
        event = Thinking("hmm")
        full_handler.emit(event)
        seen = []

        def llm_stream():
            yield "Hello "
            while True:  # DialogManager.drain_events, with the one event ever yielded counted
                try:
                    e = full_handler.events.get_nowait()
                except queue.Empty:
                    break
                seen.append(e)
                yield e
            yield "world"

        out = list(full_handler._text_only(llm_stream()))

        assert out == ["Hello ", "world"]
        assert seen == [event]
        assert list(full_handler._drain_events()) == [event]

    def test_llm_error_becomes_spoken_failure_notice(self, full_handler):
        dm = self._outage_dm(full_handler)
        dm.ai.provider.return_value = "Gemini"
        notice = "The Gemini request failed: 503 UNAVAILABLE"

        def stream():
            yield "Hel"
            raise RuntimeError("503 UNAVAILABLE")

        out = list(full_handler._text_only(stream()))

        assert out == ["Hel", notice]
        dm._surface_notice.assert_called_once_with(notice)
        assert notice in dm.llm_response
        assert RequestProcessor.LLM_OUTAGE_HISTORY_MARKER in dm.llm_response_raw

    def test_server_error_is_spoken_with_the_servers_message(self, full_handler):
        dm = self._outage_dm(full_handler)
        dm.ai.provider.return_value = "Claude"

        def stream():
            raise _billing_error()
            yield

        out = list(full_handler._text_only(stream()))

        assert out == [BILLING_NOTICE]
        dm._surface_notice.assert_called_once_with(BILLING_NOTICE)
        assert dm.llm_response == BILLING_NOTICE
        assert RequestProcessor.LLM_OUTAGE_HISTORY_MARKER in dm.llm_response_raw

    def test_llm_error_before_first_token_backfills_timestamps(self, full_handler):
        dm = self._outage_dm(full_handler)

        def stream():
            raise RuntimeError()
            yield

        out = list(full_handler._text_only(stream()))

        assert out == [RequestProcessor.LLM_OUTAGE_NOTICE]
        assert dm.first_token_got_at > 0
        assert dm.reply_token_got_at == dm.first_token_got_at

    def test_timestamps_kept_when_tokens_already_flowed(self, full_handler):
        dm = self._outage_dm(full_handler)
        dm.first_token_got_at = 123.0
        dm.reply_token_got_at = 124.0

        def stream():
            yield "partial"
            raise RuntimeError("boom")

        list(full_handler._text_only(stream()))

        assert dm.first_token_got_at == 123.0
        assert dm.reply_token_got_at == 124.0

    def test_kill_propagates_without_notice(self, full_handler):
        dm = self._outage_dm(full_handler)

        def stream():
            yield "x"
            raise RequestKilledException()

        gen = full_handler._text_only(stream())
        assert next(gen) == "x"
        with pytest.raises(RequestKilledException):
            next(gen)
        dm._surface_notice.assert_not_called()


# ---------------------------------------------------------------------------
# TestHandleFirstAudioChunk
# ---------------------------------------------------------------------------

class TestHandleFirstAudioChunk:
    def test_wait_returns_false_raises(self, full_handler):
        with patch.object(full_handler, '_wait_for_natural_timing', return_value=False):
            with pytest.raises(RequestKilledException):
                full_handler._handle_first_audio_chunk(0.5)

    def test_sets_first_sound_chunk_false(self, full_handler):
        full_handler.first_sound_chunk = True
        with patch.object(full_handler, '_wait_for_natural_timing', return_value=True):
            full_handler._handle_first_audio_chunk(0.5)
        assert full_handler.first_sound_chunk is False

    def test_checks_killed_after_wait(self, full_handler):
        full_handler.is_killed = True
        with patch.object(full_handler, '_wait_for_natural_timing', return_value=True):
            with pytest.raises(RequestKilledException):
                full_handler._handle_first_audio_chunk(0.5)

    def test_mutes_outposts(self, full_handler):
        with patch.object(full_handler, '_wait_for_natural_timing', return_value=True):
            full_handler._handle_first_audio_chunk(0.5)
        full_handler.playback.ensure_outposts_muted.assert_called_once()

    def test_mutes_outposts_even_when_streaming_to_client(self, full_handler):
        """A phone playing the reply in the same room must not be transcribed by a local
        outpost — the mute is not a server-sound-only concern."""
        full_handler.request_context.server_sound = False
        with patch.object(full_handler, '_wait_for_natural_timing', return_value=True):
            full_handler._handle_first_audio_chunk(0.5)
        full_handler.playback.ensure_outposts_muted.assert_called_once()

    def test_first_sound_chunk_reported_to_observers(self, full_handler, mock_tui_console):
        full_handler.dialog_manager.voice.last_served_from_cache = True
        with patch.object(full_handler, '_wait_for_natural_timing', return_value=True):
            full_handler._handle_first_audio_chunk(0.5)
        mock_tui_console.on_first_sound_chunk.assert_called_once_with(0.5, True)


class TestWaitForNaturalTiming:
    """The voice turn gate is consulted right before the first sound of a reply, on the
    request thread; with no gate installed (no voice interface) the reply speaks at once."""

    def test_gate_called_with_context_reply_and_timings(self, full_handler):
        full_handler.turn_gate = Mock()
        full_handler.turn_gate.wait.return_value = True
        full_handler.dialog_manager.reply_text = "User: hello"
        full_handler.dialog_manager.time_to_first_token = 0.3
        assert full_handler._wait_for_natural_timing(0.5) is True
        full_handler.turn_gate.wait.assert_called_once_with(
            full_handler.request_context, "User: hello", 0.3, 0.5)

    def test_no_gate_is_immediate(self, full_handler):
        full_handler.turn_gate = None
        with patch("assistant.RequestProcessor.time.sleep") as mock_sleep:
            assert full_handler._wait_for_natural_timing(0.5) is True
        mock_sleep.assert_not_called()

    def test_no_gate_still_honours_kill(self, full_handler):
        full_handler.turn_gate = None
        full_handler.is_killed = True
        assert full_handler._wait_for_natural_timing(0.5) is False

    def test_first_chunk_runs_gate_before_mute(self, full_handler):
        order = []
        full_handler.turn_gate = Mock()
        full_handler.turn_gate.wait.side_effect = lambda *a: order.append("gate") or True
        full_handler.playback.ensure_outposts_muted.side_effect = lambda: order.append("mute")
        full_handler._handle_first_audio_chunk(0.5)
        assert order == ["gate", "mute"]


# ---------------------------------------------------------------------------
# TestPrepareContext
# ---------------------------------------------------------------------------

class TestPrepareContext:
    @pytest.fixture(autouse=True)
    def _no_toolkit_preload(self):
        """These tests hand in a bare Mock prompt manager; the toolkit preload (its own
        tests: commands/test_toolkit_preloader.py) is stubbed out of the way."""
        with patch("assistant.RequestProcessor.ToolkitPreloader.preload") as preload:
            yield preload

    def test_voice_request_preloads_toolkits_before_the_messages_are_built(self, full_handler, _no_toolkit_preload):
        prompt_manager = Mock()
        prompt_manager.get_messages.return_value = []
        _no_toolkit_preload.side_effect = \
            lambda pm, profile, extra, is_killed: pm.add_command_result("# Toolkit: home", cap=False)
        full_handler.dialog_manager.ai.supports_native_thinking.return_value = False

        with patch("assistant.RequestProcessor.MessageFormatter") as mock_mf, \
                patch("assistant.RequestProcessor.FirstRun.preload_toolkits", return_value=("setup",)):
            mock_mf.format_message.return_value = "User: hi"
            full_handler._prepare_llm_context(prompt_manager)

        from assistant.ChannelProfile import VOICE_TO_VOICE
        _no_toolkit_preload.assert_called_once_with(prompt_manager, VOICE_TO_VOICE, ("setup",), is_killed=ANY)
        assert _no_toolkit_preload.call_args.kwargs["is_killed"]() is False
        names = [c[0] for c in prompt_manager.mock_calls if c[0] in ("add_command_result", "get_messages")]
        assert names == ["add_command_result", "get_messages"]


    def test_a_request_killed_during_the_preload_stops_before_the_messages_are_built(self, full_handler, _no_toolkit_preload):
        """A speculative partial-transcript request is usually superseded within milliseconds;
        one killed while the preload ran must not go on to assemble a prompt for nothing."""
        prompt_manager = Mock()

        def kill_during_preload(pm, profile, extra, is_killed):
            full_handler.is_killed = True

        _no_toolkit_preload.side_effect = kill_during_preload
        full_handler.dialog_manager.ai.supports_native_thinking.return_value = False

        with patch("assistant.RequestProcessor.MessageFormatter") as mock_mf, \
                patch("assistant.RequestProcessor.FirstRun.preload_toolkits", return_value=()):
            with pytest.raises(RequestKilledException):
                full_handler._prepare_llm_context(prompt_manager)

        prompt_manager.get_messages.assert_not_called()
        mock_mf.format_message.assert_not_called()

    def test_thinking_tag_instruction_for_a_model_without_internal_thinking(self, full_handler):
        """Regression for d45d4718: the <thinking>-tag instruction is unconditional (reworded as
        optional), whatever the input length — for a model that does not think on its own."""
        full_handler.dialog_manager.ai.handles_thinking_internally.return_value = False
        for text in ("hi", "tell me about the weather today please"):
            prompt_manager = Mock()
            prompt_manager.get_messages.return_value = []
            with patch("assistant.RequestProcessor.MessageFormatter") as mock_mf:
                mock_mf.format_message.return_value = f"User: {text}"
                full_handler.request_context.user_input = text
                full_handler._prepare_llm_context(prompt_manager)
            prompt_manager.get_messages.assert_called_once_with(extended_thinking=True, time_note=None, window="voice")
            assert full_handler.dialog_manager.thinking is False

    def test_no_thinking_tag_instruction_for_a_model_that_thinks_on_its_own(self, full_handler):
        """Claude handles thinking internally: the <thinking>-tag instruction would get the
        request refused as reasoning extraction, so it is never sent. Native thinking stays
        off too — these models think adaptively without the parameter."""
        full_handler.dialog_manager.ai.handles_thinking_internally.return_value = True
        prompt_manager = Mock()
        prompt_manager.get_messages.return_value = []

        with patch("assistant.RequestProcessor.MessageFormatter") as mock_mf:
            mock_mf.format_message.return_value = "User: tell me about the weather today please"
            full_handler.request_context.user_input = "tell me about the weather today please"
            full_handler._prepare_llm_context(prompt_manager)

        prompt_manager.get_messages.assert_called_once_with(extended_thinking=False, time_note=None, window="voice")
        assert full_handler.dialog_manager.thinking is False

    @pytest.mark.parametrize("picked_thinks_internally", [True, False])
    def test_thinking_tag_decision_is_for_the_model_select_ai_picks(self, full_handler, picked_thinks_internally):
        """A DialogManager starts on the fast tier and select_ai moves the request to its own
        (a chat request: Gemini -> Claude). The instruction must be decided for the model that
        will answer, not the one the DialogManager happened to hold before the pick."""
        dm = full_handler.dialog_manager
        dm.ai.handles_thinking_internally.return_value = not picked_thinks_internally
        picked = Mock()
        picked.handles_thinking_internally.return_value = picked_thinks_internally
        picked.prefers_byte_prefix_tail.return_value = False
        dm.select_ai.side_effect = lambda ctx: setattr(dm, "ai", picked)
        prompt_manager = Mock()
        prompt_manager.get_messages.return_value = []

        with patch("assistant.RequestProcessor.MessageFormatter") as mock_mf:
            mock_mf.format_message.return_value = "User: hello"
            full_handler._prepare_llm_context(prompt_manager)

        prompt_manager.get_messages.assert_called_once_with(extended_thinking=not picked_thinks_internally,
                                                            time_note=None, window="voice")

    def test_byte_prefix_tail_sends_the_turn_as_the_history_will_render_it(self, full_handler):
        """An LLM whose cache restores only at the previous request's end (Ollama's MLX
        runner) gets no <thinking> instruction, a query header dated like the history row
        the next request renders, and the clock note it saw persisted with the exchange."""
        full_handler.dialog_manager.ai.handles_thinking_internally.return_value = False
        full_handler.dialog_manager.ai.prefers_byte_prefix_tail.return_value = True
        prompt_manager = Mock()
        prompt_manager.get_messages.return_value = []
        prompt_manager.current_time_note.side_effect = PromptManager.current_time_note
        full_handler.request_context.user_input = "what's the plan for today?"

        sent = full_handler._prepare_llm_context(prompt_manager)

        assert full_handler.dialog_manager.byte_prefix_tail is True
        assert isinstance(full_handler.turn_timestamp, float)
        prompt_manager.get_messages.assert_called_once_with(extended_thinking=False,
                                                            time_note=full_handler.time_note, window="voice")
        assert full_handler.time_note == PromptManager.current_time_note(full_handler.turn_timestamp)
        assert full_handler.dialog_manager.reply_to_llm == MessageFormatter.format_message(
            "TestUser", full_handler.turn_timestamp, "VoiceToVoice", "Kitchen", "what's the plan for today?")
        assert MessageFormatter.format_timestamp(full_handler.turn_timestamp) in full_handler.dialog_manager.reply_to_llm
        # The cache key and DB row carry no time, or the same utterance a minute later misses.
        assert sent == full_handler.dialog_manager.cache_query
        assert MessageFormatter.format_timestamp(full_handler.turn_timestamp) not in sent
        assert sent == MessageFormatter.format_message("TestUser", None, "VoiceToVoice", "Kitchen",
                                                       "what's the plan for today?")

        full_handler.dialog_manager.llm_response_raw = "Nothing booked."
        full_handler._update_history(prompt_manager, None)
        prompt_manager.update_history.assert_called_once_with(
            "TestUser", "what's the plan for today?", "Nothing booked.", "VoiceToVoice", "Kitchen",
            timestamp=full_handler.turn_timestamp, time_note=full_handler.time_note)

    def test_the_byte_prefix_switch_off_restores_the_classic_tail(self, full_handler):
        """FLOXY_BYTE_PREFIX_TAIL=0 is the A/B lever: an LLM that asks for a byte-prefix tail
        still gets the classic one — <thinking> instruction, undated query header, nothing
        persisted with the exchange."""
        full_handler.dialog_manager.ai.handles_thinking_internally.return_value = False
        full_handler.dialog_manager.ai.prefers_byte_prefix_tail.return_value = True
        prompt_manager = Mock()
        prompt_manager.get_messages.return_value = []
        full_handler.request_context.user_input = "what's the plan for today?"

        with patch("assistant.RequestProcessor.BYTE_PREFIX_TAIL", False):
            sent = full_handler._prepare_llm_context(prompt_manager)

        assert full_handler.dialog_manager.byte_prefix_tail is False
        assert full_handler.turn_timestamp is None and full_handler.time_note is None
        prompt_manager.get_messages.assert_called_once_with(extended_thinking=True, time_note=None, window="voice")
        assert sent == MessageFormatter.format_message("TestUser", None, "VoiceToVoice", "Kitchen",
                                                       "what's the plan for today?")

    def test_formats_message_with_context(self, full_handler):
        prompt_manager = Mock()
        prompt_manager.get_messages.return_value = []

        with patch("assistant.RequestProcessor.MessageFormatter") as mock_mf:
            mock_mf.format_message.return_value = "TestUser (VoiceToVoice in Kitchen): hello"
            full_handler.request_context.user_input = "hello"
            result = full_handler._prepare_llm_context(prompt_manager)

        mock_mf.format_message.assert_called_once_with(
            "TestUser", None, "VoiceToVoice", "Kitchen", "hello"
        )

    def test_prepends_user_query_to_system_message(self, full_handler):
        prompt_manager = Mock()
        prompt_manager.get_messages.return_value = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]

        with patch("assistant.RequestProcessor.MessageFormatter") as mock_mf, \
             patch("assistant.RequestProcessor.prepend_user_query_override", return_value=True):
            mock_mf.format_message.return_value = "User: what is the weather"
            full_handler.request_context.user_input = "what is the weather"
            full_handler._prepare_llm_context(prompt_manager)

        system_msg = full_handler.dialog_manager.messages[0]
        assert system_msg["role"] == "system"
        assert system_msg["content"].startswith("Just a heads up")
        assert "what is the weather" in system_msg["content"]
        assert "You are a helpful assistant." in system_msg["content"]

    def test_prepending_tells_the_llm_the_prefix_is_this_requests_own(self, full_handler):
        """A prepended query sits at byte 0, so nothing after this request reads what it
        caches; the LLM picks its cache tier by the flag. The cache itself stays on — the
        follow-up rounds of the same request still read the prefix."""
        prompt_manager = Mock()
        prompt_manager.get_messages.return_value = [{"role": "system", "content": "You are a helpful assistant."}]
        for prepend in (True, False):
            with patch("assistant.RequestProcessor.MessageFormatter") as mock_mf, \
                 patch("assistant.RequestProcessor.prepend_user_query_override", return_value=prepend):
                mock_mf.format_message.return_value = "User: hello"
                full_handler.request_context.user_input = "hello"
                full_handler._prepare_llm_context(prompt_manager)
            full_handler.dialog_manager.ai.set_per_request_prefix.assert_called_with(prepend)
        full_handler.dialog_manager.ai.set_prompt_cache.assert_not_called()

    def test_prepend_disabled_leaves_system_message_unchanged(self, full_handler):
        prompt_manager = Mock()
        prompt_manager.get_messages.return_value = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]

        with patch("assistant.RequestProcessor.MessageFormatter") as mock_mf, \
             patch("assistant.RequestProcessor.prepend_user_query_override", return_value=False):
            mock_mf.format_message.return_value = "User: hello"
            full_handler.request_context.user_input = "hello"
            full_handler._prepare_llm_context(prompt_manager)

        system_msg = full_handler.dialog_manager.messages[0]
        assert system_msg["content"] == "You are a helpful assistant."

    def test_prepend_skips_when_no_system_message(self, full_handler):
        prompt_manager = Mock()
        prompt_manager.get_messages.return_value = [
            {"role": "user", "content": "hello"}
        ]

        with patch("assistant.RequestProcessor.MessageFormatter") as mock_mf, \
             patch("assistant.RequestProcessor.prepend_user_query_override", return_value=True):
            mock_mf.format_message.return_value = "User: hello"
            full_handler.request_context.user_input = "hello"
            full_handler._prepare_llm_context(prompt_manager)

        assert full_handler.dialog_manager.messages[0]["content"] == "hello"

    def test_prepend_uses_llm_default_when_override_unset(self, full_handler):
        """Override None => defer to the LLM. An LLM that allows prepending gets it."""
        prompt_manager = Mock()
        prompt_manager.get_messages.return_value = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
        full_handler.dialog_manager.ai.prefers_prepended_user_query.return_value = True

        with patch("assistant.RequestProcessor.MessageFormatter") as mock_mf, \
             patch("assistant.RequestProcessor.prepend_user_query_override", return_value=None):
            mock_mf.format_message.return_value = "User: what is the weather"
            full_handler.request_context.user_input = "what is the weather"
            full_handler._prepare_llm_context(prompt_manager)

        assert full_handler.dialog_manager.messages[0]["content"].startswith("Just a heads up")

    def test_prepend_respects_llm_optout_when_override_unset(self, full_handler):
        """Override None => defer to the LLM. An LLM that opts out (Claude/Ollama)
        is left untouched even though the global override isn't set."""
        prompt_manager = Mock()
        prompt_manager.get_messages.return_value = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
        full_handler.dialog_manager.ai.prefers_prepended_user_query.return_value = False

        with patch("assistant.RequestProcessor.MessageFormatter") as mock_mf, \
             patch("assistant.RequestProcessor.prepend_user_query_override", return_value=None):
            mock_mf.format_message.return_value = "User: hello"
            full_handler.request_context.user_input = "hello"
            full_handler._prepare_llm_context(prompt_manager)

        assert full_handler.dialog_manager.messages[0]["content"] == "You are a helpful assistant."

    def test_override_off_beats_llm_default(self, full_handler):
        """Override False forces prepend OFF even for an LLM that would allow it."""
        prompt_manager = Mock()
        prompt_manager.get_messages.return_value = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
        full_handler.dialog_manager.ai.prefers_prepended_user_query.return_value = True

        with patch("assistant.RequestProcessor.MessageFormatter") as mock_mf, \
             patch("assistant.RequestProcessor.prepend_user_query_override", return_value=False):
            mock_mf.format_message.return_value = "User: hello"
            full_handler.request_context.user_input = "hello"
            full_handler._prepare_llm_context(prompt_manager)

        assert full_handler.dialog_manager.messages[0]["content"] == "You are a helpful assistant."

    def test_images_build_content_blocks(self, full_handler):
        prompt_manager = Mock()
        prompt_manager.get_messages.return_value = []
        full_handler.request_context.images = [
            {"data": "abc123", "media_type": "image/jpeg"}
        ]

        with patch("assistant.RequestProcessor.MessageFormatter") as mock_mf, \
             patch("assistant.RequestProcessor.prepend_user_query_override", return_value=False):
            mock_mf.format_message.return_value = "User: What is this?"
            full_handler.request_context.user_input = "What is this?"
            full_handler._prepare_llm_context(prompt_manager)

        reply = full_handler.dialog_manager.reply_to_llm
        assert isinstance(reply, list)
        assert len(reply) == 2
        assert reply[0]["type"] == "image"
        assert reply[0]["source"]["data"] == "abc123"
        assert reply[1]["type"] == "text"
        assert reply[1]["text"] == "User: What is this?"

# ---------------------------------------------------------------------------
# TestUpdateHistory
# ---------------------------------------------------------------------------

class TestUpdateHistory:
    def test_removes_thinking_tags(self, full_handler):
        full_handler.dialog_manager.llm_response_raw = "<thinking>deep thought</thinking>Actual response"
        prompt_manager = Mock()

        result = full_handler._update_history(prompt_manager, None)
        assert "Actual response" in result
        assert "deep thought" not in result

    def test_calls_update_history_when_not_reported(self, full_handler):
        full_handler.dialog_manager.llm_response_raw = "response"
        prompt_manager = Mock()

        full_handler._update_history(prompt_manager, None)
        prompt_manager.update_history.assert_called_once_with("TestUser", "hello", "response", "VoiceToVoice", "Kitchen",
                                                              timestamp=None, time_note=None)

    def test_calls_update_history_none_when_already_reported(self, full_handler):
        full_handler.dialog_manager.llm_response_raw = "response"
        prompt_manager = Mock()

        full_handler._update_history(prompt_manager, "hello")
        prompt_manager.update_history.assert_called_once_with(None, None, "response")

    def test_skips_when_killed(self, full_handler):
        full_handler.is_killed = True
        full_handler.dialog_manager.llm_response_raw = "response"
        prompt_manager = Mock()

        full_handler._update_history(prompt_manager, None)
        prompt_manager.update_history.assert_not_called()

# ---------------------------------------------------------------------------
# TestSaveToDatabase
# ---------------------------------------------------------------------------

class TestSaveToDatabase:
    def test_normal_saves(self, full_handler):
        with patch("assistant.RequestProcessor.FloxyDatabase") as mock_db_cls:
            mock_db = Mock()
            mock_db_cls.get_instance.return_value = mock_db
            full_handler.dialog_manager.last_response_was_cached = False

            full_handler._save_to_database("query", "response")

            mock_db.submit.assert_called_once_with(
                "query", "", "response", False,
                source="VoiceToVoice", location="Kitchen"
            )

    def test_false_positive_skips_save(self, full_handler):
        with patch("assistant.RequestProcessor.FloxyDatabase") as mock_db_cls:
            mock_db = Mock()
            mock_db_cls.get_instance.return_value = mock_db

            full_handler._save_to_database("query", "false_positive(blah)")

            mock_db.submit.assert_not_called()

    def test_test_mode_skips_save(self, full_handler):
        with patch("assistant.RequestProcessor.FloxyDatabase") as mock_db_cls:
            mock_db = Mock()
            mock_db_cls.get_instance.return_value = mock_db
            full_handler.request_context.test = True

            full_handler._save_to_database("query", "response")

            mock_db.submit.assert_not_called()

    def test_outage_reply_skips_save(self, full_handler):
        with patch("assistant.RequestProcessor.FloxyDatabase") as mock_db_cls:
            mock_db = Mock()
            mock_db_cls.get_instance.return_value = mock_db

            full_handler._save_to_database("query", "Hel" + RequestProcessor.LLM_OUTAGE_HISTORY_MARKER)

            mock_db.submit.assert_not_called()

    def test_includes_cached_flag(self, full_handler):
        with patch("assistant.RequestProcessor.FloxyDatabase") as mock_db_cls:
            mock_db = Mock()
            mock_db_cls.get_instance.return_value = mock_db
            full_handler.dialog_manager.last_response_was_cached = True

            full_handler._save_to_database("query", "response")

            call_args = mock_db.submit.call_args
            assert call_args[0][3] is True  # cached flag


# ---------------------------------------------------------------------------
# TestWaitForInitialCommand
# ---------------------------------------------------------------------------

class TestWaitForInitialCommand:
    def test_killed_during_wait_raises(self, full_handler):
        cp = Mock()
        cp.command_being_processed = True
        cp.command_queue = Mock()
        cp.command_queue.empty.return_value = True
        full_handler.dialog_manager.command_processor = cp

        def kill_on_sleep(_):
            full_handler.is_killed = True

        with patch("assistant.RequestProcessor.time.sleep", side_effect=kill_on_sleep):
            with pytest.raises(RequestKilledException):
                full_handler._wait_for_initial_command()


# ---------------------------------------------------------------------------
# TestWaitForAudioAndNotifyOutposts
# ---------------------------------------------------------------------------

class TestWaitForAudioAndNotifyOutposts:
    def test_text_only_does_not_touch_playback(self, full_handler):
        full_handler.request_context.text_only = True
        full_handler.request_context.server_sound = True
        full_handler._wait_for_audio_and_notify_outposts()
        full_handler.playback.finish_reply.assert_not_called()

    def test_client_streamed_audio_does_not_touch_playback(self, full_handler):
        full_handler.request_context.text_only = False
        full_handler.request_context.server_sound = False
        full_handler._wait_for_audio_and_notify_outposts()
        full_handler.playback.finish_reply.assert_not_called()

    def test_server_sound_finishes_reply_with_llm_response(self, full_handler):
        full_handler.request_context.text_only = False
        full_handler.request_context.server_sound = True
        full_handler.dialog_manager.llm_response = "How are you?"
        full_handler._wait_for_audio_and_notify_outposts()
        full_handler.playback.finish_reply.assert_called_once_with("How are you?")

# ---------------------------------------------------------------------------
# TestRestoreOutpostsIfMuted / TestEnsureOutpostsMuted (delegation)
# ---------------------------------------------------------------------------

class TestNotifyThinkingStarted:
    """The thinking filler is routed through the handler's ServerPlayback, which mutes
    the outposts before the filler sounds (see test_server_playback.TestPlayThinkingFiller
    for the ordering and language tests). DialogManager only decides *whether* to play."""

    def _dialog_manager(self, full_handler, server_sound, lang=None):
        from assistant.DialogManager import DialogManager
        dm = DialogManager.__new__(DialogManager)
        dm.web_request_handler = full_handler
        full_handler.request_context = RequestContext(
            source="VoiceToVoice", location="Kitchen", user="TestUser", user_input="hi",
            server_sound=server_sound, lang=lang,
        )
        return dm

    def test_server_sound_plays_filler_through_playback_with_request_lang(self, full_handler):
        dm = self._dialog_manager(full_handler, server_sound=True, lang="sv-SE")
        dm.notify_thinking_started()
        full_handler.playback.play_thinking_filler.assert_called_once_with("sv-SE")

    def test_not_server_sound_does_nothing(self, full_handler):
        dm = self._dialog_manager(full_handler, server_sound=False)
        dm.notify_thinking_started()
        full_handler.playback.play_thinking_filler.assert_not_called()

    def test_playback_failure_is_swallowed(self, full_handler):
        dm = self._dialog_manager(full_handler, server_sound=True)
        full_handler.playback.play_thinking_filler.side_effect = RuntimeError("afplay missing")
        dm.notify_thinking_started()


# ---------------------------------------------------------------------------
# TestProcess
# ---------------------------------------------------------------------------

class TestProcess:
    def test_begin_turn_forwards_request_args_to_the_gate(self, full_handler):
        """The wake_word_detected/scribe_skipped flags reach the verdict gate only through
        this handoff (the parse itself is tested in test_voice_turn_gate.TestBeginTurn)."""
        full_handler.turn_gate = Mock()
        args = {"query": "hi", "wake_word_detected": "1", "scribe_skipped": "1"}
        full_handler.request_data = RequestData("GET", "/", args, {"Range": None, "X-Playback-Session-Id": None, "Icy-Metadata": None, "User-Agent": "test"})
        full_handler.request_context = RequestContext.from_request_data(full_handler.request_data)
        prompt_manager = Mock()
        prompt_manager.prompt_touched_at = 0

        with patch.object(full_handler, 'generate_events', return_value="response"):
            full_handler.process(prompt_manager)

        full_handler.turn_gate.begin_turn.assert_called_once_with(args)

    def test_reset_creates_new_dialog_manager(self, full_handler):
        full_handler.request_data = RequestData("GET", "/", {"query": "hi", "reset": "1"}, {"Range": None, "X-Playback-Session-Id": None, "Icy-Metadata": None, "User-Agent": "test"})
        full_handler.request_context = RequestContext.from_request_data(full_handler.request_data)
        prompt_manager = Mock()
        prompt_manager.prompt_touched_at = 0
        old_dm = full_handler.dialog_manager

        with patch.object(full_handler, 'generate_events', return_value="response"), \
             patch("assistant.RequestProcessor.DialogManager") as MockDM:
            new_dm = Mock()
            MockDM.return_value = new_dm
            full_handler.process(prompt_manager)

        assert full_handler.dialog_manager is new_dm

    def test_reset_dialog_manager_inherits_the_handlers_voice_factory(self, full_handler):
        full_handler.voice_factory = factory = Mock()
        full_handler.request_data = RequestData("GET", "/", {"query": "hi", "reset": "1"}, {"Range": None, "X-Playback-Session-Id": None, "Icy-Metadata": None, "User-Agent": "test"})
        full_handler.request_context = RequestContext.from_request_data(full_handler.request_data)
        prompt_manager = Mock()
        prompt_manager.prompt_touched_at = 0

        with patch.object(full_handler, 'generate_events', return_value="response"), \
             patch("assistant.RequestProcessor.DialogManager") as MockDM:
            full_handler.process(prompt_manager)

        MockDM.assert_called_once_with(full_handler, False, voice_factory=factory)

    def test_a_failure_before_the_generator_clears_the_in_flight_flag(self, full_handler):
        # Assistant._in_flight reads the flag to refuse a guest turn; left true by a handler
        # that never got to its generator's finally, it would refuse guests until the next
        # owner request killed the handler.
        full_handler.request_data = RequestData("GET", "/", {"query": "hi"}, {"Range": None, "X-Playback-Session-Id": None, "Icy-Metadata": None, "User-Agent": "test"})
        full_handler.request_context = RequestContext.from_request_data(full_handler.request_data)
        prompt_manager = Mock()
        prompt_manager.prompt_touched_at = 0

        with patch.object(full_handler, 'generate_events', side_effect=RuntimeError("boom")), \
             patch.object(full_handler, '_notify_autonomous_user_ended') as ended, \
             pytest.raises(RuntimeError):
            full_handler.process(prompt_manager)

        assert full_handler.is_performing_request is False
        ended.assert_called_once()

    def test_killed_returns_empty_response(self, full_handler):
        full_handler.is_killed = True
        full_handler.request_data = RequestData("GET", "/", {"query": "hi"}, {"Range": None, "X-Playback-Session-Id": None, "Icy-Metadata": None, "User-Agent": "test"})
        full_handler.request_context = RequestContext.from_request_data(full_handler.request_data)
        prompt_manager = Mock()
        prompt_manager.prompt_touched_at = 0

        result = full_handler.process(prompt_manager)

        assert list(result) == []

# ---------------------------------------------------------------------------
# TestGenerateAudio
# ---------------------------------------------------------------------------

class TestGenerateEvents:
    """generate_events is the whole request as an event stream (content types and
    delimiters now live in tests/unit/interfaces/http/test_routes.py)."""

    @staticmethod
    def _run(full_handler, **patches):
        defaults = dict(_prepare_llm_context=dict(return_value="query"),
                        stream_response=dict(return_value=iter([Token("text")])),
                        _wait_for_initial_command={},
                        _handle_background_commands=dict(return_value=iter([])),
                        _update_history=dict(return_value="resp"),
                        _wait_for_audio_and_notify_outposts={},
                        _save_to_database={})
        defaults.update(patches)
        from contextlib import ExitStack
        with ExitStack() as stack:
            for name, kw in defaults.items():
                stack.enter_context(patch.object(full_handler, name, **kw))
            return list(full_handler.generate_events(Mock()))

    def test_text_only_round_then_done(self, full_handler):
        full_handler.request_context.text_only = True
        full_handler.request_context.server_sound = False
        assert self._run(full_handler) == [Token("text"), RoundEnd("Hi there!"), Done("Hi there!")]

    def test_follow_up_rounds_come_between_first_round_and_done(self, full_handler):
        full_handler.request_context.text_only = True
        events = self._run(full_handler, _handle_background_commands=dict(
            return_value=iter([Token("more"), RoundEnd("Hi there!")])))
        assert events == [Token("text"), RoundEnd("Hi there!"), Token("more"), RoundEnd("Hi there!"), Done("Hi there!")]

    def test_denied_request_is_a_notice_only(self, full_handler):
        events = self._run(full_handler, _prepare_llm_context=dict(return_value=None))
        assert events == [Notice("Request denied.")]

    def test_killed_ends_stream_silently(self, full_handler):
        full_handler.request_context.text_only = True
        events = self._run(full_handler, stream_response=dict(side_effect=RequestKilledException))
        assert events == []

    def test_stream_error_yields_notice_with_the_exception_message_and_no_done(self, full_handler):
        """The 2026-09-17 missing OPENAI_API_KEY: never reached a server, so there is no
        server message — the exception's own text is what tells the user what to fix."""
        full_handler.request_context.text_only = True
        full_handler.dialog_manager.ai.provider.return_value = "OpenAI"
        message = "The api_key client option must be set either by passing api_key to the client or by setting the OPENAI_API_KEY environment variable"
        events = self._run(full_handler, stream_response=dict(side_effect=RuntimeError(message)))
        assert events == [Notice(f"The OpenAI request failed: {message}")]

    def test_stream_error_without_a_message_yields_the_generic_outage_notice(self, full_handler):
        full_handler.request_context.text_only = True
        events = self._run(full_handler, stream_response=dict(side_effect=RuntimeError()))
        assert events == [Notice(RequestProcessor.LLM_OUTAGE_NOTICE)]

    def test_long_multiline_exception_message_is_flattened_and_capped(self, full_handler):
        full_handler.request_context.text_only = True
        full_handler.dialog_manager.ai.provider.return_value = "OpenAI"
        events = self._run(full_handler, stream_response=dict(side_effect=RuntimeError("bad\n  gateway " + "x" * 1000)))
        detail = events[0].text.removeprefix("The OpenAI request failed: ")
        assert detail.startswith("bad gateway xxx") and detail.endswith("…")
        assert len(detail) == RequestProcessor.LLM_FAILURE_DETAIL_MAX_CHARS + 1

    def test_server_error_yields_notice_with_the_servers_message(self, full_handler):
        full_handler.request_context.text_only = True
        full_handler.dialog_manager.ai.provider.return_value = "Claude"
        events = self._run(full_handler, stream_response=dict(side_effect=_billing_error()))
        assert events == [Notice(BILLING_NOTICE)]

    def test_sets_performing_request_false(self, full_handler):
        full_handler.request_context.text_only = True
        full_handler.request_context.server_sound = True
        full_handler.is_performing_request = True
        self._run(full_handler)
        assert full_handler.is_performing_request is False

    def test_side_events_pending_at_the_end_are_flushed_before_done(self, full_handler):
        full_handler.request_context.text_only = True
        full_handler.emit(CommandRan("turn_on", "kitchen"))
        events = self._run(full_handler, stream_response=dict(return_value=iter([])))
        assert events == [RoundEnd("Hi there!"), CommandRan("turn_on", "kitchen"), Done("Hi there!")]


class TestIterUntilKilled:
    """Regression for the 60s client timeout when a superseded request's audio
    generator blocks forever (CachedVoiceSynthesizer.start's audio_queue.get with no
    timeout). _iter_until_killed must abandon the stuck producer and return promptly
    when the request is killed, so the HTTP response closes instead of hanging."""

    def test_passes_through_all_items(self, handler):
        assert list(handler._iter_until_killed(iter([1, 2, 3]))) == [1, 2, 3]

    def test_returns_promptly_when_killed_on_hung_producer(self, handler):
        started = threading.Event()
        release = threading.Event()

        def hung_gen():
            yield "first"
            started.set()
            release.wait(timeout=10)  # simulates audio_queue.get() that never returns
            yield "second"

        out = []
        t0 = time.time()
        gen = handler._iter_until_killed(hung_gen())
        out.append(next(gen))            # "first"
        assert started.wait(timeout=2)
        handler.is_killed = True          # supersede the request
        assert list(gen) == []            # consumer stops, doesn't wait for "second"
        elapsed = time.time() - t0
        assert elapsed < 2.0, f"consumer hung for {elapsed:.2f}s after kill"
        release.set()                     # let the abandoned daemon unwind

    def test_propagates_producer_exception(self, handler):
        def boom_gen():
            yield 1
            raise RuntimeError("tts blew up")

        gen = handler._iter_until_killed(boom_gen())
        assert next(gen) == 1
        with pytest.raises(RuntimeError, match="tts blew up"):
            list(gen)


class TestCommandWorkerGate:
    def test_public_gate_without_turn_gate_reports_killed(self, full_handler):
        full_handler.turn_gate = None
        full_handler.is_killed = True
        assert full_handler.wait_so_that_response_timing_is_natural(0) is False


class TestOrphanedNoteInjection:
    """_inject_orphaned_command_outcomes is the only wiring through which a killed
    request's background-command outcome (OrphanedCommands registry) actually reaches
    the LLM's history — the registry tests can't catch this glue going dead."""

    @pytest.fixture
    def registry(self):
        import assistant.commands.OrphanedCommands as oc
        fresh = oc.OrphanedCommandRegistry()
        with patch.object(oc, "_registry", fresh):
            yield fresh

    def test_note_injected_as_transient_system_entry(self, handler, registry):
        registry.register_completed("browse_act(55)", "checkout completed", is_error=False)
        prompt_manager = Mock()
        handler._inject_orphaned_command_outcomes(prompt_manager)
        call = prompt_manager.message_history.add_transient_entry.call_args
        assert call.kwargs["role"] == "system"
        assert call.kwargs["source"] == "orphaned_command"
        assert "browse_act(55)" in call.kwargs["content"]
        assert "checkout completed" in call.kwargs["content"]

    def test_empty_registry_injects_nothing(self, handler, registry):
        prompt_manager = Mock()
        handler._inject_orphaned_command_outcomes(prompt_manager)
        prompt_manager.message_history.add_transient_entry.assert_not_called()

    def test_test_mode_neither_injects_nor_consumes_the_note(self, handler, registry):
        # A test-suite request must not eat an outcome a real user has yet to hear about.
        registry.register_completed("google(label printer)", "6 results", is_error=False)
        handler.request_context.test = True
        prompt_manager = Mock()
        handler._inject_orphaned_command_outcomes(prompt_manager)
        prompt_manager.message_history.add_transient_entry.assert_not_called()
        assert registry.drain_note() is not None  # still there for the next real request


class TestCommandResultPersistence:
    """Every follow-up round hands the result it feeds the LLM to the prompt manager as
    well, so the next request's history still holds what the command returned."""

    @pytest.fixture
    def handler(self):
        return TestHandleBackgroundCommands.handler.__wrapped__(self)

    def _run(self, handler, command_reply, contains_error=False, executed=()):
        dm1 = TestHandleBackgroundCommands._make_mock_dm(
            command_running=True, command_reply=command_reply, queue_empty=True)
        dm1.command_processor.command_reply_contains_error = contains_error
        dm1.command_processor.executed_commands = list(executed)
        handler.dialog_manager = dm1
        dm2 = TestHandleBackgroundCommands._make_mock_dm(command_running=False, command_reply=None, queue_empty=True)
        with patch.object(handler, 'stream_response', return_value=iter([])), \
             patch.object(handler, '_wait_for_initial_command'), \
             patch.object(handler, '_wait_for_background_command'), \
             patch.object(handler, '_wait_for_command_reply'), \
             patch('assistant.RequestProcessor.DialogManager', return_value=dm2):
            prompt_manager = Mock()
            gen = handler._handle_background_commands(prompt_manager)
            TestHandleBackgroundCommands._exhaust_generator(gen)
        return prompt_manager, handler.dialog_manager

    def test_plain_reply_persisted_as_sent(self, handler):
        prompt_manager, dm = self._run(handler, "iMessage — disabled")
        assert dm.reply_to_llm == "result: \niMessage — disabled"
        prompt_manager.add_command_result.assert_called_once_with("result: \niMessage — disabled", cap=True)

    def test_every_command_round_is_a_follow_up_to_the_llm(self, handler):
        """A round after a command ends in the result, which the next round of the request
        repeats and no later request does (the history keeps a cut copy of a page dump): the
        LLM caches that tail for its short tier, whatever the result's size. Moving the
        breakpoint in front of an oversized result made every later round re-send the whole
        tail uncached (prod 2026-10-20: 32–78K tokens ten times in one question)."""
        from assistant.prompt.MessageHistory import MessageHistory
        _, dm = self._run(handler, "x" * (MessageHistory.COMMAND_RESULT_CHARS + 1))
        assert dm.follow_up is True
        _, dm = self._run(handler, "small")
        assert dm.follow_up is True

    def test_toolkit_round_persisted_uncapped(self, handler):
        """A round that loaded a toolkit stores its result whole, other commands in the round
        or not: capped, the reference would be cut yet still count as loaded. A round without
        one keeps the cap (its result may be a web page)."""
        prompt_manager, _ = self._run(handler, "# Toolkit: email\n…", executed=["load_toolkit(email)"])
        prompt_manager.add_command_result.assert_called_once_with("result: \n# Toolkit: email\n…", cap=False)
        prompt_manager, _ = self._run(handler, "# Toolkit: email\n…\n\npage",
                                      executed=["load_toolkit(email)", "open_url(x)"])
        prompt_manager.add_command_result.assert_called_once_with("result: \n# Toolkit: email\n…\n\npage", cap=False)
        prompt_manager, _ = self._run(handler, "page", executed=["open_url(x)"])
        prompt_manager.add_command_result.assert_called_once_with("result: \npage", cap=True)

    def test_a_load_watch_round_is_a_toolkit_load_for_the_cap_too(self, handler):
        """load_watch brings the watch toolkit's reference along with the item, so its result is
        stored whole like a load_toolkit round: cut, the reference would be half there and its
        header would still count the toolkit as loaded."""
        reply = "# Watch: solar\n…\n\n# Toolkit: watch\n…"
        prompt_manager, _ = self._run(handler, reply, executed=['load_watch("solar")'])
        prompt_manager.add_command_result.assert_called_once_with(f"result: \n{reply}", cap=False)

    def test_block_reply_persists_text_and_notes_images(self, handler):
        blocks = [{"type": "text", "text": "Front door camera:"},
                  {"type": "image", "source": {"type": "base64", "data": "xx"}}]
        prompt_manager, _ = self._run(handler, blocks)
        prompt_manager.add_command_result.assert_called_once_with(
            "Front door camera:\n[image not kept in history]", cap=True)

    def test_error_report_persisted_too(self, handler):
        prompt_manager, dm = self._run(handler, "boom", contains_error=True)
        persisted = prompt_manager.add_command_result.call_args.args[0]
        assert persisted == dm.reply_to_llm
        assert "boom" in persisted


class TestTelemetryReport:
    def test_the_command_names_of_every_round_without_their_arguments(self, handler):
        first, second = Mock(), Mock()
        first.command_processor.executed_commands = ['light_on("kitchen")', "load_toolkit(home)"]
        second.command_processor.executed_commands = ["light_off(kitchen, 50%)"]
        handler._rounds = [first, second]
        assert handler._command_names() == ["light_on", "load_toolkit", "light_off"]

    def test_the_report_carries_the_source_and_the_commands(self, handler):
        from assistant.Telemetry import Telemetry
        handler.request_context = RequestContext(source="Telegram", user_input="lights")
        handler.dialog_manager.command_processor.executed_commands = ["light_on(kitchen)"]
        handler.dialog_manager.llm_response = "Done."
        handler._rounds = [handler.dialog_manager]
        with patch.object(Telemetry, "get_instance") as get_instance:
            handler._report_turn()
        get_instance.return_value.assistant_turn.assert_called_once_with("Done.", "Telegram", ["light_on"])

    def test_the_reply_is_what_every_round_said_without_the_command_blocks(self, handler):
        from assistant.Telemetry import Telemetry
        handler.request_context = RequestContext(source="Telegram", user_input="remember the gate code")
        first, second = Mock(), Mock()
        first.llm_response_raw = '<thinking>store it</thinking>One moment.\nremember("Gate code is 4711")'
        first.llm_response = "One moment.\n"
        first.command_processor.executed_commands = ['remember("Gate code is 4711")']
        second.llm_response_raw = second.llm_response = "Saved."
        second.command_processor.executed_commands = []
        handler._rounds = [first, second]
        with patch.object(Telemetry, "get_instance") as get_instance:
            handler._report_turn()
        reply = get_instance.return_value.assistant_turn.call_args.args[0]
        assert "One moment." in reply and "Saved." in reply
        assert "4711" not in reply and "remember" not in reply and "thinking" not in reply

    def test_a_test_request_is_never_reported(self, handler):
        from assistant.Telemetry import Telemetry
        handler.request_context = RequestContext(source="Telegram", test=True)
        with patch.object(Telemetry, "get_instance") as get_instance:
            handler._report_turn()
        get_instance.assert_not_called()

    def test_a_failing_report_never_breaks_the_request(self, handler):
        from assistant.Telemetry import Telemetry
        handler.request_context = RequestContext(source="Telegram")
        handler._rounds = [handler.dialog_manager]   # a Mock: its executed_commands cannot be iterated
        with patch.object(Telemetry, "get_instance"):
            handler._report_turn()

    def test_a_real_request_reports_the_users_turn_once(self, full_handler):
        from assistant.Telemetry import Telemetry
        full_handler.request_context.text_only = True
        with patch.object(Telemetry, "get_instance") as get_instance:
            events = TestGenerateEvents._run(full_handler)
        assert isinstance(events[-1], Done)
        get_instance.return_value.user_turn.assert_called_once_with("hello", "VoiceToVoice")

    def test_a_test_request_reports_no_turn_at_all(self, full_handler):
        from assistant.Telemetry import Telemetry
        full_handler.request_context.text_only = True
        full_handler.request_context.test = True
        with patch.object(Telemetry, "get_instance") as get_instance:
            events = TestGenerateEvents._run(full_handler)
        assert isinstance(events[-1], Done)
        get_instance.return_value.user_turn.assert_not_called()
        get_instance.return_value.assistant_turn.assert_not_called()
