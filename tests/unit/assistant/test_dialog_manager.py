import pytest
from unittest.mock import Mock

from assistant.DialogManager import DialogManager
from assistant.AssistantEvents import AssistantObserver, observers
from common.RequestKilledException import RequestKilledException
from assistant.commands.CommandProcessor import CommandProcessor
from assistant.commands.formats.KeepFormat import KeepFormat


class FakeObserver(AssistantObserver):
    def __init__(self):
        self.first_token_calls = 0
        self.tokens = []
        self.time_to_first_token = None

    def on_first_token(self, seconds, from_cache):
        self.first_token_calls += 1
        self.time_to_first_token = seconds

    def on_tokens(self, text):
        self.tokens.append(text)


class FakeCommandProcessor:
    def __init__(self):
        self.process_calls = []
        self.command_reply = None
        self.command_running_in_background = False
        self.possible_commands = []
        self.killed = False
        self.invocations = []
        self.commands_dispatched = 0
        self.reset_parse_state_calls = 0
        self.format = KeepFormat([])

    def reset_parse_state(self):
        self.reset_parse_state_calls += 1

    def clear_command_reply(self):
        self.command_reply = None

    def process(self, content, dialog_manager):
        self.process_calls.append(content)
        if "turn_on(kitchen)" in content:
            self.invocations.append(("turn_on", "kitchen", dialog_manager))
            return ""
        return content

    def flush_buffer(self, dialog_manager):
        return None

    def kill(self):
        self.killed = True


class FakeAI:
    def __init__(self, chunks, cached_flag=False, thinks_internally=False):
        self._chunks = chunks
        self.cached_flag = cached_flag
        self.thinks_internally = thinks_internally
        self.last_response_was_cached = False
        self.initialized = False
        self.observed_messages = None
        self.cancelled = False

    def cancel(self):
        self.cancelled = True

    def command_format(self):
        from assistant.commands.formats.KeepFormat import KeepFormat
        return KeepFormat

    def initialize(self):
        self.initialized = True

    def create_streaming_completion(self, messages, thinking: bool = False, volatile_tail=0, follow_up=False):
        # Capture a shallow copy so later mutations don't affect expectations
        self.observed_messages = [msg.copy() for msg in messages]
        self.observed_follow_up = follow_up
        self.last_response_was_cached = self.cached_flag
        for chunk in self._chunks:
            yield chunk

    def name(self):
        return "FakeAI"

    def display_model(self):
        return self.name()

    def refusal_fallback(self):
        return None

    def handles_thinking_internally(self):
        return self.thinks_internally


class TimeStub:
    def __init__(self, values):
        if not values:
            values = [0.0]
        self._values = iter(values)
        self._last = values[-1]

    def __call__(self):
        try:
            self._last = next(self._values)
        except StopIteration:
            pass
        return self._last


def make_dialog_manager(monkeypatch, chunks, cached=False, command_processor=None):
    console = FakeObserver()
    observers().subscribe(console)
    monkeypatch.setattr('assistant.DialogManager.CachedLLM', FakeAI)
    monkeypatch.setattr('assistant.llm.CachedLLM.CachedLLM', FakeAI)

    dm = DialogManager.__new__(DialogManager)
    dm.messages = [{"role": "system", "content": "system prompt"}]
    dm.reply_from_role = "user"
    dm.reply_to_llm = "Turn on the kitchen lights"
    dm.is_killed = False
    dm.llm_response = ""
    dm.llm_response_raw = ""
    dm.time_to_first_token = 0
    dm.first_token_got_at = 0
    dm.test_suite_mode = False
    dm.web_request_handler = Mock()
    # The overlay publish hook is real so tests can drain what reached /stream_text.
    from interfaces.voice.ios.TextStreamRegistry import TextStreamRegistry
    dm.web_request_handler.publish_text_event = (
        lambda rid, ev: TextStreamRegistry.get_instance().publish(rid, ev))
    dm.voice_factory = Mock()
    dm._voices = {"en-US": Mock(), "sv-SE": Mock()}
    dm.voice = dm._voices["en-US"]

    if command_processor is None:
        processor = FakeCommandProcessor()
        processor.command_reply = "existing"
    else:
        processor = command_processor

    dm.command_processor = processor

    dm.ai = FakeAI(chunks, cached_flag=cached)
    dm.last_response_was_cached = False
    dm.thinking = False
    dm.cache_query = None
    dm.follow_up = False

    return dm, console


@pytest.mark.unit
def test_stream_llm_response_processes_chunks_and_updates_state(monkeypatch):
    dm, console = make_dialog_manager(
        monkeypatch,
        ["Hello ", "turn_on(kitchen)", " friend"],
        cached=True
    )

    time_stub = TimeStub([100.0, 100.3, 100.5])
    monkeypatch.setattr('assistant.DialogManager.time.time', time_stub)

    output = list(dm.stream_llm_response())

    assert output == ["Hello ", "", " friend"]
    assert dm.command_processor.command_reply is None
    assert dm.llm_response_raw == "Hello turn_on(kitchen) friend"
    assert dm.llm_response == "Hello  friend"
    assert dm.time_to_first_token == pytest.approx(0.3)
    assert dm.first_token_got_at == pytest.approx(100.3)
    assert dm.last_response_was_cached is True
    assert dm.messages[-1]["content"] == "Hello turn_on(kitchen) friend"
    assert dm.messages[-2]["content"] == "Turn on the kitchen lights"

    assert dm.ai.initialized is True
    # History keeps the clean query (assert above); the copy sent to the LLM has the
    # per-request reply-language directive folded onto the end of the last user turn.
    assert dm.ai.observed_messages[-1]["content"].startswith("Turn on the kitchen lights")
    assert dm.ai.observed_messages[-1]["content"] != "Turn on the kitchen lights"

    assert console.first_token_calls == 1
    assert console.tokens == ["Hello ", "", " friend"]
    assert console.time_to_first_token == pytest.approx(0.3)

    assert dm.command_processor.process_calls == ["Hello ", "turn_on(kitchen)", " friend"]


@pytest.mark.unit
def test_stream_llm_response_merges_query_without_mutating_shared_trailing_turn(monkeypatch):
    """The messages list is a shallow copy of PromptManager's, so a trailing user turn may be
    a dict PromptManager still holds (the test-suite fixtures on a self-thinking tier, where no
    time note or instruction adds a fresh turn). Merging the query must replace that dict, not
    grow it in place — otherwise every later request carries this query too."""
    dm, _ = make_dialog_manager(monkeypatch, ["ok"])
    shared_turn = {"role": "user", "content": "toolkit reference"}
    dm.messages = [{"role": "system", "content": "system prompt"}, shared_turn]

    list(dm.stream_llm_response())

    assert shared_turn == {"role": "user", "content": "toolkit reference"}
    assert dm.messages[1]["content"] == "toolkit reference\n\nTurn on the kitchen lights"
    assert dm.ai.observed_messages[1]["content"].startswith("toolkit reference\n\nTurn on the kitchen lights")


@pytest.mark.unit
def test_stream_llm_response_tells_the_llm_it_is_a_follow_up_round(monkeypatch):
    """A command follow-up round ends in a result only the next round repeats; the LLM
    needs to know with the stream to cache that tail for its short tier."""
    dm, _ = make_dialog_manager(monkeypatch, ["ok"])
    dm.follow_up = True
    list(dm.stream_llm_response())
    assert dm.ai.observed_follow_up is True
    dm.follow_up = False
    list(dm.stream_llm_response())
    assert dm.ai.observed_follow_up is False


@pytest.mark.unit
def test_stream_llm_response_propagates_cache_query_to_cached_llm(monkeypatch):
    """The clean utterance set on the DialogManager must reach the CachedLLM so it keys the
    cache on the user query rather than the volatile last-message content sent to the LLM."""
    dm, _ = make_dialog_manager(monkeypatch, ["ok"])
    dm.cache_query = "turn off the kitchen lights"

    list(dm.stream_llm_response())

    assert dm.ai.cache_query == "turn off the kitchen lights"


@pytest.mark.unit
def test_a_round_that_produced_no_text_leaves_no_empty_assistant_turn(monkeypatch):
    """The output cap spent on native thinking: the stream ends with nothing received. The
    request circles back on these messages, and the Messages API rejects an empty
    non-final assistant message, so a placeholder stands for the turn — while the round's
    own record stays empty, so nothing is read back to the user or stored as said."""
    dm, _ = make_dialog_manager(monkeypatch, [])

    list(dm.stream_llm_response())

    assert dm.llm_response_raw == ""
    assert dm.messages[-1]["role"] == "assistant"
    assert dm.messages[-1]["content"].strip()


@pytest.mark.unit
def test_stream_llm_response_respects_initial_kill(monkeypatch):
    dm, _ = make_dialog_manager(monkeypatch, [])
    dm.is_killed = True

    with pytest.raises(RequestKilledException):
        list(dm.stream_llm_response())

    assert dm.ai.initialized is False
    assert dm.llm_response == ""
    assert dm.llm_response_raw == ""


@pytest.mark.unit
def test_process_llm_stream_stops_when_dialog_manager_killed_mid_stream(monkeypatch):
    dm, console = make_dialog_manager(monkeypatch, [])

    time_stub = TimeStub([200.0, 200.1, 200.2])
    monkeypatch.setattr('assistant.DialogManager.time.time', time_stub)

    def response_stream():
        yield "First "
        dm.is_killed = True
        yield "Second chunk that should not appear"

    output = []
    with pytest.raises(RequestKilledException):
        for chunk in dm.process_llm_stream(response_stream(), 200.0, False):
            output.append(chunk)

    assert output == ["First "]
    assert dm.llm_response == "First "
    assert dm.llm_response_raw == "First "
    assert console.tokens == ["First "]
    assert dm.command_processor.process_calls == ["First "]


@pytest.mark.unit
def test_stream_triggers_command_execution(monkeypatch):
    processor = CommandProcessor(test_suite_mode=True)
    original_invoke = processor.invoke_command
    processor.invoke_command = Mock(wraps=original_invoke)

    dm, console = make_dialog_manager(
        monkeypatch,
        ["turn_on(", "kitchen)", " done"],
        cached=False,
        command_processor=processor
    )

    dm.web_request_handler.wait_so_that_response_timing_is_natural = Mock(return_value=True)

    time_stub = TimeStub([1.0, 1.05, 1.1, 1.2])
    monkeypatch.setattr('assistant.DialogManager.time.time', time_stub)

    output = list(dm.stream_llm_response())

    processor.command_queue.join()

    assert output[-1].endswith("done")
    assert dm.llm_response.endswith("done")
    assert processor.invoke_command.called
    command_name, params, dialog_manager_arg = processor.invoke_command.call_args[0]
    assert command_name == "turn_on"
    assert params == ["kitchen"]
    assert dialog_manager_arg is dm

    processor.kill()


# ===========================================================================
# Mid-stream transport drop handling (_stream_with_transport_retry)
# ===========================================================================

import httpx


class DroppingAI(FakeAI):
    """Yields per-attempt chunk lists; a trailing exception in a list is raised
    after the preceding chunks — the mid-stream disconnect shape."""

    def __init__(self, attempts):
        super().__init__([])
        self.attempts = attempts
        self.calls = 0

    def create_streaming_completion(self, messages, thinking: bool = False, volatile_tail=0, follow_up=False):
        self.observed_messages = [msg.copy() for msg in messages]
        attempt = self.attempts[self.calls]
        self.calls += 1
        for item in attempt:
            if isinstance(item, Exception):
                raise item
            yield item


class ThinkingSwallowingProcessor(FakeCommandProcessor):
    """Swallows chunks the way the real processor swallows <thinking> text."""

    def process(self, content, dialog_manager):
        self.process_calls.append(content)
        if content.startswith("<thinking>"):
            return None
        return content


def _patch_stream_time(monkeypatch, values):
    monkeypatch.setattr('assistant.DialogManager.time.time', TimeStub(values))
    monkeypatch.setattr('assistant.DialogManager.time.sleep', lambda s: None)


@pytest.mark.unit
def test_transport_drop_with_only_thinking_output_retries_transparently(monkeypatch):
    """The incident shape: stream dies mid-thinking, nothing user-facing emitted —
    must retry invisibly, without duplicating the partial thinking in the raw log."""
    processor = ThinkingSwallowingProcessor()
    dm, _ = make_dialog_manager(monkeypatch, [], command_processor=processor)
    dm.ai = DroppingAI([
        ["<thinking>half a thought", httpx.RemoteProtocolError("incomplete chunked read")],
        ["All done."],
    ])
    _patch_stream_time(monkeypatch, [100.0, 100.1, 100.2, 100.3])

    output = list(dm.stream_llm_response())

    assert output == ["All done."]
    assert dm.ai.calls == 2
    assert dm.llm_response == "All done."
    assert dm.llm_response_raw == "All done."  # snapshot restored, no duplicated thinking
    assert dm.messages[-1] == {"role": "assistant", "content": "All done."}
    assert processor.reset_parse_state_calls == 1


@pytest.mark.unit
def test_a_gemini_server_error_before_any_output_retries_transparently(monkeypatch):
    """A Gemini "500 INTERNAL" two seconds into the stream is as transient as a dropped
    socket; until 2026-10-19 it fell through to the failure notice and the request was lost."""
    from google.genai.errors import ServerError

    processor = ThinkingSwallowingProcessor()
    dm, _ = make_dialog_manager(monkeypatch, [], command_processor=processor)
    dm.ai = DroppingAI([
        [ServerError(500, {"error": {"message": "INTERNAL", "status": "INTERNAL"}})],
        ["All done."],
    ])
    _patch_stream_time(monkeypatch, [100.0, 100.1, 100.2, 100.3])

    output = list(dm.stream_llm_response())

    assert output == ["All done."]
    assert dm.ai.calls == 2
    assert dm.messages[-1] == {"role": "assistant", "content": "All done."}


@pytest.mark.unit
def test_transport_drop_after_user_facing_text_truncates_gracefully(monkeypatch):
    """Once reply text has streamed, regeneration would diverge — no retry. The turn
    must end cleanly with a truncation notice and stay in history marked interrupted."""
    dm, _ = make_dialog_manager(monkeypatch, [])
    dm.ai = DroppingAI([
        ["Here is half an ans", httpx.RemoteProtocolError("incomplete chunked read")],
    ])
    _patch_stream_time(monkeypatch, [100.0, 100.1, 100.2])

    output = list(dm.stream_llm_response())

    assert dm.ai.calls == 1
    assert output[0] == "Here is half an ans"
    assert "dropped mid-reply" in output[-1]
    assert dm.llm_response.endswith("that's all I got.")
    assert dm.messages[-1]["role"] == "assistant"
    assert dm.messages[-1]["content"].endswith(DialogManager.STREAM_DROP_HISTORY_MARKER)


@pytest.mark.unit
def test_transport_drop_after_command_dispatched_does_not_retry(monkeypatch):
    """A dispatched command is a side effect — replaying the stream could re-fire it."""
    processor = FakeCommandProcessor()
    dm, _ = make_dialog_manager(monkeypatch, [], command_processor=processor)

    def process_and_dispatch(content, dialog_manager):
        processor.commands_dispatched += 1
        return ""
    processor.process = process_and_dispatch

    dm.ai = DroppingAI([
        ["turn_on(kitchen)", httpx.RemoteProtocolError("incomplete chunked read")],
        ["should never be requested"],
    ])
    _patch_stream_time(monkeypatch, [100.0, 100.1, 100.2])

    output = list(dm.stream_llm_response())

    assert dm.ai.calls == 1
    # The consumed-command chunk yields "" before the drop; only the notice carries text.
    assert [c for c in output if c] == ["The connection to the AI dropped. Please try again in a moment."]


@pytest.mark.unit
def test_transport_drop_retries_exhausted_yields_failure_notice(monkeypatch):
    processor = ThinkingSwallowingProcessor()
    dm, _ = make_dialog_manager(monkeypatch, [], command_processor=processor)
    max_attempts = DialogManager.STREAM_DROP_RETRIES + 1
    dm.ai = DroppingAI([
        ["<thinking>hm", httpx.RemoteProtocolError("dropped")] for _ in range(max_attempts)
    ])
    _patch_stream_time(monkeypatch, [100.0 + i / 10 for i in range(2 * max_attempts + 2)])

    output = list(dm.stream_llm_response())

    assert dm.ai.calls == max_attempts
    assert output == ["The connection to the AI dropped. Please try again in a moment."]
    assert dm.messages[-1]["content"].endswith(DialogManager.STREAM_DROP_HISTORY_MARKER)


@pytest.mark.unit
def test_llm_refusal_yields_notice_and_marks_history(monkeypatch):
    """A safety refusal (stop_reason=refusal, already retried in the LLM layer) must
    reach the user as a spoken/streamed notice, not silence, and mark the history turn."""
    from assistant.llm.AbstractLLM import LLMRefusalError

    dm, _ = make_dialog_manager(monkeypatch, [])
    dm.ai = DroppingAI([
        [LLMRefusalError("claude-fable-5 refused the request")],
    ])
    _patch_stream_time(monkeypatch, [100.0, 100.1, 100.2])

    output = list(dm.stream_llm_response())

    assert dm.ai.calls == 1  # no DialogManager-level retry on refusal
    assert output == ["FakeAI refused to process this request due to security concerns. "
                      "Rephrasing or asking again usually helps."]
    assert dm.llm_response == output[0]
    assert dm.messages[-1]["role"] == "assistant"
    assert dm.messages[-1]["content"] == "[the model's safety filter refused to answer this request]"


class FallbackAI(DroppingAI):
    """A DroppingAI whose refusal hands the request to `fallback` (Fable → Opus)."""

    def __init__(self, attempts, fallback, model="Fable"):
        super().__init__(attempts)
        self.fallback = fallback
        self.model = model

    def name(self):
        return f"Claude ({self.model})"

    def display_model(self):
        return self.model

    def refusal_fallback(self):
        return self.fallback


@pytest.mark.unit
def test_llm_refusal_with_fallback_answers_on_fallback_and_appends_notice(monkeypatch):
    """Fable's refusal is deterministic, so the turn is re-run on Opus; the user gets
    Opus's reply plus a notice naming both models, the history keeps only the reply,
    and the next request is back on Fable."""
    from assistant.llm.AbstractLLM import LLMRefusalError

    dm, _ = make_dialog_manager(monkeypatch, [])
    opus = FallbackAI([["Sure, let's set up Telegram."]], fallback=None, model="Opus")
    fable = FallbackAI([[LLMRefusalError("claude-fable-5-1 refused the request")]], fallback=opus)
    dm.ai = fable
    _patch_stream_time(monkeypatch, [100.0 + i / 10 for i in range(6)])

    output = list(dm.stream_llm_response())

    assert fable.calls == 1
    assert opus.calls == 1 and opus.initialized
    assert opus.observed_messages == fable.observed_messages
    notice = "\n\n(Fable refused this request, had to use Opus fallback)"
    assert output == ["Sure, let's set up Telegram.", notice]
    assert dm.llm_response == "Sure, let's set up Telegram." + notice
    assert dm.messages[-1] == {"role": "assistant", "content": "Sure, let's set up Telegram."}
    assert dm.ai is fable


@pytest.mark.unit
def test_llm_refusal_on_fallback_too_yields_refusal_notice(monkeypatch):
    """The fallback is used once: when Opus refuses as well, the refusal notice is the
    reply and no further model is tried."""
    from assistant.llm.AbstractLLM import LLMRefusalError

    dm, _ = make_dialog_manager(monkeypatch, [])
    haiku = FallbackAI([["never asked"]], fallback=None, model="Haiku")
    opus = FallbackAI([[LLMRefusalError("opus refused too")]], fallback=haiku, model="Opus")
    fable = FallbackAI([[LLMRefusalError("fable refused")]], fallback=opus)
    dm.ai = fable
    _patch_stream_time(monkeypatch, [100.0, 100.1, 100.2])

    output = list(dm.stream_llm_response())

    assert (fable.calls, opus.calls, haiku.calls) == (1, 1, 0)
    assert output == ["Claude refused to process this request due to security concerns. "
                      "Rephrasing or asking again usually helps."]
    assert dm.messages[-1]["content"] == "[the model's safety filter refused to answer this request]"
    assert dm.ai is fable


@pytest.mark.unit
def test_llm_refusal_notice_starts_audio_and_publishes_overlay(monkeypatch):
    """The refusal notice bypasses process_llm_stream, so nothing else starts the
    audio queue-drain thread or feeds the /stream_text overlay — on a server-sound
    request the notice would be pure silence without _surface_notice."""
    from assistant.llm.AbstractLLM import LLMRefusalError
    from interfaces.voice.ios.TextStreamRegistry import TextStreamRegistry
    TextStreamRegistry.reset_instance()

    dm, _ = make_dialog_manager(monkeypatch, [])
    dm.web_request_handler.ctx.text_only = False
    dm.request_id = "req-refusal-notice"
    dm.ai = DroppingAI([
        [LLMRefusalError("claude-fable-5 refused the request")],
    ])
    _patch_stream_time(monkeypatch, [100.0, 100.1, 100.2])

    output = list(dm.stream_llm_response())

    dm.web_request_handler.initialize_audio_playback.assert_called_once()
    published = _drain_published_text("req-refusal-notice")
    assert published == output[0]
    assert "refused" in published


@pytest.mark.unit
def test_transport_drop_notice_starts_audio_and_publishes_overlay(monkeypatch):
    """The transport-drop notice has the same out-of-band hole as the refusal notice."""
    from interfaces.voice.ios.TextStreamRegistry import TextStreamRegistry
    TextStreamRegistry.reset_instance()

    processor = ThinkingSwallowingProcessor()
    dm, _ = make_dialog_manager(monkeypatch, [], command_processor=processor)
    dm.web_request_handler.ctx.text_only = False
    dm.request_id = "req-drop-notice"
    max_attempts = DialogManager.STREAM_DROP_RETRIES + 1
    dm.ai = DroppingAI([
        ["<thinking>hm", httpx.RemoteProtocolError("dropped")] for _ in range(max_attempts)
    ])
    _patch_stream_time(monkeypatch, [100.0 + i / 10 for i in range(2 * max_attempts + 2)])

    output = list(dm.stream_llm_response())

    dm.web_request_handler.initialize_audio_playback.assert_called()
    published = _drain_published_text("req-drop-notice")
    assert "connection to the AI dropped" in published
    assert output[-1] in published


# ===========================================================================
# Additional tests
# ===========================================================================

@pytest.fixture
def fake_llm_backends(monkeypatch):
    """A DialogManager whose test is about the voice, not the LLMs: the real create_llm builds
    the fast tier's Gemini client, and that import costs the worker ~0.25 s."""
    monkeypatch.setattr('assistant.DialogManager.CachedLLM', FakeAI)
    monkeypatch.setattr('assistant.llm.CachedLLM.CachedLLM', FakeAI)
    monkeypatch.setattr('assistant.DialogManager.create_llm', lambda *a, **k: Mock(model="fake-model"))


@pytest.mark.unit
def test_select_voice_without_factory_runs_text_only(fake_llm_backends):
    """A process without a voice interface passes voice_factory=None: no synthesizer is
    ever built and self.voice stays None, which RequestProcessor.stream_response reads
    as "stream the reply as text"."""

    dm = DialogManager(web_request_handler=Mock(), test_suite_mode=True)
    assert dm.voice is None
    dm.select_voice(Mock(lang="sv-SE"))
    assert dm.voice is None


@pytest.mark.unit
def test_select_voice_builds_one_synth_per_language_via_factory(fake_llm_backends):
    """The factory decides the backend; DialogManager only memoizes its result per
    request language so a warm backend connection survives across the turns of one
    dialog (as the old per-DialogManager synth pair did) and set_request_language is
    applied on every selection."""
    built = []

    def factory(lang):
        built.append(lang)
        return Mock(name=f"voice-{lang}")

    dm = DialogManager(web_request_handler=Mock(), test_suite_mode=True, voice_factory=factory)
    assert dm.voice is None, "nothing is built before the first request"

    dm.select_voice(Mock(lang="sv-SE"))
    swedish = dm.voice
    swedish.set_request_language.assert_called_once_with("sv-SE")
    dm.select_voice(Mock(lang="en-US"))
    english = dm.voice
    dm.select_voice(Mock(lang="sv-SE"))
    assert dm.voice is swedish
    assert english is not swedish
    assert built == ["sv-SE", "en-US"]


@pytest.mark.unit
@pytest.mark.parametrize("server_sound", [True, False])
def test_select_voice_allows_the_night_voice_only_on_the_house_speakers(fake_llm_backends, server_sound):
    """A reply the client plays itself (the phone apps) wakes nobody, so only a
    server_sound request may whisper."""

    dm = DialogManager(web_request_handler=Mock(), test_suite_mode=True, voice_factory=lambda lang: Mock())
    dm.select_voice(Mock(lang="en-US", server_sound=server_sound))
    dm.voice.set_whisper_allowed.assert_called_once_with(server_sound)


@pytest.mark.unit
def test_smart_llm_constructed_with_prompt_cache_enabled(monkeypatch):
    """The smart LLM must be created with prompt_cache=True in production AND
    in test_suite_mode. Pins 1671e1cd: this was previously gated on
    test_suite_mode and shipping that gating to prod re-introduces the bug
    where prompt caching was effectively off for every live voice query.

    The fast LLM has no such requirement (some backends like Gemini don't
    accept prompt_cache; create_llm forwards it only when the constructor
    accepts it). What matters is that the smart path explicitly opts in."""
    monkeypatch.setattr('assistant.DialogManager.CachedLLM', FakeAI)
    monkeypatch.setattr('assistant.llm.CachedLLM.CachedLLM', FakeAI)

    from assistant.prompt.TimeOfDay import TimeOfDay
    monkeypatch.setattr('assistant.prompt.TimeOfDay.TimeOfDay.get_current', staticmethod(lambda: TimeOfDay.DAY))

    calls = []
    def fake_create_llm(class_name, model=None, prompt_cache=None, command_format=None):
        calls.append({'class': class_name, 'model': model, 'prompt_cache': prompt_cache})
        fake = FakeAI([])
        fake.model = model or 'fake-model'  # DialogManager reads .model for TTFTMonitor
        fake.update_command_list = Mock()    # CachedLLM-style hook DialogManager calls
        return fake
    monkeypatch.setattr('assistant.DialogManager.create_llm', fake_create_llm)
    monkeypatch.setattr('assistant.DialogManager.parse_llm_config',
                        lambda tier: ('ClaudeLLM', 'claude-opus-4-6'))

    DialogManager(web_request_handler=Mock(), test_suite_mode=True)

    # Calls are ai_fast (no prompt_cache kwarg), ai_smart with True, ai_vision with True.
    assert len(calls) == 3
    assert calls[0]['prompt_cache'] is None, "fast LLM should not force prompt_cache"
    assert calls[1]['prompt_cache'] is True, "smart LLM must enable prompt_cache unconditionally"
    assert calls[2]['prompt_cache'] is True, "vision LLM must enable prompt_cache unconditionally"


@pytest.mark.unit
def test_kill_closes_every_built_voice_in_background(monkeypatch):
    import threading
    dm, _ = make_dialog_manager(monkeypatch, [])
    closed = threading.Event()
    dm._voices["sv-SE"].close.side_effect = lambda: closed.set()
    dm.kill()
    assert closed.wait(timeout=2), "voice.close was not called in background"
    dm._voices["en-US"].close.assert_called()
    dm._voices["sv-SE"].close.assert_called()


@pytest.mark.unit
def test_stream_llm_response_with_thinking_passes_flag(monkeypatch):
    dm, _ = make_dialog_manager(monkeypatch, ["response"])
    dm.thinking = True

    time_stub = TimeStub([100.0, 100.1])
    monkeypatch.setattr('assistant.DialogManager.time.time', time_stub)

    class ThinkingAwareAI(FakeAI):
        def __init__(self):
            super().__init__(["response"])
            self.thinking_flag = None

        def create_streaming_completion(self, messages, thinking=False, volatile_tail=0, follow_up=False):
            self.thinking_flag = thinking
            self.observed_messages = [msg.copy() for msg in messages]
            self.last_response_was_cached = False
            for c in self._chunks:
                yield c

    ai = ThinkingAwareAI()
    dm.ai = ai
    list(dm.stream_llm_response())
    assert ai.thinking_flag is True


@pytest.mark.unit
def test_the_first_token_records_the_models_ttft_for_the_dashboard(monkeypatch):
    """Whichever backend answers — the fast tier is not always Claude — its TTFT reaches the
    dashboard from here; a reply served from the response cache never reached a model."""
    recorded = []
    ttft = Mock(record=lambda model, seconds: recorded.append((model, seconds)))
    monkeypatch.setattr('assistant.DialogManager.TTFTMonitor.get_instance', staticmethod(lambda: ttft))
    dm, _ = make_dialog_manager(monkeypatch, [])
    dm.ai.model = "gemini-fast"
    monkeypatch.setattr('assistant.DialogManager.time.time', TimeStub([1.4]))

    list(dm.process_llm_stream(iter(["Hello"]), 1.0, False))
    assert recorded == [("gemini-fast", pytest.approx(0.4))]

    dm.ai.last_response_was_cached = True
    list(dm.process_llm_stream(iter(["Hello"]), 1.0, False))
    assert len(recorded) == 1


@pytest.mark.unit
def test_process_llm_stream_replaces_backticks(monkeypatch):
    dm, console = make_dialog_manager(monkeypatch, [])

    time_stub = TimeStub([1.0, 1.1])
    monkeypatch.setattr('assistant.DialogManager.time.time', time_stub)

    output = list(dm.process_llm_stream(iter(["hello `world`"]), 1.0, False))
    combined = "".join(output)
    assert "`" not in combined


@pytest.mark.unit
def test_process_llm_stream_preserves_trailing_space_on_first_chunk(monkeypatch):
    """lstrip() on the first chunk should only remove leading whitespace,
    preserving trailing whitespace that acts as a word separator for the next chunk."""
    dm, console = make_dialog_manager(monkeypatch, [])

    time_stub = TimeStub([1.0, 1.1, 1.2])
    monkeypatch.setattr('assistant.DialogManager.time.time', time_stub)

    output = list(dm.process_llm_stream(iter(["Hey! ", "\U0001f44b"]), 1.0, False))
    assert output == ["Hey! ", "\U0001f44b"]
    assert dm.llm_response == "Hey! \U0001f44b"


@pytest.mark.unit
def test_process_llm_stream_strips_leading_whitespace(monkeypatch):
    dm, console = make_dialog_manager(monkeypatch, [])

    time_stub = TimeStub([1.0, 1.1])
    monkeypatch.setattr('assistant.DialogManager.time.time', time_stub)

    output = list(dm.process_llm_stream(iter(["  Hello"]), 1.0, False))
    assert output[0] == "Hello"


def _drain_published_text(request_id):
    from interfaces.voice.ios.TextStreamRegistry import TextStreamRegistry
    q = TextStreamRegistry.get_instance().get_or_create(request_id)
    parts = []
    while not q.empty():
        parts.append(q.get_nowait()["text"])
    return "".join(parts)


@pytest.mark.unit
def test_process_llm_stream_strips_emotion_tags_from_published_text(monkeypatch):
    """Emotion tags are synthesis directives and must not reach the /stream_text
    overlay, but the raw chunks (with tags) must still be yielded to the synth."""
    from interfaces.voice.ios.TextStreamRegistry import TextStreamRegistry
    TextStreamRegistry.reset_instance()
    dm, console = make_dialog_manager(monkeypatch, [])
    dm.request_id = "req-tags"

    time_stub = TimeStub([1.0, 1.1, 1.2])
    monkeypatch.setattr('assistant.DialogManager.time.time', time_stub)

    output = list(dm.process_llm_stream(iter(["Hello [laughing] there", " friend"]), 1.0, False))

    # Synth still receives the raw tags
    assert "[laughing]" in "".join(output)
    # The overlay does not
    published = _drain_published_text("req-tags")
    assert "[" not in published and "laughing" not in published
    assert "Hello" in published and "there" in published and "friend" in published


@pytest.mark.unit
def test_process_llm_stream_strips_emotion_tag_split_across_chunks(monkeypatch):
    """A tag straddling chunk boundaries must still be stripped from the overlay."""
    from interfaces.voice.ios.TextStreamRegistry import TextStreamRegistry
    TextStreamRegistry.reset_instance()
    dm, console = make_dialog_manager(monkeypatch, [])
    dm.request_id = "req-split"

    time_stub = TimeStub([1.0, 1.1, 1.2, 1.3])
    monkeypatch.setattr('assistant.DialogManager.time.time', time_stub)

    list(dm.process_llm_stream(iter(["Hi [sig", "hing] now"]), 1.0, False))

    published = _drain_published_text("req-split")
    assert "[" not in published and "sighing" not in published
    assert "Hi" in published and "now" in published


# ===========================================================================
# select_ai() tests
# ===========================================================================

@pytest.mark.unit
def test_select_ai_override_fast(monkeypatch):
    dm, _ = make_dialog_manager(monkeypatch, [])
    dm.ai_fast = FakeAI(["fast"])
    dm.ai_smart = FakeAI(["smart"])

    ctx = Mock()
    ctx.images = None
    ctx.ai_override = "fast"
    ctx.source = "TextToText"

    dm.select_ai(ctx)
    assert dm.ai is dm.ai_fast


@pytest.mark.unit
def test_select_ai_override_smart(monkeypatch):
    dm, _ = make_dialog_manager(monkeypatch, [])
    dm.ai_fast = FakeAI(["fast"])
    dm.ai_smart = FakeAI(["smart"])

    ctx = Mock()
    ctx.images = None
    ctx.ai_override = "smart"
    ctx.source = "VoiceToVoice"

    dm.select_ai(ctx)
    assert dm.ai is dm.ai_smart


@pytest.mark.unit
def test_select_ai_default_voice_uses_fast(monkeypatch):
    dm, _ = make_dialog_manager(monkeypatch, [])
    dm.ai_fast = FakeAI(["fast"])
    dm.ai_smart = FakeAI(["smart"])

    ctx = Mock()
    ctx.images = None
    ctx.ai_override = None
    ctx.source = "VoiceToVoice"

    dm.select_ai(ctx)
    assert dm.ai is dm.ai_fast


@pytest.mark.unit
def test_select_ai_default_text_uses_smart(monkeypatch):
    dm, _ = make_dialog_manager(monkeypatch, [])
    dm.ai_fast = FakeAI(["fast"])
    dm.ai_smart = FakeAI(["smart"])

    ctx = Mock()
    ctx.images = None
    ctx.ai_override = None
    ctx.source = "TextToText"

    dm.select_ai(ctx)
    assert dm.ai is dm.ai_smart


@pytest.mark.unit
def test_select_ai_images_use_vision(monkeypatch):
    """Images always win — vision tier is selected regardless of override/source."""
    dm, _ = make_dialog_manager(monkeypatch, [])
    dm.ai_fast = FakeAI(["fast"])
    dm.ai_smart = FakeAI(["smart"])
    dm.ai_vision = FakeAI(["vision"])

    ctx = Mock()
    ctx.images = [{"data": "abc", "media_type": "image/jpeg"}]
    ctx.ai_override = "fast"  # explicit override must be ignored when images present
    ctx.source = "VoiceToVoice"

    dm.select_ai(ctx)
    assert dm.ai is dm.ai_vision


@pytest.mark.unit
def test_use_vision_for_image_reply_switches_on_image_blocks(monkeypatch):
    """A command reply carrying image blocks (e.g. camera snapshots) must flip
    the already-selected tier to vision, since select_ai never saw these images."""
    dm, _ = make_dialog_manager(monkeypatch, [])
    dm.ai_smart = FakeAI(["smart"])
    dm.ai_vision = FakeAI(["vision"])
    dm.ai = dm.ai_smart  # what select_ai picked before the reply arrived

    dm.reply_to_llm = [
        {"type": "text", "text": "Here are the current camera images:"},
        {"type": "image", "source": {"type": "base64", "data": "abc"}},
    ]
    dm.use_vision_for_image_reply()
    assert dm.ai is dm.ai_vision


@pytest.mark.unit
@pytest.mark.parametrize("vision_thinks_internally", [True, False])
def test_use_vision_for_image_reply_strips_the_thinking_instruction(monkeypatch, vision_thinks_internally):
    """A voice request starts on a tier that gets the <thinking>-tag instruction; when camera
    images move the follow-up round to a vision model that thinks on its own, the instruction
    must leave the carried messages or the round is refused. What is left is exactly what
    get_messages(extended_thinking=False) would have built."""
    from assistant.prompt.MessageHistory import MessageHistory, THINKING_TAG_INSTRUCTION
    dm, _ = make_dialog_manager(monkeypatch, [])
    dm.ai = dm.ai_fast = FakeAI(["fast"])
    dm.ai_vision = FakeAI(["vision"], thinks_internally=vision_thinks_internally)

    def build(with_instruction):
        messages = [{"role": "system", "content": "prompt"}]
        MessageHistory.append_user_turn(messages, "(Current time: 14:50)")
        if with_instruction:
            MessageHistory.append_user_turn(messages, THINKING_TAG_INSTRUCTION)
        MessageHistory.append_user_turn(messages, "User [Marcus]: who is at the door?")
        messages.append({"role": "assistant", "content": "camera_snapshot(front door)"})
        return messages

    dm.messages = build(with_instruction=True)
    carried_turn = dm.messages[1]
    dm.reply_to_llm = [{"type": "image", "source": {"type": "base64", "data": "abc"}}]
    dm.use_vision_for_image_reply()

    assert dm.ai is dm.ai_vision
    assert dm.messages == build(with_instruction=not vision_thinks_internally)
    assert THINKING_TAG_INSTRUCTION in carried_turn["content"]  # the shared dict is replaced, not mutated


@pytest.mark.unit
def test_use_vision_for_image_reply_noop_on_text(monkeypatch):
    dm, _ = make_dialog_manager(monkeypatch, [])
    dm.ai_smart = FakeAI(["smart"])
    dm.ai_vision = FakeAI(["vision"])
    dm.ai = dm.ai_smart
    dm.reply_to_llm = "result: \nplain text reply"
    dm.use_vision_for_image_reply()
    assert dm.ai is dm.ai_smart


# ===========================================================================
# reply_text property tests
# ===========================================================================

@pytest.mark.unit
def test_reply_text_extracts_text_from_content_blocks(monkeypatch):
    dm, _ = make_dialog_manager(monkeypatch, [])
    dm.reply_to_llm = [
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "abc"}},
        {"type": "text", "text": "What is this?"},
    ]
    assert dm.reply_text == "What is this?"


@pytest.mark.unit
def test_reply_text_joins_multiple_text_blocks(monkeypatch):
    dm, _ = make_dialog_manager(monkeypatch, [])
    dm.reply_to_llm = [
        {"type": "text", "text": "Hello"},
        {"type": "text", "text": "world"},
    ]
    assert dm.reply_text == "Hello world"


@pytest.mark.unit
def test_reply_text_empty_string_for_no_text_blocks(monkeypatch):
    dm, _ = make_dialog_manager(monkeypatch, [])
    dm.reply_to_llm = [
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "abc"}},
    ]
    assert dm.reply_text == ""


@pytest.mark.unit
def test_stream_llm_response_appends_when_reply_is_content_blocks(monkeypatch):
    """When reply_to_llm is a list of content blocks (e.g. image + text),
    it must NOT be merged via string concatenation — that would crash.
    Instead it should be appended as a new message."""
    dm, console = make_dialog_manager(monkeypatch, ["Nice photo!"])

    time_stub = TimeStub([100.0, 100.1])
    monkeypatch.setattr('assistant.DialogManager.time.time', time_stub)

    dm.messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "Previous question"},
    ]
    dm.reply_from_role = "user"
    dm.reply_to_llm = [
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "abc"}},
        {"type": "text", "text": "What is this?"},
    ]

    output = list(dm.stream_llm_response())

    assert output == ["Nice photo!"]
    user_messages = [m for m in dm.messages if m["role"] == "user"]
    assert len(user_messages) == 2
    assert user_messages[0]["content"] == "Previous question"
    assert isinstance(user_messages[1]["content"], list)


@pytest.mark.unit
def test_stream_llm_response_appends_when_last_message_has_content_blocks(monkeypatch):
    """When the last message content is a list (content blocks), even if roles match,
    merging via += would crash. Must append a new message instead."""
    dm, console = make_dialog_manager(monkeypatch, ["Sure!"])

    time_stub = TimeStub([100.0, 100.1])
    monkeypatch.setattr('assistant.DialogManager.time.time', time_stub)

    dm.messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "abc"}},
            {"type": "text", "text": "Look at this"},
        ]},
    ]
    dm.reply_from_role = "user"
    dm.reply_to_llm = "Follow-up question"

    output = list(dm.stream_llm_response())

    assert output == ["Sure!"]
    user_messages = [m for m in dm.messages if m["role"] == "user"]
    assert len(user_messages) == 2
    assert isinstance(user_messages[0]["content"], list)
    assert user_messages[1]["content"] == "Follow-up question"


@pytest.mark.unit
def test_reply_language_directive_follows_detected_language(monkeypatch):
    """A Swedish request gets the Swedish reply-language directive folded onto the end
    of the message sent to the LLM, while history keeps only the clean query — the
    directive is ephemeral so it can't accumulate or pin a stale language."""
    dm, _ = make_dialog_manager(monkeypatch, ["svar"])
    monkeypatch.setattr('assistant.DialogManager.time.time', TimeStub([100.0, 100.1]))
    dm.web_request_handler.ctx.lang = "sv-SE"

    list(dm.stream_llm_response())

    assert dm.ai.observed_messages[-1]["content"].endswith(DialogManager._LANGUAGE_DIRECTIVES["sv"])
    assert dm.messages[-2]["content"] == "Turn on the kitchen lights"   # history stays clean


@pytest.mark.unit
def test_reply_language_directive_folds_into_content_blocks(monkeypatch):
    """An image (content-block) user turn must get the directive as a text block in the
    SAME message — appending a separate user message would break Claude/Gemini's
    alternating-role requirement."""
    dm, _ = make_dialog_manager(monkeypatch, ["ok"])
    monkeypatch.setattr('assistant.DialogManager.time.time', TimeStub([100.0, 100.1]))
    dm.web_request_handler.ctx.lang = "sv-SE"
    dm.messages = [{"role": "system", "content": "system prompt"}]
    dm.reply_to_llm = [
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "abc"}},
        {"type": "text", "text": "Vad är detta?"},
    ]

    list(dm.stream_llm_response())

    sent_user = [m for m in dm.ai.observed_messages if m["role"] == "user"]
    assert len(sent_user) == 1   # one user turn, directive folded in (not a new message)
    assert sent_user[-1]["content"][-1] == {"type": "text", "text": DialogManager._LANGUAGE_DIRECTIVES["sv"]}


@pytest.mark.unit
def test_brevity_directive_precedes_language_directive_for_voice_to_voice(monkeypatch):
    """Voice-to-voice requests get the word-cap directive folded onto the end of the request,
    but BEFORE the language directive so the language directive stays the final thing read."""
    dm, _ = make_dialog_manager(monkeypatch, ["svar"])
    monkeypatch.setattr('assistant.DialogManager.time.time', TimeStub([100.0, 100.1]))
    dm.web_request_handler.ctx.source = "VoiceToVoice"
    dm.web_request_handler.ctx.lang = "sv-SE"

    list(dm.stream_llm_response())

    content = dm.ai.observed_messages[-1]["content"]
    assert DialogManager._BREVITY_DIRECTIVE in content
    assert content.endswith(DialogManager._LANGUAGE_DIRECTIVES["sv"])   # language stays last
    assert content.index(DialogManager._BREVITY_DIRECTIVE) < content.index(DialogManager._LANGUAGE_DIRECTIVES["sv"])
    assert dm.messages[-2]["content"] == "Turn on the kitchen lights"   # history stays clean


@pytest.mark.unit
def test_brevity_directive_skipped_for_text_chat(monkeypatch):
    """Text/chat replies aren't length-capped, so the brevity directive must not appear."""
    dm, _ = make_dialog_manager(monkeypatch, ["reply"])
    monkeypatch.setattr('assistant.DialogManager.time.time', TimeStub([100.0, 100.1]))
    dm.web_request_handler.ctx.source = "TextChat"

    list(dm.stream_llm_response())

    assert DialogManager._BREVITY_DIRECTIVE not in dm.ai.observed_messages[-1]["content"]


@pytest.mark.unit
def test_byte_prefix_tail_sends_the_query_with_no_trailing_directives(monkeypatch):
    """A byte-prefix request (Ollama's MLX runner) must end with exactly the turn the history
    will re-render: a voice request in Swedish, which would otherwise get both the brevity and
    the language directive, is sent with neither."""
    dm, _ = make_dialog_manager(monkeypatch, ["svar"])
    monkeypatch.setattr('assistant.DialogManager.time.time', TimeStub([100.0, 100.1]))
    dm.web_request_handler.ctx.source = "VoiceToVoice"
    dm.web_request_handler.ctx.lang = "sv-SE"
    dm.byte_prefix_tail = True

    list(dm.stream_llm_response())

    assert dm.ai.observed_messages[-1]["content"] == "Turn on the kitchen lights"
    sent = "".join(m["content"] for m in dm.ai.observed_messages)
    assert DialogManager._BREVITY_DIRECTIVE not in sent
    assert DialogManager._LANGUAGE_DIRECTIVES["sv"] not in sent


@pytest.mark.unit
def test_stream_llm_response_merges_consecutive_same_role_messages(monkeypatch):
    """When the last message already has the same role as reply_from_role,
    the new content should be merged into it instead of appending a duplicate.
    This prevents LLM API validation errors (Claude/Gemini require alternating roles)."""
    dm, console = make_dialog_manager(monkeypatch, ["response"])

    time_stub = TimeStub([100.0, 100.1])
    monkeypatch.setattr('assistant.DialogManager.time.time', time_stub)

    # Pre-populate with a user message (simulates thinking instruction appended earlier)
    dm.messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "Think step by step."},
    ]
    dm.reply_from_role = "user"
    dm.reply_to_llm = "What is the weather?"

    list(dm.stream_llm_response())

    # Should have merged into existing user message, not appended a new one
    user_messages = [m for m in dm.messages if m["role"] == "user"]
    assert len(user_messages) == 1
    assert "Think step by step." in user_messages[0]["content"]
    assert "What is the weather?" in user_messages[0]["content"]


@pytest.mark.unit
def test_stream_llm_response_appends_when_roles_differ(monkeypatch):
    """When the last message has a different role, a new message should be appended."""
    dm, console = make_dialog_manager(monkeypatch, ["response"])

    time_stub = TimeStub([100.0, 100.1])
    monkeypatch.setattr('assistant.DialogManager.time.time', time_stub)

    dm.messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "assistant", "content": "Previous reply"},
    ]
    dm.reply_from_role = "user"
    dm.reply_to_llm = "Next question"

    list(dm.stream_llm_response())

    assert dm.messages[-2]["role"] == "user"
    assert dm.messages[-2]["content"] == "Next question"
    assert dm.messages[-3]["role"] == "assistant"


@pytest.mark.unit
def test_flush_buffer_content_yielded_and_accumulated(monkeypatch):
    """When flush_buffer returns content, it should be yielded, appended to llm_response,
    and sent to the TUI console (covers DialogManager lines 151-155)."""
    fcp = FakeCommandProcessor()
    fcp.flush_buffer = lambda dm: "flushed text"

    dm, console = make_dialog_manager(monkeypatch, ["Hello "], command_processor=fcp)

    time_stub = TimeStub([100.0, 100.1])
    monkeypatch.setattr('assistant.DialogManager.time.time', time_stub)

    output = list(dm.stream_llm_response())

    assert "flushed text" in output
    assert dm.llm_response.endswith("flushed text")
    assert "flushed text" in console.tokens


@pytest.mark.unit
def test_llms_are_cached_llm_around_vault_llm(monkeypatch):
    """create_llm wraps every backend in a VaultLLM; in production mode DialogManager then
    wraps that in CachedLLM, so `isinstance(ai, CachedLLM)` keeps finding the cache layer
    and cache rows hold restored text (VAULT_PLAN §2). The model names reach TTFTMonitor
    through VaultLLM's attribute pass-through."""
    import sys
    import types
    from assistant.llm.CachedLLM import CachedLLM
    from assistant.llm.VaultLLM import VaultLLM
    from common.vault.Vault import Vault

    class _FakeBackend:
        def __init__(self, model="fake-model", prompt_cache=None):
            self.model = model

        def command_format(self):
            from assistant.commands.formats.KeepFormat import KeepFormat
            return KeepFormat

    module = types.ModuleType("assistant.llm._FakeBackend")
    module._FakeBackend = _FakeBackend
    monkeypatch.setitem(sys.modules, "assistant.llm._FakeBackend", module)
    monkeypatch.setattr('assistant.DialogManager.parse_llm_config',
                        lambda tier: ('_FakeBackend', f'{tier}-model'))
    monkeypatch.setattr('common.FloxyDatabase.FloxyDatabase.get_instance', staticmethod(lambda: Mock()))
    ttft = Mock()
    monkeypatch.setattr('assistant.DialogManager.TTFTMonitor.get_instance', staticmethod(lambda: ttft))
    Vault.reset_instance()
    try:
        dm = DialogManager(web_request_handler=Mock(), test_suite_mode=False)
    finally:
        Vault.reset_instance()

    for ai in (dm.ai_fast, dm.ai_smart, dm.ai_vision):
        assert isinstance(ai, CachedLLM)
        assert isinstance(ai.llm, VaultLLM)
        assert isinstance(ai.llm.llm, _FakeBackend)
    assert dm.ai is dm.ai_fast
    ttft.register_models.assert_called_once_with(smart_model='smart-model', fast_model='fast-model')


@pytest.mark.unit
def test_select_voice_does_not_memoize_a_language_no_engine_speaks(fake_llm_backends):
    """d93803e4: the factory answers None when no enabled engine speaks the language. That
    turn runs text-only and the question is asked again next time, so enabling an engine
    from the chat takes effect on the very next request instead of after a restart."""
    answers = [None, Mock(name="voice-sv-SE")]
    built = []

    def factory(lang):
        built.append(lang)
        return answers.pop(0)

    dm = DialogManager(web_request_handler=Mock(), test_suite_mode=True, voice_factory=factory)
    dm.select_voice(Mock(lang="sv-SE"))
    assert dm.voice is None
    assert "sv-SE" not in dm._voices

    dm.select_voice(Mock(lang="sv-SE"))
    assert dm.voice is not None
    assert dm._voices["sv-SE"] is dm.voice
    assert built == ["sv-SE", "sv-SE"]


# ===========================================================================
# The command cap: a runaway reply is cut after MAX_COMMANDS_PER_REPLY calls
# ===========================================================================

class ClosableStream:
    """A stream that records when the dialog stopped reading it."""

    def __init__(self, chunks):
        self._chunks = iter(chunks)
        self.closed = False
        self.consumed = 0

    def __iter__(self):
        return self

    def __next__(self):
        chunk = next(self._chunks)
        self.consumed += 1
        return chunk

    def close(self):
        self.closed = True


@pytest.mark.unit
def test_a_reply_past_the_command_cap_is_cut_and_the_stream_closed(monkeypatch):
    processor = CommandProcessor(test_suite_mode=True)
    dm, console = make_dialog_manager(monkeypatch, [], command_processor=processor)
    monkeypatch.setattr('assistant.DialogManager.time.time', TimeStub([1.0, 1.1]))
    cap = CommandProcessor.MAX_COMMANDS_PER_REPLY
    chunks = ["On it. "] + [f"turn_on(lamp {i}) " for i in range(cap + 20)] + ["All done."]
    stream = ClosableStream(chunks)

    output = list(dm.process_llm_stream(stream, 1.0, False))
    processor.command_queue.join()

    assert stream.closed
    assert stream.consumed <= cap + 2  # the cut chunk itself may be read, nothing past it
    assert processor.commands_dispatched == cap
    calls = processor.format.extract_all(dm.llm_response_raw)
    assert [p for _, p in calls] == [[f"lamp {i}"] for i in range(cap)]
    assert dm.llm_response_raw.startswith("On it.")
    assert "All done." not in dm.llm_response_raw
    # The stream interleaves the CommandRan events of the calls already run with the text.
    assert "".join(c for c in output if isinstance(c, str)).strip() == "On it."
    processor.kill()


@pytest.mark.unit
def test_a_reply_within_the_command_cap_is_read_to_its_end(monkeypatch):
    processor = CommandProcessor(test_suite_mode=True)
    dm, console = make_dialog_manager(monkeypatch, [], command_processor=processor)
    monkeypatch.setattr('assistant.DialogManager.time.time', TimeStub([1.0, 1.1]))
    chunks = [f"turn_on(lamp {i}) " for i in range(3)] + ["All done."]
    stream = ClosableStream(chunks)

    output = list(dm.process_llm_stream(stream, 1.0, False))
    processor.command_queue.join()

    assert not stream.closed and stream.consumed == len(chunks)
    assert "".join(output).strip() == "All done."
    assert dm.llm_response_raw.endswith("All done.")
    processor.kill()


class OwnTurnAI(FakeAI):
    """A backend with an answer to prefers_query_as_own_turn (ds4 says yes, every other no)."""

    def __init__(self, chunks, own_turn: bool):
        super().__init__(chunks)
        self._own_turn = own_turn

    def prefers_query_as_own_turn(self):
        return self._own_turn


@pytest.mark.unit
def test_the_test_suite_query_is_its_own_turn_behind_an_ack_on_a_backend_that_asks(monkeypatch):
    """ds4-server cuts its reusable disk checkpoint right before the prompt's last user turn,
    so in the test suite the query goes after the toolkit result as a user turn of its own —
    behind a short assistant turn, since two user-side turns in a row render into one block —
    and the toolkit turn, a fixture PromptManager still holds, is left exactly as it was."""
    dm, _ = make_dialog_manager(monkeypatch, ["ok"])
    dm.ai = OwnTurnAI(["ok"], own_turn=True)
    dm.test_suite_mode = True
    shared_turn = {"role": "user", "content": "toolkit reference"}
    dm.messages = [{"role": "system", "content": "system prompt"}, shared_turn]

    list(dm.stream_llm_response())

    assert dm.messages[1] is shared_turn and shared_turn == {"role": "user", "content": "toolkit reference"}
    sent = dm.ai.observed_messages
    roles = [m["role"] for m in sent]
    assert roles[:4] == ["system", "user", "assistant", "user"]
    assert all(a != b for a, b in zip(roles[1:], roles[2:]))   # never two turns of one role in a row
    assert sent[3]["content"].startswith("Turn on the kitchen lights")
    assert "Turn on the kitchen lights" not in sent[1]["content"]


@pytest.mark.unit
@pytest.mark.parametrize("test_suite_mode, own_turn", [(False, True), (True, False)])
def test_the_query_merges_into_the_toolkit_turn_in_production_and_on_a_backend_that_does_not_ask(
        monkeypatch, test_suite_mode, own_turn):
    """The own turn is the test suite's on a backend that asks for it and nothing else: a
    production request on ds4 keeps the merge, as does the suite on Claude or Gemini, which
    reject same-role adjacency."""
    dm, _ = make_dialog_manager(monkeypatch, ["ok"])
    dm.ai = OwnTurnAI(["ok"], own_turn=own_turn)
    dm.test_suite_mode = test_suite_mode
    dm.messages = [{"role": "system", "content": "system prompt"}, {"role": "user", "content": "toolkit reference"}]

    list(dm.stream_llm_response())

    user_turns = [m for m in dm.ai.observed_messages if m["role"] == "user"]
    assert len(user_turns) == 1
    assert user_turns[0]["content"].startswith("toolkit reference\n\nTurn on the kitchen lights")
    assert DialogManager.TEST_SUITE_TOOLKITS_ACK not in [m["content"] for m in dm.ai.observed_messages]


@pytest.mark.unit
def test_a_call_written_as_an_invoke_block_under_keep_runs_and_is_stored_in_keep_syntax(monkeypatch):
    # Prod 2026-10-18 to -21: Opus under the keep format wrote the whole reply as an <invoke> block.
    processor = CommandProcessor(test_suite_mode=True)
    dm, console = make_dialog_manager(monkeypatch, [], command_processor=processor)
    monkeypatch.setattr('assistant.DialogManager.time.time', TimeStub([1.0, 1.1, 1.2]))
    dm.ai = FakeAI(['<invoke name="turn_on">\n<parameter name="light_name">Kitchen lamp</parameter>\n</invoke>\n'
                    '</function_calls>'])

    output = list(dm.stream_llm_response())
    processor.command_queue.join()

    assert "".join(output).strip() == ""
    assert processor.commands_dispatched == 1
    stored = dm.messages[-1]["content"]
    assert "<invoke" not in stored and "</function_calls>" not in stored
    assert KeepFormat(["turn_on"]).extract_all(stored) == [("turn_on", ["Kitchen lamp"])]


@pytest.mark.unit
def test_kill_tells_the_llm_to_stop_re_sending(monkeypatch):
    dm, _ = make_dialog_manager(monkeypatch, [])
    dm.ai = Mock()
    dm.kill()
    dm.ai.cancel.assert_called_once()
