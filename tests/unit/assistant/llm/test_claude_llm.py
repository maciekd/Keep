import httpx
import pytest
from unittest.mock import patch, MagicMock, Mock

from anthropic import APIStatusError

from assistant.llm.AbstractLLM import LLMRefusalError, Thinking
from assistant.llm.ClaudeLLM import ClaudeLLM, WorkspaceRequiredError, WORKSPACE_CONFIG_KEY, WORKSPACE_HEADER, claude_workspace_id


def _make_api_status_error(status_code: int, message: str = "error") -> APIStatusError:
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = {"error": {"type": "api_error", "message": message}}
    response.headers = {}
    return APIStatusError(message, response=response, body={"error": {"type": "api_error", "message": message}})


def _make_overloaded_error() -> APIStatusError:
    response = MagicMock()
    response.status_code = 529
    response.json.return_value = {"error": {"type": "overloaded_error", "message": "overloaded"}}
    response.headers = {}
    return APIStatusError("overloaded", response=response, body={"error": {"type": "overloaded_error", "message": "overloaded"}})


class TestReasoningRequests:
    @pytest.mark.parametrize("model, refuses", [
        ("claude-sonnet-5-5", True), ("claude-opus-5-5", True), ("claude-fable-5-1", True),
        ("claude-haiku-5-5", False),
    ])
    def test_contract_with_anthropics_reasoning_extraction_safeguard(self, model, refuses):
        """Measured 2026-10-19 on the photo-indexing prompt: the 5-generation Sonnet, Opus and
        Fable answer a prompt with a "thinking" field by refusing the whole request; Haiku 5.5
        answers it. The photo indexer drops the field on this verdict."""
        llm = _make_llm()
        llm.model = model
        assert llm.refuses_reasoning_requests() is refuses


class TestProvider:
    def test_display_model_is_the_spoken_family(self):
        llm = _make_llm()
        llm.model = "claude-fable-5-1"
        assert llm.display_model() == "Fable"
        llm.model = "claude-opus-5"
        assert llm.display_model() == "Opus"

    @pytest.mark.parametrize("model", ["claude-opus-5-5", "claude-fable-5-1", "claude-sonnet-5-5", "claude-haiku-5-5"])
    def test_every_claude_model_writes_commands_in_keep_syntax_unless_the_tier_says_otherwise(self, model):
        """Keep's own syntax is the default for the whole family — the shorter prompt, and the
        processor rescues the <invoke> block Opus writes now and then; `command_format:` on the
        tier still wins (the invoke block was the default for one day, 2026-10-21)."""
        from assistant.commands.formats.InvokeFormat import InvokeFormat
        from assistant.commands.formats.KeepFormat import KeepFormat
        llm = _make_llm()
        llm.model = model
        assert llm.command_format() is KeepFormat
        llm.use_command_format(InvokeFormat)
        assert llm.command_format() is InvokeFormat
        llm.use_command_format(None)
        assert llm.command_format() is KeepFormat


@patch("assistant.llm.ClaudeLLM.config")
class TestRefusalFallback:
    def test_fable_falls_back_to_opus_5_with_short_lived_cache_without_fast_mode(self, mock_config):
        llm = _make_llm()
        llm.model = "claude-fable-5-1"
        llm.speed = "fast"

        fallback = llm.refusal_fallback()

        assert isinstance(fallback, ClaudeLLM)
        assert fallback.model == ClaudeLLM.REFUSAL_FALLBACK_MODEL == "claude-opus-5"
        assert fallback.speed is None  # fast mode is not carried over to the fallback
        # Refusals burst, so the fallback caches too — at the cheap 5m write tier, not 1h
        assert fallback.prompt_cache is True
        assert fallback.static_cache_ttl == ClaudeLLM.REFUSAL_FALLBACK_CACHE_TTL == "5m"
        assert fallback.refusal_fallback() is None  # the fallback's own refusal is final

    def test_non_fable_models_have_no_fallback(self, mock_config):
        llm = _make_llm()
        llm.model = "claude-sonnet-5"
        assert llm.refusal_fallback() is None


    @pytest.mark.parametrize("model", [
        "claude-fable-5-1", "claude-mythos-5-1", "claude-opus-5", "claude-sonnet-5",
        "claude-opus-4-8", "claude-sonnet-4-5", "claude-haiku-4-5",
    ])
    def test_a_fallback_thinks_the_way_its_primary_does(self, mock_config, model):
        """The fallback is sent the very messages built for the primary, <thinking>-tag
        instruction or not. refusal_fallback() and handles_thinking_internally() gate on model
        prefixes separately: if they drift apart, a self-thinking fallback gets the tag and
        refuses as well (the Sep 18 outage), or a tag-dependent one loses its reasoning."""
        llm = ClaudeLLM(model)
        fallback = llm.refusal_fallback()
        if fallback is not None:
            assert fallback.handles_thinking_internally() is llm.handles_thinking_internally()


@patch("assistant.llm.ClaudeLLM.config")
class TestCacheTtlOrder:
    """Anthropic's rule: a cache_control TTL may only get shorter through the request, and the
    history breakpoint comes after the system blocks. The prompt test runner shortens the static
    TTL alone (FLOXY_CLAUDE_STATIC_CACHE_TTL=5m), which left the history at 1h and every request
    refused with a 400."""

    def test_history_ttl_is_cut_to_a_shorter_static_ttl(self, mock_config):
        llm = ClaudeLLM("claude-opus-5-5", static_cache_ttl="5m", history_cache_ttl="1h")
        assert llm.history_cache_ttl == "5m"

    def test_the_fallback_gets_a_history_ttl_its_short_static_ttl_allows(self, mock_config):
        fallback = ClaudeLLM("claude-fable-5-1", static_cache_ttl="1h", history_cache_ttl="1h").refusal_fallback()
        assert ClaudeLLM.CACHE_TTL_RANK[fallback.history_cache_ttl] <= ClaudeLLM.CACHE_TTL_RANK[fallback.static_cache_ttl]

    def test_a_shorter_history_ttl_is_left_alone(self, mock_config):
        llm = ClaudeLLM("claude-opus-5-5", static_cache_ttl="1h", history_cache_ttl="5m")
        assert llm.history_cache_ttl == "5m"


class TestIsRetryable:
    def test_529_is_retryable(self):
        assert ClaudeLLM._is_retryable(_make_overloaded_error()) is True

    def test_500_is_retryable(self):
        assert ClaudeLLM._is_retryable(_make_api_status_error(500)) is True

    def test_overloaded_body_is_retryable(self):
        response = MagicMock()
        response.status_code = 503
        response.headers = {}
        err = APIStatusError("overloaded", response=response,
                             body={"error": {"type": "overloaded_error", "message": "overloaded"}})
        assert ClaudeLLM._is_retryable(err) is True

    def test_400_is_not_retryable(self):
        assert ClaudeLLM._is_retryable(_make_api_status_error(400)) is False

@patch("assistant.llm.ClaudeLLM.config")
class TestStreamingEventHandling:
    """Verify the streaming loop correctly handles the full Anthropic event sequence:
    message_start (cache stats), content_block_delta (text), message_delta (output tokens)."""

    def test_yields_only_text_from_mixed_events(self, mock_config):
        llm = ClaudeLLM.__new__(ClaudeLLM)
        llm.client = Mock()
        llm.model = "claude-sonnet-4-5"
        llm.speed = None
        llm.prompt_cache = True

        msg_start = MagicMock()
        msg_start.type = 'message_start'
        msg_start.message.usage.cache_creation_input_tokens = 100
        msg_start.message.usage.cache_read_input_tokens = 500
        msg_start.message.usage.input_tokens = 50

        text1 = MagicMock()
        text1.type = 'content_block_delta'
        text1.delta.type = 'text_delta'
        text1.delta.text = 'Hello'

        text2 = MagicMock()
        text2.type = 'content_block_delta'
        text2.delta.type = 'text_delta'
        text2.delta.text = ' world'

        msg_delta = MagicMock()
        msg_delta.type = 'message_delta'
        msg_delta.usage.output_tokens = 42

        llm.client.messages.create = Mock(return_value=iter([msg_start, text1, text2, msg_delta]))

        result = list(llm.create_streaming_completion([{"role": "user", "content": "hi"}]))
        assert result == ['Hello', ' world']

    def test_message_delta_without_usage_is_ignored(self, mock_config):
        llm = ClaudeLLM.__new__(ClaudeLLM)
        llm.client = Mock()
        llm.model = "claude-sonnet-4-5"
        llm.speed = None
        llm.prompt_cache = True

        text_event = MagicMock()
        text_event.type = 'content_block_delta'
        text_event.delta.type = 'text_delta'
        text_event.delta.text = 'ok'

        delta_no_usage = MagicMock()
        delta_no_usage.type = 'message_delta'
        del delta_no_usage.usage  # hasattr(event, 'usage') will be False

        llm.client.messages.create = Mock(return_value=iter([text_event, delta_no_usage]))

        result = list(llm.create_streaming_completion([{"role": "user", "content": "hi"}]))
        assert result == ['ok']

    def test_fast_mode_uses_beta_endpoint(self, mock_config):
        llm = ClaudeLLM.__new__(ClaudeLLM)
        llm.client = Mock()
        llm.model = "claude-sonnet-4-5"
        llm.speed = "fast"
        llm.prompt_cache = True

        text_event = MagicMock()
        text_event.type = 'content_block_delta'
        text_event.delta.type = 'text_delta'
        text_event.delta.text = 'fast'

        llm.client.beta.messages.create = Mock(return_value=iter([text_event]))

        result = list(llm.create_streaming_completion([{"role": "user", "content": "hi"}]))
        assert result == ['fast']
        llm.client.beta.messages.create.assert_called_once()
        call_kwargs = llm.client.beta.messages.create.call_args[1]
        assert call_kwargs['speed'] == 'fast'
        assert ClaudeLLM.FAST_MODE_BETA in call_kwargs['betas']


class ClosableEvents:
    """An SDK event stream that records how far it was read and whether it was closed."""

    def __init__(self, events):
        self._events = iter(events)
        self.consumed = 0
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        event = next(self._events)
        self.consumed += 1
        return event

    def close(self):
        self.closed = True


def _text_delta(text: str):
    event = MagicMock()
    event.type = 'content_block_delta'
    event.delta.type = 'text_delta'
    event.delta.text = text
    return event


@patch("assistant.llm.ClaudeLLM.config")
class TestStreamClosing:
    """The dialog stops reading a reply at the command cap by closing the generator; the SDK
    stream has to go with it, or the request — and the model's generation — runs on to the
    model's own end (2026-10-21: 51 invokes in one reply)."""

    def _llm(self, stream):
        llm = ClaudeLLM.__new__(ClaudeLLM)
        llm.client = Mock()
        llm.model = "claude-sonnet-4-5"
        llm.speed = None
        llm.prompt_cache = True
        llm.client.messages.create = Mock(return_value=stream)
        return llm

    def test_a_consumer_that_stops_reading_closes_the_sdk_stream_where_it_stopped(self, mock_config):
        stream = ClosableEvents([_text_delta('one'), _text_delta('two'), _text_delta('three')])
        reply = self._llm(stream).create_streaming_completion([{"role": "user", "content": "hi"}])
        assert next(reply) == 'one'
        assert not stream.closed
        reply.close()
        assert stream.closed
        assert stream.consumed == 1

    def test_a_reply_read_to_its_end_leaves_no_open_stream_behind(self, mock_config):
        stream = ClosableEvents([_text_delta('one'), _text_delta('two')])
        assert list(self._llm(stream).create_streaming_completion([{"role": "user", "content": "hi"}])) == ['one', 'two']
        assert stream.closed

    def test_a_stream_without_a_close_is_read_as_before(self, mock_config):
        reply = self._llm(iter([_text_delta('one'), _text_delta('two')])).create_streaming_completion(
            [{"role": "user", "content": "hi"}])
        assert next(reply) == 'one'
        reply.close()


@patch("assistant.llm.ClaudeLLM.config")
class TestStreamingRetry:
    def test_retries_on_500_then_succeeds(self, mock_config):
        llm = ClaudeLLM.__new__(ClaudeLLM)
        llm.client = Mock()
        llm.model = "claude-sonnet-4-5"
        llm.speed = None
        llm.prompt_cache = True

        event_ok = MagicMock()
        event_ok.type = 'content_block_delta'
        event_ok.delta.type = 'text_delta'
        event_ok.delta.text = 'hello'

        call_count = 0

        def mock_create(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise _make_api_status_error(500)
            return iter([event_ok])

        llm.client.messages.create = mock_create

        with patch("assistant.llm.ClaudeLLM.time.sleep"):
            result = list(llm.create_streaming_completion([{"role": "user", "content": "hi"}]))

        assert result == ['hello']
        assert call_count == 2

    def test_does_not_retry_on_400(self, mock_config):
        llm = ClaudeLLM.__new__(ClaudeLLM)
        llm.client = Mock()
        llm.model = "claude-sonnet-4-5"
        llm.speed = None
        llm.prompt_cache = True

        llm.client.messages.create = Mock(side_effect=_make_api_status_error(400))

        with pytest.raises(APIStatusError):
            list(llm.create_streaming_completion([{"role": "user", "content": "hi"}]))


def _make_llm() -> ClaudeLLM:
    llm = ClaudeLLM.__new__(ClaudeLLM)
    llm.client = Mock()
    llm.model = "claude-fable-5"
    llm.speed = None
    llm.prompt_cache = True
    return llm


def _empty_stream_events() -> list:
    """The Fable misfire shape: message_start + message_delta, zero text deltas."""
    msg_start = MagicMock()
    msg_start.type = 'message_start'
    msg_start.message.usage.cache_creation_input_tokens = 0
    msg_start.message.usage.cache_read_input_tokens = 0
    msg_start.message.usage.input_tokens = 100

    msg_delta = MagicMock()
    msg_delta.type = 'message_delta'
    msg_delta.delta.stop_reason = 'end_turn'
    msg_delta.usage.output_tokens = 6
    return [msg_start, msg_delta]


def _refusal_stream_events() -> list:
    """A safety-layer refusal: same empty shape, but stop_reason=refusal."""
    events = _empty_stream_events()
    events[1].delta.stop_reason = 'refusal'
    return events


def _text_stream_events(text: str) -> list:
    event = MagicMock()
    event.type = 'content_block_delta'
    event.delta.type = 'text_delta'
    event.delta.text = text
    return [event]


@patch("assistant.llm.ClaudeLLM.config")
class TestEmptyReplyRetry:
    """Fable occasionally ends the turn with 2-6 output tokens and no text block.
    The request must be re-sent since nothing reached the consumer."""

    def test_streaming_retries_empty_reply_then_succeeds(self, mock_config):
        llm = _make_llm()
        llm.client.messages.create = Mock(side_effect=[
            iter(_empty_stream_events()),
            iter(_text_stream_events('recovered')),
        ])

        result = list(llm.create_streaming_completion([{"role": "user", "content": "hi"}]))

        assert result == ['recovered']
        assert llm.client.messages.create.call_count == 2

    def test_streaming_gives_up_after_max_empty_retries(self, mock_config):
        llm = _make_llm()
        llm.client.messages.create = Mock(
            side_effect=[iter(_empty_stream_events()) for _ in range(ClaudeLLM.EMPTY_REPLY_RETRIES + 1)]
        )

        result = list(llm.create_streaming_completion([{"role": "user", "content": "hi"}]))

        assert result == []
        assert llm.client.messages.create.call_count == ClaudeLLM.EMPTY_REPLY_RETRIES + 1

    def test_streaming_does_not_retry_after_text_was_yielded(self, mock_config):
        llm = _make_llm()
        llm.client.messages.create = Mock(return_value=iter(_text_stream_events('hello')))

        result = list(llm.create_streaming_completion([{"role": "user", "content": "hi"}]))

        assert result == ['hello']
        assert llm.client.messages.create.call_count == 1

    def test_streaming_refusal_raises_without_a_retry(self, mock_config):
        """The safety classifier is deterministic on identical input, so a refusal is
        final after one request — re-sending would only add refused requests."""
        llm = _make_llm()
        llm.client.messages.create = Mock(side_effect=[
            iter(_refusal_stream_events()),
            iter(_text_stream_events('never requested')),
        ])

        with pytest.raises(LLMRefusalError):
            list(llm.create_streaming_completion([{"role": "user", "content": "hi"}]))

        assert llm.client.messages.create.call_count == 1

    def test_streaming_empty_then_refusal_raises_at_the_refusal(self, mock_config):
        """A plain empty reply is retried; the retry's refusal ends it there."""
        llm = _make_llm()
        llm.client.messages.create = Mock(side_effect=[
            iter(_empty_stream_events()),
            iter(_refusal_stream_events()),
            iter(_empty_stream_events()),
        ])

        with pytest.raises(LLMRefusalError):
            list(llm.create_streaming_completion([{"role": "user", "content": "hi"}]))

        assert llm.client.messages.create.call_count == 2

    def test_non_streaming_retries_empty_reply_then_succeeds(self, mock_config):
        llm = _make_llm()

        empty_response = MagicMock()
        empty_response.usage.cache_creation_input_tokens = 0
        empty_response.usage.cache_read_input_tokens = 0
        empty_response.usage.input_tokens = 100
        empty_response.content = []
        empty_response.stop_reason = 'end_turn'

        ok_response = MagicMock()
        ok_response.usage.cache_creation_input_tokens = 0
        ok_response.usage.cache_read_input_tokens = 0
        ok_response.usage.input_tokens = 100
        ok_response.content = [MagicMock(type='text', text='recovered')]

        llm.client.messages.create = Mock(side_effect=[empty_response, ok_response])

        result = llm.create_completion([{"role": "user", "content": "hi"}])

        assert result == 'recovered'
        assert llm.client.messages.create.call_count == 2

    def test_non_streaming_refusal_raises_without_a_retry(self, mock_config):
        """The blocking path (think_with_messages — the autonomous check) follows the
        streaming rule: the first refusal is final and surfaces as LLMRefusalError, so a
        caller can hand the turn to refusal_fallback() instead of re-sending it."""
        llm = _make_llm()

        refused = MagicMock()
        refused.usage.cache_creation_input_tokens = 0
        refused.usage.cache_read_input_tokens = 0
        refused.usage.input_tokens = 100
        refused.content = []
        refused.stop_reason = 'refusal'

        llm.client.messages.create = Mock(return_value=refused)

        with pytest.raises(LLMRefusalError):
            llm.create_completion([{"role": "user", "content": "hi"}])

        assert llm.client.messages.create.call_count == 1

    def test_think_with_messages_does_not_retry_a_refusal(self, mock_config):
        """AbstractLLM._retry re-sends on any exception; a refusal passes straight through."""
        llm = _make_llm()
        llm.create_completion = Mock(side_effect=LLMRefusalError("refused"))

        with patch("assistant.llm.AbstractLLM.time.sleep") as sleep:
            with pytest.raises(LLMRefusalError):
                llm.think_with_messages([{"role": "user", "content": "hi"}])

        assert llm.create_completion.call_count == 1
        sleep.assert_not_called()


def _dropping_stream(events, exc):
    """A stream that yields the given events, then dies with a transport error —
    the mid-stream disconnect shape (peer closed connection during chunked read)."""
    for event in events:
        yield event
    raise exc


@patch("assistant.llm.ClaudeLLM.config")
class TestTransportErrorRetry:
    """Long-lived SSE streams occasionally get cut by upstream edges mid-body
    (httpx.RemoteProtocolError). Safe to re-issue only while nothing was yielded."""

    def test_drop_before_any_output_is_retried(self, mock_config):
        llm = _make_llm()
        llm.client.messages.create = Mock(side_effect=[
            _dropping_stream([], httpx.RemoteProtocolError("incomplete chunked read")),
            iter(_text_stream_events('recovered')),
        ])

        with patch("assistant.llm.ClaudeLLM.time.sleep"):
            result = list(llm.create_streaming_completion([{"role": "user", "content": "hi"}]))

        assert result == ['recovered']
        assert llm.client.messages.create.call_count == 2

    def test_drop_after_output_propagates_without_retry(self, mock_config):
        llm = _make_llm()
        llm.client.messages.create = Mock(return_value=_dropping_stream(
            _text_stream_events('partial'), httpx.RemoteProtocolError("incomplete chunked read")))

        yielded = []
        with pytest.raises(httpx.RemoteProtocolError):
            for chunk in llm.create_streaming_completion([{"role": "user", "content": "hi"}]):
                yielded.append(chunk)

        assert yielded == ['partial']
        assert llm.client.messages.create.call_count == 1

    def test_drop_before_output_gives_up_after_max_retries(self, mock_config):
        llm = _make_llm()
        max_attempts = len(ClaudeLLM.OVERLOADED_RETRY_DELAYS) + 1
        llm.client.messages.create = Mock(
            side_effect=[_dropping_stream([], httpx.RemoteProtocolError("dropped"))
                         for _ in range(max_attempts)]
        )

        with patch("assistant.llm.ClaudeLLM.time.sleep"):
            with pytest.raises(httpx.RemoteProtocolError):
                list(llm.create_streaming_completion([{"role": "user", "content": "hi"}]))

        assert llm.client.messages.create.call_count == max_attempts


@patch("assistant.llm.ClaudeLLM.config")
class TestNonStreamingRetry:
    def test_retries_on_500_then_succeeds(self, mock_config):
        llm = ClaudeLLM.__new__(ClaudeLLM)
        llm.client = Mock()
        llm.model = "claude-sonnet-4-5"
        llm.speed = None
        llm.prompt_cache = True

        ok_response = MagicMock()
        ok_response.usage.cache_creation_input_tokens = 0
        ok_response.usage.cache_read_input_tokens = 0
        ok_response.usage.input_tokens = 10
        ok_response.content = [MagicMock(type='text', text='hello')]

        call_count = 0

        def mock_create(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise _make_api_status_error(500)
            return ok_response

        llm.client.messages.create = mock_create

        with patch("assistant.llm.ClaudeLLM.time.sleep"):
            result = llm.create_completion([{"role": "user", "content": "hi"}])

        assert result == 'hello'
        assert call_count == 2


def _batch_message(text: str = "the reply", stop_reason: str = "end_turn"):
    message = MagicMock(stop_reason=stop_reason)
    message.usage.cache_creation_input_tokens = 0
    message.usage.cache_read_input_tokens = 0
    message.usage.input_tokens = 10
    message.usage.output_tokens = 5
    message.content = [MagicMock(type='text', text=text)] if text else []
    return message


def _batch_llm(statuses: list[str], outcome: str = "succeeded", message=None) -> ClaudeLLM:
    """A ClaudeLLM whose batch reports `statuses` on successive polls (the last one for ever)
    and ends with one result of type `outcome`."""
    llm = _make_llm()
    llm.prompt_cache = False
    batches = llm.client.messages.batches
    batches.create.return_value = Mock(id="batch_1")
    remaining = list(statuses)
    batches.retrieve.side_effect = lambda _id: Mock(
        processing_status=remaining.pop(0) if len(remaining) > 1 else remaining[0])
    result = Mock()
    result.result.type = outcome
    result.result.message = message or _batch_message()
    batches.results.return_value = [result]
    return llm


MESSAGES = [{'role': 'system', 'content': 'curate'}, {'role': 'user', 'content': 'now'}]


class TestEffort:
    """The wire contract with the Messages API: every request on a model whose effort default
    may be below high carries output_config.effort = high; a model without the parameter gets none."""

    @pytest.mark.parametrize("model", ["claude-opus-5-5", "claude-fable-5-1", "claude-sonnet-5-5", "claude-haiku-5-5"])
    def test_api_contract_high_effort_sent_on_5_generation_models(self, model):
        params = ClaudeLLM(model)._build_api_params(MESSAGES, thinking=False, stream=True)
        assert params['output_config'] == {"effort": "high"}

    def test_api_contract_no_effort_on_a_model_that_rejects_it(self):
        params = ClaudeLLM("claude-sonnet-4-5")._build_api_params(MESSAGES, thinking=False, stream=True)
        assert 'output_config' not in params

    def test_effort_rides_along_with_thinking(self):
        params = ClaudeLLM("claude-opus-5-5")._build_api_params(MESSAGES, thinking=True, stream=False)
        assert params['thinking'] == {"type": "adaptive"}
        assert params['output_config'] == {"effort": "high"}

    def test_thinking_more_is_an_ordinary_request_on_claude(self):
        """The autonomous check's Thinking.MORE buys reasoning only where a backend has levels
        below its best; Claude already runs every request at high effort, so MORE must not
        turn the long-answer headroom on — the same cap and parameters as a chat turn."""
        llm = ClaudeLLM("claude-opus-5-5")
        more = llm._build_api_params(MESSAGES, thinking=Thinking.MORE, stream=False)
        normal = llm._build_api_params(MESSAGES, thinking=Thinking.NORMAL, stream=False)
        assert more == normal
        assert 'thinking' not in more and more['max_tokens'] == ClaudeLLM.MAX_TOKENS
        full = llm._build_api_params(MESSAGES, thinking=Thinking.FULL, stream=False)
        assert full['max_tokens'] == ClaudeLLM.THINKING_MAX_TOKENS and 'thinking' in full


@patch('assistant.llm.ClaudeLLM.time.sleep')
class TestBatchedCompletion:
    def test_reply_comes_from_the_batch_once_it_has_ended(self, _sleep):
        llm = _batch_llm(["in_progress", "in_progress", "ended"])
        assert llm.create_batched_completion(MESSAGES, thinking=True) == "the reply"
        assert llm.client.messages.batches.retrieve.call_count == 3
        llm.client.messages.batches.cancel.assert_not_called()
        llm.client.messages.create.assert_not_called()

    def test_batch_params_are_the_ordinary_request_without_transport_options(self, _sleep):
        """The Batches API takes Messages API bodies: `stream` and the SDK-side `timeout`
        are not part of one and get the whole batch rejected."""
        llm = _batch_llm(["ended"])
        llm.create_batched_completion(MESSAGES, thinking=True)
        (request,) = llm.client.messages.batches.create.call_args.kwargs['requests']
        assert request['params'] == {k: v for k, v in llm._build_api_params(MESSAGES, True, stream=False).items()
                                     if k not in ('stream', 'timeout')}

    def test_batch_still_running_at_the_deadline_is_cancelled_and_gives_none(self, _sleep):
        llm = _batch_llm(["in_progress", "ended"], outcome="canceled")
        assert llm.create_batched_completion(MESSAGES, max_wait_seconds=0) is None
        llm.client.messages.batches.cancel.assert_called_once_with("batch_1")

    def test_reply_that_finished_while_being_cancelled_is_used(self, _sleep):
        """Otherwise the caller would pay for the batch's answer and for its own retry."""
        llm = _batch_llm(["in_progress", "ended"], outcome="succeeded")
        assert llm.create_batched_completion(MESSAGES, max_wait_seconds=0) == "the reply"
        llm.client.messages.batches.cancel.assert_called_once_with("batch_1")

    @pytest.mark.parametrize("outcome", ["errored", "expired"])
    def test_batch_that_ended_without_a_message_gives_none(self, _sleep, outcome):
        assert _batch_llm(["ended"], outcome=outcome).create_batched_completion(MESSAGES) is None

    def test_empty_reply_gives_none(self, _sleep):
        llm = _batch_llm(["ended"], message=_batch_message(text=""))
        assert llm.create_batched_completion(MESSAGES) is None

    def test_refusal_raises_instead_of_inviting_a_second_refused_call(self, _sleep):
        llm = _batch_llm(["ended"], message=_batch_message(text="", stop_reason="refusal"))
        with pytest.raises(LLMRefusalError):
            llm.create_batched_completion(MESSAGES)

    def test_truncated_reply_reports_max_tokens(self, _sleep):
        llm = _batch_llm(["ended"], message=_batch_message(stop_reason="max_tokens"))
        llm.create_batched_completion(MESSAGES)
        assert llm.last_stop_reason == "max_tokens"

    def test_failed_poll_is_a_missed_look_not_the_end(self, _sleep):
        llm = _batch_llm(["ended"])
        ended = llm.client.messages.batches.retrieve.side_effect
        looks = iter([_make_api_status_error(529), httpx.ConnectError("down")])
        def retrieve(batch_id):
            failure = next(looks, None)
            if failure:
                raise failure
            return ended(batch_id)
        llm.client.messages.batches.retrieve.side_effect = retrieve
        assert llm.create_batched_completion(MESSAGES) == "the reply"

    def test_fast_mode_has_no_batch(self, _sleep):
        llm = _batch_llm(["ended"])
        llm.speed = "fast"
        assert llm.create_batched_completion(MESSAGES) is None
        llm.client.messages.batches.create.assert_not_called()


WORKSPACE_REQUIRED_MESSAGE = ("anthropic-workspace-id is required when authenticating with an identity-linked "
                              "API key; send the id of the workspace this request acts in.")


def _make_400(message: str) -> APIStatusError:
    response = MagicMock()
    response.status_code = 400
    response.headers = {}
    body = {"type": "error", "error": {"type": "invalid_request_error", "message": message}}
    return APIStatusError(f"Error code: 400 - {body}", response=response, body=body)


def _bare_llm() -> ClaudeLLM:
    llm = ClaudeLLM.__new__(ClaudeLLM)
    llm.client = Mock()
    llm.model = "claude-sonnet-4-5"
    llm.speed = None
    llm.prompt_cache = True
    return llm


@patch("assistant.llm.ClaudeLLM.config")
class TestWorkspace:
    """The 2026-10 beta tester: a key the Console's Create key dialog left unscoped is refused
    on every request until the workspace id travels along as a header."""

    def test_configured_workspace_rides_on_every_request_as_the_header(self, mock_config):
        mock_config.missing_required.return_value = []
        mock_config.get.return_value = " wrkspc_01TEST "
        with patch("assistant.llm.ClaudeLLM.Anthropic") as client_cls:
            llm = ClaudeLLM("claude-sonnet-4-5")
            llm.initialize()
        assert client_cls.call_args.kwargs["default_headers"] == {WORKSPACE_HEADER: "wrkspc_01TEST"}

    def test_a_key_tied_to_one_workspace_sends_no_header(self, mock_config):
        mock_config.missing_required.return_value = [WORKSPACE_CONFIG_KEY]
        with patch("assistant.llm.ClaudeLLM.Anthropic") as client_cls:
            llm = ClaudeLLM("claude-sonnet-4-5")
            llm.initialize()
        assert client_cls.call_args.kwargs["default_headers"] is None
        assert claude_workspace_id() is None

    def test_streaming_refusal_for_a_missing_workspace_says_what_to_configure(self, mock_config):
        llm = _bare_llm()
        llm.client.messages.create = Mock(side_effect=_make_400(WORKSPACE_REQUIRED_MESSAGE))
        with pytest.raises(WorkspaceRequiredError) as excinfo:
            list(llm.create_streaming_completion([{"role": "user", "content": "hi"}]))
        assert WORKSPACE_CONFIG_KEY in str(excinfo.value)
        assert "platform.claude.com/settings/keys" in str(excinfo.value)

    def test_completion_refusal_for_a_missing_workspace_says_what_to_configure(self, mock_config):
        llm = _bare_llm()
        llm.client.messages.create = Mock(side_effect=_make_400(WORKSPACE_REQUIRED_MESSAGE))
        with pytest.raises(WorkspaceRequiredError):
            llm.create_completion([{"role": "user", "content": "hi"}])

    def test_any_other_400_is_passed_through_untouched(self, mock_config):
        llm = _bare_llm()
        error = _make_400("Your credit balance is too low to access the Anthropic API.")
        llm.client.messages.create = Mock(side_effect=error)
        with pytest.raises(APIStatusError) as excinfo:
            llm.create_completion([{"role": "user", "content": "hi"}])
        assert excinfo.value is error


TOOLKIT_TAIL = "result: \n# Toolkit: home\n\nhome_status()\n\n# Toolkit: web\n\nweb_search(query)"
SYSTEM_TWO_BLOCKS = f"stable{ClaudeLLM.CACHE_BREAKPOINT_MARKER}volatile{ClaudeLLM.CACHE_BREAKPOINT_MARKER}[THINKING]"


def _breakpoints(params: dict) -> int:
    count = sum(1 for block in params["system"] if "cache_control" in block)
    for message in params["messages"]:
        if isinstance(message["content"], list):
            count += sum(1 for block in message["content"] if "cache_control" in block)
    return count


def _test_suite_messages(system: str) -> list[dict]:
    """A prompt test's request: the fixture history, the all-toolkits round and the query
    merged into its result turn, as DialogManager sends it after a reset."""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "Marcus (Console): hello"},
        {"role": "assistant", "content": "Hi."},
        {"role": "assistant", "content": 'load_toolkit("home", "web")'},
        {"role": "user", "content": TOOLKIT_TAIL + "\n\nMarcus (Console): lights on"},
    ]


class TestTestSuiteCachedTail:
    """The prompt test suite's toolkit turn gets a breakpoint of its own, the query after it."""

    @pytest.fixture(autouse=True)
    def _tail(self, monkeypatch):
        monkeypatch.setattr(ClaudeLLM, "TEST_SUITE_CACHED_TAIL", TOOLKIT_TAIL)

    def test_toolkit_turn_split_with_the_query_after_its_breakpoint(self):
        params = ClaudeLLM("claude-opus-5-5")._build_api_params(_test_suite_messages(SYSTEM_TWO_BLOCKS),
                                                                thinking=True, stream=True)
        blocks = params["messages"][-1]["content"]
        assert [b["text"] for b in blocks] == [TOOLKIT_TAIL, "\n\nMarcus (Console): lights on"]
        assert "cache_control" in blocks[0] and "cache_control" in blocks[1]
        assert _breakpoints(params) == 4

    def test_follow_up_round_keeps_the_toolkit_breakpoint_and_four_in_all(self):
        messages = _test_suite_messages(SYSTEM_TWO_BLOCKS) + [
            {"role": "assistant", "content": "home_status()"},
            {"role": "user", "content": "result: all off"},
        ]
        params = ClaudeLLM("claude-opus-5-5")._build_api_params(messages, thinking=True, stream=True,
                                                                follow_up=True)
        toolkit_turn = params["messages"][-3]["content"]
        assert toolkit_turn[0]["text"] == TOOLKIT_TAIL and "cache_control" in toolkit_turn[0]
        assert "cache_control" in params["messages"][-1]["content"][-1]
        assert _breakpoints(params) == 4

    def test_not_split_when_three_system_blocks_take_the_breakpoints(self):
        system = SYSTEM_TWO_BLOCKS.replace("[THINKING]", f"more{ClaudeLLM.CACHE_BREAKPOINT_MARKER}[THINKING]")
        params = ClaudeLLM("claude-opus-5-5")._build_api_params(_test_suite_messages(system),
                                                                thinking=True, stream=True)
        blocks = params["messages"][-1]["content"]
        assert len(blocks) == 1 and blocks[0]["text"].startswith(TOOLKIT_TAIL)
        assert _breakpoints(params) == 4

    def test_production_request_untouched_without_a_tail(self, monkeypatch):
        monkeypatch.setattr(ClaudeLLM, "TEST_SUITE_CACHED_TAIL", None)
        params = ClaudeLLM("claude-opus-5-5")._build_api_params(_test_suite_messages(SYSTEM_TWO_BLOCKS),
                                                                thinking=True, stream=True)
        assert len(params["messages"][-1]["content"]) == 1
        assert _breakpoints(params) == 3

    def test_a_turn_that_is_the_toolkit_result_alone_is_one_block_never_an_empty_one(self):
        # Contract with the Messages API: a text block with no text is refused with a 400, so a
        # turn that ends where the tail ends gets the breakpoint and nothing after it.
        messages = _test_suite_messages(SYSTEM_TWO_BLOCKS)
        messages[-1] = {"role": "user", "content": TOOLKIT_TAIL}
        params = ClaudeLLM("claude-opus-5-5")._build_api_params(messages, thinking=True, stream=True)
        blocks = params["messages"][-1]["content"]
        assert [b["text"] for b in blocks] == [TOOLKIT_TAIL]
        assert all(b["text"].strip() for m in params["messages"] if isinstance(m["content"], list) for b in m["content"])
        assert "cache_control" in blocks[0]


def _cut_stream_events(text: str | None = None) -> list:
    """A stream ended at the output cap: on Opus 5.5 at high effort the thinking alone can
    use it up, so the only block is a thinking one and no text ever comes (issue #108)."""
    events = _empty_stream_events()
    block = MagicMock()
    block.type = 'content_block_start'
    block.content_block.type = 'thinking'
    events[1].delta.stop_reason = 'max_tokens'
    events[1].usage.output_tokens = 4096
    return [events[0], block] + (_text_stream_events(text) if text else []) + [events[1]]


@patch("assistant.llm.ClaudeLLM.config")
class TestOutputCapHit:
    """A reply cut at max_tokens is truncated, not empty: the identical request would think
    itself dry the same way (three byte-identical, fully cached attempts on 2026-10-23), so
    it is not re-sent — the caller reads the stop reason and tells the model instead."""

    def test_a_stream_that_thought_through_the_cap_is_not_re_sent(self, mock_config):
        llm = _make_llm()
        llm.client.messages.create = Mock(side_effect=[iter(_cut_stream_events()), iter(_text_stream_events('x'))])

        result = list(llm.create_streaming_completion([{"role": "user", "content": "hi"}]))

        assert result == []
        assert llm.client.messages.create.call_count == 1
        assert llm.last_stop_reason == 'max_tokens'

    def test_text_cut_at_the_cap_is_passed_through_as_it_is(self, mock_config):
        llm = _make_llm()
        llm.client.messages.create = Mock(return_value=iter(_cut_stream_events('Plan: six automations, bui')))

        result = list(llm.create_streaming_completion([{"role": "user", "content": "hi"}]))

        assert result == ['Plan: six automations, bui']
        assert llm.last_stop_reason == 'max_tokens'

    def test_the_stop_reason_of_the_previous_stream_does_not_outlive_it(self, mock_config):
        llm = _make_llm()
        llm.client.messages.create = Mock(side_effect=[iter(_cut_stream_events()), iter(_text_stream_events('x'))])
        list(llm.create_streaming_completion([{"role": "user", "content": "hi"}]))

        list(llm.create_streaming_completion([{"role": "user", "content": "again"}]))

        assert llm.last_stop_reason != 'max_tokens'

    def test_a_killed_request_is_not_re_sent_after_an_empty_reply(self, mock_config):
        """A superseded request went on to make two more full Opus calls after its kill."""
        llm = _make_llm()

        def attempt(**_):
            if llm.client.messages.create.call_count > 1:
                return iter(_text_stream_events('late'))
            llm.cancel()  # the kill lands while the first attempt streams
            return iter(_empty_stream_events())
        llm.client.messages.create = Mock(side_effect=attempt)

        result = list(llm.create_streaming_completion([{"role": "user", "content": "hi"}]))

        assert result == []
        assert llm.client.messages.create.call_count == 1

    def test_non_streaming_reply_cut_at_the_cap_is_not_re_sent(self, mock_config):
        llm = _make_llm()
        cut = MagicMock()
        cut.usage.cache_creation_input_tokens = 0
        cut.usage.cache_read_input_tokens = 0
        cut.usage.input_tokens = 100
        cut.content = []
        cut.stop_reason = 'max_tokens'
        llm.client.messages.create = Mock(return_value=cut)

        assert llm.create_completion([{"role": "user", "content": "hi"}]) == ""
        assert llm.client.messages.create.call_count == 1
        assert llm.last_stop_reason == 'max_tokens'
