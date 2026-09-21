import httpx
import pytest
from unittest.mock import patch, MagicMock, Mock

from anthropic import APIStatusError

from assistant.llm.AbstractLLM import LLMRefusalError
from assistant.llm.ClaudeLLM import ClaudeLLM


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


class TestProvider:
    def test_provider_is_the_bare_vendor_name(self):
        llm = _make_llm()
        assert llm.name().startswith("Claude (")
        assert llm.provider() == "Claude"

    def test_display_model_is_the_spoken_family(self):
        llm = _make_llm()
        llm.model = "claude-fable-5-1"
        assert llm.display_model() == "Fable"
        llm.model = "claude-opus-5"
        assert llm.display_model() == "Opus"


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

    def test_401_is_not_retryable(self):
        assert ClaudeLLM._is_retryable(_make_api_status_error(401)) is False


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
