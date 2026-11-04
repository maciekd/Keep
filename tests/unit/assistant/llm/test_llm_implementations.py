import base64
import itertools
import pytest
from unittest.mock import Mock, patch, MagicMock

from assistant.llm.AbstractLLM import AbstractLLM, LLMRefusalError, Thinking, ThinkingNotSupportedError
# Imported at collection, with the rest of the tree: GeminiLLM pulls in google.genai (~0.2 s of
# pydantic models), which would otherwise land in the call of the first Gemini test a worker runs.
from assistant.llm.GeminiLLM import GeminiLLM


def _reset_tracker(tracker):
    """A PrefillTracker with no request in flight."""
    tracker._requests.clear()
    tracker._last_token = 0


# ---------------------------------------------------------------------------
# Concrete subclass of AbstractLLM for testing base-class behaviour
# ---------------------------------------------------------------------------

class ConcreteTestLLM(AbstractLLM):
    def __init__(self):
        self.client = None
        self._completion_response = "response"
        self._streaming_chunks = ["chunk1", "chunk2"]
        self._image_response = "image_response"

    def name(self) -> str:
        return "TestLLM"

    def initialize(self):
        self.client = "initialized"

    def create_streaming_completion(self, messages, thinking=False, volatile_tail=0):
        self._ensure_initialized()
        for c in self._streaming_chunks:
            yield c

    def create_completion(self, messages, thinking=False):
        self._ensure_initialized()
        return self._completion_response

    def create_completion_with_images(self, messages, images, background=False):
        self._ensure_initialized()
        return self._image_response


# ===========================================================================
# TestAbstractLLMRetry
# ===========================================================================

class TestAbstractLLMRetry:
    def test_think_delegates_to_create_completion(self):
        llm = ConcreteTestLLM()
        llm.initialize()
        result = llm.think("sys", "user")
        assert result == "response"

    def test_think_with_images_delegates_to_create_completion_with_images(self):
        llm = ConcreteTestLLM()
        llm.initialize()
        result = llm.think("sys", "user", images=["img1"])
        assert result == "image_response"

    @patch('assistant.llm.AbstractLLM.time.sleep')
    def test_retry_retries_on_failure(self, mock_sleep):
        llm = ConcreteTestLLM()
        llm.initialize()
        call_count = [0]
        def flaky():
            call_count[0] += 1
            if call_count[0] < 3:
                raise RuntimeError("fail")
            return "ok"
        result = llm._retry(flaky, max_retries=3)
        assert result == "ok"
        assert call_count[0] == 3

    @patch('assistant.llm.AbstractLLM.time.sleep')
    def test_retry_raises_after_max_retries(self, mock_sleep):
        llm = ConcreteTestLLM()
        llm.initialize()
        def always_fail():
            raise RuntimeError("permanent")
        with pytest.raises(RuntimeError, match="permanent"):
            llm._retry(always_fail, max_retries=2)

    @patch('assistant.llm.AbstractLLM.time.sleep')
    def test_retry_does_not_repeat_a_refusal(self, mock_sleep):
        """A refusal is deterministic: retrying it only burns time and money, and the caller
        (the autonomous check) has a fallback model for it."""
        llm = ConcreteTestLLM()
        llm.initialize()
        calls = [0]
        def refused():
            calls[0] += 1
            raise LLMRefusalError("refused")
        with pytest.raises(LLMRefusalError):
            llm._retry(refused, max_retries=3)
        assert calls[0] == 1
        mock_sleep.assert_not_called()


# ===========================================================================
# TestChatGPTLLM
# ===========================================================================

class TestChatGPTLLM:
    @patch('assistant.llm.ChatGPTLLM.config')
    @patch('assistant.llm.ChatGPTLLM.OpenAI')
    def test_only_full_thinking_is_refused_more_is_an_ordinary_request(self, mock_openai, mock_config):
        from assistant.llm.ChatGPTLLM import ChatGPTLLM
        mock_config.get_api_key.return_value = "k"
        llm = ChatGPTLLM()
        llm.initialize()
        reply = MagicMock()
        reply.choices = [MagicMock(message=MagicMock(content="ok"))]
        mock_openai.return_value.chat.completions.create.return_value = reply
        with pytest.raises(ThinkingNotSupportedError):
            llm.create_completion([{"role": "user", "content": "hi"}], thinking=Thinking.FULL)
        assert llm.create_completion([{"role": "user", "content": "hi"}], thinking=Thinking.MORE) == "ok"

    @patch('assistant.llm.ChatGPTLLM.config')
    @patch('assistant.llm.ChatGPTLLM.OpenAI')
    def test_initialize_with_base_url(self, MockOpenAI, mock_config):
        mock_config.get_api_key.return_value = "test-key"
        from assistant.llm.ChatGPTLLM import ChatGPTLLM
        llm = ChatGPTLLM(base_url="http://custom")
        llm.initialize()
        MockOpenAI.assert_called_once_with(api_key="test-key", base_url="http://custom")

    @patch('assistant.llm.ChatGPTLLM.config')
    def test_streaming_raises_on_thinking(self, mock_config):
        mock_config.get_api_key.return_value = "k"
        from assistant.llm.ChatGPTLLM import ChatGPTLLM
        llm = ChatGPTLLM()
        llm.client = Mock()
        with pytest.raises(ThinkingNotSupportedError):
            list(llm.create_streaming_completion([], thinking=True))

    @patch('assistant.llm.ChatGPTLLM.config')
    def test_completion_raises_on_thinking(self, mock_config):
        mock_config.get_api_key.return_value = "k"
        from assistant.llm.ChatGPTLLM import ChatGPTLLM
        llm = ChatGPTLLM()
        llm.client = Mock()
        with pytest.raises(ThinkingNotSupportedError):
            llm.create_completion([], thinking=True)

    @patch('assistant.llm.ChatGPTLLM.config')
    def test_streaming_yields_chunks(self, mock_config):
        mock_config.get_api_key.return_value = "k"
        from assistant.llm.ChatGPTLLM import ChatGPTLLM
        llm = ChatGPTLLM()

        chunk1 = Mock()
        chunk1.choices = [Mock(delta=Mock(content="Hello"))]
        chunk2 = Mock()
        chunk2.choices = [Mock(delta=Mock(content=None))]
        chunk3 = Mock()
        chunk3.choices = [Mock(delta=Mock(content=" world"))]

        llm.client = Mock()
        llm.client.chat.completions.create.return_value = [chunk1, chunk2, chunk3]

        result = list(llm.create_streaming_completion([{"role": "user", "content": "hi"}]))
        assert result == ["Hello", " world"]

    @patch('assistant.llm.ChatGPTLLM.config')
    def test_process_image_bytes_png(self, mock_config):
        mock_config.get_api_key.return_value = "k"
        from assistant.llm.ChatGPTLLM import ChatGPTLLM
        png_bytes = b'\x89PNG' + b'\x00' * 10
        result = ChatGPTLLM._process_image(png_bytes)
        assert result.startswith("data:image/png;base64,")

    @patch('assistant.llm.ChatGPTLLM.config')
    def test_process_image_bytes_jpeg(self, mock_config):
        mock_config.get_api_key.return_value = "k"
        from assistant.llm.ChatGPTLLM import ChatGPTLLM
        jpeg_bytes = b'\xff\xd8\xff' + b'\x00' * 10
        result = ChatGPTLLM._process_image(jpeg_bytes)
        assert result.startswith("data:image/jpeg;base64,")

    @patch('assistant.llm.ChatGPTLLM.config')
    def test_process_image_base64_string(self, mock_config):
        mock_config.get_api_key.return_value = "k"
        from assistant.llm.ChatGPTLLM import ChatGPTLLM
        b64 = base64.b64encode(b"test image data").decode()
        result = ChatGPTLLM._process_image(b64)
        assert result.startswith("data:image/jpeg;base64,")
        assert b64 in result

    @patch('assistant.llm.ChatGPTLLM.config')
    def test_process_image_unsupported_type(self, mock_config):
        mock_config.get_api_key.return_value = "k"
        from assistant.llm.ChatGPTLLM import ChatGPTLLM
        with pytest.raises(ValueError, match="Unsupported image format"):
            ChatGPTLLM._process_image(12345)

    @patch('assistant.llm.ChatGPTLLM.config')
    def test_create_completion_with_images(self, mock_config):
        mock_config.get_api_key.return_value = "k"
        from assistant.llm.ChatGPTLLM import ChatGPTLLM
        llm = ChatGPTLLM()
        llm.client = Mock()

        response = Mock()
        response.choices = [Mock(message=Mock(content="saw image"))]
        llm.client.chat.completions.create.return_value = response

        png_bytes = b'\x89PNG' + b'\x00' * 10
        msgs = [{"role": "user", "content": "what is this?"}]
        result = llm.create_completion_with_images(msgs, [png_bytes])
        assert result == "saw image"

        call_args = llm.client.chat.completions.create.call_args
        processed = call_args[1]['messages']
        assert processed[0]['role'] == 'user'
        assert isinstance(processed[0]['content'], list)
        assert processed[0]['content'][0]['type'] == 'text'
        assert processed[0]['content'][1]['type'] == 'image_url'

    @patch('assistant.llm.ChatGPTLLM.config')
    def test_images_ride_the_first_user_message_only(self, mock_config):
        """The photo backfill sends a system message, the user's question, and on a follow-up
        the earlier turns too: the pictures belong to the first user turn, once; a later user
        turn keeps its plain text and the system message is never turned into parts."""
        mock_config.get_api_key.return_value = "k"
        from assistant.llm.ChatGPTLLM import ChatGPTLLM
        messages = [
            {"role": "system", "content": "analyst"},
            {"role": "user", "content": "what is this?"},
            {"role": "assistant", "content": "a lamp"},
            {"role": "user", "content": "and this?"},
        ]
        out = ChatGPTLLM._image_messages(messages, [b'\x89PNG' + b'\x00' * 10, b'\xff\xd8\xff' + b'\x00' * 10])
        assert out[0] == messages[0] and out[2] == messages[2]
        assert [part["type"] for part in out[1]["content"]] == ["text", "image_url", "image_url"]
        assert out[1]["content"][0]["text"] == "what is this?"
        assert out[3] == {"role": "user", "content": "and this?"}
        assert messages[1]["content"] == "what is this?"   # the caller's list is left as it was

    @patch('assistant.llm.ChatGPTLLM.config')
    def test_completion_strips_cache_markers_from_system(self, mock_config):
        # Regression for the marker leak: OpenAI-compatible backends (and every
        # ChatGPTLLM subclass — Grok, OpenRouter, …) received Claude's cache
        # breakpoint markers as literal system-prompt text.
        mock_config.get_api_key.return_value = "k"
        from assistant.llm.ChatGPTLLM import ChatGPTLLM
        llm = ChatGPTLLM()
        llm.client = Mock()

        response = Mock()
        response.choices = [Mock(message=Mock(content="ok"))]
        llm.client.chat.completions.create.return_value = response

        llm.create_completion([
            {"role": "system", "content": "rules<<<CACHE_BREAKPOINT>>>live data"},
            {"role": "user", "content": "hi"},
        ])
        sent = llm.client.chat.completions.create.call_args[1]['messages']
        assert sent[0] == {"role": "system", "content": "ruleslive data"}
        assert sent[1] == {"role": "user", "content": "hi"}

    @patch('assistant.llm.ChatGPTLLM.config')
    def test_completion_with_images_strips_cache_markers_from_system(self, mock_config):
        # The image path builds its messages separately from create_completion, so it
        # needs the same strip — a camera query on a GPT tier must not leak markers.
        mock_config.get_api_key.return_value = "k"
        from assistant.llm.ChatGPTLLM import ChatGPTLLM
        llm = ChatGPTLLM()
        llm.client = Mock()

        response = Mock()
        response.choices = [Mock(message=Mock(content="saw image"))]
        llm.client.chat.completions.create.return_value = response

        png_bytes = b'\x89PNG' + b'\x00' * 10
        llm.create_completion_with_images([
            {"role": "system", "content": "rules<<<CACHE_BREAKPOINT>>>live data"},
            {"role": "user", "content": "what is this?"},
        ], [png_bytes])
        sent = llm.client.chat.completions.create.call_args[1]['messages']
        assert sent[0] == {"role": "system", "content": "ruleslive data"}
        assert sent[1]['content'][1]['type'] == 'image_url'

    @patch('assistant.llm.ChatGPTLLM.config')
    def test_ensure_initialized_raises(self, mock_config):
        mock_config.get_api_key.return_value = "k"
        from assistant.llm.ChatGPTLLM import ChatGPTLLM
        llm = ChatGPTLLM()
        with pytest.raises(RuntimeError, match="not initialized"):
            llm.create_completion([])


# ===========================================================================
# TestClaudeLLM
# ===========================================================================

class TestClaudeLLM:
    @pytest.mark.parametrize("model, internal", [
        ("claude-fable-5-1", True), ("claude-mythos-5-1", True), ("claude-opus-5", True), ("claude-sonnet-5", True),
        ("claude-opus-4-8", False), ("claude-sonnet-4-5", False), ("claude-haiku-4-5", False),
    ])
    @patch('assistant.llm.ClaudeLLM.config')
    def test_handles_thinking_internally(self, mock_config, model, internal):
        """The 5-generation families think adaptively on their own; a <thinking>-tag instruction
        gets the request refused as reasoning extraction, so nothing may send them one. An
        older model only thinks when asked and must keep the instruction."""
        mock_config.get_api_key.return_value = "k"
        from assistant.llm.ClaudeLLM import ClaudeLLM
        assert ClaudeLLM(model).handles_thinking_internally() is internal

    @patch('assistant.llm.ClaudeLLM.config')
    def test_prefers_prepended_user_query_true_for_non_fable(self, mock_config):
        """Non-Fable Claude prepends the user query: it helps intelligence enough to be
        worth defeating prompt caching (the caller disables caching when prepending)."""
        mock_config.get_api_key.return_value = "k"
        from assistant.llm.ClaudeLLM import ClaudeLLM
        assert ClaudeLLM(model="claude-sonnet-4-5").prefers_prepended_user_query() is True

    @patch('assistant.llm.ClaudeLLM.config')
    def test_prefers_prepended_user_query_false_for_fable(self, mock_config):
        """Fable opts out of prepending so the shared system-prompt prefix stays
        cacheable (prepending puts per-request bytes at position 0 of the prefix)."""
        mock_config.get_api_key.return_value = "k"
        from assistant.llm.ClaudeLLM import ClaudeLLM
        assert ClaudeLLM(model="claude-fable-5").prefers_prepended_user_query() is False

    @patch('assistant.llm.ClaudeLLM.config')
    def test_convert_messages_extracts_system(self, mock_config):
        mock_config.get_api_key.return_value = "k"
        from assistant.llm.ClaudeLLM import ClaudeLLM
        llm = ClaudeLLM()
        messages = [
            {"role": "system", "content": "sys msg"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "tool", "content": "should be filtered"},
        ]
        system, converted = llm._convert_messages(messages)
        assert system == "sys msg"
        assert len(converted) == 2
        assert converted[0]['role'] == 'user'
        assert converted[1]['role'] == 'assistant'

    @patch('assistant.llm.ClaudeLLM.config')
    def test_build_api_params_without_thinking(self, mock_config):
        mock_config.get_api_key.return_value = "k"
        from assistant.llm.ClaudeLLM import ClaudeLLM
        llm = ClaudeLLM()
        messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
        params = llm._build_api_params(messages, thinking=False, stream=True)
        # The cap the request is sent with is the one the request processor names in the
        # note a round that hits it circles back with.
        assert params['max_tokens'] == llm.output_cap()
        assert 'thinking' not in params
        assert params['stream'] is True

    @patch('assistant.llm.ClaudeLLM.config')
    def test_build_api_params_with_thinking(self, mock_config):
        mock_config.get_api_key.return_value = "k"
        from assistant.llm.ClaudeLLM import ClaudeLLM
        llm = ClaudeLLM()
        messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
        params = llm._build_api_params(messages, thinking=True, stream=False)
        assert params['max_tokens'] == ClaudeLLM.THINKING_MAX_TOKENS
        assert params['thinking'] == {"type": "adaptive"}
        assert params['stream'] is False

    @patch('assistant.llm.ClaudeLLM.config')
    def test_non_streaming_request_carries_an_explicit_timeout(self, mock_config):
        """Above ~21k max_tokens the SDK refuses a non-streaming call unless a timeout is
        given; a streaming call needs none."""
        mock_config.get_api_key.return_value = "k"
        from assistant.llm.ClaudeLLM import ClaudeLLM
        llm = ClaudeLLM()
        messages = [{"role": "user", "content": "hi"}]
        assert llm._build_api_params(messages, thinking=True, stream=False)['timeout'] == llm.NON_STREAMING_TIMEOUT
        assert 'timeout' not in llm._build_api_params(messages, thinking=True, stream=True)

    @patch('assistant.llm.ClaudeLLM.config')
    def test_thinking_cap_passes_the_sdk_non_streaming_guard(self, mock_config):
        """The real SDK check: the cap with the explicit timeout must not raise the
        'Streaming is required' ValueError, and would without it."""
        mock_config.get_api_key.return_value = "k"
        from anthropic import Anthropic
        from assistant.llm.ClaudeLLM import ClaudeLLM
        client = Anthropic(api_key="k")
        with pytest.raises(ValueError, match="Streaming is required"):
            client._calculate_nonstreaming_timeout(ClaudeLLM.THINKING_MAX_TOKENS, None)
        llm = ClaudeLLM()
        params = llm._build_api_params([{"role": "user", "content": "hi"}], thinking=True, stream=False)
        assert params['timeout'].read >= 3600 * params['max_tokens'] / 128_000

    @patch('assistant.llm.ClaudeLLM.config')
    def test_completion_records_the_stop_reason(self, mock_config):
        """A caller that persists a reply verbatim (the memory curator) must be able to
        tell a reply cut off at max_tokens from a finished one."""
        mock_config.get_api_key.return_value = "k"
        from assistant.llm.ClaudeLLM import ClaudeLLM
        llm = ClaudeLLM()
        assert llm.last_stop_reason is None
        llm.client = Mock()
        response = Mock()
        response.content = [Mock(type='text', text='half a memory ba')]
        response.usage = Mock(cache_creation_input_tokens=0, cache_read_input_tokens=0, input_tokens=0,
                              output_tokens=32000)
        response.stop_reason = 'max_tokens'
        llm.client.messages.create.return_value = response

        with patch('assistant.llm.ClaudeLLM.info') as mock_info:
            text = llm.create_completion([{"role": "user", "content": "curate"}], thinking=True)

        assert text == 'half a memory ba'
        assert llm.last_stop_reason == 'max_tokens'
        assert any("max_tokens" in str(c.args[0]) for c in mock_info.call_args_list)

        response.stop_reason = 'end_turn'
        llm.create_completion([{"role": "user", "content": "curate"}], thinking=True)
        assert llm.last_stop_reason == 'end_turn'

    @patch('assistant.llm.ClaudeLLM.config')
    def test_create_message_uses_beta_when_fast(self, mock_config):
        mock_config.get_api_key.return_value = "k"
        from assistant.llm.ClaudeLLM import ClaudeLLM
        llm = ClaudeLLM(speed="fast")
        llm.client = Mock()

        api_params = {"model": "test", "messages": [], "max_tokens": 100, "stream": False}
        llm._create_message(api_params)

        llm.client.beta.messages.create.assert_called_once()
        call_kwargs = llm.client.beta.messages.create.call_args
        assert call_kwargs[1]['speed'] == "fast"
        assert ClaudeLLM.FAST_MODE_BETA in call_kwargs[1]['betas']

    @patch('assistant.llm.ClaudeLLM.config')
    def test_create_message_uses_standard_when_not_fast(self, mock_config):
        mock_config.get_api_key.return_value = "k"
        from assistant.llm.ClaudeLLM import ClaudeLLM
        llm = ClaudeLLM()
        llm.client = Mock()

        api_params = {"model": "test", "messages": [], "max_tokens": 100, "stream": False}
        llm._create_message(api_params)

        llm.client.messages.create.assert_called_once()
        llm.client.beta.messages.create.assert_not_called()

    @patch('assistant.llm.ClaudeLLM.time.sleep')
    @patch('assistant.llm.ClaudeLLM.config')
    def test_streaming_529_retry_succeeds(self, mock_config, mock_sleep):
        mock_config.get_api_key.return_value = "k"
        from assistant.llm.ClaudeLLM import ClaudeLLM
        import httpx
        from anthropic import APIStatusError

        llm = ClaudeLLM()
        llm.client = Mock()

        error_response = httpx.Response(529, request=httpx.Request("POST", "https://api.anthropic.com"))
        api_error = APIStatusError("overloaded", response=error_response, body=None)

        event = Mock(type='content_block_delta', delta=Mock(type='text_delta', text='recovered'))
        llm.client.messages.create.side_effect = [api_error, [event]]

        result = list(llm.create_streaming_completion([{"role": "user", "content": "hi"}]))
        assert result == ['recovered']
        mock_sleep.assert_called_once_with(0.5)
        assert llm.client.messages.create.call_count == 2

    @patch('assistant.llm.ClaudeLLM.config')
    def test_streaming_non_529_error_raises(self, mock_config):
        mock_config.get_api_key.return_value = "k"
        from assistant.llm.ClaudeLLM import ClaudeLLM
        import httpx
        from anthropic import APIStatusError

        llm = ClaudeLLM()
        llm.client = Mock()

        error_response = httpx.Response(400, request=httpx.Request("POST", "https://api.anthropic.com"))
        api_error = APIStatusError("bad request", response=error_response, body=None)
        llm.client.messages.create.side_effect = api_error

        with pytest.raises(APIStatusError) as exc_info:
            list(llm.create_streaming_completion([{"role": "user", "content": "hi"}]))
        assert exc_info.value.status_code == 400

    @patch('assistant.llm.ClaudeLLM.time.sleep')
    @patch('assistant.llm.ClaudeLLM.config')
    def test_completion_529_retry_succeeds(self, mock_config, mock_sleep):
        mock_config.get_api_key.return_value = "k"
        from assistant.llm.ClaudeLLM import ClaudeLLM
        import httpx
        from anthropic import APIStatusError

        llm = ClaudeLLM()
        llm.client = Mock()

        error_response = httpx.Response(529, request=httpx.Request("POST", "https://api.anthropic.com"))
        api_error = APIStatusError("overloaded", response=error_response, body=None)

        block = Mock(type='text', text='recovered')
        response = Mock()
        response.content = [block]
        response.usage = Mock(input_tokens=10, cache_creation_input_tokens=0, cache_read_input_tokens=0)

        llm.client.messages.create.side_effect = [api_error, response]

        result = llm.create_completion([{"role": "user", "content": "hi"}])
        assert result == "recovered"
        mock_sleep.assert_called_once_with(0.5)

    @patch('assistant.llm.ClaudeLLM.config')
    def test_completion_non_529_error_raises(self, mock_config):
        mock_config.get_api_key.return_value = "k"
        from assistant.llm.ClaudeLLM import ClaudeLLM
        import httpx
        from anthropic import APIStatusError

        llm = ClaudeLLM()
        llm.client = Mock()

        error_response = httpx.Response(400, request=httpx.Request("POST", "https://api.anthropic.com"))
        api_error = APIStatusError("bad request", response=error_response, body=None)
        llm.client.messages.create.side_effect = api_error

        with pytest.raises(APIStatusError) as exc_info:
            llm.create_completion([{"role": "user", "content": "hi"}])
        assert exc_info.value.status_code == 400

    @patch('assistant.llm.ClaudeLLM.config')
    def test_is_retryable_status_529(self, mock_config):
        mock_config.get_api_key.return_value = "k"
        from assistant.llm.ClaudeLLM import ClaudeLLM
        import httpx
        from anthropic import APIStatusError

        error_response = httpx.Response(529, request=httpx.Request("POST", "https://api.anthropic.com"))
        e = APIStatusError("overloaded", response=error_response, body=None)
        assert ClaudeLLM._is_retryable(e) is True

    @patch('assistant.llm.ClaudeLLM.config')
    def test_is_retryable_body_type(self, mock_config):
        mock_config.get_api_key.return_value = "k"
        from assistant.llm.ClaudeLLM import ClaudeLLM
        import httpx
        from anthropic import APIStatusError

        error_response = httpx.Response(500, request=httpx.Request("POST", "https://api.anthropic.com"))
        body = {"error": {"type": "overloaded_error", "message": "Overloaded"}}
        e = APIStatusError("overloaded", response=error_response, body=body)
        assert ClaudeLLM._is_retryable(e) is True

    @patch('assistant.llm.ClaudeLLM.config')
    def test_is_retryable_non_overloaded_error(self, mock_config):
        mock_config.get_api_key.return_value = "k"
        from assistant.llm.ClaudeLLM import ClaudeLLM
        import httpx
        from anthropic import APIStatusError

        error_response = httpx.Response(400, request=httpx.Request("POST", "https://api.anthropic.com"))
        e = APIStatusError("bad request", response=error_response, body=None)
        assert ClaudeLLM._is_retryable(e) is False

    @patch('assistant.llm.ClaudeLLM.config')
    def test_streaming_yields_text_deltas(self, mock_config):
        mock_config.get_api_key.return_value = "k"
        from assistant.llm.ClaudeLLM import ClaudeLLM
        llm = ClaudeLLM()
        llm.client = Mock()

        event1 = Mock(type='content_block_delta', delta=Mock(type='text_delta', text='Hello'))
        event2 = Mock(type='content_block_start', delta=Mock(type='start'))
        event3 = Mock(type='content_block_delta', delta=Mock(type='text_delta', text=' world'))

        llm.client.messages.create.return_value = [event1, event2, event3]

        result = list(llm.create_streaming_completion([{"role": "user", "content": "hi"}]))
        assert result == ['Hello', ' world']

    @patch('assistant.llm.ClaudeLLM.config')
    def test_completion_concatenates_text_blocks(self, mock_config):
        mock_config.get_api_key.return_value = "k"
        from assistant.llm.ClaudeLLM import ClaudeLLM
        llm = ClaudeLLM()
        llm.client = Mock()

        block1 = Mock(type='text', text='Part 1')
        block2 = Mock(type='thinking', text='thought')
        block3 = Mock(type='text', text=' Part 2')

        response = Mock()
        response.content = [block1, block2, block3]
        response.usage = Mock(input_tokens=10, cache_creation_input_tokens=0, cache_read_input_tokens=0)
        llm.client.messages.create.return_value = response

        result = llm.create_completion([{"role": "user", "content": "hi"}])
        assert result == "Part 1 Part 2"

    @patch('assistant.llm.ClaudeLLM.config')
    def test_build_api_params_cache_control_with_list_content(self, mock_config):
        """When previous message content is a list (image blocks), cache_control is added to last block."""
        mock_config.get_api_key.return_value = "k"
        from assistant.llm.ClaudeLLM import ClaudeLLM
        llm = ClaudeLLM()
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "abc"}},
                {"type": "text", "text": "What is this?"},
            ]},
            {"role": "assistant", "content": "It's a cat."},
            {"role": "user", "content": "Tell me more."},
        ]
        params = llm._build_api_params(messages, thinking=False, stream=True)
        prev_msg = params['messages'][-2]
        assert isinstance(prev_msg['content'], list)
        assert prev_msg["content"][-1].get("cache_control") == {"type": "ephemeral", "ttl": llm.history_cache_ttl}

    _ROUNDS = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Marcus (Telegram): is the garage door closed?"},
        {"role": "assistant", "content": 'door_state("garage")'},
        {"role": "user", "content": "result: \nclosed"},
    ]

    @staticmethod
    def _marked(params) -> list[int]:
        return [i for i, m in enumerate(params['messages'])
                if isinstance(m['content'], list) and any('cache_control' in b for b in m['content'])]

    @patch('assistant.llm.ClaudeLLM.config')
    def test_a_model_with_a_byte_stable_tail_caches_the_last_message(self, mock_config):
        """The next round's prefix ends at this round's last message, so that is where the
        one history breakpoint goes — a command result is then written once and read after."""
        mock_config.get_api_key.return_value = "k"
        from assistant.llm.ClaudeLLM import ClaudeLLM
        params = ClaudeLLM("claude-fable-5-1")._build_api_params(self._ROUNDS, thinking=False, stream=True)
        assert self._marked(params) == [len(params['messages']) - 1]

    @patch('assistant.llm.ClaudeLLM.config')
    def test_a_model_whose_tail_is_re_rendered_caches_the_message_before_it(self, mock_config):
        mock_config.get_api_key.return_value = "k"
        from assistant.llm.ClaudeLLM import ClaudeLLM
        params = ClaudeLLM("claude-sonnet-4-5")._build_api_params(self._ROUNDS, thinking=False, stream=True)
        assert self._marked(params) == [len(params['messages']) - 2]

    @patch('assistant.llm.ClaudeLLM.config')
    def test_a_volatile_tail_moves_the_breakpoint_off_the_messages_nobody_sends_again(self, mock_config):
        """The autonomous check folds its instruction onto the history for one call: a
        breakpoint there is a write no later request reads, so it goes on the message before."""
        mock_config.get_api_key.return_value = "k"
        from assistant.llm.ClaudeLLM import ClaudeLLM
        llm = ClaudeLLM("claude-fable-5-1")
        params = llm._build_api_params(self._ROUNDS, thinking=False, stream=False, volatile_tail=1)
        assert self._marked(params) == [len(params['messages']) - 2]
        assert llm._message_cache_index(volatile_tail=2) == -3
        assert llm._message_cache_index(volatile_tail=0) == -1

    @patch('assistant.llm.ClaudeLLM.config')
    def test_a_follow_up_round_caches_its_whole_tail_for_the_short_tier(self, mock_config):
        """A round after a command ends in the result the next round repeats verbatim and no
        later request does: the breakpoint sits on the last message, volatile tail or not, at
        the follow-up TTL — a 5m write where nothing reads it past the request, instead of a
        1h write at twice the price or, with the breakpoint moved in front of an oversized
        result, the whole tail re-sent uncached by every later round."""
        mock_config.get_api_key.return_value = "k"
        from assistant.llm.ClaudeLLM import ClaudeLLM
        llm = ClaudeLLM("claude-fable-5-1", history_cache_ttl="1h")
        params = llm._build_api_params(self._ROUNDS, thinking=False, stream=True, volatile_tail=3, follow_up=True)
        assert self._marked(params) == [len(params['messages']) - 1]
        breakpoint = params['messages'][-1]['content'][-1]['cache_control']
        assert breakpoint['ttl'] == ClaudeLLM.FOLLOW_UP_CACHE_TTL
        assert breakpoint['ttl'] != llm.history_cache_ttl
        first_round = llm._build_api_params(self._ROUNDS, thinking=False, stream=True)
        assert first_round['messages'][-1]['content'][-1]['cache_control']['ttl'] == "1h"

    @patch('assistant.llm.ClaudeLLM.config')
    def test_a_follow_up_round_on_a_re_rendered_tail_model_keeps_the_message_before(self, mock_config):
        """A model whose trailing turn is re-rendered each request still gets the breakpoint
        one message back on a follow-up round; the follow-up only lifts the volatile tail."""
        mock_config.get_api_key.return_value = "k"
        from assistant.llm.ClaudeLLM import ClaudeLLM
        llm = ClaudeLLM("claude-sonnet-4-5")
        assert llm._message_cache_index(volatile_tail=3, follow_up=True) == -2

    @patch('assistant.llm.ClaudeLLM.config')
    def test_the_follow_up_flag_reaches_the_request_through_the_stream(self, mock_config):
        """A chat round's follow-up is streamed, so the flag must travel that path."""
        mock_config.get_api_key.return_value = "k"
        from assistant.llm.ClaudeLLM import ClaudeLLM
        llm = ClaudeLLM("claude-fable-5-1")
        llm.client = Mock()
        seen = {}
        def capture(api_params):
            seen['messages'] = api_params['messages']; yield "ok"
        with patch.object(llm, '_stream_with_status_retry', side_effect=capture):
            list(llm.create_streaming_completion(self._ROUNDS, volatile_tail=1, follow_up=True))
        sent = seen['messages']
        assert sent[-1]['content'][-1]['cache_control'] == {"type": "ephemeral", "ttl": ClaudeLLM.FOLLOW_UP_CACHE_TTL}

    @patch('assistant.llm.ClaudeLLM.config')
    def test_the_volatile_tail_reaches_the_request_through_create_completion(self, mock_config):
        mock_config.get_api_key.return_value = "k"
        from assistant.llm.ClaudeLLM import ClaudeLLM
        llm = ClaudeLLM("claude-fable-5-1")
        llm.client = Mock()
        text = Mock(); text.type = "text"; text.text = "NO"
        llm.client.messages.create.return_value = Mock(content=[text], usage=None, stop_reason="end_turn")
        llm.create_completion(self._ROUNDS, volatile_tail=1)
        sent = llm.client.messages.create.call_args.kwargs['messages']
        assert [i for i, m in enumerate(sent) if isinstance(m['content'], list)
                and any('cache_control' in b for b in m['content'])] == [len(sent) - 2]

    @patch('assistant.llm.ClaudeLLM.config')
    def test_the_volatile_tail_reaches_the_request_through_the_stream(self, mock_config):
        """A chat round's follow-up is streamed: an oversized command result at its tail
        must move the breakpoint there too, not only on the non-streaming path."""
        mock_config.get_api_key.return_value = "k"
        from assistant.llm.ClaudeLLM import ClaudeLLM
        llm = ClaudeLLM("claude-fable-5-1")
        llm.client = Mock()
        seen = {}
        def capture(api_params):
            seen['messages'] = api_params['messages']; yield "ok"
        with patch.object(llm, '_stream_with_status_retry', side_effect=capture):
            list(llm.create_streaming_completion(self._ROUNDS, volatile_tail=1))
        sent = seen['messages']
        assert [i for i, m in enumerate(sent) if isinstance(m['content'], list)
                and any('cache_control' in b for b in m['content'])] == [len(sent) - 2]

    @patch('assistant.llm.ClaudeLLM.config')
    def test_an_image_turn_at_the_tail_is_not_the_breakpoint(self, mock_config):
        """The history keeps an image turn as its text alone, so the turn as sent never comes
        back: caching it would be a write without a read."""
        mock_config.get_api_key.return_value = "k"
        from assistant.llm.ClaudeLLM import ClaudeLLM
        messages = self._ROUNDS + [
            {"role": "assistant", "content": "Closed. Anything else?"},
            {"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "abc"}},
                {"type": "text", "text": "Marcus (Telegram): what is this?"},
            ]},
        ]
        params = ClaudeLLM("claude-fable-5-1")._build_api_params(messages, thinking=False, stream=True)
        assert self._marked(params) == [len(params['messages']) - 2]

    @patch('assistant.llm.ClaudeLLM.config')
    def test_a_lone_message_gets_no_history_breakpoint(self, mock_config):
        mock_config.get_api_key.return_value = "k"
        from assistant.llm.ClaudeLLM import ClaudeLLM
        lone = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
        params = ClaudeLLM("claude-fable-5-1")._build_api_params(lone, thinking=False, stream=True)
        assert self._marked(params) == []
        params = ClaudeLLM("claude-fable-5-1")._build_api_params(lone, thinking=False, stream=True, volatile_tail=1)
        assert self._marked(params) == []

    @patch('assistant.llm.ClaudeLLM.BYTE_PREFIX_TAIL', False)
    @patch('assistant.llm.ClaudeLLM.config')
    def test_with_the_byte_prefix_tail_switched_off_the_last_message_is_not_cached(self, mock_config):
        mock_config.get_api_key.return_value = "k"
        from assistant.llm.ClaudeLLM import ClaudeLLM
        params = ClaudeLLM("claude-fable-5-1")._build_api_params(self._ROUNDS, thinking=False, stream=True)
        assert self._marked(params) == [len(params['messages']) - 2]

    @patch('assistant.llm.ClaudeLLM.config')
    def test_only_a_model_that_uses_the_prefix_cache_and_thinks_on_its_own_gets_the_byte_stable_tail(self, mock_config):
        """An older model would lose its <thinking> tag instruction in that mode, and one that
        gets the query prepended to the prompt has no prefix cache to protect."""
        mock_config.get_api_key.return_value = "k"
        from assistant.llm.ClaudeLLM import ClaudeLLM
        assert ClaudeLLM("claude-fable-5-1").prefers_byte_prefix_tail()
        assert ClaudeLLM("claude-opus-5-5").prefers_byte_prefix_tail()
        assert not ClaudeLLM("claude-sonnet-4-5").prefers_byte_prefix_tail()
        assert not ClaudeLLM("claude-opus-5").prefers_byte_prefix_tail()

    @patch('assistant.llm.ClaudeLLM.config')
    def test_opus_5_5_keeps_its_prompt_prefix_cacheable_and_opus_5_does_not(self, mock_config):
        """"claude-opus-5" is a prefix of "claude-opus-5-5": the older model must still get
        the prepended query."""
        mock_config.get_api_key.return_value = "k"
        from assistant.llm.ClaudeLLM import ClaudeLLM
        assert not ClaudeLLM("claude-opus-5-5").prefers_prepended_user_query()
        assert ClaudeLLM("claude-opus-5").prefers_prepended_user_query()

    @patch('assistant.llm.ClaudeLLM.config')
    def test_build_api_params_cache_control_with_string_content(self, mock_config):
        """When previous message content is a string, it's wrapped in a list with cache_control."""
        mock_config.get_api_key.return_value = "k"
        from assistant.llm.ClaudeLLM import ClaudeLLM
        llm = ClaudeLLM()
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "follow up"},
        ]
        params = llm._build_api_params(messages, thinking=False, stream=True)
        prev_msg = params['messages'][-2]
        assert isinstance(prev_msg['content'], list)
        assert prev_msg['content'][0]['type'] == 'text'
        assert prev_msg["content"][0].get("cache_control") == {"type": "ephemeral", "ttl": llm.history_cache_ttl}

    @patch('assistant.llm.ClaudeLLM.config')
    def test_history_breakpoint_takes_its_own_ttl_after_the_system_blocks(self, mock_config):
        # The system blocks keep the static TTL and the history breakpoint its own, shorter one —
        # the order Anthropic accepts (longer first); the reverse is rejected by the API.
        mock_config.get_api_key.return_value = "k"
        from assistant.llm.ClaudeLLM import ClaudeLLM
        llm = ClaudeLLM("claude-fable-5-1", static_cache_ttl="1h", history_cache_ttl="5m")
        messages = [
            {"role": "system", "content": f"stable{ClaudeLLM.CACHE_BREAKPOINT_MARKER}tail"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "follow up"},
        ]
        params = llm._build_api_params(messages, thinking=False, stream=True)
        assert params['system'][0]['cache_control']['ttl'] == "1h"
        history_breakpoint = params['messages'][llm._message_cache_index()]['content'][-1]['cache_control']
        assert history_breakpoint['ttl'] == "5m"

    @patch('assistant.llm.ClaudeLLM.config')
    def test_a_per_request_prefix_puts_every_block_on_the_short_tier(self, mock_config):
        """With the user query prepended at byte 0 no later request shares the prefix: only
        the request's own follow-up rounds read it, seconds later, so the system blocks and
        the history breakpoint both take the follow-up TTL instead of the 1h tiers — the
        same reads for a 1.25x write instead of 2x. Off again, the tiers come back: the flag
        is set per request on a long-lived instance."""
        mock_config.get_api_key.return_value = "k"
        from assistant.llm.ClaudeLLM import ClaudeLLM
        llm = ClaudeLLM("claude-sonnet-4-5", static_cache_ttl="1h", history_cache_ttl="1h")
        messages = [
            {"role": "system", "content": f"Just a heads up: the query{ClaudeLLM.CACHE_BREAKPOINT_MARKER}"
                                          f"stable{ClaudeLLM.CACHE_BREAKPOINT_MARKER}tail"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "the query"},
        ]
        llm.set_per_request_prefix(True)
        params = llm._build_api_params(messages, thinking=False, stream=True)
        assert [b['cache_control']['ttl'] for b in params['system'] if 'cache_control' in b] == ["5m", "5m"]
        history_breakpoint = params['messages'][llm._message_cache_index()]['content'][-1]['cache_control']
        assert history_breakpoint['ttl'] == ClaudeLLM.FOLLOW_UP_CACHE_TTL
        llm.set_per_request_prefix(False)
        params = llm._build_api_params(messages, thinking=False, stream=True)
        assert [b['cache_control']['ttl'] for b in params['system'] if 'cache_control' in b] == ["1h", "1h"]
        assert params['messages'][llm._message_cache_index()]['content'][-1]['cache_control']['ttl'] == "1h"

    @patch('assistant.llm.ClaudeLLM.config')
    def test_build_api_params_skips_cache_on_empty_string_block(self, mock_config):
        """A blank [-2] turn (e.g. an interrupted reply stored empty) must not get
        cache_control — Anthropic rejects cache_control on empty text blocks."""
        mock_config.get_api_key.return_value = "k"
        from assistant.llm.ClaudeLLM import ClaudeLLM
        llm = ClaudeLLM()
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "   "},
            {"role": "user", "content": "hej"},
        ]
        params = llm._build_api_params(messages, thinking=False, stream=True)
        prev_msg = params['messages'][-2]
        assert prev_msg['content'] == "   "

    @patch('assistant.llm.ClaudeLLM.config')
    def test_build_api_params_skips_cache_on_empty_text_list_block(self, mock_config):
        """When the [-2] block is a list ending in an empty text block, skip caching."""
        mock_config.get_api_key.return_value = "k"
        from assistant.llm.ClaudeLLM import ClaudeLLM
        llm = ClaudeLLM()
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": [{"type": "text", "text": ""}]},
            {"role": "user", "content": "hej"},
        ]
        params = llm._build_api_params(messages, thinking=False, stream=True)
        prev_msg = params['messages'][-2]
        assert 'cache_control' not in prev_msg['content'][-1]

    @patch('assistant.llm.ClaudeLLM.config')
    def test_cache_breakpoint_marker_splits_system_into_two_blocks(self, mock_config):
        """Marker in system message produces a 1h-cached static prefix and an uncached volatile tail."""
        mock_config.get_api_key.return_value = "k"
        from assistant.llm.ClaudeLLM import ClaudeLLM
        llm = ClaudeLLM(static_cache_ttl="1h")
        sys_msg = "static stable rules\n<<<CACHE_BREAKPOINT>>>\nlive home automation data"
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": "hi"},
        ]
        params = llm._build_api_params(messages, thinking=False, stream=True)
        system = params['system']
        assert isinstance(system, list)
        assert len(system) == 2
        assert system[0]['text'] == "static stable rules"
        assert system[0]['cache_control'] == {"type": "ephemeral", "ttl": "1h"}
        assert system[1]['text'] == "live home automation data"
        assert 'cache_control' not in system[1]

    @patch('assistant.llm.ClaudeLLM.config')
    def test_no_marker_falls_back_to_single_5m_block(self, mock_config):
        """Without the marker, system stays as one ephemeral block (existing behavior)."""
        mock_config.get_api_key.return_value = "k"
        from assistant.llm.ClaudeLLM import ClaudeLLM
        llm = ClaudeLLM()
        messages = [
            {"role": "system", "content": "no marker here"},
            {"role": "user", "content": "hi"},
        ]
        params = llm._build_api_params(messages, thinking=False, stream=True)
        system = params['system']
        assert isinstance(system, list)
        assert len(system) == 1
        assert system[0]['text'] == "no marker here"
        assert system[0]['cache_control'] == {"type": "ephemeral"}

    @patch('assistant.llm.ClaudeLLM.config')
    def test_marker_stripped_when_prompt_cache_disabled(self, mock_config):
        """When prompt_cache=False, the marker must not leak into the system prompt."""
        mock_config.get_api_key.return_value = "k"
        from assistant.llm.ClaudeLLM import ClaudeLLM
        llm = ClaudeLLM(prompt_cache=False)
        sys_msg = "before<<<CACHE_BREAKPOINT>>>after"
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": "hi"},
        ]
        params = llm._build_api_params(messages, thinking=False, stream=True)
        assert params['system'] == "beforeafter"

    @patch('assistant.llm.ClaudeLLM.config')
    def test_marker_with_empty_tail_omits_second_block(self, mock_config):
        """If the tail after the marker is empty, only the cached static block is sent."""
        mock_config.get_api_key.return_value = "k"
        from assistant.llm.ClaudeLLM import ClaudeLLM
        llm = ClaudeLLM()
        messages = [
            {"role": "system", "content": "all static<<<CACHE_BREAKPOINT>>>"},
            {"role": "user", "content": "hi"},
        ]
        params = llm._build_api_params(messages, thinking=False, stream=True)
        system = params['system']
        assert len(system) == 1
        assert system[0]['text'] == "all static"
        assert system[0]['cache_control'] == {"type": "ephemeral", "ttl": llm.static_cache_ttl}

    @patch('assistant.llm.ClaudeLLM.config')
    def test_multiple_markers_produce_one_cached_block_each(self, mock_config):
        """Each marker ends a cached block (stable → volatile); text after the
        last marker is the uncached tail."""
        mock_config.get_api_key.return_value = "k"
        from assistant.llm.ClaudeLLM import ClaudeLLM
        llm = ClaudeLLM(static_cache_ttl="1h")
        sys_msg = ("instructions\n<<<CACHE_BREAKPOINT>>>\nplaylists\n"
                   "<<<CACHE_BREAKPOINT>>>\nweather and home data\n"
                   "<<<CACHE_BREAKPOINT>>>\nvolatile tail")
        params = llm._build_api_params(
            [{"role": "system", "content": sys_msg}, {"role": "user", "content": "hi"}],
            thinking=False, stream=True,
        )
        system = params['system']
        assert [b['text'] for b in system] == [
            "instructions", "playlists", "weather and home data", "volatile tail",
        ]
        for cached_block in system[:3]:
            assert cached_block['cache_control'] == {"type": "ephemeral", "ttl": "1h"}
        assert 'cache_control' not in system[3]

    @patch('assistant.llm.ClaudeLLM.config')
    def test_markers_beyond_limit_join_the_uncached_tail(self, mock_config):
        """Anthropic allows 4 breakpoints per request and one is reserved for the
        message-history breakpoint — extra system markers must not produce a 5th
        block; their segments ride uncached instead."""
        mock_config.get_api_key.return_value = "k"
        from assistant.llm.ClaudeLLM import ClaudeLLM
        llm = ClaudeLLM()
        sys_msg = "<<<CACHE_BREAKPOINT>>>".join(["s1", "s2", "s3", "s4", "s5"])  # 4 markers
        params = llm._build_api_params(
            [{"role": "system", "content": sys_msg}, {"role": "user", "content": "hi"}],
            thinking=False, stream=True,
        )
        system = params['system']
        cached = [b for b in system if 'cache_control' in b]
        assert len(cached) == ClaudeLLM.MAX_SYSTEM_CACHE_BLOCKS
        assert [b['text'] for b in cached] == ["s1", "s2", "s3"]
        assert system[-1]['text'] == "s4\n\ns5"
        assert 'cache_control' not in system[-1]

    @patch('assistant.llm.ClaudeLLM.config')
    def test_adjacent_markers_skip_empty_cached_block(self, mock_config):
        """Two markers with nothing between them must not emit an empty cached
        block — the API rejects cache_control on empty text."""
        mock_config.get_api_key.return_value = "k"
        from assistant.llm.ClaudeLLM import ClaudeLLM
        llm = ClaudeLLM()
        sys_msg = "stable<<<CACHE_BREAKPOINT>>>\n\n<<<CACHE_BREAKPOINT>>>tail"
        params = llm._build_api_params(
            [{"role": "system", "content": sys_msg}, {"role": "user", "content": "hi"}],
            thinking=False, stream=True,
        )
        system = params['system']
        assert [b['text'] for b in system] == ["stable", "tail"]
        assert 'cache_control' in system[0]
        assert 'cache_control' not in system[1]

    @staticmethod
    def _image_llm(mock_config, **kwargs):
        mock_config.get_api_key.return_value = "k"
        from assistant.llm.ClaudeLLM import ClaudeLLM
        llm = ClaudeLLM(**kwargs)
        llm.client = Mock()
        response = Mock()
        response.content = [Mock(type='text', text='ok')]
        response.usage = Mock(input_tokens=1200, cache_creation_input_tokens=0, cache_read_input_tokens=5000,
                              output_tokens=500)
        llm.client.messages.create.return_value = response
        return llm

    @patch('assistant.llm.ClaudeLLM.config')
    def test_image_completion_caches_system_by_breakpoint(self, mock_config):
        """A chat prompt sent with an image (a camera question) is cached block by block
        like any chat request: the markers become breakpoints and never reach the model."""
        llm = self._image_llm(mock_config)

        llm.create_completion_with_images(
            [{"role": "system", "content": "rules\n<<<CACHE_BREAKPOINT>>>\nlive data"},
             {"role": "user", "content": "what's on the camera?"}],
            [b'\x89PNG1234'],
        )

        system = llm.client.messages.create.call_args.kwargs['system']
        texts = [b['text'] for b in system]
        assert not any("<<<CACHE_BREAKPOINT>>>" in t for t in texts)
        assert texts == ["rules", "live data"]
        assert 'cache_control' in system[0] and 'cache_control' not in system[1]

    @patch('assistant.llm.ClaudeLLM.config')
    def test_image_completion_caches_a_markerless_system_whole(self, mock_config):
        """The photo indexer's prompt has no markers: the whole of it (instructions and memory
        bank, the same bytes for every photo of a backfill) goes out as one cached block, and
        the image stays in the user turn behind it."""
        llm = self._image_llm(mock_config)

        llm.create_completion_with_images(
            [{"role": "system", "content": "You are a photo analyst.\n# memory\n..."},
             {"role": "user", "content": "- Date taken: 2026-10-01\nAnalyze this photo."}],
            [b'\xff\xd8jpeg'],
        )

        params = llm.client.messages.create.call_args.kwargs
        assert params['system'] == [{"type": "text", "text": "You are a photo analyst.\n# memory\n...",
                                     "cache_control": {"type": "ephemeral"}}]
        assert [b['type'] for b in params['messages'][-1]['content']] == ['image', 'text']

    @patch('assistant.llm.ClaudeLLM.config')
    def test_image_completion_without_prompt_cache_strips_markers(self, mock_config):
        """With caching off the markers are still stripped — they must never reach the model
        as literal prompt text."""
        llm = self._image_llm(mock_config, prompt_cache=False)

        llm.create_completion_with_images(
            [{"role": "system", "content": "rules\n<<<CACHE_BREAKPOINT>>>\nlive data"},
             {"role": "user", "content": "what's on the camera?"}],
            [b'\x89PNG1234'],
        )

        system = llm.client.messages.create.call_args.kwargs['system']
        assert isinstance(system, str)
        assert "<<<CACHE_BREAKPOINT>>>" not in system
        assert "rules" in system and "live data" in system

    @patch('assistant.llm.ClaudeLLM.config')
    def test_image_completion_empty_reply_returns_empty_and_logs_diagnostics(self, mock_config):
        """The Fable empty-reply quirk on image calls: no text blocks must surface as ''
        (so ImageAnalyzer's retry/fallback kicks in) rather than raising, and the
        diagnostic log must carry stop_reason + block types to tell refusal apart
        from the empty-token quirk."""
        mock_config.get_api_key.return_value = "k"
        from assistant.llm.ClaudeLLM import ClaudeLLM
        llm = ClaudeLLM()
        llm.client = Mock()
        response = Mock()
        response.content = [Mock(type='thinking')]
        response.stop_reason = 'end_turn'
        response.usage = Mock(input_tokens=1200, cache_creation_input_tokens=0, cache_read_input_tokens=0,
                              output_tokens=4)
        llm.client.messages.create.return_value = response

        with patch('assistant.llm.ClaudeLLM.info') as mock_info:
            result = llm.create_completion_with_images(
                [{"role": "user", "content": "what's in this photo?"}],
                [b'\x89PNG1234'],
            )

        assert result == ''
        logged = " ".join(str(c.args[0]) for c in mock_info.call_args_list)
        assert "stop_reason=end_turn" in logged
        assert "thinking" in logged

    def test_cache_creation_breakdown_splits_5m_and_1h(self):
        """Anthropic's per-TTL split is read through so each write bills at its own rate."""
        from assistant.llm.ClaudeLLM import ClaudeLLM
        usage = Mock(cache_creation=Mock(ephemeral_5m_input_tokens=300,
                                         ephemeral_1h_input_tokens=1200))
        assert ClaudeLLM._cache_creation_breakdown(usage, 1500) == (300, 1200)

    def test_cache_creation_breakdown_falls_back_to_5m_when_detail_absent(self):
        """Without the per-TTL detail (older API / bare mock), all created tokens count as 5m."""
        from assistant.llm.ClaudeLLM import ClaudeLLM
        usage = Mock(spec=['cache_creation_input_tokens'])  # no .cache_creation attr
        assert ClaudeLLM._cache_creation_breakdown(usage, 800) == (800, 0)


# ===========================================================================
# TestGeminiLLM
# ===========================================================================

class TestGeminiLLM:
    @patch('assistant.llm.GeminiLLM.config')
    def test_convert_messages_maps_roles(self, mock_config):
        mock_config.get_api_key.return_value = "k"
        llm = GeminiLLM()
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "response"},
        ]
        sys_instr, converted = llm._convert_messages(messages)
        assert sys_instr == "sys"
        assert len(converted) == 2
        assert converted[0]['role'] == 'user'
        assert converted[1]['role'] == 'model'
        assert converted[0]['parts'] == [{'text': 'hello'}]

    @patch('assistant.llm.GeminiLLM.config')
    def test_convert_messages_strips_cache_markers_from_system(self, mock_config):
        """Regression for the marker leak: Gemini (the voice tier) received Claude's
        <<<CACHE_BREAKPOINT>>> markers as literal system-prompt text. They mean
        nothing to Gemini and must never reach the model."""
        mock_config.get_api_key.return_value = "k"
        sys_instr, _ = GeminiLLM()._convert_messages([
            {"role": "system", "content": "rules<<<CACHE_BREAKPOINT>>>live data"},
            {"role": "user", "content": "hello"},
        ])
        assert sys_instr == "ruleslive data"

    @patch('assistant.llm.GeminiLLM.config')
    @patch('assistant.llm.GeminiLLM.types')
    def test_streaming_completion(self, mock_types, mock_config):
        mock_config.get_api_key.return_value = "k"
        llm = GeminiLLM()
        llm.client = Mock()

        chunk1 = Mock(text="Hello")
        chunk2 = Mock(text="")
        chunk3 = Mock(text=" world")

        llm.client.models.generate_content_stream.return_value = [chunk1, chunk2, chunk3]

        result = list(llm.create_streaming_completion(
            [{"role": "user", "content": "hi"}], thinking=False
        ))
        assert result == ["Hello", " world"]

    @patch('assistant.llm.GeminiLLM.config')
    def test_min_thinking_level_respects_model_floor(self, mock_config):
        """gemini-3.7-flash and gemini-3.8-flash reject MINIMAL with 400
        INVALID_ARGUMENT — their floor is LOW. Older Flash generations must
        keep MINIMAL (the voice loop's latency floor)."""
        mock_config.get_api_key.return_value = "k"
        from google.genai.types import ThinkingLevel
        assert GeminiLLM("gemini-3.7-flash")._min_thinking_level() == ThinkingLevel.LOW
        assert GeminiLLM("gemini-3.8-flash")._min_thinking_level() == ThinkingLevel.LOW
        assert GeminiLLM("gemini-3.5-flash")._min_thinking_level() == ThinkingLevel.MINIMAL

    @patch('assistant.llm.GeminiLLM.config')
    @patch('assistant.llm.GeminiLLM.types')
    def test_thinking_more_is_an_ordinary_request_on_gemini(self, mock_types, mock_config):
        """The autonomous check's Thinking.MORE buys reasoning only where a backend has levels
        below its best to give; on Gemini it is a chat turn's floor level, never the deep level
        FULL asks for — the same contract Claude and ChatGPT keep for MORE."""
        mock_config.get_api_key.return_value = "k"
        llm = GeminiLLM("gemini-3.5-flash")
        llm.client = Mock()
        llm.client.models.generate_content.return_value = Mock(text="ok")
        levels = {}
        for thinking in (Thinking.NORMAL, Thinking.MORE, Thinking.FULL):
            llm.create_completion([{"role": "user", "content": "hi"}], thinking=thinking)
            levels[thinking] = mock_types.ThinkingConfig.call_args.kwargs["thinking_level"]
        assert levels[Thinking.MORE] == levels[Thinking.NORMAL] == llm._min_thinking_level()
        assert levels[Thinking.FULL] != levels[Thinking.MORE]

    @patch('assistant.llm.GeminiLLM.config')
    @patch('assistant.llm.GeminiLLM.types')
    def test_streaming_37_flash_never_requests_minimal(self, mock_types, mock_config):
        """The non-thinking streaming path is the production voice loop on the
        fast tier — sending MINIMAL to gemini-3.7-flash 400s every request."""
        mock_config.get_api_key.return_value = "k"
        from google.genai.types import ThinkingLevel
        llm = GeminiLLM("gemini-3.7-flash")
        llm.client = Mock()
        llm.client.models.generate_content_stream.return_value = []

        list(llm.create_streaming_completion(
            [{"role": "user", "content": "hi"}], thinking=False
        ))
        assert mock_types.ThinkingConfig.call_args.kwargs['thinking_level'] == ThinkingLevel.LOW

    @patch('assistant.llm.GeminiLLM.config')
    @patch('assistant.llm.GeminiLLM.types')
    def test_streaming_older_flash_keeps_minimal(self, mock_types, mock_config):
        """The floor helper must not blanket-raise every model to LOW — that
        would silently slow the voice loop on models that accept MINIMAL."""
        mock_config.get_api_key.return_value = "k"
        from google.genai.types import ThinkingLevel
        llm = GeminiLLM()
        llm.client = Mock()
        llm.client.models.generate_content_stream.return_value = []

        list(llm.create_streaming_completion(
            [{"role": "user", "content": "hi"}], thinking=False
        ))
        assert mock_types.ThinkingConfig.call_args.kwargs['thinking_level'] == ThinkingLevel.MINIMAL

    @patch('assistant.llm.GeminiLLM.config')
    @patch('assistant.llm.GeminiLLM.types')
    def test_completion_37_flash_never_requests_minimal(self, mock_types, mock_config):
        mock_config.get_api_key.return_value = "k"
        from google.genai.types import ThinkingLevel
        llm = GeminiLLM("gemini-3.7-flash")
        llm.client = Mock()
        llm.client.models.generate_content.return_value = Mock(text="ok")

        llm.create_completion([{"role": "user", "content": "hi"}], thinking=False)
        assert mock_types.ThinkingConfig.call_args.kwargs['thinking_level'] == ThinkingLevel.LOW

    @patch('assistant.llm.GeminiLLM.config')
    @patch('assistant.llm.GeminiLLM.types')
    def test_streaming_thinking_caps_at_low(self, mock_types, mock_config):
        """Streaming is the voice loop — thinking there is deliberately capped
        at LOW (HIGH/MEDIUM add ~5s, too slow for conversation). The completion
        path's near-identical block uses HIGH, so a copy-paste between them
        would silently regress voice latency."""
        mock_config.get_api_key.return_value = "k"
        from google.genai.types import ThinkingLevel
        llm = GeminiLLM()
        llm.client = Mock()
        llm.client.models.generate_content_stream.return_value = []

        list(llm.create_streaming_completion(
            [{"role": "user", "content": "hi"}], thinking=True
        ))
        assert mock_types.ThinkingConfig.call_args.kwargs['thinking_level'] == ThinkingLevel.LOW

    @patch('assistant.llm.GeminiLLM.config')
    @patch('assistant.llm.GeminiLLM.types')
    def test_completion_thinking_uses_high(self, mock_types, mock_config):
        """Non-streaming completions have no voice-latency constraint — thinking
        must stay HIGH, not drift down to the streaming path's LOW cap."""
        mock_config.get_api_key.return_value = "k"
        from google.genai.types import ThinkingLevel
        llm = GeminiLLM()
        llm.client = Mock()
        llm.client.models.generate_content.return_value = Mock(text="ok")

        llm.create_completion([{"role": "user", "content": "hi"}], thinking=True)
        assert mock_types.ThinkingConfig.call_args.kwargs['thinking_level'] == ThinkingLevel.HIGH


# ===========================================================================
# TestGrokLLM
# ===========================================================================

class TestGrokLLM:
    @patch('assistant.llm.ChatGPTLLM.config')
    def test_uses_grok_api_key(self, mock_config):
        mock_config.get_api_key.return_value = "grok-key"
        from assistant.llm.GrokLLM import GrokLLM
        assert GrokLLM().api_key == "grok-key"   # read at use, not captured at construction
        mock_config.get_api_key.assert_called_with('grok')

# ===========================================================================
# TestOpenRouterLLM
# ===========================================================================

class TestOpenRouterLLM:
    @patch('assistant.llm.ChatGPTLLM.config')
    def test_uses_openrouter_api_key(self, mock_config):
        mock_config.get_api_key.return_value = "or-key"
        from assistant.llm.OpenRouterLLM import OpenRouterLLM
        assert OpenRouterLLM(model='test/model').api_key == "or-key"
        mock_config.get_api_key.assert_called_with('openrouter')

# ===========================================================================
# TestQwenLLM
# ===========================================================================

# ===========================================================================
# TestGemmaLLM
# ===========================================================================

# ===========================================================================
# TestChatGPTLLMImageConversion
# ===========================================================================

class TestChatGPTLLMImageConversion:
    @patch('assistant.llm.ChatGPTLLM.config')
    def test_convert_anthropic_content_with_image_blocks(self, mock_config):
        mock_config.get_api_key.return_value = "k"
        from assistant.llm.ChatGPTLLM import ChatGPTLLM
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "abc123"}},
                {"type": "text", "text": "What is this?"},
            ]},
        ]
        result = ChatGPTLLM._convert_anthropic_content(messages)
        assert result[0] == messages[0]  # system message unchanged
        user_content = result[1]['content']
        assert user_content[0] == {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,abc123"}}
        assert user_content[1] == {"type": "text", "text": "What is this?"}

    @patch('assistant.llm.ChatGPTLLM.config')
    def test_convert_anthropic_content_passes_string_content_through(self, mock_config):
        mock_config.get_api_key.return_value = "k"
        from assistant.llm.ChatGPTLLM import ChatGPTLLM
        messages = [
            {"role": "user", "content": "just text"},
        ]
        result = ChatGPTLLM._convert_anthropic_content(messages)
        assert result == messages

    @patch('assistant.llm.ChatGPTLLM.config')
    def test_streaming_with_image_content_converts_messages(self, mock_config):
        mock_config.get_api_key.return_value = "k"
        from assistant.llm.ChatGPTLLM import ChatGPTLLM
        llm = ChatGPTLLM()
        llm.client = Mock()

        chunk = Mock()
        chunk.choices = [Mock(delta=Mock(content="I see a cat"))]
        llm.client.chat.completions.create.return_value = [chunk]

        messages = [
            {"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "imgdata"}},
                {"type": "text", "text": "What is this?"},
            ]},
        ]
        result = list(llm.create_streaming_completion(messages))
        assert result == ["I see a cat"]

        call_args = llm.client.chat.completions.create.call_args
        sent_messages = call_args[1]['messages']
        assert sent_messages[0]['content'][0]['type'] == 'image_url'
        assert 'data:image/jpeg;base64,imgdata' in sent_messages[0]['content'][0]['image_url']['url']

    @patch('assistant.llm.ChatGPTLLM.config')
    def test_completion_with_image_content_converts_messages(self, mock_config):
        mock_config.get_api_key.return_value = "k"
        from assistant.llm.ChatGPTLLM import ChatGPTLLM
        llm = ChatGPTLLM()
        llm.client = Mock()

        response = Mock()
        response.choices = [Mock(message=Mock(content="It's a dog"))]
        llm.client.chat.completions.create.return_value = response

        messages = [
            {"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "imgdata"}},
                {"type": "text", "text": "What is this?"},
            ]},
        ]
        result = llm.create_completion(messages)
        assert result == "It's a dog"

        call_args = llm.client.chat.completions.create.call_args
        sent_messages = call_args[1]['messages']
        assert sent_messages[0]['content'][0]['type'] == 'image_url'


# ===========================================================================
# TestGeminiLLMImageConversion
# ===========================================================================

class TestGeminiLLMImageConversion:
    @patch('assistant.llm.GeminiLLM.config')
    @patch('assistant.llm.GeminiLLM.types')
    def test_convert_messages_with_image_blocks(self, mock_types, mock_config):
        mock_config.get_api_key.return_value = "k"
        mock_types.Part.from_bytes.return_value = "mock_image_part"
        llm = GeminiLLM()

        img_data = base64.b64encode(b"fake image").decode()
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_data}},
                {"type": "text", "text": "What is this?"},
            ]},
        ]
        sys_instr, converted = llm._convert_messages(messages)
        assert sys_instr == "sys"
        assert len(converted) == 1
        assert converted[0]['role'] == 'user'
        assert converted[0]['parts'][0] == "mock_image_part"
        assert converted[0]['parts'][1] == {'text': 'What is this?'}
        mock_types.Part.from_bytes.assert_called_once()

    @patch('assistant.llm.GeminiLLM.config')
    @patch('assistant.llm.GeminiLLM.types')
    def test_convert_messages_string_content_unchanged(self, mock_types, mock_config):
        mock_config.get_api_key.return_value = "k"
        llm = GeminiLLM()
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        _, converted = llm._convert_messages(messages)
        assert converted[0]['parts'] == [{'text': 'hello'}]
        assert converted[1]['role'] == 'model'
        assert converted[1]['parts'] == [{'text': 'hi there'}]

    @patch('assistant.llm.GeminiLLM.config')
    @patch('assistant.llm.GeminiLLM.types')
    def test_create_completion_with_images(self, mock_types, mock_config):
        mock_config.get_api_key.return_value = "k"
        mock_types.Part.from_bytes.return_value = "mock_image_part"
        llm = GeminiLLM()
        llm.client = Mock()

        response = Mock(text="I see a cat")
        llm.client.models.generate_content.return_value = response

        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "What is in this image?"},
        ]
        png_bytes = b'\x89PNG' + b'\x00' * 10
        result = llm.create_completion_with_images(messages, [png_bytes])
        assert result == "I see a cat"
        mock_types.Part.from_bytes.assert_called_once_with(data=png_bytes, mime_type="image/png")


# ===========================================================================
# TestGeminiLLMVideo
# ===========================================================================

class TestGeminiLLMVideo:
    @pytest.fixture(autouse=True)
    def _fresh_outage(self):
        from assistant.llm import GeminiLLM as module
        module._outage.clear()
        yield
        module._outage.clear()

    @patch('assistant.llm.GeminiLLM.config')
    @patch('assistant.llm.GeminiLLM.types')
    def test_think_with_video_sends_inline_clip_before_text(self, mock_types, mock_config):
        mock_config.get_api_key.return_value = "k"
        mock_types.Part.return_value = "video_part"
        llm = GeminiLLM(model="gemini-video")
        llm.client = Mock()
        llm.client.models.generate_content.return_value = Mock(text="A dog runs", usage_metadata=None)

        result = llm.think_with_video("sys<<<CACHE_BREAKPOINT>>>tem", "describe", b"vid", "video/quicktime")

        assert result == "A dog runs"
        mock_types.Blob.assert_called_once_with(data=b"vid", mime_type="video/quicktime")
        mock_types.Part.assert_called_once_with(inline_data=mock_types.Blob.return_value, video_metadata=None)
        mock_types.VideoMetadata.assert_not_called()
        assert mock_types.CreateCachedContentConfig.call_args.kwargs["system_instruction"] == "system"
        call = llm.client.models.generate_content.call_args[1]
        assert call["model"] == "gemini-video"
        assert call["contents"] == [{'role': 'user', 'parts': ["video_part", {'text': "describe"}]}]

    @patch('assistant.llm.GeminiLLM.config')
    @patch('assistant.llm.GeminiLLM.types')
    def test_clips_and_stills_with_the_same_instruction_share_one_cache(self, mock_types, mock_config):
        """The indexer's system instruction is the same from clip to clip, and a clip must not pay
        full price for it when the stills already have it cached."""
        mock_config.get_api_key.return_value = "k"
        llm = GeminiLLM(model="gemini-video")
        llm.client = Mock()
        llm.client.caches.create.return_value = Mock(usage_metadata=Mock(total_token_count=1868))
        llm.client.caches.create.return_value.name = "cachedContents/abc"
        llm.client.models.generate_content.return_value = Mock(text="{}", usage_metadata=None)

        llm.think(system_message="BANK", user_message="d", images=[b"img"], background=True)
        llm.think_with_video("BANK", "describe", b"vid", "video/mp4")
        llm.think_with_video("BANK", "describe", b"vid2", "video/mp4")

        llm.client.caches.create.assert_called_once()
        configs = [c.kwargs for c in mock_types.GenerateContentConfig.call_args_list]
        assert configs == [{"cached_content": "cachedContents/abc"}] * 3

    @patch('assistant.llm.GeminiLLM.config')
    @patch('assistant.llm.GeminiLLM.types')
    def test_a_dropped_cache_resends_the_clip_inline_and_still_deletes_the_upload(self, mock_types, mock_config):
        from assistant.llm.GeminiLLM import INLINE_VIDEO_LIMIT
        llm = self._uploading_llm(mock_config)
        llm.client.caches.create.return_value = Mock(usage_metadata=Mock(total_token_count=1868))
        llm.client.caches.create.return_value.name = "cachedContents/abc"
        llm.client.models.generate_content.side_effect = [
            RuntimeError("403 PERMISSION_DENIED: CachedContent not found"), Mock(text="ok", usage_metadata=None)]

        assert llm.think_with_video("BANK", "describe", b"v" * (INLINE_VIDEO_LIMIT + 1), "video/mp4") == "ok"

        configs = [c.kwargs for c in mock_types.GenerateContentConfig.call_args_list]
        assert configs == [{"cached_content": "cachedContents/abc"}, {"system_instruction": "BANK"}]
        llm.client.files.upload.assert_called_once(), "the clip is not uploaded twice for the resend"
        llm.client.files.delete.assert_called_once_with(name="files/abc")
        assert llm._system_caches == {}

    @patch('assistant.llm.GeminiLLM.config')
    @patch('assistant.llm.GeminiLLM.types')
    def test_think_with_video_passes_fps_as_video_metadata(self, mock_types, mock_config):
        mock_config.get_api_key.return_value = "k"
        llm = GeminiLLM()
        llm.client = Mock()
        llm.client.models.generate_content.return_value = Mock(text="ok", usage_metadata=None)

        llm.think_with_video("sys", "user", b"vid", "video/mp4", fps=0.5)

        mock_types.VideoMetadata.assert_called_once_with(fps=0.5)
        mock_types.Part.assert_called_once_with(
            inline_data=mock_types.Blob.return_value, video_metadata=mock_types.VideoMetadata.return_value)

    @staticmethod
    def _uploading_llm(mock_config, states=("ACTIVE",)):
        """A GeminiLLM whose files.upload returns a file in states[0] and files.get walks the rest."""
        mock_config.get_api_key.return_value = "k"
        llm = GeminiLLM(model="gemini-video")
        llm.client = Mock()
        files = []
        for state in states:
            f = Mock(uri="https://files/abc")
            f.name = "files/abc"
            f.state = Mock()
            f.state.name = state
            files.append(f)
        llm.client.files.upload.return_value = files[0]
        llm.client.files.get.side_effect = files[1:]
        llm.client.models.generate_content.return_value = Mock(text="A long clip", usage_metadata=None)
        return llm

    @patch('assistant.llm.GeminiLLM.config')
    @patch('assistant.llm.GeminiLLM.types')
    def test_large_clip_goes_through_the_files_api(self, mock_types, mock_config):
        from assistant.llm.GeminiLLM import INLINE_VIDEO_LIMIT
        llm = self._uploading_llm(mock_config)
        clip = b"v" * (INLINE_VIDEO_LIMIT + 1)

        result = llm.think_with_video("sys", "describe", clip, "video/mp4")

        assert result == "A long clip"
        upload = llm.client.files.upload.call_args.kwargs
        assert upload["file"].read() == clip
        mock_types.UploadFileConfig.assert_called_once_with(mime_type="video/mp4")
        mock_types.FileData.assert_called_once_with(file_uri="https://files/abc", mime_type="video/mp4")
        mock_types.Part.assert_called_once_with(file_data=mock_types.FileData.return_value, video_metadata=None)
        mock_types.Blob.assert_not_called()
        llm.client.files.delete.assert_called_once_with(name="files/abc")

    @patch('assistant.llm.GeminiLLM.config')
    @patch('assistant.llm.GeminiLLM.types')
    def test_clip_at_the_inline_limit_stays_inline(self, mock_types, mock_config):
        from assistant.llm.GeminiLLM import INLINE_VIDEO_LIMIT
        llm = self._uploading_llm(mock_config)

        llm.think_with_video("sys", "describe", b"v" * INLINE_VIDEO_LIMIT, "video/mp4")

        llm.client.files.upload.assert_not_called()
        mock_types.Blob.assert_called_once()

    @patch('assistant.llm.GeminiLLM.time.sleep')
    @patch('assistant.llm.GeminiLLM.config')
    @patch('assistant.llm.GeminiLLM.types')
    def test_upload_waits_for_processing_to_finish(self, mock_types, mock_config, sleep):
        from assistant.llm.GeminiLLM import INLINE_VIDEO_LIMIT
        llm = self._uploading_llm(mock_config, states=("PROCESSING", "PROCESSING", "ACTIVE"))

        assert llm.think_with_video("sys", "describe", b"v" * (INLINE_VIDEO_LIMIT + 1), "video/mp4") == "A long clip"

        assert llm.client.files.get.call_count == 2
        assert sleep.call_count == 2
        llm.client.files.delete.assert_called_once_with(name="files/abc")

    @patch('assistant.llm.AbstractLLM.time.sleep')
    @patch('assistant.llm.GeminiLLM.config')
    @patch('assistant.llm.GeminiLLM.types')
    def test_failed_processing_raises_and_deletes(self, mock_types, mock_config, sleep):
        from assistant.llm.GeminiLLM import INLINE_VIDEO_LIMIT
        llm = self._uploading_llm(mock_config, states=("FAILED",))

        with pytest.raises(RuntimeError, match="state FAILED"):
            llm.think_with_video("sys", "describe", b"v" * (INLINE_VIDEO_LIMIT + 1), "video/mp4")

        llm.client.models.generate_content.assert_not_called()
        assert llm.client.files.delete.call_count == 3  # once per retry attempt

    @patch('assistant.llm.GeminiLLM.time.monotonic')
    @patch('assistant.llm.GeminiLLM.time.sleep')
    @patch('assistant.llm.GeminiLLM.config')
    @patch('assistant.llm.GeminiLLM.types')
    def test_processing_timeout_raises_and_deletes(self, mock_types, mock_config, sleep, monotonic):
        from assistant.llm.GeminiLLM import INLINE_VIDEO_LIMIT, FILE_PROCESSING_TIMEOUT
        llm = self._uploading_llm(mock_config, states=("PROCESSING",) * 10)
        # The patch replaces the global time.monotonic, so daemon threads can consume
        # entries too: a finite list ran dry under the full suite (StopIteration).
        monotonic.side_effect = itertools.chain([0, 1, 1], itertools.repeat(FILE_PROCESSING_TIMEOUT + 2))

        with pytest.raises(TimeoutError, match="still processing"):
            llm._create_completion_with_video("sys", "describe", b"v" * (INLINE_VIDEO_LIMIT + 1), "video/mp4", None)

        llm.client.files.delete.assert_called_once_with(name="files/abc")

    @patch('assistant.llm.AbstractLLM.time.sleep')
    @patch('assistant.llm.GeminiLLM.config')
    @patch('assistant.llm.GeminiLLM.types')
    def test_uploaded_file_is_deleted_when_generation_fails(self, mock_types, mock_config, sleep):
        from assistant.llm.GeminiLLM import INLINE_VIDEO_LIMIT
        llm = self._uploading_llm(mock_config)
        llm.client.models.generate_content.side_effect = RuntimeError("503")

        with pytest.raises(RuntimeError):
            llm.think_with_video("sys", "describe", b"v" * (INLINE_VIDEO_LIMIT + 1), "video/mp4")

        assert llm.client.files.delete.call_count == 3

    @patch('assistant.llm.GeminiLLM.config')
    @patch('assistant.llm.GeminiLLM.types')
    def test_failed_cleanup_does_not_lose_the_description(self, mock_types, mock_config):
        """The delete is best effort (files expire on their own); a hiccup there must not
        turn a finished, paid-for analysis into a retry."""
        from assistant.llm.GeminiLLM import INLINE_VIDEO_LIMIT
        llm = self._uploading_llm(mock_config)
        llm.client.files.delete.side_effect = RuntimeError("404 already gone")

        assert llm.think_with_video("sys", "describe", b"v" * (INLINE_VIDEO_LIMIT + 1), "video/mp4") == "A long clip"
        assert llm.client.models.generate_content.call_count == 1

    @patch('assistant.llm.GeminiLLM.config')
    @patch('assistant.llm.GeminiLLM.types')
    def test_an_outage_refuses_a_large_clip_before_it_is_uploaded(self, mock_types, mock_config):
        """A refused request must leave no private clip on Gemini: the outage check comes before
        the Files API upload, not after it (where the raise escaped the delete)."""
        from assistant.llm import GeminiLLM as module
        from assistant.llm.AbstractLLM import LLMUnavailable
        from assistant.llm.GeminiLLM import INLINE_VIDEO_LIMIT
        llm = self._uploading_llm(mock_config)
        module._outage.note(RuntimeError("429 RESOURCE_EXHAUSTED spend-based rate limit"), "gemini-video")

        with pytest.raises(LLMUnavailable):
            llm._create_completion_with_video("sys", "describe", b"v" * (INLINE_VIDEO_LIMIT + 1), "video/mp4", None)

        llm.client.files.upload.assert_not_called()
        llm.client.models.generate_content.assert_not_called()

    @patch('assistant.llm.GeminiLLM.config')
    @patch('assistant.llm.GeminiLLM.types')
    def test_empty_response_returns_empty_string_instead_of_none(self, mock_types, mock_config):
        mock_config.get_api_key.return_value = "k"
        llm = GeminiLLM()
        llm.client = Mock()
        blocked = Mock(text=None, usage_metadata=None, prompt_feedback=Mock(block_reason="SAFETY"))
        blocked.candidates = []
        llm.client.models.generate_content.return_value = blocked

        assert llm.think_with_video("sys", "user", b"vid", "video/mp4") == ""
        assert llm.client.models.generate_content.call_count == 1

    @patch('assistant.llm.GeminiLLM.config')
    @patch('assistant.llm.GeminiLLM.types')
    def test_empty_image_response_returns_empty_string_instead_of_none(self, mock_types, mock_config):
        mock_config.get_api_key.return_value = "k"
        llm = GeminiLLM()
        llm.client = Mock()
        blocked = Mock(text=None, usage_metadata=None, prompt_feedback=Mock(block_reason="SAFETY"))
        blocked.candidates = []
        llm.client.models.generate_content.return_value = blocked

        assert llm.think("sys", "describe", images=[b'\xff\xd8\xff' + b'\x00' * 10]) == ""

    @patch('assistant.llm.GeminiLLM.info')
    @patch('assistant.llm.GeminiLLM.config')
    @patch('assistant.llm.GeminiLLM.types')
    def test_image_request_logs_token_usage_with_cache_hits(self, mock_types, mock_config, mock_info):
        """The photo backfill is thousands of image calls; the log has to show what each one cost
        and how much of the prompt the implicit cache served — the video path logged it, the
        image path did not."""
        mock_config.get_api_key.return_value = "k"
        llm = GeminiLLM()
        llm.client = Mock()
        usage = Mock(prompt_token_count=8900, cached_content_token_count=6500,
                     candidates_token_count=480, thoughts_token_count=None, total_token_count=9380)
        llm.client.models.generate_content.return_value = Mock(text="{}", usage_metadata=usage)

        llm.think("sys", "describe", images=[b'\xff\xd8\xff' + b'\x00' * 10])

        usage_lines = [c.args[0] for c in mock_info.call_args_list if "image request used" in c.args[0]]
        assert len(usage_lines) == 1
        assert "8900" in usage_lines[0] and "6500 cached" in usage_lines[0] and "480 output" in usage_lines[0]


    @patch('assistant.llm.GeminiLLM.config')
    @patch('assistant.llm.GeminiLLM.types')
    def test_think_with_video_retries_transient_failure(self, mock_types, mock_config):
        mock_config.get_api_key.return_value = "k"
        llm = GeminiLLM()
        llm.client = Mock()
        llm.client.models.generate_content.side_effect = [RuntimeError("503"), Mock(text="ok", usage_metadata=None)]

        with patch('assistant.llm.AbstractLLM.time.sleep'):
            assert llm.think_with_video("sys", "user", b"vid", "video/mp4") == "ok"
        assert llm.client.models.generate_content.call_count == 2


@pytest.mark.unit
class TestGeminiSystemCache:
    """Bulk image work (background=True) serves its long, identical system instruction from an
    explicit context cache — a tenth of the input price for those tokens — and a user's one-off
    image question does not pay for a cache it would use once."""

    @pytest.fixture(autouse=True)
    def _fresh_outage(self):
        from assistant.llm import GeminiLLM as module
        module._outage.clear()
        yield
        module._outage.clear()

    @staticmethod
    def _llm():
        llm = GeminiLLM(model="gemini-3.8-flash")
        llm.client = Mock()
        llm.client.caches.create.return_value = Mock(name="cachedContents/abc", usage_metadata=Mock(total_token_count=1868))
        llm.client.caches.create.return_value.name = "cachedContents/abc"
        llm.client.models.generate_content.return_value = Mock(text="{}", usage_metadata=None)
        return llm

    @patch('assistant.llm.GeminiLLM.config')
    @patch('assistant.llm.GeminiLLM.types')
    def test_background_images_share_one_cache_and_send_no_system_instruction(self, mock_types, mock_config):
        llm = self._llm()
        for _ in range(3):
            llm.think(system_message="BANK", user_message="describe", images=[b"img"], background=True)
        llm.client.caches.create.assert_called_once()
        assert llm.client.caches.create.call_args.kwargs["model"] == "gemini-3.8-flash"
        assert mock_types.CreateCachedContentConfig.call_args.kwargs["system_instruction"] == "BANK"
        configs = [c.kwargs for c in mock_types.GenerateContentConfig.call_args_list]
        assert len(configs) == 3 and all(c == {"cached_content": "cachedContents/abc"} for c in configs)

    @patch('assistant.llm.GeminiLLM.config')
    @patch('assistant.llm.GeminiLLM.types')
    def test_a_foreground_image_question_never_makes_a_cache(self, mock_types, mock_config):
        llm = self._llm()
        llm.think(system_message="Answer briefly", user_message="what is this", images=[b"img"])
        llm.client.caches.create.assert_not_called()
        assert mock_types.GenerateContentConfig.call_args.kwargs == {"system_instruction": "Answer briefly"}

    @patch('assistant.llm.GeminiLLM.config')
    @patch('assistant.llm.GeminiLLM.types')
    def test_a_system_instruction_gemini_refuses_to_cache_is_sent_inline_and_not_retried(self, mock_types, mock_config):
        llm = self._llm()
        llm.client.caches.create.side_effect = RuntimeError("400 INVALID_ARGUMENT: below the minimum token count")
        for _ in range(2):
            llm.think(system_message="short", user_message="describe", images=[b"img"], background=True)
        llm.client.caches.create.assert_called_once()
        assert all(c.kwargs == {"system_instruction": "short"} for c in mock_types.GenerateContentConfig.call_args_list)

    @patch('assistant.llm.GeminiLLM.config')
    @patch('assistant.llm.GeminiLLM.types')
    def test_a_different_system_instruction_gets_its_own_cache(self, mock_types, mock_config):
        llm = self._llm()
        llm.think(system_message="BANK v1", user_message="d", images=[b"img"], background=True)
        llm.think(system_message="BANK v2", user_message="d", images=[b"img"], background=True)
        assert llm.client.caches.create.call_count == 2

    @patch('assistant.llm.GeminiLLM.config')
    @patch('assistant.llm.GeminiLLM.types')
    def test_a_cache_gemini_has_dropped_is_forgotten_and_the_image_resent_inline(self, mock_types, mock_config):
        llm = self._llm()
        llm.client.models.generate_content.side_effect = [
            RuntimeError("403 PERMISSION_DENIED: CachedContent not found"), Mock(text="ok", usage_metadata=None)]
        assert llm.think(system_message="BANK", user_message="d", images=[b"img"], background=True) == "ok"
        configs = [c.kwargs for c in mock_types.GenerateContentConfig.call_args_list]
        assert configs == [{"cached_content": "cachedContents/abc"}, {"system_instruction": "BANK"}]
        assert llm._system_caches == {}, "the next photo makes a fresh cache instead of hitting the dead one"

    @patch('assistant.llm.GeminiLLM.time')
    @patch('assistant.llm.GeminiLLM.config')
    @patch('assistant.llm.GeminiLLM.types')
    def test_a_cache_past_half_its_life_is_extended_so_a_steady_stream_keeps_one_cache(self, mock_types, mock_config, mock_time):
        from assistant.llm import GeminiLLM as module
        llm = self._llm()
        clock = [1000.0]
        mock_time.monotonic.side_effect = lambda: clock[0]
        llm.think(system_message="BANK", user_message="d", images=[b"img"], background=True)
        clock[0] += module.SYSTEM_CACHE_TTL_SECONDS * 0.4
        llm.think(system_message="BANK", user_message="d", images=[b"img"], background=True)
        llm.client.caches.update.assert_not_called()
        clock[0] += module.SYSTEM_CACHE_TTL_SECONDS * 0.2
        llm.think(system_message="BANK", user_message="d", images=[b"img"], background=True)
        llm.client.caches.update.assert_called_once()
        assert llm.client.caches.update.call_args.kwargs["name"] == "cachedContents/abc"
        llm.client.caches.create.assert_called_once()


@pytest.mark.unit
class TestGeminiBatchedImages:
    """The backfill's stills as inline batch jobs: half price, one reply per request in order,
    None for what the job did not answer, and a job that outlives the wait is cancelled."""

    @pytest.fixture(autouse=True)
    def _fresh_outage(self):
        from assistant.llm import GeminiLLM as module
        module._outage.clear()
        yield
        module._outage.clear()

    @staticmethod
    def _job(name, state, replies=None):
        job = Mock()
        job.name = name
        job.state = Mock()
        job.state.name = state
        job.error = None
        if replies is not None:
            job.dest.inlined_responses = [
                Mock(response=None, error="boom") if r is None else Mock(response=Mock(text=r, usage_metadata=None), error=None)
                for r in replies]
        return job

    @staticmethod
    def _llm():
        llm = GeminiLLM(model="gemini-3.8-flash")
        llm.client = Mock()
        llm.client.caches.create.side_effect = RuntimeError("400 below minimum")    # no cache: plain system instruction
        return llm

    @patch('assistant.llm.GeminiLLM.time.sleep')
    @patch('assistant.llm.GeminiLLM.config')
    def test_replies_come_back_in_request_order_with_none_for_a_failed_request(self, mock_config, mock_sleep):
        llm = self._llm()
        llm.client.batches.create.return_value = self._job("batches/1", "JOB_STATE_PENDING")
        llm.client.batches.get.side_effect = [self._job("batches/1", "JOB_STATE_RUNNING"),
                                              self._job("batches/1", "JOB_STATE_SUCCEEDED", ["one", None, "three"])]
        replies = llm.think_batched_with_images("SYS", [("u1", b"a"), ("u2", b"b"), ("u3", b"c")], max_wait_seconds=600)
        assert replies == ["one", None, "three"]
        src = llm.client.batches.create.call_args.kwargs["src"]
        assert [r["contents"][0]["parts"][1]["text"] for r in src] == ["u1", "u2", "u3"]
        assert [r["contents"][0]["parts"][0]["inline_data"]["data"] for r in src] == [b"a", b"b", b"c"]
        assert all(r["config"] == {"system_instruction": "SYS"} for r in src)

    @patch('assistant.llm.GeminiLLM.time.sleep')
    @patch('assistant.llm.GeminiLLM.config')
    def test_a_cached_system_instruction_rides_in_every_batched_request(self, mock_config, mock_sleep):
        llm = self._llm()
        cache = Mock(usage_metadata=Mock(total_token_count=1868))
        cache.name = "cachedContents/abc"
        llm.client.caches.create.side_effect = None
        llm.client.caches.create.return_value = cache
        llm.client.batches.create.return_value = self._job("batches/1", "JOB_STATE_SUCCEEDED", ["one"])
        llm.client.batches.get.return_value = self._job("batches/1", "JOB_STATE_SUCCEEDED", ["one"])
        with patch('assistant.llm.GeminiLLM.types') as mock_types:
            mock_types.CreateCachedContentConfig = Mock()
            assert llm.think_batched_with_images("SYS", [("u1", b"a")]) == ["one"]
        src = llm.client.batches.create.call_args.kwargs["src"]
        assert src[0]["config"] == {"cached_content": "cachedContents/abc"}

    @patch('assistant.llm.GeminiLLM.time.sleep')
    @patch('assistant.llm.GeminiLLM.config')
    def test_images_are_split_into_jobs_under_the_inline_limit(self, mock_config, mock_sleep):
        from assistant.llm import GeminiLLM as module
        llm = self._llm()
        big = b"x" * (module.INLINE_BATCH_LIMIT // 2 + 1)
        jobs = [self._job("batches/1", "JOB_STATE_SUCCEEDED", ["a"]), self._job("batches/2", "JOB_STATE_SUCCEEDED", ["b", "c"])]
        llm.client.batches.create.side_effect = jobs
        llm.client.batches.get.side_effect = lambda name: next(j for j in jobs if j.name == name)
        replies = llm.think_batched_with_images("SYS", [("u1", big), ("u2", big), ("u3", b"small")])
        assert replies == ["a", "b", "c"]
        assert [len(c.kwargs["src"]) for c in llm.client.batches.create.call_args_list] == [1, 2]

    def test_the_split_budgets_the_base64_the_images_become_on_the_wire(self):
        """Two images whose raw bytes fit the limit together but whose base64 does not go in
        separate jobs — a job over the inline limit is rejected whole and falls back to live calls."""
        from assistant.llm import GeminiLLM as module
        raw = int(module.INLINE_BATCH_LIMIT * 0.4)    # 2 × 0.4 raw fits; 2 × 0.4 × 4/3 = 1.07 does not
        assert module.GeminiLLM._batch_chunks([("u1", b"x" * raw), ("u2", b"x" * raw)]) == [[0], [1]]
        assert module.GeminiLLM._batch_chunks([("u1", b"x" * raw), ("u2", b"x" * (raw // 2))]) == [[0, 1]]

    @patch('assistant.llm.GeminiLLM.time.sleep')
    @patch('assistant.llm.GeminiLLM.config')
    def test_a_job_that_outlives_the_wait_is_cancelled_and_its_requests_unanswered(self, mock_config, mock_sleep):
        llm = self._llm()
        llm.client.batches.create.return_value = self._job("batches/1", "JOB_STATE_PENDING")
        llm.client.batches.get.return_value = self._job("batches/1", "JOB_STATE_RUNNING")
        with patch('assistant.llm.GeminiLLM.time.time', side_effect=[0, 0, 10, 20, 2000, 2000, 2000]):
            replies = llm.think_batched_with_images("SYS", [("u1", b"a"), ("u2", b"b")], max_wait_seconds=1800)
        assert replies == [None, None]
        llm.client.batches.cancel.assert_called_once_with(name="batches/1")

    @patch('assistant.llm.GeminiLLM.config')
    def test_a_refused_submission_means_no_batch_at_all(self, mock_config):
        llm = self._llm()
        llm.client.batches.create.side_effect = RuntimeError("429 RESOURCE_EXHAUSTED: enqueued tokens")
        assert llm.think_batched_with_images("SYS", [("u1", b"a")]) is None

    @patch('assistant.llm.GeminiLLM.config')
    def test_no_requests_is_an_empty_answer_without_a_job(self, mock_config):
        llm = self._llm()
        assert llm.think_batched_with_images("SYS", []) == []
        llm.client.batches.create.assert_not_called()

    @patch('assistant.llm.GeminiLLM.config')
    def test_a_refused_submission_for_credits_is_recorded_for_every_other_caller(self, mock_config):
        from assistant.llm.AbstractLLM import LLMUnavailable
        llm = self._llm()
        llm.client.batches.create.side_effect = RuntimeError("402 RESOURCE_EXHAUSTED: Your prepayment credits are depleted")
        assert llm.think_batched_with_images("SYS", [("u1", b"a")]) is None
        with pytest.raises(LLMUnavailable):
            llm.think("s", "u", images=[b"a"], background=True)
        llm.client.models.generate_content.assert_not_called()


@pytest.mark.unit
class TestGeminiOutage:
    """One 402 or 429 is enough: every GeminiLLM shares the verdict, background work fails fast
    without a call, and a probe that gets through clears it."""

    @pytest.fixture(autouse=True)
    def _fresh_outage(self):
        from assistant.llm import GeminiLLM as module
        module._outage.clear()
        yield
        module._outage.clear()

    @staticmethod
    def _llm():
        llm = GeminiLLM(model="gemini-3.8-flash")
        llm.client = Mock()
        llm.client.caches.create.side_effect = RuntimeError("400 below minimum")
        return llm

    CREDITS = RuntimeError("402 RESOURCE_EXHAUSTED. {'error': {'code': 402, 'message': 'Your prepayment credits are depleted.'}}")
    RATE = RuntimeError("429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your spend-based rate limit.'}}")

    @patch('assistant.llm.GeminiLLM.config')
    @patch('assistant.llm.GeminiLLM.types')
    def test_a_402_is_raised_once_as_unavailable_and_then_refused_without_a_call(self, mock_types, mock_config):
        from assistant.llm.AbstractLLM import LLMUnavailable
        llm = self._llm()
        llm.client.models.generate_content.side_effect = self.CREDITS
        with pytest.raises(LLMUnavailable):
            llm.think("s", "u", images=[b"a"], background=True)
        assert llm.client.models.generate_content.call_count == 1, "not retried: a 402 does not pass on the second try"
        other = GeminiLLM(model="gemini-3.5-flash")
        other.client = Mock()
        with pytest.raises(LLMUnavailable):
            other.think("s", "u")    # a user's request, another instance: the key is empty for everyone
        other.client.models.generate_content.assert_not_called()
        assert llm.out_of_credits() and other.unavailable() and "402" in other.unavailable()

    @patch('assistant.llm.GeminiLLM.config')
    @patch('assistant.llm.GeminiLLM.types')
    def test_a_stream_refused_on_its_first_chunk_records_the_outage_and_raises_unavailable(self, mock_types, mock_config):
        """generate_content_stream is lazy: the 429 arrives on the first next(), not on the call."""
        from assistant.llm.AbstractLLM import LLMUnavailable
        llm = self._llm()

        def refused():
            raise self.RATE
            yield    # noqa — a generator that fails on its first chunk
        llm.client.models.generate_content_stream.return_value = refused()
        with pytest.raises(LLMUnavailable):
            list(llm.create_streaming_completion([{"role": "user", "content": "hi"}]))
        assert llm.unavailable() and not llm.out_of_credits(), "the stream's 429 backs background work off"

    @patch('assistant.llm.GeminiLLM.config')
    @patch('assistant.llm.GeminiLLM.types')
    def test_a_stream_clears_a_backoff_only_once_a_chunk_arrived(self, mock_types, mock_config):
        from assistant.llm import GeminiLLM as module
        llm = self._llm()
        module._outage.note(self.RATE, "gemini-3.8-flash")

        def broken():
            raise RuntimeError("connection reset")
            yield
        llm.client.models.generate_content_stream.return_value = broken()
        with pytest.raises(RuntimeError):
            list(llm.create_streaming_completion([{"role": "user", "content": "hi"}]))
        assert llm.unavailable(), "a stream that never produced a chunk proves nothing about Gemini"

        llm.client.models.generate_content_stream.return_value = iter([Mock(text="h"), Mock(text=""), Mock(text="i")])
        assert "".join(llm.create_streaming_completion([{"role": "user", "content": "hi"}])) == "hi"
        assert llm.unavailable() is None

    @patch('assistant.llm.GeminiLLM.config')
    @patch('assistant.llm.GeminiLLM.types')
    def test_a_429_backs_off_background_work_but_lets_a_user_request_through(self, mock_types, mock_config):
        from assistant.llm.AbstractLLM import LLMUnavailable
        llm = self._llm()
        llm.client.models.generate_content.side_effect = [self.RATE, Mock(text="hi", usage_metadata=None)]
        with pytest.raises(LLMUnavailable):
            llm.think("s", "u", images=[b"a"], background=True)
        with pytest.raises(LLMUnavailable):
            llm.think("s", "u", images=[b"b"], background=True)
        assert llm.client.models.generate_content.call_count == 1
        assert llm.unavailable() and not llm.out_of_credits()
        assert llm.think("s", "u") == "hi", "the user's own request is not told no on the background's behalf"
        assert llm.unavailable() is None, "a request that got through clears the backoff"

    @patch('assistant.llm.GeminiLLM.time')
    @patch('assistant.llm.GeminiLLM.config')
    @patch('assistant.llm.GeminiLLM.types')
    def test_the_backoff_expires_on_its_own_and_doubles_while_the_429s_keep_coming(self, mock_types, mock_config, mock_time):
        from assistant.llm.AbstractLLM import LLMUnavailable
        from assistant.llm import GeminiLLM as module
        clock = [1000.0]
        mock_time.monotonic.side_effect = lambda: clock[0]
        llm = self._llm()
        llm.client.models.generate_content.side_effect = self.RATE
        with pytest.raises(LLMUnavailable):
            llm.think("s", "u", images=[b"a"], background=True)
        clock[0] += module.RATE_BACKOFF_SECONDS + 1
        with pytest.raises(LLMUnavailable):
            llm.think("s", "u", images=[b"a"], background=True)
        assert llm.client.models.generate_content.call_count == 2, "after the wait one call probes again"
        assert f"{module.RATE_BACKOFF_SECONDS * 2:.0f}s" in llm.unavailable()

    @patch('assistant.llm.GeminiLLM.time')
    def test_two_429s_in_the_same_window_double_the_backoff_once(self, mock_time):
        """Two workers hitting the limit together: the second notes the outage already on and
        must not double the wait on its own account."""
        from assistant.llm.AbstractLLM import LLMUnavailable
        from assistant.llm import GeminiLLM as module
        clock = [1000.0]
        mock_time.monotonic.side_effect = lambda: clock[0]
        base = module.RATE_BACKOFF_SECONDS
        first = module._outage.note(self.RATE, "gemini-3.8-flash")
        second = module._outage.note(self.RATE, "gemini-3.8-flash")
        assert isinstance(first, LLMUnavailable) and isinstance(second, LLMUnavailable)
        assert f"{base:.0f}s" in str(second) and f"{base * 2:.0f}s" not in str(second)
        clock[0] += base + 1
        third = module._outage.note(self.RATE, "gemini-3.8-flash")
        assert f"{base * 2:.0f}s" in str(third) and f"{base * 4:.0f}s" not in str(third)

    @patch('assistant.llm.GeminiLLM.config')
    @patch('assistant.llm.GeminiLLM.types')
    def test_an_image_request_during_an_outage_is_refused_before_any_cache_is_made(self, mock_types, mock_config):
        from assistant.llm.AbstractLLM import LLMUnavailable
        from assistant.llm import GeminiLLM as module
        module._outage.note(self.RATE, "gemini-3.8-flash")
        llm = self._llm()
        with pytest.raises(LLMUnavailable):
            llm.think("s", "u", images=[b"a"], background=True)
        llm.client.caches.create.assert_not_called()
        llm.client.models.generate_content.assert_not_called()

    @patch('assistant.llm.GeminiLLM.config')
    @patch('assistant.llm.GeminiLLM.types')
    def test_a_batch_during_a_402_is_not_submitted(self, mock_types, mock_config):
        from assistant.llm import GeminiLLM as module
        module._outage.note(self.CREDITS, "gemini-3.8-flash")
        llm = self._llm()
        assert llm.think_batched_with_images("SYS", [("u1", b"a")]) is None
        llm.client.batches.create.assert_not_called()
        llm.client.caches.create.assert_not_called()

    @patch('assistant.llm.GeminiLLM.config')
    @patch('assistant.llm.GeminiLLM.types')
    def test_probe_clears_an_outage_only_when_gemini_answers(self, mock_types, mock_config):
        from assistant.llm.AbstractLLM import LLMUnavailable
        llm = self._llm()
        llm.client.models.generate_content.side_effect = [self.CREDITS, self.CREDITS, Mock(text="ok", usage_metadata=None)]
        with pytest.raises(LLMUnavailable):
            llm.think("s", "u", images=[b"a"], background=True)
        assert llm.probe() is False and llm.out_of_credits()
        assert llm.probe() is True and llm.unavailable() is None

    def test_the_status_code_decides_not_a_402_somewhere_in_the_text(self):
        from assistant.llm import GeminiLLM as module
        from google.genai import errors
        assert module._outage.note(RuntimeError("500 INTERNAL: request 402-77 failed"), "m") is None
        assert module._outage.active(background=True) is None
        verdict = module._outage.note(errors.APIError(402, {"error": {"message": "Payment required."}}), "m")
        assert verdict is not None and module._outage.active(background=True)
        assert module._outage.kind == "credits"

    def test_a_429_by_code_backs_off_background_work(self):
        from assistant.llm import GeminiLLM as module
        from google.genai import errors
        module._outage.note(errors.APIError(429, {"error": {"message": "Quota exceeded."}}), "m")
        assert module._outage.active(background=True) and module._outage.active(background=False) is None

    @patch('assistant.llm.GeminiLLM.time')
    @patch('assistant.llm.GeminiLLM.config')
    @patch('assistant.llm.GeminiLLM.types')
    def test_a_user_request_tries_again_once_per_recheck_window_and_a_success_clears_the_credits_outage(
            self, mock_types, mock_config, mock_time):
        """A 402 the chat hit first must not lock Gemini out until a restart: one user request
        per window is a live attempt, background work stays refused until one succeeds."""
        from assistant.llm.AbstractLLM import LLMUnavailable
        from assistant.llm import GeminiLLM as module
        clock = [1000.0]
        mock_time.monotonic.side_effect = lambda: clock[0]
        llm = self._llm()
        llm.client.models.generate_content.side_effect = [self.CREDITS, self.CREDITS, Mock(text="hi", usage_metadata=None)]
        with pytest.raises(LLMUnavailable):
            llm.think("s", "u")
        with pytest.raises(LLMUnavailable):
            llm.think("s", "u")
        assert llm.client.models.generate_content.call_count == 1, "inside the window the chat is told no without a call"
        clock[0] += module.CREDITS_RECHECK_SECONDS + 1
        with pytest.raises(LLMUnavailable):
            llm.think("s", "u")
        assert llm.client.models.generate_content.call_count == 2 and llm.out_of_credits(), "the window's one attempt, still refused"
        with pytest.raises(LLMUnavailable):
            llm.think("s", "u", images=[b"a"], background=True)
        clock[0] += module.CREDITS_RECHECK_SECONDS + 1
        assert llm.think("s", "u") == "hi"
        assert llm.unavailable() is None, "the attempt that got through clears it for everyone"

    @patch('assistant.llm.GeminiLLM.config')
    @patch('assistant.llm.GeminiLLM.types')
    def test_any_other_error_is_still_the_callers_to_handle(self, mock_types, mock_config):
        llm = self._llm()
        llm.client.models.generate_content.side_effect = RuntimeError("503 UNAVAILABLE")
        with patch('assistant.llm.AbstractLLM.time.sleep'), pytest.raises(RuntimeError):
            llm.think("s", "u", images=[b"a"], background=True)
        assert llm.client.models.generate_content.call_count == 3, "retried as before"
        assert llm.unavailable() is None


# ===========================================================================
# TestOllamaLLM
# ===========================================================================

class TestOllamaLLM:
    DONE = {"done": True, "prompt_eval_count": 1, "prompt_eval_duration": 1e9,
            "eval_count": 1, "eval_duration": 1e9, "total_duration": 2e9}

    def _make_llm(self):
        from assistant.llm.OllamaLLM import OllamaLLM
        return OllamaLLM(model='gemma4:26b', base_url='http://localhost:11434')

    @pytest.fixture(autouse=True)
    def _fresh_thinking_probe(self):
        from assistant.llm.OllamaLLM import OllamaLLM
        from assistant.commands.formats.CommandFormats import CommandFormats
        from assistant.commands.formats.QwenFormat import QwenFormat
        OllamaLLM._capabilities.clear()
        CommandFormats.set_active(QwenFormat)  # an Ollama smart tier: the parsed calls go back as Qwen text
        yield
        OllamaLLM._capabilities.clear()
        CommandFormats.reset()

    def test_under_another_smart_tier_a_parsed_call_is_written_back_in_that_tiers_format(self):
        # The mixed-tier case: an Ollama fast tier under a Claude (Keep) or ds4 (DSML) smart tier
        # must hand the active parser a call it reads, not a <tool_call> block it takes for prose.
        from assistant.llm.OllamaLLM import OllamaLLM
        from assistant.commands.formats.CommandFormats import CommandFormats
        from assistant.commands.formats.DsmlFormat import DsmlFormat
        from assistant.commands.formats.KeepFormat import KeepFormat
        call = {"function": {"name": "dim", "arguments": {"light_name": "Lamp", "dim_percentage": "5"}}}
        for fmt in (KeepFormat, DsmlFormat):
            CommandFormats.set_active(fmt)
            text = OllamaLLM._content_of({"message": {"content": "", "tool_calls": [call]}})
            assert "<tool_call>" not in text
            assert fmt(["dim"]).extract_all(text) == [("dim", ["Lamp", "5"])]

    @patch('assistant.llm.OllamaLLM.config')
    def test_base_url_from_config(self, mock_config):
        mock_config.get.side_effect = \
            lambda key, default=None: 'http://studio:11434' if key == 'ollama.base_url' else default
        from assistant.llm.OllamaLLM import OllamaLLM
        try:
            assert OllamaLLM().base_url == 'http://studio:11434'
        finally:
            OllamaLLM._ssh_log = None

    @patch('assistant.llm.OllamaLLM.config')
    def test_explicit_base_url_wins_over_config(self, mock_config):
        mock_config.get.side_effect = \
            lambda key, default=None: 'http://studio:11434' if key == 'ollama.base_url' else default
        from assistant.llm.OllamaLLM import OllamaLLM
        try:
            assert OllamaLLM(base_url='http://other:11434').base_url == 'http://other:11434'
        finally:
            OllamaLLM._ssh_log = None

    @patch('assistant.llm.OllamaLLM.trace')
    def test_config_trace_logged_once_per_model_and_base_url(self, mock_trace):
        # The handler pool constructs dozens of identical instances (25 handlers
        # x 2 tiers), which flooded startup with ~50 identical config lines.
        # The line must appear once per distinct (model, base_url) — repeated
        # instances stay silent, a genuinely new config still logs.
        from assistant.llm.OllamaLLM import OllamaLLM
        saved = OllamaLLM._logged_configs
        OllamaLLM._logged_configs = set()
        try:
            for _ in range(3):
                OllamaLLM(model='gemma4:26b', base_url='http://localhost:11434')
            OllamaLLM(model='qwen3.8:27b', base_url='http://localhost:11434')
            config_lines = [c for c in mock_trace.call_args_list
                            if 'using base_url' in str(c)]
            assert len(config_lines) == 2
        finally:
            OllamaLLM._logged_configs = saved

    @patch('assistant.llm.OllamaLLM.OllamaLLM.thinks', return_value=True)
    def test_build_request(self, mock_thinks):
        llm = self._make_llm()
        req = llm._build_request([{"role": "user", "content": "hi"}], stream=True)
        assert req == {
            "model": "gemma4:26b",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "keep_alive": -1,
            "think": "medium",   # the default level (OllamaLLM.DEFAULT_THINK)
            "tools": [type(llm).STATIC_TOOL],
        }

    def test_every_request_declares_one_tool_as_the_contract_with_ollamas_call_parser(self):
        # Ollama runs its tool-call parser only when the request declares a tool: with none,
        # a <tool_call> block the model writes is swallowed (plan, host run 2026-10-20). The
        # declared tool must be the same bytes in every request, pre-heats included, since it
        # renders into the prefix the KV cache is matched on.
        llm = self._make_llm()
        real = llm._build_request([{"role": "user", "content": "hi"}], stream=True)
        pre_heat = llm._build_request([{"role": "user", "content": "hi"}], stream=True, num_predict=1)
        assert len(real["tools"]) == 1 and real["tools"][0]["function"]["name"] == "load_toolkit"
        assert pre_heat["tools"] == real["tools"]

    def test_ollama_runs_the_qwen_command_format(self):
        from assistant.commands.formats.QwenFormat import QwenFormat
        assert self._make_llm().command_format() is QwenFormat

    @patch('assistant.llm.OllamaLLM.OllamaLLM.thinks', return_value=True)
    @patch('assistant.llm.OllamaLLM.OllamaLLM._rendered_prompt', return_value=None)
    @patch('assistant.llm.OllamaLLM.requests.post')
    def test_a_parsed_tool_call_is_written_back_as_qwen_text_the_parser_reads(self, mock_post, mock_rendered, mock_thinks):
        # Ollama parses the model's <tool_call> block into message.tool_calls (arguments as
        # strings) in a chunk after the content; the stream must carry it as the text the
        # model wrote, after the content and outside the thinking block.
        import json
        from assistant.commands.formats.QwenFormat import QwenFormat
        call = {"function": {"index": 0, "name": "turn_on", "arguments": {"light_name": "Kitchen lamp"}}}
        mock_post.return_value.iter_lines.return_value = [
            json.dumps({"message": {"thinking": "plan"}}),
            json.dumps({"message": {"content": "Turning it on."}}),
            json.dumps({"message": {"content": "", "tool_calls": [call]}}),
            json.dumps({"message": {"content": ""}, **self.DONE}),
        ]
        out = "".join(self._make_llm().create_streaming_completion([{"role": "user", "content": "hi"}]))
        assert out.startswith("<thinking>plan</thinking>Turning it on.\n<tool_call>")
        fmt = QwenFormat(["turn_on"], table={"turn_on": ["light_name"]})
        assert fmt.extract_all(out) == [("turn_on", ["Kitchen lamp"])]

    @patch('assistant.llm.OllamaLLM.OllamaLLM.thinks', return_value=True)
    @patch('assistant.llm.OllamaLLM.OllamaLLM._rendered_prompt', return_value=None)
    @patch('assistant.llm.OllamaLLM.requests.post')
    def test_a_tool_call_straight_after_thinking_closes_the_block_first(self, mock_post, mock_rendered, mock_thinks):
        import json
        from assistant.commands.formats.QwenFormat import QwenFormat
        call = {"function": {"name": "load_toolkit", "arguments": {"name": ["email", "web"]}}}
        mock_post.return_value.iter_lines.return_value = [
            json.dumps({"message": {"thinking": "plan"}}),
            json.dumps({"message": {"content": "", "tool_calls": [call]}, **self.DONE}),
        ]
        out = "".join(self._make_llm().create_streaming_completion([{"role": "user", "content": "hi"}]))
        assert out.startswith("<thinking>plan</thinking><tool_call>")
        fmt = QwenFormat(["load_toolkit"], table={"load_toolkit": ["name"]}, variadic={"load_toolkit"})
        assert fmt.extract_all(out) == [("load_toolkit", ["email", "web"])]

    def test_a_non_streamed_reply_carries_its_tool_calls_as_text_too(self):
        import json
        from assistant.llm.OllamaLLM import OllamaLLM
        from assistant.commands.formats.QwenFormat import QwenFormat
        data = {"message": {"content": "Dimming.", "tool_calls": [
            {"function": {"name": "dim", "arguments": json.dumps({"light_name": "Lamp", "dim_percentage": 1})}}]}}
        text = OllamaLLM._content_of(data)
        fmt = QwenFormat(["dim"], table={"dim": ["light_name", "dim_percentage"]})
        assert text.startswith("Dimming.\n") and fmt.extract_all(text) == [("dim", ["Lamp", "1"])]

    @patch('assistant.llm.OllamaLLM.note')
    def test_tool_call_text_left_in_the_content_is_noted_but_passed_on(self, mock_note):
        from assistant.llm.OllamaLLM import OllamaLLM
        text = OllamaLLM._content_of({"message": {"content": "<tool_call>\nbroken\n</tool_call>"}})
        assert text == "<tool_call>\nbroken\n</tool_call>"
        assert mock_note.called
        mock_note.reset_mock()
        OllamaLLM._content_of({"message": {"content": "fine", "tool_calls": [{"function": {"name": "restart_server", "arguments": {}}}]}})
        assert not mock_note.called

    def test_build_request_flattens_anthropic_image_blocks(self):
        # Ollama's /api/chat rejects list content with a 400. A message whose
        # content is a list of Anthropic-style blocks (e.g. camera snapshots)
        # must become string content plus a message-level `images` list of raw
        # base64 — otherwise the whole request fails (the real-world 400 bug).
        llm = self._make_llm()
        b64 = base64.b64encode(b"jpegbytes").decode()
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Camera: Front door"},
                {"type": "image", "source": {"type": "base64",
                                             "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": "Camera: Garden"},
            ],
        }]
        req = llm._build_request(messages, stream=True)
        msg = req["messages"][0]
        assert msg["content"] == "Camera: Front door\nCamera: Garden"
        assert msg["images"] == [b64]

    def test_build_request_strips_cache_markers_from_system(self):
        # Regression for the marker leak: Ollama received Claude's cache breakpoint
        # markers as literal system-prompt text. Stripping is deterministic, so
        # Ollama's own KV-cache prefix matching is unaffected.
        llm = self._make_llm()
        req = llm._build_request([
            {"role": "system", "content": "rules<<<CACHE_BREAKPOINT>>>live data"},
            {"role": "user", "content": "hi"},
        ], stream=False)
        assert req["messages"][0]["content"] == "ruleslive data"
        assert req["messages"][1] == {"role": "user", "content": "hi"}

    def test_build_request_collects_multiple_image_blocks(self):
        # The real-world bug surfaced with camera snapshots (plural). A regression
        # that kept only the first image would still pass the single-image test
        # above, so pin the multi-image contract explicitly: every image block's
        # base64 must reach Ollama's message-level `images` list, in order.
        llm = self._make_llm()
        front_b64 = base64.b64encode(b"pool_jpeg").decode()
        garden_b64 = base64.b64encode(b"altan_jpeg").decode()
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Cameras:"},
                {"type": "image", "source": {"type": "base64",
                                             "media_type": "image/jpeg", "data": front_b64}},
                {"type": "image", "source": {"type": "base64",
                                             "media_type": "image/jpeg", "data": garden_b64}},
            ],
        }]
        req = llm._build_request(messages, stream=True)
        assert req["messages"][0]["images"] == [front_b64, garden_b64]

    def test_build_request_caps_tokens_for_pre_heat(self):
        # num_predict is what turns a request into a pre-heat: generate just one
        # token to warm the KV cache instead of a full ~50s completion. A real
        # request passes None and must NOT carry num_predict (test_build_request
        # pins that). Here we pin the pre-heat side so the cap can't silently drop.
        llm = self._make_llm()
        req = llm._build_request([{"role": "user", "content": "hi"}], stream=True, num_predict=1)
        assert req["options"] == {"num_predict": 1}

    @patch('assistant.llm.OllamaLLM.requests.post')
    def test_a_model_without_the_thinking_capability_runs_without_it(self, mock_post):
        # Ollama answers think=true with a 400 for such a model, so the probe of /api/show
        # decides the request flag, and the <thinking>-tag instruction stays available to it.
        mock_post.return_value.json.return_value = {"capabilities": ["completion", "vision"]}
        llm = self._make_llm()
        req = llm._build_request([{"role": "user", "content": "hi"}], stream=True)
        assert req["think"] is False
        assert "options" not in req
        assert llm.handles_thinking_internally() is False

    @patch('assistant.llm.OllamaLLM.requests.post')
    def test_thinking_probe_runs_once_per_model(self, mock_post):
        mock_post.return_value.json.return_value = {"capabilities": ["completion", "thinking"]}
        assert self._make_llm().thinks() is True
        assert self._make_llm().handles_thinking_internally() is True
        assert mock_post.call_count == 1
        assert mock_post.call_args.args[0].endswith("/api/show")

    @patch('assistant.llm.OllamaLLM.requests.post', side_effect=ConnectionError("down"))
    def test_an_unreachable_host_is_assumed_to_think(self, mock_post):
        assert self._make_llm().thinks() is True

    @patch('assistant.llm.OllamaLLM.requests.post')
    def test_a_model_without_the_tools_capability_gets_no_tool_declared(self, mock_post):
        # Ollama answers a request declaring a tool with a 400 ("does not support tools") for
        # a gemma3- or llava-class model, so such a model could not answer at all.
        mock_post.return_value.json.return_value = {"capabilities": ["completion", "thinking"]}
        llm = self._make_llm()
        real = llm._build_request([{"role": "user", "content": "hi"}], stream=True)
        pre_heat = llm._build_request([{"role": "user", "content": "hi"}], stream=True, num_predict=1)
        assert "tools" not in real and "tools" not in pre_heat
        assert real["think"] == "medium"
        assert mock_post.call_count == 1, "one probe of /api/show answers both capabilities"

    @patch('assistant.llm.OllamaLLM.requests.post')
    def test_a_model_with_the_tools_capability_gets_the_static_tool(self, mock_post):
        mock_post.return_value.json.return_value = {"capabilities": ["completion", "tools"]}
        req = self._make_llm()._build_request([{"role": "user", "content": "hi"}], stream=True)
        assert req["tools"][0]["function"]["name"] == "load_toolkit" and req["think"] is False

    @patch('assistant.llm.OllamaLLM.OllamaLLM.thinks', return_value=True)
    @patch('assistant.llm.OllamaLLM.OllamaLLM._rendered_prompt', return_value=None)
    @patch('assistant.llm.OllamaLLM.requests.post')
    def test_native_thinking_streams_as_a_thinking_block(self, mock_post, mock_rendered, mock_thinks):
        # The command processor only knows <thinking> tags, so the chunks' `thinking` field
        # arrives inside one, closed the moment the first reply token follows.
        import json
        mock_post.return_value.iter_lines.return_value = [
            json.dumps({"message": {"thinking": "plan "}}),
            json.dumps({"message": {"thinking": "it"}}),
            json.dumps({"message": {"content": "Hi"}}),
            json.dumps({"message": {"content": "!"}, **self.DONE}),
        ]
        chunks = list(self._make_llm().create_streaming_completion([{"role": "user", "content": "hi"}]))
        assert "".join(chunks) == "<thinking>plan it</thinking>Hi!"

    @patch('assistant.llm.OllamaLLM.OllamaLLM.thinks', return_value=True)
    @patch('assistant.llm.OllamaLLM.OllamaLLM._rendered_prompt', return_value=None)
    @patch('assistant.llm.OllamaLLM.requests.post')
    def test_a_stream_cut_off_while_thinking_still_closes_the_block(self, mock_post, mock_rendered, mock_thinks):
        import json
        mock_post.return_value.iter_lines.return_value = [
            json.dumps({"message": {"thinking": "hmm"}}),
            json.dumps({"message": {"content": ""}, **self.DONE}),
        ]
        chunks = list(self._make_llm().create_streaming_completion([{"role": "user", "content": "hi"}]))
        assert "".join(chunks) == "<thinking>hmm</thinking>"

    @patch('assistant.llm.OllamaLLM.requests.post')
    def test_rendered_prompt_post_carries_timeout(self, mock_post):
        from assistant.llm.OllamaLLM import OllamaLLM
        llm = self._make_llm()
        mock_post.return_value.json.return_value = {"_debug_info": {"rendered_template": "x"}}
        assert llm._rendered_prompt([{"role": "user", "content": "hi"}]) == "x"
        assert mock_post.call_args.kwargs["timeout"] == OllamaLLM._RENDER_TIMEOUT

    @patch('assistant.llm.OllamaLLM.requests.post')
    def test_raw_pre_heat_post_carries_timeout(self, mock_post):
        from assistant.llm.OllamaLLM import OllamaLLM
        llm = self._make_llm()
        rendered = "system prefix<|im_end|>\n<|im_start|>assistant\n<think>"
        with patch.object(llm, '_rendered_prompt', return_value=rendered):
            assert llm._pre_heat_raw_truncated([{"role": "user", "content": "hi"}]) is True
        assert mock_post.call_args.kwargs["timeout"] == OllamaLLM._PRE_HEAT_TIMEOUT

    @patch('assistant.llm.OllamaLLM.OllamaLLM._rendered_prompt', return_value=None)
    @patch('assistant.llm.OllamaLLM.requests.post')
    def test_streaming_timeout_only_for_pre_heat(self, mock_post, mock_rendered):
        from assistant.llm.OllamaLLM import OllamaLLM
        llm = self._make_llm()
        mock_post.return_value.iter_lines.return_value = []
        list(llm.create_streaming_completion([{"role": "user", "content": "hi"}], num_predict=1))
        assert mock_post.call_args.kwargs["timeout"] == OllamaLLM._PRE_HEAT_TIMEOUT
        list(llm.create_streaming_completion([{"role": "user", "content": "hi"}]))
        assert mock_post.call_args.kwargs["timeout"] is None

    @patch('assistant.llm.OllamaLLM.OllamaLLM.capabilities', return_value={"thinking", "tools"})
    @patch('assistant.llm.OllamaLLM.OllamaLLM._rendered_prompt', return_value=None)
    @patch('assistant.llm.OllamaLLM.requests.post')
    def test_create_completion(self, mock_post, mock_rendered, mock_capabilities):
        llm = self._make_llm()
        mock_resp = Mock()
        mock_resp.json.return_value = {
            "message": {"content": "hello back"},
            "done": True,
            "prompt_eval_count": 10, "prompt_eval_duration": 1_000_000_000,
            "eval_count": 5, "eval_duration": 500_000_000,
            "total_duration": 1_500_000_000,
        }
        mock_post.return_value = mock_resp

        result = llm.create_completion([{"role": "user", "content": "hi"}])
        assert result == "hello back"
        mock_resp.raise_for_status.assert_called_once()

    @patch('assistant.llm.OllamaLLM.OllamaLLM._rendered_prompt', return_value=None)
    @patch('assistant.llm.OllamaLLM.requests.post')
    def test_create_streaming_completion(self, mock_post, mock_rendered):
        import json as _json
        llm = self._make_llm()

        lines = [
            _json.dumps({"message": {"content": "Hello"}, "done": False}).encode(),
            _json.dumps({"message": {"content": " world"}, "done": False}).encode(),
            b"",
            _json.dumps({"message": {"content": ""}, "done": True,
                         "prompt_eval_count": 10, "prompt_eval_duration": 1e9,
                         "eval_count": 5, "eval_duration": 5e8,
                         "total_duration": 1.5e9}).encode(),
        ]
        mock_resp = Mock()
        mock_resp.iter_lines.return_value = lines
        mock_post.return_value = mock_resp

        result = list(llm.create_streaming_completion([{"role": "user", "content": "hi"}]))
        assert result == ["Hello", " world"]

    @patch('assistant.llm.OllamaLLM.requests.post')
    def test_create_completion_with_images_bytes(self, mock_post):
        llm = self._make_llm()
        mock_resp = Mock()
        mock_resp.json.return_value = {
            "message": {"content": "I see a photo"},
            "done": True,
            "prompt_eval_count": 50, "prompt_eval_duration": 2e9,
            "eval_count": 10, "eval_duration": 1e9,
            "total_duration": 3e9,
        }
        mock_post.return_value = mock_resp

        msgs = [{"role": "user", "content": "what is this?"}]
        result = llm.create_completion_with_images(msgs, [b'\x89PNG\x00\x00'])
        assert result == "I see a photo"

        call_json = mock_post.call_args[1]['json']
        assert 'images' in call_json['messages'][0]

    @patch('assistant.llm.OllamaLLM.requests.post')
    def test_create_completion_with_images_base64_string(self, mock_post):
        llm = self._make_llm()
        mock_resp = Mock()
        mock_resp.json.return_value = {
            "message": {"content": "nice"}, "done": True,
            "prompt_eval_count": 10, "prompt_eval_duration": 1e9,
            "eval_count": 5, "eval_duration": 5e8, "total_duration": 1.5e9,
        }
        mock_post.return_value = mock_resp

        b64 = base64.b64encode(b"test image").decode()
        result = llm.create_completion_with_images(
            [{"role": "user", "content": "describe"}], [b64]
        )
        assert result == "nice"
        sent_images = mock_post.call_args[1]['json']['messages'][0]['images']
        assert sent_images == [b64]

    @patch('assistant.llm.OllamaLLM.OllamaLLM.thinks', return_value=True)
    @patch('assistant.llm.OllamaLLM.requests.post')
    def test_create_completion_with_images_routes_through_build_request(self, mock_post, mock_thinks):
        # ea11de61 replaced the inline payload in create_completion_with_images
        # with self._build_request(...) so vision requests carry the same fields
        # as every other request. The visible behavioural side-effect is that
        # `think` now flows through — a revert to the inline dict would silently
        # drop it, and the existing image tests wouldn't notice. Pin the
        # construction contract on the vision path.
        llm = self._make_llm()
        mock_resp = Mock()
        mock_resp.json.return_value = {"message": {"content": "ok"}, **self.DONE}
        mock_post.return_value = mock_resp

        llm.create_completion_with_images(
            [{"role": "user", "content": "what?"}], [b'\x89PNG\x00\x00']
        )

        call_json = mock_post.call_args[1]['json']
        assert call_json["model"] == "gemma4:26b"
        assert call_json["stream"] is False
        assert call_json["keep_alive"] == -1
        assert call_json["think"] == "medium"

    # --- the prefill tracker, fed from Ollama's log ----------------------
    # The estimate rules live in PrefillTracker (tests/unit/assistant/llm/test_prefill_tracker.py);
    # here: Ollama's log lines, the anchor at request start, and the rate the done-time metrics teach.

    @pytest.fixture
    def tracker(self, tmp_path):
        """OllamaLLM's tracker pointed at a tmp log, fresh, and restored afterwards."""
        from assistant.llm.OllamaLLM import OllamaLLM
        t = OllamaLLM._prefill
        saved_log, saved_tps = t.log, t.tps
        t.use_log(str(tmp_path / "ollama.log"))
        _reset_tracker(t)
        yield t
        _reset_tracker(t)
        t.log, t.tps = saved_log, saved_tps

    def test_the_llamacpp_cache_slot_line_gives_the_workload(self):
        from assistant.llm.OllamaLLM import OllamaLogFormat
        text = ('time=... level=DEBUG source=cache.go:151 msg="loading cache slot" '
                'id=0 cache=6533 prompt=6344 used=5786 remaining=558\n')
        assert OllamaLogFormat().work(text) == (5786, 558)
        assert OllamaLogFormat().progress(text) is None, "the llama.cpp runner logs no progress"

    def test_a_partial_cache_slot_line_is_not_a_workload_yet(self):
        # A poll landing mid-write sees the line without its remaining= — not a match, so the
        # tracker re-reads the same bytes on the next poll instead of latching garbage.
        from assistant.llm.OllamaLLM import OllamaLogFormat
        assert OllamaLogFormat().work('msg="loading cache slot" id=0 cache=6533 prompt=6344 used=5786 ') is None

    def test_a_stale_line_below_the_anchor_is_excluded_and_the_new_one_found(self, tracker, tmp_path):
        # begin() runs before the POST, so the anchor lands at the current EOF: the previous
        # request's line stays below it and the one Ollama logs afterwards is this request's.
        log_file = tmp_path / "ollama.log"
        log_file.write_text('msg="loading cache slot" id=0 cache=1 prompt=1 used=999 remaining=999\n')
        token = tracker.begin()
        assert tracker._find(token).anchor == log_file.stat().st_size
        assert tracker.work(token, force=True) is None
        with open(log_file, 'a') as f:
            f.write('msg="loading cache slot" id=0 cache=2 prompt=2 used=5786 remaining=558\n')
        assert tracker.work(token, force=True) == (5786, 558)
        assert tracker.work(token, force=True) == (5786, 558), "polls never consume the line"
        tracker.end(token)

    def test_progress_follows_the_mlx_progress_lines_in_the_log(self, tracker, tmp_path):
        import time
        token = tracker.begin()
        (tmp_path / "ollama.log").write_text(
            'msg="cache miss" total=17531 matched=2 cached=0 left=17531\n'
            'msg="Prompt processing progress" processed=2048 total=17531\n'
            'msg="Prompt processing progress" processed=8192 total=17531\n')
        tracker._find(token).start_ts = time.monotonic() - 9999  # the time estimate alone would say 99
        assert tracker.progress() == pytest.approx(8192 / 17531 * 100.0)
        tracker.end(token)

    # --- prefill-rate self-calibration (EMA) ----------------------------
    # The time estimate hinges on the tracker's rate, which _log_metrics refines from each
    # completed request's prompt_eval_duration over the tokens the cache line said were left.

    def test_log_metrics_refines_prefill_rate_via_ema(self, tracker):
        token = tracker.begin()
        request = tracker._find(token)
        request.work, request.anchor = (5000, 558), None   # remaining=558 over 1.0s => 558 t/s
        tracker.tps = 350.0
        self._make_llm()._log_metrics({
            "prompt_eval_count": 5558, "prompt_eval_duration": 1_000_000_000,
            "eval_count": 5, "eval_duration": 500_000_000,
            "total_duration": 1_500_000_000,
        }, token=token)
        assert tracker.tps == pytest.approx(0.5 * 350.0 + 0.5 * 558.0)

    def test_log_metrics_skips_rate_update_on_near_full_cache_hit(self, tracker):
        token = tracker.begin()
        request = tracker._find(token)
        request.work, request.anchor = (5000, 50), None   # <=100 tokens left: too noisy to learn from
        tracker.tps = 350.0
        self._make_llm()._log_metrics({
            "prompt_eval_count": 5050, "prompt_eval_duration": 1_000_000_000,
            "eval_count": 5, "eval_duration": 500_000_000,
            "total_duration": 1_500_000_000,
        }, token=token)
        assert tracker.tps == 350.0

    # --- pre-heat in-flight guard ---------------------------------------
    # Ollama processes one request at a time. The guard ensures at most one
    # pre-heat is ever fired; a newer prompt arriving mid-flight is stored and
    # fired when the current one completes (rather than piling up requests).

    def _reset_preheat_state(self):
        from assistant.llm.OllamaLLM import OllamaLLM
        OllamaLLM._pre_heat_in_flight = False
        OllamaLLM._pre_heat_pending_messages = None

    def test_pre_heat_fires_when_idle(self):
        from assistant.llm.OllamaLLM import OllamaLLM
        self._reset_preheat_state()
        try:
            llm = self._make_llm()
            msgs = [{"role": "user", "content": "warm me"}]
            with patch.object(OllamaLLM, '_fire_pre_heat') as mock_fire:
                llm.pre_heat_cache(msgs)
            assert OllamaLLM._pre_heat_in_flight is True
            mock_fire.assert_called_once_with(msgs)
            assert OllamaLLM._pre_heat_pending_messages is None
        finally:
            self._reset_preheat_state()

    def test_pre_heat_defers_when_in_flight(self):
        from assistant.llm.OllamaLLM import OllamaLLM
        self._reset_preheat_state()
        try:
            llm = self._make_llm()
            OllamaLLM._pre_heat_in_flight = True
            msgs = [{"role": "user", "content": "latest"}]
            with patch.object(OllamaLLM, '_fire_pre_heat') as mock_fire:
                llm.pre_heat_cache(msgs)
            mock_fire.assert_not_called()
            assert OllamaLLM._pre_heat_pending_messages == msgs
        finally:
            self._reset_preheat_state()

    def test_pre_heat_appends_stub_user_message_to_system_only_prompt(self):
        # Ollama 500s ("no user query found in messages") on a system-only
        # request, which is exactly what the prompt-assembled callback sends.
        import time
        self._reset_preheat_state()
        try:
            llm = self._make_llm()
            captured = {}
            def fake_stream(messages, num_predict=None):
                captured['messages'] = messages
                return iter(())
            with patch.object(llm, 'create_streaming_completion', side_effect=fake_stream), \
                 patch.object(llm, '_pre_heat_raw_truncated', return_value=False):
                llm.pre_heat_cache([{"role": "system", "content": "sys"}])
                for _ in range(100):
                    if 'messages' in captured:
                        break
                    time.sleep(0.01)
            assert captured['messages'][0] == {"role": "system", "content": "sys"}
            assert captured['messages'][-1]["role"] == "user"
        finally:
            self._reset_preheat_state()

    def test_pre_heat_leaves_prompt_with_user_message_untouched(self):
        import time
        self._reset_preheat_state()
        try:
            llm = self._make_llm()
            msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
            captured = {}
            def fake_stream(messages, num_predict=None):
                captured['messages'] = messages
                return iter(())
            with patch.object(llm, 'create_streaming_completion', side_effect=fake_stream), \
                 patch.object(llm, '_pre_heat_raw_truncated', return_value=False):
                llm.pre_heat_cache(msgs)
                for _ in range(100):
                    if 'messages' in captured:
                        break
                    time.sleep(0.01)
            assert captured['messages'] == msgs
        finally:
            self._reset_preheat_state()

    # --- raw truncated pre-heat -----------------------------------------
    # The MLX runner snapshots the KV cache at end-of-prompt minus 4 tokens.
    # A chat pre-heat ends with the template's generation suffix, placing that
    # snapshot past where the next real request diverges (unusable). The raw
    # path cuts the suffix so the snapshot lands inside the shared prefix.

    _RENDERED = ("<|im_start|>system\nsys<|im_end|>\n"
                 "<|im_start|>user\nnote content<|im_end|>\n"
                 "<|im_start|>assistant\n<think>\n\n</think>\n\n")

    def test_raw_pre_heat_truncates_generation_suffix(self):
        llm = self._make_llm()
        with patch.object(llm, '_rendered_prompt', return_value=self._RENDERED), \
             patch('assistant.llm.OllamaLLM.requests.post') as mock_post:
            mock_post.return_value = MagicMock(status_code=200)
            assert llm._pre_heat_raw_truncated([{"role": "user", "content": "note content"}]) is True
        assert mock_post.call_count == 1
        url = mock_post.call_args[0][0]
        body = mock_post.call_args[1]["json"]
        assert url.endswith("/api/generate")
        assert body["raw"] is True
        assert body["options"]["num_predict"] == 1
        assert body["prompt"].endswith("note content"), "prompt must be cut at the end of the last turn's content"
        assert "<|im_start|>assistant" not in body["prompt"]

    def test_raw_pre_heat_falls_back_when_render_unavailable(self):
        llm = self._make_llm()
        with patch.object(llm, '_rendered_prompt', return_value=None), \
             patch('assistant.llm.OllamaLLM.requests.post') as mock_post:
            assert llm._pre_heat_raw_truncated([{"role": "user", "content": "x"}]) is False
        mock_post.assert_not_called()

    def test_raw_pre_heat_falls_back_when_suffix_marker_missing(self):
        llm = self._make_llm()
        with patch.object(llm, '_rendered_prompt', return_value="some other template format"), \
             patch('assistant.llm.OllamaLLM.requests.post') as mock_post:
            assert llm._pre_heat_raw_truncated([{"role": "user", "content": "x"}]) is False
        mock_post.assert_not_called()

    def test_pre_heat_keeps_only_latest_pending(self):
        from assistant.llm.OllamaLLM import OllamaLLM
        self._reset_preheat_state()
        try:
            llm = self._make_llm()
            OllamaLLM._pre_heat_in_flight = True
            with patch.object(OllamaLLM, '_fire_pre_heat'):
                llm.pre_heat_cache([{"role": "user", "content": "first"}])
                llm.pre_heat_cache([{"role": "user", "content": "second"}])
            assert OllamaLLM._pre_heat_pending_messages == [{"role": "user", "content": "second"}]
        finally:
            self._reset_preheat_state()


# ===========================================================================
# TestOllamaRemoteLog
# ===========================================================================

class TestOllamaRemoteLog:
    """Remote-Ollama diagnostics: the tracker reads the log over ssh when base_url points at
    another machine, and goes back to the local file for a local one. The ssh reader itself is
    tested in test_prefill_tracker.py."""

    @pytest.fixture(autouse=True)
    def _restore_log(self):
        from assistant.llm.OllamaLLM import OllamaLLM
        saved = OllamaLLM._prefill.log
        yield
        OllamaLLM._prefill.log = saved

    @patch('assistant.llm.OllamaLLM.config')
    def test_remote_base_url_reads_the_log_over_ssh_to_that_host(self, mock_config):
        mock_config.get.side_effect = lambda key, default=None: default
        from assistant.llm.OllamaLLM import OllamaLLM
        OllamaLLM(base_url='http://100.64.0.10:11434')
        log = OllamaLLM._prefill.log
        assert log.ssh_host == '100.64.0.10'
        assert log.path.endswith('/log/ollama.log')

    @patch('assistant.llm.OllamaLLM.config')
    def test_ssh_host_and_log_path_from_config(self, mock_config):
        overrides = {'ollama.ssh_host': 'user@studio.local', 'ollama.remote_log_path': '/srv/ollama.log'}
        mock_config.get.side_effect = lambda key, default=None: overrides.get(key, default)
        from assistant.llm.OllamaLLM import OllamaLLM
        OllamaLLM(base_url='http://studio.local:11434')
        log = OllamaLLM._prefill.log
        assert (log.ssh_host, log.path) == ('user@studio.local', '/srv/ollama.log')

    @patch('assistant.llm.OllamaLLM.config')
    def test_local_base_url_goes_back_to_the_local_file(self, mock_config):
        mock_config.get.side_effect = lambda key, default=None: default
        from assistant.llm.OllamaLLM import OllamaLLM
        OllamaLLM(base_url='http://studio.local:11434')
        assert OllamaLLM._prefill.log.ssh_host is not None
        OllamaLLM(base_url='http://localhost:11434')
        log = OllamaLLM._prefill.log
        assert log.ssh_host is None and log.path == OllamaLLM._ollama_log_path


# ===========================================================================
# TestOllamaMlxLogFormat
# ===========================================================================

class TestOllamaMlxLogFormat:
    """The MLX-engine Ollama build logs a different format than the llama.cpp
    runner: 'cache hit/miss ... cached=C left=L' instead of 'loading cache slot
    ... used=N remaining=M', plus real incremental prefill progress lines."""

    def _format(self):
        from assistant.llm.OllamaLLM import OllamaLogFormat
        return OllamaLogFormat()

    def test_parses_mlx_cache_hit(self):
        text = ('time=... level=INFO source=prefix_cache.go:124 '
                'msg="cache hit" total=25572 matched=25553 cached=24575 left=997\n')
        assert self._format().work(text) == (24575, 997)

    def test_parses_mlx_cache_miss(self):
        text = ('time=... level=INFO source=prefix_cache.go:124 '
                'msg="cache miss" total=17531 matched=2 cached=0 left=17531\n')
        assert self._format().work(text) == (0, 17531)

    def test_progress_is_the_last_progress_line(self):
        text = ('msg="cache miss" total=17531 matched=2 cached=0 left=17531\n'
                'msg="Prompt processing progress" processed=2048 total=17531\n'
                'msg="Prompt processing progress" processed=4096 total=17531\n')
        assert self._format().progress(text) == (4096, 17531)

    def test_no_progress_without_progress_lines(self):
        assert self._format().progress('msg="loading cache slot" id=0 cache=1 prompt=1 used=1 remaining=1\n') is None

# ===========================================================================
# missing_api_key — the question a caller asks before initialize() when it would rather
# sit out than crash the boot (the cache auditor, on an install without a Gemini key).
# ===========================================================================

class TestMissingApiKey:
    @patch('assistant.llm.GeminiLLM.config')
    def test_gemini_names_its_key_only_when_unset(self, mock_config):
        mock_config.get_api_key.return_value = ""
        assert GeminiLLM().missing_api_key() == "api_keys.gemini"
        mock_config.get_api_key.return_value = "g-key"
        assert GeminiLLM().missing_api_key() is None

    @patch('assistant.llm.ClaudeLLM.config')
    def test_claude_names_its_key_only_when_unset(self, mock_config):
        from assistant.llm.ClaudeLLM import ClaudeLLM
        mock_config.get_api_key.return_value = None
        assert ClaudeLLM().missing_api_key() == "api_keys.anthropic"
        mock_config.get_api_key.return_value = "a-key"
        assert ClaudeLLM().missing_api_key() is None

    @patch('assistant.llm.ChatGPTLLM.config')
    def test_chatgpt_family_names_the_key_it_was_built_for(self, mock_config):
        # Grok and the other OpenAI-compatible backends share the class; the answer must
        # point at *their* key, not at api_keys.openai.
        from assistant.llm.ChatGPTLLM import ChatGPTLLM
        mock_config.get_api_key.return_value = ""
        assert ChatGPTLLM(api_key_name='grok').missing_api_key() == "api_keys.grok"
        assert ChatGPTLLM().missing_api_key() == "api_keys.openai"
        mock_config.get_api_key.return_value = "o-key"
        assert ChatGPTLLM(api_key_name='grok').missing_api_key() is None

    @patch('assistant.llm.ChatGPTLLM.config')
    @patch('assistant.llm.ClaudeLLM.config')
    @patch('assistant.llm.GeminiLLM.config')
    def test_a_key_stored_after_construction_reaches_the_same_instance(self, gemini_cfg, claude_cfg, gpt_cfg):
        """dfa4b591: the key is read at every use, not captured in __init__. The LLM objects
        are built once at boot; a key the user stores from the chat (config_request) must
        reach the next request without a restart."""
        from assistant.llm.ClaudeLLM import ClaudeLLM
        from assistant.llm.ChatGPTLLM import ChatGPTLLM
        for cfg in (gemini_cfg, claude_cfg, gpt_cfg):
            cfg.get_api_key.return_value = None
        llms = [GeminiLLM(), ClaudeLLM(), ChatGPTLLM(api_key_name='grok')]
        assert [llm.missing_api_key() for llm in llms] == ["api_keys.gemini", "api_keys.anthropic", "api_keys.grok"]

        for cfg in (gemini_cfg, claude_cfg, gpt_cfg):
            cfg.get_api_key.return_value = "stored-later"
        assert [llm.missing_api_key() for llm in llms] == [None, None, None]
        assert [llm.api_key for llm in llms] == ["stored-later"] * 3

    def test_tier_api_key_name_builds_the_tier_backend_uninitialized(self, monkeypatch):
        from assistant.llm import tier_api_key_name
        monkeypatch.setattr('assistant.llm.parse_llm_config', lambda tier: ('GeminiLLM', 'gemini-3.8-flash'))
        with patch('assistant.llm.GeminiLLM.GeminiLLM.initialize') as init:
            assert tier_api_key_name('photo_indexing') == "api_keys.gemini"
        init.assert_not_called()
        monkeypatch.setattr('assistant.llm.parse_llm_config', lambda tier: ('OllamaLLM', None))
        assert tier_api_key_name('photo_indexing') is None

    def test_the_wrappers_forward_the_backend_answer(self, monkeypatch):
        # create_llm() always hands out a VaultLLM, and the cache auditor wraps that in a
        # CachedLLM: a wrapper that fell back to the base class's None would send the
        # auditor into initialize() and crash the boot without the key.
        from assistant.llm.VaultLLM import VaultLLM
        from assistant.llm.CachedLLM import CachedLLM
        monkeypatch.setattr('common.FloxyDatabase.FloxyDatabase.get_instance', staticmethod(lambda: Mock()))
        backend = Mock(spec=AbstractLLM)
        backend.missing_api_key.return_value = "api_keys.gemini"
        assert VaultLLM(backend, Mock()).missing_api_key() == "api_keys.gemini"
        assert CachedLLM(VaultLLM(backend, Mock())).missing_api_key() == "api_keys.gemini"
        backend.missing_api_key.return_value = None
        assert CachedLLM(VaultLLM(backend, Mock())).missing_api_key() is None
        backend.api_key_name.return_value = "api_keys.gemini"
        assert CachedLLM(VaultLLM(backend, Mock())).api_key_name() == "api_keys.gemini"
