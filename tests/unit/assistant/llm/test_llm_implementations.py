import base64
import itertools
import pytest
from unittest.mock import Mock, patch, MagicMock

from assistant.llm.AbstractLLM import AbstractLLM, LLMRefusalError, ThinkingNotSupportedError


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

    def create_streaming_completion(self, messages, thinking=False):
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
        assert params['max_tokens'] == 4096
        assert 'thinking' not in params
        assert params['stream'] is True

    @patch('assistant.llm.ClaudeLLM.config')
    def test_build_api_params_with_thinking(self, mock_config):
        mock_config.get_api_key.return_value = "k"
        from assistant.llm.ClaudeLLM import ClaudeLLM
        llm = ClaudeLLM()
        messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
        params = llm._build_api_params(messages, thinking=True, stream=False)
        assert params['max_tokens'] == 32000
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
        assert prev_msg["content"][-1].get("cache_control") == {"type": "ephemeral", "ttl": llm.static_cache_ttl}

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
        assert prev_msg["content"][0].get("cache_control") == {"type": "ephemeral", "ttl": llm.static_cache_ttl}

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

    @patch('assistant.llm.ClaudeLLM.config')
    def test_image_completion_strips_cache_markers_from_system(self, mock_config):
        """The one-shot image path skips _build_cached_system entirely, so it must
        strip the breakpoint markers itself — otherwise they reach the model as
        literal prompt text (the pre-existing leak fixed alongside multi-breakpoint)."""
        mock_config.get_api_key.return_value = "k"
        from assistant.llm.ClaudeLLM import ClaudeLLM
        llm = ClaudeLLM()
        llm.client = Mock()
        response = Mock()
        response.content = [Mock(type='text', text='ok')]
        llm.client.messages.create.return_value = response

        llm.create_completion_with_images(
            [{"role": "system", "content": "rules\n<<<CACHE_BREAKPOINT>>>\nlive data"},
             {"role": "user", "content": "what's on the camera?"}],
            [b'\x89PNG1234'],
        )

        system = llm.client.messages.create.call_args.kwargs['system']
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
        from assistant.llm.GeminiLLM import GeminiLLM
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
        from assistant.llm.GeminiLLM import GeminiLLM
        sys_instr, _ = GeminiLLM()._convert_messages([
            {"role": "system", "content": "rules<<<CACHE_BREAKPOINT>>>live data"},
            {"role": "user", "content": "hello"},
        ])
        assert sys_instr == "ruleslive data"

    @patch('assistant.llm.GeminiLLM.config')
    @patch('assistant.llm.GeminiLLM.types')
    def test_streaming_completion(self, mock_types, mock_config):
        mock_config.get_api_key.return_value = "k"
        from assistant.llm.GeminiLLM import GeminiLLM
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
        from assistant.llm.GeminiLLM import GeminiLLM
        from google.genai.types import ThinkingLevel
        assert GeminiLLM("gemini-3.7-flash")._min_thinking_level() == ThinkingLevel.LOW
        assert GeminiLLM("gemini-3.8-flash")._min_thinking_level() == ThinkingLevel.LOW
        assert GeminiLLM("gemini-3.5-flash")._min_thinking_level() == ThinkingLevel.MINIMAL

    @patch('assistant.llm.GeminiLLM.config')
    @patch('assistant.llm.GeminiLLM.types')
    def test_streaming_37_flash_never_requests_minimal(self, mock_types, mock_config):
        """The non-thinking streaming path is the production voice loop on the
        fast tier — sending MINIMAL to gemini-3.7-flash 400s every request."""
        mock_config.get_api_key.return_value = "k"
        from assistant.llm.GeminiLLM import GeminiLLM
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
        from assistant.llm.GeminiLLM import GeminiLLM
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
        from assistant.llm.GeminiLLM import GeminiLLM
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
        from assistant.llm.GeminiLLM import GeminiLLM
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
        from assistant.llm.GeminiLLM import GeminiLLM
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
        from assistant.llm.GeminiLLM import GeminiLLM
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
        from assistant.llm.GeminiLLM import GeminiLLM
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
        from assistant.llm.GeminiLLM import GeminiLLM
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
    @patch('assistant.llm.GeminiLLM.config')
    @patch('assistant.llm.GeminiLLM.types')
    def test_think_with_video_sends_inline_clip_before_text(self, mock_types, mock_config):
        mock_config.get_api_key.return_value = "k"
        mock_types.Part.return_value = "video_part"
        from assistant.llm.GeminiLLM import GeminiLLM
        llm = GeminiLLM(model="gemini-video")
        llm.client = Mock()
        llm.client.models.generate_content.return_value = Mock(text="A dog runs", usage_metadata=None)

        result = llm.think_with_video("sys<<<CACHE_BREAKPOINT>>>tem", "describe", b"vid", "video/quicktime")

        assert result == "A dog runs"
        mock_types.Blob.assert_called_once_with(data=b"vid", mime_type="video/quicktime")
        mock_types.Part.assert_called_once_with(inline_data=mock_types.Blob.return_value, video_metadata=None)
        mock_types.VideoMetadata.assert_not_called()
        mock_types.GenerateContentConfig.assert_called_once_with(system_instruction="system")
        call = llm.client.models.generate_content.call_args[1]
        assert call["model"] == "gemini-video"
        assert call["contents"] == [{'role': 'user', 'parts': ["video_part", {'text': "describe"}]}]

    @patch('assistant.llm.GeminiLLM.config')
    @patch('assistant.llm.GeminiLLM.types')
    def test_think_with_video_passes_fps_as_video_metadata(self, mock_types, mock_config):
        mock_config.get_api_key.return_value = "k"
        from assistant.llm.GeminiLLM import GeminiLLM
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
        from assistant.llm.GeminiLLM import GeminiLLM
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
    def test_empty_response_returns_empty_string_instead_of_none(self, mock_types, mock_config):
        mock_config.get_api_key.return_value = "k"
        from assistant.llm.GeminiLLM import GeminiLLM
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
        from assistant.llm.GeminiLLM import GeminiLLM
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
        from assistant.llm.GeminiLLM import GeminiLLM
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
        from assistant.llm.GeminiLLM import GeminiLLM
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
        from assistant.llm.GeminiLLM import GeminiLLM
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
        from assistant.llm.GeminiLLM import GeminiLLM
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
        from assistant.llm.GeminiLLM import GeminiLLM
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
        from assistant.llm.GeminiLLM import GeminiLLM
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
        OllamaLLM._thinking_capable.clear()
        yield
        OllamaLLM._thinking_capable.clear()

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
            "think": True,
        }

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

    @patch('assistant.llm.OllamaLLM.OllamaLLM.thinks', return_value=True)
    @patch('assistant.llm.OllamaLLM.OllamaLLM._rendered_prompt', return_value=None)
    @patch('assistant.llm.OllamaLLM.requests.post')
    def test_create_completion(self, mock_post, mock_rendered, mock_thinks):
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
        assert call_json["think"] is True

    def test_read_cache_metrics_parses_log(self, tmp_path):
        from assistant.llm.OllamaLLM import OllamaLLM
        log_file = tmp_path / "ollama.log"
        log_file.write_text(
            'time=... level=DEBUG source=cache.go:151 msg="loading cache slot" '
            'id=0 cache=6533 prompt=6344 used=5786 remaining=558\n'
        )
        original_path = OllamaLLM._ollama_log_path
        original_anchor = OllamaLLM._prefill_log_start
        try:
            OllamaLLM._ollama_log_path = str(log_file)
            OllamaLLM._prefill_log_start = 0
            result = OllamaLLM._read_cache_metrics()
            assert result == (5786, 558)
        finally:
            OllamaLLM._ollama_log_path = original_path
            OllamaLLM._prefill_log_start = original_anchor

    def test_read_cache_metrics_returns_none_when_no_file(self, tmp_path):
        from assistant.llm.OllamaLLM import OllamaLLM
        original_path = OllamaLLM._ollama_log_path
        original_anchor = OllamaLLM._prefill_log_start
        try:
            OllamaLLM._ollama_log_path = str(tmp_path / "nonexistent.log")
            OllamaLLM._prefill_log_start = 0
            assert OllamaLLM._read_cache_metrics() is None
        finally:
            OllamaLLM._ollama_log_path = original_path
            OllamaLLM._prefill_log_start = original_anchor

    def test_read_cache_metrics_returns_none_when_no_new_content(self, tmp_path):
        from assistant.llm.OllamaLLM import OllamaLLM
        log_file = tmp_path / "ollama.log"
        log_file.write_text("some old content\n")
        original_path = OllamaLLM._ollama_log_path
        original_anchor = OllamaLLM._prefill_log_start
        try:
            OllamaLLM._ollama_log_path = str(log_file)
            OllamaLLM._prefill_log_start = log_file.stat().st_size
            assert OllamaLLM._read_cache_metrics() is None
        finally:
            OllamaLLM._ollama_log_path = original_path
            OllamaLLM._prefill_log_start = original_anchor

    def test_read_cache_metrics_is_idempotent_across_polls(self, tmp_path):
        # Regression: the read must NOT advance a shared cursor. The render thread
        # polls this many times per second; every poll must return the same tuple,
        # never consume the line so a later read (or done-time log) sees None.
        from assistant.llm.OllamaLLM import OllamaLLM
        log_file = tmp_path / "ollama.log"
        log_file.write_text(
            'msg="loading cache slot" id=0 cache=6533 prompt=6344 used=5786 remaining=558\n'
        )
        original_path = OllamaLLM._ollama_log_path
        original_anchor = OllamaLLM._prefill_log_start
        try:
            OllamaLLM._ollama_log_path = str(log_file)
            OllamaLLM._prefill_log_start = 0
            assert OllamaLLM._read_cache_metrics() == (5786, 558)
            assert OllamaLLM._read_cache_metrics() == (5786, 558)
            assert OllamaLLM._read_cache_metrics() == (5786, 558)
        finally:
            OllamaLLM._ollama_log_path = original_path
            OllamaLLM._prefill_log_start = original_anchor

    def test_read_cache_metrics_recovers_from_partial_line(self, tmp_path):
        # Regression (stuck-at-0%): a poll landing mid-write sees a partial line
        # with no remaining= yet. It must return None and leave the bytes for the
        # next poll to re-read once the line is complete — never discard them.
        from assistant.llm.OllamaLLM import OllamaLLM
        log_file = tmp_path / "ollama.log"
        log_file.write_text('msg="loading cache slot" id=0 cache=6533 prompt=6344 used=5786 ')
        original_path = OllamaLLM._ollama_log_path
        original_anchor = OllamaLLM._prefill_log_start
        try:
            OllamaLLM._ollama_log_path = str(log_file)
            OllamaLLM._prefill_log_start = 0
            assert OllamaLLM._read_cache_metrics() is None  # partial line, no match
            with open(log_file, 'a') as f:
                f.write('remaining=558\n')
            assert OllamaLLM._read_cache_metrics() == (5786, 558)  # re-read, now complete
        finally:
            OllamaLLM._ollama_log_path = original_path
            OllamaLLM._prefill_log_start = original_anchor

    def test_read_cache_metrics_excludes_lines_before_anchor(self, tmp_path):
        # Regression (stale latching): a cache-slot line from a prior request or a
        # preheat sits below the anchor and must be ignored; the first line AFTER
        # the anchor is the one for the current request.
        from assistant.llm.OllamaLLM import OllamaLLM
        log_file = tmp_path / "ollama.log"
        log_file.write_text(
            'msg="loading cache slot" id=0 cache=1 prompt=1 used=111 remaining=222\n'
        )
        original_path = OllamaLLM._ollama_log_path
        original_anchor = OllamaLLM._prefill_log_start
        try:
            OllamaLLM._ollama_log_path = str(log_file)
            OllamaLLM._prefill_log_start = log_file.stat().st_size  # anchor past stale line
            with open(log_file, 'a') as f:
                f.write('msg="loading cache slot" id=0 cache=2 prompt=2 used=5786 remaining=558\n')
            assert OllamaLLM._read_cache_metrics() == (5786, 558)
        finally:
            OllamaLLM._ollama_log_path = original_path
            OllamaLLM._prefill_log_start = original_anchor

    def test_begin_prefill_anchors_at_current_log_eof(self, tmp_path):
        # _begin_prefill is called before the POST; the cache-slot line Ollama
        # logs afterwards is what we want. Verify the anchor lands at EOF so a
        # pre-existing (prior-request) line is excluded but the new one is found.
        from assistant.llm.OllamaLLM import OllamaLLM
        log_file = tmp_path / "ollama.log"
        log_file.write_text(
            'msg="loading cache slot" id=0 cache=1 prompt=1 used=999 remaining=999\n'
        )
        original_path = OllamaLLM._ollama_log_path
        original_anchor = OllamaLLM._prefill_log_start
        try:
            OllamaLLM._ollama_log_path = str(log_file)
            token = OllamaLLM._begin_prefill()  # anchors at current EOF (past the stale line)
            assert OllamaLLM._prefill_log_start == log_file.stat().st_size
            with open(log_file, 'a') as f:
                f.write('msg="loading cache slot" id=0 cache=2 prompt=2 used=5786 remaining=558\n')
            assert OllamaLLM._consume_prefill_metrics() == (5786, 558)
        finally:
            OllamaLLM._end_prefill(token)
            OllamaLLM._ollama_log_path = original_path
            OllamaLLM._prefill_log_start = original_anchor

    def test_consume_prefill_metrics_parses_log_once_then_caches(self):
        # prefill_progress() is polled many times per second by the render thread.
        # _consume_prefill_metrics parses the log only on the first successful poll
        # and serves a cached value thereafter — an optimization that avoids
        # re-reading the log file every render frame once the line is found.
        from assistant.llm.OllamaLLM import OllamaLLM
        with patch.object(OllamaLLM, '_read_cache_metrics',
                          side_effect=[(5786, 558), None, None]) as mock_read:
            token = OllamaLLM._begin_prefill()  # resets _prefill_cache_metrics to None
            try:
                assert OllamaLLM._consume_prefill_metrics() == (5786, 558)
                # Subsequent polls return the cached tuple and never re-read the log.
                assert OllamaLLM._consume_prefill_metrics() == (5786, 558)
                assert OllamaLLM._consume_prefill_metrics() == (5786, 558)
                mock_read.assert_called_once()
            finally:
                OllamaLLM._end_prefill(token)

    def test_consume_prefill_metrics_throttles_rereads_until_line_appears(self):
        # Before the cache-slot line lands in the log, every poll used to re-read
        # it — over ssh that's a subprocess per TUI render frame. Failed reads
        # are now throttled to _PROGRESS_POLL_INTERVAL_S; force=True (the
        # done-time read in _log_metrics) bypasses the throttle.
        from assistant.llm.OllamaLLM import OllamaLLM
        with patch.object(OllamaLLM, '_read_cache_metrics', return_value=None) as mock_read:
            token = OllamaLLM._begin_prefill()
            try:
                assert OllamaLLM._consume_prefill_metrics() is None  # first poll reads
                assert OllamaLLM._consume_prefill_metrics() is None  # throttled
                assert OllamaLLM._consume_prefill_metrics() is None  # throttled
                mock_read.assert_called_once()
                assert OllamaLLM._consume_prefill_metrics(force=True) is None
                assert mock_read.call_count == 2
            finally:
                OllamaLLM._end_prefill(token)

    def test_prefill_log_reads_run_outside_prefill_lock(self):
        # _mark_first_token takes _prefill_lock per streamed token and the TUI
        # render thread takes it per frame. Over ssh a log read can take seconds,
        # so holding the lock across it would freeze token delivery and the TUI.
        # Pin that both readers are invoked with the lock free: a non-blocking
        # acquire from inside the (patched) reader must succeed.
        from assistant.llm.OllamaLLM import OllamaLLM

        def assert_lock_free_and_return(value):
            def _reader():
                acquired = OllamaLLM._prefill_lock.acquire(blocking=False)
                assert acquired, "log read must not run under _prefill_lock"
                OllamaLLM._prefill_lock.release()
                return value
            return _reader

        with patch.object(OllamaLLM, '_read_cache_metrics',
                          side_effect=assert_lock_free_and_return((100, 50))), \
             patch.object(OllamaLLM, '_read_prefill_progress',
                          side_effect=assert_lock_free_and_return((2048, 17531))):
            token = OllamaLLM._begin_prefill()
            try:
                assert OllamaLLM._consume_prefill_metrics() == (100, 50)
                assert OllamaLLM._poll_prefill_progress() == (2048, 17531)
            finally:
                OllamaLLM._end_prefill(token)

    def test_prefill_progress_none_when_idle(self):
        from assistant.llm.OllamaLLM import OllamaLLM
        OllamaLLM._end_prefill(OllamaLLM._begin_prefill())
        assert OllamaLLM.prefill_progress() is None

    def test_pre_heat_overlapping_a_user_request_leaves_its_prefill_state_alone(self):
        # The pre-heat fired at the end of one turn is often still prefilling when the next
        # user request goes out. Its exit used to blank the user's progress bar and swallow
        # the first-token mark; a pre-heat starting mid-request used to restart the clock.
        from assistant.llm.OllamaLLM import OllamaLLM
        with patch.object(OllamaLLM, '_log_size', return_value=0), \
             patch.object(OllamaLLM, '_consume_prefill_metrics', return_value=(0, 1000)), \
             patch.object(OllamaLLM, '_poll_prefill_progress', return_value=None):
            # pre-heat first, user request takes over, pre-heat exits
            pre_heat = OllamaLLM._begin_prefill(interactive=False)
            user = OllamaLLM._begin_prefill()
            OllamaLLM._end_prefill(pre_heat)
            assert OllamaLLM.prefill_progress() is not None
            OllamaLLM._mark_first_token(pre_heat)
            assert OllamaLLM.prefill_progress() < 100.0
            OllamaLLM._mark_first_token(user)
            assert OllamaLLM.prefill_progress() == 100.0
            # a pre-heat starting mid-request is not tracked and cannot end the user's state
            late = OllamaLLM._begin_prefill(interactive=False)
            assert late is None
            assert OllamaLLM.prefill_progress() == 100.0
            OllamaLLM._end_prefill(late)
            assert OllamaLLM.prefill_progress() == 100.0
            OllamaLLM._end_prefill(user)
            assert OllamaLLM.prefill_progress() is None
            # once the user request is over, a pre-heat is tracked again
            pre_heat = OllamaLLM._begin_prefill(interactive=False)
            assert pre_heat is not None and OllamaLLM.prefill_progress() is not None
            OllamaLLM._end_prefill(pre_heat)

    def test_prefill_progress_zero_before_cache_line(self, tmp_path):
        from assistant.llm.OllamaLLM import OllamaLLM
        original_path = OllamaLLM._ollama_log_path
        original_anchor = OllamaLLM._prefill_log_start
        try:
            # Log present but no "loading cache slot" line yet at request start.
            log_file = tmp_path / "ollama.log"
            log_file.write_text("nothing useful here\n")
            OllamaLLM._ollama_log_path = str(log_file)
            token = OllamaLLM._begin_prefill()
            assert OllamaLLM.prefill_progress() == 0.0
        finally:
            OllamaLLM._end_prefill(token)
            OllamaLLM._ollama_log_path = original_path
            OllamaLLM._prefill_log_start = original_anchor

    def test_prefill_progress_full_cache_hit_is_complete(self, tmp_path):
        from assistant.llm.OllamaLLM import OllamaLLM
        original_path = OllamaLLM._ollama_log_path
        original_anchor = OllamaLLM._prefill_log_start
        try:
            log_file = tmp_path / "ollama.log"
            OllamaLLM._ollama_log_path = str(log_file)
            token = OllamaLLM._begin_prefill()  # anchors before Ollama logs the line
            log_file.write_text(
                'msg="loading cache slot" id=0 cache=100 prompt=100 used=100 remaining=0\n'
            )
            assert OllamaLLM.prefill_progress() == 100.0
        finally:
            OllamaLLM._end_prefill(token)
            OllamaLLM._ollama_log_path = original_path
            OllamaLLM._prefill_log_start = original_anchor

    def test_prefill_progress_estimates_from_elapsed(self, tmp_path):
        import time
        from assistant.llm.OllamaLLM import OllamaLLM
        original_path = OllamaLLM._ollama_log_path
        original_anchor = OllamaLLM._prefill_log_start
        original_tps = OllamaLLM._prefill_tps
        try:
            log_file = tmp_path / "ollama.log"
            OllamaLLM._ollama_log_path = str(log_file)
            OllamaLLM._prefill_tps = 1000.0  # 1000 tokens @ 1000 t/s => 1.0s eta
            token = OllamaLLM._begin_prefill()
            log_file.write_text(
                'msg="loading cache slot" id=0 cache=0 prompt=1000 used=0 remaining=1000\n'
            )
            OllamaLLM._prefill_start_ts = time.monotonic() - 0.5  # half the eta elapsed
            pct = OllamaLLM.prefill_progress()
            assert 40.0 < pct < 60.0
        finally:
            OllamaLLM._end_prefill(token)
            OllamaLLM._ollama_log_path = original_path
            OllamaLLM._prefill_log_start = original_anchor
            OllamaLLM._prefill_tps = original_tps

    def test_prefill_progress_capped_until_first_token(self, tmp_path):
        import time
        from assistant.llm.OllamaLLM import OllamaLLM
        original_path = OllamaLLM._ollama_log_path
        original_anchor = OllamaLLM._prefill_log_start
        try:
            log_file = tmp_path / "ollama.log"
            OllamaLLM._ollama_log_path = str(log_file)
            token = OllamaLLM._begin_prefill()
            log_file.write_text(
                'msg="loading cache slot" id=0 cache=0 prompt=1000 used=0 remaining=1000\n'
            )
            OllamaLLM._prefill_start_ts = time.monotonic() - 9999  # way past eta
            assert OllamaLLM.prefill_progress() == 99.0
            OllamaLLM._mark_first_token(token)
            assert OllamaLLM.prefill_progress() == 100.0
        finally:
            OllamaLLM._end_prefill(token)
            OllamaLLM._ollama_log_path = original_path
            OllamaLLM._prefill_log_start = original_anchor

    # --- prefill-rate self-calibration (EMA) ----------------------------
    # The progress estimate's accuracy hinges on _prefill_tps, which
    # _log_metrics refines via an EMA from each completed request's observed
    # prefill rate. Near-full cache hits (tiny `remaining` over a tiny
    # duration) yield a noisy rate and are deliberately skipped so they don't
    # corrupt the estimate that drives the KV-Cache bar.

    def test_log_metrics_refines_prefill_rate_via_ema(self):
        from assistant.llm.OllamaLLM import OllamaLLM
        original_tps = OllamaLLM._prefill_tps
        original_metrics = OllamaLLM._prefill_cache_metrics
        try:
            # remaining=558 over 1.0s => observed 558 t/s; EMA blends 50/50.
            OllamaLLM._prefill_cache_metrics = (5000, 558)
            OllamaLLM._prefill_tps = 350.0
            self._make_llm()._log_metrics({
                "prompt_eval_count": 5558, "prompt_eval_duration": 1_000_000_000,
                "eval_count": 5, "eval_duration": 500_000_000,
                "total_duration": 1_500_000_000,
            })
            assert OllamaLLM._prefill_tps == pytest.approx(0.5 * 350.0 + 0.5 * 558.0)
        finally:
            OllamaLLM._prefill_tps = original_tps
            OllamaLLM._prefill_cache_metrics = original_metrics

    def test_log_metrics_skips_rate_update_on_near_full_cache_hit(self):
        from assistant.llm.OllamaLLM import OllamaLLM
        original_tps = OllamaLLM._prefill_tps
        original_metrics = OllamaLLM._prefill_cache_metrics
        try:
            # remaining=50 (<=100) is a near-full hit — too noisy to learn from.
            OllamaLLM._prefill_cache_metrics = (5000, 50)
            OllamaLLM._prefill_tps = 350.0
            self._make_llm()._log_metrics({
                "prompt_eval_count": 5050, "prompt_eval_duration": 1_000_000_000,
                "eval_count": 5, "eval_duration": 500_000_000,
                "total_duration": 1_500_000_000,
            })
            assert OllamaLLM._prefill_tps == 350.0
        finally:
            OllamaLLM._prefill_tps = original_tps
            OllamaLLM._prefill_cache_metrics = original_metrics

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

def _ssh_result(returncode=0, stdout='', stderr=''):
    result = MagicMock()
    result.returncode = returncode
    result.stdout = stdout
    result.stderr = stderr
    return result


class TestOllamaRemoteLog:
    """Remote-Ollama diagnostics: cache-metric log reads go over ssh when base_url
    points at another machine."""

    @patch('assistant.llm.OllamaLLM.config')
    def test_remote_base_url_enables_ssh_log(self, mock_config):
        mock_config.get.side_effect = lambda key, default=None: default
        from assistant.llm.OllamaLLM import OllamaLLM
        try:
            OllamaLLM(base_url='http://100.64.0.10:11434')
            assert OllamaLLM._ssh_log is not None
            assert OllamaLLM._ssh_log.host == '100.64.0.10'
            assert OllamaLLM._ssh_log.path.endswith('/log/ollama.log')
        finally:
            OllamaLLM._ssh_log = None

    @patch('assistant.llm.OllamaLLM.config')
    def test_ssh_host_and_log_path_from_config(self, mock_config):
        overrides = {'ollama.ssh_host': 'user@studio', 'ollama.remote_log_path': '/srv/ollama.log'}
        mock_config.get.side_effect = lambda key, default=None: overrides.get(key, default)
        from assistant.llm.OllamaLLM import OllamaLLM
        try:
            OllamaLLM(base_url='http://studio:11434')
            assert OllamaLLM._ssh_log.host == 'user@studio'
            assert OllamaLLM._ssh_log.path == '/srv/ollama.log'
        finally:
            OllamaLLM._ssh_log = None

    @patch('assistant.llm.OllamaLLM.config')
    def test_local_base_url_clears_previous_ssh_log(self, mock_config):
        mock_config.get.side_effect = lambda key, default=None: default
        from assistant.llm.OllamaLLM import OllamaLLM
        try:
            OllamaLLM(base_url='http://studio:11434')
            assert OllamaLLM._ssh_log is not None
            OllamaLLM(base_url='http://localhost:11434')
            assert OllamaLLM._ssh_log is None
        finally:
            OllamaLLM._ssh_log = None

    @patch('assistant.llm.OllamaLLM.subprocess.run')
    def test_ssh_size_parses_wc_output(self, mock_run):
        from assistant.llm.OllamaLLM import _SshRemoteLog
        mock_run.return_value = _ssh_result(stdout='   1234\n')
        assert _SshRemoteLog('studio', '/srv/ollama.log').size() == 1234

    @patch('assistant.llm.OllamaLLM.subprocess.run')
    def test_ssh_size_zero_when_remote_file_missing(self, mock_run):
        from assistant.llm.OllamaLLM import _SshRemoteLog
        mock_run.return_value = _ssh_result(stdout='MISSING\n')
        assert _SshRemoteLog('studio', '/srv/ollama.log').size() == 0

    @patch('assistant.llm.OllamaLLM.subprocess.run')
    def test_ssh_read_from_tails_at_offset(self, mock_run):
        from assistant.llm.OllamaLLM import _SshRemoteLog
        line = 'msg="loading cache slot" id=0 cache=6533 prompt=6344 used=5786 remaining=558\n'
        mock_run.return_value = _ssh_result(stdout=line)
        assert _SshRemoteLog('studio', '/srv/ollama.log').read_from(100) == line
        remote_cmd = mock_run.call_args[0][0][-1]
        assert 'tail -c +101' in remote_cmd  # tail offsets are 1-based

    @patch('assistant.llm.OllamaLLM.subprocess.run')
    def test_ssh_failure_engages_cooldown(self, mock_run):
        # A dead ssh connection must not stall the request path: after one
        # failure, further calls return None without spawning ssh again.
        from assistant.llm.OllamaLLM import _SshRemoteLog
        mock_run.return_value = _ssh_result(returncode=255, stderr='Connection refused')
        log = _SshRemoteLog('studio', '/srv/ollama.log')
        assert log.size() is None
        assert log.read_from(0) is None
        mock_run.assert_called_once()

    @patch('assistant.llm.OllamaLLM.subprocess.run')
    def test_ssh_cooldown_expires_and_retries(self, mock_run):
        from assistant.llm.OllamaLLM import _SshRemoteLog
        mock_run.side_effect = [_ssh_result(returncode=255, stderr='down'),
                                _ssh_result(stdout='42\n')]
        log = _SshRemoteLog('studio', '/srv/ollama.log')
        assert log.size() is None
        log._disabled_until = 0.0  # simulate cooldown expiry
        assert log.size() == 42

    def test_read_cache_metrics_via_ssh_log(self):
        from assistant.llm.OllamaLLM import OllamaLLM, _SshRemoteLog
        line = 'msg="loading cache slot" id=0 cache=6533 prompt=6344 used=5786 remaining=558\n'
        original_anchor = OllamaLLM._prefill_log_start
        try:
            OllamaLLM._ssh_log = MagicMock(spec=_SshRemoteLog)
            OllamaLLM._ssh_log.read_from.return_value = line
            OllamaLLM._prefill_log_start = 500
            assert OllamaLLM._read_cache_metrics() == (5786, 558)
            OllamaLLM._ssh_log.read_from.assert_called_with(500)
        finally:
            OllamaLLM._ssh_log = None
            OllamaLLM._prefill_log_start = original_anchor

    def test_begin_prefill_disables_metrics_when_log_unreachable(self):
        # A None anchor (ssh down at request start) must disable cache metrics
        # for the request instead of scanning the log from byte 0.
        from assistant.llm.OllamaLLM import OllamaLLM
        original_anchor = OllamaLLM._prefill_log_start
        try:
            with patch.object(OllamaLLM, '_log_size', return_value=None):
                token = OllamaLLM._begin_prefill()
            assert OllamaLLM._prefill_log_start is None
            assert OllamaLLM._read_cache_metrics() is None
        finally:
            OllamaLLM._end_prefill(token)
            OllamaLLM._prefill_log_start = original_anchor


# ===========================================================================
# TestOllamaMlxLogFormat
# ===========================================================================

class TestOllamaMlxLogFormat:
    """The MLX-engine Ollama build logs a different format than the llama.cpp
    runner: 'cache hit/miss ... cached=C left=L' instead of 'loading cache slot
    ... used=N remaining=M', plus real incremental prefill progress lines."""

    def _with_log(self, tmp_path, content):
        log_file = tmp_path / "ollama.log"
        log_file.write_text(content)
        return log_file

    def _patched_state(self, tmp_path):
        from assistant.llm.OllamaLLM import OllamaLLM
        return patch.multiple(OllamaLLM,
                              _ollama_log_path=str(tmp_path / "ollama.log"),
                              _prefill_log_start=0)

    def test_read_cache_metrics_parses_mlx_cache_hit(self, tmp_path):
        from assistant.llm.OllamaLLM import OllamaLLM
        self._with_log(tmp_path,
                       'time=... level=INFO source=prefix_cache.go:124 '
                       'msg="cache hit" total=25572 matched=25553 cached=24575 left=997\n')
        with self._patched_state(tmp_path):
            assert OllamaLLM._read_cache_metrics() == (24575, 997)

    def test_read_cache_metrics_parses_mlx_cache_miss(self, tmp_path):
        from assistant.llm.OllamaLLM import OllamaLLM
        self._with_log(tmp_path,
                       'time=... level=INFO source=prefix_cache.go:124 '
                       'msg="cache miss" total=17531 matched=2 cached=0 left=17531\n')
        with self._patched_state(tmp_path):
            assert OllamaLLM._read_cache_metrics() == (0, 17531)

    def test_read_cache_metrics_still_parses_llamacpp_format(self, tmp_path):
        from assistant.llm.OllamaLLM import OllamaLLM
        self._with_log(tmp_path,
                       'msg="loading cache slot" id=0 cache=6533 prompt=6344 used=5786 remaining=558\n')
        with self._patched_state(tmp_path):
            assert OllamaLLM._read_cache_metrics() == (5786, 558)

    def test_read_prefill_progress_returns_last_line(self, tmp_path):
        from assistant.llm.OllamaLLM import OllamaLLM
        self._with_log(tmp_path,
                       'msg="cache miss" total=17531 matched=2 cached=0 left=17531\n'
                       'msg="Prompt processing progress" processed=2048 total=17531\n'
                       'msg="Prompt processing progress" processed=4096 total=17531\n')
        with self._patched_state(tmp_path):
            assert OllamaLLM._read_prefill_progress() == (4096, 17531)

    def test_read_prefill_progress_none_without_progress_lines(self, tmp_path):
        from assistant.llm.OllamaLLM import OllamaLLM
        self._with_log(tmp_path,
                       'msg="loading cache slot" id=0 cache=1 prompt=1 used=1 remaining=1\n')
        with self._patched_state(tmp_path):
            assert OllamaLLM._read_prefill_progress() is None

    def test_prefill_progress_uses_mlx_progress_lines(self):
        from assistant.llm.OllamaLLM import OllamaLLM
        with patch.object(OllamaLLM, '_log_size', return_value=0):
            token = OllamaLLM._begin_prefill()
        try:
            with patch.object(OllamaLLM, '_read_cache_metrics', return_value=(0, 17531)), \
                 patch.object(OllamaLLM, '_read_prefill_progress', return_value=(8192, 17531)):
                progress = OllamaLLM.prefill_progress()
            assert progress == pytest.approx(8192 / 17531 * 100.0)
        finally:
            OllamaLLM._end_prefill(token)

    def test_prefill_progress_throttles_progress_log_reads(self):
        # Over ssh every progress read is a round trip; polls inside the
        # throttle window must serve the cached value without re-reading.
        from assistant.llm.OllamaLLM import OllamaLLM
        with patch.object(OllamaLLM, '_log_size', return_value=0):
            token = OllamaLLM._begin_prefill()
        try:
            with patch.object(OllamaLLM, '_read_cache_metrics', return_value=(0, 17531)), \
                 patch.object(OllamaLLM, '_read_prefill_progress',
                              return_value=(2048, 17531)) as mock_read:
                first = OllamaLLM.prefill_progress()
                second = OllamaLLM.prefill_progress()
            assert first == second == pytest.approx(2048 / 17531 * 100.0)
            mock_read.assert_called_once()
        finally:
            OllamaLLM._end_prefill(token)

# ===========================================================================
# missing_api_key — the question a caller asks before initialize() when it would rather
# sit out than crash the boot (the cache auditor, on an install without a Gemini key).
# ===========================================================================

class TestMissingApiKey:
    @patch('assistant.llm.GeminiLLM.config')
    def test_gemini_names_its_key_only_when_unset(self, mock_config):
        from assistant.llm.GeminiLLM import GeminiLLM
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
        from assistant.llm.GeminiLLM import GeminiLLM
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
