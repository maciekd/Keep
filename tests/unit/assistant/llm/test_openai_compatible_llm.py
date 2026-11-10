"""OpenAICompatibleLLM — any local OpenAI-compatible server (oMLX, llama.cpp) as a backend: the
config that names it, the request it sends, and the prefill bar fed from the server's own
progress chunks instead of a log."""
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest

from assistant.llm import create_llm, local_only_class, parse_llm_config, tier_api_key_name
from assistant.llm.Ds4LLM import Ds4LLM
from assistant.llm.OpenAICompatibleLLM import OpenAICompatibleLLM
from assistant.llm.PrefillTracker import LogSource
from common import ConfigSchema


def _config(values: dict):
    def get(key, default=None):
        return values.get(key, default)
    return patch('assistant.llm.OpenAICompatibleLLM.config.get', side_effect=get)


def _chunk(content=None, reasoning=None, usage=None, choices=True, finish_reason=None):
    delta = SimpleNamespace(content=content, tool_calls=None)
    if reasoning is not None:
        delta.reasoning_content = reasoning
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=finish_reason)] if choices else [],
                           usage=usage)


def _progress(total, cache, processed, time_ms=None):
    """A `prompt_progress` chunk as the SDK hands it back: no choices, the extra field a dict."""
    return SimpleNamespace(choices=[], usage=None,
                           prompt_progress={'total': total, 'cache': cache, 'processed': processed, 'time_ms': time_ms})


def _usage(prompt_tokens, cached):
    return SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=3,
                           prompt_tokens_details=SimpleNamespace(cached_tokens=cached))


def _llm(values=None, chunks=None) -> OpenAICompatibleLLM:
    with _config(values or {}):
        llm = OpenAICompatibleLLM()
    llm.client = MagicMock()
    llm.client.with_options.return_value = llm.client
    if chunks is not None:
        llm.client.chat.completions.create.return_value = chunks
    return llm


def _stream(llm, values=None, **kwargs) -> list[str]:
    with _config(values or {}):
        return list(llm.create_streaming_completion([{'role': 'user', 'content': 'hi'}], **kwargs))


@pytest.fixture(autouse=True)
def _clean_class_state():
    OpenAICompatibleLLM._models_checked.clear()
    OpenAICompatibleLLM._discovered_models.clear()
    OpenAICompatibleLLM._discovery_failed.clear()
    with patch.object(LogSource, 'size', return_value=None):
        yield
    OpenAICompatibleLLM._prefill._requests.clear()
    Ds4LLM._prefill._requests.clear()


class TestConfig:
    def test_local_backend_openai_routes_every_tier_there(self):
        # Contract with the `local_backend` config key: `openai` is the third local backend.
        def get(key, default=None):
            return 'openai' if key == 'local_backend' else default
        with patch('assistant.llm.config.get', side_effect=get):
            assert local_only_class() == 'OpenAICompatibleLLM'
            assert parse_llm_config('smart') == ('OpenAICompatibleLLM', None)
            assert parse_llm_config('code_quality')[0] == 'ClaudeLLM'

    def test_the_section_keys_are_in_the_schema(self):
        # Contract with config.example.yaml: `config_help` and the setup read the section from it.
        for key in ('base_url', 'model', 'api_key', 'reasoning_effort', 'return_progress', 'background_idle_seconds'):
            assert ConfigSchema.is_known_key(f'openai_compatible.{key}'), key

    def test_the_tier_model_wins_over_the_section_and_an_unset_model_waits_for_the_server(self):
        with _config({'openai_compatible.model': 'qwen3-27b'}):
            assert OpenAICompatibleLLM().model == 'qwen3-27b'
            assert OpenAICompatibleLLM(model='gemma-4').model == 'gemma-4'
        with _config({}):
            assert OpenAICompatibleLLM().model is None
            assert OpenAICompatibleLLM().display_model(), "a notice still has something to say"

    def test_keyless_for_the_setup_code_but_the_client_gets_a_key(self):
        # The SDK refuses to construct without a key; a server that takes none ignores the placeholder.
        with _config({'ai.photo_indexing': {'llm': 'OpenAICompatibleLLM'}}), \
                patch('assistant.llm.config.get', side_effect=lambda k, d=None: {'ai.photo_indexing': {'llm': 'OpenAICompatibleLLM'}}.get(k, d)):
            assert OpenAICompatibleLLM().api_key_name() is None
            assert tier_api_key_name('photo_indexing') is None
            assert OpenAICompatibleLLM().api_key
        with _config({'openai_compatible.api_key': 'sk-local-1'}):
            assert OpenAICompatibleLLM().api_key == 'sk-local-1'

    def test_the_client_has_no_read_timeout_and_the_configured_server(self):
        with _config({'openai_compatible.base_url': 'http://192.0.2.7:8000/v1'}), \
                patch('assistant.llm.OpenAICompatibleLLM.OpenAI') as openai_cls:
            OpenAICompatibleLLM().initialize()
        kwargs = openai_cls.call_args.kwargs
        assert kwargs['base_url'] == 'http://192.0.2.7:8000/v1'
        assert kwargs['timeout'].read is None and kwargs['timeout'].connect is not None

    def test_its_local_preferences_reach_the_dialog_through_the_wrapper(self):
        with _config({}):
            wrapped = create_llm('OpenAICompatibleLLM')
        assert wrapped.is_local() is True
        assert wrapped.prefers_prepended_user_query() is False
        assert wrapped.prefers_byte_prefix_tail() is True
        assert wrapped.prefers_query_as_own_turn() is False, "ds4's cold-anchor trick, not a generic server's"


class TestCommandFormat:
    def test_a_qwen_model_writes_qwen_syntax_and_any_other_keeps_own(self):
        from assistant.commands.formats.KeepFormat import KeepFormat
        from assistant.commands.formats.QwenFormat import QwenFormat
        with _config({}):
            assert OpenAICompatibleLLM(model='Qwen3.8-27B-oQ4e-mtp').command_format() is QwenFormat
            assert OpenAICompatibleLLM(model='gemma-4-26b').command_format() is KeepFormat
            assert OpenAICompatibleLLM().command_format() is KeepFormat, "no model name to read yet"
            assert create_llm('OpenAICompatibleLLM', model='qwen3-27b').command_format() is QwenFormat

    def test_the_tier_key_wins_over_the_model_name(self):
        from assistant.commands.formats.KeepFormat import KeepFormat
        with _config({}):
            assert create_llm('OpenAICompatibleLLM', model='qwen3-27b', command_format='keep').command_format() is KeepFormat

    def test_ds4_keeps_its_own_choice_whatever_the_model_is_called(self):
        from assistant.commands.formats.KeepFormat import KeepFormat
        with _config({}):
            assert Ds4LLM(model='qwen3.8-flash-next').command_format() is KeepFormat


class TestModelCheck:
    """`initialize` reads `GET /v1/models` once: the configured model is checked, an unset one
    becomes the first listed, and the server's context length is logged when it says."""

    def _listing(self, llm, models):
        llm.client.models.list.return_value = models

    def test_an_unset_model_becomes_the_first_the_server_lists(self):
        llm = _llm({})
        self._listing(llm, [SimpleNamespace(id='qwen3-27b', max_model_len=32768), SimpleNamespace(id='gemma-4')])
        llm._check_model()
        assert llm.model == 'qwen3-27b'

    def test_a_model_the_server_does_not_list_is_kept_and_noted(self):
        llm = _llm({'openai_compatible.model': 'typo-7b'})
        self._listing(llm, [SimpleNamespace(id='qwen3-27b')])
        with patch('assistant.llm.OpenAICompatibleLLM.note') as noted:
            llm._check_model()
        assert llm.model == 'typo-7b', "a server may take a name it does not list"
        assert noted.call_count == 1 and 'qwen3-27b' in noted.call_args[0][0]

    def test_an_unreadable_list_leaves_a_configured_model_unchecked(self):
        llm = _llm({'openai_compatible.model': 'qwen3-27b'})
        llm.client.models.list.side_effect = ConnectionError("refused")
        llm._check_model()
        assert llm.model == 'qwen3-27b'

    def test_no_model_anywhere_is_an_error_naming_the_key_but_not_a_refusal(self):
        # A single-model server (llama.cpp) ignores the field, so the requests still go out.
        llm = _llm({})
        self._listing(llm, [])
        with patch('assistant.llm.OpenAICompatibleLLM.error') as err:
            llm._check_model()
        assert err.call_count == 1 and 'openai_compatible.model' in err.call_args[0][0]
        assert llm.model is None

    def test_the_check_runs_once_per_server_and_model(self):
        llm = _llm({'openai_compatible.model': 'qwen3-27b'})
        self._listing(llm, [SimpleNamespace(id='qwen3-27b')])
        llm._check_model()
        llm._check_model()
        assert llm.client.models.list.call_count == 1

    def test_the_discovered_model_reaches_every_later_instance_of_the_server_without_a_second_read(self):
        """The dialog builds an instance per tier and request; with `model` unset each of them
        must send the discovered id, not null (2026-11-10: only the first ever did)."""
        first = _llm({})
        self._listing(first, [SimpleNamespace(id='qwen3-27b')])
        first._check_model()
        second = _llm({})
        second._check_model()
        assert second.model == 'qwen3-27b'
        assert not second.client.models.list.called

    def test_an_unreadable_list_leaves_an_unset_model_to_the_next_instance(self):
        first = _llm({})
        first.client.models.list.side_effect = ConnectionError("refused")
        with patch('assistant.llm.OpenAICompatibleLLM.error') as err:
            first._check_model()
            second = _llm({})
            self._listing(second, [SimpleNamespace(id='qwen3-27b')])
            second._check_model()
        assert first.model is None and second.model == 'qwen3-27b'
        assert err.call_count == 1, "the empty list is one error line per server, not one per instance"

    def test_the_discovery_is_per_server(self):
        here = _llm({})
        self._listing(here, [SimpleNamespace(id='qwen3-27b')])
        here._check_model()
        there = _llm({'openai_compatible.base_url': 'http://192.0.2.7:8000/v1'})
        self._listing(there, [SimpleNamespace(id='gemma-4')])
        there._check_model()
        assert (here.model, there.model) == ('qwen3-27b', 'gemma-4')


class TestTheRequest:
    def test_a_streamed_request_asks_for_usage_and_progress(self):
        llm = _llm({}, chunks=iter([_chunk(content="Hi."), _chunk(choices=False)]))
        _stream(llm)
        kwargs = llm.client.chat.completions.create.call_args.kwargs
        assert kwargs['stream_options'] == {'include_usage': True}
        assert kwargs['extra_body']['return_progress'] is True
        assert kwargs['extra_body']['reasoning_effort'] == 'low'
        assert 'tools' not in kwargs

    def test_return_progress_false_leaves_the_key_out_for_a_strict_server(self):
        llm = _llm({}, chunks=iter([_chunk(content="Hi."), _chunk(choices=False)]))
        _stream(llm, values={'openai_compatible.return_progress': False})
        assert 'return_progress' not in llm.client.chat.completions.create.call_args.kwargs['extra_body']

    def test_ds4_never_asks_for_progress_since_its_bar_reads_the_log(self):
        with _config({}), patch.object(Ds4LLM, 'replay_store', return_value=MagicMock(turn=lambda k: None)):
            llm = Ds4LLM()
            kwargs = llm._request_kwargs([{'role': 'user', 'content': 'hi'}], False, stream=True)
        assert 'return_progress' not in kwargs['extra_body']

    def test_a_dsml_looking_reply_passes_through_visible(self):
        # The withholding is ds4's: a generic server's content is the reply, sentinel or not.
        from assistant.commands.formats.DsmlFormat import OPEN
        llm = _llm({}, chunks=iter([_chunk(content="Sure " + OPEN), _chunk(content=" and more"), _chunk(choices=False)]))
        assert "".join(_stream(llm)) == "Sure " + OPEN + " and more"

    def test_reasoning_is_wrapped_in_thinking_tags(self):
        llm = _llm({}, chunks=iter([_chunk(reasoning="hm"), _chunk(content="Hi."), _chunk(choices=False)]))
        assert _stream(llm) == ["<thinking>", "hm", "</thinking>", "Hi."]


class TestPrefillBar:
    """The bar's numbers come off the wire: each `prompt_progress` chunk reaches the tracker,
    the first one is the workload the early cache report is made from."""

    def test_progress_chunks_drive_the_bar_while_the_server_prefills(self):
        seen = []
        tracker = OpenAICompatibleLLM._prefill

        def chunks():
            yield _progress(total=5000, cache=4000, processed=4500, time_ms=250.0)
            seen.append((tracker.progress(), tracker.rate()))
            yield _chunk(content="Hi.")
            seen.append((tracker.progress(), None))
            yield _chunk(choices=False, usage=_usage(5000, 4000))

        llm = _llm({}, chunks=chunks())
        _stream(llm)
        assert seen[0] == (pytest.approx(50.0), pytest.approx(2000.0)), "500 of 1000 uncached tokens in 0.25 s"
        assert seen[1][0] == 100.0, "the first token ends the prefill"
        assert tracker.progress() is None, "the request is out of the tracker once the stream ends"

    def test_the_cache_report_is_made_from_the_first_progress_chunk_and_not_again_from_the_usage(self):
        llm = _llm({}, chunks=iter([_progress(total=5000, cache=4000, processed=4096, time_ms=10.0),
                                    _progress(total=5000, cache=4000, processed=5000, time_ms=400.0),
                                    _chunk(content="Hi."), _chunk(choices=False, usage=_usage(5000, 4000))]))
        reports = []
        with patch.object(llm, 'report_cache_usage', side_effect=lambda m, c, t, note=None: reports.append((c, t))):
            _stream(llm)
        assert reports == [(4000, 5000)]

    def test_without_progress_chunks_the_usage_makes_the_report(self):
        llm = _llm({}, chunks=iter([_chunk(content="Hi."), _chunk(choices=False, usage=_usage(5000, 4000))]))
        reports = []
        with patch.object(llm, 'report_cache_usage', side_effect=lambda m, c, t, note=None: reports.append((c, t))):
            _stream(llm)
        assert reports == [(4000, 5000)]

    def test_the_servers_prefill_time_teaches_the_rate(self):
        llm = _llm({}, chunks=iter([_progress(total=5000, cache=4000, processed=5000, time_ms=500.0),
                                    _chunk(content="Hi."), _chunk(choices=False, usage=_usage(5000, 4000))]))
        with patch.object(OpenAICompatibleLLM._prefill, 'observe_rate') as observe:
            _stream(llm)
        observe.assert_called_once_with(1000, pytest.approx(0.5))

    def test_a_malformed_progress_chunk_is_ignored(self):
        bad = SimpleNamespace(choices=[], usage=None, prompt_progress={'total': 'lots'})
        llm = _llm({}, chunks=iter([bad, _chunk(content="Hi."), _chunk(choices=False)]))
        assert _stream(llm) == ["Hi."]

    def test_a_raw_sse_progress_chunk_is_read_the_same_way(self):
        # The background image request reads the stream itself: the chunk is a dict there.
        llm = _llm({})
        token = OpenAICompatibleLLM._prefill.begin()
        try:
            assert llm._take_progress(token, {'choices': [], 'prompt_progress': {'total': 100, 'cache': 0, 'processed': 50}})
            assert OpenAICompatibleLLM._prefill.progress() == pytest.approx(50.0)
            assert not llm._take_progress(token, {'choices': []})
        finally:
            OpenAICompatibleLLM._prefill.end(token)
