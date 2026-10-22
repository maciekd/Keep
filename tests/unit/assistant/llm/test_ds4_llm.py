"""Ds4LLM — the OpenAI-style backend for a local ds4-server, the reasoning stream and the DSML
calls it writes back as text."""
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest

from assistant.commands.formats.CommandFormats import CommandFormats
from assistant.commands.formats.DsmlFormat import DsmlFormat, OPEN
from assistant.commands.formats.KeepFormat import KeepFormat
from assistant.llm import create_llm, local_only_class, parse_llm_config, tier_api_key_name
from assistant.llm.Ds4LLM import Ds4LLM, Ds4LogFormat
from assistant.llm.Ds4Replay import ReplayStore
from assistant.llm.PrefillTracker import LogSource

CALL_TEXT = ('\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="turn_on">\n'
             '<｜DSML｜ parameter name="light_name" string="true">kitchen lamp</｜DSML｜ parameter>\n'
             '</｜DSML｜ invoke>\n</｜DSML｜ calls>')


def _config(values: dict):
    """Patch the config both the factory and Ds4LLM read: known keys from *values*, the rest default."""
    def get(key, default=None):
        return values.get(key, default)
    return patch('assistant.llm.config.get', side_effect=get), patch('assistant.llm.Ds4LLM.config.get', side_effect=get)


def _chunk(content=None, reasoning=None, tool_calls=None, usage=None, choices=True):
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    if reasoning is not None:
        delta.reasoning_content = reasoning
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)] if choices else [], usage=usage)


def _call_delta(index=0, id=None, name=None, arguments=None):
    return SimpleNamespace(index=index, id=id, type="function", function=SimpleNamespace(name=name, arguments=arguments))


def _llm_with_stream(chunks, values=None) -> tuple[Ds4LLM, MagicMock]:
    p1, p2 = _config(values or {})
    with p1, p2:
        llm = Ds4LLM()
    llm.client = MagicMock()
    llm.client.chat.completions.create.return_value = iter(chunks)
    return llm, llm.client.chat.completions.create


def _stream(llm, **kwargs) -> list[str]:
    p1, p2 = _config({})
    with p1, p2:
        return list(llm.create_streaming_completion([{'role': 'user', 'content': 'hi'}], **kwargs))


FMT = DsmlFormat(["turn_on", "dim"], table={"turn_on": ["light_name"], "dim": ["light_name", "dim_percentage"]})


@pytest.fixture(autouse=True)
def _dsml_active_and_a_memory_store():
    """A ds4 smart tier: the process format is DSML, the replay store stays off the disk, and the
    prefill tracker's log is unreachable (no anchor, so a request never reads a file or ssh)."""
    CommandFormats.set_active(DsmlFormat)
    Ds4LLM._replay_store = ReplayStore(None)
    saved_log, saved_tps = Ds4LLM._prefill.log, Ds4LLM._prefill.tps
    with patch.object(LogSource, 'size', return_value=None):
        yield
    CommandFormats.reset()
    Ds4LLM._replay_store = None
    Ds4LLM._prefill._requests.clear()
    Ds4LLM._prefill.log, Ds4LLM._prefill.tps = saved_log, saved_tps


class TestSetup:
    def test_is_keyless_for_the_setup_code_like_ollama(self):
        # Contract with the integrations that run on a tier (Photos asks tier_api_key_name for the key it
        # needs): a local server needs none, so the answer is None, not an api_keys entry.
        p1, p2 = _config({'ai.photo_indexing': {'llm': 'Ds4LLM'}})
        with p1, p2:
            assert Ds4LLM().api_key_name() is None
            assert Ds4LLM().missing_api_key() is None
            assert tier_api_key_name('photo_indexing') is None

    def test_the_openai_client_still_gets_a_key_and_the_configured_server(self):
        # The OpenAI SDK refuses to construct without a key; the server ignores whatever it is.
        p1, p2 = _config({'ds4.base_url': 'http://192.0.2.7:8000/v1'})
        with p1, p2, patch('assistant.llm.ChatGPTLLM.OpenAI') as openai_cls:
            Ds4LLM().initialize()
        kwargs = openai_cls.call_args.kwargs
        assert kwargs['base_url'] == 'http://192.0.2.7:8000/v1'
        assert kwargs['api_key']

    def test_the_factory_builds_it_by_class_name(self):
        p1, p2 = _config({})
        with p1, p2:
            llm = create_llm('Ds4LLM', model='glm-5.3')
        assert llm.name() == "DS4" and llm.display_model() == 'glm-5.3'

    def test_tier_model_wins_over_the_ds4_section_which_wins_over_the_default(self):
        p1, p2 = _config({'ds4.model': 'glm-5.3'})
        with p1, p2:
            assert Ds4LLM().model == 'glm-5.3'
            assert Ds4LLM(model='qwen3.8-flash-next').model == 'qwen3.8-flash-next'
        p1, p2 = _config({})
        with p1, p2:
            assert Ds4LLM().model == Ds4LLM.DEFAULT_MODEL

    def test_runs_the_dsml_command_format_and_a_wrapper_forwards_it(self):
        p1, p2 = _config({})
        with p1, p2:
            assert Ds4LLM().command_format() is DsmlFormat
            assert create_llm('Ds4LLM').command_format() is DsmlFormat

    def test_the_query_is_not_prepended_because_ds4s_cache_matches_from_byte_zero(self):
        p1, p2 = _config({})
        with p1, p2:
            assert Ds4LLM().prefers_prepended_user_query() is False

    def test_its_cache_preferences_reach_the_dialog_through_the_wrapper_create_llm_puts_around_it(self):
        """ds4 reuses its live context only for a prompt that extends the previous one token for
        token, and its disk checkpoint is cut before the last user turn: the dialog must hear
        both answers through the VaultLLM every backend is wrapped in — AbstractLLM's own
        defaults say no to both, and they win over the wrapper's __getattr__ — or the directive
        tail diverges the prefix every round and the suite's query merges into the toolkit turn."""
        p1, p2 = _config({})
        with p1, p2:
            wrapped = create_llm('Ds4LLM')
            assert wrapped.prefers_byte_prefix_tail() is True
            assert wrapped.prefers_query_as_own_turn() is True
            assert wrapped.prefers_prepended_user_query() is False


class TestReasoningEffort:
    def test_an_unknown_effort_spelling_falls_back_to_the_default(self):
        p1, p2 = _config({'ds4.reasoning_effort': 'lots'})
        with p1, p2:
            assert Ds4LLM.reasoning_effort() == Ds4LLM.DEFAULT_REASONING_EFFORT

    def test_a_request_without_thinking_sends_the_configured_effort_and_a_thinking_one_high(self):
        # Contract with ds4-server: the effort travels as the `reasoning_effort` body field and
        # "none" is its spelling for no reasoning at all.
        llm, create = _llm_with_stream([_chunk(content="x")])
        p1, p2 = _config({'ds4.reasoning_effort': 'none'})
        with p1, p2:
            list(llm.create_streaming_completion([{'role': 'user', 'content': 'hi'}]))
            assert create.call_args.kwargs['extra_body'] == {'reasoning_effort': 'none'}
            assert not llm.handles_thinking_internally()
            create.return_value = iter([_chunk(content="y")])
            list(llm.create_streaming_completion([{'role': 'user', 'content': 'hi'}], thinking=True))
            assert create.call_args.kwargs['extra_body'] == {'reasoning_effort': Ds4LLM.THINKING_REASONING_EFFORT}

    def test_the_model_counts_as_thinking_internally_unless_reasoning_is_off(self):
        p1, p2 = _config({'ds4.reasoning_effort': 'medium'})
        with p1, p2:
            assert Ds4LLM().handles_thinking_internally()


class TestStreaming:
    def test_reasoning_deltas_are_wrapped_in_one_thinking_block_before_the_reply(self):
        usage = SimpleNamespace(prompt_tokens=61, completion_tokens=92,
                                prompt_tokens_details=SimpleNamespace(cached_tokens=40))
        llm, _ = _llm_with_stream([
            _chunk(reasoning="Let me "), _chunk(reasoning="see."), _chunk(content="Turn "),
            _chunk(content="it on."), _chunk(choices=False, usage=usage),
        ])
        assert _stream(llm) == ["<thinking>", "Let me ", "see.", "</thinking>", "Turn ", "it on."]

    def test_a_stream_that_ends_while_reasoning_still_closes_the_block(self):
        llm, _ = _llm_with_stream([_chunk(reasoning="hmm"), _chunk(choices=False)])
        out = _stream(llm)
        assert out[-1] == "</thinking>" and "".join(out).count("<thinking>") == 1

    def test_a_reply_without_reasoning_carries_no_thinking_tags(self):
        llm, _ = _llm_with_stream([_chunk(content="Hej!"), _chunk(choices=False)])
        assert _stream(llm) == ["Hej!"]

    def test_the_dsml_content_is_withheld_and_the_tool_call_delta_is_written_back_as_dsml_text(self):
        # The probe's streamed shape (plan, 2026-10-20): the raw block as content deltas, then
        # one tool_calls delta with the whole call. The processor must see the block once.
        pieces = [CALL_TEXT[i:i + 7] for i in range(0, len(CALL_TEXT), 7)]
        llm, _ = _llm_with_stream(
            [_chunk(reasoning="just turn_on.")] + [_chunk(content=p) for p in pieces]
            + [_chunk(tool_calls=[_call_delta(id="call_1", name="turn_on", arguments='{"light_name":"kitchen lamp"}')])]
            + [_chunk(choices=False)])
        out = "".join(_stream(llm))
        assert out.startswith("<thinking>just turn_on.</thinking>")
        body = out[len("<thinking>just turn_on.</thinking>"):]
        assert body.count(OPEN) == 1 and body.lstrip().startswith(OPEN)
        assert FMT.extract_all(body) == [("turn_on", ["kitchen lamp"])]

    def test_reply_text_before_the_block_is_shown_and_the_block_follows_on_its_own_line(self):
        llm, _ = _llm_with_stream([
            _chunk(content="Turning it on."), _chunk(content="\n\n<｜DSML｜"), _chunk(content=" calls>\n…garbage…"),
            _chunk(tool_calls=[_call_delta(name="turn_on", arguments='{"light_name":"Hall lamp"}')]), _chunk(choices=False)])
        out = "".join(_stream(llm))
        assert out.startswith("Turning it on.\n")
        assert "garbage" not in out
        assert FMT.extract_all(out) == [("turn_on", ["Hall lamp"])]

    def test_tool_call_deltas_sharing_an_index_are_merged_and_several_calls_come_in_index_order(self):
        llm, _ = _llm_with_stream([
            _chunk(tool_calls=[_call_delta(index=1, name="dim", arguments='{"light_name": "B"')]),
            _chunk(tool_calls=[_call_delta(index=0, id="a", name="turn_on", arguments='{"light_name": "A"}')]),
            _chunk(tool_calls=[_call_delta(index=1, arguments=', "dim_percentage": 5}')]),
            _chunk(choices=False)])
        assert FMT.extract_all("".join(_stream(llm))) == [("turn_on", ["A"]), ("dim", ["B", "5"])]

    @patch('assistant.llm.Ds4LLM.note')
    def test_withheld_text_with_no_tool_call_behind_it_is_released_as_written_and_noted(self, mock_note):
        llm, _ = _llm_with_stream([_chunk(content="Sure."), _chunk(content=CALL_TEXT), _chunk(choices=False)])
        out = "".join(_stream(llm))
        assert out == "Sure." + CALL_TEXT
        assert mock_note.called

    def test_withheld_text_behind_a_nameless_tool_call_is_not_lost(self):
        llm, _ = _llm_with_stream([_chunk(content="Sure."), _chunk(content=CALL_TEXT),
                                   _chunk(tool_calls=[_call_delta(index=0, id="c1", name=None, arguments="{}")]),
                                   _chunk(choices=False)])
        out = "".join(_stream(llm))
        assert "kitchen lamp" in out, "the call the model wrote reaches the parser one way or the other"

    def test_a_split_opener_in_plain_text_is_not_lost(self):
        llm, _ = _llm_with_stream([_chunk(content="a <"), _chunk(content="b"), _chunk(choices=False)])
        assert "".join(_stream(llm)) == "a <b"

    def test_an_opener_prefix_at_the_very_end_is_still_released(self):
        llm, _ = _llm_with_stream([_chunk(content="look: <｜DSML"), _chunk(choices=False)])
        assert "".join(_stream(llm)) == "look: <｜DSML"


class TestTheRequestShape:
    """Contract with ds4-server's live KV cache (plan, 2026-10-20): a call turn is kept only in
    the tool shape, which a request enters by declaring a tool — the same one every time."""

    def test_every_request_declares_the_one_static_tool(self):
        llm, create = _llm_with_stream([_chunk(content="x")])
        _stream(llm)
        tools = create.call_args.kwargs['tools']
        assert len(tools) == 1 and tools[0]['function']['name'] == "load_toolkit"
        assert tools[0] is Ds4LLM.STATIC_TOOL

    def test_a_streamed_call_fills_the_store_and_the_next_request_replays_it_by_id(self):
        llm, create = _llm_with_stream([
            _chunk(reasoning="lamp on."), _chunk(content="\n\n" + CALL_TEXT.strip()),
            _chunk(tool_calls=[_call_delta(id="call_7", name="turn_on", arguments='{"light_name":"kitchen lamp"}')]),
            _chunk(choices=False)])
        out = "".join(_stream(llm))
        row = out.replace("<thinking>lamp on.</thinking>", "").strip()  # what DialogManager's history keeps
        create.return_value = iter([_chunk(content="Done."), _chunk(choices=False)])
        p1, p2 = _config({})
        with p1, p2:
            list(llm.create_streaming_completion([{'role': 'user', 'content': 'hi'}, {'role': 'assistant', 'content': row},
                                                  {'role': 'user', 'content': 'result: \nKitchen lamp is on.'}]))
        sent = create.call_args.kwargs['messages']
        assert sent[1]['reasoning_content'] == "lamp on."
        assert sent[1]['content'] == "" and sent[1]['tool_calls'][0]['id'] == "call_7"
        assert sent[2] == {'role': 'tool', 'tool_call_id': 'call_7', 'content': 'result: \nKitchen lamp is on.'}

    def test_a_text_reply_keeps_its_reasoning_for_the_replay(self):
        llm, create = _llm_with_stream([_chunk(reasoning="say hi"), _chunk(content="Hej!"), _chunk(choices=False)])
        _stream(llm)
        create.return_value = iter([_chunk(content="x"), _chunk(choices=False)])
        p1, p2 = _config({})
        with p1, p2:
            list(llm.create_streaming_completion([{'role': 'user', 'content': 'hi'}, {'role': 'assistant', 'content': 'Hej!'},
                                                  {'role': 'user', 'content': 'again'}]))
        assert create.call_args.kwargs['messages'][1] == {'role': 'assistant', 'content': 'Hej!', 'reasoning_content': 'say hi'}

    def test_a_parsed_call_is_written_back_in_the_active_format_not_always_dsml(self):
        # The mixed-tier case: ds4 as the fast tier under a Claude smart tier parses Keep's syntax.
        CommandFormats.set_active(KeepFormat)
        llm, _ = _llm_with_stream([
            _chunk(tool_calls=[_call_delta(id="c", name="dim", arguments='{"light_name": "Lamp", "dim_percentage": 5}')]),
            _chunk(choices=False)])
        out = "".join(_stream(llm))
        assert OPEN not in out
        assert KeepFormat(["dim"]).extract_all(out) == [("dim", ["Lamp", "5"])]


class TestNonStreaming:
    def _completion(self, message):
        llm, create = _llm_with_stream([])
        create.return_value = SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)
        p1, p2 = _config({})
        with p1, p2:
            return llm.create_completion([{'role': 'user', 'content': 'hi'}])

    def test_returns_the_text_and_keeps_the_reasoning_out_of_it(self):
        message = SimpleNamespace(content="Hej!", reasoning_content="Swedish hello, one line.", tool_calls=None)
        assert self._completion(message) == "Hej!"

    def test_tool_calls_are_written_back_as_dsml_text_after_the_content(self):
        # ds4 non-streamed: content empty, the call in tool_calls (plan, 2026-10-20).
        call = SimpleNamespace(id="call_1", function=SimpleNamespace(name="dim", arguments='{"light_name": "Lamp", "dim_percentage": 1}'))
        text = self._completion(SimpleNamespace(content="", reasoning_content="", tool_calls=[call]))
        assert text.startswith(OPEN) and FMT.extract_all(text) == [("dim", ["Lamp", "1"])]
        text = self._completion(SimpleNamespace(content="Dimming.", reasoning_content=None, tool_calls=[call]))
        assert text.startswith("Dimming.\n") and FMT.extract_all(text) == [("dim", ["Lamp", "1"])]


class TestPrefillBar:
    """The "Prefilling…" bar: ds4's log lines, the tracker bracketed around a request, and the
    rate each reply's usage teaches."""

    LOG = ('1021 11:39:28 ds4-server: chat ctx=24576..26339:1763 TOOLS prompt start\n'
           '1021 11:39:44 ds4-server: chat ctx=24576..26339:1763 TOOLS prefill chunk 1024/1763 (58.1%) chunk=0.00 t/s avg=1518.73 t/s 16.182s\n'
           '1021 11:39:47 ds4-server: chat ctx=24576..26339:1763 TOOLS prefill chunk 1700/1763 (96.4%) chunk=656.29 t/s avg=1396.54 t/s 18.853s\n')

    def test_the_prompt_start_line_gives_cached_and_remaining_tokens(self):
        assert Ds4LogFormat().work(self.LOG) == (24576, 1763)
        assert Ds4LogFormat().work("1021 11:39:28 ds4-server: kv cache stored tokens=26329\n") is None

    def test_progress_is_the_last_chunk_line(self):
        assert Ds4LogFormat().progress(self.LOG) == (1700, 1763)
        assert Ds4LogFormat().progress(self.LOG.splitlines()[0]) is None

    def test_a_streamed_request_is_bracketed_and_the_first_delta_ends_the_prefill(self):
        llm, create = _llm_with_stream([_chunk(reasoning="hm"), _chunk(content="Hi."), _chunk(choices=False)])
        order = []
        create.side_effect = lambda **kw: (order.append("create"), iter([_chunk(reasoning="hm"), _chunk(content="Hi."), _chunk(choices=False)]))[1]
        with patch.object(Ds4LLM._prefill, 'begin', side_effect=lambda **kw: (order.append("begin"), 7)[1]), \
             patch.object(Ds4LLM._prefill, 'first_token', side_effect=lambda t: order.append(("first", t))), \
             patch.object(Ds4LLM._prefill, 'end', side_effect=lambda t: order.append(("end", t))):
            _stream(llm)
        assert order[:3] == ["begin", "create", ("first", 7)], "the tracker runs before the request leaves; the first delta ends the prefill"
        assert order.count(("first", 7)) == 1 and order[-1] == ("end", 7)

    def test_the_tracker_is_released_when_the_stream_raises(self):
        def broken():
            yield _chunk(content="Hi")
            raise ConnectionError("gone")
        llm, create = _llm_with_stream([])
        create.return_value = broken()
        with patch.object(Ds4LLM._prefill, 'end') as end, pytest.raises(ConnectionError):
            _stream(llm)
        end.assert_called_once()

    def test_a_non_streamed_request_is_bracketed_too(self):
        p1, p2 = _config({})
        with p1, p2:
            llm = Ds4LLM()
        llm.client = MagicMock()
        message = SimpleNamespace(content="Hi.", reasoning_content=None, tool_calls=None)
        llm.client.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)
        with patch.object(Ds4LLM._prefill, 'begin', return_value=3) as begin, patch.object(Ds4LLM._prefill, 'end') as end, p1, p2:
            llm.create_completion([{'role': 'user', 'content': 'hi'}])
        begin.assert_called_once()
        end.assert_called_once_with(3)

    def test_the_usage_teaches_the_rate_from_the_uncached_tokens_over_the_time_to_the_first_token(self):
        usage = SimpleNamespace(prompt_tokens=5000, completion_tokens=3,
                                prompt_tokens_details=SimpleNamespace(cached_tokens=4000))
        llm, _ = _llm_with_stream([_chunk(content="Hi."), _chunk(choices=False, usage=usage)])
        with patch.object(Ds4LLM._prefill, 'observe_rate') as observe:
            _stream(llm)
        (tokens, seconds), _ = observe.call_args
        assert tokens == 1000 and 0 <= seconds < 5

    def test_no_usage_means_no_rate_lesson(self):
        llm, _ = _llm_with_stream([_chunk(content="Hi."), _chunk(choices=False)])
        with patch.object(Ds4LLM._prefill, 'observe_rate') as observe:
            _stream(llm)
        observe.assert_not_called()

    def test_a_local_server_means_the_log_file_here_and_a_remote_one_ssh_to_its_host(self):
        p1, p2 = _config({})
        with p1, p2:
            Ds4LLM(base_url='http://127.0.0.1:8000/v1')
            log = Ds4LLM._prefill.log
            assert log.ssh_host is None and log.path == Ds4LLM.DEFAULT_LOG_PATH
            Ds4LLM(base_url='http://100.64.0.10:8000/v1')
            log = Ds4LLM._prefill.log
            assert log.ssh_host == '100.64.0.10' and log.path == Ds4LLM.DEFAULT_LOG_PATH

    def test_the_log_path_and_ssh_host_keys_override_the_defaults(self):
        p1, p2 = _config({'ds4.log_path': '/srv/ds4.log', 'ds4.ssh_host': 'user@studio.local'})
        with p1, p2:
            Ds4LLM(base_url='http://studio.local:8000/v1')
        log = Ds4LLM._prefill.log
        assert (log.path, log.ssh_host) == ('/srv/ds4.log', 'user@studio.local')


class TestLocalOnlyRouting:
    """`ollama.local_only` holding a class name routes every tier to that backend (the switch
    the dev machine's config.yaml uses to run the whole assistant on ds4)."""

    def _get(self, tier_cfg, local_only):
        def get(key, default=None):
            if key == 'ollama.local_only':
                return local_only
            if key.startswith('ai.'):
                return tier_cfg
            return default
        return patch('assistant.llm.config.get', side_effect=get)

    def test_true_still_means_ollama(self):
        with self._get(None, True):
            assert local_only_class() == 'OllamaLLM'
            assert parse_llm_config('smart') == ('OllamaLLM', None)

    def test_the_text_a_config_set_stores_means_the_same_as_the_bool(self):
        with self._get(None, "true"):
            assert local_only_class() == 'OllamaLLM'
        with self._get(None, "false"):
            assert local_only_class() is None

    def test_a_class_name_routes_every_cloud_tier_there_without_its_cloud_model(self):
        with self._get(None, 'Ds4LLM'):
            for tier in ('fast', 'smart', 'vision', 'video', 'photo_indexing', 'memory_curator', 'cache_auditor'):
                assert parse_llm_config(tier) == ('Ds4LLM', None)

    def test_a_tier_already_on_the_named_class_keeps_its_model(self):
        with self._get({'llm': 'Ds4LLM', 'model': 'glm-5.3'}, 'Ds4LLM'):
            assert parse_llm_config('smart') == ('Ds4LLM', 'glm-5.3')

    def test_a_tier_on_ollama_moves_too_and_loses_its_ollama_model(self):
        with self._get({'llm': 'OllamaLLM', 'model': 'qwen3.8:4b'}, 'Ds4LLM'):
            assert parse_llm_config('cache_auditor') == ('Ds4LLM', None)

    def test_code_quality_stays_on_the_claude_cli(self):
        with self._get({'model': 'claude-sonnet-5-5'}, 'Ds4LLM'):
            assert parse_llm_config('code_quality') == ('ClaudeLLM', 'claude-sonnet-5-5')

    @pytest.mark.parametrize("off", [False, None, "", "   "])
    def test_off_values_leave_the_cloud_tiers_alone(self, off):
        with self._get({'llm': 'GeminiLLM', 'model': 'gemini-3.8-flash'}, off):
            assert local_only_class() is None
            assert parse_llm_config('video') == ('GeminiLLM', 'gemini-3.8-flash')
