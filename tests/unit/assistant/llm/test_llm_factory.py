import sys
import types

import pytest
from unittest.mock import patch

from assistant.llm import create_llm, parse_llm_config, command_format_override, command_format_chooser, speed_override
from assistant.llm.VaultLLM import VaultLLM
from common.vault.Vault import Vault


# ===========================================================================
# parse_llm_config — picks the LLM class/model for a tier from config,
# falling back to the cloud defaults that production relies on.
#
# The branching here guards a real near-miss bug (see commit "default to cloud
# LLMs when no `ai` config block is present"): without it, merging the local-AI
# branch would silently switch prod off Gemini+Claude and onto local Ollama.
# ===========================================================================

class TestParseLLMConfig:
    def _patch_config(self, tier_value, local_only='none'):
        """Patch assistant.llm.config.get: every ai.<tier> lookup yields tier_value (or raises it),
        local_backend yields local_only, anything else its default."""
        def get(key, default=None):
            if key == 'local_backend':
                return local_only
            if key.startswith('ai.'):
                if isinstance(tier_value, Exception):
                    raise tier_value
                return tier_value
            return default
        p = patch('assistant.llm.config.get', side_effect=get)
        return p, p.start()

    def test_string_value_yields_class_with_no_model(self):
        p, _ = self._patch_config('OllamaLLM')
        try:
            assert parse_llm_config('fast') == ('OllamaLLM', None)
        finally:
            p.stop()

    def test_dict_does_not_inherit_default_model_for_different_class(self):
        # The crucial guard: an explicit Ollama backend must NOT pick up Claude's
        # default model.
        p, _ = self._patch_config({'llm': 'OllamaLLM'})
        try:
            assert parse_llm_config('smart') == ('OllamaLLM', None)
        finally:
            p.stop()

    def test_dict_explicit_model_is_used(self):
        p, _ = self._patch_config({'llm': 'OllamaLLM', 'model': 'gemma4:26b'})
        try:
            assert parse_llm_config('smart') == ('OllamaLLM', 'gemma4:26b')
        finally:
            p.stop()

    def test_unknown_tier_falls_back_to_ollama(self):
        p, _ = self._patch_config(KeyError("ai.bogus"))
        try:
            assert parse_llm_config('bogus') == ('OllamaLLM', None)
        finally:
            p.stop()

    def test_local_only_routes_cloud_default_tier_to_ollama_without_its_model(self):
        # No ai block at all: the cloud default (a Claude/Gemini model name) must not leak to Ollama.
        p, _ = self._patch_config(None, local_only='ollama')
        try:
            for tier in ('fast', 'smart', 'vision', 'video', 'photo_indexing', 'photo_backfill', 'memory_curator'):
                assert parse_llm_config(tier) == ('OllamaLLM', None)
        finally:
            p.stop()

    def test_local_only_overrides_an_explicit_cloud_tier(self):
        p, _ = self._patch_config({'llm': 'GeminiLLM', 'model': 'gemini-3.8-flash'}, local_only='ollama')
        try:
            assert parse_llm_config('video') == ('OllamaLLM', None)
        finally:
            p.stop()

    def test_local_only_keeps_the_model_of_a_tier_already_on_ollama(self):
        p, _ = self._patch_config({'llm': 'OllamaLLM', 'model': 'qwen3.8:4b'}, local_only='ollama')
        try:
            assert parse_llm_config('cache_auditor') == ('OllamaLLM', 'qwen3.8:4b')
        finally:
            p.stop()

    def test_local_only_leaves_code_quality_alone(self):
        # The nightly Claude Code CLI runs cannot move to Ollama; only its model is read.
        p, _ = self._patch_config({'model': 'claude-sonnet-5-5'}, local_only='ollama')
        try:
            assert parse_llm_config('code_quality') == ('ClaudeLLM', 'claude-sonnet-5-5')
        finally:
            p.stop()

    def test_local_only_off_leaves_cloud_tiers_untouched(self):
        p, _ = self._patch_config({'llm': 'GeminiLLM', 'model': 'gemini-3.8-flash'}, local_only='none')
        try:
            assert parse_llm_config('video') == ('GeminiLLM', 'gemini-3.8-flash')
        finally:
            p.stop()

# ===========================================================================
# create_llm — instantiates an LLM by class name, forwarding model and (only
# when accepted) prompt_cache, and wraps the result in a VaultLLM. We inject a
# fake module so the test exercises the factory's kwargs/introspection logic
# without constructing real backends (which need API keys).
# ===========================================================================

class _FakeLLMWithCache:
    def __init__(self, model='default-model', prompt_cache=True):
        self.model = model
        self.prompt_cache = prompt_cache


class _FakeLLMNoCache:
    def __init__(self, model='default-model'):
        self.model = model


class TestCreateLLM:
    @pytest.fixture(autouse=True)
    def fresh_vault_singleton(self):
        # VaultLLM reads Vault.get_instance(); never let one test's singleton leak into another.
        Vault.reset_instance()
        yield
        Vault.reset_instance()

    def _inject(self, class_name, cls):
        """Register a fake assistant.llm.<class_name> module exposing cls."""
        module = types.ModuleType(f"assistant.llm.{class_name}")
        setattr(module, class_name, cls)
        sys.modules[f"assistant.llm.{class_name}"] = module
        return module

    def test_model_override_forwarded(self):
        self._inject("_FakeLLMNoCache", _FakeLLMNoCache)
        try:
            inst = create_llm("_FakeLLMNoCache", model="my-model")
            assert inst.model == "my-model"
        finally:
            del sys.modules["assistant.llm._FakeLLMNoCache"]

    def test_no_model_uses_constructor_default(self):
        self._inject("_FakeLLMNoCache", _FakeLLMNoCache)
        try:
            inst = create_llm("_FakeLLMNoCache")
            assert inst.model == "default-model"
        finally:
            del sys.modules["assistant.llm._FakeLLMNoCache"]

    def test_prompt_cache_forwarded_when_accepted(self):
        self._inject("_FakeLLMWithCache", _FakeLLMWithCache)
        try:
            inst = create_llm("_FakeLLMWithCache", prompt_cache=False)
            assert inst.prompt_cache is False
        finally:
            del sys.modules["assistant.llm._FakeLLMWithCache"]

    def test_prompt_cache_ignored_when_not_accepted(self):
        # Must not raise even though the constructor has no prompt_cache param.
        self._inject("_FakeLLMNoCache", _FakeLLMNoCache)
        try:
            inst = create_llm("_FakeLLMNoCache", prompt_cache=True)
            assert isinstance(inst.llm, _FakeLLMNoCache)
        finally:
            del sys.modules["assistant.llm._FakeLLMNoCache"]

    def test_result_is_a_vault_llm_around_the_backend(self):
        # Every LLM the system builds goes through the vault boundary (VAULT_PLAN §3.6);
        # the backend itself is reachable as .llm, and its attributes pass through.
        self._inject("_FakeLLMWithCache", _FakeLLMWithCache)
        try:
            inst = create_llm("_FakeLLMWithCache", model="my-model", prompt_cache=True)
            assert isinstance(inst, VaultLLM)
            assert isinstance(inst.llm, _FakeLLMWithCache)
            assert inst.model == "my-model" and inst.prompt_cache is True
        finally:
            del sys.modules["assistant.llm._FakeLLMWithCache"]


# ===========================================================================
# command_format — a tier's `ai.<tier>.command_format` key overrides the syntax the
# LLM class would write commands in; the plan's check is an Ollama tier pointed at
# `keep` and back.
# ===========================================================================

class TestCommandFormatOverride:
    def _patch_config(self, tier_value):
        def get(key, default=None):
            if key == 'local_backend':
                return 'none'
            if key.startswith('ai.'):
                return tier_value
            return default
        return patch('assistant.llm.config.get', side_effect=get)

    def _smart_llm(self, tier_value):
        with self._patch_config(tier_value):
            name, model = parse_llm_config('smart')
            return create_llm(name, model, command_format=command_format_override('smart'))

    def test_an_ollama_tier_pointed_at_keep_and_back(self):
        from assistant.commands.formats.KeepFormat import KeepFormat
        from assistant.commands.formats.QwenFormat import QwenFormat
        assert self._smart_llm({'llm': 'OllamaLLM'}).command_format() is QwenFormat
        assert self._smart_llm({'llm': 'OllamaLLM', 'command_format': 'keep'}).command_format() is KeepFormat
        assert self._smart_llm({'llm': 'OllamaLLM'}).command_format() is QwenFormat

    def test_the_override_names_any_of_the_formats_whatever_the_class(self):
        from assistant.commands.formats.CommandFormats import CommandFormats
        for name in ("keep", "qwen", "dsml", "invoke", " Qwen "):
            with self._patch_config({'llm': 'OllamaLLM', 'command_format': name}):
                assert command_format_override('smart') == name.strip().lower()
            assert create_llm("OllamaLLM", command_format=name).command_format() is CommandFormats.by_name(name)

    def test_no_key_leaves_the_class_its_choice(self):
        with self._patch_config({'llm': 'OllamaLLM', 'model': 'gemma4:26b'}):
            assert command_format_override('smart') is None
        with self._patch_config('OllamaLLM'):
            assert command_format_override('smart') is None
        with self._patch_config(None):
            assert command_format_override('smart') is None

    def test_the_environment_variable_names_the_format_for_every_tier_over_the_config(self, monkeypatch):
        monkeypatch.setenv("FLOXY_COMMAND_FORMAT", " DSML ")
        with self._patch_config({'llm': 'OllamaLLM', 'command_format': 'keep'}):
            assert command_format_override('smart') == 'dsml'
            assert 'FLOXY_COMMAND_FORMAT' in command_format_chooser('smart')
        with self._patch_config('ClaudeLLM'):
            assert command_format_override('fast') == 'dsml'

    def test_an_unknown_environment_name_is_logged_and_the_config_key_stands(self, monkeypatch):
        monkeypatch.setenv("FLOXY_COMMAND_FORMAT", "xml")
        with self._patch_config({'llm': 'OllamaLLM', 'command_format': 'keep'}), \
                patch('assistant.llm.error') as logged:
            assert command_format_override('smart') == 'keep'
            assert 'ai.smart' in command_format_chooser('smart')
            assert logged.call_count == 1 and 'FLOXY_COMMAND_FORMAT' in logged.call_args[0][0]

    def test_a_blank_environment_variable_counts_as_unset(self, monkeypatch):
        monkeypatch.setenv("FLOXY_COMMAND_FORMAT", "  ")
        with self._patch_config({'llm': 'OllamaLLM'}):
            assert command_format_override('smart') is None
            assert command_format_chooser('smart') is None

    def test_an_unknown_name_is_logged_and_ignored_by_the_reader_but_refused_by_the_factory(self):
        with self._patch_config({'llm': 'OllamaLLM', 'command_format': 'xml'}), \
                patch('assistant.llm.error') as logged:
            assert command_format_override('smart') is None
            assert logged.call_count == 1 and 'ai.smart' in logged.call_args[0][0]
        self._inject("_FakeLLMNoCache", _FakeLLMNoCache)
        try:
            with pytest.raises(ValueError):
                create_llm("_FakeLLMNoCache", command_format="xml")
        finally:
            del sys.modules["assistant.llm._FakeLLMNoCache"]

    def _inject(self, class_name, cls):
        module = types.ModuleType(f"assistant.llm.{class_name}")
        setattr(module, class_name, cls)
        sys.modules[f"assistant.llm.{class_name}"] = module
        return module


# ===========================================================================
# speed — `speed: fast` in a tier's block is Anthropic's fast mode, at twice
# the price: nothing runs fast unless a block asks for it, and a tier that
# cannot (another model, another backend) runs as it did instead of failing.
# ===========================================================================

class TestTierSpeed:
    def _tier_llm(self, tier_value):
        def get(key, default=None):
            if key == 'local_backend':
                return 'none'
            if key.startswith('ai.'):
                return tier_value
            return default
        with patch('assistant.llm.config.get', side_effect=get):
            name, model = parse_llm_config('smart')
            return create_llm(name, model, speed=speed_override('smart'))

    def test_a_tier_without_the_key_is_not_fast(self):
        assert self._tier_llm({'llm': 'ClaudeLLM', 'model': 'claude-opus-5-5'}).speed is None
        assert self._tier_llm('ClaudeLLM').speed is None
        assert self._tier_llm(None).speed is None

    def test_speed_fast_reaches_a_model_that_has_fast_mode(self):
        for model in ('claude-opus-5-5', 'claude-fable-5-2'):
            for value in ('fast', ' Fast '):
                assert self._tier_llm({'llm': 'ClaudeLLM', 'model': model, 'speed': value}).speed == 'fast'

    def test_standard_and_a_typo_are_the_ordinary_speed(self):
        for value in ('standard', 'turbo', True, 2):
            assert self._tier_llm({'llm': 'ClaudeLLM', 'model': 'claude-opus-5-5', 'speed': value}).speed is None

    def test_a_model_without_fast_mode_runs_at_the_ordinary_speed(self):
        for model in ('claude-sonnet-5-5', 'claude-haiku-5-5', 'claude-fable-5-1'):
            assert self._tier_llm({'llm': 'ClaudeLLM', 'model': model, 'speed': 'fast'}).speed is None

    def test_a_backend_without_fast_mode_is_built_all_the_same(self):
        llm = self._tier_llm({'llm': 'OllamaLLM', 'model': 'gemma4:26b', 'speed': 'fast'})
        assert llm.model == 'gemma4:26b'
