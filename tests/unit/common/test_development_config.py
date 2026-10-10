"""Defaults that we rely on in production.

These flags live in `DevelopmentConfig.py` and are read once at import time.
Both have caused real incidents — the prompt-prepend flag silently destroyed
prompt caching, and the cache TTL controls the per-request bill on every
voice query — so the defaults are pinned here against accidental flips.
"""

import importlib.util

import pytest


def _reimport_development_config():
    """A fresh copy of DevelopmentConfig evaluated under the env just set, loaded beside the
    imported one and never in its place: swapping it into sys.modules left every module that
    had imported it (CodeQualityMaintainer, Telemetry) holding the old object, where a later
    test's patch on the module name could not reach — and the copy carried this test's env."""
    spec = importlib.util.find_spec("common.DevelopmentConfig")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def isolated_env(monkeypatch):
    """Strip both env vars so the default branch is what we read back. The
    autouse fixture in tests/unit/conftest.py doesn't touch these, so without
    this fixture a shell that exported FLOXY_PREPEND_USER_QUERY would mask the
    test."""
    monkeypatch.delenv("FLOXY_PREPEND_USER_QUERY", raising=False)
    monkeypatch.delenv("FLOXY_CLAUDE_STATIC_CACHE_TTL", raising=False)
    return monkeypatch


@pytest.mark.unit
def test_prepend_user_query_defaults_to_no_override(isolated_env):
    """Unset env => None => defer to each LLM's own prefers_prepended_user_query()
    default (Claude-on-Fable/Ollama off so caching survives, the rest on for accuracy).
    Pins b351f5b5: the override must NOT default to forcing prepend on, which gave
    0% cache reads (every request a write at byte 0 of the cached block)."""
    cfg = _reimport_development_config()
    assert cfg.PREPEND_USER_QUERY_OVERRIDE is None


@pytest.mark.unit
def test_prepend_user_query_force_on_via_env(isolated_env):
    """The escape hatch documented in the source comment must still work —
    the prompt test runner relies on toggling this at runtime to A/B accuracy."""
    isolated_env.setenv("FLOXY_PREPEND_USER_QUERY", "1")
    cfg = _reimport_development_config()
    assert cfg.PREPEND_USER_QUERY_OVERRIDE is True


@pytest.mark.unit
def test_prepend_user_query_force_off_via_env(isolated_env):
    """=0 forces prepend OFF for every LLM regardless of per-LLM defaults — the
    cheap, cache-preserving run mode the prompt test runner exports."""
    isolated_env.setenv("FLOXY_PREPEND_USER_QUERY", "0")
    cfg = _reimport_development_config()
    assert cfg.PREPEND_USER_QUERY_OVERRIDE is False


@pytest.mark.unit
def test_claude_static_cache_ttl_env_override(isolated_env):
    """Test/prompt runner overrides this to 5m to save money on dense, short
    requests — preserve the env-driven override path."""
    isolated_env.setenv("FLOXY_CLAUDE_STATIC_CACHE_TTL", "5m")
    cfg = _reimport_development_config()
    assert cfg.CLAUDE_STATIC_CACHE_TTL == "5m"
