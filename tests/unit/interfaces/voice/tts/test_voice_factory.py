"""interfaces.voice.tts.VoiceFactory — which TTS engine serves which request language.

The factory speaks through the first engine in `engine_order` that is enabled in
`enabled_integrations` and has its config keys (ElevenLabs is on by default, Swedish prefers
Gemini); with none it returns None and the request runs text-only. The conftest stubs
`Integration.enabled` to True for every class, so the tests here decide per engine."""
import pytest

from assistant.integration.Integration import Integration
from interfaces.voice.tts.CachedVoiceSynthesizer import CachedVoiceSynthesizer
from interfaces.voice.tts.CartesiaSonicVoiceSynthesizer import CartesiaSonicVoiceSynthesizer
from interfaces.voice.tts.ElevenLabsVoiceSynthesizer import ElevenLabsVoiceSynthesizer
from interfaces.voice.tts.GeminiVoiceSynthesizer import GeminiVoiceSynthesizer
from interfaces.voice.tts.SupertonicVoiceSynthesizer import SupertonicVoiceSynthesizer
from interfaces.voice.tts.VoiceFactory import ENGINES, create_voice, engine_order


@pytest.fixture
def engines(monkeypatch):
    """Every engine enabled and configured unless a test says otherwise:
    `engines.disable(cls)` and `engines.unconfigure(cls, key)` narrow the field."""
    disabled, unconfigured = set(), {}
    monkeypatch.setattr(Integration, "enabled", classmethod(lambda cls, config: cls not in disabled))
    monkeypatch.setattr(Integration, "missing_config_keys", lambda self: unconfigured.get(type(self), []))

    class Control:
        @staticmethod
        def disable(cls):
            disabled.add(cls)

        @staticmethod
        def unconfigure(cls, key):
            unconfigured[cls] = [key]
    return Control()


@pytest.mark.unit
@pytest.mark.parametrize("time_of_day, expected_voice_id", [
    ("DAY", "QtY3JBOUKEB5xzrRfOKc"),    # Maisie, normal speech
    ("NIGHT", "geVuXdUpU0lgUukWJCUE"),  # Shannon, whispering
])
def test_english_voice_id_selected_by_time_of_day(monkeypatch, engines, time_of_day, expected_voice_id):
    """English speaks through ElevenLabs; time of day selects its voice: Maisie (normal) in
    the daytime, a whisper at night. Pins both the engine and the day/night voice ids."""
    from assistant.prompt.TimeOfDay import TimeOfDay
    monkeypatch.setattr('interfaces.voice.tts.ElevenLabsVoiceSynthesizer.TimeOfDay.get_current',
                        staticmethod(lambda: TimeOfDay[time_of_day]))

    voice = create_voice("en-US")
    assert isinstance(voice, CachedVoiceSynthesizer)
    assert isinstance(voice.impl, ElevenLabsVoiceSynthesizer)
    assert voice.impl.voice_id() == expected_voice_id


@pytest.mark.unit
@pytest.mark.parametrize("lang, expected_impl", [
    ("sv-SE", GeminiVoiceSynthesizer),
    ("sv", GeminiVoiceSynthesizer),
    ("en-US", ElevenLabsVoiceSynthesizer),
    ("en", ElevenLabsVoiceSynthesizer),
    (None, GeminiVoiceSynthesizer),  # falls back to configured language (sv-SE)
])
def test_backend_routes_by_lang(monkeypatch, engines, lang, expected_impl):
    """With every engine available Swedish prefers Gemini and everything else takes the
    default order, ElevenLabs first. Missing lang falls back to the configured language."""
    monkeypatch.setattr('common.Language.configured_language', lambda: 'sv-SE')
    voice = create_voice(lang)
    assert isinstance(voice, CachedVoiceSynthesizer)
    assert isinstance(voice.impl, expected_impl)


@pytest.mark.unit
def test_engine_order_puts_the_preferred_engine_first_and_keeps_the_rest():
    assert engine_order("sv-SE")[0] is GeminiVoiceSynthesizer
    assert engine_order("en-US")[0] is ElevenLabsVoiceSynthesizer
    assert set(engine_order("sv")) == set(ENGINES) and len(engine_order("sv")) == len(ENGINES)


@pytest.mark.unit
def test_disabled_engine_is_skipped_for_the_next_enabled_one(engines):
    """disable_integration("ElevenLabs TTS") with Cartesia enabled: English speaks through Cartesia."""
    engines.disable(ElevenLabsVoiceSynthesizer)
    engines.disable(GeminiVoiceSynthesizer)
    for cls in ENGINES:
        if cls not in (ElevenLabsVoiceSynthesizer, GeminiVoiceSynthesizer, CartesiaSonicVoiceSynthesizer):
            engines.disable(cls)
    assert isinstance(create_voice("en").impl, CartesiaSonicVoiceSynthesizer)


@pytest.mark.unit
def test_engine_without_its_keys_is_skipped(engines):
    """Swedish with Gemini enabled but google_service_account unset falls back to ElevenLabs."""
    engines.unconfigure(GeminiVoiceSynthesizer, "google_service_account")
    assert isinstance(create_voice("sv-SE").impl, ElevenLabsVoiceSynthesizer)


@pytest.mark.unit
def test_no_engine_means_text_only(engines):
    """ElevenLabs on by default but not configured, nothing else enabled: None, and the
    request runs text-only instead of failing at the first synthesis."""
    engines.unconfigure(ElevenLabsVoiceSynthesizer, "api_keys.elevenlabs")
    for cls in ENGINES:
        if cls is not ElevenLabsVoiceSynthesizer:
            engines.disable(cls)
    assert create_voice("en") is None


@pytest.mark.unit
def test_disabled_engine_is_never_constructed(engines, monkeypatch):
    """A disabled engine is skipped on the class alone — Supertonic's constructor would
    start the ONNX preload."""
    monkeypatch.setattr(SupertonicVoiceSynthesizer, "__init__",
                        lambda self: (_ for _ in ()).throw(AssertionError("constructed a disabled engine")))
    engines.disable(SupertonicVoiceSynthesizer)
    assert isinstance(create_voice("en").impl, ElevenLabsVoiceSynthesizer)


@pytest.mark.unit
def test_every_call_builds_a_fresh_instance(engines):
    """Callers own the lifetime (DialogManager memoizes per language and closes on kill);
    a shared instance would let one dialog's kill close the websocket under another."""
    assert create_voice("en") is not create_voice("en")


@pytest.mark.unit
@pytest.mark.parametrize("local, default_engine, other", [
    (False, ElevenLabsVoiceSynthesizer, SupertonicVoiceSynthesizer),
    (True, SupertonicVoiceSynthesizer, ElevenLabsVoiceSynthesizer),
])
def test_the_default_engine_follows_cloud_or_local_ai(monkeypatch, local, default_engine, other):
    """Unlisted in enabled_integrations, ElevenLabs speaks on cloud AI and Supertonic on local AI,
    so with `local_backend` set nothing spoken leaves the machine; a listed engine is on either way."""
    monkeypatch.setattr("assistant.llm.local_only_mode", lambda: local)

    class Listed:
        def __init__(self, names):
            self.names = names

        def integration_enabled(self, name, default=False):
            return name in self.names or default

    enabled = Integration.__dict__["enabled"].original.__func__
    unlisted = Listed(set())
    assert enabled(default_engine, unlisted)
    assert not enabled(other, unlisted)
    assert enabled(other, Listed({other.config_name()}))
