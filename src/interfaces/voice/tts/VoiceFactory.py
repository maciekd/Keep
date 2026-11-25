from common.ConfigManager import config
from common.Language import language_code
from common.logging import info, trace
from .CachedVoiceSynthesizer import CachedVoiceSynthesizer
from .CartesiaSonicVoiceSynthesizer import CartesiaSonicVoiceSynthesizer
from .Chirp3VoiceSynthesizer import Chirp3VoiceSynthesizer
from .ElevenLabsVoiceSynthesizer import ElevenLabsVoiceSynthesizer
from .FishAudioVoiceSynthesizer import FishAudioVoiceSynthesizer
from .FishLocalVoiceSynthesizer import FishLocalVoiceSynthesizer
from .GeminiVoiceSynthesizer import GeminiVoiceSynthesizer
from .GrokVoiceSynthesizer import GrokVoiceSynthesizer
from .SupertonicVoiceSynthesizer import SupertonicVoiceSynthesizer
from .VoiceSynthesizer import VoiceSynthesizer

# Every TTS engine, in the order a request tries them: the first one that is enabled in
# `enabled_integrations` and has its config keys speaks. ElevenLabs is on by default, so a
# voice install with only its key speaks through it; the others are opt-in alternatives. Of the
# two local ones Fish Local stands first: while its model is not ready, Supertonic speaks.
ENGINES: list[type] = [
    ElevenLabsVoiceSynthesizer,
    GeminiVoiceSynthesizer,
    Chirp3VoiceSynthesizer,
    CartesiaSonicVoiceSynthesizer,
    FishAudioVoiceSynthesizer,
    GrokVoiceSynthesizer,
    FishLocalVoiceSynthesizer,
    SupertonicVoiceSynthesizer,
]

# A language that prefers one engine over the default order: Swedish was tuned on Gemini.
PREFERRED: dict[str, type] = {'sv': GeminiVoiceSynthesizer}


def engine_order(lang: str | None) -> list[type]:
    """The engines to try for a request language, the language's preferred one first."""
    preferred = PREFERRED.get(language_code(lang))
    return ([preferred] if preferred else []) + [cls for cls in ENGINES if cls is not preferred]


def create_voice(lang: str | None) -> VoiceSynthesizer | None:
    """Build the TTS synthesizer for a request language: the first engine in engine_order
    that is enabled, ready and configured, behind the on-disk CachedVoiceSynthesizer. None when
    no engine can speak — the request then runs text-only, which DialogManager handles.

    Every call returns a fresh instance. The backends are cheap to construct (they connect
    lazily on the first start()), and callers own the lifetime: DialogManager keeps one per
    language for as long as it lives (so the ElevenLabs websocket stays warm across the
    turns of one dialog) and closes them on kill, exactly as it did when it built the pair
    itself. Sharing an instance across dialogs would let one request's kill close the
    websocket under the next one's synthesis."""
    code = language_code(lang)
    skipped = []
    for cls in engine_order(lang):
        name = cls.config_name()
        if not cls.enabled(config):
            skipped.append(f"{name} disabled")
            continue
        if not cls.ready():
            skipped.append(f"{name} not ready ({cls.not_ready_reason()})")
            continue
        impl = cls()
        missing = impl.missing_config_keys()
        if missing:
            skipped.append(f"{name} missing {', '.join(missing)}")
            continue
        trace(f"Voice for lang={lang} ({code}): {type(impl).__name__}"
             + (f" (skipped: {'; '.join(skipped)})" if skipped else ""))
        return CachedVoiceSynthesizer(impl)
    info(f"No TTS engine can speak lang={lang} ({code}), the reply is text-only: {'; '.join(skipped)}")
    return None
