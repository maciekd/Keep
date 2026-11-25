import sys
import threading
import time
from typing import Iterator, Optional, ClassVar, List, Tuple

from .ClauseBatcher import speakable_units, to_pcm16_24k
from .VoiceSynthesizer import VoiceSynthesizer
from assistant.integration.Integration import Integration
from common.logging import error, info, trace
from common.ConfigManager import config
from common.Language import language_code


class SupertonicVoiceSynthesizer(VoiceSynthesizer, Integration):
    """A local voice from the Supertonic model on this CPU: no cloud, no key.

    Speaks Keep's replies with the Supertonic model on this machine's CPU: no network and no API
    key.

    Runs the Supertonic ONNX model (Supertone/supertonic). It is batch-only, so the LLM's text
    is cut into clause and sentence units (ClauseBatcher) and its 44.1 kHz float32 audio packed
    as the pipeline's 24 kHz int16 PCM."""

    CONFIG_SECTION = "supertonic"

    # The default voice of local AI (`local_backend`), where ElevenLabs is off: nothing spoken
    # leaves the machine. On cloud AI it is opt-in.
    @classmethod
    def default_enabled(cls) -> bool:
        from assistant.llm import local_only_mode
        return local_only_mode()

    SUPERTONIC_SAMPLE_RATE = 44100

    # Sleep before the None sentinel so the client/sounddevice can finish draining
    # its playback buffer. Without this, the tail of the last clause sometimes gets
    # clipped (intermittent — depends on timing). Mirrors the same workaround in
    # CartesiaSonicVoiceSynthesizer._on_message and ElevenLabsVoiceSynthesizer.
    END_OF_STREAM_GRACE_S = 2.0

    _shared_tts: ClassVar[Optional[object]] = None
    _shared_style: ClassVar[Optional[object]] = None
    _shared_voice_name: ClassVar[Optional[str]] = None
    _shared_warmed_up: ClassVar[bool] = False
    _load_lock: ClassVar[threading.Lock] = threading.Lock()
    _preload_started: ClassVar[bool] = False

    @property
    def name(self) -> str:
        return "Supertonic TTS"

    @property
    def config_keys(self) -> List[str]:
        return []  # local model, no credentials required

    def check_health(self) -> bool:
        try:
            import supertonic  # noqa: F401
            return True
        except Exception:
            return False

    def __init__(self):
        VoiceSynthesizer.__init__(self)
        Integration.__init__(self)
        cfg = config.get('tts.supertonic', {}) or {}
        self._voice_name: str = cfg.get('voice', 'F3')
        self._lang: str = self.configured_language_code()
        # Two-step quality: opener uses fewer diffusion steps for fast first-sound,
        # subsequent batched chunks use more steps for cleaner prosody (they have
        # the playback-time budget to afford it — RTF is ~0.3 even at 10 steps).
        self._first_chunk_steps: int = int(cfg.get('first_chunk_steps', 5))
        self._total_steps: int = int(cfg.get('total_steps', 10))
        self._speed: float = float(cfg.get('speed', 1.0))
        self.synthesis_thread: Optional[threading.Thread] = None
        self._maybe_start_background_preload()

    def _maybe_start_background_preload(self) -> None:
        """Kick off a one-shot background thread that loads + warms the ONNX
        model so the first user query doesn't pay the ~100MB download /
        cold-start cost. Runs at most once per process; skipped under pytest,
        where tests drive model loading explicitly."""
        if SupertonicVoiceSynthesizer._preload_started or "pytest" in sys.modules:
            trace("Supertonic background preload not started (already started or under pytest)")
            return
        SupertonicVoiceSynthesizer._preload_started = True
        self._spawn_preload_thread()

    def _spawn_preload_thread(self) -> None:
        """Spawn the named daemon thread that warms the model. Kept as its own
        method so tests can assert the spawn decision without patching the
        process-global threading.Thread (the synthesis path also spawns threads,
        so counting global Thread() calls is inherently racy)."""
        def _run():
            try:
                info("Pre-loading Supertonic TTS model in background...")
                self.preload()
            except Exception as exc:
                error(f"Supertonic preload failed: {exc}")

        threading.Thread(target=_run, name="SupertonicPreload", daemon=True).start()

    @staticmethod
    def voice_id() -> str:
        return (config.get('tts.supertonic', {}) or {}).get('voice', 'F3')

    def preload(self) -> None:
        """Force the ONNX model to load (download on first call). Safe to call
        from a background thread at server startup so the first user query doesn't
        pay the load cost."""
        self._get_tts()

    def _get_tts(self) -> Tuple[object, object]:
        """Return (tts, style), loading both lazily on first use. Shared across
        all instances via class-level cache. Runs a one-shot warmup synth the
        first time the model is loaded so the next real call doesn't pay the
        ONNX graph cold-start cost."""
        with SupertonicVoiceSynthesizer._load_lock:
            if (SupertonicVoiceSynthesizer._shared_tts is None
                    or SupertonicVoiceSynthesizer._shared_voice_name != self._voice_name):
                info(f"Loading Supertonic TTS model (voice={self._voice_name})...")
                t0 = time.time()
                from supertonic import TTS
                tts = TTS(auto_download=True)
                style = tts.get_voice_style(voice_name=self._voice_name)
                SupertonicVoiceSynthesizer._shared_tts = tts
                SupertonicVoiceSynthesizer._shared_style = style
                SupertonicVoiceSynthesizer._shared_voice_name = self._voice_name
                SupertonicVoiceSynthesizer._shared_warmed_up = False
                info(f"Supertonic TTS model ready in {time.time() - t0:.1f}s")
            if not SupertonicVoiceSynthesizer._shared_warmed_up:
                self._warmup_locked()
            return SupertonicVoiceSynthesizer._shared_tts, SupertonicVoiceSynthesizer._shared_style

    def _warmup_locked(self) -> None:
        """Run a tiny synth to compile/warm the ONNX inference graph. Must be
        called while holding _load_lock. Failures are non-fatal — we just skip
        the warmup."""
        try:
            info("Warming up Supertonic inference graph...")
            t0 = time.time()
            wake_word = config.get('wake_word', 'Michelle')
            SupertonicVoiceSynthesizer._shared_tts.synthesize(
                text=f"Hello and welcome to this automation system, I'm your trusty AI, {wake_word}. What going on with you?",
                lang=self._lang,
                voice_style=SupertonicVoiceSynthesizer._shared_style,
                total_steps=self._first_chunk_steps,
                speed=self._speed,
                silence_duration=0.0,
            )
            info(f"Supertonic warmup done in {(time.time() - t0) * 1000:.0f}ms")
        except Exception as e:
            error(f"Supertonic warmup failed (continuing anyway): {e}")
        finally:
            SupertonicVoiceSynthesizer._shared_warmed_up = True

    def start(self, llm_text_stream: Iterator[str]) -> Iterator[bytes]:
        self.llm_text_stream = llm_text_stream
        self.is_closed = False
        self._clear_queue(self.audio_queue)
        self._clear_queue(self.text_queue)

        threading.Thread(target=self._feed_text_to_queue, daemon=True).start()

        self.synthesis_thread = threading.Thread(target=self._run_synthesis, daemon=True)
        self.synthesis_thread.start()

        self.audio_stream = self._audio_generator()
        return self.audio_stream

    def _run_synthesis(self) -> None:
        """Drain text_queue, batch into clause-sized speakable units, synthesize,
        and push int16 PCM bytes onto audio_queue. Terminates with a None sentinel."""
        try:
            tts, style = self._get_tts()
        except Exception as e:
            error(f"Supertonic model load failed: {e}")
            self.audio_queue.put(None)
            return

        try:
            # The opener takes fewer diffusion steps for a fast first sound; the batched
            # units after it have the playback time to afford more.
            for text, is_first in speakable_units(self.text_queue, lambda: self.is_closed):
                self._synth_and_emit(text, tts, style, self._first_chunk_steps if is_first else self._total_steps)
        except Exception as e:
            error(f"Supertonic synthesis error: {e}")
        finally:
            # Brief grace so the client can finish playing the tail of the final
            # clause before the stream closes. Skipped if killed mid-flight.
            if not self.is_closed and self.END_OF_STREAM_GRACE_S > 0:
                time.sleep(self.END_OF_STREAM_GRACE_S)
            self.audio_queue.put(None)

    @staticmethod
    def _strip_unsupported_chars(text: str, tts) -> str:
        """Drop characters the model can't synthesize (emoji, ⏰, ...) instead of
        letting synthesize() reject the whole clause. Asks the model's own charset
        validator so supported non-ASCII (åäö, ...) is preserved."""
        if not text:
            return text
        try:
            is_valid, unsupported = tts.model.text_processor.validate_text(text)
        except Exception as e:
            trace(f"Supertonic charset validation unavailable ({e}); passing text through")
            return text
        if is_valid:
            return text
        bad = set(unsupported)
        cleaned = "".join(c for c in text if c not in bad).strip()
        info(f"Supertonic: stripped unsupported character(s) {unsupported} from {text!r}")
        return cleaned

    def _synthesis_language(self) -> str:
        """The request's language when Supertonic speaks it (a Swedish request on an en-US
        install), else the configured one — the model reads a text in the language it is told."""
        if not self.request_language:
            return self._lang
        code = language_code(self.request_language)
        try:
            from supertonic.config import SUPPORTED_LANGUAGES
        except ImportError:
            return code
        if code in SUPPORTED_LANGUAGES:
            return code
        trace(f"Supertonic does not speak request language {self.request_language!r} — using {self._lang!r}")
        return self._lang

    def _synth_and_emit(self, text: str, tts, style, total_steps: int) -> None:
        text = self._strip_unsupported_chars(text.strip(), tts)
        if not text or not any(c.isalnum() for c in text):
            trace("Supertonic: nothing speakable left in segment, skipping")
            return
        try:
            t0 = time.time()
            wav, duration = tts.synthesize(
                text=text,
                lang=self._synthesis_language(),
                voice_style=style,
                total_steps=total_steps,
                speed=self._speed,
                silence_duration=0.0,
            )
            pcm = to_pcm16_24k(wav, self.SUPERTONIC_SAMPLE_RATE)
        except Exception as e:
            error(f"Supertonic synth failed for {text!r}: {e}")
            return

        if pcm and not self.is_closed:
            self.audio_queue.put(pcm)

        # duration is a numpy scalar in some supertonic versions; coerce before formatting.
        try:
            dur_s = float(duration)
        except (TypeError, ValueError):
            dur_s = 0.0
        trace(f"Supertonic synth ({total_steps} steps): {text!r} ({len(text)} chars) → "
              f"{dur_s:.2f}s audio in {(time.time() - t0) * 1000:.0f}ms ({len(pcm)} bytes PCM)")

    def _join_threads(self) -> None:
        if self.synthesis_thread and self.synthesis_thread.is_alive():
            self.synthesis_thread.join(timeout=2)
