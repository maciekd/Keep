import queue
import sys
import threading
import time
from typing import Iterator, Optional, ClassVar, List, Tuple

import numpy as np

from .VoiceSynthesizer import VoiceSynthesizer
from assistant.integration.Integration import Integration
from common.logging import error, info, trace
from common.ConfigManager import config
from common.Language import language_code


class SupertonicVoiceSynthesizer(VoiceSynthesizer, Integration):
    """A local voice from the Supertonic model on this CPU: no cloud, no key.

    Speaks Keep's replies with the Supertonic model on this machine's CPU: no network and no API
    key.

    Runs the Supertonic ONNX model (Supertone/supertonic). It is batch-only, so LLM text chunks
    are buffered and flushed at clause boundaries to keep first-audio latency low while
    preserving prosodic context. Output is 24 kHz / mono / int16 PCM to match the rest of the
    pipeline (Supertonic natively emits 44.1 kHz float32, which is resampled and packed to
    int16)."""

    CONFIG_SECTION = "supertonic"

    # The default voice of local AI (`local_backend`), where ElevenLabs is off: nothing spoken
    # leaves the machine. On cloud AI it is opt-in.
    @classmethod
    def default_enabled(cls) -> bool:
        from assistant.llm import local_only_mode
        return local_only_mode()

    SUPERTONIC_SAMPLE_RATE = 44100
    OUTPUT_SAMPLE_RATE = 24000

    CLAUSE_PUNCT = set(",.?!:;—-")
    SENTENCE_PUNCT = set(".?!")
    MIN_WORDS_FLUSH = 20  # safety net only — primary trigger is clause/sentence punctuation

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

        buffer = ""
        chunks_emitted = 0
        try:
            while True:
                if self.is_closed:
                    break
                try:
                    chunk = self.text_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                if chunk is None:
                    if buffer.strip():
                        steps = self._first_chunk_steps if chunks_emitted == 0 else self._total_steps
                        self._synth_and_emit(buffer, tts, style, steps)
                    break

                if not chunk:
                    continue
                buffer += chunk

                # Mode flips after the first emit: first chunk prioritises latency
                # (split on earliest clause punct), subsequent chunks prioritise
                # prosody (batch up to the latest sentence-end so the model has
                # more context per call).
                while True:
                    is_first = chunks_emitted == 0
                    mode = 'first' if is_first else 'batch'
                    speakable, buffer = self._pop_speakable(buffer, mode)
                    if speakable is None:
                        break
                    steps = self._first_chunk_steps if is_first else self._total_steps
                    self._synth_and_emit(speakable, tts, style, steps)
                    chunks_emitted += 1
        except Exception as e:
            error(f"Supertonic synthesis error: {e}")
        finally:
            # Brief grace so the client can finish playing the tail of the final
            # clause before the stream closes. Skipped if killed mid-flight.
            if not self.is_closed and self.END_OF_STREAM_GRACE_S > 0:
                time.sleep(self.END_OF_STREAM_GRACE_S)
            self.audio_queue.put(None)

    @classmethod
    def _pop_speakable(cls, buffer: str, mode: str = 'first') -> Tuple[Optional[str], str]:
        """Try to extract one speakable unit from the front of `buffer`.

        mode='first'  — split at the earliest clause-punctuation boundary
                        (",.?!:;—-"). Optimises for first-sound latency: flush
                        as soon as any speakable fragment exists.
        mode='batch'  — split at the **latest** sentence-end (".?!") in the
                        buffer, batching multiple complete sentences into one
                        synth call. Optimises for prosody (the model gets full
                        sentence context) once we're already producing audio.

        Both modes fall back to `MIN_WORDS_FLUSH` as a safety net so a
        punctuation-free stream still flushes eventually. Returns
        (speakable_or_None, remainder)."""
        if mode == 'first':
            for i, ch in enumerate(buffer):
                if ch in cls.CLAUSE_PUNCT:
                    return buffer[:i + 1], buffer[i + 1:]
        else:  # 'batch' — find the rightmost sentence-end and flush through it
            cut = -1
            for i, ch in enumerate(buffer):
                if ch in cls.SENTENCE_PUNCT:
                    cut = i
            if cut >= 0:
                return buffer[:cut + 1], buffer[cut + 1:]

        words = buffer.split()
        if len(words) >= cls.MIN_WORDS_FLUSH:
            # Flush all complete words, keep the trailing partial word (if buffer
            # doesn't end on whitespace) so we never cut mid-word.
            if buffer and not buffer[-1].isspace():
                cut = buffer.rfind(' ')
                if cut > 0:
                    return buffer[:cut], buffer[cut:]
            return buffer, ""
        return None, buffer

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
            pcm = self._to_pcm16_24k(wav)
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

    @classmethod
    def _to_pcm16_24k(cls, wav) -> bytes:
        """Convert Supertonic's float32 @ 44.1 kHz → int16 mono @ 24 kHz raw PCM bytes."""
        arr = np.asarray(wav, dtype=np.float32)
        if arr.ndim > 1:
            # Supertonic returns shape (1, samples) — channels-first. Pick the longer
            # axis as the sample axis and mean over the shorter (channel) axis so this
            # works for either layout.
            channel_axis = 0 if arr.shape[0] <= arr.shape[1] else 1
            arr = arr.mean(axis=channel_axis)
        if arr.size == 0:
            return b""

        # Linear resample 44100 → 24000. Speech-grade quality; no scipy dep needed.
        src_n = arr.size
        dst_n = int(round(src_n * cls.OUTPUT_SAMPLE_RATE / cls.SUPERTONIC_SAMPLE_RATE))
        if dst_n <= 0:
            return b""
        xs = np.linspace(0.0, src_n - 1, dst_n, dtype=np.float64)
        resampled = np.interp(xs, np.arange(src_n, dtype=np.float64), arr).astype(np.float32)

        clipped = np.clip(resampled, -1.0, 1.0)
        pcm = (clipped * 32767.0).astype(np.int16)
        return pcm.tobytes()

    def _join_threads(self) -> None:
        if self.synthesis_thread and self.synthesis_thread.is_alive():
            self.synthesis_thread.join(timeout=2)
