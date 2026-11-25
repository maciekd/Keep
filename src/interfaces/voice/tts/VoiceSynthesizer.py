import queue
import re
from abc import ABC, abstractmethod
from typing import Iterator, Optional

from common import Language
from common.ConfigManager import config


class VoiceSynthesizer(ABC):
    """Turns a streamed LLM text into a stream of PCM audio chunks; the base of every TTS backend.

    Phonetic name handling comes from config (see name_replacements); CONFIG_SECTION is the
    engine's key under `tts`, where a per-engine pronunciation overrides the shared one.
    """

    CONFIG_SECTION: str = ""

    # Whether this synthesizer renders bracket emotion tags ([laughing], [sighing], ...)
    # as audio. When False, such tags are stripped from the text before synthesis so
    # they aren't spoken literally. Subclasses that support them override to True.
    # The voice toolkit's EMOTION TAGS block asks the LLM for them while `tts.emotions` is
    # on (off by default — then every engine strips them); ElevenLabs decides per model
    # (only the v3/v4 family performs them), so the flag is a property there.
    SUPPORTS_EMOTION_TAGS: bool = False

    def __init__(self):
        self.llm_text_stream: Optional[Iterator[str]] = None
        self.audio_stream: Optional[Iterator[bytes]] = None
        self.is_closed = False
        self.audio_queue = queue.Queue()
        self.text_queue = queue.Queue()
        # Per-request BCP-47 locale (e.g. 'sv-SE'); None falls back to global config.
        self.request_language: Optional[str] = None
        # Whether this request's reply may use the engine's whispering night voice: only a
        # reply played on the house speakers wakes anyone, a phone gets the normal voice.
        self.whisper_allowed = True

    def name_replacements(self) -> list[tuple[str, str]]:
        """(name, replacement) pairs, lowercase names. `tts.names` lists names that only get
        the comma-prefix stripping (", Sam" -> " Sam"); `tts.name_pronunciations` maps a name
        to how it is spelled for the engine, and `tts.<engine>.name_pronunciations` overrides
        that per engine (one engine wants a respelling, another IPA)."""
        tts = config.get('tts', {}) or {}
        pronunciations = {str(name).lower(): str(name).lower() for name in tts.get('names') or []}
        for source in (tts, tts.get(self.CONFIG_SECTION) or {}):
            for name, spoken in (source.get('name_pronunciations') or {}).items():
                pronunciations[str(name).lower()] = str(spoken)
        return list(pronunciations.items())

    @staticmethod
    def configured_voice(section: str, default: str, key: str = 'voice') -> str:
        """The voice id `tts.<section>.<key>` names, else the engine's stock voice."""
        return ((config.get('tts', {}) or {}).get(section) or {}).get(key) or default

    @staticmethod
    def emotions_enabled() -> bool:
        """`tts.emotions`: whether the voice performs emotion tags at all (off by default)."""
        return bool((config.get('tts', {}) or {}).get('emotions', False))

    @classmethod
    def ready(cls) -> bool:
        """Whether the engine can speak now. False for one that is still fetching or loading
        its model: VoiceFactory then goes on to the next engine without constructing this one."""
        return True

    @classmethod
    def not_ready_reason(cls) -> str:
        """What a not-ready engine is waiting for, for the log."""
        return ""

    def impl_name(self) -> str:
        """Human-readable name of the actual synthesizer doing the work.
        Decorators (e.g. caching) override to report the wrapped impl."""
        return type(self).__name__

    def cache_variant(self) -> str:
        """What besides engine and voice changes the sound of a phrase (a model), so the WAV
        cache keeps one recording per variant. Empty for an engine with only one."""
        return ""

    def set_request_language(self, lang: Optional[str]):
        """Set the locale for the current request. Synthesizers that emit a
        language to the TTS backend should read it via effective_language()
        so the spoken locale tracks the request, not just the global config."""
        self.request_language = lang

    def set_whisper_allowed(self, allowed: bool):
        """Allow or forbid the night voice for the current request (see whisper_allowed)."""
        self.whisper_allowed = allowed

    @abstractmethod
    def start(self, llm_text_stream: Iterator[str]) -> Iterator[bytes]:
        """Start voice synthesis with text chunks from the LLM.
        Returns an iterator yielding PCM audio bytes."""
        pass

    @staticmethod
    @abstractmethod
    def voice_id() -> str:
        """Return the voice ID to use for synthesis."""
        pass

    @staticmethod
    def configured_languages() -> list[str]:
        """See common.Language.configured_languages."""
        return Language.configured_languages()

    @staticmethod
    def configured_language() -> str:
        """Primary (default) BCP-47 language tag; synthesizers that accept a single
        locale use this directly."""
        return Language.configured_language()

    @staticmethod
    def configured_language_code() -> str:
        """Primary language subtag only (e.g. 'en' from 'en-US'). Synthesizers
        that want a bare ISO-639-1 code use this."""
        return Language.configured_language().split('-')[0]

    def effective_language(self) -> str:
        """Full BCP-47 locale for the current request: the per-request language
        when one was set (e.g. a Swedish request on an en-US server), otherwise
        the global config."""
        return self.request_language or self.configured_language()

    def _audio_generator(self) -> Iterator[bytes]:
        """Yield audio chunks from the queue until None sentinel."""
        while True:
            chunk = self.audio_queue.get()
            if chunk is None:
                break
            yield chunk
            self.audio_queue.task_done()

    @staticmethod
    def _clear_queue(q: queue.Queue):
        """Drain a queue, calling task_done for each item."""
        while not q.empty():
            try:
                q.get_nowait()
                q.task_done()
            except queue.Empty:
                break

    def _feed_text_to_queue(self, text_preprocessor=None):
        """Read from llm_text_stream, apply name replacements,
        put processed text into text_queue. Meant to run in a thread."""
        from common.logging import error, trace
        from common.RequestKilledException import RequestKilledException

        replacements = self.name_replacements()
        names = [name for name, _ in replacements]
        max_hold = max((len(n) for n in names), default=0) + 1

        def should_hold_back(suffix):
            s = suffix.lower()
            return any(n.startswith(s) or (", " + n).startswith(s) for n in names)

        def apply_replacements(text):
            for name, replacement in replacements:
                if replacement != name:
                    text = re.sub(r",\s*" + re.escape(name), " " + replacement, text, flags=re.IGNORECASE)
                    text = re.sub(re.escape(name), replacement, text, flags=re.IGNORECASE)
                else:
                    text = re.sub(r",\s*" + re.escape(name), " " + name, text, flags=re.IGNORECASE)
            return text

        strip_tags = not (self.emotions_enabled() and self.SUPPORTS_EMOTION_TAGS)

        def open_tag_hold(text):
            """Hold-back length for a trailing, not-yet-closed emotion tag (an
            unmatched '[' with no ']' after it), so we never emit a half tag."""
            if not strip_tags:
                return 0
            open_idx = text.rfind("[")
            return len(text) - open_idx if open_idx != -1 and "]" not in text[open_idx:] else 0

        def process(text):
            if strip_tags and "[" in text:
                stripped = re.sub(r"\s*\[[^\]]*\]\s*", " ", text)
                if stripped != text:
                    trace(f"Stripped emotion tag(s) for {type(self).__name__} (tts.emotions off or no tag support)")
                    text = stripped
            return apply_replacements(text)

        try:
            buffer = ""
            for text_chunk in self.llm_text_stream:
                if self.is_closed:
                    break
                if text_preprocessor:
                    text_chunk = text_preprocessor(text_chunk)
                buffer += text_chunk

                hold = 0
                for i in range(min(max_hold, len(buffer)), 0, -1):
                    if should_hold_back(buffer[-i:]):
                        hold = i
                        break
                hold = max(hold, open_tag_hold(buffer))

                safe_to_send = buffer[:len(buffer) - hold] if hold else buffer
                hold_back = buffer[len(buffer) - hold:] if hold else ""

                if safe_to_send:
                    self.text_queue.put(process(safe_to_send))
                buffer = hold_back

            if buffer:
                self.text_queue.put(process(buffer))
            self.text_queue.put(None)
        except (RequestKilledException, GeneratorExit):
            trace("LLM stream killed mid-synthesis (request superseded); stopping feed.")
            self.text_queue.put(None)
        except Exception as e:
            error(f"Error feeding LLM response: {type(e).__name__}: {e}")
            self.text_queue.put(None)

    def close(self):
        """Close and clean up. Subclasses override _cleanup() for specific teardown."""
        self.is_closed = True
        self._cleanup()
        self._clear_queue(self.text_queue)
        self.text_queue.put(None)
        self._clear_queue(self.audio_queue)
        self._join_threads()
        self.llm_text_stream = None
        self.audio_stream = None

    def _cleanup(self):
        """Hook for subclass-specific teardown (e.g. close WebSocket). Default: no-op."""
        pass

    @abstractmethod
    def _join_threads(self):
        """Join worker threads with timeout. Each subclass knows its own threads."""
        pass
