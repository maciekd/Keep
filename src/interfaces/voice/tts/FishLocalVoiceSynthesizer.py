import re
import threading
import time
from pathlib import Path
from typing import Iterator, List, Optional

from . import FishLocalModel as fish
from .ClauseBatcher import speakable_units, to_pcm16_24k
from .FishLocalModel import FishLocalModel, Settings
from .VoiceSynthesizer import VoiceSynthesizer
from assistant.integration.Integration import Integration, IntegrationState
from common.ConfigManager import config
from common.logging import error, info, trace

PAIRED_TAG = re.compile(r"<(/?)([A-Za-z][\w-]*)>")
EMPTY_PAIR = re.compile(r"<([A-Za-z][\w-]*)>\s*</\1>")
MODEL_TOKEN = re.compile(r"<\|[^|>]*\|>")    # the model's own structure; it refuses a text holding one


class TagBalance:
    """Keeps the delivery tags whole in each unit of a text that was cut into several.

    A `<whisper>` still open where a unit ends is closed there and opened again in front of
    the next, so both halves are whispered; a closing tag with nothing open and a `[cue]` cut
    in two are dropped — the model would read a half tag aloud."""

    def __init__(self):
        self.open: list[str] = []
        self.in_cue = False

    def balance(self, text: str) -> str:
        text = MODEL_TOKEN.sub("", text)
        if self.in_cue:
            end = text.find("]")
            if end == -1:
                return ""
            text, self.in_cue = text[end + 1:], False
        start = text.rfind("[")
        if start != -1 and "]" not in text[start:]:
            text, self.in_cue = text[:start], True

        inherited = list(self.open)

        def keep(match: re.Match) -> str:
            closing, name = match.group(1), match.group(2)
            if not closing:
                self.open.append(name)
            elif self.open and self.open[-1] == name:
                self.open.pop()
            else:
                return ""
            return match.group(0)

        text = ("".join(f"<{name}>" for name in inherited) + PAIRED_TAG.sub(keep, text)
                + "".join(f"</{name}>" for name in reversed(self.open)))
        # A unit that only closes what the last one left open would start with an empty pair.
        while (emptied := EMPTY_PAIR.sub("", text)) != text:
            text = emptied
        return text


class FishLocalVoiceSynthesizer(VoiceSynthesizer, Integration):
    """A local, expressive voice from Fish Audio's Drama 3: no cloud, no key.

    Speaks Keep's replies with the open-weight Drama 3 model on this Mac's GPU, in a voice that
    sighs, laughs and whispers on cue. The model is downloaded the first time it is switched on
    (about 22 GB); its licence, the Fish Audio Community License, is the user's to meet.

    The model, its download and the thread it runs on are FishLocalModel; this is the engine a
    request speaks through. Until the model is ready `ready()` is false and VoiceFactory speaks
    through the next engine; on a Mac that cannot carry the build nothing is started and the
    monitor lists it not configured, with the reason. The text is cut into clause and sentence
    units (ClauseBatcher), each generated in the install's reference voice and streamed as
    24 kHz int16 PCM."""

    CONFIG_SECTION = "fish_local"
    SUPPORTS_EMOTION_TAGS = True
    mac_only = True

    # The same wait as Supertonic's before the stream ends: the tail of the last unit is
    # otherwise sometimes cut off in playback.
    END_OF_STREAM_GRACE_S = 2.0

    @property
    def name(self) -> str:
        return "Fish Local TTS"

    @property
    def config_keys(self) -> List[str]:
        return []

    @staticmethod
    def settings() -> Settings:
        cfg = (config.get('tts', {}) or {}).get('fish_local') or {}
        return Settings(quant=str(cfg.get('quant') or fish.DEFAULT_QUANT),
                        voice=str(cfg.get('voice') or ""), voice_text=str(cfg.get('voice_text') or ""))

    def __init__(self):
        VoiceSynthesizer.__init__(self)
        Integration.__init__(self)
        self.synthesis_thread: Optional[threading.Thread] = None
        settings = self.settings()
        unmet = FishLocalModel.unmet(settings)
        if unmet:
            # Not a health state: no check can change it, restart_integration asks again.
            info(f"Fish Local TTS cannot run here: {unmet}")
            self.model: Optional[FishLocalModel] = None
            self.state, self._last_error = IntegrationState.NOT_CONFIGURED, unmet
            return
        self.model = FishLocalModel.ensure(settings)

    def _on_reset(self) -> None:
        """A restart loads the model afresh (a deleted voice clip is made anew); switched off, it stays unloaded."""
        FishLocalModel.reset()

    @classmethod
    def ready(cls) -> bool:
        run = FishLocalModel.current(cls.settings())
        return run is not None and run.state == fish.READY

    @classmethod
    def not_ready_reason(cls) -> str:
        run = FishLocalModel.current(cls.settings())
        return run.status() if run is not None else "not started"

    def check_health(self) -> bool:
        """A failed run is started again here, so a download that broke resumes at the next check."""
        settings = self.settings()
        unmet = FishLocalModel.unmet(settings)
        if unmet:
            raise RuntimeError(unmet)
        run = FishLocalModel.current(settings)
        failed = run.reason if run is not None and run.state == fish.FAILED else ""
        self.model = FishLocalModel.ensure(settings)
        if failed:
            raise RuntimeError(f"{failed} — trying again")
        return self.model.health()

    @staticmethod
    def voice_id() -> str:
        """The reference clip and when it was made, so the phrase cache follows a new voice."""
        voice = FishLocalVoiceSynthesizer.settings().voice
        clip = Path(voice).expanduser() if voice else fish.own_voice()[0]
        try:
            return f"{clip.stem}-{int(clip.stat().st_mtime)}"
        except OSError:
            return clip.stem

    def cache_variant(self) -> str:
        return self.settings().quant

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
        tags = TagBalance()
        perform = self.emotions_enabled()
        try:
            if self.model is None:
                raise RuntimeError(self.last_error or "no model")
            for unit, _ in speakable_units(self.text_queue, lambda: self.is_closed):
                # The base strips the [cues] while tts.emotions is off; the paired tags are this model's own.
                text = tags.balance(unit).strip() if perform else PAIRED_TAG.sub("", MODEL_TOKEN.sub("", unit)).strip()
                if not any(c.isalnum() for c in PAIRED_TAG.sub("", text)):
                    trace("Fish Local: nothing speakable in the unit, skipped")
                    continue
                for audio in self.model.speak(text, lambda: self.is_closed):
                    pcm = to_pcm16_24k(audio, self.model.sample_rate)
                    if pcm and not self.is_closed:      # the model ends some texts on an empty piece
                        self.audio_queue.put(pcm)
        except Exception as e:
            error(f"Fish Local synthesis error: {type(e).__name__}: {e}")
        finally:
            if not self.is_closed and self.END_OF_STREAM_GRACE_S > 0:
                time.sleep(self.END_OF_STREAM_GRACE_S)
            self.audio_queue.put(None)

    def _join_threads(self) -> None:
        if self.synthesis_thread and self.synthesis_thread.is_alive():
            self.synthesis_thread.join(timeout=2)
