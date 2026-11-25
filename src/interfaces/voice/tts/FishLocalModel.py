"""Fish Audio's Drama 3 on this Mac through MLX: every fact about the model, its download and the one thread that runs it."""
import os
import queue
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, ClassVar, Iterator, Optional

import numpy as np

from assistant.integration.Integration import HealthWarning
from common import Platform
from common.Paths import CACHE_DIR
from common.logging import error, info, trace

GB = 10 ** 9


@dataclass(frozen=True)
class Build:
    """One quantisation of the model: the conversion's repo at the revision that was tested."""
    repo: str
    revision: str
    size: int            # bytes of the repo at that revision
    min_memory_gb: int   # the Mac's memory it is offered from


# Community conversions Fish Audio neither released nor endorsed, so an install loads the
# revision that was tested and never whatever the repo holds that day. 4-bit was measured on an
# M5 Ultra (peak 26 GB, 2x real time); the floors of 3-bit and 8-bit are the mlx-audio docs'.
BUILDS = {
    "3bit": Build("mlx-community/fish-audio-drama-3-3bit", "7120e8b11630aa6ae614224074457ef9bb4f48c0", 19_754_587_845, 32),
    "4bit": Build("mlx-community/fish-audio-drama-3-4bit", "2b0c8028998533972e77359cce56b8bf1c5d6d4a", 23_781_119_297, 48),
    "8bit": Build("mlx-community/fish-audio-drama-3-8bit", "fedc7cc92bab5a180709cbaae6208d10b390bb97", 42_177_609_321, 64),
}
DEFAULT_QUANT = "4bit"
DISK_MARGIN_GB = 8
# Outside the checkout, so a reinstall of Keep does not fetch the model again.
MODELS_DIR = Path.home() / ".cache" / "keep" / "fish_local"
VOICE_DIR = CACHE_DIR / "fish_local"
# What the install's own voice is made from when the config names no reference clip: read once
# without a reference, and that clip is the reference from then on.
VOICE_TEXT = ("Good afternoon. I keep an eye on the house, the calendar and the weather, and I tell you "
              "what matters when it matters. Ask me anything, at any time, and I will do my best to help.")
WARMUP_TEXT = "Ready when you are."
# The first streamed piece, shorter than the rest for a fast first sound. With 4 frames
# playback ran dry once (245 ms) while the local LLM generated beside it; 8 did not.
FIRST_PIECE_FRAMES = 8
PARTIAL_SUFFIX = ".incomplete"   # the hub's name for a file still arriving
PROGRESS_STEP = 5          # percent between two log lines of a download
POLL_SECONDS = 5

IDLE, DOWNLOADING, LOADING, READY, FAILED, STOPPED = "idle", "downloading", "loading", "ready", "failed", "stopped"


@dataclass(frozen=True)
class Settings:
    """`tts.fish_local` as the model needs it; a change of any field is another model run."""
    quant: str = DEFAULT_QUANT
    voice: str = ""         # a reference clip of the user's own, with what is said in it
    voice_text: str = ""


def _hub_download(build: Build, target: Path) -> None:
    """The one call that reaches Hugging Face (the unit conftest refuses it)."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import disable_progress_bars
    disable_progress_bars()
    snapshot_download(build.repo, revision=build.revision, local_dir=str(target))


def _mlx_load(path: Path):
    """The one call that loads the weights into memory (the unit conftest refuses it)."""
    from mlx_audio.tts.utils import load_model
    return load_model(path)


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    import soundfile
    soundfile.write(str(path), audio, sample_rate)


class FishLocalModel:
    """One loaded Drama 3 and the thread everything of it runs on.

    `ensure(settings)` hands out the process's one run, starting it when there is none for those
    settings: the thread fetches the build if it is not here, loads it, makes or reads the
    reference voice, and then speaks one text at a time for whoever calls `speak` — so two
    dialogs take turns, and every MLX call of the model is made from one thread.
    """

    _current: ClassVar[Optional["FishLocalModel"]] = None
    _guard: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, settings: Settings):
        self.settings = settings
        self.build = BUILDS[settings.quant]
        self.state = IDLE
        self.reason = ""
        self.sample_rate = 0
        self._model = None
        self._voice: tuple[str, str] = ("", "")
        self._jobs: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------- the one run --

    @classmethod
    def ensure(cls, settings: Settings) -> "FishLocalModel":
        """The run for `settings`: the current one, or a new one in place of a failed run or of
        one with other settings (which is stopped, its memory given back)."""
        with cls._guard:
            current = cls._current
            if current is not None and current.settings == settings and current.state not in (FAILED, STOPPED):
                return current
            if current is not None:
                info(f"Fish Local: replacing the {current.settings.quant} run ({current.state}) with a new one")
                current.stop()
            cls._current = cls(settings)
            cls._current.start()
            return cls._current

    @classmethod
    def current(cls, settings: Settings) -> Optional["FishLocalModel"]:
        """The run for `settings` if there is one — never starts one."""
        current = cls._current
        return current if current is not None and current.settings == settings else None

    @classmethod
    def reset(cls) -> None:
        with cls._guard:
            if cls._current is not None:
                cls._current.stop()
            cls._current = None

    # ------------------------------------------------------------- the machine --

    @staticmethod
    def unmet(settings: Settings) -> str:
        """The first thing this machine lacks for the build, in words; "" when it can carry it."""
        build = BUILDS.get(settings.quant)
        if build is None:
            return f"tts.fish_local.quant is {settings.quant!r}; it takes {', '.join(BUILDS)}"
        if not Platform.is_apple_silicon():
            return "the model runs on a Mac with Apple Silicon only"
        memory = Platform.memory_gb()
        if memory < build.min_memory_gb:
            return (f"this Mac has {memory or '?'} GB of memory, and the {settings.quant} model needs "
                    f"{build.min_memory_gb} GB")
        if settings.voice and not Path(settings.voice).expanduser().is_file():
            return f"tts.fish_local.voice names {settings.voice}, and there is no such file"
        if settings.voice and not settings.voice_text:
            return "tts.fish_local.voice needs tts.fish_local.voice_text: what is said in the clip"
        missing = build.size - on_disk(model_dir(settings.quant), partials=False)
        if not downloaded(settings.quant) and missing > 0:
            need, free = missing // GB + DISK_MARGIN_GB, free_gb()
            if free < need:
                return f"the disk has {free} GB free, and the download of the {settings.quant} model needs {need} GB"
        return ""

    # -------------------------------------------------------------- the thread --

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="FishLocalModel", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.state = STOPPED
        self._jobs.put(None)

    def _run(self) -> None:
        try:
            self.prepare()
        except Exception as e:
            self.state, self.reason = FAILED, f"{type(e).__name__}: {e}"
            error(f"Fish Local: the {self.settings.quant} model did not come up: {self.reason}")
            self._release()
            return
        while (job := self._jobs.get()) is not None:
            self._speak(*job)
        info(f"Fish Local: the {self.settings.quant} run ended, its memory is given back")
        self._release()

    def _release(self) -> None:
        """Drop the weights and answer whoever still waits for audio."""
        if self._model is not None:
            self._model = None
            try:
                import mlx.core as mx
                mx.clear_cache()
            except ImportError:
                pass
        while True:
            try:
                job = self._jobs.get_nowait()
            except queue.Empty:
                return
            if job is not None:
                job[1].put(None)

    def prepare(self) -> None:
        """Fetch, load, voice, warm-up — READY after it, or it raises with the reason."""
        target = model_dir(self.settings.quant)
        if self.state == STOPPED:
            return
        if downloaded(self.settings.quant):
            trace(f"Fish Local: the {self.settings.quant} model is here already ({target})")
        else:
            self._download(target)
        if self.state == STOPPED:
            return
        self.state = LOADING
        started = time.time()
        info(f"Fish Local: loading {self.build.repo} from {target}...")
        self._model = _mlx_load(target)
        self.sample_rate = self._model.sample_rate
        self._voice = self._reference()
        # Not optional: the first generation after a load is slow, and a reference voice costs
        # about 3.5 s the first time it is used — both are paid here, before a reply waits on them.
        for _ in self._generate(WARMUP_TEXT):
            pass
        if self.state == STOPPED:
            return
        self.state = READY
        info(f"Fish Local: the {self.settings.quant} model speaks, {time.time() - started:.1f}s after the load began "
             f"(voice: {self._voice[0]})")

    def _download(self, target: Path) -> None:
        self.state = DOWNLOADING
        info(f"Fish Local: downloading {self.build.repo}@{self.build.revision[:8]} "
             f"({self.build.size / GB:.0f} GB) to {target}")
        target.mkdir(parents=True, exist_ok=True)
        drop_partials(target)
        failure: list[BaseException] = []
        done = threading.Event()

        def fetch():
            try:
                _hub_download(self.build, target)
            except BaseException as e:
                failure.append(e)
            finally:
                done.set()

        threading.Thread(target=fetch, name="FishLocalDownload", daemon=True).start()
        logged = self.percent() // PROGRESS_STEP
        while not done.wait(POLL_SECONDS):
            if self.state == STOPPED:
                info("Fish Local: the run was stopped during its download; what arrived stays on disk")
                return
            step = self.percent() // PROGRESS_STEP
            if step > logged:
                logged = step
                info(f"Fish Local: download at {self.percent()} %")
        if failure:
            raise failure[0]
        if self.state == STOPPED:
            return
        marker(self.settings.quant).write_text(self.build.revision)
        info(f"Fish Local: download of {self.build.repo} complete")

    def percent(self) -> int:
        """How much of the build is on disk, the files still arriving among them."""
        return min(100, on_disk(model_dir(self.settings.quant)) * 100 // self.build.size)

    def _reference(self) -> tuple[str, str]:
        """(clip, what is said in it): the configured one, else this install's own — made here
        the first time, by reading VOICE_TEXT without a reference, and the same from then on."""
        if self.settings.voice:
            return str(Path(self.settings.voice).expanduser()), self.settings.voice_text
        clip, transcript = own_voice()
        if clip.is_file() and transcript.is_file():
            trace(f"Fish Local: this install's own voice is {clip}")
            return str(clip), transcript.read_text().strip()
        info(f"Fish Local: no reference voice yet — making this install's own at {clip}")
        audio = np.concatenate([np.asarray(result.audio, dtype=np.float32).reshape(-1)
                                for result in self._model.generate(VOICE_TEXT)])
        clip.parent.mkdir(parents=True, exist_ok=True)
        _write_wav(clip, audio, self.sample_rate)
        transcript.write_text(VOICE_TEXT)
        return str(clip), VOICE_TEXT

    def _generate(self, text: str):
        return self._model.generate(text, ref_audio=self._voice[0], ref_text=self._voice[1],
                                    stream=True, first_chunk_frames=FIRST_PIECE_FRAMES)

    def _speak(self, text: str, out: queue.Queue, cancelled: Callable[[], bool]) -> None:
        pieces = None
        try:
            if cancelled():
                trace("Fish Local: the reply was dropped before its turn came")
                return
            started, first, samples = time.time(), None, 0
            pieces = self._generate(text)
            for result in pieces:
                if cancelled() or self.state == STOPPED:
                    trace(f"Fish Local: generation stopped after {samples / self.sample_rate:.1f}s of audio")
                    return
                audio = np.asarray(result.audio, dtype=np.float32).reshape(-1)
                first = first or time.time() - started
                samples += audio.size
                out.put(audio)
            trace(f"Fish Local: {text!r} → {samples / self.sample_rate:.1f}s of audio in {time.time() - started:.1f}s, "
                  f"first after {(first or 0) * 1000:.0f}ms")
        except Exception as e:
            error(f"Fish Local: synthesis failed for {text!r}: {type(e).__name__}: {e}")
        finally:
            if pieces is not None:
                pieces.close()
            out.put(None)

    # ----------------------------------------------------------- for the engine --

    def speak(self, text: str, cancelled: Callable[[], bool]) -> Iterator[np.ndarray]:
        """The audio of `text` at `sample_rate`, piece by piece as it is generated; after the
        texts queued before it. Nothing when the model is not ready."""
        if self.state != READY:
            trace(f"Fish Local: nothing spoken, the model is {self.state}")
            return
        out: queue.Queue = queue.Queue()
        self._jobs.put((text, out, cancelled))
        while (piece := out.get()) is not None:
            yield piece

    def status(self) -> str:
        if self.state == DOWNLOADING:
            return f"downloading the {self.settings.quant} model, {self.percent()} % of {self.build.size / GB:.0f} GB"
        if self.state == LOADING:
            return f"loading the {self.settings.quant} model"
        if self.state == FAILED:
            return self.reason
        return self.state

    def health(self) -> bool:
        """True once it speaks; a download or a load is a warning, a failure raises its reason."""
        if self.state == READY:
            return True
        if self.state == FAILED:
            raise RuntimeError(self.reason)
        raise HealthWarning(self.status())


def model_dir(quant: str) -> Path:
    return MODELS_DIR / BUILDS[quant].repo.split("/")[-1]


def marker(quant: str) -> Path:
    """Written after a complete download, with its revision: a start then asks the hub nothing,
    so the voice comes up without a network."""
    return model_dir(quant) / ".keep-revision"


def downloaded(quant: str) -> bool:
    try:
        return marker(quant).read_text().strip() == BUILDS[quant].revision
    except OSError:
        return False


def own_voice() -> tuple[Path, Path]:
    return VOICE_DIR / "voice.wav", VOICE_DIR / "voice.txt"


def drop_partials(target: Path) -> None:
    """Delete what an interrupted download left of the files it was in the middle of: the hub
    resumes file by file — it starts those again under other names and never removes these
    (6.8 GB stayed behind a download killed at 39 %), and they would count as progress."""
    for partial in target.rglob("*" + PARTIAL_SUFFIX):
        try:
            size = partial.stat().st_size
            partial.unlink()
            info(f"Fish Local: dropped {partial.name[:12]}… ({size / GB:.1f} GB) of an interrupted download")
        except OSError as e:
            trace(f"Fish Local: could not drop {partial}: {e}")


def on_disk(target: Path, partials: bool = True) -> int:
    """The bytes under `target`; without `partials`, only the files that arrived whole."""
    total = 0
    for folder, _, names in os.walk(target):
        for name in names:
            if not partials and name.endswith(PARTIAL_SUFFIX):
                continue
            try:
                total += os.lstat(os.path.join(folder, name)).st_size
            except OSError:
                pass        # a part file renamed between the listing and the reading
    return total


def free_gb() -> int:
    folder = MODELS_DIR
    while not folder.exists():
        folder = folder.parent
    return shutil.disk_usage(folder).free // GB
