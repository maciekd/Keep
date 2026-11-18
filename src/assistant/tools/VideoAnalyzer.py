import json
import math
import threading
from typing import Optional

from common.logging import info, error, trace
from common.ConfigManager import config
from common.Mp4Metadata import mp4_duration
from .ImageAnalyzer import parse_description_json

DESCRIPTION_PROMPT_SYSTEM = """You are a video analyst. Watch the provided video, including its audio track, and return a JSON object with these keys:
- "short_description": A concise 1-sentence description (max 50 chars)
- "detailed_description": A rich description of what happens over the course of the video: subjects, setting, actions, mood, and notable details. Include any visible text, landmarks, or activities.
- "spoken_content": What is said or sung in the video, quoted or closely summarized. An empty string if nobody speaks or sings.

{context}

Return ONLY the JSON object, no other text."""

DESCRIPTION_PROMPT_USER = "Analyze this video and return the JSON object as described."

# Clips above `ai.video.max_size_mb` are skipped rather than shrunk. The model's cost is set by
# duration, not bytes, so this only bounds upload time (a 1080p phone original is ~1 MB/s, so
# the default is roughly three minutes). No probing, no transcoding.
DEFAULT_MAX_SIZE_MB = 200
DEFAULT_MIME_TYPE = "video/mp4"

# Gemini samples video at 1 frame/s unless told otherwise, so a sub-second time-lapse, slo-mo or
# GIF yields zero frames and the request is rejected with INVALID_ARGUMENT. Clips shorter than
# this many seconds get an fps override that samples at least this many frames. Gemini caps fps at 24.
SHORT_CLIP_MIN_FRAMES = 2
MAX_VIDEO_FPS = 24


def short_clip_fps(duration: float | None) -> float | None:
    """The fps override for a clip, or None when Gemini's default 1 frame/s already yields
    SHORT_CLIP_MIN_FRAMES frames. A clip of unknown or zero duration gets the maximum."""
    if not duration or duration <= 0:
        return float(MAX_VIDEO_FPS)
    if duration >= SHORT_CLIP_MIN_FRAMES:
        return None
    return float(min(MAX_VIDEO_FPS, math.ceil(SHORT_CLIP_MIN_FRAMES / duration)))


class VideoKeyMissing(Exception):
    """The video tier's backend has no API key yet — a state of the install, not a failure."""


class VideoAnalyzer:
    """Describes video clips with the `ai.video` tier LLM.

    The raw bytes go to the model exactly as received; there is no re-encoding step. If the
    configured LLM has no video input (anything but Gemini today) clips are left undescribed —
    by decision, no frame-sampling workaround. They are left undescribed as well while the
    tier's backend has no API key: the tier defaults to Gemini, which an install running on
    Claude alone never gave a key, and the key is asked for again at every clip, so setting it
    needs no restart. Use `video_analyzer()` for the process-wide instance.
    """

    def __init__(self, llm=None):
        self._llm = llm
        self._llm_lock = threading.Lock()

    @staticmethod
    def configured_max_size() -> int:
        return int(config.get('ai.video.max_size_mb', DEFAULT_MAX_SIZE_MB)) * 1024 * 1024

    @classmethod
    def max_video_size(cls, llm) -> int:
        """The configured byte cap, never above what the LLM itself can take."""
        configured = cls.configured_max_size()
        return min(configured, llm.max_video_bytes() or configured)

    def max_size(self) -> int:
        """The byte cap for the configured video LLM; integrations check it before downloading.
        Never raises: a video tier that cannot be built is describe_bytes()' problem, not the
        caller's (the Telegram feed would otherwise drop the whole message, text included)."""
        try:
            return self.max_video_size(self.llm)
        except VideoKeyMissing as e:
            trace(f"VideoAnalyzer: {e}, using the configured size cap")
            return self.configured_max_size()
        except Exception as e:
            error(f"VideoAnalyzer: could not initialize the video LLM for its size cap ({e}), using the configured cap")
            return self.configured_max_size()

    @property
    def llm(self):
        if self._llm is None:
            with self._llm_lock:
                if self._llm is None:
                    from ..llm import create_llm, parse_llm_config
                    name, model = parse_llm_config('video')
                    llm = create_llm(name, model)
                    missing = llm.missing_api_key()
                    if missing:
                        raise VideoKeyMissing(f"the video tier's {llm.name()} has no {missing}")
                    llm.initialize()
                    self._llm = llm
                    trace(f"VideoAnalyzer: initialized {name} ({model}) for video analysis")
        return self._llm

    def describe_bytes(self, raw_bytes: bytes, mime_type: Optional[str] = None, context: str = "",
                       fallback: str = "[Video]") -> str:
        """Full pipeline: video-capable LLM check → size cap → LLM → formatted description.
        Returns *fallback* on any skip or failure, so callers never handle the steps themselves."""
        try:
            if not self.llm.supports_video():
                info(f"VideoAnalyzer: {self.llm.name()} cannot take video input, leaving the video undescribed")
                return fallback

            size = len(raw_bytes)
            max_size = self.max_video_size(self.llm)
            if size > max_size:
                info(f"VideoAnalyzer: video too large ({size / 1024 / 1024:.1f} MB > "
                     f"{max_size / 1024 / 1024:.0f} MB), skipping analysis")
                return fallback

            mime_type = mime_type or DEFAULT_MIME_TYPE
            info(f"VideoAnalyzer: analyzing video ({size / 1024 / 1024:.1f} MB, {mime_type})"
                 f"{f' — {context}' if context else ''}")
            result = self.analyze(raw_bytes, mime_type, context=context)
            if not result:
                info("VideoAnalyzer: video LLM returned no description")
                return fallback

            info(f"VideoAnalyzer: analyzed video — {result['short_description']}")
            return self.format_description(result)
        except VideoKeyMissing as e:
            info(f"VideoAnalyzer: {e}, leaving the video undescribed")
            return fallback
        except Exception as e:
            error(f"VideoAnalyzer: video analysis failed: {e}")
            return fallback

    def analyze(self, video_bytes: bytes, mime_type: str, context: str = "") -> Optional[dict]:
        context_block = f"# Additional context\n{context}" if context else ""
        system = DESCRIPTION_PROMPT_SYSTEM.format(context=context_block)
        fps = self._fps_override(video_bytes)

        for attempt in range(3):
            try:
                response = self.llm.think_with_video(
                    system_message=system,
                    user_message=DESCRIPTION_PROMPT_USER,
                    video_bytes=video_bytes,
                    mime_type=mime_type,
                    fps=fps,
                )
                return self._parse_response(response)
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                if attempt < 2:
                    trace(f"VideoAnalyzer: LLM response parse failed, retrying: {e}")
                    continue
                error(f"VideoAnalyzer: analysis failed after retry: {e}")
                return None
            except Exception as e:
                error(f"VideoAnalyzer: LLM call failed: {e}")
                return None
        return None

    @staticmethod
    def _fps_override(video_bytes: bytes) -> Optional[float]:
        """The sampling override for a short clip, read from its MP4 header. Clips without a
        readable duration (other containers, fragmented files) keep Gemini's default rather
        than risk sampling a long clip at 24 fps."""
        duration = mp4_duration(video_bytes)
        if duration is None:
            trace("VideoAnalyzer: no duration in the clip header, leaving frame sampling to the model")
            return None
        fps = short_clip_fps(duration)
        if fps:
            info(f"VideoAnalyzer: {duration:.1f}s clip is too short for Gemini's 1 frame/s default, sampling at {fps:.0f} fps")
        return fps

    @staticmethod
    def format_description(result: dict) -> str:
        text = f"Video description: [{result['short_description']}] {result['detailed_description']}"
        spoken = (result.get('spoken_content') or "").strip()
        if spoken:
            text += f" Spoken: {spoken}"
        return text

    @staticmethod
    def _parse_response(response: str) -> dict:
        return parse_description_json(response, ("short_description", "detailed_description"), "VideoAnalyzer")


_default = VideoAnalyzer()


def video_analyzer() -> VideoAnalyzer:
    """The process-wide analyzer; all integrations share its lazily created video LLM."""
    return _default
