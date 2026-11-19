"""Frame-driven utterance segmentation: the pre-roll buffer, in-speech flag, captured
frames, and trailing-silence bookkeeping that decide where an utterance starts and ends.

Extracted from LocalOutpost._endpoint_loop so the live outpost and the offline STT
harness (Tools/STTTest/endpoint.py) segment audio with the SAME code — the harness can
never drift from what production actually feeds Whisper. The VAD verdict stays outside
(callers score each frame and pass `loud`): the live loop needs the raw score for
logging and the energy gate, and the harness drives its own per-clip VAD instance.

How much trailing quiet ends an utterance depends on how long it has run (`patience`):
a short command ends on SILENCE, so its reply is as fast as the endpoint can be, and
somebody who has been speaking for a while may pause to think — up to SILENCE_LONG —
without the pause being taken for the end.
"""

import collections

import numpy as np

from .constants import (
    FRAME_DURATION, MAX_SEGMENT, PATIENCE_FROM, PATIENCE_FULL, PRE_ROLL, SILENCE, SILENCE_LONG,
)

STARTED = "started"   # this frame opened an utterance
STOPPED = "stopped"   # the utterance is complete — read audio(), then reset()

_ROUNDING = 1e-6      # the quiet run is a sum of frame durations and lands a hair off a threshold


def patience(spoken: float) -> float:
    """The trailing quiet that ends an utterance `spoken` seconds long (pre-roll included,
    the quiet being measured not): SILENCE up to PATIENCE_FROM, then growing in a straight
    line to SILENCE_LONG at PATIENCE_FULL."""
    if spoken <= PATIENCE_FROM:
        return SILENCE
    share = min(1.0, (spoken - PATIENCE_FROM) / (PATIENCE_FULL - PATIENCE_FROM))
    return SILENCE + share * (SILENCE_LONG - SILENCE)


class Endpointer:
    def __init__(self):
        pre_roll_frames = max(1, int(PRE_ROLL / FRAME_DURATION))
        self._pre_buffer = collections.deque(maxlen=pre_roll_frames)
        self._speech = []
        self.in_speech = False
        self.silence_run = 0.0    # consecutive quiet seconds per the caller's `loud` verdicts

    @property
    def speech_secs(self) -> float:
        return len(self._speech) * FRAME_DURATION

    @property
    def has_audio(self) -> bool:
        """An utterance is in progress or captured frames are pending."""
        return self.in_speech or bool(self._speech)

    @property
    def silence_needed(self) -> float:
        """The trailing quiet that ends the utterance in progress, at its length so far."""
        return patience(self.speech_secs - self.silence_run)

    def audio(self) -> np.ndarray:
        return np.concatenate(self._speech) if self._speech else np.zeros(0, np.float32)

    def reset(self):
        self._speech = []
        self.in_speech = False
        self.silence_run = 0.0

    def clear_pre_roll(self):
        """Drop buffered pre-roll frames (e.g. after a mute, so stale audio of Floxy's
        own voice can't open the next utterance)."""
        self._pre_buffer.clear()

    def feed(self, frame: np.ndarray, loud: bool) -> str | None:
        """Advance one frame. Returns STARTED when this frame opens an utterance,
        STOPPED when the utterance is complete (silence_needed of trailing quiet, or
        MAX_SEGMENT of speech), else None. After STOPPED the caller reads audio()
        and calls reset(); further feeds before that keep returning STOPPED."""
        if not self.in_speech:
            self._pre_buffer.append(frame)
            if not loud:
                return None
            self.in_speech = True
            self._speech = list(self._pre_buffer)
            self._pre_buffer.clear()
            self.silence_run = 0.0
            return STARTED
        self._speech.append(frame)
        if loud:
            self.silence_run = 0.0
        else:
            self.silence_run += FRAME_DURATION
        if self.silence_run + _ROUNDING >= self.silence_needed or self.speech_secs >= MAX_SEGMENT:
            return STOPPED
        return None
