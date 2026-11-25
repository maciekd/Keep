"""Cuts a streamed text into the units a batch-only TTS model speaks, and packs its audio for the pipeline.

A local model (Supertonic, Fish Local) takes a whole text per call, the LLM hands over a stream:
the first unit ends at the earliest clause mark, for a fast first sound, and every later one at
the latest sentence end, so the model has the sentence for its prosody.
"""
import queue
from typing import Callable, Iterator, Optional, Tuple

import numpy as np

CLAUSE_PUNCT = set(",.?!:;—-")
SENTENCE_PUNCT = set(".?!")
MIN_WORDS_FLUSH = 20  # safety net only — the primary trigger is the punctuation
OUTPUT_SAMPLE_RATE = 24000


def pop_speakable(buffer: str, mode: str = 'first') -> Tuple[Optional[str], str]:
    """(unit, remainder) from the front of `buffer`; (None, buffer) while there is none yet.

    mode='first' cuts at the earliest clause mark, mode='batch' at the latest sentence end.
    Both flush a text of MIN_WORDS_FLUSH words without punctuation, never inside a word."""
    if mode == 'first':
        for i, ch in enumerate(buffer):
            if ch in CLAUSE_PUNCT:
                return buffer[:i + 1], buffer[i + 1:]
    else:
        cut = max((i for i, ch in enumerate(buffer) if ch in SENTENCE_PUNCT), default=-1)
        if cut >= 0:
            return buffer[:cut + 1], buffer[cut + 1:]

    if len(buffer.split()) >= MIN_WORDS_FLUSH:
        if buffer and not buffer[-1].isspace():
            cut = buffer.rfind(' ')
            if cut > 0:
                return buffer[:cut], buffer[cut:]
        return buffer, ""
    return None, buffer


def speakable_units(text_queue: queue.Queue, closed: Callable[[], bool]) -> Iterator[Tuple[str, bool]]:
    """(unit, is_first) for the text arriving on `text_queue`, until its None or `closed()`."""
    buffer, emitted = "", 0
    while not closed():
        try:
            chunk = text_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        if chunk is None:
            if buffer.strip():
                yield buffer, emitted == 0
            return
        if not chunk:
            continue
        buffer += chunk
        while True:
            speakable, buffer = pop_speakable(buffer, 'first' if emitted == 0 else 'batch')
            if speakable is None:
                break
            yield speakable, emitted == 0
            emitted += 1


def to_pcm16_24k(wav, source_rate: int) -> bytes:
    """A model's float audio at `source_rate` as the pipeline's int16 mono 24 kHz PCM."""
    arr = np.asarray(wav, dtype=np.float32)
    if arr.ndim > 1:
        # Channels-first (1, samples) or channels-last: the longer axis is the samples.
        arr = arr.mean(axis=0 if arr.shape[0] <= arr.shape[1] else 1)
    if arr.size == 0:
        return b""

    # Linear resampling: speech-grade, and no scipy in the path.
    dst_n = int(round(arr.size * OUTPUT_SAMPLE_RATE / source_rate))
    if dst_n <= 0:
        return b""
    xs = np.linspace(0.0, arr.size - 1, dst_n, dtype=np.float64)
    resampled = np.interp(xs, np.arange(arr.size, dtype=np.float64), arr).astype(np.float32)
    return (np.clip(resampled, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
