"""interfaces.voice.tts.ClauseBatcher — the units a batch-only model speaks, and its audio as the pipeline's PCM."""
import queue

import numpy as np

from interfaces.voice.tts.ClauseBatcher import pop_speakable, speakable_units, to_pcm16_24k


class TestClauseBoundaryFlusher:
    def _pop(self, buffer):
        return pop_speakable(buffer)

    def test_flush_at_comma(self):
        speakable, rest = self._pop("Hello, world.")
        assert speakable == "Hello,"
        assert rest == " world."

    def test_word_count_fallback_keeps_partial_word(self):
        # 20 complete words + a partial word at the end → flush the 20 words,
        # keep the partial word for the next round.
        words = " ".join(f"w{i}" for i in range(20))
        speakable, rest = self._pop(f"{words} etacontinues")
        assert speakable == words
        assert rest == " etacontinues"

    def test_word_count_fallback_when_ending_on_space(self):
        # Already ends on whitespace — safe to flush everything.
        words = " ".join(f"w{i}" for i in range(20))
        speakable, rest = self._pop(f"{words} ")
        assert speakable == f"{words} "
        assert rest == ""

    def test_word_count_fallback_does_not_fire_under_threshold(self):
        # 19 words with no punctuation → no flush (threshold is 20).
        words = " ".join(f"w{i}" for i in range(19))
        speakable, rest = self._pop(words)
        assert speakable is None
        assert rest == words

    def test_empty_buffer(self):
        speakable, rest = self._pop("")
        assert speakable is None
        assert rest == ""


class TestBatchModeFlusher:
    """`mode='batch'` is the prosody path used once audio is already flowing: it
    flushes through the *rightmost* sentence-end (".?!") so the model gets full
    sentence context, rather than the earliest clause boundary like 'first' mode."""

    def _pop_batch(self, buffer):
        return pop_speakable(buffer, mode='batch')

    def test_flushes_through_rightmost_sentence_end(self):
        # Two complete sentences plus a partial third → batch the first two,
        # keep the trailing partial. (First mode would have stopped at "One.")
        speakable, rest = self._pop_batch("One. Two. Three")
        assert speakable == "One. Two."
        assert rest == " Three"

    def test_ignores_clause_punct_without_sentence_end(self):
        # Commas/colons are clause punctuation, not sentence-ends — batch mode
        # must NOT flush on them (this is what distinguishes it from 'first').
        speakable, rest = self._pop_batch("Hello, there: friend")
        assert speakable is None
        assert rest == "Hello, there: friend"

    def test_single_sentence_flushes_whole(self):
        speakable, rest = self._pop_batch("Just one sentence.")
        assert speakable == "Just one sentence."
        assert rest == ""

    def test_word_count_fallback_still_applies(self):
        # No sentence punctuation at all, but past the word threshold → flush the
        # complete words, keep the trailing partial word.
        words = " ".join(f"w{i}" for i in range(20))
        speakable, rest = self._pop_batch(f"{words} partial")
        assert speakable == words
        assert rest == " partial"


class TestPcmConversion:
    def test_mono_float32_to_pcm16(self):
        # 1 second of non-silent audio @ 44.1 kHz → ~24000 samples @ 24 kHz = 48000 bytes
        wav = np.full(44100, 0.3, dtype=np.float32)
        pcm = to_pcm16_24k(wav, 44100)
        assert isinstance(pcm, bytes)
        # int16 is 2 bytes/sample; 24000 samples = 48000 bytes (allow ±2 for rounding)
        assert abs(len(pcm) - 48000) <= 4

    def test_empty_input_returns_empty_bytes(self):
        assert to_pcm16_24k(np.array([], dtype=np.float32), 44100) == b""

    def test_stereo_is_downmixed(self):
        wav = np.full((44100, 2), 0.3, dtype=np.float32)
        pcm = to_pcm16_24k(wav, 44100)
        assert abs(len(pcm) - 48000) <= 4

    def test_supertonic_native_shape_1_x_samples(self):
        """Supertonic returns shape (1, samples) — channels-first. Earlier bug:
        meaning over axis=-1 collapsed the waveform to a single scalar."""
        # Non-zero so we'd notice if values got averaged out incorrectly
        wav = np.full((1, 44100), 0.5, dtype=np.float32)
        pcm = to_pcm16_24k(wav, 44100)
        assert abs(len(pcm) - 48000) <= 4
        as_int16 = np.frombuffer(pcm, dtype=np.int16)
        # Every sample should be ~0.5 * 32767, not zero
        assert as_int16.min() > 16000

    def test_clipping(self):
        # Values >1.0 should be clipped to int16 max (32767)
        wav = np.full(100, 2.5, dtype=np.float32)
        pcm = to_pcm16_24k(wav, 44100)
        as_int16 = np.frombuffer(pcm, dtype=np.int16)
        assert as_int16.max() == 32767
        assert as_int16.min() == 32767  # all values were positive


class TestSpeakableUnits:
    @staticmethod
    def _units(*chunks, closed=lambda: False):
        q = queue.Queue()
        for chunk in chunks:
            q.put(chunk)
        return list(speakable_units(q, closed))

    def test_the_opener_is_a_clause_and_what_follows_whole_sentences(self):
        units = self._units("Sure, the lights ", "are on. The door is locked. And", " the car charges", None)
        assert units == [("Sure,", True), (" the lights are on. The door is locked.", False),
                         (" And the car charges", False)]

    def test_a_text_without_punctuation_is_spoken_at_its_end_as_the_opener(self):
        assert self._units("no punctuation here", None) == [("no punctuation here", True)]

    def test_nothing_but_white_space_left_at_the_end_is_no_unit(self):
        assert self._units("Done.", "  ", None) == [("Done.", True)]

    def test_a_closed_stream_yields_nothing_more(self):
        assert self._units("One. Two.", None, closed=lambda: True) == []

    def test_other_rates_keep_their_duration(self):
        """A second of audio is a second of PCM whatever rate the model emits it at."""
        assert abs(len(to_pcm16_24k(np.full(16000, 0.1, dtype=np.float32), 16000)) - 48000) <= 4
