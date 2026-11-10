import sys

import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock


def _make_vs(language="en-US", **cfg_overrides):
    """Build a SupertonicVoiceSynthesizer with config.get patched.

    Returns the instance. cfg_overrides lets a test pin specific
    `tts.supertonic.*` values; defaults match the class defaults. `language`
    pins the top-level `language` config key that drives the synthesis lang.
    """
    supertonic_cfg = {
        'voice': cfg_overrides.get('voice', 'F1'),
        'first_chunk_steps': cfg_overrides.get('first_chunk_steps', 5),
        'total_steps': cfg_overrides.get('total_steps', 8),
        'speed': cfg_overrides.get('speed', 1.3),
    }
    with patch('interfaces.voice.tts.SupertonicVoiceSynthesizer.config') as mock_config, \
            patch('common.Language.config') as mock_base_config:
        mock_config.get.side_effect = (
            lambda key, default=None: supertonic_cfg if key == 'tts.supertonic' else default
        )
        mock_base_config.get.side_effect = (
            lambda key, default=None: language if key == 'language' else default
        )
        from interfaces.voice.tts.SupertonicVoiceSynthesizer import SupertonicVoiceSynthesizer
        # Wipe class-level cache so successive tests don't reuse a mocked TTS
        SupertonicVoiceSynthesizer._shared_tts = None
        SupertonicVoiceSynthesizer._shared_style = None
        SupertonicVoiceSynthesizer._shared_voice_name = None
        return SupertonicVoiceSynthesizer()


class TestStaticBehavior:
    def test_voice_id_reflects_config(self):
        with patch('interfaces.voice.tts.SupertonicVoiceSynthesizer.config') as mock_config:
            mock_config.get.side_effect = (
                lambda key, default=None: {'voice': 'F2'} if key == 'tts.supertonic' else default
            )
            from interfaces.voice.tts.SupertonicVoiceSynthesizer import SupertonicVoiceSynthesizer
            assert SupertonicVoiceSynthesizer.voice_id() == "F2"

class TestConfig:
    def test_lang_derives_from_top_level_language(self):
        """Supertonic has no separate lang setting — it takes the primary subtag
        of the global `language` config (e.g. 'sv-SE' → 'sv')."""
        assert _make_vs(language="sv-SE")._lang == "sv"
        assert _make_vs(language="en")._lang == "en"


class TestRequestLanguage:
    """The language a segment is synthesized in follows the request, not only the config."""

    def _synth_langs(self, vs):
        tts = MagicMock()
        tts.synthesize.return_value = (np.zeros(441, dtype=np.float32), 0.01)
        tts.model.text_processor.validate_text.return_value = (True, [])
        vs.text_queue.put("Hej, världen.")
        vs.text_queue.put(None)
        # The end-of-stream grace is a real 2 s sleep for the player's sake; nothing here plays.
        with patch.object(vs, '_get_tts', return_value=(tts, MagicMock())), \
                patch.object(type(vs), 'END_OF_STREAM_GRACE_S', 0):
            vs._run_synthesis()
        return {c.kwargs['lang'] for c in tts.synthesize.call_args_list}

    def test_swedish_request_on_an_english_install_is_read_as_swedish(self):
        vs = _make_vs(language="en-US")
        vs.set_request_language("sv-SE")
        assert self._synth_langs(vs) == {"sv"}

    def test_without_a_request_language_the_configured_one_is_used(self):
        assert self._synth_langs(_make_vs(language="sv-SE")) == {"sv"}

    def test_a_request_language_supertonic_lacks_falls_back_to_the_configured_one(self):
        vs = _make_vs(language="en-US")
        vs.set_request_language("is-IS")
        assert self._synth_langs(vs) == {"en"}


class TestClauseBoundaryFlusher:
    def _pop(self, buffer):
        from interfaces.voice.tts.SupertonicVoiceSynthesizer import SupertonicVoiceSynthesizer
        return SupertonicVoiceSynthesizer._pop_speakable(buffer)

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
        from interfaces.voice.tts.SupertonicVoiceSynthesizer import SupertonicVoiceSynthesizer
        return SupertonicVoiceSynthesizer._pop_speakable(buffer, mode='batch')

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
        from interfaces.voice.tts.SupertonicVoiceSynthesizer import SupertonicVoiceSynthesizer
        # 1 second of non-silent audio @ 44.1 kHz → ~24000 samples @ 24 kHz = 48000 bytes
        wav = np.full(44100, 0.3, dtype=np.float32)
        pcm = SupertonicVoiceSynthesizer._to_pcm16_24k(wav)
        assert isinstance(pcm, bytes)
        # int16 is 2 bytes/sample; 24000 samples = 48000 bytes (allow ±2 for rounding)
        assert abs(len(pcm) - 48000) <= 4

    def test_empty_input_returns_empty_bytes(self):
        from interfaces.voice.tts.SupertonicVoiceSynthesizer import SupertonicVoiceSynthesizer
        assert SupertonicVoiceSynthesizer._to_pcm16_24k(np.array([], dtype=np.float32)) == b""

    def test_stereo_is_downmixed(self):
        from interfaces.voice.tts.SupertonicVoiceSynthesizer import SupertonicVoiceSynthesizer
        wav = np.full((44100, 2), 0.3, dtype=np.float32)
        pcm = SupertonicVoiceSynthesizer._to_pcm16_24k(wav)
        assert abs(len(pcm) - 48000) <= 4

    def test_supertonic_native_shape_1_x_samples(self):
        """Supertonic returns shape (1, samples) — channels-first. Earlier bug:
        meaning over axis=-1 collapsed the waveform to a single scalar."""
        from interfaces.voice.tts.SupertonicVoiceSynthesizer import SupertonicVoiceSynthesizer
        # Non-zero so we'd notice if values got averaged out incorrectly
        wav = np.full((1, 44100), 0.5, dtype=np.float32)
        pcm = SupertonicVoiceSynthesizer._to_pcm16_24k(wav)
        assert abs(len(pcm) - 48000) <= 4
        as_int16 = np.frombuffer(pcm, dtype=np.int16)
        # Every sample should be ~0.5 * 32767, not zero
        assert as_int16.min() > 16000

    def test_clipping(self):
        from interfaces.voice.tts.SupertonicVoiceSynthesizer import SupertonicVoiceSynthesizer
        # Values >1.0 should be clipped to int16 max (32767)
        wav = np.full(100, 2.5, dtype=np.float32)
        pcm = SupertonicVoiceSynthesizer._to_pcm16_24k(wav)
        as_int16 = np.frombuffer(pcm, dtype=np.int16)
        assert as_int16.max() == 32767
        assert as_int16.min() == 32767  # all values were positive


class TestRunSynthesis:
    @pytest.fixture(autouse=True)
    def _no_grace(self):
        """Skip the end-of-stream grace sleep so the test suite stays fast.
        The grace behavior itself has a dedicated test below."""
        from interfaces.voice.tts.SupertonicVoiceSynthesizer import SupertonicVoiceSynthesizer
        with patch.object(SupertonicVoiceSynthesizer, 'END_OF_STREAM_GRACE_S', 0):
            yield

    def _fake_tts_pair(self):
        """Return a (tts, style) pair whose synthesize() returns a short fake wav."""
        fake_tts = Mock()
        fake_style = Mock()
        # 0.1s of audio at 44.1kHz
        fake_wav = np.full(4410, 0.3, dtype=np.float32)
        fake_tts.synthesize.return_value = (fake_wav, 0.1)
        fake_tts.model.text_processor.validate_text.return_value = (True, [])
        return fake_tts, fake_style

    def test_sleeps_before_none_sentinel_to_let_client_drain(self):
        """Before signaling end-of-stream, sleep END_OF_STREAM_GRACE_S so the
        sounddevice/client playback buffer can drain. Without this the tail of
        the last clause intermittently clips."""
        vs = _make_vs()
        tts, style = self._fake_tts_pair()

        vs.text_queue.put("Hello, world.")
        vs.text_queue.put(None)

        with patch.object(type(vs), 'END_OF_STREAM_GRACE_S', 0.25):
            with patch.object(vs, '_get_tts', return_value=(tts, style)):
                with patch('interfaces.voice.tts.SupertonicVoiceSynthesizer.time.sleep') as mock_sleep:
                    vs._run_synthesis()

        sleep_durations = [c.args[0] for c in mock_sleep.call_args_list]
        assert 0.25 in sleep_durations, f"expected a 0.25s grace sleep, got {sleep_durations}"

    def test_grace_skipped_when_closed(self):
        """If close() was called mid-stream, skip the grace — no point making a
        torn-down request wait."""
        vs = _make_vs()
        tts, style = self._fake_tts_pair()

        vs.text_queue.put(None)
        vs.is_closed = True

        with patch.object(type(vs), 'END_OF_STREAM_GRACE_S', 0.5):
            with patch.object(vs, '_get_tts', return_value=(tts, style)):
                with patch('interfaces.voice.tts.SupertonicVoiceSynthesizer.time.sleep') as mock_sleep:
                    vs._run_synthesis()

        sleep_durations = [c.args[0] for c in mock_sleep.call_args_list]
        assert 0.5 not in sleep_durations

    def test_emits_audio_then_none_sentinel(self):
        vs = _make_vs()
        tts, style = self._fake_tts_pair()

        vs.text_queue.put("Hello, world.")
        vs.text_queue.put(None)

        with patch.object(vs, '_get_tts', return_value=(tts, style)):
            vs._run_synthesis()

        items = []
        while not vs.audio_queue.empty():
            items.append(vs.audio_queue.get_nowait())

        assert items[-1] is None
        audio_chunks = [x for x in items if x is not None]
        assert len(audio_chunks) >= 1
        # Each chunk is non-empty PCM bytes
        for chunk in audio_chunks:
            assert isinstance(chunk, bytes) and len(chunk) > 0

    def test_clause_flusher_calls_synthesize_per_clause(self):
        vs = _make_vs()
        tts, style = self._fake_tts_pair()

        vs.text_queue.put("Hello, ")
        vs.text_queue.put("world.")
        vs.text_queue.put(None)

        with patch.object(vs, '_get_tts', return_value=(tts, style)):
            vs._run_synthesis()

        # Expect 2 synthesize calls: one for "Hello,", one for "world."
        assert tts.synthesize.call_count == 2
        texts = [c.kwargs.get('text') or c.args[0] for c in tts.synthesize.call_args_list]
        joined = " ".join(texts)
        assert "Hello" in joined
        assert "world" in joined

    def test_first_chunk_uses_first_chunk_steps_subsequent_uses_total_steps(self):
        vs = _make_vs(language="ko", first_chunk_steps=4, total_steps=11, speed=1.1)
        tts, style = self._fake_tts_pair()

        # Two sentences in one chunk → first emit (opener, 'first' mode) plus a
        # batched emit covering the remainder.
        vs.text_queue.put("Hi there. How are you today?")
        vs.text_queue.put(None)

        with patch.object(vs, '_get_tts', return_value=(tts, style)):
            vs._run_synthesis()

        calls = tts.synthesize.call_args_list
        assert len(calls) == 2, f"expected 2 synth calls, got {len(calls)}"

        first_kwargs = calls[0].kwargs
        assert first_kwargs['lang'] == "ko"
        assert first_kwargs['total_steps'] == 4   # first_chunk_steps
        assert first_kwargs['speed'] == 1.1
        assert first_kwargs['voice_style'] is style

        subsequent_kwargs = calls[1].kwargs
        assert subsequent_kwargs['total_steps'] == 11  # total_steps for batched chunks
        assert subsequent_kwargs['speed'] == 1.1

    def test_skips_text_with_no_alphanumerics(self):
        vs = _make_vs()
        tts, style = self._fake_tts_pair()

        vs.text_queue.put("...,?!")  # all punctuation
        vs.text_queue.put(None)

        with patch.object(vs, '_get_tts', return_value=(tts, style)):
            vs._run_synthesis()

        tts.synthesize.assert_not_called()
        assert vs.audio_queue.get_nowait() is None

    def test_model_load_failure_queues_none(self):
        vs = _make_vs()

        with patch.object(vs, '_get_tts', side_effect=RuntimeError("oops")):
            vs._run_synthesis()

        assert vs.audio_queue.get_nowait() is None

    def test_unsupported_chars_stripped_before_synthesis(self):
        """An emoji in a clause must not kill the whole segment — strip exactly
        the characters the model's validator rejects and synthesize the rest."""
        vs = _make_vs()
        tts, style = self._fake_tts_pair()
        tts.model.text_processor.validate_text.return_value = (False, ['⏰'])

        vs.text_queue.put("48 PM ⏰.")
        vs.text_queue.put(None)

        with patch.object(vs, '_get_tts', return_value=(tts, style)):
            vs._run_synthesis()

        tts.synthesize.assert_called_once()
        assert tts.synthesize.call_args.kwargs['text'] == "48 PM ."

    def test_segment_of_only_unsupported_chars_is_skipped(self):
        vs = _make_vs()
        tts, style = self._fake_tts_pair()
        tts.model.text_processor.validate_text.return_value = (False, ['⏰', '🎉'])

        vs.text_queue.put("⏰🎉.")
        vs.text_queue.put(None)

        with patch.object(vs, '_get_tts', return_value=(tts, style)):
            vs._run_synthesis()

        tts.synthesize.assert_not_called()
        assert vs.audio_queue.get_nowait() is None

    def test_text_passes_through_when_validator_unavailable(self):
        """If the supertonic API doesn't expose the validator (older version,
        odd mock), don't filter — let synthesize() decide."""
        vs = _make_vs()
        tts, style = self._fake_tts_pair()
        tts.model.text_processor.validate_text.side_effect = AttributeError("no validator")

        vs.text_queue.put("Hello there ⏰.")
        vs.text_queue.put(None)

        with patch.object(vs, '_get_tts', return_value=(tts, style)):
            vs._run_synthesis()

        tts.synthesize.assert_called_once()
        assert tts.synthesize.call_args.kwargs['text'] == "Hello there ⏰."

    def test_synth_failure_for_one_clause_doesnt_break_stream(self):
        vs = _make_vs()
        fake_tts = Mock()
        fake_style = Mock()
        fake_wav = np.full(4410, 0.3, dtype=np.float32)
        # First call raises, second succeeds
        fake_tts.synthesize.side_effect = [RuntimeError("boom"), (fake_wav, 0.1)]

        vs.text_queue.put("First clause, ")
        vs.text_queue.put("second clause.")
        vs.text_queue.put(None)

        with patch.object(vs, '_get_tts', return_value=(fake_tts, fake_style)):
            vs._run_synthesis()

        items = []
        while not vs.audio_queue.empty():
            items.append(vs.audio_queue.get_nowait())

        assert items[-1] is None
        assert any(isinstance(x, bytes) for x in items)


class TestStartAndLifecycle:
    @pytest.fixture(autouse=True)
    def _no_grace(self):
        from interfaces.voice.tts.SupertonicVoiceSynthesizer import SupertonicVoiceSynthesizer
        with patch.object(SupertonicVoiceSynthesizer, 'END_OF_STREAM_GRACE_S', 0):
            yield

    def test_start_resets_state(self):
        vs = _make_vs()
        vs.is_closed = True
        vs.audio_queue.put(b"stale")
        vs.text_queue.put("stale")

        tts, style = Mock(), Mock()
        tts.synthesize.return_value = (np.full(4410, 0.3, dtype=np.float32), 0.1)

        with patch.object(vs, '_get_tts', return_value=(tts, style)):
            stream = vs.start(iter(["Hello, world."]))
            chunks = list(stream)

        assert vs.is_closed is False
        # stale text shouldn't have been synthesized — only "Hello, world."
        synthesized = " ".join(c.kwargs.get('text') or c.args[0] for c in tts.synthesize.call_args_list)
        assert "stale" not in synthesized

class TestPreloadAndHealth:
    def test_get_tts_caches_at_class_level(self):
        vs = _make_vs()
        from interfaces.voice.tts.SupertonicVoiceSynthesizer import SupertonicVoiceSynthesizer

        fake_tts = MagicMock()
        fake_tts.get_voice_style.return_value = "fake-style"
        fake_module = MagicMock(TTS=Mock(return_value=fake_tts))

        with patch.dict('sys.modules', {'supertonic': fake_module}):
            t1, s1 = vs._get_tts()
            t2, s2 = vs._get_tts()

        assert t1 is t2
        assert s1 == s2 == "fake-style"
        # TTS constructor called exactly once — second call hit the class cache
        assert fake_module.TTS.call_count == 1

        # Cleanup so the cached fake doesn't bleed into other tests
        SupertonicVoiceSynthesizer._shared_tts = None
        SupertonicVoiceSynthesizer._shared_style = None
        SupertonicVoiceSynthesizer._shared_voice_name = None
        SupertonicVoiceSynthesizer._shared_warmed_up = False

    def test_warmup_runs_once_on_first_load(self):
        vs = _make_vs()
        from interfaces.voice.tts.SupertonicVoiceSynthesizer import SupertonicVoiceSynthesizer

        fake_tts = MagicMock()
        fake_tts.get_voice_style.return_value = "fake-style"
        fake_module = MagicMock(TTS=Mock(return_value=fake_tts))

        with patch.dict('sys.modules', {'supertonic': fake_module}):
            vs._get_tts()  # first call: load + warmup
            vs._get_tts()  # second call: cached, no extra warmup

        # Warmup synth runs exactly once even though _get_tts was called twice
        assert fake_tts.synthesize.call_count == 1
        call = fake_tts.synthesize.call_args
        # Warmup feeds a non-empty priming utterance at first_chunk_steps (5 by
        # default in _make_vs). We don't pin the exact text — it's free to change.
        assert isinstance(call.kwargs['text'], str) and call.kwargs['text'].strip()
        assert call.kwargs['total_steps'] == vs._first_chunk_steps
        assert SupertonicVoiceSynthesizer._shared_warmed_up is True

        SupertonicVoiceSynthesizer._shared_tts = None
        SupertonicVoiceSynthesizer._shared_style = None
        SupertonicVoiceSynthesizer._shared_voice_name = None
        SupertonicVoiceSynthesizer._shared_warmed_up = False

    def test_warmup_failure_does_not_block(self):
        vs = _make_vs()
        from interfaces.voice.tts.SupertonicVoiceSynthesizer import SupertonicVoiceSynthesizer

        fake_tts = MagicMock()
        fake_tts.get_voice_style.return_value = "fake-style"
        fake_tts.synthesize.side_effect = RuntimeError("warmup boom")
        fake_module = MagicMock(TTS=Mock(return_value=fake_tts))

        with patch.dict('sys.modules', {'supertonic': fake_module}):
            tts, style = vs._get_tts()

        # Even though warmup raised, _get_tts still returned the model and
        # marked it warm so we don't retry forever.
        assert tts is fake_tts
        assert SupertonicVoiceSynthesizer._shared_warmed_up is True

        SupertonicVoiceSynthesizer._shared_tts = None
        SupertonicVoiceSynthesizer._shared_style = None
        SupertonicVoiceSynthesizer._shared_voice_name = None
        SupertonicVoiceSynthesizer._shared_warmed_up = False


class TestBackgroundPreload:
    """Construction kicks off a one-shot background ONNX preload (moved here
    from floxy_ai_server in 3e6e8d3d). It must spawn at most one daemon thread
    per process and must be skipped entirely under pytest, so the test suite
    never pays the cold-start cost or leaks stray daemon threads."""

    @pytest.fixture(autouse=True)
    def _reset_flag(self):
        """Clear the once-per-process guard around each test so cases don't leak
        into one another (the flag is class-level state)."""
        from interfaces.voice.tts.SupertonicVoiceSynthesizer import SupertonicVoiceSynthesizer
        SupertonicVoiceSynthesizer._preload_started = False
        yield
        SupertonicVoiceSynthesizer._preload_started = False

    # NOTE: these cases patch the per-class _spawn_preload_thread seam rather than
    # the module-global threading.Thread. The synthesis path (start/streaming)
    # also spawns threads via the same module-global Thread, so leaked synthesis
    # daemons from other tests would race into a global-Thread mock and flakily
    # inflate its call count. _spawn_preload_thread is only ever called by the
    # preload guard, so counting it is deterministic.

    def test_skipped_under_pytest(self):
        # Under pytest, construction must skip the one-shot background preload.
        from interfaces.voice.tts.SupertonicVoiceSynthesizer import SupertonicVoiceSynthesizer
        with patch.dict('sys.modules', {'pytest': pytest}), \
             patch.object(SupertonicVoiceSynthesizer, '_spawn_preload_thread') as spawn:
            _make_vs()
        spawn.assert_not_called()
        assert SupertonicVoiceSynthesizer._preload_started is False

    def test_spawns_preload_once_when_not_under_pytest(self):
        from interfaces.voice.tts.SupertonicVoiceSynthesizer import SupertonicVoiceSynthesizer
        with patch.dict('sys.modules'), \
             patch.object(SupertonicVoiceSynthesizer, '_spawn_preload_thread') as spawn:
            sys.modules.pop('pytest', None)  # simulate "not under pytest"
            _make_vs()
            _make_vs()  # second construction must NOT spawn a second thread

        spawn.assert_called_once()
        assert SupertonicVoiceSynthesizer._preload_started is True

    def test_preload_thread_target_calls_preload_and_swallows_errors(self):
        vs = _make_vs()
        with patch('interfaces.voice.tts.SupertonicVoiceSynthesizer.threading.Thread') as mock_thread:
            vs._spawn_preload_thread()

        preload_calls = [c for c in mock_thread.call_args_list
                         if c.kwargs.get('name') == 'SupertonicPreload']
        target = preload_calls[0].kwargs['target']

        with patch.object(vs, 'preload') as mock_preload:
            target()
            mock_preload.assert_called_once()

        # A failure inside preload must be swallowed, not propagated out of the
        # daemon thread (otherwise the thread dies noisily and the server logs a
        # bare traceback instead of our error() line).
        with patch.object(vs, 'preload', side_effect=RuntimeError("boom")):
            target()  # must not raise
