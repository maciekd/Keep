import queue
import sys
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from interfaces.voice.microphones.local_outposts.LocalOutpost import (
    LocalOutpost, strip_wake_word, wake_word_prefix, deep_wake_word_index, clean_query,
    is_priming_echo, doubled_wake_word, near_wake_word_token, _StreamStalled,
    command_similarity, scribe_verify_disagreement_veto, shares_content_word,
    gate_whisper_transcript, gate_scribe_verify, gate_scribe_rescue, matching_command_prefix,
)
from interfaces.voice.microphones.local_outposts.constants import (
    INITIAL_PROMPT_PHRASES, RECOVERY_OFFBUS_GRACE, USB_POWER_CYCLE_MAX_ATTEMPTS,
    CAPTURE_DROP_STARTUP_GRACE, CAPTURE_DROP_MIN_GAP, WAKE_WORD_ANCHOR_CHARS,
    WAKE_WORD_PREFIX_CONFIDENCE_MIN_PCT, SCRIBE_CONF_CALIBRATION, LISTEN_NO_NEW_WORDS_TIMEOUT,
)
from interfaces.voice.microphones.local_outposts.LocalOutpostManager import (
    LocalOutpostManager, filter_input_devices, disambiguate_mic_names,
)
from interfaces.voice.microphones.local_outposts.OutpostStreamClient import OutpostStreamClient
from interfaces.voice.microphones.local_outposts.TextLanguageDetector import TextLanguageDetector
from interfaces.voice.microphones.local_outposts.WhisperWorker import segment_metrics, metrics_risky, wake_word_hallucination_risky, normalize_loudness


@pytest.fixture(autouse=True)
def cancel_outpost_timers(monkeypatch):
    """An outpost arms 2.5/5/15 s listen timers as it runs; none may outlive its test."""
    built = []
    real_init = LocalOutpost.__init__

    def tracking_init(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        built.append(self)

    monkeypatch.setattr(LocalOutpost, "__init__", tracking_init)
    yield
    for outpost in built:
        outpost._cancel_listen_timers()


def make_outpost(awaiting_wakeword=True, muted=False, languages=("sv-SE",), lang_detector=None):
    worker, client = MagicMock(), MagicMock()
    default_lang = languages[0]
    lang_by_subtag = {}
    for tag in languages:
        lang_by_subtag.setdefault(tag.split("-")[0].lower(), tag)
    outpost = LocalOutpost(0, "Test Mic", worker, client, default_lang, lang_by_subtag,
                           lang_detector=lang_detector)
    outpost.awaiting_wakeword = awaiting_wakeword
    outpost.muted = muted
    return outpost, client


CLEAN = {"conf": 0.8, "logprob": -0.2, "no_speech": 0.1, "compression": 0.8, "temperature": 0.0}
AUDIO = np.zeros(8000, np.float32)  # stand-in utterance audio for the transcript gate


@pytest.mark.unit
class TestStripWakeWord:
    def test_strips_wake_word_and_keeps_remainder(self):
        # Mirrors Android: substring after the wake word, trimmed (leading comma kept).
        assert strip_wake_word("Michelle, turn on the lights") == (True, ", turn on the lights")

    def test_case_insensitive(self):
        assert strip_wake_word("hey MICHELLE play music")[0] is True

    def test_no_wake_word_returns_original(self):
        assert strip_wake_word("turn on the lights") == (False, "turn on the lights")

    def test_wake_word_only(self):
        assert strip_wake_word("Michelle")[1] == ""

    def test_mid_sentence_wake_word_is_not_an_address(self):
        # An overheard sentence naming the assistant mid-flow.
        # "Michelle" sits 19 chars in, past the anchor, so it must NOT trigger.
        text = "Vill du hälsa till Michelle från oss, Frances?"
        assert strip_wake_word(text) == (False, text)

    def test_wake_word_just_past_anchor_is_ignored(self):
        # "Michelle" one char past the window (index == anchor) — rejected.
        text = "x" * WAKE_WORD_ANCHOR_CHARS + "Michelle hello"
        assert strip_wake_word(text)[0] is False

    def test_wake_word_at_anchor_edge_still_triggers(self):
        # Starts at the last in-window position (index == anchor - 1) — still an address.
        text = "x" * (WAKE_WORD_ANCHOR_CHARS - 1) + "Michelle hello"
        assert strip_wake_word(text)[0] is True

    def test_deep_wake_word_index_reports_position(self):
        assert deep_wake_word_index("Vill du hälsa till Michelle från oss?") == 19

    def test_deep_wake_word_index_minus_one_when_anchored(self):
        assert deep_wake_word_index("Michelle turn on the lights") == -1
        assert deep_wake_word_index("turn on the lights") == -1


@pytest.mark.unit
class TestDoubledWakeWord:
    """doubled_wake_word: "Michelle Michelle …" is a hallucination signature — production
    false positives decoded Swedish household speech into a doubled wake word (both engines primed
    with it), and no genuine WakeWordTest fire ever doubles it."""

    def test_adjacent_doubling_detected(self):
        assert doubled_wake_word("Michelle Michelle") is True

    def test_doubling_with_command_detected(self):
        # A production false-positive decode shape: doubled opener, command-ish tail.
        assert doubled_wake_word("Michelle Michelle, is it over there? Hmm?") is True

    def test_punctuation_between_still_counts(self):
        assert doubled_wake_word("Michelle, Michelle!") is True

    def test_case_insensitive(self):
        assert doubled_wake_word("MICHELLE michelle") is True

    def test_single_wake_word_passes(self):
        assert doubled_wake_word("Michelle, turn on the lights") is False
        assert doubled_wake_word("Michelle") is False

    def test_non_adjacent_repeats_pass(self):
        # Only back-to-back repetition is the signature; a later re-mention is not.
        assert doubled_wake_word("Michelle, tell Michelle's story") is False

    def test_doubled_non_wake_token_passes(self):
        # Only doubling of the WAKE word is the signature — a stutter or decode
        # repeat of an ordinary word in a genuine command must not veto the fire.
        assert doubled_wake_word("Michelle, turn turn on the lights") is False


@pytest.mark.unit
class TestCommandSimilarity:
    def test_spacing_and_punctuation_are_ignored(self):
        assert command_similarity("goodnight", "Good night!") == 1.0
        assert command_similarity("", "") == 1.0
        assert command_similarity("", "volume down") == 0.0

    def test_the_two_correlated_fps_are_far_apart(self):
        assert command_similarity("close the window", "did you bring any of it?") <= 0.20
        assert command_similarity("close the window", "is there more of this?") <= 0.20
        assert command_similarity("who's there?", "all good?") <= 0.20
        # a mishear that still shares substance is not a disagreement
        assert command_similarity("run with the trains", "when will it rain?") > 0.20

    def test_disagreement_veto_needs_both_sides_and_both_confidences(self):
        v = scribe_verify_disagreement_veto
        assert v("", "volume down", 0.4, 0.5) is None
        assert v("volume down", "!", 0.4, 0.5) is None
        assert v("close the window", "all good?", None, 0.6) is None
        assert v("close the window", "all good?", 0.4, None) is None
        assert v("close the window", "all good?", 0.4, 0.9) is None      # scribe sure
        assert v("close the window", "all good?", 0.56, 0.6) is None     # whisper sure
        assert v("close the window", "all good?", 0.4, 0.6)

    def test_a_shared_content_word_is_never_a_disagreement(self):
        # WakeWordTest "will it rain today": Whisper "what would you like to drink today?"
        # 42% / Scribe "will it rain today?" 0.83 — bigram similarity 0.19, but "today".
        assert shares_content_word("what would you like to drink today?", "will it rain today?")
        assert not shares_content_word("I told you. I told you so.", "You're the chief camera")
        assert not shares_content_word("what do you think?", "wood")
        assert scribe_verify_disagreement_veto("what would you like to drink today?",
                                               "will it rain today?", 0.42, 0.83) is None
        # "please" counts: a WakeWordTest clip — Whisper
        # "one minute, please" 50% / Scribe "volume really please" 0.57, both botched.
        assert shares_content_word("one minute, please", "volume really please")
        assert scribe_verify_disagreement_veto("one minute, please", "volume really please",
                                               0.497, 0.57) is None


@pytest.mark.unit
class TestNearMissWakeWord:
    """near_wake_word_token: the opening word of a Scribe transcript when it is one of the
    configured WAKE_WORD_NEAR_MISSES (words Scribe renders a spoken "Michelle" as —
    "Mission complete"), used to keep a genuine fire alive through a Scribe misspelling
    on the verify path."""

    def test_suite_and_prod_near_misses(self):
        assert near_wake_word_token("Mission complete") == "mission"
        assert near_wake_word_token("Missions to kitchen temperature") == "missions"

    def test_multi_word_near_miss(self):
        # A WakeWordTest clip: Scribe "We shall volume down".
        assert near_wake_word_token("We shall volume down") == "we shall"
        assert near_wake_word_token("We, shall we?") == "we shall"
        # "we" alone (or followed by anything else) is not the rendering.
        assert near_wake_word_token("We should go") is None
        assert near_wake_word_token("We") is None

    def test_case_and_punctuation_insensitive(self):
        assert near_wake_word_token("MISSION, complete!") == "mission"

    def test_exact_wake_word_is_not_a_near_miss(self):
        # The exact wake word is strip_wake_word's business; the near-miss path never
        # touches it (nor the correlated-hallucination decodes that carry it).
        assert near_wake_word_token("Michelle, good night") is None

    def test_only_the_opening_word_counts(self):
        assert near_wake_word_token("okay so mission complete") is None

    def test_unrelated_openers_are_none(self):
        # Prod-veto openers and ordinary words: only a listed word counts, nothing fuzzy.
        for text in ("Mycket bra väder", "Much better", "Sure. Put the kettle on now",
                     "Let me show you how", "Missing keys", "Michigan", "Vi ska äta nu", "", "..."):
            assert near_wake_word_token(text) is None, text


@pytest.mark.unit
class TestWakeWordPrefix:
    def test_flush_wake_word_has_no_prefix(self):
        assert wake_word_prefix("Michelle, turn on the lights") == ""

    def test_spoken_lead_in_is_returned(self):
        # A false activation shape: hallucinated words ahead
        # of the wake word, still inside the anchor window.
        assert wake_word_prefix("Okay, Michelle, that is all for now.") == "Okay, "

    def test_no_wake_word_returns_empty(self):
        assert wake_word_prefix("turn on the lights") == ""

    def test_wake_word_past_anchor_returns_empty(self):
        # Past the anchor window the wake word is an overheard mention, not a fire —
        # so there is no prefix to gate on.
        assert wake_word_prefix("x" * WAKE_WORD_ANCHOR_CHARS + "Michelle hello") == ""


@pytest.mark.unit
class TestCleanQuery:
    def test_removes_leading_comma_and_capitalizes(self):
        # The classic "Michelle, what's the outdoor temperature?" leftover.
        assert clean_query(", what's the outdoor temperature?") == "What's the outdoor temperature?"

    def test_strips_leading_separators_and_whitespace(self):
        assert clean_query("  -  turn on the lights") == "Turn on the lights"

    def test_preserves_trailing_punctuation_and_inner_casing(self):
        assert clean_query(", play some IDM in the kitchen!") == "Play some IDM in the kitchen!"

    def test_only_separators_becomes_empty(self):
        assert clean_query(" , . ") == ""


@pytest.mark.unit
class TestPrimingEcho:
    def test_exact_phrase_is_echo(self):
        for phrase in INITIAL_PROMPT_PHRASES:
            assert is_priming_echo(phrase) is True

    def test_match_ignores_punctuation_and_casing(self):
        phrase = INITIAL_PROMPT_PHRASES[0]
        assert is_priming_echo(phrase.lower()) is True
        assert is_priming_echo(f"  {phrase.upper()}!!  ") is True

    def test_real_command_is_not_echo(self):
        assert is_priming_echo("Michelle, turn on the kitchen lights") is False

    def test_empty_is_not_echo(self):
        assert is_priming_echo("") is False

    def test_identity_sentence_prefix_is_echo(self):
        # Whisper parroting only the prompt's opening sentence ("You are Michelle, our AI home
        # assistant.") must be caught too — that leading run opens with the wake word and was
        # false-firing in prod, even though it's not the whole phrase.
        for phrase in INITIAL_PROMPT_PHRASES:
            first_sentence = phrase.split(". ")[0] + "."
            assert is_priming_echo(first_sentence) is True

    def test_leading_run_is_echo(self):
        phrase = INITIAL_PROMPT_PHRASES[0]
        five_words = " ".join(phrase.split()[:5])  # a leading run mid-prompt
        assert is_priming_echo(five_words) is True

    def test_short_generic_prefix_is_not_echo(self):
        # A one/two-word prefix is too generic to attribute to the prompt — only longer
        # word-for-word echoes of its opening qualify.
        assert is_priming_echo("You are") is False

    def test_mid_prompt_command_examples_are_not_echoes(self):
        # The prompt lists genuine command examples mid-stream ("pause the music", "turn off
        # the wardrobe lights"). Echo detection matches a *leading* run from word one only, so
        # these embedded commands — exactly what a user really says — must NOT be flagged and
        # silently dropped. Guards the word-for-word-from-start semantics against a regression
        # to a naive substring match, which the existing tests (non-embedded commands, leading
        # runs) would not catch.
        prompt = " ".join(INITIAL_PROMPT_PHRASES).lower()
        for command in ["pause the music", "turn off the wardrobe lights",
                        "show me the camera feeds"]:
            assert command in prompt          # really is embedded mid-prompt
            assert is_priming_echo(command) is False  # ...but not a leading-run echo


@pytest.mark.unit
class TestMetricsRisk:
    def test_clean_metrics_not_risky(self):
        assert metrics_risky(CLEAN) is False

    def test_none_metrics_not_risky(self):
        assert metrics_risky(None) is False

    @pytest.mark.parametrize("override", [
        {"conf": 0.30},
        {"no_speech": 0.7},
        {"compression": 3.0},
        {"temperature": 0.4},
    ])
    def test_each_threshold_trips(self, override):
        assert metrics_risky({**CLEAN, **override}) is True

    def test_looser_confidence_floor_override_spares_low_confidence_clip(self):
        # A bare "Michelle" averages low on context alone (e.g. 33%) and trips the standard
        # floor. metrics_risky honours a looser floor passed by the caller (the wake-word path
        # passes WAKE_WORD_RISK_CONFIDENCE_MIN_PCT). This guards the override mechanism, not any
        # specific tuned value, so it stays valid however that constant is re-tuned.
        clip = {**CLEAN, "conf": 0.33}
        assert metrics_risky(clip) is True            # standard floor (RISK_CONFIDENCE_MIN_PCT = 40%)
        assert metrics_risky(clip, 25) is False       # a looser floor spares it

    def test_wake_word_floor_still_drops_truly_bad_confidence(self):
        from interfaces.voice.microphones.local_outposts.constants import WAKE_WORD_RISK_CONFIDENCE_MIN_PCT
        assert metrics_risky({**CLEAN, "conf": 0.20}, WAKE_WORD_RISK_CONFIDENCE_MIN_PCT) is True

    def test_wake_word_floor_keeps_other_gates(self):
        # Relaxing confidence must not disarm the silence/repetition gates.
        from interfaces.voice.microphones.local_outposts.constants import WAKE_WORD_RISK_CONFIDENCE_MIN_PCT
        assert metrics_risky({**CLEAN, "no_speech": 0.7}, WAKE_WORD_RISK_CONFIDENCE_MIN_PCT) is True

    def test_segment_metrics_aggregates_worst_segment(self):
        result = {"segments": [
            {"avg_logprob": -0.2, "no_speech_prob": 0.1, "compression_ratio": 1.0, "temperature": 0.0},
            {"avg_logprob": -0.4, "no_speech_prob": 0.5, "compression_ratio": 2.0, "temperature": 0.2},
        ]}
        m = segment_metrics(result)
        assert m["no_speech"] == 0.5 and m["compression"] == 2.0 and m["temperature"] == 0.2
        assert m["logprob"] == pytest.approx(-0.3)  # mean

    def test_segment_metrics_none_without_segments(self):
        assert segment_metrics({"segments": []}) is None

@pytest.mark.unit
class TestWakeWordHallucinationRisk:
    """The correlated cross-engine hallucination veto (8b7159ae): a wake-word fire is dropped
    only when it is BOTH low-confidence AND breathy. The whole point is the CONJUNCTION — each
    signal alone overlaps genuine short commands, so a regression to OR (or to either single
    gate) would either let the FP through or start killing real fires. These pin the AND."""

    def test_both_low_conf_and_breathy_is_risky(self):
        # The surviving Swedish FP signature: conf 37% / no_speech 0.196 — both bad at once.
        assert wake_word_hallucination_risky({**CLEAN, "conf": 0.37, "no_speech": 0.196}) is True

    def test_breathy_but_confident_is_not_risky(self):
        # A genuine breathy clip ("Michelle, turn off the TV." no_speech 0.193) stays confident,
        # so the conjunction must spare it even though no_speech alone is high.
        assert wake_word_hallucination_risky({**CLEAN, "conf": 0.80, "no_speech": 0.196}) is False

    def test_low_conf_but_speechy_is_not_risky(self):
        # A genuinely unsure command still sounds like speech (low no_speech) — spared.
        assert wake_word_hallucination_risky({**CLEAN, "conf": 0.33, "no_speech": 0.10}) is False

    def test_widened_box_catches_the_low_no_speech_fps(self):
        # A run of production false activations (both engines hallucinating a short
        # "Michelle, …" command from Swedish background speech) sit OUTSIDE the
        # original (0.17, 45) box — their no_speech is too low — and are exactly what the
        # widening to (0.10, 42) exists to catch. Narrowing either threshold
        # back resurrects them.
        assert wake_word_hallucination_risky({**CLEAN, "conf": 0.382, "no_speech": 0.132}) is True
        assert wake_word_hallucination_risky({**CLEAN, "conf": 0.408, "no_speech": 0.123}) is True

    def test_nearest_genuine_fire_survives_the_widened_box(self):
        # "Michelle volume down 5" (conf 42.2% / no_speech 0.134) is the genuine fire
        # closest to the box — 1.4pp above the conf ceiling. Raising the ceiling past it
        # starts killing real commands; this pins that thin margin.
        assert wake_word_hallucination_risky({**CLEAN, "conf": 0.422, "no_speech": 0.134}) is False

    def test_none_metrics_not_risky(self):
        assert wake_word_hallucination_risky(None) is False

    def test_thresholds_are_strict_inequalities_at_the_edge(self):
        # no_speech must EXCEED the floor and conf must be strictly BELOW the ceiling; a clip
        # sitting exactly on either boundary is not vetoed (the gaps are centred off the edges).
        from interfaces.voice.microphones.local_outposts.constants import (
            WAKE_WORD_HALLUCINATION_NO_SPEECH_MIN, WAKE_WORD_HALLUCINATION_CONFIDENCE_MAX_PCT,
        )
        on_no_speech_edge = {**CLEAN, "conf": 0.30,
                             "no_speech": WAKE_WORD_HALLUCINATION_NO_SPEECH_MIN}
        on_conf_edge = {**CLEAN, "no_speech": 0.50,
                        "conf": WAKE_WORD_HALLUCINATION_CONFIDENCE_MAX_PCT / 100}
        assert wake_word_hallucination_risky(on_no_speech_edge) is False
        assert wake_word_hallucination_risky(on_conf_edge) is False


@pytest.mark.unit
class TestNormalizeLoudness:
    """Peak-normalization run on every segment before Whisper decode. The win is on
    quiet captures (Whisper mis-hears soft speech), and the guard is what keeps near-silent
    noise from being amplified into a phantom signal."""

    def test_scales_quiet_audio_up_to_target_peak(self):
        audio = np.array([0.1, -0.05, 0.08], dtype=np.float32)  # peak 0.1
        out = normalize_loudness(audio, target_peak=0.95)
        assert float(np.abs(out).max()) == pytest.approx(0.95)

    def test_preserves_relative_shape_and_sign(self):
        audio = np.array([0.2, -0.1, 0.05], dtype=np.float32)
        out = normalize_loudness(audio, target_peak=0.95)
        # Uniform gain → every sample scaled by the same factor, signs intact.
        np.testing.assert_allclose(out, audio * (0.95 / 0.2), rtol=1e-6)

    def test_near_silent_buffer_left_untouched(self):
        # Below the 1e-4 guard: pure noise must not be blown up into a phantom signal.
        audio = np.array([5e-5, -3e-5, 1e-5], dtype=np.float32)
        out = normalize_loudness(audio)
        np.testing.assert_array_equal(out, audio)

    def test_empty_array_returned_unchanged(self):
        audio = np.array([], dtype=np.float32)
        out = normalize_loudness(audio)
        assert out.size == 0

    def test_output_is_float32(self):
        audio = np.array([0.3, -0.2], dtype=np.float64)
        out = normalize_loudness(audio)
        assert out.dtype == np.float32


@pytest.mark.unit
class TestMuteWatchdog:
    """A muted mic that never receives the server's unmute ('listen'/'wakeword') would
    stay muted forever and silently drop everything; the watchdog auto-recovers it."""

    def test_on_listen_clears_mute_time(self):
        outpost, _ = make_outpost()
        outpost.on_mute()
        outpost.on_listen()
        assert outpost.muted is False
        assert outpost._muted_at is None

    def test_on_wakeword_clears_mute_time(self):
        outpost, _ = make_outpost()
        outpost.on_mute()
        outpost.on_wakeword()
        assert outpost.muted is False
        assert outpost._muted_at is None

    def test_watchdog_auto_unmutes_after_timeout(self):
        from interfaces.voice.microphones.local_outposts.constants import MUTE_WATCHDOG_TIMEOUT
        outpost, _ = make_outpost(awaiting_wakeword=False, muted=True)
        outpost._muted_at = time.time() - (MUTE_WATCHDOG_TIMEOUT + 1.0)
        outpost._check_mute_watchdog()
        assert outpost.muted is False
        assert outpost.awaiting_wakeword is True   # reverts to requiring the wake word
        assert outpost._muted_at is None

    def test_watchdog_keeps_mute_within_timeout(self):
        outpost, _ = make_outpost(awaiting_wakeword=False, muted=True)
        outpost._muted_at = time.time()            # just muted
        outpost._check_mute_watchdog()
        assert outpost.muted is True

@pytest.mark.unit
class TestStreamWatchdog:
    """PortAudio can silently stop delivering frames (USB/CoreAudio hiccup); the capture
    loop then waits on an empty queue forever and the mic goes deaf with no error. The
    stream watchdog detects the frame drought and the supervisor reopens the stream."""

    def test_audio_callback_stamps_last_frame(self):
        outpost, _ = make_outpost()
        outpost._last_frame_at = 0.0
        outpost._audio_callback(np.zeros((1600, 1), np.float32), 1600, None, None)
        assert outpost._last_frame_at > 0.0
        assert not outpost._audio_q.empty()

    def test_watchdog_raises_when_no_frames_for_timeout(self):
        from interfaces.voice.microphones.local_outposts.constants import STREAM_SILENCE_TIMEOUT
        outpost, _ = make_outpost()
        outpost._last_frame_at = time.time() - (STREAM_SILENCE_TIMEOUT + 1.0)
        with pytest.raises(_StreamStalled):
            outpost._check_stream_watchdog()

    def test_watchdog_silent_while_frames_flow(self):
        outpost, _ = make_outpost()
        outpost._last_frame_at = time.time()       # a frame just arrived
        outpost._check_stream_watchdog()           # must not raise

    def test_supervisor_reopens_stream_on_stall(self):
        outpost, _ = make_outpost()
        outpost._vad = MagicMock()
        outpost._vad.name = "silero"
        outpost._energy_vad = MagicMock()
        outpost._energy_vad.threshold = 0.01       # already calibrated → skip ambient calibration in _run
        outpost._open_stream = MagicMock()
        outpost._close_stream = MagicMock()

        calls = {"n": 0}

        def fake_loop():
            calls["n"] += 1
            if calls["n"] == 1:
                raise _StreamStalled()             # first pass: stream looks dead
            outpost._stop.set()                    # second pass: end the supervisor

        outpost._endpoint_loop = fake_loop
        outpost._run()

        outpost._close_stream.assert_called_once()
        outpost._open_stream.assert_called_once()

    def test_endpoint_loop_entry_clears_stuck_capturing_speech(self):
        # A crash or stall mid-utterance restarts _endpoint_loop with fresh locals, but
        # capturing_speech is instance state: left True, it silently disables the GPU
        # keep-warm until this mic completes another utterance.
        outpost, _ = make_outpost()
        outpost.capturing_speech = True
        outpost._stop.set()
        outpost._endpoint_loop()
        assert outpost.capturing_speech is False


@pytest.mark.unit
class TestCaptureDropDetection:
    """CoreAudio drops input samples when the audio callback is starved (the gap the user
    hears mid-recording). PortAudio's status flag is unreliable on macOS, so drops are
    detected from the hardware ADC capture clock: consecutive callbacks should advance
    inputBufferAdcTime by exactly FRAME_DURATION; any excess is dropped audio."""

    @staticmethod
    def _ti(adc):
        # Mimics the PortAudio CFFI time struct sounddevice passes to the callback.
        return SimpleNamespace(inputBufferAdcTime=adc, currentTime=adc, outputBufferDacTime=adc)

    def _feed(self, outpost, adc, status=None):
        outpost._audio_callback(np.zeros((1600, 1), np.float32), 1600, self._ti(adc), status)

    def test_no_drop_when_adc_advances_normally(self):
        outpost, _ = make_outpost()
        for adc in (0.0, 0.1, 0.2, 0.3, 0.4):      # contiguous 100 ms buffers
            self._feed(outpost, adc)
        assert outpost._drop_count == 0
        assert outpost._total_dropped_s == 0.0
        assert outpost._pending_drops == []

    def test_detects_drop_when_adc_clock_jumps(self):
        outpost, _ = make_outpost()
        for adc in (0.0, 0.1, 0.2):                # prime past the warm-up window
            self._feed(outpost, adc)
        self._feed(outpost, 1.5)                   # 1.3 s of device time, only 0.1 s expected
        assert outpost._drop_count == 1
        assert outpost._total_dropped_s == pytest.approx(1.2, abs=1e-6)
        assert len(outpost._pending_drops) == 1
        assert outpost._pending_drops[0][1] == pytest.approx(1.2, abs=1e-6)

    def test_subthreshold_jitter_is_not_a_drop(self):
        outpost, _ = make_outpost()
        for adc in (0.0, 0.1, 0.2):
            self._feed(outpost, adc)
        self._feed(outpost, 0.34)                  # +0.14 advance → 0.04 s hole, under the floor
        assert outpost._drop_count == 0

    def test_warmup_callbacks_are_ignored(self):
        outpost, _ = make_outpost()
        self._feed(outpost, 0.0)
        self._feed(outpost, 5.0)                   # huge jump but within warm-up → ignored
        assert outpost._drop_count == 0

    def test_none_time_info_does_not_crash(self):
        # Backends that don't populate the ADC clock (and the watchdog test's None) must
        # degrade to "no detection", not raise.
        outpost, _ = make_outpost()
        outpost._audio_callback(np.zeros((1600, 1), np.float32), 1600, None, None)
        assert outpost._drop_count == 0
        assert not outpost._audio_q.empty()

    def test_status_flag_recorded_but_not_counted_as_drop(self):
        outpost, _ = make_outpost()
        self._feed(outpost, 0.0)
        self._feed(outpost, 0.1)
        self._feed(outpost, 0.2, status="input overflow")   # flag set, but ADC is contiguous
        assert outpost._drop_count == 0
        assert outpost._pending_status == ["input overflow"]

    def test_baseline_reset_on_stream_reopen_skips_false_gap(self, monkeypatch):
        import sounddevice as sd
        monkeypatch.setattr(sd, "InputStream", lambda *a, **k: MagicMock())
        outpost, _ = make_outpost()
        for adc in (0.0, 0.1, 0.2):
            self._feed(outpost, adc)
        outpost._open_stream()                     # a reopen restarts the ADC clock near 0
        assert outpost._last_adc is None and outpost._cb_count == 0
        for adc in (0.0, 0.1, 0.2, 0.3):           # fresh contiguous stream — no false drop
            self._feed(outpost, adc)
        assert outpost._drop_count == 0

    def test_startup_grace_suppresses_drop(self):
        # A drop in the first CAPTURE_DROP_STARTUP_GRACE seconds after the mic starts is a
        # boot-time false positive (model-load contention starves the callback), not a gap.
        outpost, _ = make_outpost()
        outpost._started_at = time.time()
        for adc in (0.0, 0.1, 0.2):                # prime past the warm-up window
            self._feed(outpost, adc)
        self._feed(outpost, 1.5)                   # 1.3 s jump → would be a 1.2 s drop
        assert outpost._drop_count == 0
        assert outpost._total_dropped_s == 0.0
        assert outpost._pending_drops == []
        assert outpost._suppressed_drop_count == 1
        assert outpost._pending_suppressed == [pytest.approx(1.2, abs=1e-6)]

    def test_drop_counted_after_startup_grace(self):
        outpost, _ = make_outpost()
        outpost._started_at = time.time() - (CAPTURE_DROP_STARTUP_GRACE + 1)
        for adc in (0.0, 0.1, 0.2):
            self._feed(outpost, adc)
        self._feed(outpost, 1.5)
        assert outpost._drop_count == 1
        assert outpost._suppressed_drop_count == 0
        assert len(outpost._pending_drops) == 1

    def test_drain_logs_suppressed_drop_to_verbose(self, monkeypatch):
        import interfaces.voice.microphones.local_outposts.LocalOutpost as mod
        fines, errors = [], []
        monkeypatch.setattr(mod, "fine", lambda m: fines.append(m))
        monkeypatch.setattr(mod, "error", lambda m: errors.append(m))
        outpost, _ = make_outpost()
        outpost._started_at = time.time()
        for adc in (0.0, 0.1, 0.2):
            self._feed(outpost, adc)
        self._feed(outpost, 1.5)
        outpost._drain_capture_drops()
        assert errors == []                        # no AUDIO DROP error during startup grace
        assert any("startup grace" in f for f in fines)
        assert outpost._pending_suppressed == []   # drained

    def test_drain_logs_error_and_clears_pending(self, monkeypatch):
        import interfaces.voice.microphones.local_outposts.LocalOutpost as mod
        errors = []
        monkeypatch.setattr(mod, "error", lambda m: errors.append(m))
        outpost, _ = make_outpost()
        for adc in (0.0, 0.1, 0.2):
            self._feed(outpost, adc)
        self._feed(outpost, 1.5)
        outpost._drain_capture_drops()
        assert len(errors) == 1 and "AUDIO DROP" in errors[0]
        assert outpost._pending_drops == []        # drained
        assert outpost._drop_count == 1            # cumulative total persists


@pytest.mark.unit
class TestTranscriptGate:
    def test_muted_drops_transcript(self):
        outpost, client = make_outpost(awaiting_wakeword=False, muted=True)
        outpost._on_transcript("turn on the lights", CLEAN, AUDIO)
        client.send_query.assert_not_called()

    def test_risky_transcript_dropped(self):
        outpost, client = make_outpost(awaiting_wakeword=False)
        outpost._on_transcript("turn on the lights", {**CLEAN, "no_speech": 0.9}, AUDIO)
        client.send_query.assert_not_called()

    def test_priming_echo_dropped(self):
        outpost, client = make_outpost(awaiting_wakeword=True)
        outpost._on_transcript(INITIAL_PROMPT_PHRASES[0], CLEAN, AUDIO)
        client.send_query.assert_not_called()
        client.notify_wake_word.assert_not_called()

    def test_identity_sentence_echo_does_not_fire(self):
        # The prod false positive: Whisper parrots its prompt's identity sentence, the wake
        # word at the front would otherwise fire. It must be dropped as a priming echo.
        outpost, client = make_outpost(awaiting_wakeword=True, languages=("en-US",))
        outpost._on_transcript("You are Michelle, our AI home assistant.", CLEAN, AUDIO, language="en")
        client.send_query.assert_not_called()
        client.notify_wake_word.assert_not_called()

    def test_wake_word_folds_signal_into_query_send(self):
        outpost, client = make_outpost(awaiting_wakeword=True)
        outpost._on_transcript("Michelle turn on the lights", CLEAN, AUDIO)
        # No separate /wake_word_detected call — it's folded into the /stream send.
        client.notify_wake_word.assert_not_called()
        client.send_query.assert_called_once()
        sent_query, sent_lang = client.send_query.call_args[0][0], client.send_query.call_args[0][1]
        assert sent_query == "Turn on the lights" and sent_lang == "sv-SE"  # cleaned + capitalized
        assert client.send_query.call_args.kwargs.get("wake_word") is True
        assert outpost.awaiting_wakeword is False

    def test_wake_word_only_uses_standalone_notify(self):
        outpost, client = make_outpost(awaiting_wakeword=True)
        outpost._on_transcript("Michelle", CLEAN, AUDIO)
        client.notify_wake_word.assert_called_once()
        client.send_query.assert_not_called()

    def test_doubled_wake_word_dropped_as_hallucination(self):
        # A production false-positive decode shape ("Michelle Michelle, …" from Swedish household
        # speech): even a clean-metrics doubled fire must not open the wake word.
        outpost, client = make_outpost(awaiting_wakeword=True)
        outpost._on_transcript("Michelle Michelle, is it over there?", CLEAN, AUDIO)
        client.send_query.assert_not_called()
        client.notify_wake_word.assert_not_called()
        assert outpost.awaiting_wakeword is True

    def test_low_confidence_wake_word_admitted_at_relaxed_floor(self):
        # The wake-word confidence floor was relaxed (constants.py: 40%→25%) now that Scribe
        # verifies every Whisper wake-word fire: a bare "Michelle." at conf 33% — above the
        # 25% floor — is admitted and opens the listen window, where it would once have been
        # dropped as a likely hallucination. Scribe vetoes any false fire this lets through.
        outpost, client = make_outpost(awaiting_wakeword=True)
        outpost._on_transcript("Michelle.", {**CLEAN, "logprob": -1.11, "conf": 0.33}, AUDIO)
        client.notify_wake_word.assert_called_once()
        assert outpost.awaiting_wakeword is False

    def test_very_low_confidence_wake_word_still_dropped(self):
        # Below the relaxed floor (conf 20% < 25%) a wake word is still dropped as a likely
        # hallucination and the mic keeps waiting.
        outpost, client = make_outpost(awaiting_wakeword=True)
        outpost._on_transcript("Michelle.", {**CLEAN, "logprob": -1.6, "conf": 0.20}, AUDIO)
        client.notify_wake_word.assert_not_called()
        assert outpost.awaiting_wakeword is True

    def test_low_confidence_non_wake_word_still_dropped(self):
        # The relaxed floor is wake-word-only: a low-confidence query gets no reprieve.
        outpost, client = make_outpost(awaiting_wakeword=False)
        outpost._on_transcript("turn on the lights", {**CLEAN, "logprob": -1.11, "conf": 0.33}, AUDIO)
        client.send_query.assert_not_called()

    def test_follow_up_send_has_no_wake_word_flag(self):
        outpost, client = make_outpost(awaiting_wakeword=False)
        outpost._on_transcript("turn on the lights", CLEAN, AUDIO)
        assert client.send_query.call_args.kwargs.get("wake_word") in (False, None)

    def test_no_wake_word_while_awaiting_drops(self):
        outpost, client = make_outpost(awaiting_wakeword=True)
        outpost._on_transcript("turn on the lights", CLEAN, AUDIO)
        client.send_query.assert_not_called()
        client.notify_wake_word.assert_not_called()

    def test_forwards_speculative_identify_future_on_follow_up(self):
        # The future started at the speculative pause must reach send_query so the
        # GET waits on it instead of identifying inline.
        outpost, client = make_outpost(awaiting_wakeword=False)
        fut = object()
        outpost._on_transcript("turn on the lights", CLEAN, AUDIO, identify_future=fut)
        assert client.send_query.call_args.kwargs.get("identify_future") is fut

    def test_forwards_speculative_identify_future_on_wake_word(self):
        outpost, client = make_outpost(awaiting_wakeword=True)
        fut = object()
        outpost._on_transcript("Michelle turn on the lights", CLEAN, AUDIO, language="sv", identify_future=fut)
        assert client.send_query.call_args.kwargs.get("identify_future") is fut

    def test_detected_language_routes_to_its_locale(self):
        # Bilingual outpost: the reply locale (and so the TTS voice) follows whatever
        # Whisper detected for this utterance, not a fixed per-mic language.
        outpost, client = make_outpost(awaiting_wakeword=False, languages=("en-US", "sv-SE"))
        outpost._on_transcript("släck lamporna", CLEAN, AUDIO, language="sv")
        assert client.send_query.call_args[0][1] == "sv-SE"
        outpost._on_transcript("turn on the lights", CLEAN, AUDIO, language="en")
        assert client.send_query.call_args[0][1] == "en-US"

    def test_out_of_set_language_dropped(self):
        # Whisper auto-detect ranges over all languages; a clip it reads as Norwegian
        # (a misdetection, or a genuine foreign speaker) is ignored, never sent.
        outpost, client = make_outpost(awaiting_wakeword=False, languages=("en-US", "sv-SE"))
        outpost._on_transcript("skru på lyset", CLEAN, AUDIO, language="no")
        client.send_query.assert_not_called()

    def test_missing_detection_falls_back_to_primary(self):
        outpost, client = make_outpost(awaiting_wakeword=False, languages=("en-US", "sv-SE"))
        outpost._on_transcript("turn on the lights", CLEAN, AUDIO, language=None)
        assert client.send_query.call_args[0][1] == "en-US"

    # ── Scribe is gated behind an accepted Whisper send (SCRIBE_VERIFY_WAKE_WORD) ──
    # MID_CONF sits below the SCRIBE_SKIP_CONFIDENCE_PCT skip floor (61%) but above every
    # risk floor, so Scribe still runs for these fires.
    MID_CONF = {**CLEAN, "conf": 0.5, "logprob": -0.69}

    def test_scribe_fired_only_after_wake_word_accept(self):
        # Whisper accepts a wake-word command → Scribe runs once, on that audio, as a verifier.
        outpost, client = make_outpost(awaiting_wakeword=True)
        outpost._scribe = MagicMock()
        outpost._on_transcript("Michelle turn on the lights", self.MID_CONF, AUDIO)
        outpost._scribe.submit.assert_called_once()
        assert outpost._scribe.submit.call_args[0][0] is AUDIO   # the final audio, not a partial
        assert client.send_query.call_args.kwargs.get("scribe_skipped") is False

    def test_scribe_not_fired_when_whisper_drops(self):
        # No wake word while awaiting one → Whisper drops it → Scribe (paid) is never called.
        outpost, client = make_outpost(awaiting_wakeword=True)
        outpost._scribe = MagicMock()
        outpost._on_transcript("turn on the lights", CLEAN, AUDIO)
        outpost._scribe.submit.assert_not_called()

    def test_scribe_verifies_bare_wake_word(self):
        # A bare wake word arms a listen window, which used to go unverified (a production
        # FP path: hallucinated "Michelle." → 15s of listening → background phrase became a
        # query). Scribe now verifies the fire just like a command fire.
        outpost, client = make_outpost(awaiting_wakeword=True)
        outpost._scribe = MagicMock()
        outpost._on_transcript("Michelle", CLEAN, AUDIO)
        outpost._scribe.submit.assert_called_once()
        assert outpost._scribe.submit.call_args[0][0] is AUDIO

    def test_bare_wake_word_veto_unarms_listen_window(self):
        # Scribe hears no wake word in the bare fire → the fire was a hallucination: the
        # listen window is un-armed (back to awaiting the wake word) and no server request
        # is killed — there never was one (whisper_query is empty).
        outpost, client = make_outpost(awaiting_wakeword=True)
        outpost._scribe = MagicMock()
        outpost._on_transcript("Michelle", CLEAN, AUDIO)
        assert outpost.awaiting_wakeword is False   # window armed on Whisper's fire
        callback = outpost._scribe.submit.call_args[0][2]
        callback("oh well", 0.64, "en")             # Scribe hears no wake word
        assert outpost.awaiting_wakeword is True    # window un-armed by the veto
        client.send_false_positive.assert_not_called()

    def test_bare_wake_word_confirmed_keeps_listening(self):
        # Scribe hears the bare wake word too → genuine attention call: the window stays
        # armed for the follow-up command, and nothing is sent or vetoed (the bare-echo
        # agreement veto needs a non-empty whisper command, so bare-bare is confirmation).
        outpost, client = make_outpost(awaiting_wakeword=True)
        outpost._scribe = MagicMock()
        outpost._on_transcript("Michelle", CLEAN, AUDIO)
        callback = outpost._scribe.submit.call_args[0][2]
        callback("Michelle", 0.9, "en")
        assert outpost.awaiting_wakeword is False
        client.send_false_positive.assert_not_called()
        client.send_query.assert_not_called()

    def test_scribe_fired_on_follow_up_as_pure_vote(self):
        # Inside a listen window a follow-up is accepted → Scribe still runs as an accuracy
        # vote, but it can never veto (no wake word to disagree about).
        outpost, client = make_outpost(awaiting_wakeword=False)
        outpost._scribe = MagicMock()
        outpost._on_transcript("turn on the lights", self.MID_CONF, AUDIO)
        outpost._scribe.submit.assert_called_once()
        callback = outpost._scribe.submit.call_args[0][2]
        callback("turn on the lights", 0.9, "en")   # Scribe returns, no wake word
        client.send_false_positive.assert_not_called()

    def test_wake_word_verifier_callback_vetoes_on_disagreement(self):
        # End-to-end wiring: the verifier callback _on_transcript hands Scribe, when Scribe
        # doesn't hear the wake word, kills the request for the exact phrase Whisper sent.
        outpost, client = make_outpost(awaiting_wakeword=True)
        outpost._scribe = MagicMock()
        outpost._on_transcript("Michelle turn on the lights", self.MID_CONF, AUDIO)
        callback = outpost._scribe.submit.call_args[0][2]
        callback("some unrelated chatter", 0.9, "en")   # Scribe: no wake word here
        client.send_false_positive.assert_called_once()
        assert client.send_false_positive.call_args[0][0] == "Turn on the lights"

    # ── High-confidence skip (SCRIBE_SKIP_CONFIDENCE_PCT): in two months of prod no fire
    # at ≥52% Whisper confidence was ever Scribe-vetoed, and above 1/SCRIBE_CONF_CALIBRATION
    # (≈60.6%) Whisper wins the transcript vote regardless of Scribe — so ≥61% skips Scribe
    # entirely and flags the send so the server's verdict gate doesn't hold the reply. ──
    def test_scribe_skipped_on_high_confidence_wake_word(self):
        outpost, client = make_outpost(awaiting_wakeword=True)
        outpost._scribe = MagicMock()
        outpost._on_transcript("Michelle turn on the lights", CLEAN, AUDIO)   # conf 0.8
        outpost._scribe.submit.assert_not_called()
        client.send_query.assert_called_once()
        assert client.send_query.call_args.kwargs.get("scribe_skipped") is True

    def test_scribe_skipped_on_high_confidence_follow_up(self):
        outpost, client = make_outpost(awaiting_wakeword=False)
        outpost._scribe = MagicMock()
        outpost._on_transcript("turn on the lights", CLEAN, AUDIO)   # conf 0.8
        outpost._scribe.submit.assert_not_called()
        assert client.send_query.call_args.kwargs.get("scribe_skipped") is True

    def test_scribe_skips_at_exactly_the_floor(self):
        # The floor is inclusive: exactly 61% skips (>= comparison)…
        outpost, client = make_outpost(awaiting_wakeword=True)
        outpost._scribe = MagicMock()
        outpost._on_transcript("Michelle turn on the lights",
                               {**CLEAN, "conf": 0.61, "logprob": -0.49}, AUDIO)
        outpost._scribe.submit.assert_not_called()
        assert client.send_query.call_args.kwargs.get("scribe_skipped") is True

    def test_scribe_runs_just_below_the_floor(self):
        # …and just below it (the 60.006% "room vs outdoor temperature" clip that motivated
        # the 61 floor) Scribe still verifies.
        outpost, client = make_outpost(awaiting_wakeword=True)
        outpost._scribe = MagicMock()
        outpost._on_transcript("Michelle turn on the lights",
                               {**CLEAN, "conf": 0.60, "logprob": -0.51}, AUDIO)
        outpost._scribe.submit.assert_called_once()
        assert client.send_query.call_args.kwargs.get("scribe_skipped") is False

    def test_missing_metrics_never_skip_scribe(self):
        # No confidence available → can't justify skipping the verifier.
        outpost, client = make_outpost(awaiting_wakeword=False)
        outpost._scribe = MagicMock()
        outpost._on_transcript("turn on the lights", None, AUDIO)
        outpost._scribe.submit.assert_called_once()
        assert client.send_query.call_args.kwargs.get("scribe_skipped") is False


@pytest.fixture
def logged(monkeypatch):
    """Every message logged while the test runs, at any level."""
    lines = []
    monkeypatch.setattr("common.logging.log", lambda message, *args, **kwargs: lines.append(message))
    return lines


@pytest.mark.unit
class TestOverheardTranscriptLogging:
    OVERHEARD = "we should call the dentist tomorrow"

    def _allow_overheard(self, monkeypatch, allowed):
        from interfaces.voice.microphones.local_outposts import constants
        monkeypatch.setattr(constants, "LOG_OVERHEARD_TRANSCRIPTS", allowed)

    def test_speech_without_wake_word_keeps_its_words_out_of_the_logs(self, logged, monkeypatch):
        self._allow_overheard(monkeypatch, False)
        outpost, client = make_outpost(awaiting_wakeword=True)
        outpost._on_transcript(self.OVERHEARD, CLEAN, AUDIO, vad_stop_at=time.time())
        client.send_query.assert_not_called()
        assert logged, "the drop itself is still logged"
        assert not any("dentist" in line for line in logged)

    def test_muted_overheard_speech_keeps_its_words_out_of_the_logs(self, logged, monkeypatch):
        self._allow_overheard(monkeypatch, False)
        outpost, _ = make_outpost(awaiting_wakeword=True, muted=True)
        outpost._on_transcript(self.OVERHEARD, CLEAN, AUDIO)
        assert logged and not any("dentist" in line for line in logged)

    def test_mid_sentence_wake_word_mention_is_still_overheard(self, logged, monkeypatch):
        self._allow_overheard(monkeypatch, False)
        outpost, _ = make_outpost(awaiting_wakeword=True)
        outpost._on_transcript("the dentist asked whether Michelle works", CLEAN, AUDIO)
        assert logged and not any("dentist" in line for line in logged)

    def test_overheard_language_override_is_logged_without_its_words(self, logged, monkeypatch):
        self._allow_overheard(monkeypatch, False)
        det = TextLanguageDetector({"en", "sv"})
        outpost, _ = make_outpost(awaiting_wakeword=True, languages=("en-US", "sv-SE"),
                                  lang_detector=det)
        outpost._on_transcript("Vi borde ringa tandläkaren i morgon bitti.", CLEAN, AUDIO, language="en")
        assert any("arbitration" in line for line in logged)
        assert not any("tandläkaren" in line for line in logged)

    def test_wake_word_utterance_is_logged_in_clear(self, logged, monkeypatch):
        self._allow_overheard(monkeypatch, False)
        outpost, _ = make_outpost(awaiting_wakeword=True)
        outpost._on_transcript("Michelle call the dentist", CLEAN, AUDIO, vad_stop_at=time.time())
        assert any("call the dentist" in line for line in logged)

    def test_wake_word_fire_the_gate_drops_is_logged_in_clear(self, logged, monkeypatch):
        self._allow_overheard(monkeypatch, False)
        outpost, client = make_outpost(awaiting_wakeword=True)
        outpost._on_transcript("Michelle call the dentist", {**CLEAN, "no_speech": 0.9}, AUDIO)
        client.send_query.assert_not_called()
        assert any("call the dentist" in line for line in logged)

    def test_follow_up_in_the_listen_window_is_logged_in_clear(self, logged, monkeypatch):
        self._allow_overheard(monkeypatch, False)
        outpost, _ = make_outpost(awaiting_wakeword=False)
        outpost._on_transcript(self.OVERHEARD, CLEAN, AUDIO, vad_stop_at=time.time())
        assert any("dentist" in line for line in logged)

    def test_flag_on_logs_overheard_speech_in_clear(self, logged, monkeypatch):
        self._allow_overheard(monkeypatch, True)
        outpost, _ = make_outpost(awaiting_wakeword=True)
        outpost._on_transcript(self.OVERHEARD, CLEAN, AUDIO, vad_stop_at=time.time())
        assert any("dentist" in line for line in logged)

    def test_whisper_worker_logs_a_decode_without_its_words(self, logged, monkeypatch):
        # The worker decodes before the gate, so it cannot know whether the text is addressed.
        import interfaces.voice.microphones.local_outposts.WhisperWorker as mod
        self._allow_overheard(monkeypatch, False)
        monkeypatch.setattr(mod, "transcribe",
                            lambda *args, **kwargs: (self.OVERHEARD, "en", 0.1, None, 0.2, CLEAN))
        delivered = []
        worker = mod.WhisperWorker("repo", "en")
        worker._run_final(mod._Job("final", np.zeros(8000, np.float32),
                                   on_result=lambda text, *rest: delivered.append(text), label="m1"))
        assert delivered == [self.OVERHEARD]
        assert logged and not any("dentist" in line for line in logged)


@pytest.mark.unit
class TestWakeWordPrefixConfidenceGate:
    """A wake-word fire behind a spoken lead-in ("Okay, Michelle…") must clear
    WAKE_WORD_PREFIX_CONFIDENCE_MIN_PCT — 48 of 49 such prod fires were junk. Flush
    fires and punctuation-only lead-ins keep the ordinary floors."""

    def test_prefixed_fire_below_floor_dropped(self):
        # A production false activation: spoken prefix at 50% confidence — under the
        # 55% floor, so nothing fires and the mic keeps waiting for a real address.
        outpost, client = make_outpost(awaiting_wakeword=True)
        outpost._on_transcript("Okay, Michelle, that is all for now.",
                               {**CLEAN, "conf": 0.50}, AUDIO)
        client.send_query.assert_not_called()
        client.notify_wake_word.assert_not_called()
        assert outpost.awaiting_wakeword is True

    def test_prefixed_fire_above_floor_admitted(self):
        # A genuine prefixed fire ("Who's there? Michelle, open the blinds.")
        # scored 56.6% — decisive confidence must still rescue a prefixed address.
        outpost, client = make_outpost(awaiting_wakeword=True)
        outpost._on_transcript("Who's there? Michelle, open the blinds.",
                               {**CLEAN, "conf": 0.566}, AUDIO)
        client.send_query.assert_called_once()
        assert client.send_query.call_args.kwargs.get("wake_word") is True
        assert outpost.awaiting_wakeword is False

    def test_punctuation_only_lead_in_is_exempt(self):
        # A stray dash/quote before the wake word is decoder noise, not spoken words —
        # the same 50% confidence that kills a spoken prefix fires here.
        outpost, client = make_outpost(awaiting_wakeword=True)
        outpost._on_transcript("— Michelle, turn on the lights",
                               {**CLEAN, "conf": 0.50}, AUDIO)
        client.send_query.assert_called_once()

    def test_flush_fire_keeps_ordinary_floor(self):
        # No prefix → the 55% floor must not apply; 50% clears the ordinary wake-word
        # risk floor and fires.
        outpost, client = make_outpost(awaiting_wakeword=True)
        outpost._on_transcript("Michelle, turn on the lights",
                               {**CLEAN, "conf": 0.50}, AUDIO)
        client.send_query.assert_called_once()

    def test_prefixed_fire_without_metrics_fails_closed(self):
        # No confidence data at all (segment_metrics can return None) → a prefixed
        # fire can't prove it clears the floor, so it must be dropped.
        outpost, client = make_outpost(awaiting_wakeword=True)
        outpost._on_transcript("Okay, Michelle, turn on the lights", None, AUDIO)
        client.send_query.assert_not_called()
        client.notify_wake_word.assert_not_called()

@pytest.mark.unit
class TestSttLatencyMeasurement:
    """stt_ms is user-perceived STT latency: measured from speech_ended_at (VAD stop
    minus the trailing-silence wait), falling back to vad_stop_at for callers that
    don't provide it. The endpoint share travels separately as stt_wait_ms."""

    def test_stt_ms_measured_from_speech_end_not_vad_stop(self):
        outpost, client = make_outpost(awaiting_wakeword=False)
        now = time.time()
        outpost._on_transcript("turn on the lights", CLEAN, AUDIO,
                               vad_stop_at=now - 0.4, speech_ended_at=now - 2.0)
        stt_ms = client.send_query.call_args.kwargs["stt_ms"]
        # ~2000ms from speech end; measuring from VAD stop would give ~400ms.
        assert 2000 <= stt_ms < 3000

    def test_stt_ms_falls_back_to_vad_stop(self):
        outpost, client = make_outpost(awaiting_wakeword=False)
        outpost._on_transcript("turn on the lights", CLEAN, AUDIO,
                               vad_stop_at=time.time() - 0.4)
        stt_ms = client.send_query.call_args.kwargs["stt_ms"]
        assert 400 <= stt_ms < 1400

    def test_stt_wait_ms_is_endpoint_share_of_breakdown(self):
        # endpoint = decode submitted 0.5s after speech ended → stt_wait_ms == 500,
        # deterministic because both stamps are supplied.
        outpost, client = make_outpost(awaiting_wakeword=False)
        now = time.time()
        timing = {"submitted_at": now - 1.5, "started_at": now - 1.2, "done_at": now - 0.3}
        outpost._on_transcript("turn on the lights", CLEAN, AUDIO,
                               vad_stop_at=now - 1.6, speech_ended_at=now - 2.0,
                               timing=timing)
        assert client.send_query.call_args.kwargs["stt_wait_ms"] == 500

    def test_trigger_forwarded_when_speculative_decode_won(self):
        # The energy gate fired the speculative and its decode produced the text →
        # the Silence detect row is attributed to RMS.
        outpost, client = make_outpost(awaiting_wakeword=False)
        now = time.time()
        timing = {"submitted_at": now - 1.5, "started_at": now - 1.2, "done_at": now - 0.3,
                  "decode": "spec"}
        outpost._on_transcript("turn on the lights", CLEAN, AUDIO,
                               vad_stop_at=now - 1.6, speech_ended_at=now - 2.0,
                               timing=timing, spec_trigger="rms")
        assert client.send_query.call_args.kwargs["stt_trigger"] == "rms"

    def test_trigger_is_vad_when_final_decoded_fresh(self):
        # An RMS-fired speculative was cancelled/skipped — the fresh final started at
        # silero's VAD stop, so the spec's trigger must not be attributed to it.
        outpost, client = make_outpost(awaiting_wakeword=False)
        now = time.time()
        timing = {"submitted_at": now - 1.5, "started_at": now - 1.2, "done_at": now - 0.3,
                  "decode": "final"}
        outpost._on_transcript("turn on the lights", CLEAN, AUDIO,
                               vad_stop_at=now - 1.6, speech_ended_at=now - 2.0,
                               timing=timing, spec_trigger="rms")
        assert client.send_query.call_args.kwargs["stt_trigger"] == "vad"

@pytest.mark.unit
class TestSaveRecordingAsync:
    """The debug WAV write moved off the transcript hot path onto a daemon thread —
    the save itself and the INCOMPLETE-recording diagnostic must survive the move."""

    def _run_inline(self, monkeypatch):
        import interfaces.voice.microphones.local_outposts.LocalOutpost as mod

        class InlineThread:
            def __init__(self, target=None, **kwargs):
                self._target = target

            def start(self):
                self._target()

        monkeypatch.setattr(mod.threading, "Thread", InlineThread)
        return mod

    def test_nothing_is_saved_while_file_logging_is_disabled(self, monkeypatch):
        # The debug WAV is a log artefact: with file logging off (every unit test) no thread is
        # started and nothing lands in log/audio.
        mod = self._run_inline(monkeypatch)
        monkeypatch.setattr("common.logging.disable_file_logging", True)
        saved = []
        monkeypatch.setattr(mod, "save_pcm_wav", lambda *a, **k: saved.append(a) or "p.wav")
        outpost, _ = make_outpost()
        outpost._save_recording_async("turn on the lights", AUDIO, 0.0)
        assert saved == []

    def test_saves_pcm_converted_audio(self, monkeypatch, file_logging_enabled):
        mod = self._run_inline(monkeypatch)
        saved = []
        monkeypatch.setattr(mod, "save_pcm_wav",
                            lambda d, text, pcm, sample_rate, metadata=None: saved.append((text, pcm)) or "p.wav")
        outpost, _ = make_outpost()
        outpost._save_recording_async("turn on the lights", AUDIO, 0.0)
        assert len(saved) == 1
        text, pcm = saved[0]
        assert text == "turn on the lights"
        assert len(pcm) == 2 * len(AUDIO)   # float32 → int16 bytes

    def test_promoted_spec_saves_full_utterance_and_records_whisper_span(self, monkeypatch, file_logging_enabled):
        # A promoted speculative decoded only the utterance-so-far, while Scribe verifies
        # the whole utterance. The WAV must hold the whole utterance (Scribe's prod input)
        # and carry the prefix length Whisper saw, so WakeWordTest replays both engines
        # on their prod spans — saving just the spec span broke the Scribe half of replay.
        mod = self._run_inline(monkeypatch)
        saved = []
        monkeypatch.setattr(mod, "save_pcm_wav",
                            lambda d, text, pcm, sample_rate, metadata=None:
                                saved.append((pcm, metadata)) or "p.wav")
        outpost, _ = make_outpost()
        spec_audio = np.zeros(len(AUDIO) // 2, np.float32)
        now = time.time()
        outpost._on_transcript("turn on the lights", CLEAN, AUDIO, vad_stop_at=now,
                               timing={"submitted_at": now, "started_at": now, "done_at": now,
                                       "decode": "spec", "audio": spec_audio})
        assert len(saved) == 1
        pcm, metadata = saved[0]
        assert len(pcm) == 2 * len(AUDIO)
        assert metadata == {"whisper_span_samples": len(spec_audio), "decode": "spec"}

    def test_fresh_final_saves_full_utterance_without_span(self, monkeypatch, file_logging_enabled):
        mod = self._run_inline(monkeypatch)
        saved = []
        monkeypatch.setattr(mod, "save_pcm_wav",
                            lambda d, text, pcm, sample_rate, metadata=None:
                                saved.append((pcm, metadata)) or "p.wav")
        outpost, _ = make_outpost()
        now = time.time()
        outpost._on_transcript("turn on the lights", CLEAN, AUDIO, vad_stop_at=now,
                               timing={"submitted_at": now, "started_at": now, "done_at": now,
                                       "decode": "final", "audio": AUDIO})
        outpost._on_transcript("turn on the lights", CLEAN, AUDIO)
        assert [(len(pcm), m) for pcm, m in saved] == [(2 * len(AUDIO), None)] * 2

    def test_dropped_audio_flags_recording_incomplete(self, monkeypatch, file_logging_enabled):
        mod = self._run_inline(monkeypatch)
        errors = []
        monkeypatch.setattr(mod, "save_pcm_wav", lambda *a, **k: "p.wav")
        monkeypatch.setattr(mod, "error", lambda m: errors.append(m))
        outpost, _ = make_outpost()
        outpost._save_recording_async("hello", AUDIO, CAPTURE_DROP_MIN_GAP)
        assert len(errors) == 1 and "INCOMPLETE" in errors[0]
        outpost._save_recording_async("hello", AUDIO, 0.0)
        assert len(errors) == 1              # no drop → no warning

    def test_an_overheard_incomplete_recording_is_flagged_without_its_words(self, monkeypatch, file_logging_enabled, logged):
        from interfaces.voice.microphones.local_outposts import constants
        monkeypatch.setattr(constants, "LOG_OVERHEARD_TRANSCRIPTS", False)
        mod = self._run_inline(monkeypatch)
        text = "we should call the dentist tomorrow"
        monkeypatch.setattr(mod, "save_pcm_wav", lambda folder, name, *a, **k: f"log/audio/{name}.wav")
        outpost, _ = make_outpost()
        outpost._save_recording_async(text, AUDIO, CAPTURE_DROP_MIN_GAP, addressed=False)
        incomplete = [line for line in logged if "INCOMPLETE" in line]
        assert len(incomplete) == 1
        assert "dentist" not in incomplete[0]


@pytest.mark.unit
class TestScribeTranscriptGate:
    """The parallel Scribe vote: _on_scribe_transcript mirrors _on_transcript's gates but is
    read-only (no accumulator, no wake-word reset) and de-scales confidence onto Whisper's
    range (conf / SCRIBE_CONF_CALIBRATION)."""

    def test_wake_word_stripped_sent_without_reset_or_state_change(self):
        # In a live session (awaiting_wakeword=False — Whisper's accept opened it before this
        # Scribe job returned): Scribe strips the wake word and votes, but never carries the reset.
        outpost, client = make_outpost(awaiting_wakeword=False, languages=("en-US",))
        outpost._on_scribe_transcript("Michelle turn on the lights", 0.9, "en", 0.0)
        client.send_query.assert_called_once()
        q, lang = client.send_query.call_args[0][0], client.send_query.call_args[0][1]
        assert q == "Turn on the lights" and lang == "en-US"
        # Scribe never carries the wake-word reset and never flips the wake-word state —
        # only Whisper's send owns that.
        assert client.send_query.call_args.kwargs.get("wake_word") is False
        assert outpost.awaiting_wakeword is False

    def test_muted_drops(self):
        outpost, client = make_outpost(awaiting_wakeword=False, muted=True)
        outpost._on_scribe_transcript("turn on the lights", 0.9, "sv", 0.0)
        client.send_query.assert_not_called()

    def test_no_wake_word_while_awaiting_drops(self):
        outpost, client = make_outpost(awaiting_wakeword=True)
        outpost._on_scribe_transcript("turn on the lights", 0.9, "sv", 0.0)
        client.send_query.assert_not_called()

    def test_priming_echo_dropped(self):
        outpost, client = make_outpost(awaiting_wakeword=False)
        outpost._on_scribe_transcript(INITIAL_PROMPT_PHRASES[0], 0.9, "sv", 0.0)
        client.send_query.assert_not_called()

    def test_low_calibrated_confidence_dropped(self):
        # 0.4 / 1.65 = 0.24 < SCRIBE_MIN_CONFIDENCE (0.3) → dropped locally so it never trips
        # the server's false-positive kill path.
        outpost, client = make_outpost(awaiting_wakeword=False)
        outpost._on_scribe_transcript("turn on the lights", 0.4, "sv", 0.0)
        client.send_query.assert_not_called()

    def test_send_carries_no_audio(self):
        # audio=None → no duplicate speaker identification; Whisper's send already populated it.
        outpost, client = make_outpost(awaiting_wakeword=False)
        outpost._on_scribe_transcript("turn on the lights", 0.9, "sv", 0.0)
        assert client.send_query.call_args.kwargs.get("audio") is None

    def test_detected_language_routes_to_its_locale(self):
        outpost, client = make_outpost(awaiting_wakeword=False, languages=("en-US", "sv-SE"))
        outpost._on_scribe_transcript("släck lamporna", 0.9, "sv", 0.0)
        assert client.send_query.call_args[0][1] == "sv-SE"

    def test_iso639_3_language_code_normalized(self):
        # Scribe reports ISO-639-3 ('swe'); it must map to the 'sv' subtag, not fall back.
        outpost, client = make_outpost(awaiting_wakeword=False, languages=("en-US", "sv-SE"))
        outpost._on_scribe_transcript("släck lamporna", 0.9, "swe", 0.0)
        assert client.send_query.call_args[0][1] == "sv-SE"

    def test_unmapped_language_falls_back_to_default_not_dropped(self):
        # Unlike Whisper (which drops out-of-set detections), Scribe has no restricted re-pick,
        # so an unmapped code is treated as a label quirk: route to the default and still vote.
        outpost, client = make_outpost(awaiting_wakeword=False, languages=("en-US", "sv-SE"))
        outpost._on_scribe_transcript("turn on the lights", 0.9, "no", 0.0)
        client.send_query.assert_called_once()
        assert client.send_query.call_args[0][1] == "en-US"

    # ── wake-word verification (verify_wake_word=True) ──
    def test_verify_disagreement_vetoes_and_reverts_to_wake_word(self):
        # Scribe verifying a Whisper wake-word send but hearing no wake word → kill the request
        # for Whisper's phrase and drop this mic's spuriously-armed listen window.
        outpost, client = make_outpost(awaiting_wakeword=False, languages=("en-US",))
        outpost._on_scribe_transcript("turn on the lights", 0.9, "en", 0.0,
                                      verify_wake_word=True, whisper_query="turn on the lights")
        client.send_false_positive.assert_called_once_with("turn on the lights", outpost.label)
        client.send_query.assert_not_called()
        assert outpost.awaiting_wakeword is True

    def test_verify_confirmation_votes_and_does_not_veto(self):
        # Scribe also hears the wake word → it's a real command: vote, never veto.
        outpost, client = make_outpost(awaiting_wakeword=False, languages=("en-US",))
        outpost._on_scribe_transcript("Michelle turn on the lights", 0.9, "en", 0.0,
                                      verify_wake_word=True, whisper_query="turn on the lights")
        client.send_query.assert_called_once()
        client.send_false_positive.assert_not_called()

    def test_verify_doubled_wake_word_vetoes(self):
        # Production false positives: Scribe "confirmed" with a doubled wake word ("Michelle
        # Michelle …") hallucinated from Swedish speech. The doubling refutes the fire —
        # even though a wake word (and command text) is nominally present.
        outpost, client = make_outpost(awaiting_wakeword=False, languages=("en-US",))
        outpost._on_scribe_transcript("Michelle Michelle, is it over there?", 0.9, "en", 0.0,
                                      verify_wake_word=True, whisper_query="is it there")
        client.send_false_positive.assert_called_once_with("is it there", outpost.label)
        client.send_query.assert_not_called()
        assert outpost.awaiting_wakeword is True

    def test_follow_up_vote_with_doubled_wake_word_still_votes(self):
        # The doubling veto is verify-only: a mid-session Scribe vote never vetoes,
        # matching the existing follow-up contract.
        outpost, client = make_outpost(awaiting_wakeword=False, languages=("en-US",))
        outpost._on_scribe_transcript("Michelle Michelle turn on the lights", 0.9, "en", 0.0)
        client.send_false_positive.assert_not_called()
        client.send_query.assert_called_once()

    def test_verify_deep_wake_word_confirms_not_vetoes(self):
        # Scribe endpoints differently from Whisper and often grabs leading speech, pushing the
        # wake word past the anchor: Whisper "Michelle, turn on the lights" vs Scribe
        # "okay so anyway Michelle, turn on the lights". That is Scribe CONFIRMING the wake word,
        # not disagreeing — it must vote, never veto, and the vote is the command, not the preamble.
        outpost, client = make_outpost(awaiting_wakeword=False, languages=("en-US",))
        outpost._on_scribe_transcript("okay so anyway Michelle, turn on the lights", 0.9, "en", 0.0,
                                      verify_wake_word=True, whisper_query="turn on the lights")
        client.send_false_positive.assert_not_called()
        client.send_query.assert_called_once()
        assert client.send_query.call_args.args[0] == "Turn on the lights"

    def test_verify_vote_correlates_by_whisper_phrase_not_scribe_transcript(self):
        # The vote must carry whisper_query as verified_phrase — the exact phrase the server
        # is running, hence the only string that can settle the verdict gate for the RIGHT
        # utterance. Scribe's own transcript diverges here (deep wake word grabbed leading
        # speech), so using it as the correlator would leave the gate waiting out the cap.
        outpost, client = make_outpost(awaiting_wakeword=False, languages=("en-US",))
        outpost._on_scribe_transcript("okay so anyway Michelle, turn on the lights", 0.9, "en", 0.0,
                                      verify_wake_word=True, whisper_query="turn on the lights")
        assert client.send_query.call_args.kwargs["verified_phrase"] == "turn on the lights"

    def test_follow_up_vote_carries_no_verified_phrase(self):
        # A follow-up vote (no whisper_query) has no verdict gate to settle: verified_phrase
        # stays empty and the server falls back to Scribe's own transcript for attribution.
        outpost, client = make_outpost(awaiting_wakeword=False, languages=("en-US",))
        outpost._on_scribe_transcript("turn on the lights", 0.9, "en", 0.0)
        assert client.send_query.call_args.kwargs["verified_phrase"] == ""

    def test_only_a_verify_vote_tells_the_server_the_wake_word_was_confirmed(self):
        # The server lets a confirmation outweigh another mic's veto, so a follow-up vote
        # (no wake word was verified) must never claim one.
        outpost, client = make_outpost(awaiting_wakeword=False, languages=("en-US",))
        outpost._on_scribe_transcript("Michelle, turn on the lights", 0.9, "en", 0.0,
                                      verify_wake_word=True, whisper_query="turn on the lights")
        assert client.send_query.call_args.kwargs["wake_word_confirmed"] is True
        client.send_query.reset_mock()
        outpost._on_scribe_transcript("turn on the lights", 0.9, "en", 0.0)
        assert client.send_query.call_args.kwargs["wake_word_confirmed"] is False

    def test_verify_empty_transcript_vetoes(self):
        # An empty Scribe result means it ran and heard no speech (a network error never
        # reaches the callback) — that refutes Whisper's wake word as strongly as a missing
        # one, so verifying must kill the request, not skip.
        outpost, client = make_outpost(awaiting_wakeword=False, languages=("en-US",))
        outpost._on_scribe_transcript("", 0.9, "en", 0.0,
                                      verify_wake_word=True, whisper_query="turn on the lights")
        client.send_false_positive.assert_called_once_with("turn on the lights", outpost.label)
        client.send_query.assert_not_called()
        assert outpost.awaiting_wakeword is True

    def test_empty_follow_up_vote_skips(self):
        # An empty transcript on a follow-up vote (verify_wake_word=False) has nothing to add
        # and nothing to refute — skip silently, never veto.
        outpost, client = make_outpost(awaiting_wakeword=False, languages=("en-US",))
        outpost._on_scribe_transcript("", 0.9, "en", 0.0)
        client.send_false_positive.assert_not_called()
        client.send_query.assert_not_called()

    def test_verify_bare_wake_word_echo_vetoes(self):
        # A production false positive: Whisper heard "Michelle is ready." but Scribe heard ONLY the
        # bare wake word — no command behind it. Background chatter naming the assistant, not
        # an address: veto. (Zero genuine WakeWordTest fires have this shape — when Whisper
        # hears a command, Scribe always hears one too.)
        outpost, client = make_outpost(awaiting_wakeword=False, languages=("en-US",))
        outpost._on_scribe_transcript("Michelle", 0.92, "en", 0.0,
                                      verify_wake_word=True, whisper_query="is ready")
        client.send_false_positive.assert_called_once_with("is ready", outpost.label)
        client.send_query.assert_not_called()
        assert outpost.awaiting_wakeword is True

    # ── near-miss wake word: Scribe misspells the opening as a near-homophone ──
    def test_verify_near_miss_wake_word_stands_but_casts_no_vote(self):
        # WakeWordTest "Michelle good night" ×2: Whisper "Michelle, goodnight." / Scribe
        # "Mission complete". The wake-word SOUND was there ("mission" is a listed Scribe
        # rendering of it, in the same position), so the fire stands — but Scribe's command text is suspect,
        # so it must not vote ("complete" beating "goodnight" would be worse than no vote).
        outpost, client = make_outpost(awaiting_wakeword=False, languages=("en-US",))
        outpost._on_scribe_transcript("Mission complete", 0.9, "en", 0.0,
                                      verify_wake_word=True, whisper_query="goodnight")
        client.send_false_positive.assert_not_called()
        client.send_query.assert_not_called()
        assert outpost.awaiting_wakeword is False

    def test_verify_near_miss_prod_kitchen_temperature(self):
        # A false veto: Whisper "'s the kitchen temperature", Scribe "Missions to
        # kitchen temperature" — a genuine address killed for a spelling.
        outpost, client = make_outpost(awaiting_wakeword=False, languages=("en-US",))
        outpost._on_scribe_transcript("Missions to kitchen temperature", 0.9, "en", 0.0,
                                      verify_wake_word=True, whisper_query="'s the kitchen temperature")
        client.send_false_positive.assert_not_called()

    def test_verify_multi_word_near_miss_volume_down(self):
        # A WakeWordTest clip: Whisper "Michelle, both of
        # them are down" fired, Scribe "We shall volume down" vetoed a genuine address.
        outpost, client = make_outpost(awaiting_wakeword=False, languages=("en-US",))
        outpost._on_scribe_transcript("We shall volume down", 0.9, "en", 0.0,
                                      verify_wake_word=True, whisper_query="both of them are down")
        client.send_false_positive.assert_not_called()
        client.send_query.assert_not_called()
        client.send_scribe_settled.assert_called_once_with("both of them are down", outpost.label)

    # ── disagreement veto: both engines guess a different command, Scribe unsure ──
    def test_verify_disagreement_vetoes_correlated_hallucination(self):
        # A correlated production false positive: Whisper "Michelle, close
        # the window." 43%, Scribe "Michelle, did you bring any of it?" raw 0.69.
        outpost, client = make_outpost(awaiting_wakeword=False, languages=("en-US",))
        outpost._on_scribe_transcript("Michelle, did you bring any of it?", 0.69, "en", 0.0,
                                      verify_wake_word=True, whisper_query="close the window",
                                      whisper_conf=0.43)
        client.send_false_positive.assert_called_once_with("close the window", outpost.label)
        client.send_query.assert_not_called()

    def test_verify_disagreement_vetoes_the_long_standing_fp(self):
        # The long-standing accepted FP: Whisper "what's that?" 47% / Scribe "all good?" 0.69.
        outpost, client = make_outpost(awaiting_wakeword=False, languages=("en-US",))
        outpost._on_scribe_transcript("Michelle, all good?", 0.69, "en", 0.0,
                                      verify_wake_word=True, whisper_query="who's there",
                                      whisper_conf=0.47)
        client.send_false_positive.assert_called_once_with("who's there", outpost.label)

    def test_verify_disagreement_stands_when_either_engine_is_confident(self):
        # Genuine WakeWordTest fires where the engines disagree: Whisper botched the command
        # and Scribe is sure — or Whisper was sure and Scribe botched it (model x).
        for whisper, wconf, scribe, sconf in [("hold your jaw", 0.51, "Michelle, volume down", 1.0),
                                              ("I'm gonna be back", 0.42, "Michelle volume down", 0.92),
                                              ("rolling back", 0.37, "Michelle volume up", 0.97),
                                              ("where's my charger?", 0.56, "Michelle, we must not move", 0.53)]:
            outpost, client = make_outpost(awaiting_wakeword=False, languages=("en-US",))
            outpost._on_scribe_transcript(scribe, sconf, "en", 0.0, verify_wake_word=True,
                                          whisper_query=whisper, whisper_conf=wconf)
            client.send_false_positive.assert_not_called()
            client.send_query.assert_called_once()

    def test_verify_similar_commands_stand_even_when_scribe_is_unsure(self):
        outpost, client = make_outpost(awaiting_wakeword=False, languages=("en-US",))
        outpost._on_scribe_transcript("Michelle, good night", 0.5, "en", 0.0,
                                      verify_wake_word=True, whisper_query="goodnight",
                                      whisper_conf=0.40)
        client.send_false_positive.assert_not_called()
        client.send_query.assert_called_once()

    def test_verify_disagreement_never_bites_a_near_miss(self):
        # A near-miss opening already drops Scribe's vote as untrustworthy; it must not
        # additionally veto for disagreeing ("Mission complete" vs "goodnight" at 0.71).
        outpost, client = make_outpost(awaiting_wakeword=False, languages=("en-US",))
        outpost._on_scribe_transcript("Mission complete", 0.71, "en", 0.0,
                                      verify_wake_word=True, whisper_query="goodnight",
                                      whisper_conf=0.47)
        client.send_false_positive.assert_not_called()

    def test_verify_disagreement_is_off_without_whisper_conf(self):
        outpost, client = make_outpost(awaiting_wakeword=False, languages=("en-US",))
        outpost._on_scribe_transcript("Michelle, all good?", 0.69, "en", 0.0,
                                      verify_wake_word=True, whisper_query="who's there")
        client.send_false_positive.assert_not_called()

    def test_verify_near_miss_needs_a_command_on_both_sides(self):
        # A bare Whisper fire (empty query) arms an unguarded listen window and a bare
        # near-miss is the bare-echo shape — both keep the strict veto.
        outpost, client = make_outpost(awaiting_wakeword=False, languages=("en-US",))
        outpost._on_scribe_transcript("Mission complete", 0.9, "en", 0.0,
                                      verify_wake_word=True, whisper_query="")
        client.send_query.assert_not_called()
        assert outpost.awaiting_wakeword is True   # the bare listen window is un-armed
        outpost, client = make_outpost(awaiting_wakeword=False, languages=("en-US",))
        outpost._on_scribe_transcript("Mission.", 0.9, "en", 0.0,
                                      verify_wake_word=True, whisper_query="goodnight")
        client.send_false_positive.assert_called_once_with("goodnight", outpost.label)

    # ── no-vote exits must still settle the server's verdict gate (see _settle_scribe_gate) ──
    def test_verify_near_miss_settles_the_verdict_gate(self):
        # A rescued fire is below the Scribe-skip floor, so the reply is holding for a verdict
        # that never comes as a vote — the no-vote exit must release it, or the rescue costs
        # the full MAX_WAIT_FOR_SCRIBE_VERDICT cap.
        outpost, client = make_outpost(awaiting_wakeword=False, languages=("en-US",))
        outpost._on_scribe_transcript("Mission complete", 0.9, "en", 0.0,
                                      verify_wake_word=True, whisper_query="goodnight")
        client.send_scribe_settled.assert_called_once_with("goodnight", outpost.label)
        client.send_false_positive.assert_not_called()

    def test_verify_priming_echo_settles_the_verdict_gate(self):
        outpost, client = make_outpost(awaiting_wakeword=False, languages=("en-US",))
        outpost._on_scribe_transcript(INITIAL_PROMPT_PHRASES[0], 0.9, "en", 0.0,
                                      verify_wake_word=True, whisper_query="turn on the lights")
        client.send_scribe_settled.assert_called_once_with("turn on the lights", outpost.label)
        client.send_false_positive.assert_not_called()
        client.send_query.assert_not_called()

    def test_verify_low_confidence_drop_settles_the_verdict_gate(self):
        outpost, client = make_outpost(awaiting_wakeword=False, languages=("en-US",))
        outpost._on_scribe_transcript("Michelle turn on the lights", 0.4, "en", 0.0,
                                      verify_wake_word=True, whisper_query="turn on the lights")
        client.send_query.assert_not_called()
        client.send_scribe_settled.assert_called_once_with("turn on the lights", outpost.label)

    def test_verify_vote_and_veto_do_not_double_settle(self):
        # A real vote settles via /stream (verified_phrase) and a veto via /false_positive —
        # neither needs the lightweight signal on top.
        outpost, client = make_outpost(awaiting_wakeword=False, languages=("en-US",))
        outpost._on_scribe_transcript("Michelle turn on the lights", 0.9, "en", 0.0,
                                      verify_wake_word=True, whisper_query="turn on the lights")
        client.send_query.assert_called_once()
        client.send_scribe_settled.assert_not_called()
        outpost, client = make_outpost(awaiting_wakeword=False, languages=("en-US",))
        outpost._on_scribe_transcript("okay so mission complete", 0.9, "en", 0.0,
                                      verify_wake_word=True, whisper_query="goodnight")
        client.send_false_positive.assert_called_once()
        client.send_scribe_settled.assert_not_called()

    def test_follow_up_no_vote_does_not_settle(self):
        # Only verify jobs have a reply holding on them; follow-up votes never touch the gate.
        outpost, client = make_outpost(awaiting_wakeword=False, languages=("en-US",))
        outpost._on_scribe_transcript("turn on the lights", 0.4, "en", 0.0)
        client.send_scribe_settled.assert_not_called()

    def test_verify_near_miss_must_open_the_transcript(self):
        # Unlike a real wake word (deep promotion), a near-miss buried in the sentence is no
        # evidence of an address.
        outpost, client = make_outpost(awaiting_wakeword=False, languages=("en-US",))
        outpost._on_scribe_transcript("okay so mission complete", 0.9, "en", 0.0,
                                      verify_wake_word=True, whisper_query="goodnight")
        client.send_false_positive.assert_called_once_with("goodnight", outpost.label)

    def test_verify_unrelated_opening_word_still_vetoes(self):
        # Prod vetoes that must keep vetoing: ordinary words and Swedish "my-" words are not
        # on the near-miss list — nothing fuzzy happens here.
        for scribe, whisper in [("Sure. Put the kettle on now", "could you hand me the towel"),
                                ("Mycket bra väder för oss", "was a finalist"),
                                ("Much better now", "better now"),
                                ("Let me show you how", "you have a question")]:
            outpost, client = make_outpost(awaiting_wakeword=False, languages=("en-US",))
            outpost._on_scribe_transcript(scribe, 0.9, "en", 0.0,
                                          verify_wake_word=True, whisper_query=whisper)
            client.send_false_positive.assert_called_once_with(whisper, outpost.label)

    def test_verify_bare_wake_word_on_both_engines_does_not_veto(self):
        # A genuine bare address ("Michelle!") is bare in BOTH engines. Whisper's query for it
        # can be punctuation-only ("!" survives clean_query's separator strip), so the
        # bare-echo veto must compare WORD content, not string emptiness — a wordless
        # whisper query is not a command Scribe failed to hear.
        outpost, client = make_outpost(awaiting_wakeword=False, languages=("en-US",))
        outpost._on_scribe_transcript("Michelle", 0.9, "en", 0.0,
                                      verify_wake_word=True, whisper_query="!")
        client.send_false_positive.assert_not_called()

    def test_wake_word_vote_after_turn_closed_drops(self):
        # A production incident: a wake-word VERIFY job
        # returned 0.5s after the reply had ended (the outpost was already reset to
        # awaiting_wakeword), carrying a wake word and a phrase that diverged from whisper_query.
        # Confirming the wake word, it skipped the veto — then, because it had a wake word, it
        # slipped past the stale-vote guard and was injected as a brand-new query. Scribe never
        # opens a session: once the turn has closed its vote is stale and must be dropped even
        # with a wake word present (a genuinely fresh wake word arrives via Whisper, which
        # reopens the session before its Scribe job returns).
        outpost, client = make_outpost(awaiting_wakeword=True, languages=("en-US",))
        outpost._on_scribe_transcript("Michelle can you hear the rain", 0.9, "en", 0.0,
                                      verify_wake_word=True, whisper_query="turn on the controls")
        client.send_query.assert_not_called()       # not injected as a new query…
        client.send_false_positive.assert_not_called()  # …and the wake word was confirmed, so no veto
        assert outpost.awaiting_wakeword is True


@pytest.mark.unit
class TestUtteranceConcatenation:
    """A command split by a mid-utterance pause > SILENCE arrives as disjoint Whisper
    segments. The outpost re-joins consecutive fragments into one cumulative growing
    phrase (the contract the server's voting expects) when the inter-fragment gap is
    within CONCAT_MAX_GAP — otherwise they're treated as separate commands."""

    def test_short_gap_fragments_are_joined_cumulatively(self):
        outpost, client = make_outpost(awaiting_wakeword=True)
        # Fragment 1: the wake-word command, cut off mid-sentence by a pause.
        outpost._on_transcript("Michelle what's the", CLEAN, AUDIO,
                               vad_stop_at=100.0, vad_start_at=99.0)
        # Fragment 2: resumes 0.5s later — within CONCAT_MAX_GAP → continues the utterance.
        outpost._on_transcript("kitchen temperature", CLEAN, AUDIO,
                               vad_stop_at=101.5, vad_start_at=100.5)

        assert client.send_query.call_count == 2
        first, second = client.send_query.call_args_list
        assert first[0][0] == "What's the"
        assert first.kwargs.get("wake_word") is True
        # The second send is the CUMULATIVE phrase, not the bare fragment…
        assert second[0][0] == "What's the kitchen temperature"
        # …and it must not re-reset the conversation the wake word just opened.
        assert second.kwargs.get("wake_word") in (False, None)
        outpost._cancel_listen_timers()

    def test_long_gap_starts_a_new_command(self):
        outpost, client = make_outpost(awaiting_wakeword=True)
        outpost._on_transcript("Michelle what's the", CLEAN, AUDIO,
                               vad_stop_at=100.0, vad_start_at=99.0)
        # Resumes 3s later (> CONCAT_MAX_GAP 2.5) → a separate command, NOT joined.
        outpost._on_transcript("turn on the lights", CLEAN, AUDIO,
                               vad_stop_at=104.0, vad_start_at=103.0)
        assert client.send_query.call_args_list[1][0][0] == "Turn on the lights"
        outpost._cancel_listen_timers()

    def test_follow_up_fragments_join_without_wake_word(self):
        # Inside a listen window (no wake word): two halves of one follow-up still join.
        outpost, client = make_outpost(awaiting_wakeword=False)
        outpost._on_transcript("play some", CLEAN, AUDIO, vad_stop_at=10.0, vad_start_at=9.0)
        outpost._on_transcript("music in the kitchen", CLEAN, AUDIO,
                               vad_stop_at=11.5, vad_start_at=10.4)
        second = client.send_query.call_args_list[1]
        assert second[0][0] == "Play some music in the kitchen"
        assert second.kwargs.get("wake_word") in (False, None)
        outpost._cancel_listen_timers()

    def test_new_wake_word_starts_fresh_utterance(self):
        outpost, client = make_outpost(awaiting_wakeword=True)
        outpost._on_transcript("Michelle what's the", CLEAN, AUDIO,
                               vad_stop_at=100.0, vad_start_at=99.0)
        # A second wake word soon after is a brand-new command, not a continuation.
        outpost._on_transcript("Michelle play music", CLEAN, AUDIO,
                               vad_stop_at=101.0, vad_start_at=100.5)
        second = client.send_query.call_args_list[1]
        assert second[0][0] == "Play music"            # fresh, not "What's the play music"
        assert second.kwargs.get("wake_word") is True  # resets the conversation again
        outpost._cancel_listen_timers()

    def test_mute_clears_accumulator(self):
        outpost, client = make_outpost(awaiting_wakeword=True)
        outpost._on_transcript("Michelle what's the", CLEAN, AUDIO,
                               vad_stop_at=100.0, vad_start_at=99.0)
        assert outpost._acc_text                       # accumulator holds the first half
        outpost.on_mute()
        assert outpost._acc_text == ""                 # Floxy speaking ends the utterance
        outpost._cancel_listen_timers()

    def test_missing_timing_does_not_blindly_join(self):
        # Defensive: without vad_start_at the gap is unknowable, so a wake-word-less
        # fragment starts fresh rather than concatenating onto a stale accumulator.
        outpost, client = make_outpost(awaiting_wakeword=False)
        outpost._on_transcript("play some", CLEAN, AUDIO, vad_stop_at=10.0, vad_start_at=9.0)
        outpost._on_transcript("music", CLEAN, AUDIO)  # no timing → no gap → no join
        assert client.send_query.call_args_list[1][0][0] == "Music"
        outpost._cancel_listen_timers()


@pytest.mark.unit
class TestSpeechActiveSignal:
    """At VAD onset the outpost pings /speech_active so the server keeps an in-flight
    incomplete request waiting through a mid-utterance pause — but only when the onset
    looks like a continuation of an utterance it's mid-sending, so room noise can't hold
    every request open."""

    def _mid_utterance(self):
        outpost, client = make_outpost(awaiting_wakeword=False)
        outpost._acc_text = "what is the"          # an utterance is mid-flight
        outpost._acc_last_stop_at = 100.0
        return outpost, client

    def test_signals_for_continuation_within_gap(self):
        outpost, _ = self._mid_utterance()
        assert outpost._should_signal_speech_active(100.5) is True   # 0.5s after last fragment

    def test_silent_when_gap_too_long(self):
        from interfaces.voice.microphones.local_outposts.constants import CONCAT_MAX_GAP
        outpost, _ = self._mid_utterance()
        assert outpost._should_signal_speech_active(100.0 + CONCAT_MAX_GAP + 0.5) is False

    def test_silent_when_awaiting_wake_word(self):
        outpost, _ = self._mid_utterance()
        outpost.awaiting_wakeword = True            # a fresh address, not a continuation
        assert outpost._should_signal_speech_active(100.5) is False

    def test_silent_without_in_flight_utterance(self):
        outpost, _ = make_outpost(awaiting_wakeword=False)   # listen window, but nothing sent yet
        assert outpost._should_signal_speech_active(100.5) is False

    def test_notify_speech_active_pings_the_endpoint(self):
        client = OutpostStreamClient(8888)
        got = []
        client._get = lambda url, label: got.append(url)
        client._executor.submit = lambda fn, *a: fn(*a)  # run inline
        client.notify_speech_active("m1")
        assert len(got) == 1 and got[0].endswith("/speech_active")

    def test_send_false_positive_carries_phrase_and_mic(self):
        from urllib.parse import parse_qs, urlparse
        client = OutpostStreamClient(8888)
        got = []
        client._get = lambda url, label: got.append(url)
        client._executor.submit = lambda fn, *a: fn(*a)  # run inline
        client.send_false_positive("turn on the lights", "Kitchen Mic")
        assert len(got) == 1
        parsed = urlparse(got[0])
        assert parsed.path == "/false_positive"
        q = parse_qs(parsed.query)
        assert q["query"] == ["turn on the lights"] and q["mic"] == ["Kitchen Mic"]

    def test_send_scribe_settled_carries_phrase_and_mic(self):
        from urllib.parse import parse_qs, urlparse
        client = OutpostStreamClient(8888)
        got = []
        client._get = lambda url, label: got.append(url)
        client._executor.submit = lambda fn, *a: fn(*a)  # run inline
        client.send_scribe_settled("goodnight", "Kitchen Mic")
        assert len(got) == 1
        parsed = urlparse(got[0])
        assert parsed.path == "/scribe_settled"
        q = parse_qs(parsed.query)
        assert q["query"] == ["goodnight"] and q["mic"] == ["Kitchen Mic"]


@pytest.mark.unit
class TestStreamUrl:
    def test_marks_request_as_voice_to_voice(self):
        from urllib.parse import parse_qs, urlparse
        url = OutpostStreamClient(8888).build_stream_url("turn on the lights", "sv-SE", 0.82)
        q = parse_qs(urlparse(url).query)
        assert q["source"] == ["VoiceToVoice"]
        assert q["server_sound"] == ["true"] and q["low_stability"] == ["false"]
        assert q["lang"] == ["sv-SE"] and q["query"] == ["turn on the lights"]
        assert q["confidence"] == ["0.8200"]

    def test_sends_the_configured_location(self, monkeypatch):
        # The server only auto-defaults a location for recognized mobile apps; our HTTP
        # client isn't one, so the outpost sends `local_outposts.location` itself.
        from urllib.parse import parse_qs, urlparse
        monkeypatch.setattr(
            "interfaces.voice.microphones.local_outposts.OutpostStreamClient.OUTPOST_LOCATION", "Kitchen")
        url = OutpostStreamClient(8888).build_stream_url("play music", "en-US")
        assert parse_qs(urlparse(url).query)["location"] == ["Kitchen"]

    def test_no_location_sent_when_none_is_configured(self, monkeypatch):
        from urllib.parse import parse_qs, urlparse
        monkeypatch.setattr(
            "interfaces.voice.microphones.local_outposts.OutpostStreamClient.OUTPOST_LOCATION", None)
        url = OutpostStreamClient(8888).build_stream_url("play music", "en-US")
        assert "location" not in parse_qs(urlparse(url).query)

    def test_confidence_omitted_when_none(self):
        from urllib.parse import parse_qs, urlparse
        url = OutpostStreamClient(8888).build_stream_url("play music", "en-US", None)
        assert "confidence" not in parse_qs(urlparse(url).query)

    def test_wake_word_flag_added_when_set(self):
        from urllib.parse import parse_qs, urlparse
        url = OutpostStreamClient(8888).build_stream_url("turn on the lights", "en-US", wake_word=True)
        assert parse_qs(urlparse(url).query)["wake_word_detected"] == ["1"]

    def test_stt_ms_added_when_set(self):
        from urllib.parse import parse_qs, urlparse
        url = OutpostStreamClient(8888).build_stream_url("turn on the lights", "en-US", stt_ms=180)
        assert parse_qs(urlparse(url).query)["stt_ms"] == ["180"]

    def test_stt_trigger_added_when_set(self):
        from urllib.parse import parse_qs, urlparse
        url = OutpostStreamClient(8888).build_stream_url("turn on the lights", "en-US", stt_trigger="rms")
        assert parse_qs(urlparse(url).query)["stt_trigger"] == ["rms"]

    def test_scribe_skipped_flag_added_when_set(self):
        from urllib.parse import parse_qs, urlparse
        url = OutpostStreamClient(8888).build_stream_url("turn on the lights", "en-US", scribe_skipped=True)
        assert parse_qs(urlparse(url).query)["scribe_skipped"] == ["1"]

    def test_mic_added_and_stripped_when_set(self):
        # The outpost label is space-padded to 30 chars for log alignment; the URL
        # should carry the trimmed device name so the Home tab attributes the phrase
        # to a specific microphone.
        from urllib.parse import parse_qs, urlparse
        url = OutpostStreamClient(8888).build_stream_url(
            "turn on the lights", "en-US", mic="Studio Display Microphone     ")
        assert parse_qs(urlparse(url).query)["mic"] == ["Studio Display Microphone"]

    def test_send_query_forwards_verified_phrase(self):
        # verified_phrase is the correlator the server's wake-word verdict gate settles on.
        # If it silently drops out of the /stream URL, the server falls back to matching
        # Scribe's OWN transcript — which legitimately diverges from Whisper's — and the
        # stale-verdict race 4db1a9fc closed quietly reopens. End-to-end through send_query
        # so both the kwarg forwarding and the URL encoding are covered.
        from urllib.parse import parse_qs, urlparse
        client = OutpostStreamClient(8888)
        got = []
        client._identify_then_get = lambda url, audio, label, wake_word: got.append(url)
        client._executor.submit = lambda fn, *a: fn(*a)  # run inline
        client.send_query("turn the lights on", "en-US", engine="scribe",
                          verified_phrase="turn on the lights")
        assert len(got) == 1
        assert parse_qs(urlparse(got[0]).query)["verified_phrase"] == ["turn on the lights"]

    def test_wake_word_confirmed_is_the_stream_param_the_server_reads(self):
        # Contract with MicrophoneRoutes.accept_transcription: wake_word_confirmed=1 on /stream.
        from urllib.parse import parse_qs, urlparse
        client = OutpostStreamClient(8888)
        confirmed = client.build_stream_url("turn the lights on", "en-US", engine="scribe",
                                            verified_phrase="turn on the lights",
                                            wake_word_confirmed=True)
        plain = client.build_stream_url("turn the lights on", "en-US", engine="scribe")
        assert parse_qs(urlparse(confirmed).query)["wake_word_confirmed"] == ["1"]
        assert "wake_word_confirmed" not in parse_qs(urlparse(plain).query)

@pytest.mark.unit
class TestSpeakerIdentification:
    def _fake_identifier(self, monkeypatch, instance):
        # Patch the class the client lazily imports so get_instance() yields our fake.
        from interfaces.voice.speaker_id.SpeakerIdentifier import SpeakerIdentifier
        monkeypatch.setattr(SpeakerIdentifier, "get_instance",
                            classmethod(lambda cls: instance))

    def test_identify_runs_before_get(self):
        # The ordering guarantee: history must be populated before the server votes.
        client = OutpostStreamClient(8888)
        order = []
        client._identify_speaker = lambda audio, label: order.append("identify")
        client._get = lambda url, label: order.append("get")
        client._identify_then_get("http://x/stream", AUDIO, "m1")
        assert order == ["identify", "get"]

    def test_identify_converts_float32_to_16bit_pcm(self, monkeypatch):
        fake = MagicMock()
        fake.identify.return_value = ("Sam", 0.91)
        self._fake_identifier(monkeypatch, fake)
        client = OutpostStreamClient(8888)
        client._identify_speaker(np.array([0.0, 1.0, -1.0, 0.5], np.float32), "m1")
        pcm, sample_rate, bit_depth = fake.identify.call_args[0]
        assert sample_rate == 16000 and bit_depth == 16
        assert np.frombuffer(pcm, dtype=np.int16).tolist() == [0, 32767, -32767, 16383]

    def test_identify_clips_out_of_range_samples(self, monkeypatch):
        fake = MagicMock()
        fake.identify.return_value = ("Sam", 0.9)
        self._fake_identifier(monkeypatch, fake)
        client = OutpostStreamClient(8888)
        client._identify_speaker(np.array([2.0, -2.0], np.float32), "m1")  # beyond [-1, 1]
        assert np.frombuffer(fake.identify.call_args[0][0], dtype=np.int16).tolist() == [32767, -32767]

    def test_identify_noop_without_identifier(self, monkeypatch):
        self._fake_identifier(monkeypatch, None)  # not initialized
        OutpostStreamClient(8888)._identify_speaker(AUDIO, "m1")  # must not raise

    def test_identify_skips_empty_audio(self, monkeypatch):
        fake = MagicMock()
        self._fake_identifier(monkeypatch, fake)
        client = OutpostStreamClient(8888)
        client._identify_speaker(None, "m1")
        client._identify_speaker(np.zeros(0, np.float32), "m1")
        fake.identify.assert_not_called()

    def test_send_query_forwards_audio_to_identify(self, monkeypatch):
        # The whole path: send_query schedules an identify-then-get carrying the audio.
        client = OutpostStreamClient(8888)
        captured = {}
        client._identify_then_get = lambda url, audio, label, wake_word=False: captured.update(url=url, audio=audio)
        client._executor.submit = lambda fn, *a: fn(*a)  # run inline
        client.send_query("Turn on the lights", "sv-SE", 0.8, "m1", audio=AUDIO)
        assert captured["audio"] is AUDIO and "/stream?" in captured["url"]

    def test_identify_does_not_record_without_wake_word(self, monkeypatch):
        # Speculative identification on background speech must NOT update the identified
        # user — record=False is passed so the vote history stays clean.
        fake = MagicMock()
        fake.identify.return_value = ("Sam", 0.9)
        self._fake_identifier(monkeypatch, fake)
        client = OutpostStreamClient(8888)
        client._identify_speaker(AUDIO, "m1")
        assert fake.identify.call_args.kwargs.get("record") is False
        fake.record_identification.assert_not_called()

    def test_wake_word_records_deferred_identification(self, monkeypatch):
        # On a wake word, the deferred identify result is committed to the vote history.
        fake = MagicMock()
        self._fake_identifier(monkeypatch, fake)
        client = OutpostStreamClient(8888)
        client._get = lambda url, label: None
        client._identify_speaker = lambda audio, label: ("Sam", 0.9)
        client._identify_then_get("http://x/stream", AUDIO, "m1", wake_word=True)
        fake.record_identification.assert_called_once_with("Sam", 0.9)

    def test_no_wake_word_skips_deferred_recording(self, monkeypatch):
        fake = MagicMock()
        self._fake_identifier(monkeypatch, fake)
        client = OutpostStreamClient(8888)
        client._get = lambda url, label: None
        client._identify_speaker = lambda audio, label: ("Sam", 0.9)
        client._identify_then_get("http://x/stream", AUDIO, "m1", wake_word=False)
        fake.record_identification.assert_not_called()

    def test_send_query_with_future_waits_instead_of_reidentifying(self):
        # When identification was started early, send_query waits on that future and
        # does NOT identify a second time — but still GETs after the wait (ordering).
        client = OutpostStreamClient(8888)
        order = []
        client._identify_speaker = lambda audio, label: order.append("identify")
        client._get = lambda url, label: order.append("get")
        client._executor.submit = lambda fn, *a: fn(*a)  # run inline
        fut = MagicMock()
        fut.result = lambda timeout=None: order.append("wait")
        client.send_query("Lights on", "sv-SE", 0.8, "m1", audio=AUDIO, identify_future=fut)
        assert order == ["wait", "get"]  # waited on the pre-started identify, no re-identify

    def test_send_query_identifies_inline_without_future(self):
        # No speculative future (speculative never fired) → fall back to inline order.
        client = OutpostStreamClient(8888)
        order = []
        client._identify_speaker = lambda audio, label: order.append("identify")
        client._get = lambda url, label: order.append("get")
        client._executor.submit = lambda fn, *a: fn(*a)  # run inline
        client.send_query("Lights on", "sv-SE", 0.8, "m1", audio=AUDIO)
        assert order == ["identify", "get"]

    def test_wait_then_get_survives_slow_identify(self):
        # A future that blows the timeout must not block the GET — graceful fallback.
        client = OutpostStreamClient(8888)
        got = []
        client._get = lambda url, label: got.append(url)
        fut = MagicMock()
        fut.result = MagicMock(side_effect=TimeoutError("still running"))
        client._wait_then_get(fut, "http://x/stream", "m1")
        assert got == ["http://x/stream"]


class FakeTimer:
    """Stand-in for threading.Timer: never fires on its own; tests fire it by reason."""
    def __init__(self, delay, fn, args=()):
        self.delay, self.fn, self.args, self.cancelled = delay, fn, args, False
        self.daemon = False

    def start(self):
        pass

    def cancel(self):
        self.cancelled = True


class FakeTimers:
    def __init__(self):
        self.created = []

    def __call__(self, delay, fn, args=()):
        timer = FakeTimer(delay, fn, args)
        self.created.append(timer)
        return timer

    def fire(self, reason):
        live = [t for t in self.created if not t.cancelled and t.args == (reason,)]
        assert live, f"no armed timer for {reason!r}"
        live[-1].fn(*live[-1].args)


@pytest.fixture
def fake_timers(monkeypatch):
    import interfaces.voice.microphones.local_outposts.LocalOutpost as mod
    timers = FakeTimers()
    monkeypatch.setattr(mod.threading, "Timer", timers)
    return timers


@pytest.mark.unit
class TestFollowUpTimeout:
    def test_on_wakeword_reverts_and_cancels_timers(self):
        outpost, _ = make_outpost()
        outpost.on_listen()
        outpost.on_wakeword()
        assert outpost.awaiting_wakeword is True
        assert outpost._no_words_timer is None and outpost._max_listen_timer is None

    def test_reverts_to_wakeword_after_silence(self, fake_timers):
        outpost, _ = make_outpost()
        outpost.on_listen()
        assert outpost.awaiting_wakeword is False
        fake_timers.fire("no follow-up spoken")
        assert outpost.awaiting_wakeword is True  # no follow-up → reverted

    def test_followup_utterance_resets_between_words_timer(self, fake_timers):
        outpost, _ = make_outpost()
        outpost.on_listen()
        first_word = outpost._no_words_timer
        outpost._note_followup_utterance()  # got a follow-up → switch to longer between-words window
        assert first_word.cancelled
        assert outpost._no_words_timer.delay == LISTEN_NO_NEW_WORDS_TIMEOUT
        assert outpost.awaiting_wakeword is False
        fake_timers.fire("no further follow-up spoken")
        assert outpost.awaiting_wakeword is True

    def test_mute_cancels_timers(self):
        outpost, _ = make_outpost()
        outpost.on_listen()
        outpost.on_mute()
        assert outpost._no_words_timer is None and outpost._max_listen_timer is None

    def test_speech_started_holds_false_positive_timer(self, fake_timers):
        # A follow-up that began in time but is slow to transcribe must not be cut
        # off: speech START holds the between-words timer until STOP re-arms it.
        outpost, _ = make_outpost()
        outpost.on_listen()
        first_word = outpost._no_words_timer
        outpost._note_speech_started()  # user is speaking before the first-word window elapses
        assert first_word.cancelled and outpost._no_words_timer is None  # held open
        assert outpost.awaiting_wakeword is False
        outpost._note_speech_ended()    # utterance ended → between-words timer re-armed
        assert outpost._no_words_timer.delay == LISTEN_NO_NEW_WORDS_TIMEOUT
        assert outpost.awaiting_wakeword is False
        fake_timers.fire("no further follow-up spoken")
        assert outpost.awaiting_wakeword is True   # finally reverts when truly quiet

    def test_speech_started_ignored_outside_listen_window(self):
        outpost, _ = make_outpost()  # awaiting_wakeword True by default
        outpost._note_speech_started()
        outpost._note_speech_ended()
        assert outpost.awaiting_wakeword is True
        assert outpost._no_words_timer is None


@pytest.mark.unit
class TestAdaptiveRmsVAD:
    """Two-rule ambient tracking (RAISE_FRAMES=10, SETTLE_FRAMES=50 at 100ms frames):
    a fully-elevated 1s window lifts the floor to its minimum (fast attack, so hiss
    re-arms the gate within a second), and a fully sub-threshold 5s window re-anchors
    the floor to its maximum (hugging the quiet room from above, undoing any
    speech-inflated raise)."""

    def _vad(self, threshold=0.015, floor=0.005):
        from interfaces.voice.microphones.local_outposts.VAD import RmsVAD
        v = RmsVAD(threshold=threshold, label="test")
        v.noise_floor = floor                    # as seeded by the startup calibration
        return v

    @staticmethod
    def _feed(v, level, frames):
        for _ in range(frames):
            v.is_speech(np.full(1600, level, np.float32))

    def test_sustained_noise_raises_floor_within_a_second(self):
        from interfaces.voice.microphones.local_outposts.constants import RMS_THRESHOLD_MULTIPLIER
        v = self._vad()
        self._feed(v, 0.02, 10)                  # dishwasher: one full raise window
        assert v.noise_floor == pytest.approx(0.02)
        assert v.threshold == pytest.approx(0.02 * RMS_THRESHOLD_MULTIPLIER)

    def test_post_reply_hiss_rearms_the_gate(self):
        # A production failure: hiss above the threshold pegged the gate 'loud'
        # for minutes. Now the floor lifts within ~1s and hiss reads as quiet again.
        v = self._vad(threshold=0.006, floor=0.0005)
        assert v.is_speech(np.full(1600, 0.008, np.float32)) is True   # hiss starts: loud
        self._feed(v, 0.008, 9)                  # …one second of it
        assert v.noise_floor == pytest.approx(0.008)
        assert v.is_speech(np.full(1600, 0.008, np.float32)) is False  # adapted: quiet again

    def test_speech_with_quiet_gaps_does_not_raise_floor(self):
        # The raise rule drops its single lowest outlier (10th percentile), so speech
        # needs at least TWO closure/breath dips per second to hold the floor down.
        v = self._vad()
        for _ in range(8):
            self._feed(v, 0.5, 8)
            self._feed(v, 0.005, 2)
        assert v.noise_floor == pytest.approx(0.005)

    def test_speech_hint_freezes_adaptation(self):
        # Frames silero attributes to speech never feed the tracker — the floor
        # models the room, not the speaker. Same signal without the hint adapts.
        v = self._vad()
        for _ in range(20):
            v.is_speech(np.full(1600, 0.05, np.float32), speech_hint=True)
        assert v.noise_floor == pytest.approx(0.005)   # frozen while "speaking"
        self._feed(v, 0.05, 10)                        # same level, no hint → noise
        assert v.noise_floor == pytest.approx(0.05)

    def test_settle_survives_above_threshold_clicks(self):
        # A few loud clicks inside the settle window no longer disarm it: 'quiet'
        # means the 90th percentile is sub-threshold, not every single frame.
        v = self._vad()
        self._feed(v, 0.002, 47)
        self._feed(v, 0.02, 3)                   # clicks ABOVE the 0.015 threshold
        assert v.noise_floor == pytest.approx(0.002)

    def test_quiet_settle_reanchors_floor_down(self):
        v = self._vad()
        self._feed(v, 0.02, 10)                  # noise raised the floor…
        assert v.noise_floor == pytest.approx(0.02)
        self._feed(v, 0.004, 50)                 # …then a fully quiet 5s window
        assert v.noise_floor == pytest.approx(0.004)   # re-anchored to ambient's top
        from interfaces.voice.microphones.local_outposts.constants import RMS_THRESHOLD_MULTIPLIER, RMS_THRESHOLD_FLOOR
        assert v.threshold == pytest.approx(max(0.004 * RMS_THRESHOLD_MULTIPLIER,
                                                RMS_THRESHOLD_FLOOR))

    def test_settle_ignores_subthreshold_spikes(self):
        # A single click/clink below the threshold must not hold the floor up at
        # spike level — the settle anchor is the window's 90th percentile, not max.
        v = self._vad()
        self._feed(v, 0.002, 49)
        self._feed(v, 0.005, 1)                  # sub-threshold spike inside the window
        assert v.noise_floor == pytest.approx(0.002)

    def test_threshold_never_drops_below_absolute_floor(self):
        from interfaces.voice.microphones.local_outposts.constants import RMS_THRESHOLD_FLOOR
        v = self._vad()
        self._feed(v, 0.0001, 50)                # near-silent room
        assert v.threshold == pytest.approx(RMS_THRESHOLD_FLOOR)

    def test_clamp_sits_below_every_real_room_floor(self):
        # The clamp is a degenerate-silence guard, not a tuning knob: at the
        # quietest floor measured in production (0.0014), floor x
        # MULTIPLIER must be the operative threshold. A clamp at or above
        # 0.0021 re-introduces the old 0.006 bug — permanently overriding the
        # adaptive floor and clipping fading word endings ~0.5s early.
        from interfaces.voice.microphones.local_outposts.constants import RMS_THRESHOLD_MULTIPLIER, RMS_THRESHOLD_FLOOR
        v = self._vad()
        self._feed(v, 0.0014, 50)                # settle at the quietest measured room
        assert v.noise_floor == pytest.approx(0.0014)
        assert v.threshold == pytest.approx(0.0014 * RMS_THRESHOLD_MULTIPLIER)
        assert v.threshold > RMS_THRESHOLD_FLOOR

    def test_noise_floor_none_until_first_raise_window(self):
        from interfaces.voice.microphones.local_outposts.VAD import RmsVAD
        v = RmsVAD(threshold=0.015, label="test")   # unseeded (no calibration)
        self._feed(v, 0.003, 9)
        assert v.noise_floor is None
        self._feed(v, 0.003, 1)                  # first full 1s window initializes it
        assert v.noise_floor == pytest.approx(0.003)

    def test_one_spurious_dip_cannot_block_a_raise(self):
        # The raise reference is the window's 10th percentile, not its min — a single
        # anomalously low frame must not veto a legitimate ambience raise. A regression
        # to plain min() would let one dropped/glitched buffer per second keep the gate
        # pegged 'loud' through sustained noise.
        v = self._vad()
        self._feed(v, 0.02, 9)
        self._feed(v, 0.001, 1)                  # one dip inside the 1s raise window
        assert v.noise_floor == pytest.approx(0.02)
        assert v.raise_min == pytest.approx(0.02)  # the Stats panel reads this same reference

    def test_speech_hint_suppresses_tracking_not_detection(self):
        # The hint freezes the ambient tracker only; the verdict itself must stay live,
        # or the energy gate goes deaf exactly while the user is speaking.
        v = self._vad(threshold=0.015)
        assert v.is_speech(np.full(1600, 0.05, np.float32), speech_hint=True) is True
        assert v.last_score == pytest.approx(0.05, rel=0.01)
        assert v.is_speech(np.full(1600, 0.005, np.float32), speech_hint=True) is False

    def test_reset_preserves_the_ambient_tracker(self):
        # reset() runs at every utterance end; the room model must survive it — clearing
        # the tracker there would re-learn the floor from scratch after each query.
        from interfaces.voice.microphones.local_outposts.constants import RMS_THRESHOLD_MULTIPLIER
        v = self._vad()
        self._feed(v, 0.02, 10)
        v.reset()
        assert v.noise_floor == pytest.approx(0.02)
        assert v.threshold == pytest.approx(0.02 * RMS_THRESHOLD_MULTIPLIER)


@pytest.mark.unit
class TestMakeVad:
    """backend='silero' is strict: RMS-as-main-VAD is never wanted in production (its
    noise floor adapts to the user's own speech — no silero verdict exists to exclude
    it), so an import failure must fail the outpost start loudly instead of silently
    degrading into that mode."""

    def test_silero_import_failure_is_loud(self, monkeypatch):
        import interfaces.voice.microphones.local_outposts.VAD as mod

        class Broken:
            def __init__(self, *a, **k):
                raise ImportError("No module named 'torch'")

        monkeypatch.setattr(mod, "SileroVAD", Broken)
        with pytest.raises(RuntimeError, match="refusing to fall back"):
            mod.make_vad("silero", 0.5, label="test")

@pytest.mark.unit
class TestKeepWarm:
    """GPU keep-warm: a lowest-priority silence decode after KEEPWARM_IDLE_SECONDS of
    decode-idle, so the first real query of a quiet period doesn't pay Metal's
    idle-decay cold penalty. Never submitted while a mic is mid-utterance."""

    def _worker(self):
        from interfaces.voice.microphones.local_outposts.WhisperWorker import WhisperWorker
        return WhisperWorker("repo", "en")

    def test_keepwarm_decodes_when_idle(self, monkeypatch):
        import interfaces.voice.microphones.local_outposts.WhisperWorker as mod
        calls = []
        monkeypatch.setattr(mod, "transcribe",
                            lambda *a, **k: calls.append(1) or ("", "en", 0.6, 1.0, 0.02, None))
        w = self._worker()
        w._last_decode_at = time.time() - 100
        assert w.submit_keepwarm() is True
        _, _, job = w._jobs.get_nowait()
        assert job.kind == "keepwarm"
        w._run_keepwarm()
        assert calls == [1]
        assert w.idle_seconds() < 1              # decode stamped the idle clock
        assert w._keepwarm_pending is False      # slot freed for the next cycle
        assert w.metrics()["keepwarm_total"] == 1

    def test_keepwarm_skips_when_a_real_decode_just_ran(self, monkeypatch):
        import interfaces.voice.microphones.local_outposts.WhisperWorker as mod
        calls = []
        monkeypatch.setattr(mod, "transcribe", lambda *a, **k: calls.append(1))
        w = self._worker()
        w._last_decode_at = time.time() - 100
        w.submit_keepwarm()
        w._last_decode_at = time.time()          # a real decode landed while queued
        w._run_keepwarm()
        assert calls == []                       # GPU already warm — no decode

    def test_only_one_keepwarm_pending(self):
        w = self._worker()
        w._last_decode_at = time.time() - 100
        assert w.submit_keepwarm() is True
        assert w.submit_keepwarm() is False      # capped at one queued-or-running
        w._last_decode_at = time.time()          # take the skip path (no real decode)
        w._run_keepwarm()                        # …which still frees the pending slot
        assert w.submit_keepwarm() is True

    def test_real_jobs_dequeue_before_keepwarm(self):
        w = self._worker()
        w.submit_keepwarm()
        w.submit_speculative(None, "m1")
        w.submit_final(None, lambda *a: None, label="m1")
        kinds = [w._jobs.get_nowait()[2].kind for _ in range(3)]
        assert kinds == ["final", "spec", "keepwarm"]

    def test_manager_noise_floors(self):
        import threading
        from types import SimpleNamespace
        from interfaces.voice.microphones.local_outposts.LocalOutpostManager import LocalOutpostManager
        m = LocalOutpostManager.__new__(LocalOutpostManager)
        m._lock = threading.RLock()
        m.outposts = [
            SimpleNamespace(label="Mic A", _energy_vad=SimpleNamespace(
                noise_floor=0.003, threshold=0.009, raise_min=0.001)),
            SimpleNamespace(label="Mic B"),  # not started yet — no energy gate
        ]
        assert m.noise_floors() == [("Mic A", 0.003, 0.009, 0.001)]

    def test_manager_gate(self):
        import threading
        from types import SimpleNamespace
        from interfaces.voice.microphones.local_outposts.LocalOutpostManager import LocalOutpostManager
        m = LocalOutpostManager.__new__(LocalOutpostManager)
        m._lock = threading.RLock()
        submitted = []
        m._worker = SimpleNamespace(idle_seconds=lambda: 100.0,
                                    submit_keepwarm=lambda: submitted.append(1))
        m.outposts = [SimpleNamespace(capturing_speech=False),
                      SimpleNamespace(capturing_speech=True)]
        m._keep_warm_tick()
        assert submitted == []                   # a mic is mid-utterance — hold off
        m.outposts[1].capturing_speech = False
        m._keep_warm_tick()
        assert submitted == [1]                  # idle + quiet → keep-warm submitted
        m._worker = SimpleNamespace(idle_seconds=lambda: 5.0,
                                    submit_keepwarm=lambda: submitted.append(2))
        m._keep_warm_tick()
        assert submitted == [1]                  # GPU recently used — nothing to do

    def test_manager_gate_survives_missing_worker(self):
        import threading
        from interfaces.voice.microphones.local_outposts.LocalOutpostManager import LocalOutpostManager
        m = LocalOutpostManager.__new__(LocalOutpostManager)
        m._lock = threading.RLock()
        m._worker = None
        m.outposts = []
        # No worker yet must be a no-op, not an AttributeError: the tick runs first in
        # _recovery_loop's shared try, so raising here would abort mic auto-recovery.
        m._keep_warm_tick()

    def test_keepwarm_ticks_with_auto_recovery_disabled(self, monkeypatch):
        # The keep-warm tick shares the recovery loop's thread but must not die with
        # the recovery FEATURE: auto_recovery=false only gates the mic watchdog.
        import threading
        import importlib
        mod = importlib.import_module("interfaces.voice.microphones.local_outposts.LocalOutpostManager")
        monkeypatch.setattr(mod, "RECOVERY_TICK_INTERVAL", 0.01)
        m = mod.LocalOutpostManager.__new__(mod.LocalOutpostManager)
        m.auto_recovery = False
        m._stop_recovery = threading.Event()
        calls = []

        def warm():
            calls.append("warm")
            m._stop_recovery.set()

        m._keep_warm_tick = warm
        m._recovery_tick = lambda: calls.append("recover")
        m._recovery_loop()
        assert calls == ["warm"]                 # keep-warm ran; mic watchdog gated off

    def test_recovery_loop_runs_mic_watchdog_when_auto_recovery_enabled(self, monkeypatch):
        # The enabled half of the gate: the fix turned an unconditional
        # _recovery_tick call into a gated one, and only the disabled side was
        # pinned — inverting the gate would silently kill the mic watchdog with
        # every existing test staying green.
        import threading
        import importlib
        mod = importlib.import_module("interfaces.voice.microphones.local_outposts.LocalOutpostManager")
        monkeypatch.setattr(mod, "RECOVERY_TICK_INTERVAL", 0.01)
        m = mod.LocalOutpostManager.__new__(mod.LocalOutpostManager)
        m.auto_recovery = True
        m._stop_recovery = threading.Event()
        calls = []

        def warm():
            calls.append("warm")
            m._stop_recovery.set()

        m._keep_warm_tick = warm
        m._recovery_tick = lambda: calls.append("recover")
        m._recovery_loop()
        assert calls == ["warm", "recover"]      # watchdog runs in the same tick

    def test_ticker_thread_starts_with_auto_recovery_disabled(self):
        import threading
        from interfaces.voice.microphones.local_outposts.LocalOutpostManager import LocalOutpostManager
        m = LocalOutpostManager.__new__(LocalOutpostManager)
        m.auto_recovery = False
        m._stop_recovery = threading.Event()
        m._stop_recovery.set()                   # loop exits on its first wait
        m._start_recovery_loop()
        assert m._recovery_thread is not None
        m._recovery_thread.join(timeout=2)
        assert not m._recovery_thread.is_alive()

    def test_real_decodes_stamp_the_idle_clock(self, monkeypatch):
        # The linchpin of the whole feature: every REAL decode must stamp
        # _last_decode_at. Lose the spec stamp and keep-warm fires mid-conversation;
        # lose the final stamp and it never fires at all — either way the gate tests
        # above stay green because they set the clock by hand.
        import interfaces.voice.microphones.local_outposts.WhisperWorker as mod
        monkeypatch.setattr(mod, "transcribe",
                            lambda *a, **k: ("hi", "en", 0.1, 1.0, 0.1, {"conf": 0.9}))
        w = self._worker()
        w._last_decode_at = time.time() - 100
        w._run_spec(mod._Job("spec", np.zeros(8000, np.float32), spec_id=1, label="m1"))
        assert w.idle_seconds() < 1
        w._last_decode_at = time.time() - 100
        w._run_final(mod._Job("final", np.zeros(8000, np.float32),
                              on_result=lambda *a: None, label="m1"))
        assert w.idle_seconds() < 1

    def test_run_loop_dispatches_keepwarm(self, monkeypatch):
        # The gate tests call _run_keepwarm directly; this pins run()'s dispatch of the
        # "keepwarm" kind — a dropped elif leaves the job falling silently through the
        # chain and the feature dead in prod with a green suite.
        import threading
        import interfaces.voice.microphones.local_outposts.WhisperWorker as mod
        decoded = threading.Event()
        monkeypatch.setattr(mod, "transcribe",
                            lambda *a, **k: decoded.set() or ("", "en", 0.6, 1.0, 0.02, None))
        w = self._worker()
        w._last_decode_at = time.time() - 100
        w.submit_keepwarm()
        w.start()
        assert decoded.wait(2.0)
        w.stop()
        w.join(2.0)


@pytest.mark.unit
class TestWorkerPriority:
    def _worker(self):
        from interfaces.voice.microphones.local_outposts.WhisperWorker import WhisperWorker
        return WhisperWorker("repo", "en")

    def test_final_dequeued_before_pending_speculative(self):
        # The core fix: a final must jump ahead of an already-queued speculative,
        # so the first mic's result isn't stuck behind a second mic's speculative.
        w = self._worker()
        sid = w.submit_speculative(None, "m1")
        w.submit_final(None, lambda *a: None, spec_id=sid, label="m1")
        kinds = [w._jobs.get_nowait()[2].kind for _ in range(2)]
        assert kinds == ["final", "spec"]

    def test_final_promotes_cached_spec_without_decoding(self, monkeypatch):
        # A cached speculative result is delivered synchronously at submit time — the
        # worker may be mid-decode on another mic's final, which no queue jump can
        # preempt, so queuing the promotion would lose the server's first-arrival race.
        import interfaces.voice.microphones.local_outposts.WhisperWorker as mod
        calls = []
        monkeypatch.setattr(mod, "transcribe", lambda *a, **k: calls.append(1))
        w = self._worker()
        spec_audio = np.zeros(8000, np.float32)
        w._spec_results[5] = ("cached text", "en", {"conf": 0.8}, 0.7, 0.35, 1.0, 1.1, 1.8,
                              spec_audio)
        got = []
        final_audio = np.zeros(16000, np.float32)
        w.submit_final(final_audio, lambda t, m, a, l, tm: got.append((t, l, a, tm)),
                       spec_id=5, label="m1")
        assert len(got) == 1
        text, language, audio, timing = got[0]
        assert text == "cached text" and language == "en"
        assert audio is final_audio            # full-utterance audio, not the spec's
        assert timing["decode"] == "spec"
        assert timing["audio"] is spec_audio   # …but the span that was decoded rides along
        assert calls == []                     # promoted, no decode
        assert w._jobs.empty()                 # delivered synchronously — nothing queued
        assert not w._spec_results             # cache consumed

    def test_promoted_delivery_failure_does_not_propagate(self, monkeypatch):
        # Synchronous delivery runs the callback on the calling (endpoint) thread —
        # the exception guard is all that keeps a callback failure from killing the
        # mic loop, which the old queued promotion never risked.
        import interfaces.voice.microphones.local_outposts.WhisperWorker as mod
        calls = []
        monkeypatch.setattr(mod, "transcribe", lambda *a, **k: calls.append(1))
        w = self._worker()
        w._spec_results[5] = ("cached text", "en", {"conf": 0.8}, 0.7, 0.35, 1.0, 1.1, 1.8, np.zeros(8000, np.float32))

        def boom(*a):
            raise RuntimeError("callback exploded")

        w.submit_final(np.zeros(16000, np.float32), boom, spec_id=5, label="m1")
        assert w._jobs.empty()                 # failed delivery is not retried via the queue
        assert not w._spec_results             # cache consumed exactly once
        assert calls == []                     # and no fallback decode either

    def test_final_cancels_pending_spec_when_no_cached_result(self, monkeypatch):
        import interfaces.voice.microphones.local_outposts.WhisperWorker as mod
        monkeypatch.setattr(mod, "transcribe", lambda *a, **k: ("hi", "sv", 0.1, 1.0, 0.1, {"conf": 0.9}))
        w = self._worker()
        got = []
        audio = np.zeros(8000, np.float32)  # > MIN_SPEECH
        w._run_final(mod._Job("final", audio, spec_id=7, on_result=lambda t, m, a, l, tm: got.append((t, l)), label="m1"))
        assert got == [("hi", "sv")]   # decoded fresh, detected language forwarded
        assert 7 in w._cancelled       # its stale speculative will be skipped

    def test_cancelled_speculative_is_skipped(self, monkeypatch):
        import interfaces.voice.microphones.local_outposts.WhisperWorker as mod
        calls = []
        monkeypatch.setattr(mod, "transcribe", lambda *a, **k: calls.append(1) or ("x", "en", 0, 1, 0, None))
        w = self._worker()
        w.cancel(9)
        w._run_spec(mod._Job("spec", np.zeros(8000, np.float32), spec_id=9, label="m1"))
        assert calls == []  # skipped, never decoded

    def test_only_one_speculative_in_flight(self):
        # Every mic fires a speculative on each pause; unbounded they pile decodes onto the
        # single GPU/GIL thread and starve capture. The 2nd is skipped while one is in flight.
        w = self._worker()
        audio = np.zeros(8000, np.float32)
        assert w.submit_speculative(audio, "m1") is not None
        assert w.submit_speculative(audio, "m2") is None   # capped

    def test_speculative_slot_frees_after_decode(self, monkeypatch):
        import interfaces.voice.microphones.local_outposts.WhisperWorker as mod
        monkeypatch.setattr(mod, "transcribe", lambda *a, **k: ("hi", "en", 0.1, 1.0, 0.1, {"conf": 0.9}))
        w = self._worker()
        audio = np.zeros(8000, np.float32)
        w.submit_speculative(audio, "m1")
        assert w.submit_speculative(audio, "m2") is None    # one in flight
        _, _, job = w._jobs.get_nowait()                    # the worker thread would dequeue + run it
        w._run_spec(job)
        assert w.submit_speculative(audio, "m3") is not None  # slot freed, a new spec can run

    def test_running_spec_delivers_waiting_final_on_completion(self, monkeypatch):
        # The scheduling fix: a final whose speculative is mid-decode is
        # delivered the moment that decode completes, instead of its ready result
        # queuing behind another mic's redundant full decode (~0.7s of dead latency).
        import interfaces.voice.microphones.local_outposts.WhisperWorker as mod
        monkeypatch.setattr(mod, "transcribe", lambda *a, **k: ("hi", "en", 0.1, 1.0, 0.1, {"conf": 0.9}))
        w = self._worker()
        spec_audio = np.zeros(8000, np.float32)
        final_audio = np.zeros(16000, np.float32)
        sid = w.submit_speculative(spec_audio, "m1")
        _, _, spec_job = w._jobs.get_nowait()   # worker thread picked up the spec…
        got = []
        w.submit_final(final_audio, lambda t, m, a, l, tm: got.append((t, a, tm)), spec_id=sid, label="m1")
        w._run_spec(spec_job)                   # …and completes it while the final waits
        assert len(got) == 1
        text, audio, timing = got[0]
        assert text == "hi" and audio is final_audio  # final's full-utterance audio, not the spec's
        assert timing["audio"] is spec_audio          # the span the transcript came from
        assert timing["submitted_at"] <= timing["started_at"] <= timing["done_at"]
        _, _, final_job = w._jobs.get_nowait()
        w._run_final(final_job)                 # queued final pops later — must be a no-op
        assert len(got) == 1

    def test_cached_spec_result_beats_other_finals(self):
        # A final whose speculative already finished delivers synchronously at submit —
        # it never enters the queue, so another mic's queued (or even already-running)
        # fresh decode can't delay the ready transcript.
        w = self._worker()
        w._spec_results[3] = ("cached", "en", {"conf": 0.8}, 0.7, 0.35, 1.0, 1.1, 1.8,
                              np.zeros(8000, np.float32))
        got = []
        w.submit_final(None, lambda *a: None, label="m2")             # other mic, fresh decode
        w.submit_final(None, lambda t, *a: got.append(t), spec_id=3, label="m1")
        assert got == ["cached"]       # delivered before m2's decode ever runs
        labels = [w._jobs.get_nowait()[2].label for _ in range(1)]
        assert labels == ["m2"]        # only the fresh decode is queued

    def test_pending_final_cleared_when_final_decodes_fresh(self, monkeypatch):
        # Final pops before its speculative ever ran: it decodes fresh and cancels the
        # spec — the pending registration must go with it so the skipped spec can never
        # deliver a second copy.
        import interfaces.voice.microphones.local_outposts.WhisperWorker as mod
        monkeypatch.setattr(mod, "transcribe", lambda *a, **k: ("fresh", "en", 0.1, 1.0, 0.1, {"conf": 0.9}))
        w = self._worker()
        audio = np.zeros(8000, np.float32)
        sid = w.submit_speculative(audio, "m1")
        _, _, spec_job = w._jobs.get_nowait()
        got = []
        w.submit_final(audio, lambda t, m, a, l, tm: got.append(t), spec_id=sid, label="m1")
        _, _, final_job = w._jobs.get_nowait()
        w._run_final(final_job)
        assert got == ["fresh"]
        assert not w._pending_finals
        w._run_spec(spec_job)   # cancelled — skipped, no double delivery
        assert got == ["fresh"]

    def test_slot_frees_even_when_decode_raises(self, monkeypatch):
        import interfaces.voice.microphones.local_outposts.WhisperWorker as mod
        monkeypatch.setattr(mod, "transcribe", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        w = self._worker()
        audio = np.zeros(8000, np.float32)
        w.submit_speculative(audio, "m1")
        _, _, job = w._jobs.get_nowait()
        with pytest.raises(RuntimeError):
            w._run_spec(job)
        assert w.submit_speculative(audio, "m2") is not None  # finally released the slot


class _ScriptedVAD:
    """Feeds the real _endpoint_loop a pre-planned per-frame verdict sequence."""
    name = "scripted"

    def __init__(self, verdicts):
        self._verdicts = list(verdicts)
        self.threshold = 0.01
        self.last_score = 0.0

    def is_speech(self, frame, speech_hint=False):
        return bool(self._verdicts.pop(0)) if self._verdicts else False

    def reset(self):
        pass


class _ScriptedFrames:
    """Stands in for the capture queue: one 100ms frame per scripted verdict, then
    stops the loop."""

    def __init__(self, n, stop_event):
        self._left = n
        self._stop = stop_event

    def get(self, timeout=None):
        if self._left == 0:
            self._stop.set()
            raise queue.Empty
        self._left -= 1
        return np.zeros(1600, np.float32)


def drive_endpoint_loop(outpost, silero, energy):
    """Run the real _endpoint_loop over parallel per-frame verdict scripts (1=loud,
    0=quiet) for the main (silero) VAD and the shadow energy gate."""
    assert len(silero) == len(energy)
    outpost._vad = _ScriptedVAD(silero)
    outpost._energy_vad = _ScriptedVAD(energy)
    if outpost._worker.submit_speculative.side_effect is None:
        outpost._worker.submit_speculative.side_effect = list(range(10, 20))
    outpost._last_frame_at = time.time()
    outpost._audio_q = _ScriptedFrames(len(silero), outpost._stop)
    outpost._endpoint_loop()
    return outpost


@pytest.mark.unit
class TestEndpointLoopSpeculativeCancel:
    """The trigger-symmetric speculative cancel: an energy-fired spec dies
    when ENERGY resumes; a silero-fired spec dies only when SILERO hears speech again.
    Cross-cancelling silero-fired specs on energy-loud killed every fallback spec
    ~100ms after birth whenever post-reply hiss sat above the energy threshold.
    Frame math at 100ms frames: a spec fires after 2 quiet frames (SPEC_SILENCE=0.15)
    once >=3 speech frames are captured (MIN_SPEECH=0.3); 4 silero-quiet frames end
    the utterance (SILENCE=0.4)."""

    def test_energy_fired_spec_cancelled_when_energy_resumes(self):
        # Silero rides loud through its offset decay while energy goes quiet → the
        # spec fires on the energy gate; energy coming back means speech restarted.
        o = drive_endpoint_loop(make_outpost()[0],
                                silero=[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                                energy=[1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0])
        assert o._worker.cancel.call_count == 1
        o._worker.cancel.assert_called_once_with(10)
        assert o._worker.submit_final.call_count == 1

    def test_energy_fired_spec_survives_silero_offset_decay(self):
        # Silero staying loud over an energy-fired spec is just offset decay (under the
        # SPEC_STALE_SPEECH_GRACE) — the spec must survive to be promoted at VAD stop.
        o = drive_endpoint_loop(make_outpost()[0],
                                silero=[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                                energy=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        o._worker.cancel.assert_not_called()
        assert o._worker.submit_final.call_args.kwargs["spec_id"] == 10

    def test_energy_fired_spec_cancelled_at_stop_after_soft_speech(self):
        # Silero kept reporting speech past the grace after the spec fired — that was
        # likely genuinely soft speech the spec's audio lacks. Promote nothing: cancel
        # at VAD stop so the final decodes the full utterance fresh.
        o = drive_endpoint_loop(make_outpost()[0],
                                silero=[1] * 12 + [0] * 4,
                                energy=[1] * 4 + [0] * 12)
        o._worker.cancel.assert_called_once_with(10)
        assert o._worker.submit_final.call_args.kwargs["spec_id"] is None

    def test_silero_fired_spec_survives_energy_loud(self):
        # The bug itself: a silero-fallback spec fired precisely because the
        # energy gate was misreading sustained noise as loud, so energy-loud frames are
        # its normal environment — they must NOT cancel it.
        o = drive_endpoint_loop(make_outpost()[0],
                                silero=[1, 1, 1, 1, 0, 0, 0, 0],
                                energy=[0, 0, 0, 0, 0, 0, 1, 1])
        o._worker.cancel.assert_not_called()
        assert o._worker.submit_final.call_args.kwargs["spec_id"] == 10

    def test_silero_fired_spec_cancelled_when_silero_hears_speech(self):
        # The pre-hybrid rule still holds for silero-fired specs: silero hearing speech
        # again makes the utterance-so-far stale.
        o = drive_endpoint_loop(make_outpost()[0],
                                silero=[1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0],
                                energy=[0] * 11)
        o._worker.cancel.assert_called_once_with(10)

    def test_capturing_speech_true_until_final_submitted(self):
        outpost, _ = make_outpost()
        seen = []
        outpost._worker.submit_final.side_effect = (
            lambda *a, **k: seen.append(outpost.capturing_speech))
        assert outpost.capturing_speech is False
        drive_endpoint_loop(outpost, silero=[1, 1, 1, 1, 0, 0, 0, 0], energy=[0] * 8)
        # The keep-warm gate reads this flag across threads: submit_final must run
        # while it is still True (a real decode is queued before the gate reopens),
        # and the utterance reset must clear it afterwards — a flag stuck True
        # disables keep-warm system-wide.
        assert seen == [True]
        assert outpost.capturing_speech is False


@pytest.mark.unit
class TestRestrictedDetection:
    """Lazy restricted detection: open-set auto-detect is flaky on short audio (it can
    return a spurious language like 'ar' even when the text is fine). When it lands
    outside the configured set we re-pick within the set instead of dropping the command."""

    def test_in_set_language_kept_without_extra_pass(self, monkeypatch):
        import interfaces.voice.microphones.local_outposts.WhisperWorker as mod
        called = []
        monkeypatch.setattr(mod, "_detect_restricted", lambda *a: called.append(1) or "en")
        assert mod._apply_restricted_detection(AUDIO, "repo", "sv", {"en", "sv"}) == "sv"
        assert called == []   # already allowed → no extra (expensive) detection pass

    def test_out_of_set_language_repicked_and_logged(self, monkeypatch):
        import interfaces.voice.microphones.local_outposts.WhisperWorker as mod
        monkeypatch.setattr(mod, "_detect_restricted", lambda audio, repo, allowed: "en")
        logged = []
        monkeypatch.setattr(mod, "fine", lambda m: logged.append(m))
        assert mod._apply_restricted_detection(AUDIO, "repo", "ar", {"en", "sv"}) == "en"
        assert logged and "'ar'" in logged[0]   # the extra pass is logged so it's not missed

    def test_no_allowed_set_is_noop(self, monkeypatch):
        import interfaces.voice.microphones.local_outposts.WhisperWorker as mod
        called = []
        monkeypatch.setattr(mod, "_detect_restricted", lambda *a: called.append(1))
        assert mod._apply_restricted_detection(AUDIO, "repo", "ar", None) == "ar"
        assert called == []   # single-language / unrestricted → never re-picks


@pytest.mark.unit
class TestMicDiscovery:
    def test_filters_non_input_virtual_and_excluded(self):
        devices = [
            {"name": "MacBook Pro Microphone", "max_input_channels": 1},
            {"name": "MacBook Pro Speakers", "max_input_channels": 0},
            {"name": "BlackHole 2ch", "max_input_channels": 2},
            {"name": "Aggregate Device", "max_input_channels": 2},
            {"name": "USB Conference Mic", "max_input_channels": 1},
        ]
        mics = filter_input_devices(devices, exclude_names=["conference"])
        assert mics == [(0, "MacBook Pro Microphone")]

    def test_a_bare_string_exclude_is_one_hint_not_one_per_character(self):
        devices = [
            {"name": "MacBook Pro Microphone", "max_input_channels": 1},
            {"name": "Acme Headset", "max_input_channels": 1},
        ]
        assert filter_input_devices(devices, exclude_names="acme") == [(0, "MacBook Pro Microphone")]

    def test_teams_and_iphone_skipped_by_default(self):
        devices = [
            {"name": "MacBook Pro Microphone", "max_input_channels": 1},
            {"name": "Microsoft Teams Audio", "max_input_channels": 1},
            {"name": "iPhone Microphone", "max_input_channels": 1},
        ]
        assert filter_input_devices(devices, exclude_names=[]) == [(0, "MacBook Pro Microphone")]

    def test_no_exclude_keeps_all_real_mics(self):
        devices = [
            {"name": "Built-in Mic", "max_input_channels": 1},
            {"name": "Studio Mic", "max_input_channels": 2},
        ]
        assert filter_input_devices(devices, exclude_names=[]) == [(0, "Built-in Mic"), (1, "Studio Mic")]

    def test_duplicate_names_get_numbered(self):
        mics = [(1, "Studio Display Microphone"), (3, "Studio Display Microphone"), (5, "Built-in")]
        assert disambiguate_mic_names(mics) == [
            (1, "Studio Display Microphone #1"),
            (3, "Studio Display Microphone #2"),
            (5, "Built-in"),  # unique name left untouched
        ]


@pytest.mark.unit
class TestMaxOutposts:
    def test_discover_caps_to_max_outposts(self, monkeypatch):
        import sys
        mod = sys.modules["interfaces.voice.microphones.local_outposts.LocalOutpostManager"]
        fake_sd = MagicMock()
        fake_sd.query_devices.return_value = [
            {"name": "Mic One", "max_input_channels": 1},
            {"name": "Mic Two", "max_input_channels": 1},
            {"name": "Mic Three", "max_input_channels": 1},
        ]
        monkeypatch.setitem(__import__("sys").modules, "sounddevice", fake_sd)
        monkeypatch.setattr(mod, "MAX_OUTPOSTS", 1)
        manager = LocalOutpostManager(8888)
        assert manager._discover_mics() == [(0, "Mic One")]

    def test_none_keeps_all(self, monkeypatch):
        import sys
        mod = sys.modules["interfaces.voice.microphones.local_outposts.LocalOutpostManager"]
        fake_sd = MagicMock()
        fake_sd.query_devices.return_value = [
            {"name": "Mic One", "max_input_channels": 1},
            {"name": "Mic Two", "max_input_channels": 1},
        ]
        monkeypatch.setitem(__import__("sys").modules, "sounddevice", fake_sd)
        monkeypatch.setattr(mod, "MAX_OUTPOSTS", None)
        manager = LocalOutpostManager(8888)
        assert manager._discover_mics() == [(0, "Mic One"), (1, "Mic Two")]


@pytest.mark.unit
class TestManagerCommandRouting:
    def _manager_with_outposts(self):
        manager = LocalOutpostManager(8888, languages=["sv-SE"])
        manager.outposts = [make_outpost()[0], make_outpost()[0]]
        manager._active = True
        return manager

    def test_single_language_forces_whisper(self):
        manager = LocalOutpostManager(8888, languages=["sv-SE"])
        assert manager.whisper_language == "sv"      # forced, not auto-detected
        assert manager.default_lang == "sv-SE"
        assert manager.lang_by_subtag == {"sv": "sv-SE"}

    def test_multi_language_autodetects_and_maps(self):
        manager = LocalOutpostManager(8888, languages=["en-US", "sv-SE"])
        assert manager.whisper_language is None       # auto-detect across the set
        assert manager.default_lang == "en-US"        # primary = first
        assert manager.lang_by_subtag == {"en": "en-US", "sv": "sv-SE"}

    def test_default_language_when_none_given(self):
        manager = LocalOutpostManager(8888)
        assert manager.whisper_language == "en" and manager.default_lang == "en-US"

    def test_stop_mutes_all(self):
        manager = self._manager_with_outposts()
        manager.on_command("stop")
        assert all(o.muted for o in manager.outposts)

    def test_listen_unmutes_and_drops_wakeword_gate(self):
        manager = self._manager_with_outposts()
        manager.on_command("stop")
        manager.on_command("listen")
        assert all(not o.muted and not o.awaiting_wakeword for o in manager.outposts)

    def test_wakeword_returns_to_idle(self):
        manager = self._manager_with_outposts()
        manager.on_command("listen")
        manager.on_command("wakeword")
        assert all(not o.muted and o.awaiting_wakeword for o in manager.outposts)

    def test_commands_ignored_when_inactive(self):
        manager = self._manager_with_outposts()
        manager._active = False
        manager.on_command("stop")
        assert all(not o.muted for o in manager.outposts)


@pytest.mark.unit
class TestScribeStatusPublishing:
    """The manager must publish RUNTIME Scribe availability to scribe_status: the wake-word
    verdict gate keys off it, and holding replies for a verdict that can never arrive
    (ScribeWorker failed at boot → Whisper-only fallback) would silently tax every
    wake-word reply with the full wait cap — the exact hole 4db1a9fc closed."""

    @pytest.fixture(autouse=True)
    def reset_flag(self):
        from interfaces.voice.microphones.local_outposts.scribe_status import set_scribe_running
        set_scribe_running(False)
        yield
        set_scribe_running(False)

    def _started_manager(self, monkeypatch, scribe_ctor):
        mod = sys.modules["interfaces.voice.microphones.local_outposts.LocalOutpostManager"]
        monkeypatch.setattr(mod, "WhisperWorker", MagicMock())
        monkeypatch.setattr(mod, "model_cached", lambda repo: True)
        monkeypatch.setattr(mod, "OutpostStreamClient", MagicMock())
        monkeypatch.setattr(mod, "ScribeWorker", scribe_ctor)
        monkeypatch.setattr(mod, "SCRIBE_ENABLED", True)
        manager = LocalOutpostManager(8888)
        monkeypatch.setattr(manager, "_discover_mics", lambda: [(0, "Test Mic")])
        monkeypatch.setattr(manager, "_start_outpost",
                            lambda idx, name: manager.outposts.append(MagicMock(device_name=name)))
        monkeypatch.setattr(manager, "_start_recovery_loop", lambda: None)
        monkeypatch.setattr(manager, "_start_monitor_loop", lambda: None)
        assert manager.start() is True
        return manager

    def test_scribe_up_publishes_running(self, monkeypatch):
        from interfaces.voice.microphones.local_outposts.scribe_status import scribe_is_running
        self._started_manager(monkeypatch, MagicMock())
        assert scribe_is_running() is True

    def test_scribe_boot_failure_publishes_not_running(self, monkeypatch):
        # Missing API key / deps: the manager continues Whisper-only. The gate must learn
        # this — with the flag stuck True no verdict can ever settle it.
        from interfaces.voice.microphones.local_outposts.scribe_status import scribe_is_running
        self._started_manager(monkeypatch, MagicMock(side_effect=RuntimeError("no API key")))
        assert scribe_is_running() is False

    def test_stop_clears_running_flag(self, monkeypatch):
        from interfaces.voice.microphones.local_outposts.scribe_status import scribe_is_running
        manager = self._started_manager(monkeypatch, MagicMock())
        assert scribe_is_running() is True
        manager.stop()
        assert scribe_is_running() is False


# ===========================================================================
# Wake-word configuration (commit ae8f065a)
# ===========================================================================
#
# constants.WAKE_WORDS and INITIAL_PROMPT_PHRASES are derived from
# config.yaml's `wake_word` (+ optional `wake_word_aliases`). The values are
# computed at module import, so the test reloads the module under a patched
# config to verify the derivation — guarding against a refactor that
# accidentally pins the wake word back to a literal.


def _reload_constants_with_wake_word(wake_word, aliases):
    """Reload interfaces.voice.microphones.local_outposts.constants with config.get('wake_word') /
    ('wake_word_aliases') overridden — returns (reloaded_module, restore_fn).
    Always call restore_fn (in a try/finally) so later tests in this session
    see the real config-derived constants."""
    import importlib
    from common.ConfigManager import config as cfg
    from interfaces.voice.microphones.local_outposts import constants as constants_module

    real_get = cfg.get

    def fake_get(key, default=None):
        if key == 'wake_word':
            return wake_word if wake_word is not None else default
        if key == 'wake_word_aliases':
            return aliases if aliases is not None else (default if default is not None else [])
        return real_get(key, default)

    cfg.get = fake_get
    try:
        reloaded = importlib.reload(constants_module)
    except Exception:
        cfg.get = real_get
        importlib.reload(constants_module)
        raise

    def restore():
        cfg.get = real_get
        importlib.reload(constants_module)

    return reloaded, restore


@pytest.mark.unit
class TestWakeWordConfig:
    def test_wake_words_uses_configured_value_and_includes_aliases(self):
        # Aliases compensate for Whisper mishearing the wake word; both the
        # canonical name and each alias must end up in WAKE_WORDS, lowercased.
        constants, restore = _reload_constants_with_wake_word("Floxy", ["floxi", "phlox"])
        try:
            assert constants.WAKE_WORD == "Floxy"
            assert constants.WAKE_WORDS == ("floxy", "floxi", "phlox")
        finally:
            restore()

    def test_wake_words_falls_back_to_michelle_when_unset(self):
        # Existing deployments without the new key must keep matching "Michelle".
        constants, restore = _reload_constants_with_wake_word(None, None)
        try:
            assert constants.WAKE_WORD == "Michelle"
            assert "michelle" in constants.WAKE_WORDS
        finally:
            restore()

    def test_wake_words_no_aliases_yields_only_canonical(self):
        # Aliases are optional. Missing key must not crash and must produce
        # a single-entry tuple — not an unpacked-None failure.
        constants, restore = _reload_constants_with_wake_word("Aria", None)
        try:
            assert constants.WAKE_WORDS == ("aria",)
        finally:
            restore()

    def test_initial_prompt_phrases_name_the_configured_wake_word(self):
        # The Whisper priming phrase embeds the wake word so the model biases
        # toward the assistant's actual name. If the substitution ever breaks
        # the priming would seed a stale name, hurting recognition.
        constants, restore = _reload_constants_with_wake_word("Aria", None)
        try:
            assert all("Aria" in phrase for phrase in constants.INITIAL_PROMPT_PHRASES)
        finally:
            restore()


# ===========================================================================
# Near-miss config derivation (commits d19b5f01, 889f1485)
# ===========================================================================
#
# WAKE_WORD_NEAR_MISSES comes from config.yaml's optional `wake_word_near_misses`,
# with an _UNSET sentinel distinguishing "key missing" (apply the Michelle-tuned
# defaults, and note() at startup if the wake word is custom) from "explicitly
# set" (including [] = rescue disabled). Like WAKE_WORDS above, the value is
# computed at module import, so these tests reload the module under a patched
# config — guarding the sentinel dance, the whitespace normalization the
# gate_scribe_verify matcher relies on, and the startup note.

_KEY_UNSET = object()   # simulate `wake_word_near_misses` missing from config.yaml


def _reload_constants_with_near_misses(near_misses, wake_word='Michelle'):
    """Reload constants with config.get('wake_word_near_misses') and ('wake_word')
    overridden, capturing note() calls emitted during the import. Returns
    (reloaded_module, notes, restore_fn); always call restore_fn (in a
    try/finally) so later tests see the real config-derived constants."""
    import importlib
    from common import logging as logging_module
    from common.ConfigManager import config as cfg
    from interfaces.voice.microphones.local_outposts import constants as constants_module

    real_get = cfg.get
    real_note = logging_module.note
    notes = []

    def fake_get(key, default=None):
        if key == 'wake_word_near_misses':
            return default if near_misses is _KEY_UNSET else near_misses
        if key == 'wake_word':
            return wake_word
        return real_get(key, default)

    cfg.get = fake_get
    logging_module.note = notes.append
    try:
        reloaded = importlib.reload(constants_module)
    except Exception:
        cfg.get = real_get
        logging_module.note = real_note
        importlib.reload(constants_module)
        raise

    def restore():
        cfg.get = real_get
        logging_module.note = real_note
        importlib.reload(constants_module)

    return reloaded, notes, restore


@pytest.mark.unit
class TestNearMissConfig:
    def test_unset_key_applies_defaults_quietly_under_michelle(self):
        # Existing deployments without the key keep the evidence-backed Michelle
        # renderings, and the startup note must not nag the default setup.
        constants, notes, restore = _reload_constants_with_near_misses(_KEY_UNSET)
        try:
            assert constants.WAKE_WORD_NEAR_MISSES == ('mission', 'missions', 'we shall')
            assert notes == []
        finally:
            restore()

    def test_configured_entries_replace_the_defaults(self):
        constants, notes, restore = _reload_constants_with_near_misses(['Flossy', 'flock C'])
        try:
            assert constants.WAKE_WORD_NEAR_MISSES == ('flossy', 'flock c')
        finally:
            restore()

    def test_empty_list_disables_the_rescue(self):
        # config.example.yaml documents [] as "disable": it must yield an empty
        # tuple, not fall back to the Michelle defaults.
        constants, notes, restore = _reload_constants_with_near_misses([])
        try:
            assert constants.WAKE_WORD_NEAR_MISSES == ()
        finally:
            restore()

    def test_entries_are_whitespace_normalized(self):
        # gate_scribe_verify matches space-joined lowercased tokens; an entry
        # configured with stray spacing must still hit ("We  Shall " ≡ "we shall").
        constants, notes, restore = _reload_constants_with_near_misses(['We  Shall '])
        try:
            assert constants.WAKE_WORD_NEAR_MISSES == ('we shall',)
        finally:
            restore()

    def test_custom_wake_word_with_defaulted_near_misses_notes_at_startup(self):
        # Under another wake word the Michelle renderings are ordinary words that
        # would corroborate a Whisper fire instead of vetoing it — the operator
        # must be told to set the key.
        constants, notes, restore = _reload_constants_with_near_misses(
            _KEY_UNSET, wake_word='Aria')
        try:
            assert len(notes) == 1
            assert 'wake_word_near_misses' in notes[0]
        finally:
            restore()

    def test_custom_wake_word_with_explicit_near_misses_stays_quiet(self):
        constants, notes, restore = _reload_constants_with_near_misses(
            ['aria'], wake_word='Aria')
        try:
            assert notes == []
        finally:
            restore()


# ===========================================================================
# Multi-language Whisper priming (commit e4114e6a)
# ===========================================================================
#
# Whisper continues in the language of its initial_prompt. With a single English
# priming phrase, Swedish audio was *translated* to English instead of being
# transcribed ("vad är klockan?" came back as "what is the clock?"). The fix
# builds one priming phrase per configured language. These tests pin that
# behavior so the translation regression cannot quietly reappear.

@pytest.mark.unit
class TestConfiguredPrimingPhrases:
    def _call_with_languages(self, languages):
        from unittest.mock import patch
        from interfaces.voice.microphones.local_outposts import constants
        with patch("interfaces.voice.tts.VoiceSynthesizer.VoiceSynthesizer.configured_languages",
                   return_value=languages):
            return constants._configured_priming_phrases()

    def test_english_only_yields_single_english_phrase(self):
        phrases = self._call_with_languages(["en-US"])
        assert len(phrases) == 1
        assert "home assistant" in phrases[0]

    def test_swedish_added_when_configured_so_whisper_stops_translating(self):
        # The regression that motivated this feature: English-only priming made
        # Whisper translate Swedish to English. Adding the Swedish phrase keeps
        # Swedish audio Swedish.
        phrases = self._call_with_languages(["en-US", "sv-SE"])
        assert any("home assistant" in p for p in phrases)
        assert any("personliga AI-hemassistent" in p for p in phrases)

    def test_order_follows_config_with_primary_first(self):
        # Primary language is first in config; priming order should mirror that
        # so the model's most-recent context is the primary language.
        phrases = self._call_with_languages(["sv-SE", "en-US"])
        assert "personliga AI-hemassistent" in phrases[0]
        assert "home assistant" in phrases[1]

    def test_duplicate_subtag_is_deduplicated(self):
        # en-US and en-GB share the 'en' subtag — they must produce one phrase,
        # not two identical ones (priming-echo matching uses the phrase set as a
        # frozenset, but the prompt itself would still double the same sentence).
        phrases = self._call_with_languages(["en-US", "en-GB"])
        assert len(phrases) == 1

    def test_unknown_language_falls_back_to_english(self):
        # A configured language with no template (e.g. French) must still produce
        # a usable priming phrase rather than a KeyError — the wake word still
        # primes, only the cadence sentence stays English.
        phrases = self._call_with_languages(["fr-FR"])
        assert len(phrases) == 1
        assert "home assistant" in phrases[0]


@pytest.mark.unit
class TestTextLanguageDetector:
    def test_inactive_with_single_language(self):
        det = TextLanguageDetector({"sv"})
        assert det.active is False
        # No arbitration: the audio label passes straight through.
        assert det.arbitrate("Spela sportbil med Miriam Bryant.", "en") == "en"

    def test_overrides_confident_disagreement(self):
        # The exact bug: Whisper heard 'en' for a plainly Swedish command. Text wins.
        det = TextLanguageDetector({"en", "sv"})
        assert det.active is True
        assert det.arbitrate("Spela sportbil med Miriam Bryant.", "en") == "sv"

    def test_keeps_label_on_agreement(self):
        det = TextLanguageDetector({"en", "sv"})
        assert det.arbitrate("Turn on the kitchen lights", "en") == "en"

    def test_keeps_label_below_confidence_threshold(self):
        # A near-tie must not flip the label — override only on confident disagreement.
        det = TextLanguageDetector({"en", "sv"}, min_confidence=0.99)
        assert det.arbitrate("Spela sportbil med Miriam Bryant.", "en") == "en"

    def test_empty_text_keeps_label(self):
        det = TextLanguageDetector({"en", "sv"})
        assert det.arbitrate("", "en") == "en"
        assert det.arbitrate("   ", "sv") == "sv"

    def test_arbitrate_does_not_route_to_unaccepted_candidate(self):
        # for_outpost loads every language as a *candidate* but accepts only 'en'. When the
        # text classifies confidently as a candidate we do NOT accept (Swedish here),
        # arbitrate must KEEP Whisper's audio label, not route to that language — routing is
        # for accepted command languages only; an overheard foreign language is handled by
        # is_foreign's veto, never by routing to it. This is the accepted-vs-candidate split
        # the is_foreign refactor added to arbitrate, and the only arbitrate branch the other
        # tests (all confined to accepted languages) never exercise. Without the guard,
        # arbitrate would start routing commands to languages the server doesn't accept.
        det = TextLanguageDetector.for_outpost({"en"})
        assert det.is_foreign("Spela sportbil med Miriam Bryant.") is True   # confidently sv
        assert det.arbitrate("Spela sportbil med Miriam Bryant.", "en") == "en"

    def test_is_foreign_flags_unaccepted_language(self):
        # English-only deployment: overheard Swedish (with an English wake word prefix) is a
        # non-command language. for_outpost loads ambient 'sv' as a candidate so it classifies.
        det = TextLanguageDetector.for_outpost({"en"})
        assert det.active is True
        assert det.is_foreign("Michelle, har du sett var fjärrkontrollen är?") is True
        assert det.is_foreign("Michelle, turn off the kitchen lights") is False

    def test_is_foreign_false_when_language_accepted(self):
        # Bilingual deployment accepts Swedish — nothing is foreign, the veto is a no-op.
        det = TextLanguageDetector.for_outpost({"en", "sv"})
        assert det.is_foreign("Michelle, har du sett var fjärrkontrollen är?") is False

    def test_is_foreign_false_when_inactive(self):
        # A plain single-language detector (no ambient candidates) can't classify → never veto.
        det = TextLanguageDetector({"en"})
        assert det.active is False
        assert det.is_foreign("Michelle, har du sett var fjärrkontrollen är?") is False

    def test_is_foreign_false_on_empty(self):
        det = TextLanguageDetector.for_outpost({"en"})
        assert det.is_foreign("") is False
        assert det.is_foreign("   ") is False

    def test_is_foreign_requires_min_confidence(self, monkeypatch):
        # The veto's central fail-safe: only a *confident* foreign classification vetoes —
        # below the floor it's a no-op so uncertainty never drops a genuine fire. The other
        # is_foreign tests feed clearly-Swedish text that always classifies far above 0.70, so
        # none of them exercise the threshold gate; stub the classifier to sit either side of
        # it. Guards against the gate being dropped or inverted.
        det = TextLanguageDetector.for_outpost({"en"}, min_confidence=0.70)
        monkeypatch.setattr(det, "_classify", lambda text: ("sv", 0.69))
        assert det.is_foreign("borderline overheard speech") is False  # below floor: no veto
        monkeypatch.setattr(det, "_classify", lambda text: ("sv", 0.70))
        assert det.is_foreign("confident overheard speech") is True   # at floor: veto


@pytest.mark.unit
class TestLanguageArbitrationInGate:
    def test_swedish_text_overrides_whisper_english_label(self):
        # End-to-end through the transcript gate: a Swedish command mislabelled 'en-US'
        # by Whisper must be sent to the server as 'sv-SE'.
        det = TextLanguageDetector({"en", "sv"})
        outpost, client = make_outpost(awaiting_wakeword=True, languages=("en-US", "sv-SE"),
                                       lang_detector=det)
        outpost._on_transcript("Michelle, spela sportbil med Miriam Bryant.", CLEAN, AUDIO, language="en")
        client.send_query.assert_called_once()
        assert client.send_query.call_args[0][1] == "sv-SE"

    def test_wake_word_stripped_before_classification(self):
        # The English wake word "Michelle" must not pull a short Swedish command to 'en'.
        det = TextLanguageDetector({"en", "sv"})
        outpost, client = make_outpost(awaiting_wakeword=True, languages=("en-US", "sv-SE"),
                                       lang_detector=det)
        outpost._on_transcript("Michelle, tänd taklampan.", CLEAN, AUDIO, language="en")
        client.send_query.assert_called_once()
        assert client.send_query.call_args[0][1] == "sv-SE"

    def test_no_detector_keeps_whisper_label(self):
        outpost, client = make_outpost(awaiting_wakeword=True, languages=("en-US", "sv-SE"))
        outpost._on_transcript("Michelle, turn on the lights.", CLEAN, AUDIO, language="en")
        client.send_query.assert_called_once()
        assert client.send_query.call_args[0][1] == "en-US"


@pytest.mark.unit
class TestForeignLanguageVeto:
    """English-only deployment: a wake word inside overheard non-command-language speech is a
    hallucinated address. The same TextLanguageDetector.is_foreign gate the WakeWordTest runs."""

    def test_whisper_foreign_wake_word_is_vetoed(self):
        det = TextLanguageDetector.for_outpost({"en"})
        outpost, client = make_outpost(awaiting_wakeword=True, languages=("en-US",), lang_detector=det)
        outpost._on_transcript("Michelle, har du sett var fjärrkontrollen är?", CLEAN, AUDIO, language="en")
        client.send_query.assert_not_called()

    def test_whisper_english_command_still_fires(self):
        # The veto must not touch genuine English commands in the same English-only setup.
        det = TextLanguageDetector.for_outpost({"en"})
        outpost, client = make_outpost(awaiting_wakeword=True, languages=("en-US",), lang_detector=det)
        outpost._on_transcript("Michelle, turn off the kitchen lights.", CLEAN, AUDIO, language="en")
        client.send_query.assert_called_once()

    def test_bilingual_swedish_command_not_vetoed(self):
        # Swedish IS accepted in a bilingual deployment — the veto stays out of the way and the
        # command routes to sv-SE (arbitration), exactly as before.
        det = TextLanguageDetector.for_outpost({"en", "sv"})
        outpost, client = make_outpost(awaiting_wakeword=True, languages=("en-US", "sv-SE"), lang_detector=det)
        outpost._on_transcript("Michelle, tänd taklampan.", CLEAN, AUDIO, language="en")
        client.send_query.assert_called_once()
        assert client.send_query.call_args[0][1] == "sv-SE"

    def test_scribe_verify_foreign_language_vetoes(self):
        # Scribe verifying a Whisper wake-word send, but the transcript is overheard Swedish
        # (English-only) → treat Whisper's detection as a false positive and kill the request.
        det = TextLanguageDetector.for_outpost({"en"})
        outpost, client = make_outpost(awaiting_wakeword=False, languages=("en-US",), lang_detector=det)
        outpost._on_scribe_transcript("Michelle, har du sett var fjärrkontrollen är?", 0.9, "en", 0.0,
                                      verify_wake_word=True, whisper_query="have you seen where the remote is")
        client.send_false_positive.assert_called_once()
        client.send_query.assert_not_called()
        assert outpost.awaiting_wakeword is True

    def test_is_foreign_strips_wake_word_before_classifying(self):
        # a2ff5427: the English wake word must be stripped before language classification,
        # mirroring _arbitrate_language. On a short overheard clip the wake word would
        # otherwise dilute the foreign signal below threshold and the veto would miss.
        outpost, _ = make_outpost(awaiting_wakeword=True, languages=("en-US",),
                                  lang_detector=MagicMock())
        outpost._is_foreign("Michelle, ja")
        classified = outpost._lang_detector.is_foreign.call_args[0][0]
        assert "michelle" not in classified.lower()  # wake word gone before classification

    def test_is_foreign_false_without_detector(self):
        # The guard: no detector means no veto (the fail-safe path).
        outpost, _ = make_outpost(awaiting_wakeword=True, languages=("en-US",))
        assert outpost._is_foreign("Michelle, har du sett var fjärrkontrollen är?") is False


@pytest.mark.unit
class TestSentenceBoundaryAnchor:
    """A wake word past WAKE_WORD_ANCHOR_CHARS still addresses us when it opens its own
    sentence — the text before it ends with sentence-final punctuation (a production
    miss like "What time is it there? Michelle, pause the music."). Mid-sentence mentions
    stay overheard."""

    def test_wake_word_after_question_addresses_us(self):
        found, rest = strip_wake_word("What time is it there? Michelle, pause the music.")
        assert found is True and rest == ", pause the music."

    def test_boundary_fire_carries_spoken_prefix(self):
        # The prefix confidence floor must guard boundary fires, so the prefix is reported.
        assert wake_word_prefix("What time is it there? Michelle, pause the music.") == \
            "What time is it there? "

    def test_mid_sentence_mention_still_overheard(self):
        text = "Vill du hälsa till Michelle från oss, Frances?"
        assert strip_wake_word(text) == (False, text)

    def test_deep_boundary_still_addresses_us(self):
        # Several completed sentences before the address — depth doesn't matter.
        found, rest = strip_wake_word("Hold on. Let me check. Michelle, turn off the lights")
        assert found is True and rest == ", turn off the lights"

    def test_comma_is_not_a_boundary(self):
        # Only completed speech counts; a clause break mid-sentence is still a mention.
        assert strip_wake_word("and then we told, Michelle about the trip")[0] is False

    def test_later_occurrence_wins_when_first_is_mid_sentence(self):
        # First mention is buried mid-sentence (past the anchor, overheard), the second
        # opens its own sentence — the addressing one wins.
        found, rest = strip_wake_word("They already told Michelle about it. Michelle, lights on")
        assert found is True and rest == ", lights on"


@pytest.mark.unit
class TestWhisperRescueEligibility:
    """gate_whisper_transcript marks a risky drop rescueable (held for Scribe) only for the
    background-music shape: flush wake word + command at conf ≥ 50%, clear of every
    hallucination signature. Everything else drops exactly as before."""

    MUSIC = {"conf": 0.58, "no_speech": 0.51, "compression": 0.76, "temperature": 0.0}

    def test_music_masked_command_is_rescueable(self):
        v = gate_whisper_transcript("Michelle, pause the music.", self.MUSIC)
        assert v.drop is not None and v.rescue is True

    def test_low_confidence_is_not_rescueable(self):
        v = gate_whisper_transcript("Michelle, turn the kitchen lights on.",
                                    {**self.MUSIC, "conf": 0.37})
        assert v.drop is not None and v.rescue is False

    def test_spoken_prefix_is_not_rescueable(self):
        # The 59%-conf priming-echo variant ("you are Michelle. Next sentence…") that a
        # blanket relaxation would admit — the flush requirement excludes it.
        v = gate_whisper_transcript(
            "you are Michelle. Next sentence will be a query to you. Next sentence will "
            "be a query to you.", {**self.MUSIC, "conf": 0.59})
        assert v.drop is not None and v.rescue is False

    def test_bare_wake_word_is_not_rescueable(self):
        # A rescued bare fire would arm a listen window — scope stays commands-only.
        v = gate_whisper_transcript("Michelle.", self.MUSIC)
        assert v.drop is not None and v.rescue is False

    def test_doubled_wake_word_is_not_rescueable(self):
        v = gate_whisper_transcript("Michelle Michelle, pause the music.", self.MUSIC)
        assert v.drop is not None and v.rescue is False

    def test_clean_metrics_fire_normally(self):
        v = gate_whisper_transcript("Michelle, pause the music.", CLEAN)
        assert v.drop is None and v.rescue is False


@pytest.mark.unit
class TestGateScribeRescue:
    """The rescue must POSITIVELY confirm: verify stands, Scribe near-certain, commands
    agree. Weaker confirmations (echo, near-miss) that leave a verify standing refuse."""

    def test_exact_agreement_at_full_confidence_rescues(self):
        v = gate_scribe_rescue("Michelle, pause the music", "Pause the music", scribe_conf=1.0)
        assert v.veto is None

    def test_scribe_confidence_below_floor_refuses(self):
        # A background-speech false positive (Scribe 0.83): agreement alone must not rescue.
        v = gate_scribe_rescue("Michelle, get this", "Get this", scribe_conf=0.83)
        assert v.veto is not None

    def test_disagreeing_commands_refuse(self):
        v = gate_scribe_rescue("Michelle, open the door", "Pause the music", scribe_conf=0.99)
        assert v.veto is not None and "disagree" in v.veto

    def test_no_wake_word_refuses(self):
        v = gate_scribe_rescue("pause the music", "Pause the music", scribe_conf=1.0)
        assert v.veto is not None

    def test_empty_transcript_refuses(self):
        v = gate_scribe_rescue("", "Pause the music", scribe_conf=1.0)
        assert v.veto is not None

    def test_near_miss_opening_refuses(self):
        # "Mission complete" keeps a verify alive but is no positive confirmation.
        v = gate_scribe_rescue("Mission complete", "Good night", scribe_conf=1.0)
        assert v.veto is not None and "near-miss" in v.veto


@pytest.mark.unit
class TestMatchingCommandPrefix:
    """When Scribe's command STARTS with exactly Whisper's command, trailing speech is
    background bleed: the foreign veto is waived and the vote truncated to the agreed
    command (a production miss: Swedish room chatter after the command)."""

    def test_prefix_match_truncates_to_agreed_command(self):
        got = matching_command_prefix(
            "Pause the music", "pause the music. Vi kan väl ta en kopp kaffe sen. Det blir nog bra")
        assert got == "pause the music"

    def test_non_ascii_command_truncates_at_the_agreed_words(self):
        # "släck" is one Unicode \w+ token but TWO _normalize_for_match tokens
        # ("sl ck"); a \w+-based cut counted words differently and kept the bleed.
        got = matching_command_prefix("släck i köket", "släck i köket och sen då")
        assert got == "släck i köket"

    def test_trailing_non_ascii_letter_is_kept(self):
        # "på" normalizes to "p": the ASCII-token cut ended mid-word and returned "sätt p".
        assert matching_command_prefix("sätt på", "sätt på lampan i köket") == "sätt på"
        assert matching_command_prefix("sätt på", "Sätt på.") == "Sätt på"

    def test_no_match_when_commands_differ(self):
        assert matching_command_prefix(
            "Have you seen where the remote is?", "har du sett var fjärrkontrollen är?") is None

    def test_empty_whisper_command_never_matches(self):
        assert matching_command_prefix("", "pause the music") is None

    def test_partial_word_is_not_a_prefix(self):
        assert matching_command_prefix("pause the mu", "pause the music") is None

    def test_verify_waives_foreign_veto_on_agreed_prefix(self):
        det = TextLanguageDetector.for_outpost({"en"})
        v = gate_scribe_verify("Michelle, pause the music. Vi kan väl ta en kopp kaffe sen. "
                               "Det kanske hjälper oss.", "Pause the music", det)
        assert v.veto is None
        assert "kopp kaffe" not in v.rest  # the bleed never becomes part of the vote

    def test_verify_foreign_veto_stands_without_agreement(self):
        det = TextLanguageDetector.for_outpost({"en"})
        v = gate_scribe_verify("Michelle, har du sett var fjärrkontrollen är?",
                               "Have you seen where the remote is?", det)
        assert v.veto is not None and "non-command language" in v.veto


@pytest.mark.unit
class TestScribeRescueWiring:
    """_on_transcript holds a rescueable risky drop for Scribe; _on_scribe_rescue fires
    late on a confirmation, stays silent otherwise. No server request exists until the
    rescue fires, so a refusal has nothing to kill."""

    MUSIC = {"conf": 0.58, "no_speech": 0.51, "compression": 0.76, "temperature": 0.0}

    def _outpost_with_scribe(self):
        outpost, client = make_outpost(awaiting_wakeword=True, languages=("en-US",))
        outpost._scribe = MagicMock()
        return outpost, client, outpost._scribe

    def test_rescueable_drop_submits_scribe_and_sends_nothing(self):
        outpost, client, scribe = self._outpost_with_scribe()
        outpost._on_transcript("Michelle, pause the music.", self.MUSIC, AUDIO, language="en")
        scribe.submit.assert_called_once()
        client.send_query.assert_not_called()
        assert outpost.awaiting_wakeword is True

    def test_confirmed_rescue_fires_with_scribe_command(self):
        outpost, client, scribe = self._outpost_with_scribe()
        outpost._on_transcript("Michelle, pause the music.", self.MUSIC, AUDIO, language="en")
        callback = scribe.submit.call_args[0][2]
        callback("Michelle, pause the music", 1.0, "eng")
        client.send_query.assert_called_once()
        assert client.send_query.call_args[0][0] == "Pause the music"
        assert client.send_query.call_args.kwargs.get("wake_word") is True
        assert client.send_query.call_args.kwargs.get("scribe_skipped") is True
        assert outpost.awaiting_wakeword is False

    def test_refused_rescue_stays_dropped(self):
        outpost, client, scribe = self._outpost_with_scribe()
        outpost._on_transcript("Michelle, pause the music.", self.MUSIC, AUDIO, language="en")
        callback = scribe.submit.call_args[0][2]
        callback("Det var visst ganska kallt ute idag", 0.86, "swe")
        client.send_query.assert_not_called()
        client.send_false_positive.assert_not_called()  # nothing in flight to kill
        assert outpost.awaiting_wakeword is True

    def test_late_rescue_after_another_fire_is_dropped(self):
        outpost, client, scribe = self._outpost_with_scribe()
        outpost._on_transcript("Michelle, pause the music.", self.MUSIC, AUDIO, language="en")
        callback = scribe.submit.call_args[0][2]
        outpost.awaiting_wakeword = False  # another utterance opened a session meanwhile
        callback("Michelle, pause the music", 1.0, "eng")
        client.send_query.assert_not_called()

    def test_non_rescueable_risky_drop_never_consults_scribe(self):
        outpost, client, scribe = self._outpost_with_scribe()
        outpost._on_transcript("Michelle, turn the kitchen lights on.",
                               {**self.MUSIC, "conf": 0.37}, AUDIO, language="en")
        scribe.submit.assert_not_called()
        client.send_query.assert_not_called()


@pytest.mark.unit
class TestWakeWordHallucinationVeto:
    """8b7159ae end-to-end: a wake-word-opening clip with the correlated-hallucination
    signature (both low-confidence AND breathy) is dropped before send_query, while genuine
    fires that trip only one of the two signals still go through."""

    FP = {**CLEAN, "conf": 0.37, "no_speech": 0.196}  # the surviving Swedish FP signature

    def test_correlated_hallucination_signature_is_vetoed(self):
        outpost, client = make_outpost(awaiting_wakeword=True, languages=("en-US",))
        outpost._on_transcript("Michelle, take it all out.", self.FP, AUDIO, language="en")
        client.send_query.assert_not_called()

    def test_breathy_but_confident_fire_still_sent(self):
        outpost, client = make_outpost(awaiting_wakeword=True, languages=("en-US",))
        outpost._on_transcript("Michelle, turn off the TV.",
                               {**CLEAN, "conf": 0.80, "no_speech": 0.196}, AUDIO, language="en")
        client.send_query.assert_called_once()

    def test_veto_does_not_touch_command_continuation(self):
        # The veto applies only to wake-word-opening clips. These metrics (conf 42% / no_speech
        # 0.196) sit in the band that trips the hallucination veto (conf <45 AND no_speech >0.17)
        # yet clears the standard risk floor (conf >40), so a continuation transcript with no
        # wake word still goes through — proving the veto is gated on `found`, not the metrics.
        outpost, client = make_outpost(awaiting_wakeword=False, languages=("en-US",))
        outpost._on_transcript("take it all out.",
                               {**CLEAN, "conf": 0.42, "no_speech": 0.196}, AUDIO, language="en")
        client.send_query.assert_called_once()


# ===========================================================================
# Auto-recovery: USB scan, power-cycle, and offline detection
# ===========================================================================

_UHUBCTL_OUTPUT = (
    "Current status for hub 8-3 [05ac:8009 Apple USB2 Hub 0000A00, USB 2.10, 4 ports, ppps]\n"
    "  Port 1: 0100 power\n"  # powered but nothing connected → skipped
    "  Port 4: 0503 power highspeed enable connect [2752:001c miniDSP VocalFusion Spk (UAC2.0)]\n"
    "Current status for hub 9-1 [1234:5678 Generic Hub, USB 2.10, 2 ports]\n"  # no ppps → unswitchable
    "  Port 1: 0503 power enable connect [aaaa:bbbb USB audio CODEC]\n"
)


@pytest.mark.unit
class TestAutoRecoveryUSB:
    def _manager(self, monkeypatch, output, expected, returncode=0):
        mod = sys.modules["interfaces.voice.microphones.local_outposts.LocalOutpostManager"]
        manager = LocalOutpostManager(8888)
        manager._uhubctl = "/usr/bin/uhubctl"
        manager._expected_names = list(expected)
        fake_run = MagicMock(return_value=SimpleNamespace(stdout=output, stderr="", returncode=returncode))
        monkeypatch.setattr(mod.subprocess, "run", fake_run)
        return manager, fake_run

    def test_scan_maps_mic_to_ppps_port(self, monkeypatch):
        manager, _ = self._manager(monkeypatch, _UHUBCTL_OUTPUT, ["miniDSP VocalFusion Spk (UAC2.0)"])
        assert manager._scan_usb_ports() == {"miniDSP VocalFusion Spk (UAC2.0)": ("8-3", 4)}

    def test_scan_skips_non_ppps_hub(self, monkeypatch):
        # USB audio CODEC sits on a hub without ppps — not power-switchable, so unmapped.
        manager, _ = self._manager(monkeypatch, _UHUBCTL_OUTPUT, ["USB audio CODEC"])
        assert manager._scan_usb_ports() == {}

    def test_scan_returns_none_without_uhubctl(self, monkeypatch):
        # None = "bus state unknown", distinct from {} = "scanned, nothing found".
        manager, fake_run = self._manager(monkeypatch, _UHUBCTL_OUTPUT, ["miniDSP VocalFusion Spk (UAC2.0)"])
        manager._uhubctl = None
        assert manager._scan_usb_ports() is None
        fake_run.assert_not_called()

    def test_scan_returns_none_when_uhubctl_fails(self, monkeypatch):
        mod = sys.modules["interfaces.voice.microphones.local_outposts.LocalOutpostManager"]
        manager, _ = self._manager(monkeypatch, "", ["mic"])
        monkeypatch.setattr(mod.subprocess, "run", MagicMock(side_effect=OSError("boom")))
        assert manager._scan_usb_ports() is None

    def test_power_cycle_issues_uhubctl_cycle(self, monkeypatch):
        manager, fake_run = self._manager(monkeypatch, "", ["mic"])
        manager._port_by_mic = {"mic": ("8-3", 4)}
        assert manager._maybe_power_cycle("mic") is True
        argv = fake_run.call_args[0][0]
        assert argv[:5] == ["/usr/bin/uhubctl", "-a", "cycle", "-l", "8-3"]
        assert "-p" in argv and "4" in argv

    def test_power_cycle_respects_cooldown(self, monkeypatch):
        manager, fake_run = self._manager(monkeypatch, "", ["mic"])
        manager._port_by_mic = {"mic": ("8-3", 4)}
        assert manager._maybe_power_cycle("mic") is True
        fake_run.reset_mock()
        assert manager._maybe_power_cycle("mic") is False  # within cooldown → not re-issued
        fake_run.assert_not_called()

    def test_power_cycle_without_learned_port_is_noop(self, monkeypatch):
        manager, fake_run = self._manager(monkeypatch, "", ["mic"])
        manager._port_by_mic = {}
        assert manager._maybe_power_cycle("mic") is False
        fake_run.assert_not_called()

    def test_power_cycle_reports_uhubctl_failure(self, monkeypatch):
        manager, _ = self._manager(monkeypatch, "", ["mic"], returncode=1)
        manager._port_by_mic = {"mic": ("8-3", 4)}
        assert manager._maybe_power_cycle("mic") is False  # non-zero exit → failed

    def test_power_cycle_gives_up_after_max_attempts(self, monkeypatch):
        # A hung mic that never re-enumerates was cycled ~230 times over
        # 4 h to no effect. Consecutive no-effect cycles must stop at the cap.
        manager, fake_run = self._manager(monkeypatch, "", ["mic"])
        manager._port_by_mic = {"mic": ("8-3", 4)}
        for _ in range(USB_POWER_CYCLE_MAX_ATTEMPTS):
            manager._last_cycle_at.clear()  # bypass the cooldown between attempts
            assert manager._maybe_power_cycle("mic") is True
        assert "mic" in manager._gave_up_cycling
        fake_run.reset_mock()
        manager._last_cycle_at.clear()
        assert manager._maybe_power_cycle("mic") is False  # given up → no more cycles
        fake_run.assert_not_called()

    def test_bus_return_rearms_power_cycling(self, monkeypatch):
        # The give-up is per-drop: the device reappearing (replug, reboot) proves the
        # port works again, so the next drop gets fresh cycle attempts.
        manager = LocalOutpostManager(8888)
        manager._expected_names = ["mic"]
        manager._gave_up_cycling = {"mic"}
        manager._cycle_failures = {"mic": USB_POWER_CYCLE_MAX_ATTEMPTS}
        monkeypatch.setattr(manager, "_scan_usb_ports", lambda: {"mic": ("8-3", 4)})
        manager.outposts = [self._healthy_outpost()]
        manager._recovery_tick()
        assert "mic" not in manager._gave_up_cycling
        assert "mic" not in manager._cycle_failures

    def _healthy_outpost(self):
        o = MagicMock()
        o.device_name = "mic"
        o.muted = False
        o.is_alive.return_value = True
        o.seconds_since_last_frame.return_value = 0.5
        return o


@pytest.mark.unit
class TestAutoRecoveryTick:
    def _manager(self, monkeypatch, usb_map):
        manager = LocalOutpostManager(8888)
        manager._expected_names = ["mic"]
        monkeypatch.setattr(manager, "_scan_usb_ports", lambda: usb_map)
        monkeypatch.setattr(manager, "_learn_ports", lambda usb: None)
        return manager

    def _outpost(self, alive=True, silent_for=0.0, muted=False):
        o = MagicMock()
        o.device_name = "mic"
        o.muted = muted
        o.is_alive.return_value = alive
        o.seconds_since_last_frame.return_value = silent_for
        return o

    def test_healthy_mic_takes_no_action(self, monkeypatch):
        manager = self._manager(monkeypatch, {"mic": ("8-3", 4)})
        manager.outposts = [self._outpost(alive=True, silent_for=1.0)]
        cycle = MagicMock(); reattach = MagicMock()
        monkeypatch.setattr(manager, "_maybe_power_cycle", cycle)
        monkeypatch.setattr(manager, "_reattach", reattach)
        manager._recovery_tick()
        cycle.assert_not_called(); reattach.assert_not_called()

    def test_dropped_mic_power_cycles_not_reattaches(self, monkeypatch):
        manager = self._manager(monkeypatch, {})  # not on the USB bus
        manager.outposts = [self._outpost(alive=True, silent_for=999)]
        cycle = MagicMock(return_value=True); reattach = MagicMock()
        monkeypatch.setattr(manager, "_maybe_power_cycle", cycle)
        monkeypatch.setattr(manager, "_reattach", reattach)
        manager._recovery_tick()
        cycle.assert_called_once_with("mic")
        reattach.assert_not_called()  # device absent → nothing to attach yet

    def test_present_but_missing_outpost_reattaches(self, monkeypatch):
        manager = self._manager(monkeypatch, {"mic": ("8-3", 4)})  # back on the bus
        manager.outposts = []  # no live outpost
        cycle = MagicMock(return_value=False); reattach = MagicMock()
        monkeypatch.setattr(manager, "_maybe_power_cycle", cycle)
        monkeypatch.setattr(manager, "_reattach", reattach)
        manager._recovery_tick()
        reattach.assert_called_once()

    def test_recovered_mic_clears_offline_flag(self, monkeypatch):
        manager = self._manager(monkeypatch, {"mic": ("8-3", 4)})
        manager._offline_names = {"mic"}
        manager.outposts = [self._outpost(alive=True, silent_for=0.5)]  # healthy again
        manager._recovery_tick()
        assert manager._offline_names == set()

    def test_offbus_mic_with_fresh_frames_flagged_after_grace(self, monkeypatch):
        # The macOS reroute case: device gone from the bus but its stream still delivers
        # frames (cloned from another input), so staleness never fires. The bus scan must
        # catch it — but only after the grace period, to ride out re-enumeration blips.
        manager = self._manager(monkeypatch, {})  # scanned fine, mic absent
        manager._port_by_mic = {"mic": ("8-3", 4)}
        manager.outposts = [self._outpost(alive=True, silent_for=0.5)]
        cycle = MagicMock(return_value=True); reattach = MagicMock()
        monkeypatch.setattr(manager, "_maybe_power_cycle", cycle)
        monkeypatch.setattr(manager, "_reattach", reattach)
        manager._recovery_tick()                # first absence starts the grace timer
        cycle.assert_not_called(); reattach.assert_not_called()
        manager._offbus_since["mic"] -= RECOVERY_OFFBUS_GRACE + 1
        manager._recovery_tick()                # grace elapsed → offline, power-cycle
        cycle.assert_called_once_with("mic")
        reattach.assert_not_called()            # still absent → nothing to attach yet
        assert "mic" in manager._pending_reattach

    def test_offbus_mic_reattaches_once_back_on_bus(self, monkeypatch):
        # After the drop, frames look healthy (rerouted) — the pending flag must still
        # force a re-attach so the stream points at the right device again.
        manager = self._manager(monkeypatch, {"mic": ("8-3", 4)})
        manager._port_by_mic = {"mic": ("8-3", 4)}
        manager._pending_reattach = {"mic"}
        manager._offline_names = {"mic"}
        manager.outposts = [self._outpost(alive=True, silent_for=0.5)]
        cycle = MagicMock(); reattach = MagicMock()
        monkeypatch.setattr(manager, "_maybe_power_cycle", cycle)
        monkeypatch.setattr(manager, "_reattach", reattach)
        manager._recovery_tick()
        reattach.assert_called_once()
        cycle.assert_not_called()

    def test_return_to_bus_within_grace_resets_timer(self, monkeypatch):
        manager = self._manager(monkeypatch, {"mic": ("8-3", 4)})
        manager._port_by_mic = {"mic": ("8-3", 4)}
        manager._offbus_since = {"mic": 0.0}    # was missing, now present again
        manager.outposts = [self._outpost(alive=True, silent_for=0.5)]
        manager._recovery_tick()
        assert "mic" not in manager._offbus_since

    def test_unlearned_mic_absence_is_not_offline(self, monkeypatch):
        # A mic on a non-switchable hub never appears in the scan — its absence must not
        # be read as a drop.
        manager = self._manager(monkeypatch, {})
        manager.outposts = [self._outpost(alive=True, silent_for=0.5)]
        cycle = MagicMock(); reattach = MagicMock()
        monkeypatch.setattr(manager, "_maybe_power_cycle", cycle)
        monkeypatch.setattr(manager, "_reattach", reattach)
        manager._recovery_tick()
        cycle.assert_not_called(); reattach.assert_not_called()
        assert "mic" not in manager._offbus_since

    def test_unknown_bus_state_reattaches_stale_mic(self, monkeypatch):
        # No uhubctl → scan is None. A stale mic must still reach the re-attach path
        # rather than being treated as off-bus and waiting forever for a power-cycle.
        manager = self._manager(monkeypatch, None)
        manager.outposts = [self._outpost(alive=True, silent_for=999)]
        cycle = MagicMock(return_value=False); reattach = MagicMock()
        monkeypatch.setattr(manager, "_maybe_power_cycle", cycle)
        monkeypatch.setattr(manager, "_reattach", reattach)
        manager._recovery_tick()
        reattach.assert_called_once()

    def test_reattach_clears_pending_flag(self, monkeypatch):
        manager = LocalOutpostManager(8888)
        manager._expected_names = ["mic"]
        manager._pending_reattach = {"mic"}
        fake_sd = SimpleNamespace(_terminate=lambda: None, _initialize=lambda: None)
        monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
        monkeypatch.setattr(manager, "_discover_mics", lambda: [(3, "mic")])
        monkeypatch.setattr(manager, "_start_outpost", lambda idx, name: True)
        manager._reattach()
        assert manager._pending_reattach == set()


@pytest.mark.unit
class TestAudioDiagnostics:
    """Forensics for the -9986 wedge: the capture must name the real CoreAudio error,
    discriminate process-local vs system-wide via the subprocess probe, and rate-limit
    its expensive parts so the 1s reopen-retry loop can't stack log-show calls."""

    def test_fourcc_renders_printable_osstatus(self):
        from interfaces.voice.microphones.local_outposts.AudioDiagnostics import fourcc
        assert fourcc(0x21646174) == "'!dat'"     # kAudioHardwareUnsupportedOperationError
        assert fourcc(0x77686F3F) == "'who?'"     # kAudioHardwareBadObjectError

    def test_fourcc_falls_back_to_int_for_unprintable(self):
        from interfaces.voice.microphones.local_outposts.AudioDiagnostics import fourcc
        assert fourcc(5) == "5"
        assert fourcc(-50) == "-50"               # paramErr: bytes not printable

    def test_base_device_name_strips_disambiguation_suffix(self):
        from interfaces.voice.microphones.local_outposts.AudioDiagnostics import base_device_name
        assert base_device_name("Studio Display Microphone #2") == "Studio Display Microphone"
        assert base_device_name("USB audio CODEC") == "USB audio CODEC"
        assert base_device_name("ReSpeaker 4 Mic Array (UAC1.0)") == "ReSpeaker 4 Mic Array (UAC1.0)"

    def test_probe_parses_subprocess_verdict(self, monkeypatch):
        import interfaces.voice.microphones.local_outposts.AudioDiagnostics as diag
        fake = SimpleNamespace(stdout="PROBE:ok\n", stderr="", returncode=0)
        monkeypatch.setattr(diag.subprocess, "run", lambda *a, **k: fake)
        assert diag.probe_open_in_subprocess("mic") == "ok"
        fake.stdout = "PROBE:fail: Internal PortAudio error\n"
        assert diag.probe_open_in_subprocess("mic").startswith("fail")

    def test_probe_survives_subprocess_error(self, monkeypatch):
        import interfaces.voice.microphones.local_outposts.AudioDiagnostics as diag
        def boom(*a, **k):
            raise OSError("spawn failed")
        monkeypatch.setattr(diag.subprocess, "run", boom)
        assert diag.probe_open_in_subprocess("mic").startswith("probe-error")

    def test_full_capture_rate_limited(self, monkeypatch):
        import interfaces.voice.microphones.local_outposts.AudioDiagnostics as diag
        monkeypatch.setattr(diag, "_last_full_capture", 0.0)
        monkeypatch.setattr(diag, "last_host_error", lambda: "OSStatus 0")
        monkeypatch.setattr(diag, "snapshot_devices", lambda: "none")
        probes = []
        monkeypatch.setattr(diag, "probe_open_in_subprocess", lambda n: probes.append(n) or "ok")
        monkeypatch.setattr(diag, "coreaudiod_log_excerpt", lambda: ["x"])
        diag.capture_open_failure("mic", "mic", RuntimeError("-9986"))
        diag.capture_open_failure("mic", "mic", RuntimeError("-9986"))  # within the window
        assert probes == ["mic"]                  # expensive probe ran exactly once

    def test_full_capture_runs_again_after_interval(self, monkeypatch):
        import interfaces.voice.microphones.local_outposts.AudioDiagnostics as diag
        monkeypatch.setattr(diag, "_last_full_capture", 0.0)
        monkeypatch.setattr(diag, "last_host_error", lambda: "OSStatus 0")
        monkeypatch.setattr(diag, "snapshot_devices", lambda: "none")
        probes = []
        monkeypatch.setattr(diag, "probe_open_in_subprocess", lambda n: probes.append(n) or "fail: x")
        monkeypatch.setattr(diag, "coreaudiod_log_excerpt", lambda: ["x"])
        diag.capture_open_failure("mic", "mic", RuntimeError("-9986"))
        monkeypatch.setattr(diag, "_last_full_capture",
                            time.time() - diag.FULL_CAPTURE_INTERVAL - 1)
        diag.capture_open_failure("mic", "mic", RuntimeError("-9986"))
        assert len(probes) == 2


@pytest.mark.unit
class TestSileroVADNoGradLeak:
    """Silero VAD is a stateful LSTM whose hidden state is carried across calls. Run with
    autograd enabled, every forward chains onto the previous one's graph and leaks ~25 KB,
    freed only by reset_states() — which Floxy calls only after a detected utterance. On a
    silent night (no utterances → no reset) that grew ~5 GB/h across both mics until GC/swap
    stalls starved the capture callback, dropping recorded audio. is_speech must run the
    forward under inference_mode so no graph is built however long the VAD runs unreset."""

    class _Torch:
        """The two things SileroVAD asks of torch, with inference_mode observable — importing
        the real torch costs half a second, and what matters here is where the forward runs."""

        def __init__(self):
            self.inferring = False

        @contextmanager
        def inference_mode(self):
            self.inferring = True
            try:
                yield
            finally:
                self.inferring = False

        @staticmethod
        def from_numpy(chunk):
            return chunk

    def _bare_vad(self, torch, model):
        # Build a SileroVAD without the heavyweight model download; inject a probe model.
        from interfaces.voice.microphones.local_outposts.VAD import SileroVAD
        vad = object.__new__(SileroVAD)
        vad._torch, vad.model, vad.threshold = torch, model, 0.5
        vad.last_score, vad._buf = 0.0, np.empty(0, np.float32)
        return vad

    def test_forward_runs_with_autograd_disabled(self):
        from interfaces.voice.microphones.local_outposts.VAD import SileroVAD
        torch = self._Torch()
        seen = []
        vad = self._bare_vad(
            torch, lambda chunk, sr: seen.append(torch.inferring) or SimpleNamespace(item=lambda: 0.0))
        vad.is_speech(np.zeros(SileroVAD.WINDOW, np.float32))
        assert seen == [True], \
            "Silero forward ran outside inference_mode — the unbounded-graph VAD leak is back"


class TestWhisperDownloadNotes:
    """The first start on a machine downloads the model inside warm_up with no visible
    progress, so the manager announces the download's start and end (or failure) as
    system rows the Home tab shows; a cached model says nothing."""

    def _manager(self, monkeypatch, cached, warm_up_error=None, size=3_083_151_518):
        mod = sys.modules[LocalOutpostManager.__module__]
        sent = []
        self.pushed = {}

        def note(source, text, push=True):
            sent.append((source, text))
            self.pushed[text] = push

        monkeypatch.setattr(mod, "system_note", note)
        monkeypatch.setattr(mod, "model_cached", lambda repo: cached)
        monkeypatch.setattr(mod, "model_download_size", lambda repo: size)
        worker = MagicMock()
        if warm_up_error is not None:
            worker.warm_up.side_effect = warm_up_error
        monkeypatch.setattr(mod, "WhisperWorker", lambda *a, **k: worker)
        monkeypatch.setattr(mod, "OutpostStreamClient", MagicMock())
        monkeypatch.setattr(mod, "SCRIBE_ENABLED", False)
        manager = LocalOutpostManager(8888, model_repo="mlx-community/whisper-large-v2-mlx")
        monkeypatch.setattr(manager, "_discover_mics", lambda: [(0, "Test Mic")])
        monkeypatch.setattr(manager, "_start_outpost",
                            lambda idx, name: manager.outposts.append(MagicMock(device_name=name)))
        monkeypatch.setattr(manager, "_start_recovery_loop", lambda: None)
        monkeypatch.setattr(manager, "_start_monitor_loop", lambda: None)
        return manager, sent

    def test_cached_model_announces_nothing_about_a_download(self, monkeypatch):
        manager, sent = self._manager(monkeypatch, cached=True)
        assert manager.start() is True
        assert [t for _, t in sent if "ownload" in t] == []

    def test_download_start_and_end_are_announced(self, monkeypatch):
        manager, sent = self._manager(monkeypatch, cached=False)
        assert manager.start() is True
        download_notes = [t for _, t in sent if "ownload" in t]
        assert download_notes[0] == (
            "Downloading the Whisper speech-to-text model 'mlx-community/whisper-large-v2-mlx' "
            "from Hugging Face (3.1 GB, a one-time download). The local microphones are not "
            "listening until it is done.")
        assert download_notes[1].startswith("Whisper speech-to-text model downloaded and loaded (")
        assert download_notes[1].endswith(" min).")
        assert sent.index(("voice", download_notes[1])) < len(sent) - 1   # before the running note

    def test_only_the_running_note_stays_out_of_the_chats(self, monkeypatch):
        """Every start says voice is ready, so that row is for the history alone; a download
        is news and still goes to the chats."""
        manager, sent = self._manager(monkeypatch, cached=False)
        assert manager.start() is True
        running_note = sent[-1][1]
        assert "Test Mic" in running_note
        assert [text for text, push in self.pushed.items() if not push] == [running_note]

    def test_unknown_size_says_several_gb(self, monkeypatch):
        manager, sent = self._manager(monkeypatch, cached=False, size=None)
        manager.start()
        assert "(several GB, a one-time download)" in sent[0][1]

    def test_failed_download_is_announced(self, monkeypatch):
        manager, sent = self._manager(monkeypatch, cached=False, warm_up_error=OSError("no route to host"))
        assert manager.start() is False
        assert sent[-1] == ("voice", "The Whisper speech-to-text model download failed (no route to host). "
                                     "Voice will not work now; it is retried at the next start.")

    def test_failed_warm_up_of_a_cached_model_is_not_a_download_failure(self, monkeypatch):
        manager, sent = self._manager(monkeypatch, cached=True, warm_up_error=RuntimeError("no MLX"))
        assert manager.start() is False
        assert sent == []


class TestModelCached:
    def test_local_directory_counts_as_cached(self, tmp_path):
        from interfaces.voice.microphones.local_outposts.WhisperWorker import model_cached
        assert model_cached(str(tmp_path)) is True

    def test_missing_hub_snapshot_is_not_cached(self, monkeypatch):
        import huggingface_hub
        from huggingface_hub.errors import LocalEntryNotFoundError
        from interfaces.voice.microphones.local_outposts.WhisperWorker import model_cached

        def missing(repo_id, local_files_only):
            assert local_files_only is True   # never a download
            raise LocalEntryNotFoundError("not on disk")
        monkeypatch.setattr(huggingface_hub, "snapshot_download", missing)
        assert model_cached("mlx-community/whisper-nope-mlx") is False

    def test_present_hub_snapshot_is_cached(self, monkeypatch, tmp_path):
        import huggingface_hub
        from interfaces.voice.microphones.local_outposts.WhisperWorker import model_cached
        monkeypatch.setattr(huggingface_hub, "snapshot_download",
                            lambda repo_id, local_files_only: str(tmp_path))
        assert model_cached("mlx-community/whisper-large-v2-mlx") is True
