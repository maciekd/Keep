"""Reply-timing gate tests, moved from tests/unit/test_web_request_handler.py when the
gate left RequestProcessor (Phase 3). Assertions are unchanged; only the fixture changed:
`handler` is now a small stand-in carrying the request context, a mocked dialog manager
and the gate, and `wait()` maps the old wait_so_that_response_timing_is_natural(chunk_time)
call onto VoiceTurnGate.wait(ctx, reply_text, time_to_first_token, chunk_time)."""
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from assistant.AssistantRequest import RequestContext
from interfaces.voice.VoiceTurnGate import VoiceTurnGate


@pytest.fixture
def handler():
    h = SimpleNamespace()
    h.is_killed = False
    h.kill_reason = ""
    h.request_context = RequestContext()
    h.dialog_manager = Mock()
    h.dialog_manager.reply_text = "User: hello how are you"
    h.dialog_manager.time_to_first_token = 0.1
    h.gate = VoiceTurnGate(Mock(), lambda: h.is_killed, lambda: h.kill_reason)
    return h


def wait(h, chunk_time: float) -> bool:
    return h.gate.wait(h.request_context, h.dialog_manager.reply_text,
                       h.dialog_manager.time_to_first_token, chunk_time)


@pytest.fixture
def mock_tui_console():
    """A Mock observer subscribed to the hub (the conftest fixture clears it afterwards)."""
    from assistant.AssistantEvents import observers
    observer = Mock()
    observers().subscribe(observer)
    return observer


class GateClock:
    """The gate module's `time`: a clock that moves only when the gate sleeps, each sleep
    recorded in `sleeps` (a Mock a test may give a side effect). Patched on the module alone:
    a patch of `time.sleep` itself is every thread's, and a stray thread's sleep would count."""

    def __init__(self):
        self.now = 1_000_000.0
        self.sleeps = Mock()

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds
        self.sleeps(seconds)

    def slept(self) -> list[float]:
        return [c.args[0] for c in self.sleeps.call_args_list]


@pytest.fixture(autouse=True)
def clock():
    fake = GateClock()
    with patch("interfaces.voice.VoiceTurnGate.time", fake):
        yield fake


@pytest.fixture
def mock_sleep(clock):
    return clock.sleeps


class TestBeginTurn:
    """The outpost's wake_word_detected/scribe_skipped query params only reach the verdict
    gate through this parse — the client tests assert the URL carries them and the gate
    tests assert the attributes work, but a param rename on either side would silently
    re-hold every skipped send for the full verdict cap. (Replaces
    TestProcess.test_parses_wake_word_and_scribe_skipped_flags / _default_false.)"""

    def test_parses_wake_word_and_scribe_skipped_flags(self, handler):
        handler.gate.speech_analyzer.utterance_generation = 7
        handler.gate.begin_turn({"query": "hi", "wake_word_detected": "1", "scribe_skipped": "1"})
        assert handler.gate._is_wake_word_send is True
        assert handler.gate._scribe_skipped is True
        assert handler.gate._utterance_generation == 7

    def test_wake_word_and_scribe_skipped_default_false(self, handler):
        handler.gate.begin_turn({"query": "hi"})
        assert handler.gate._is_wake_word_send is False
        assert handler.gate._scribe_skipped is False

# ---------------------------------------------------------------------------
# TestWaitSoThatResponseTimingIsNatural
# ---------------------------------------------------------------------------

class TestWaitSoThatResponseTimingIsNatural:
    def test_common_query_uses_minimum_wait(self, handler, mock_tui_console, mock_sleep):
        with patch("interfaces.voice.VoiceTurnGate.FloxyDatabase") as mock_db_cls, \
             patch("interfaces.voice.VoiceTurnGate.SentenceCompletenessDetector") as mock_scd:
            mock_db_cls.get_instance.return_value.is_query_common.return_value = True
            handler.dialog_manager.time_to_first_token = 0.3
            result = wait(handler, 0.2)
            assert result is True
            if mock_sleep.called:
                sleep_val = mock_sleep.call_args[0][0]
                assert sleep_val <= 0.6

    def test_single_word_uses_full_wait(self, handler, mock_tui_console, mock_sleep):
        handler.dialog_manager.reply_to_llm = "Hello"
        handler.dialog_manager.reply_text = "Hello"
        with patch("interfaces.voice.VoiceTurnGate.FloxyDatabase") as mock_db_cls, \
             patch("interfaces.voice.VoiceTurnGate.SentenceCompletenessDetector"):
            handler.dialog_manager.time_to_first_token = 0.1
            result = wait(handler, 0.1)
            assert result is True
            assert mock_sleep.called
            sleep_val = mock_sleep.call_args[0][0]
            assert sleep_val > 1.0

    def test_killed_mid_sleep_returns_false(self, handler, mock_tui_console, mock_sleep):
        handler.dialog_manager.reply_to_llm = "Hello"
        handler.dialog_manager.reply_text = "Hello"

        def kill_on_sleep(_):
            handler.is_killed = True

        mock_sleep.side_effect = kill_on_sleep
        with patch("interfaces.voice.VoiceTurnGate.FloxyDatabase"), \
             patch("interfaces.voice.VoiceTurnGate.SentenceCompletenessDetector"):
            handler.dialog_manager.time_to_first_token = 0.1
            result = wait(handler, 0.1)
            assert result is False

    def test_no_sleep_when_total_exceeds_target(self, handler, mock_tui_console, mock_sleep):
        with patch("interfaces.voice.VoiceTurnGate.FloxyDatabase") as mock_db_cls, \
             patch("interfaces.voice.VoiceTurnGate.SentenceCompletenessDetector") as mock_scd:
            mock_db_cls.get_instance.return_value.is_query_common.return_value = True
            handler.dialog_manager.time_to_first_token = 0.5
            result = wait(handler, 0.5)
            assert result is True
            mock_sleep.assert_not_called()

    def test_is_complete_gets_raw_utterance_not_formatted(self, handler, mock_tui_console, mock_sleep):
        # reply_text carries the LLM-bound header; is_complete must see the bare utterance.
        handler.request_context = RequestContext(user_input="turn off the kitchen lights")
        handler.dialog_manager.reply_text = "Marcus (VoiceToVoice in Kitchen): turn off the kitchen lights"
        with patch("interfaces.voice.VoiceTurnGate.FloxyDatabase") as mock_db_cls, \
             patch("interfaces.voice.VoiceTurnGate.SentenceCompletenessDetector") as mock_scd:
            mock_db_cls.get_instance.return_value.is_query_common.return_value = False
            mock_scd.is_complete.return_value = True
            wait(handler, 0.2)
            mock_scd.is_complete.assert_called_once_with("turn off the kitchen lights")
            # is_query_common still keys on the formatted text the DB stores.
            mock_db_cls.get_instance.return_value.is_query_common.assert_called_once_with(
                "Marcus (VoiceToVoice in Kitchen): turn off the kitchen lights"
            )

    def test_outpost_engine_skips_completeness_model(self, handler, mock_tui_console, mock_sleep):
        # Local-outpost STT (whisper/scribe) sends a final, already-complete segment, so the
        # completeness model must be skipped entirely — it can only add a needless hold.
        handler.request_context = RequestContext(user_input="turn off the kitchen lights", engine="whisper")
        handler.dialog_manager.reply_text = "Marcus (VoiceToVoice in Kitchen): turn off the kitchen lights"
        with patch("interfaces.voice.VoiceTurnGate.FloxyDatabase") as mock_db_cls, \
             patch("interfaces.voice.VoiceTurnGate.SentenceCompletenessDetector") as mock_scd:
            handler.dialog_manager.time_to_first_token = 0.3
            result = wait(handler, 0.2)
            assert result is True
            mock_scd.is_complete.assert_not_called()
            mock_db_cls.get_instance.return_value.is_query_common.assert_not_called()

    def test_is_complete_falls_back_to_reply_text_when_no_utterance(self, handler, mock_tui_console, mock_sleep):
        # No raw utterance available (e.g. command follow-up turn) -> use reply_text.
        handler.request_context = RequestContext(user_input=None)
        handler.dialog_manager.reply_text = "result: the lights are now off"
        with patch("interfaces.voice.VoiceTurnGate.FloxyDatabase") as mock_db_cls, \
             patch("interfaces.voice.VoiceTurnGate.SentenceCompletenessDetector") as mock_scd:
            mock_db_cls.get_instance.return_value.is_query_common.return_value = False
            mock_scd.is_complete.return_value = True
            wait(handler, 0.2)
            mock_scd.is_complete.assert_called_once_with("result: the lights are now off")

    def test_speech_end_anchor_skips_sleep_when_pause_already_elapsed(self, handler, mock_tui_console, mock_sleep,
                                                                      clock):
        # The pause guards against talking over the user, so it must be measured from when
        # they stopped speaking. An outpost's STT endpointing (0.4s silence) + decode means
        # ~1s has usually already passed by reply time — no additional sleep is owed, even
        # when the reply itself was instant (cached LLM + cached voice).
        handler.request_context = RequestContext(
            user_input="turn off the kitchen lights", engine="whisper",
            speech_end_epoch=clock.now - 1.0)
        handler.dialog_manager.time_to_first_token = 0.0
        with patch("interfaces.voice.VoiceTurnGate.FloxyDatabase"), \
             patch("interfaces.voice.VoiceTurnGate.SentenceCompletenessDetector"):
            result = wait(handler, 0.0)
            assert result is True
            mock_sleep.assert_not_called()

    def test_speech_end_anchor_sleeps_only_the_remainder(self, handler, mock_tui_console, mock_sleep, clock):
        # Speech ended 0.2s ago against a 0.6s target -> only ~0.4s remains to sleep,
        # regardless of how fast the LLM/TTS were.
        handler.request_context = RequestContext(
            user_input="turn off the kitchen lights", engine="whisper",
            speech_end_epoch=clock.now - 0.2)
        handler.dialog_manager.time_to_first_token = 0.0
        with patch("interfaces.voice.VoiceTurnGate.FloxyDatabase"), \
             patch("interfaces.voice.VoiceTurnGate.SentenceCompletenessDetector"):
            result = wait(handler, 0.0)
            assert result is True
            assert mock_sleep.called
            sleep_val = mock_sleep.call_args[0][0]
            assert sleep_val == pytest.approx(0.4)

    def test_no_speech_end_anchor_falls_back_to_server_latency(self, handler, mock_tui_console, mock_sleep):
        # Phones don't send stt_ms -> no anchor -> the old proxy (LLM first token + first
        # audio chunk) still governs the wait.
        handler.request_context = RequestContext(user_input="turn off the kitchen lights", engine="whisper")
        handler.dialog_manager.time_to_first_token = 0.1
        with patch("interfaces.voice.VoiceTurnGate.FloxyDatabase"), \
             patch("interfaces.voice.VoiceTurnGate.SentenceCompletenessDetector"):
            result = wait(handler, 0.1)
            assert result is True
            assert mock_sleep.called
            sleep_val = mock_sleep.call_args[0][0]
            assert sleep_val == pytest.approx(0.4, abs=0.01)

    def test_live_more_speech_signal_overrides_stale_anchor(self, handler, mock_tui_console, mock_sleep, clock):
        # Branch order matters: when the user has started speaking again, the
        # still-speaking hold must govern even though the (now stale) end-of-speech
        # anchor says the pause was paid long ago. A refactor hoisting the anchor above
        # the more_speech branch would zero out the hold exactly when speaking early is
        # most damaging.
        handler.request_context = RequestContext(
            user_input="turn off the kitchen lights", engine="whisper",
            speech_end_epoch=clock.now - 5.0)
        handler.gate.more_speech_is_incoming = clock.now
        mock_sleep.side_effect = lambda s: setattr(handler.gate, "more_speech_is_incoming", None)
        with patch("interfaces.voice.VoiceTurnGate.FloxyDatabase"), \
             patch("interfaces.voice.VoiceTurnGate.SentenceCompletenessDetector"):
            result = wait(handler, 0.0)
        assert result is True
        assert clock.slept() == [1.0]


class TestWaitForScribeVerdict:
    """The Scribe gate: a Whisper wake-word reply holds until Scribe has voted or vetoed
    (or the cap elapses), so Scribe's more-accurate verdict lands before playback."""

    @pytest.fixture
    def scribe_running(self):
        # The gate keys off runtime Scribe availability (not the SCRIBE_ENABLED constant),
        # so tests exercising the hold must simulate a live ScribeWorker.
        with patch("interfaces.voice.VoiceTurnGate.scribe_is_running", return_value=True):
            yield

    def _wake_word_whisper(self, handler, gen=5):
        handler.request_context = RequestContext(user_input="turn on the lights", engine="whisper")
        handler.gate._is_wake_word_send = True
        handler.gate._utterance_generation = gen

    def test_non_whisper_engine_skips_gate(self, handler, scribe_running):
        handler.request_context = RequestContext(user_input="x", engine="scribe")
        handler.gate._is_wake_word_send = True
        handler.gate.speech_analyzer.scribe_has_settled.return_value = False
        assert handler.gate._wait_for_scribe_verdict(handler.request_context) is True
        handler.gate.speech_analyzer.scribe_has_settled.assert_not_called()

    def test_follow_up_without_wake_word_skips_gate(self, handler, scribe_running):
        handler.request_context = RequestContext(user_input="and the fan too", engine="whisper")
        handler.gate._is_wake_word_send = False
        handler.gate.speech_analyzer.scribe_has_settled.return_value = False
        assert handler.gate._wait_for_scribe_verdict(handler.request_context) is True
        handler.gate.speech_analyzer.scribe_has_settled.assert_not_called()

    def test_scribe_skipped_send_skips_gate(self, handler, scribe_running, mock_sleep):
        # The outpost skipped Scribe on high Whisper confidence (scribe_skipped=1): no
        # verdict is ever coming, so the reply must speak without waiting out the cap.
        self._wake_word_whisper(handler)
        handler.gate._scribe_skipped = True
        handler.gate.speech_analyzer.scribe_has_settled.return_value = False
        assert handler.gate._wait_for_scribe_verdict(handler.request_context) is True
        handler.gate.speech_analyzer.scribe_has_settled.assert_not_called()
        mock_sleep.assert_not_called()

    def test_whisper_only_fallback_skips_gate(self, handler, mock_sleep):
        # ScribeWorker never came up (scribe_is_running() is False by default): no verdict
        # can ever arrive, so a wake-word reply must NOT wait out the cap.
        self._wake_word_whisper(handler)
        handler.gate.speech_analyzer.scribe_has_settled.return_value = False
        assert handler.gate._wait_for_scribe_verdict(handler.request_context) is True
        handler.gate.speech_analyzer.scribe_has_settled.assert_not_called()
        mock_sleep.assert_not_called()

    def test_already_settled_returns_immediately(self, handler, scribe_running, mock_sleep):
        self._wake_word_whisper(handler)
        handler.gate.speech_analyzer.scribe_has_settled.return_value = True
        assert handler.gate._wait_for_scribe_verdict(handler.request_context) is True
        mock_sleep.assert_not_called()

    def _kill_on_second_poll(self, handler, mock_sleep, reason):
        polls = [0]

        def sleep_side_effect(_):
            polls[0] += 1
            if polls[0] >= 2:
                handler.is_killed = True
                handler.kill_reason = reason

        mock_sleep.side_effect = sleep_side_effect

    def test_veto_kills_request_while_waiting(self, handler, scribe_running, mock_tui_console, mock_sleep):
        self._wake_word_whisper(handler)
        handler.gate.speech_analyzer.scribe_has_settled.return_value = False
        # The veto arrives as a kill mid-wait: is_killed flips True on the second poll.
        self._kill_on_second_poll(handler, mock_sleep, "Scribe veto from kitchen")
        with patch("interfaces.voice.VoiceTurnGate.info") as log:
            assert handler.gate._wait_for_scribe_verdict(handler.request_context) is False
        assert "Scribe veto from kitchen" in log.call_args_list[-1].args[0]

    def test_superseding_transcription_kill_is_not_logged_as_a_veto(self, handler, scribe_running,
                                                                    mock_tui_console, mock_sleep):
        """A production case: Scribe CONFIRMED the wake word, its phrase won the vote and the new
        request killed this one — the old log line called that a veto and misled the triage."""
        self._wake_word_whisper(handler)
        handler.gate.speech_analyzer.scribe_has_settled.return_value = False
        self._kill_on_second_poll(handler, mock_sleep,
                                  "superseded by new request 'Did you bring any of it?'")
        with patch("interfaces.voice.VoiceTurnGate.info") as log:
            assert handler.gate._wait_for_scribe_verdict(handler.request_context) is False
        message = log.call_args_list[-1].args[0]
        assert "superseded by new request" in message
        assert "veto" not in message

    def test_cap_elapses_speaks_on_whisper_alone(self, handler, scribe_running, mock_tui_console, clock):
        self._wake_word_whisper(handler)
        handler.gate.speech_analyzer.scribe_has_settled.return_value = False
        # Scribe never settles → the hold runs the cap out and speaks anyway.
        assert handler.gate._wait_for_scribe_verdict(handler.request_context) is True
        assert sum(clock.slept()) == pytest.approx(VoiceTurnGate.MAX_WAIT_FOR_SCRIBE_VERDICT, abs=0.02)


class TestMoreSpeechWaitSignal:
    """The tentative VAD onset signal (speech_onset_at) must hold the reply only briefly —
    a phantom onset (multi-mic echo, room noise) that never grows the transcript cannot
    escalate to the full hold the way a confirmed length-growth signal does."""

    def test_confirmed_signal_drives_full_hold(self, handler, clock):
        handler.gate.more_speech_is_incoming = clock.now
        active, _, target = handler.gate._more_speech_wait()
        assert active is True
        assert target == VoiceTurnGate.SECONDS_TO_WAIT_BEFORE_SPEAKING_IF_SENTENCE_INCOMPLETE

    def test_onset_within_grace_holds_tentatively(self, handler, clock):
        handler.gate.speech_onset_at = clock.now - VoiceTurnGate.SPEECH_ONSET_GRACE / 2
        active, elapsed, target = handler.gate._more_speech_wait()
        assert active is True
        assert target == VoiceTurnGate.SPEECH_ONSET_GRACE
        assert elapsed == VoiceTurnGate.SPEECH_ONSET_GRACE / 2

    def test_onset_past_grace_is_inactive(self, handler, clock):
        handler.gate.speech_onset_at = clock.now - VoiceTurnGate.SPEECH_ONSET_GRACE - 0.1
        active, _, _ = handler.gate._more_speech_wait()
        assert active is False

    def test_confirmed_wins_over_onset(self, handler, clock):
        handler.gate.speech_onset_at = clock.now
        handler.gate.more_speech_is_incoming = clock.now
        active, _, target = handler.gate._more_speech_wait()
        assert active is True
        assert target == VoiceTurnGate.SECONDS_TO_WAIT_BEFORE_SPEAKING_IF_SENTENCE_INCOMPLETE

    def test_single_phantom_onset_proceeds_without_looping(self, handler, mock_tui_console, clock):
        # One onset ping that never grows into a longer transcript holds at most the base
        # wait, then speaks — it must not enter the escalating hold loop.
        handler.dialog_manager.reply_to_llm = "Hello"
        handler.dialog_manager.reply_text = "Hello"
        handler.gate.speech_onset_at = clock.now

        with patch("interfaces.voice.VoiceTurnGate.FloxyDatabase") as mock_db_cls, \
             patch("interfaces.voice.VoiceTurnGate.SentenceCompletenessDetector"):
            mock_db_cls.get_instance.return_value.is_query_common.return_value = False
            handler.dialog_manager.time_to_first_token = 0.1
            result = wait(handler, 0.1)
            assert result is True
            # Slept once past the grace, then the onset aged out → no further holding.
            assert sum(clock.slept()) < VoiceTurnGate.MAX_TOTAL_WAIT_FOR_MORE_SPEECH

    def test_sustained_phantom_onset_capped_not_infinite(self, handler, mock_tui_console, clock, mock_sleep):
        # A noisy room re-arming the onset every sleep keeps the tentative signal live, but
        # it can only ramp to the grace ceiling, so the absolute MAX_TOTAL_WAIT cap ends it
        # rather than it holding the reply forever.
        handler.dialog_manager.reply_to_llm = "Hello"
        handler.dialog_manager.reply_text = "Hello"
        handler.gate.speech_onset_at = clock.now
        mock_sleep.side_effect = lambda s: setattr(handler.gate, "speech_onset_at", clock.now)  # noise keeps firing onsets

        with patch("interfaces.voice.VoiceTurnGate.FloxyDatabase") as mock_db_cls, \
             patch("interfaces.voice.VoiceTurnGate.SentenceCompletenessDetector"):
            mock_db_cls.get_instance.return_value.is_query_common.return_value = False
            handler.dialog_manager.time_to_first_token = 0.1
            result = wait(handler, 0.1)
            assert result is True
            # Total hold lands at the cap (within one grace step), never unbounded.
            assert sum(clock.slept()) <= VoiceTurnGate.MAX_TOTAL_WAIT_FOR_MORE_SPEECH + VoiceTurnGate.SPEECH_ONSET_GRACE
