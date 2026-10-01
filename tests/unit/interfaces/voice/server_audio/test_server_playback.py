import time
from types import SimpleNamespace
from unittest.mock import Mock, patch, MagicMock

import pytest

from common.RequestKilledException import RequestKilledException
from interfaces.voice.server_audio.ServerPlayback import ServerPlayback

MOD = "interfaces.voice.server_audio.ServerPlayback"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _new_playback() -> ServerPlayback:
    """A ServerPlayback with an injected outpost server / speech analyzer and a kill flag
    the test flips via `pb.state.killed = True` (the handler owns that flag in prod)."""
    state = SimpleNamespace(killed=False)
    pb = ServerPlayback(outpost_server=Mock(loop=None), speech_analyzer=Mock(),
                        is_killed=lambda: state.killed)
    pb.state = state
    return pb


@pytest.fixture
def playback():
    return _new_playback()


@pytest.fixture
def mock_tui_console():
    """A Mock observer subscribed to the hub (the conftest fixture clears it afterwards)."""
    from assistant.AssistantEvents import observers
    observer = Mock()
    observers().subscribe(observer)
    return observer


@pytest.fixture(autouse=True)
def _reset_mute_class_state():
    """The mute generation counters live on the class, so a test that claims a mute would
    otherwise leak nonzero state into the next test in the same worker. Reset before and
    after each test so every test starts from 'no reply has ever muted'."""
    ServerPlayback._mute_generation = 0
    ServerPlayback._active_mute_generation = 0
    yield
    ServerPlayback._mute_generation = 0
    ServerPlayback._active_mute_generation = 0


def _live_outposts(pb: ServerPlayback):
    pb.outpost_server.loop = MagicMock()
    pb.outpost_server.loop.is_closed.return_value = False


# ---------------------------------------------------------------------------
# TestStop
# ---------------------------------------------------------------------------

class TestStop:
    def test_stop_skips_close_when_worker_wedged(self, playback):
        """kill() path: closing a stream the wedged worker still owns would block on the
        same CoreAudio hang — it must be skipped, leaving cleanup to the daemon worker."""
        wedged = Mock()
        wedged.is_alive.return_value = True  # still alive even after the join timeout
        playback.audio_thread = wedged

        with patch.object(playback, "_close_audio_stream") as mock_close, \
             patch(f"{MOD}.error"):
            playback.stop()

        wedged.join.assert_called_once_with(timeout=1.0)
        mock_close.assert_not_called()
        assert playback.audio_thread is None


# ---------------------------------------------------------------------------
# TestCloseAudioStream
# ---------------------------------------------------------------------------

class TestCloseAudioStream:
    def test_active_stream_stop_and_close(self, playback):
        mock_stream = Mock()
        mock_stream.active = True
        playback.audio_stream = mock_stream

        playback._close_audio_stream()

        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()
        assert playback.audio_stream is None

    def test_idempotent(self, playback):
        mock_stream = Mock()
        mock_stream.active = True
        playback.audio_stream = mock_stream

        playback._close_audio_stream()
        playback._close_audio_stream()  # second call should not raise

        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()


# ---------------------------------------------------------------------------
# TestAudioPlaybackWorker
# ---------------------------------------------------------------------------

class TestAudioPlaybackWorker:
    """audio_stream.start() moved BEFORE the first-chunk wait, keeping CoreAudio's
    ~44ms start cost off the time-to-first-sound path. The stream is therefore
    live even when no audio ever arrives — the sentinel path must still stop and
    close it, or every request killed before TTS leaks a running stream."""

    @pytest.fixture
    def mock_sd(self):
        with patch(f"{MOD}.sd") as m:
            m.query_devices.return_value = {"name": "Test Device",
                                            "default_low_output_latency": 0.01}
            m.OutputStream.return_value = Mock(active=True)
            yield m

    def _run(self, playback):
        playback.sample_rate = 16000
        playback.channels = 1
        playback.audio_playback_worker()

    def test_stream_started_before_waiting_for_first_chunk(
            self, playback, mock_sd, mock_tui_console):
        stream = mock_sd.OutputStream.return_value
        real_queue = playback.audio_queue
        real_queue.put(None)
        started_at_wait = []

        def recording_get(timeout=None):
            started_at_wait.append(stream.start.called)
            return real_queue.get(timeout=timeout)

        playback.audio_queue = Mock(get=recording_get)

        self._run(playback)

        assert started_at_wait[0] is True, "stream must start before the first-chunk wait"

    def test_sentinel_before_any_chunk_closes_the_started_stream(
            self, playback, mock_sd, mock_tui_console):
        stream = mock_sd.OutputStream.return_value
        playback.audio_queue.put(None)     # request killed before any TTS chunk

        self._run(playback)

        stream.start.assert_called_once()
        stream.write.assert_not_called()
        stream.stop.assert_called_once()
        stream.close.assert_called_once()
        assert playback.audio_stream is None

    def test_plays_chunks_then_closes_on_sentinel(
            self, playback, mock_sd, mock_tui_console):
        stream = mock_sd.OutputStream.return_value
        playback.audio_queue.put(b"\x00\x01" * 320)
        playback.audio_queue.put(b"\x00\x01" * 320)
        playback.audio_queue.put(None)

        self._run(playback)

        assert stream.write.call_count == 2
        assert playback._chunks_played == 2
        stream.stop.assert_called_once()
        stream.close.assert_called_once()
        assert playback.audio_stream is None

    def test_first_chunk_reports_speaker_output_started(
            self, playback, mock_sd, mock_tui_console):
        """The on_speaker_output_started measurement point stays at the first write."""
        playback.audio_queue.put(b"\x00\x01" * 320)
        playback.audio_queue.put(None)

        self._run(playback)

        mock_tui_console.on_speaker_output_started.assert_called_once_with(0.05)


# ---------------------------------------------------------------------------
# TestEnsureStarted
# ---------------------------------------------------------------------------

class TestEnsureStarted:
    def test_creates_daemon_thread(self, playback):
        with patch.object(playback, 'audio_playback_worker'):
            playback.ensure_started()
            assert playback.audio_thread is not None
            assert playback.audio_thread.daemon is True
            # Clean up
            playback.audio_queue.put(None)
            playback.audio_thread.join(timeout=1)

    def test_thread_alive_no_new_thread(self, playback):
        mock_thread = Mock()
        mock_thread.is_alive.return_value = True
        playback.audio_thread = mock_thread
        playback.ensure_started()
        assert playback.audio_thread is mock_thread  # unchanged


# ---------------------------------------------------------------------------
# TestEnqueue
# ---------------------------------------------------------------------------

class TestEnqueue:
    def test_queues_chunk(self, playback):
        with patch(f"{MOD}.PlaySound"), \
             patch(f"{MOD}.DynamicServerVolumeController") as mock_dvc:
            mock_dvc.get_instance.return_value = Mock()
            playback.enqueue(b"audio_data")
            assert playback.audio_queue.get_nowait() == b"audio_data"

    def test_updates_play_timestamp(self, playback):
        with patch(f"{MOD}.PlaySound") as mock_ps, \
             patch(f"{MOD}.DynamicServerVolumeController") as mock_dvc:
            mock_dvc.get_instance.return_value = Mock()
            before = time.time()
            playback.enqueue(b"data")
            assert mock_ps.last_time_sound_played_on_server >= before

    def test_pauses_volume_controller(self, playback):
        with patch(f"{MOD}.PlaySound"), \
             patch(f"{MOD}.DynamicServerVolumeController") as mock_dvc:
            mock_vc = Mock()
            mock_dvc.get_instance.return_value = mock_vc
            playback.enqueue(b"data")
            mock_vc.pause.assert_called_once()

# ---------------------------------------------------------------------------
# TestFinishReply
# ---------------------------------------------------------------------------

class TestFinishReply:
    def _running(self, playback):
        mock_thread = Mock()
        mock_thread.is_alive.return_value = True
        playback.audio_thread = mock_thread
        return mock_thread

    def test_no_audio_thread_returns_immediately(self, playback):
        playback.audio_thread = None
        playback.finish_reply("Hello")
        playback.speech_analyzer.clean.assert_not_called()

    def test_waits_for_queue_and_joins(self, playback):
        mock_thread = self._running(playback)

        with patch(f"{MOD}.time.sleep"):
            playback.finish_reply("Hello")

        mock_thread.join.assert_called_once_with(timeout=2.0)

    def test_calls_speech_analyzer_clean(self, playback):
        self._running(playback)

        with patch(f"{MOD}.time.sleep"):
            playback.finish_reply("Hello")

        playback.speech_analyzer.clean.assert_called_once()

    def test_question_broadcasts_listen(self, playback):
        self._running(playback)
        # Production reality (post-ea772691): finish_reply only broadcasts the unmute if
        # THIS reply muted and still owns the current mute. A spoken question implies audio
        # played, which implies the reply muted at the first sound chunk — mirror that here.
        playback._outposts_muted = True
        playback._claim_outpost_mute()
        _live_outposts(playback)

        mock_future = MagicMock()
        with patch(f"{MOD}.time.sleep"), \
             patch(f"{MOD}.asyncio.run_coroutine_threadsafe", return_value=mock_future):
            playback.finish_reply("How are you?")

        playback.outpost_server.broadcast_command.assert_called_with("listen")

    def test_skips_unmute_when_superseded_by_newer_request(self, playback):
        """Barge-in regression (ea772691): if a newer request claimed the mute while we were
        playing audio, the end-of-reply unmute must NOT broadcast 'listen'/'wakeword' — that
        would reopen the mics in the middle of the newer reply (mic captures Floxy's own voice).
        Generation mismatch wins; we just clear our own _outposts_muted flag and bail."""
        self._running(playback)
        playback._outposts_muted = True
        _live_outposts(playback)
        playback._claim_outpost_mute()  # reply A claims...
        newer = _new_playback()
        newer._claim_outpost_mute()  # ...newer barge-in supersedes

        with patch(f"{MOD}.time.sleep"), \
             patch(f"{MOD}.note"):
            playback.finish_reply("Sure thing.")

        playback.outpost_server.broadcast_command.assert_not_called()
        assert playback._outposts_muted is False

    def test_no_audio_reply_defers_reset_while_other_reply_speaking(self, playback):
        """Barge-in safety: a no-audio reply still wants to reset the outpost state, but if
        ANOTHER reply currently holds the mute and is speaking, broadcasting 'wakeword' would
        reopen the mics mid-reply (mic captures Floxy's own voice). When a live mute is held
        by someone else, the no-audio path must defer the reset and stay quiet."""
        self._running(playback)
        _live_outposts(playback)
        # Another request is currently muted (claimed and not yet released) — it's speaking now.
        other = _new_playback()
        other._claim_outpost_mute()

        with patch(f"{MOD}.time.sleep"), \
             patch(f"{MOD}.note") as mock_note:
            playback.finish_reply("")

        playback.outpost_server.broadcast_command.assert_not_called()
        mock_note.assert_not_called()

    def test_no_audio_reply_resets_outpost_state_when_mute_idle(self, playback):
        """A false-accept regression: a wake word opened a follow-up
        listen window, then the reply produced NO audio. The old code returned without any
        broadcast, leaving the window to the outpost's local timers — which room chatter kept
        re-arming, so a stray phrase was forwarded with no wake word. Now, when no other reply
        holds the mute, a no-audio reply must broadcast 'wakeword' to close the window itself."""
        self._running(playback)
        _live_outposts(playback)

        mock_future = MagicMock()
        with patch(f"{MOD}.time.sleep"), \
             patch(f"{MOD}.note") as mock_note, \
             patch(f"{MOD}.asyncio.run_coroutine_threadsafe", return_value=mock_future):
            playback.finish_reply("")  # no '?', so reset is 'wakeword'

        playback.outpost_server.broadcast_command.assert_called_once_with("wakeword")
        mock_note.assert_not_called()

    def test_no_audio_question_reply_keeps_window_open_with_listen(self, playback):
        """A no-audio reply that is itself a question should reopen the follow-up window
        ('listen'), not close it — mirroring the audio question path."""
        self._running(playback)
        _live_outposts(playback)

        mock_future = MagicMock()
        with patch(f"{MOD}.time.sleep"), \
             patch(f"{MOD}.asyncio.run_coroutine_threadsafe", return_value=mock_future):
            playback.finish_reply("Which room?")

        playback.outpost_server.broadcast_command.assert_called_once_with("listen")


# ---------------------------------------------------------------------------
# TestPlaybackStallFailsafe
# ---------------------------------------------------------------------------

class TestPlaybackStallFailsafe:
    """Wedge regression: sd.OutputStream() creation hung inside CoreAudio, the
    worker never consumed a chunk, and the previously unbounded drain loop wedged the whole
    request — reply never played, HTTP response never completed, outposts stayed muted
    until their own 90s failsafe. The drain wait must be bounded."""

    def test_stalled_worker_bails_without_unmute(self, playback):
        """Zero consumption progress with chunks queued → bail past the stall timeout:
        no broadcast from the normal path (mute left set for the finally failsafe), no
        join on the wedged thread, and a None queued so the worker can exit if it ever
        unwedges."""
        playback._outposts_muted = True
        wedged = Mock()
        wedged.is_alive.return_value = True
        playback.audio_thread = wedged
        playback.audio_queue.put(b"chunk")
        _live_outposts(playback)

        # A timeout already behind the first check: the drain loop polls every 0.1 s, and a
        # positive one would cost the test a whole poll before the stall is seen.
        with patch.object(ServerPlayback, "PLAYBACK_STALL_TIMEOUT", -1.0), \
             patch.object(ServerPlayback, "_capture_playback_stall_forensics"):
            playback.finish_reply("Hello")

        playback.outpost_server.broadcast_command.assert_not_called()
        assert playback._outposts_muted is True
        wedged.join.assert_not_called()
        assert playback.audio_queue.get_nowait() == b"chunk"
        assert playback.audio_queue.get_nowait() is None

    def test_dead_worker_bails(self, playback):
        """A worker that died with audio still queued must not be waited on forever."""
        dead = Mock()
        dead.is_alive.side_effect = [True, False]  # alive at the outer guard, dead in the drain
        playback.audio_thread = dead
        playback.audio_queue.put(b"chunk")

        playback.finish_reply("Hello")

        dead.join.assert_not_called()

    def test_progress_resets_stall_clock(self, playback):
        """A worker that IS consuming never trips the stall bail; the drain finishes
        normally and the end-of-reply path joins the thread."""
        alive = Mock()
        alive.is_alive.return_value = True
        playback.audio_thread = alive
        for _ in range(3):
            playback.audio_queue.put(b"c")

        def consume(_):
            playback.audio_queue.get_nowait()
            playback._chunks_played += 1

        with patch(f"{MOD}.time.sleep", side_effect=consume):
            playback.finish_reply("Hello")

        alive.join.assert_called_once_with(timeout=2.0)

    def test_killed_during_drain_raises(self, playback):
        alive = Mock()
        alive.is_alive.return_value = True
        playback.audio_thread = alive
        playback.audio_queue.put(b"c")

        def kill(_):
            playback.state.killed = True

        with patch(f"{MOD}.time.sleep", side_effect=kill):
            with pytest.raises(RequestKilledException):
                playback.finish_reply("Hello")


# ---------------------------------------------------------------------------
# TestRestoreOutpostsIfMuted
# ---------------------------------------------------------------------------

class TestRestoreOutpostsIfMuted:
    """The 'stop' broadcast that mutes outposts must always be paired with an unmute,
    even when the reply is killed/errored before the normal end-of-reply unmute —
    otherwise the muted mics can never produce the query that would unmute them."""

    def test_broadcasts_wakeword_when_left_muted(self, playback):
        playback._claim_outpost_mute()  # this reply owns the current mute
        playback._outposts_muted = True
        _live_outposts(playback)
        mock_future = MagicMock()
        with patch(f"{MOD}.asyncio.run_coroutine_threadsafe", return_value=mock_future):
            playback.restore_outposts_if_muted()

        playback.outpost_server.broadcast_command.assert_called_with("wakeword")
        assert playback._outposts_muted is False

    def test_noop_when_not_muted(self, playback):
        playback._outposts_muted = False
        playback.restore_outposts_if_muted()
        playback.outpost_server.broadcast_command.assert_not_called()

    def test_skips_unmute_when_superseded_by_newer_request(self, playback):
        """Barge-in regression: a killed/superseded reply's finally must NOT unmute,
        or it opens the mic while the newer request is still speaking (mic captures
        Floxy's own voice). It still clears its own flag and leaves the mute to the owner."""
        playback._claim_outpost_mute()  # reply A mutes and owns it...
        playback._outposts_muted = True
        newer = _new_playback()
        newer._claim_outpost_mute()  # ...then a newer (barge-in) request takes over the mute

        with patch(f"{MOD}.note"):
            playback.restore_outposts_if_muted()

        playback.outpost_server.broadcast_command.assert_not_called()
        assert playback._outposts_muted is False

    def test_flag_cleared_even_if_broadcast_raises(self, playback):
        playback._claim_outpost_mute()  # owns the mute, so the failsafe unmute is attempted
        playback._outposts_muted = True
        with patch.object(ServerPlayback, "_broadcast_outpost_command", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError):
                playback.restore_outposts_if_muted()
        assert playback._outposts_muted is False


# ---------------------------------------------------------------------------
# TestEnsureOutpostsMuted
# ---------------------------------------------------------------------------

class TestEnsureOutpostsMuted:
    """The 'stop' mute must fire before the FIRST server sound of a reply — the thinking
    filler OR the first response chunk — so the mic never transcribes Floxy's own voice.
    It must be idempotent so the two callers don't double-broadcast within one reply."""

    def test_claims_and_broadcasts_stop_when_unmuted(self, playback):
        playback._outposts_muted = False
        with patch.object(ServerPlayback, "_broadcast_outpost_command") as mock_bc:
            playback.ensure_outposts_muted()
        mock_bc.assert_called_once_with("stop")
        assert playback._outposts_muted is True
        assert playback._mute_generation == ServerPlayback._mute_generation

    def test_idempotent_second_call_is_noop(self, playback):
        playback._outposts_muted = False
        with patch.object(ServerPlayback, "_broadcast_outpost_command") as mock_bc:
            playback.ensure_outposts_muted()  # thinking filler mutes first…
            playback.ensure_outposts_muted()  # …first audio chunk must not re-broadcast
        mock_bc.assert_called_once_with("stop")

    def test_broadcast_uses_injected_outpost_server(self, playback):
        _live_outposts(playback)
        mock_future = MagicMock()
        with patch(f"{MOD}.asyncio.run_coroutine_threadsafe", return_value=mock_future) as mock_rcts:
            playback.ensure_outposts_muted()
        mock_rcts.assert_called_once()
        playback.outpost_server.broadcast_command.assert_called_once_with("stop")

    def test_broadcast_without_loop_is_logged_not_raised(self, playback):
        playback.outpost_server.loop = None
        with patch(f"{MOD}.error") as mock_error:
            playback.ensure_outposts_muted()
        mock_error.assert_called_once()
        assert playback._outposts_muted is True


# ---------------------------------------------------------------------------
# play_thinking_filler
# ---------------------------------------------------------------------------

class TestPlayThinkingFiller:
    """The thinking filler plays during the LLM thinking phase, before the first response
    chunk. The outposts must be muted first, or they hear the filler and the open follow-up
    listen window sends it back as a query (the 'Hold on.' loop seen in prod)."""

    def test_mutes_outposts_before_playing(self, playback):
        order = []
        with patch.object(playback, "ensure_outposts_muted", side_effect=lambda: order.append("mute")), \
             patch(f"{MOD}.PlaySound") as mock_ps:
            mock_ps.get_instance.return_value.playsound_thinking.side_effect = lambda *a: order.append("play")
            playback.play_thinking_filler("en-US")
        assert order == ["mute", "play"]

    def test_mute_failure_does_not_block_the_filler(self, playback):
        with patch.object(playback, "ensure_outposts_muted", side_effect=RuntimeError("no loop")), \
             patch(f"{MOD}.PlaySound") as mock_ps:
            playback.play_thinking_filler("en-US")
        mock_ps.get_instance.return_value.playsound_thinking.assert_called_once_with("en")

    def test_uses_request_language_subtag(self, playback):
        """The filler picked must match the reply language, not a fixed default —
        otherwise a Swedish reply gets an English "Hmm…". The BCP-47 tag is reduced to
        its subtag because PlaySound groups samples by folder name (en/, sv/, …)."""
        with patch.object(playback, "ensure_outposts_muted"), \
             patch(f"{MOD}.PlaySound") as mock_ps:
            playback.play_thinking_filler("sv-SE")
        mock_ps.get_instance.return_value.playsound_thinking.assert_called_once_with("sv")

    def test_falls_back_to_configured_language_when_lang_is_none(self, playback):
        with patch.object(playback, "ensure_outposts_muted"), \
             patch(f"{MOD}.PlaySound") as mock_ps, \
             patch("common.Language.configured_language", return_value="en-US"):
            playback.play_thinking_filler(None)
        mock_ps.get_instance.return_value.playsound_thinking.assert_called_once_with("en")
