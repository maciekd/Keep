import asyncio
import queue
import threading
import time
from typing import Callable

import numpy as np
import sounddevice as sd

from assistant.AssistantEvents import observers
from common.Language import language_code
from common.RequestKilledException import RequestKilledException
from common.logging import info, trace, note, error, fine
from .DynamicServerVolumeController import DynamicServerVolumeController
from .PlaySound import PlaySound


class ServerPlayback:
    """Plays one reply's TTS chunks through the server speakers and mutes the local mics around it.

    Mutes the outposts ('stop') before the first sound, drains at the end, then sends
    'listen' / 'wakeword' so the mics reopen. One instance per RequestProcessor (handlers are
    single-use), created by the voice interface's playback factory, so a process without a voice
    interface never constructs one and never imports sounddevice. Latency: enqueue() only does a
    queue.put; the worker opens and starts the output stream before the first chunk exists so
    CoreAudio's start cost is off the time-to-first-sound path.
    """

    # How long the end-of-reply audio drain tolerates the playback worker consuming
    # NOTHING while chunks sit queued before declaring it wedged. On 2026-06-11
    # sd.OutputStream() creation hung forever inside CoreAudio (the output-side cousin
    # of the -9986 input wedge), and the unbounded drain loop wedged the whole request:
    # reply never played, HTTP response never completed, outposts stayed muted. Normal
    # consumption is one ~40ms chunk per write, so 10s of zero progress is decisive.
    PLAYBACK_STALL_TIMEOUT = 10.0

    # Class-wide owner of the local-outpost mute. Each time any reply mutes the
    # outposts it bumps _mute_generation and records the new value as its own claim.
    # The failsafe unmute (restore_outposts_if_muted) only fires if this reply is
    # still the latest claimant — so a killed/superseded request (barge-in) can't
    # unmute mics that a newer reply has since re-muted and is still speaking through.
    _mute_lock = threading.Lock()
    _mute_generation = 0
    # Generation of the reply currently holding the mute (i.e. speaking right now), or 0
    # when no reply is muted. Unlike _mute_generation (monotonic — only ever increases),
    # this is cleared back to 0 on unmute. It lets a reply that played NO audio (and so
    # never muted) tell "another reply is speaking through a mute right now" (must not
    # reopen the mics) from "the mute is idle" (safe to reset the outpost's listen state).
    _active_mute_generation = 0

    def __init__(self, outpost_server, speech_analyzer, is_killed: Callable[[], bool]):
        self.outpost_server = outpost_server
        self.speech_analyzer = speech_analyzer
        # Owned by the request handler: True once the request was killed (barge-in, Scribe veto).
        self._is_killed = is_killed

        # True once we've broadcast "stop" to mute outposts for this reply and not yet
        # broadcast the matching unmute. Lets the handler's finally block recover if the reply
        # is killed/errored before the normal end-of-reply unmute (see restore_outposts_if_muted).
        self._outposts_muted = False
        # This reply's claim on the class-wide mute (0 = never muted). Compared against
        # ServerPlayback._mute_generation to detect that a newer request has taken over.
        self._mute_generation = 0

        self.audio_queue = queue.Queue()
        self.audio_thread = None
        self.audio_stream = None
        self._audio_stream_lock = threading.Lock()
        # Chunks the playback worker has actually written to the device. The end-of-reply
        # drain wait watches this for progress instead of queue size (which the TTS
        # producer can still be growing) so a worker wedged inside CoreAudio is detected.
        self._chunks_played = 0

        self.sample_rate = 24000
        self.channels = 1

    @property
    def is_killed(self) -> bool:
        return self._is_killed()

    def _check_killed(self):
        if self.is_killed:
            raise RequestKilledException()

    def is_running(self) -> bool:
        return self.audio_thread is not None and self.audio_thread.is_alive()

    # ------------------------------------------------------------------ playback

    def ensure_started(self):
        """Start the worker (which opens + starts the output stream) if it isn't running.
        Called at first token so the stream is live before the first chunk arrives."""
        if self.audio_thread is None or not self.audio_thread.is_alive():
            info("Initiating audio playback in background.")
            self.audio_thread = threading.Thread(target=self.audio_playback_worker, daemon=True)
            self.audio_thread.start()

    def enqueue(self, chunk: bytes):
        # Runs on: the thread consuming TTS chunks — keep this a plain queue.put.
        fine("Playing " + str(len(chunk)) + " bytes of audio.")
        self.mark_sound_played()
        DynamicServerVolumeController.get_instance().pause()
        self.audio_queue.put(chunk)

    @staticmethod
    def log_ambient_noise():
        vc = DynamicServerVolumeController.get_instance()
        fine(f"Background noise: {round(vc.current_noise_db)} Db Volume: {round(vc.current_volume)}%")

    @staticmethod
    def mark_sound_played():
        """Stamp 'the server speakers were just used' — PlaySound uses it to space out
        notification sounds. Called per chunk and once more when the reply completes."""
        PlaySound.last_time_sound_played_on_server = time.time()

    def audio_playback_worker(self):
        # Runs on: audio playback thread (daemon, started by ensure_started)
        """Worker thread for audio playback using sounddevice blocking write"""
        try:
            t0 = time.time()
            device_info = sd.query_devices(sd.default.device[1], 'output')
            trace(f"Audio device: {device_info['name']}, low latency: {device_info['default_low_output_latency'] * 1000:.0f}ms")

            # Create stream with low latency
            self.audio_stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.int16,
                latency=0.05
            )
            # Start now, before the first chunk exists: a started blocking-write
            # stream just plays silence, and this keeps CoreAudio's start cost
            # (~44ms) off the time-to-first-sound critical path.
            self.audio_stream.start()
            trace(f"Audio stream setup+start: {(time.time() - t0) * 1000:.0f}ms (in background, before first chunk)")

            # Wait for first chunk
            first_chunk = self.audio_queue.get(timeout=30.0)
            if first_chunk is None:
                trace("Audio playback worker: no first chunk (sentinel), closing stream")
                return

            observers().on_speaker_output_started(0.05)

            # Write first chunk
            audio_array = np.frombuffer(first_chunk, dtype=np.int16).reshape(-1, self.channels)
            self.audio_stream.write(audio_array)
            self._chunks_played += 1

            # Continue writing chunks
            while not self.is_killed:
                try:
                    chunk = self.audio_queue.get(timeout=0.1)
                    if chunk is None:
                        break
                    audio_array = np.frombuffer(chunk, dtype=np.int16).reshape(-1, self.channels)
                    self.audio_stream.write(audio_array)
                    self._chunks_played += 1
                except queue.Empty:
                    continue

        except Exception as e:
            error(f"Failed to initialize audio stream: {e!r}")
        finally:
            self._close_audio_stream()

    def _close_audio_stream(self):
        """Close the current audio stream exactly once."""
        stream = None
        with self._audio_stream_lock:
            if self.audio_stream is None:
                return
            stream = self.audio_stream
            self.audio_stream = None

        if stream is None:
            return

        try:
            if stream.active:
                stream.stop()
        except Exception as e:
            trace(f"Ignoring audio stop error during cleanup: {e}")
        try:
            stream.close()
        except Exception as e:
            trace(f"Ignoring audio close error during cleanup: {e}")

    def stop(self):
        """Clean up audio resources (kill path and handler teardown)."""
        thread = self.audio_thread
        if thread and thread.is_alive():
            # Signal audio thread to stop and wait for it to clean up the stream.
            self.audio_queue.put(None)
            thread.join(timeout=1.0)
            if thread.is_alive():
                # Worker is wedged inside a CoreAudio call (stream open/write hang).
                # Closing its stream from here would block on the same wedge — leave
                # cleanup to the daemon worker's finally if it ever unwedges.
                error("Audio playback worker did not exit within 1s (wedged in CoreAudio?) "
                      "— skipping stream close so this thread doesn't hang too. "
                      "Capturing forensics in background.")
                # Recover at the first sign: left to a later reply's drain wait, the wedge
                # was only diagnosed once that reply had stalled for PLAYBACK_STALL_TIMEOUT
                # (35 minutes of silent replies on the incident that showed it).
                self._start_stall_forensics()
                self.audio_thread = None
                return

        # Ensure residual stream resources are released once.
        self._close_audio_stream()
        self.audio_thread = None

    # ------------------------------------------------------------------ end of reply

    def finish_reply(self, reply_text: str):
        """Drain the queued audio, then tell the outposts to listen again: 'listen' if the
        reply asked the user a question, else 'wakeword'. No-op unless the worker is running."""
        if not (self.audio_thread and self.audio_thread.is_alive()):
            return
        if not self._wait_for_playback_to_drain():
            # Worker is wedged (CoreAudio hung the stream open or a write) — abandon
            # the audio instead of waiting forever. Don't touch its stream from this
            # thread (closing blocks on the same wedge); the queued None lets the
            # daemon worker exit if it ever unwedges. _outposts_muted stays set so the
            # handler's finally-block failsafe (restore_outposts_if_muted) reopens the mics.
            self.audio_queue.put(None)
            return
        self.audio_queue.put(None)
        self.audio_thread.join(timeout=2.0)

        trace("We're done playing sound - notify all clients to start listening for wake word again.")
        self.speech_analyzer.clean()
        if "?" in reply_text:
            command = "listen"
            info("AI seems to have a question to the user: commanding outposts to start listening for STT again.")
        else:
            command = "wakeword"
        # Same atomicity requirement as the restore_outposts_if_muted failsafe: gate the
        # unmute on still owning the mute, and keep the check + broadcast under one lock so
        # a barge-in that claimed the mute (and broadcast 'stop') can't be undone by our
        # 'listen'/'wakeword' reopening the mics while its newer reply is still speaking.
        with ServerPlayback._mute_lock:
            if not self._outposts_muted:
                # Command-only / no-audio reply: no filler or TTS chunk ever played, so
                # we never claimed the mute — there is nothing to *unmute*. But the wake
                # word that opened this turn left the outpost in a follow-up listen window;
                # without a state-reset broadcast the server never closes it and stray
                # room speech keeps re-arming it, so an unrelated phrase lands inside the
                # window and is forwarded with no wake word (2026-06-15 'good night, love
                # you' false-accept). Send the reset ('wakeword', or 'listen' if we asked
                # a question) so the window closes deterministically — UNLESS another reply
                # currently holds the mute and is speaking, in which case reopening the
                # mics would feed its own audio back as a query (leave the reset to it).
                if ServerPlayback._active_mute_generation != 0:
                    trace("Reply produced no server audio and another reply currently holds "
                          "the outpost mute — leaving the outpost state reset to it.")
                    return
                trace(f"Reply produced no server audio (outposts were never muted) — "
                      f"broadcasting '{command}' so the outpost closes any dangling follow-up "
                      f"listen window instead of waiting on its local timers.")
                self._broadcast_outpost_command(command)
                return
            self._outposts_muted = False
            if self._mute_generation != ServerPlayback._mute_generation:
                note("Reply finished, but a newer request has taken over the outpost mute — "
                     "leaving the unmute to it so we don't open mics while its reply is speaking.")
                return
            ServerPlayback._active_mute_generation = 0
            self._broadcast_outpost_command(command)

    def _wait_for_playback_to_drain(self) -> bool:
        """Wait until the playback worker has consumed the queued audio. Bounded: returns
        False when the worker dies or makes zero progress for PLAYBACK_STALL_TIMEOUT —
        e.g. the 2026-06-11 wedge where sd.OutputStream() creation hung inside CoreAudio
        and the previously unbounded loop here spun forever. Progress is measured by
        _chunks_played, not queue size, since the TTS producer may still be queueing."""
        last_played = self._chunks_played
        last_progress = time.time()
        while not self.audio_queue.empty():
            self._check_killed()
            played = self._chunks_played
            if played != last_played:
                last_played, last_progress = played, time.time()
            elif not self.audio_thread.is_alive():
                error("Audio playback worker died with audio still queued — abandoning playback wait.")
                return False
            elif time.time() - last_progress > ServerPlayback.PLAYBACK_STALL_TIMEOUT:
                error(f"Audio playback stalled: worker consumed nothing for "
                      f"{ServerPlayback.PLAYBACK_STALL_TIMEOUT:.0f}s with "
                      f"{self.audio_queue.qsize()} chunks queued — abandoning playback "
                      f"(CoreAudio output wedge?). Capturing forensics in background.")
                self._start_stall_forensics()
                return False
            time.sleep(0.1)
        return True

    def _start_stall_forensics(self):
        """Diagnose a wedged worker — and restart the process when the wedge is ours — off
        the calling thread, which must not block on the same CoreAudio hang."""
        threading.Thread(target=self._capture_playback_stall_forensics,
                         name="PlaybackStallForensics", daemon=True).start()

    @staticmethod
    def _capture_playback_stall_forensics():
        # Lazy import: interfaces.voice.microphones.local_outposts.__init__ pulls in the whole outpost manager.
        try:
            from interfaces.voice.microphones.local_outposts import AudioDiagnostics
            verdict = AudioDiagnostics.capture_playback_stall("server playback")
        except Exception as e:
            error(f"Playback-stall forensics failed: {e}")
            return

        # An in-process wedge poisons this process's audio session for good — every
        # subsequent reply spawns a fresh worker that wedges too (8 leaked threads on the
        # 2026-06-14 incident). The forensics prove a fresh process plays fine, so restart
        # to recover. AudioRecovery owns the loop guard so this can't boot-loop.
        if verdict == "in_process_wedge":
            try:
                from interfaces.voice.microphones.local_outposts import AudioRecovery
                AudioRecovery.trigger("server playback wedged in CoreAudio")
            except Exception as e:
                error(f"Audio recovery trigger failed: {e}")

    # ------------------------------------------------------------------ outpost mute

    def _broadcast_outpost_command(self, command: str):
        loop = self.outpost_server.loop
        if loop is None or loop.is_closed():
            error("Outpost server event loop not available")
            return
        future = asyncio.run_coroutine_threadsafe(self.outpost_server.broadcast_command(command), loop)
        try:
            future.result(timeout=10)
        except Exception as e:
            error(f"Failed to broadcast '{command}' to outposts: {e}")

    def play_thinking_filler(self, lang: str | None):
        """Play a short spoken filler ("Hmm...", "Let's see...") over the LLM's thinking
        phase. It is the first sound of the reply, so the local mic outposts are muted
        first — or they transcribe the filler ("Hold on") and the open follow-up listen
        window sends it back as a query. `lang` is the request language; None falls back
        to the configured default. No samples for that language -> silence."""
        try:
            self.ensure_outposts_muted()
        except Exception as e:
            fine(f"Failed to mute outposts before thinking filler: {e}")
        PlaySound.get_instance().playsound_thinking(language_code(lang))

    def ensure_outposts_muted(self):
        """Mute the local mic outposts before the FIRST server sound of this reply —
        the spoken 'thinking' filler (DialogManager.notify_thinking_started) or the first
        response audio chunk, whichever comes first — so the mic never transcribes Floxy's
        own voice and feeds it back as a follow-up query. (The filler plays during the LLM
        thinking phase, well before the first response chunk; muting only at that chunk left
        the filler audible, and the still-open listen window accepted it as a follow-up.)

        Idempotent within a reply: the first caller claims the mute and broadcasts 'stop';
        later callers are no-ops. Paired with the end-of-reply unmute (finish_reply) and
        the restore_outposts_if_muted failsafe."""
        with ServerPlayback._mute_lock:
            if self._outposts_muted:
                return
            self._claim_outpost_mute_locked()
            self._outposts_muted = True
        self._broadcast_outpost_command("stop")

    def _claim_outpost_mute(self):
        """Take ownership of the local-outpost mute right before broadcasting 'stop'.
        Bumps the class-wide generation and records it as this reply's claim so a
        later barge-in can be detected by restore_outposts_if_muted. Claiming MUST
        happen before the 'stop' broadcast: otherwise an older reply's failsafe could
        run in the gap, still see itself as owner, and undo our mute."""
        with ServerPlayback._mute_lock:
            self._claim_outpost_mute_locked()

    def _claim_outpost_mute_locked(self):
        """Generation bump for _claim_outpost_mute / ensure_outposts_muted. Caller MUST
        already hold _mute_lock."""
        ServerPlayback._mute_generation += 1
        self._mute_generation = ServerPlayback._mute_generation
        ServerPlayback._active_mute_generation = self._mute_generation

    def restore_outposts_if_muted(self):
        """Failsafe pairing for the 'stop' broadcast. While speaking we mute the local
        mic outposts ('stop'); they're unmuted again only by finish_reply at the end of
        the reply. If this request is killed or errors in between (barge-in, TTS/stream
        failure, is_killed) that unmute never runs — and since a muted mic can't produce
        the next query, nothing ever unmutes it and the whole local-outpost flow wedges
        until restart. Re-broadcast 'wakeword' here so they always recover.

        But only if we're still the mute owner: on barge-in the superseding request mutes
        again with a newer generation, and unmuting from this stale finally would open the
        mic while that newer reply is still speaking (mic captures Floxy's own voice). When
        a newer request has taken over, leave the mute to it."""
        if not self._outposts_muted:
            return
        # The owner check and the 'wakeword' broadcast MUST stay atomic: if we drop the
        # lock in between, a newer reply can claim the mute and broadcast 'stop' in the
        # gap, after which our 'wakeword' reopens the mics while the newer reply is still
        # speaking. Hold _mute_lock through the broadcast so no claim can interleave.
        with ServerPlayback._mute_lock:
            self._outposts_muted = False
            if self._mute_generation != ServerPlayback._mute_generation:
                note("Reply ended mid-speech, but a newer request has since taken over the "
                     "outpost mute — not broadcasting 'wakeword' so we don't unmute mics while "
                     "the newer reply is still speaking.")
                return
            ServerPlayback._active_mute_generation = 0
            info("Reply ended without the normal outpost unmute (killed/errored mid-speech) — "
                 "broadcasting 'wakeword' so local mics don't stay muted.")
            self._broadcast_outpost_command("wakeword")
