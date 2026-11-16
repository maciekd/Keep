"""HTTP routes the microphone outposts (Android, iOS, local mics) call, plus the cross-mic
vote /stream runs before dispatching a transcription. Registered as a Quart blueprint by the
launcher when the voice interface is enabled; everything it needs arrives through
MicrophoneDeps, never by importing the launcher.
"""
import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from quart import Blueprint, request

from assistant.AssistantEvents import observers
from common.STTConfidenceMonitor import STTConfidenceMonitor
from common.audio_log import save_pcm_wav
from common.logging import info, trace, error, fine, note, external_info, external_trace, external_fine
from interfaces.voice.microphones.speech_analysis.SpeechAnalyzer import SpeechAnalyzer
from interfaces.voice.speaker_id import SpeakerIdentifier


@dataclass
class MicrophoneDeps:
    """What the routes need from the running server.

    running_handlers returns the in-flight WebRequestHandlers (each with a `turn_gate`,
    `request_context`, `is_performing_request`, `kill()`) — Assistant.running_handlers; the
    routes only ever read them. speaker_identifier / stt_confidence_monitor are callables so the lazily-created
    singletons are looked up at call time, like the launcher did."""
    running_handlers: Callable[[], list]
    speech_analyzer: SpeechAnalyzer
    wake_word: str
    configured_languages: Callable[[], list]
    # The server's own speakers; None on a server without local audio (common/Platform).
    play_sound: Any = None
    speaker_identifier: Callable[[], Optional[SpeakerIdentifier]] = SpeakerIdentifier.get_instance
    stt_confidence_monitor: Callable[[], STTConfidenceMonitor] = STTConfidenceMonitor.get_instance
    # Optional: exposes current_noise_db / current_volume for the wake-word log line.
    volume_controller: Any = None
    false_positive_timeout_ms: int = field(default=3500)


class MicrophoneRoutes:
    """Serves the microphone outposts' HTTP routes and runs the cross-mic vote before dispatching a query."""

    # Debounce window for wake-word signals from several mics hearing the same utterance.
    WAKE_WORD_TIMEOUT = 1.6

    def __init__(self, deps: MicrophoneDeps):
        self.deps = deps
        self.last_wake_word_time = 0.0
        # Utterance generation an STT time was already reported for — later (slower) mics
        # transcribing the same utterance must not overwrite the first mic's STT time.
        self._stt_recorded_utterance = -1

    # --- shared logic -------------------------------------------------------------------
    def handle_wake_word_detected(self) -> bool:
        """Run the (debounced) wake-word side effects: reset the SpeechAnalyzer for a fresh
        conversation and tell observers (the TUI resets its timings).

        Shared by the /wake_word_detected route and the /stream?wake_word_detected=1 param,
        so a client can fold the wake-word signal into the same request as its query — no
        separate fire-and-forget call that could arrive out of order.
        Returns True if it ran, False if it was debounced."""
        current_time = time.time()

        # Check if enough time has passed since last wake word detection.
        # Sliding window: a debounced signal still refreshes the timestamp, so late
        # mics finishing the same utterance keep extending the window instead of
        # triggering a mid-utterance reset once the first mic's window expires (1.6s: a slow mic's
        # Whisper landed 1.53s after the first on 2026-10-18 and ran 'volume down' twice).
        if current_time - self.last_wake_word_time < self.WAKE_WORD_TIMEOUT:
            trace(f"Wake word debounced ({current_time - self.last_wake_word_time:.2f}s since last) - sliding window.")
            self.last_wake_word_time = current_time
            return False

        trace("Wake word detected!")
        vc = self.deps.volume_controller
        if vc is not None:
            trace(f"Background noise: {round(vc.current_noise_db)} Db Volume: {round(vc.current_volume)}%")

        self.last_wake_word_time = current_time

        self.deps.speech_analyzer.clean()
        observers().on_wake_word_detected()

        # Intentionally NOT broadcasting "listen" here: only the outpost that actually
        # heard the wake word should start listening (it arms itself locally), the reason
        # is that we want to use scribe as a secondary verifier of the wake word. Having
        # all outposts wake here would make that complicated.
        trace("Wake-word side effects done — not waking other outposts (local arming only).")
        return True

    def _running(self) -> list:
        return list(self.deps.running_handlers())

    def arm_speech_onset(self, now: float) -> int:
        """Arm the TENTATIVE speech-onset signal on every in-flight request's turn gate.
        Returns how many were armed."""
        armed = 0
        for h in self._running():
            if h.is_performing_request and h.turn_gate:
                h.turn_gate.arm_speech_onset(now)
                armed += 1
        return armed

    def kill_running_request_matching(self, phrase: str, reason: str = "",
                                      include_grown: bool = False) -> int:
        """Kill any in-flight request whose query normalizes to `phrase`, flagging it a false
        positive. Phrase-targeted on purpose: a request the server has since moved on to (a
        different phrase) is left running, so a stale veto can never kill the wrong reply.
        With `include_grown` a request whose query STARTS with the phrase is killed too: the
        outpost's accumulator only ever appends, so by the time Scribe's veto lands the
        vetoed fire may already run as "<phrase> <next fragment>" (2026-10-12 prod: the veto
        for 'Sir, get dressed, …' missed 'Sir, get dressed, … make an orange water.' and
        Floxy answered). Returns how many were killed. Shared by the low-confidence guard in
        accept_transcription (exact match) and the /false_positive veto route (grown too)."""
        analyzer = self.deps.speech_analyzer
        normalized = analyzer.normalize_query(phrase)
        if not normalized:
            return 0

        def matches(user_input):
            running = analyzer.normalize_query(user_input)
            return running == normalized or (include_grown and running.startswith(normalized + " "))

        killed = 0
        for h in self._running():
            if (h.request_context and h.request_context.user_input
                    and matches(h.request_context.user_input)
                    and h.is_performing_request):
                note(f"Killing in-flight request as a false positive: '{phrase}'"
                     + (f" ({reason})." if reason else "."))
                h.dialog_manager.command_processor.commands.cmd_false_positive("")
                h.kill(reason or "false positive")
                killed += 1
        return killed

    def _record_stt_timing(self, req_data):
        """Winning transcription only — report the outpost's STT latency (VAD stop →
        transcript) to observers. Only the FIRST winner per utterance counts: with several
        mics, a slower mic's transcription can win the vote later and would otherwise
        overwrite the first (fastest) mic's STT time in the UI."""
        stt_ms = req_data.args.get('stt_ms')
        if stt_ms is None:
            return
        generation = self.deps.speech_analyzer.utterance_generation
        if self._stt_recorded_utterance == generation:
            trace(f"STT time already recorded for this utterance, ignoring stt_ms={stt_ms} from a later mic.")
            return
        try:
            stt_s = round(float(stt_ms) / 1000.0, 2)
            # stt_wait_ms (local outposts only) is the endpointing share of stt_ms —
            # time spent waiting out trailing silence before the winning decode was
            # submitted. Split it out so the Home tab shows silence-detect and
            # transcription as separate stages; clients that don't send it (phones)
            # keep the whole time on the transcription row.
            stt_wait_ms = req_data.args.get('stt_wait_ms')
            if stt_wait_ms is not None:
                wait_s = round(float(stt_wait_ms) / 1000.0, 2)
                # Which gate started the winning decode ("rms" energy / "vad" silero),
                # shown as a suffix on the Home tab's Silence detect row.
                observers().on_stt_timing(wait_s, round(stt_s - wait_s, 2), req_data.args.get('stt_trigger') or "")
            else:
                observers().on_stt_timing(-1, stt_s, "")
            self._stt_recorded_utterance = generation
        except ValueError:
            trace(f"Ignoring malformed stt_ms/stt_wait_ms param: {stt_ms!r}")

    def accept_transcription(self, req_data, context) -> Optional[str]:
        """The /stream pre-dispatch hook: cross-mic voting and everything the microphones
        need to happen before a transcription reaches the assistant. Returns None when the
        request should be dispatched, or the reason it was rejected (the caller answers
        204). May mutate `context.user` (speaker identification)."""
        analyzer = self.deps.speech_analyzer
        low_stability = req_data.args.get('low_stability', '') == "true"
        confidence = float(req_data.args.get('confidence', 0.5))
        mic = req_data.args.get('mic', '')
        engine = req_data.args.get('engine', '')
        query = req_data.args.get('query', '')
        user_agent = req_data.headers.get('User-Agent', '')
        ios_app = 'CFNetwork' in user_agent
        text_source = 'text' in req_data.args

        if ios_app or text_source:
            trace("Running text mode, reset SpeechAnalyzer since that app doesn't support that.")
            analyzer.clean()
            # iOS now sends voice samples to /identify_voice, so keep its identification.
            # Only genuine text clients (no audio) have nothing to identify.
            if text_source:
                speaker_id = self.deps.speaker_identifier()
                if speaker_id:
                    trace("Text client can't be identified by voice, clearing last identified user.")
                    speaker_id.clear_last_identified_user()

        ls = ""
        if low_stability:
            ls = " (low stability)"
        ls += f" (confidence: {confidence})"
        fine(" >>> " + query + ls)

        # A Scribe vote arriving (even a duplicate that loses the vote below) means the more
        # accurate engine has weighed in on this utterance and did NOT veto it — release any
        # Whisper reply holding at the natural-timing gate for Scribe's verdict. The veto path
        # marks it too (see /false_positive). Empty/low-confidence Scribe results never reach
        # here — the outpost drops them — so this only fires on a genuine second opinion.
        # verified_phrase is the Whisper phrase the Scribe job ran on; matching IT (not
        # Scribe's own transcript, which can legitimately differ) against the current
        # utterance window ties the settle to the right utterance — a late vote for a
        # previous utterance must not release the new one's gate.
        if engine == 'scribe' and query:
            scribe_correlator = req_data.args.get('verified_phrase', '')
            gen = analyzer.mark_scribe_settled(scribe_correlator or query)
            if gen is not None:
                trace(f"Scribe vote landed for utterance generation {gen}: '{query}'")
                # A vote from a wake-word VERIFY job means that mic's Scribe heard the wake
                # word; remember it so another mic's veto on this utterance can't kill it.
                if (req_data.args.get('wake_word_confirmed') in ('1', 'true')
                        and analyzer.mark_wake_word_confirmed(scribe_correlator or query, mic)):
                    info(f"Scribe on '{mic or 'outpost'}' confirmed the wake word for utterance "
                         f"generation {gen}.")
            elif scribe_correlator:
                # The outpost named the exact Whisper phrase this Scribe job verified, and
                # it is NOT in the current utterance window — the vote belongs to a
                # previous utterance (back-to-back wake words inside Scribe's cloud
                # round-trip; clean() has since wiped the window). It rightly failed to
                # settle the gate above, but letting it continue into the vote below could
                # fire a duplicate of the OLD utterance's command, so drop it entirely.
                # Without a correlator we fall through: Scribe's own transcript can
                # legitimately differ from the window's Whisper phrases, so a failed
                # match proves nothing about staleness.
                info(f"Dropping stale Scribe vote '{query}' — its Whisper phrase "
                     f"'{scribe_correlator}' is not in the current utterance window.")
                return "stale Scribe vote for a previous utterance"

        # A client (e.g. a local mic outpost) can fold the wake-word signal into the
        # same request as its query via wake_word_detected=1, so the reset + this
        # query can't arrive out of order. Runs before voting so this query starts
        # the fresh conversation.
        if req_data.args.get('wake_word_detected') in ('1', 'true'):
            self.handle_wake_word_detected()

        if confidence < 0.3:
            trace(f"Low confidence query detected (c{confidence}), making sure we're not using this as our running query.")
            self.kill_running_request_matching(query, reason=f"low-confidence transcription (c{confidence})")
            return "dropped low-confidence query"  # Ignore this request, it's a shitty transcription

        # Content-aware length-growth check: a genuinely longer, unique transcript for an
        # in-flight request CONFIRMS the user is still speaking (see VoiceTurnGate).
        for h in self._running():
            if h.request_context and h.request_context.user_input is not None and analyzer.is_unique_phrase(query) and len(h.request_context.user_input) + 2 < len(query) and h.is_performing_request and h.turn_gate:
                trace("...user is still speaking...")
                h.turn_gate.confirm_more_speech(time.time())

        # Record each mic's own STT confidence BEFORE the cross-mic vote. Several mics
        # can hear the same utterance, but only one wins is_best_transcription() — the
        # rest return 204 below. Gating the recording on the winner meant the losing
        # mic(s) never appeared in the DashboardTab. The dedup inside record() keeps it to
        # one bar per mic per utterance (highest confidence wins).
        if mic and query and not text_source:
            self.deps.stt_confidence_monitor().record(mic, confidence, analyzer.utterance_generation)

        if not query or (not text_source and not analyzer.is_best_transcription(query, low_stability, confidence, mic, engine)):
            return "empty/duplicate/non-best transcription"  # No content for bad transcription (and duplicates)

        self._record_stt_timing(req_data)

        if not context.user:
            speaker_id = self.deps.speaker_identifier()
            identified_user = speaker_id.get_last_identified_user(max_age_seconds=20.0) if speaker_id else None
            from assistant import VoiceCandidates
            turn_note = None
            if identified_user:
                context.user = identified_user
                info(f"User identified by voice: {identified_user}")
                if speaker_id.is_candidate(identified_user):
                    turn_note = VoiceCandidates.unnamed_speaker_note(identified_user)
                    info(f"The speaker is the unnamed voice {identified_user}: the turn note asks for the name")
            elif speaker_id:
                turn_note = VoiceCandidates.unrecognised_speaker_note()
                info("Could not identify user using voice: the turn note says the speaker is not known")
            else:
                info("Could not identify user using voice (no speaker identifier).")
            if turn_note:
                context.note = f"{context.note}\n{turn_note}" if context.note else turn_note

        fine(f"Votes: {analyzer.num_votes(query)}")
        return None

    # --- routes -------------------------------------------------------------------------
    def blueprint(self) -> Blueprint:
        bp = Blueprint('microphones', __name__)
        deps = self.deps

        @bp.route('/wake_word_detected')
        async def wake_word_detected():
            try:
                self.handle_wake_word_detected()
                return "", 204
            except Exception as e:
                error(f"Error: {str(e)}")
                return {"error": "An internal error occurred"}, 500

        @bp.route('/speech_active')
        async def speech_active():
            """A local outpost detected speech onset (VAD start) — possibly the user talking
            again. Arm the TENTATIVE speech_onset_at signal on every in-flight request so an
            incomplete one holds briefly (SPEECH_ONSET_GRACE) for the continuation's transcript
            instead of speaking a half-answer and muting the mic mid-sentence. Deliberately NOT
            the full hold: this fires at onset before any transcript exists, so it can't tell a
            real continuation from the same command re-detected by another mic (multi-mic echo,
            trailing audio). The content-aware length-growth check in accept_transcription is
            what confirms it — only a genuinely longer transcript sets more_speech_is_incoming
            and escalates to the full wait."""
            try:
                armed = self.arm_speech_onset(time.time())
                if armed:
                    trace(f"...possible continuation... (speech-active onset ping, arming grace on {armed} in-flight request(s))")
                return "", 204
            except Exception as e:
                error(f"Error: {str(e)}")
                return {"error": "An internal error occurred"}, 500

        @bp.route('/false_positive')
        async def false_positive():
            """A local outpost's Scribe verifier could not confirm the wake word Whisper fired
            on — veto it: kill the in-flight request running on that phrase before Floxy
            speaks. Matches on the normalized phrase, so a request that has already moved on
            to a different phrase is left alone (see kill_running_request_matching).
            Fire-and-forget, like /wake_word_detected."""
            try:
                query = request.args.get('query', '')
                mic = request.args.get('mic', '')
                if not query:
                    trace("false_positive called with no query — nothing to kill.")
                    return "", 204
                info(f"Scribe veto from '{mic or 'outpost'}' — verifying false positive for '{query}'.")
                # Kill FIRST, settle SECOND: the gate polls settled-then-killed, so marking settled
                # before the kill lands would let the vetoed reply wake up, see is_killed=False, and
                # speak. Kill-first closes that race for any DISPATCHED request — one whose
                # user_input is set and is_performing_request is True, which is all the phrase match
                # in kill_running_request_matching can see. A vetoed request still queued on
                # request_processing_lock is a known residual gap: the kill can't find it yet, but
                # the settle below still lands, so once dispatched it passes the gate un-killed and
                # speaks. Marking settled still frees a same-utterance reply the veto's phrase
                # doesn't match (another mic's transcript won the vote) from waiting out the cap;
                # the phrase match inside mark_scribe_settled keeps a STALE veto (previous
                # utterance) from settling a new utterance's gate.
                # Scribe garbles the wake word on one mic while hearing it on another; a
                # confirmation already recorded for this exact Whisper phrase outweighs the
                # veto (a different phrase in the same utterance is not covered — see
                # SpeechAnalyzer.scribe_confirmed_phrases). Only that order is covered — a
                # veto that lands first has already killed.
                confirmed_by = deps.speech_analyzer.wake_word_confirmed_by(query)
                if confirmed_by is not None:
                    note(f"Ignoring Scribe veto from '{mic or 'outpost'}' for '{query}': Scribe on "
                         f"'{confirmed_by or 'outpost'}' already confirmed the wake word for that phrase.")
                    deps.speech_analyzer.mark_scribe_settled(query)
                    return "", 204
                killed = self.kill_running_request_matching(query, reason=f"Scribe veto from {mic or 'outpost'}",
                                                            include_grown=True)
                deps.speech_analyzer.mark_scribe_settled(query)
                if not killed:
                    trace(f"Scribe veto for '{query}': no matching in-flight request "
                          f"(server already moved on, or nothing running).")
                return "", 204
            except Exception as e:
                error(f"Error: {str(e)}")
                return {"error": "An internal error occurred"}, 500

        @bp.route('/scribe_settled')
        async def scribe_settled():
            """A local outpost's Scribe verifier finished on a Whisper wake-word fire WITHOUT a
            vote or a veto — near-miss wake word, priming echo, low confidence, nothing left after
            cleanup. The fire stands, but no /stream vote will ever follow, so release the reply's
            verdict gate now instead of letting it burn MAX_WAIT_FOR_SCRIBE_VERDICT (every rescued
            near-miss is below the Scribe-skip floor, so it would pay the full cap). Settle-only:
            nothing is killed. Fire-and-forget, like /false_positive."""
            try:
                query = request.args.get('query', '')
                mic = request.args.get('mic', '')
                if not query:
                    trace("scribe_settled called with no query — nothing to settle.")
                    return "", 204
                gen = deps.speech_analyzer.mark_scribe_settled(query)
                if gen is None:
                    trace(f"Scribe no-vote from '{mic or 'outpost'}' for '{query}' matches no current "
                          f"utterance — stale, gate untouched.")
                else:
                    info(f"Scribe no-vote from '{mic or 'outpost'}' for '{query}' — releasing the "
                         f"verdict gate for utterance generation {gen}.")
                return "", 204
            except Exception as e:
                error(f"Error: {str(e)}")
                return {"error": "An internal error occurred"}, 500

        @bp.route('/end')
        async def end():
            try:
                if deps.play_sound is None:
                    trace("No end sound: this server has no speakers of its own")
                    return "", 204
                info("Playing end sound")
                deps.play_sound.playsound_end()
                return "", 204
            except Exception as e:
                error(f"Error: {str(e)}")
                return {"error": "An internal error occurred"}, 500

        @bp.route('/upload_audio_session', methods=['POST'])
        async def upload_audio_session():
            try:
                duration = request.headers.get('X-Session-Duration', 'unknown')
                audio_size = request.headers.get('X-Audio-Size', 'unknown')
                sample_rate = int(request.headers.get('X-Sample-Rate', '16000'))
                bit_depth = int(request.headers.get('X-Bit-Depth', '16'))
                channels = int(request.headers.get('X-Channels', '1'))
                query = request.headers.get('X-Query', 'unknown')

                audio_data = await request.get_data()

                filename = os.path.basename(
                    await asyncio.to_thread(
                        save_pcm_wav, "log/audio", query, audio_data,
                        sample_rate=sample_rate, channels=channels, bit_depth=bit_depth,
                    )
                )
                fine(f"Audio sample stored: {filename} (duration: {duration}s, size: {audio_size} bytes)")

                return {"status": "success", "filename": filename}, 200

            except Exception as e:
                error(f"Error uploading audio session: {str(e)}")
                return {"error": "Failed to save audio session"}, 500

        @bp.route('/identify_voice', methods=['POST'])
        async def identify_voice():
            """Identify the speaker from raw PCM audio data."""
            try:
                speaker_id = deps.speaker_identifier()
                if speaker_id is None:
                    return {"error": "Speaker identification not initialized"}, 503

                sample_rate = int(request.headers.get('X-Sample-Rate', '16000'))
                bit_depth = int(request.headers.get('X-Bit-Depth', '16'))
                channels = 1

                audio_data = await request.get_data()
                if not audio_data:
                    return {"error": "No audio data received"}, 400

                # Generate unique ID and save audio for debugging
                audio_id = f"{int(time.time() * 1000)}"
                await asyncio.to_thread(
                    save_pcm_wav, "log/audio/identification", audio_id, audio_data,
                    sample_rate=sample_rate, channels=channels, bit_depth=bit_depth,
                )

                start_time = time.time()
                speaker, confidence = speaker_id.identify(audio_data, sample_rate, bit_depth)
                elapsed_ms = int((time.time() - start_time) * 1000)
                info(f"Identified speaker: {speaker} (confidence: {confidence:.2f}) [audio_id: {audio_id}] ({elapsed_ms}ms)")

                return {"speaker": speaker, "confidence": confidence}, 200

            except Exception as e:
                error(f"Error identifying voice: {str(e)}")
                return {"error": "Failed to identify voice"}, 500

        @bp.route('/log')
        async def log():
            """Outposts forward their own log lines here so one floxy.out tells the story."""
            try:
                i = request.args.get('info', 'no')
                level = request.args.get('level', '').lower()
                name = request.args.get('name', '<no name>')
                message = request.args.get('message', '<no message>')
                if i == 'yes' or level == 'info':
                    external_info(name, message)
                elif level == 'fine' or request.args.get('fine', 'no') == 'yes':
                    external_fine(name, message)
                else:
                    external_trace(name, message)
                return "", 204
            except Exception as e:
                error(f"Error: {str(e)}")
                return {"error": "An internal error occurred"}, 500

        @bp.route('/outpost_settings')
        async def outpost_settings():
            try:
                client_name = request.args.get('client', 'unknown')
                trace(f"Settings requested by client: {client_name}")

                languages = deps.configured_languages()
                return {
                    "wake_words": [deps.wake_word.lower()],
                    "false_positive_timeout_ms": deps.false_positive_timeout_ms,  # Timeout in milliseconds for false positive detection
                    "language": languages[0],   # primary; kept scalar for backward-compatible clients
                    "languages": languages,     # full configured set, for clients that listen for several
                    "audio_processing": "VOICE_RECOGNITION"
                }, 200
            except Exception as e:
                error(f"Error: {str(e)}")
                return {"error": "An internal error occurred"}, 500

        return bp


def register(app, deps: MicrophoneDeps) -> MicrophoneRoutes:
    """Register the microphone blueprint on `app` and return the routes object; the
    launcher keeps it for the /stream pre-dispatch hook (accept_transcription)."""
    routes = MicrophoneRoutes(deps)
    app.register_blueprint(routes.blueprint())
    info("Microphone routes registered (/wake_word_detected, /speech_active, /false_positive, "
         "/end, /upload_audio_session, /identify_voice, /log, /outpost_settings).")
    return routes
